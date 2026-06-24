"""Project-side shared-policy MAPPO with one joint optimizer/update."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.multi_agents.torch import MultiAgent
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.multi_agents.torch.mappo.mappo import compute_gae
from skrl.utils import ScopedTimer


def linear_schedule(start: float, end: float, timestep: int, timesteps: int) -> float:
    if timesteps <= 1:
        return end
    alpha = min(1.0, max(0.0, timestep / float(timesteps - 1)))
    return (1.0 - alpha) * start + alpha * end


def scheduled_entropy_scale(
    start: float,
    end: float,
    timestep: int,
    timesteps: int,
    schedule_timesteps: int | None = None,
) -> float:
    horizon = min(timesteps, schedule_timesteps or timesteps)
    return linear_schedule(start, end, timestep, horizon)


def _assert_matching_tensor_dict(
    name: str,
    values: dict[str, torch.Tensor | None],
    agents: list[str],
) -> None:
    reference = values[agents[0]]
    for agent in agents[1:]:
        candidate = values[agent]
        if reference is None or candidate is None:
            if reference is not candidate:
                raise RuntimeError(f"SharedPolicyMAPPO requires matching {name} for all agents.")
            continue
        if reference.shape != candidate.shape or not torch.equal(reference, candidate):
            raise RuntimeError(f"SharedPolicyMAPPO requires matching {name} for all agents.")


class SharedPolicyMAPPO(MAPPO):
    """MAPPO variant that jointly updates one shared actor and one shared critic."""

    def __init__(
        self,
        *,
        entropy_loss_scale_end: float | None = None,
        entropy_schedule_timesteps: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        first = self.possible_agents[0]
        self.shared_policy = self.policies[first]
        self.shared_value = self.values[first]
        if self.shared_policy is None or self.shared_value is None:
            raise ValueError("SharedPolicyMAPPO requires both policy and value models.")
        if any(self.policies[uid] is not self.shared_policy for uid in self.possible_agents):
            raise ValueError("SharedPolicyMAPPO requires one shared policy instance.")
        if any(self.values[uid] is not self.shared_value for uid in self.possible_agents):
            raise ValueError("SharedPolicyMAPPO requires one shared value instance.")
        if any(self.cfg.learning_rate_scheduler[uid][0] is not None for uid in self.possible_agents):
            raise ValueError("SharedPolicyMAPPO does not support per-agent learning-rate schedulers.")

        learning_rates = {float(self.cfg.learning_rate[uid][0]) for uid in self.possible_agents}
        if len(learning_rates) != 1:
            raise ValueError("SharedPolicyMAPPO requires one learning rate for all agents.")
        parameters = itertools.chain(
            self.shared_policy.parameters(),
            self.shared_value.parameters(),
        )
        self.shared_optimizer = torch.optim.Adam(parameters, lr=learning_rates.pop())
        self.optimizers = {uid: self.shared_optimizer for uid in self.possible_agents}
        self.schedulers = {uid: None for uid in self.possible_agents}
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["optimizer"] = self.shared_optimizer

        self.entropy_loss_scale_start = float(self.cfg.entropy_loss_scale[first])
        self.entropy_loss_scale_end = float(
            self.entropy_loss_scale_start
            if entropy_loss_scale_end is None
            else entropy_loss_scale_end
        )
        if entropy_schedule_timesteps is not None and entropy_schedule_timesteps <= 0:
            raise ValueError("entropy_schedule_timesteps must be positive when provided.")
        self.entropy_schedule_timesteps = entropy_schedule_timesteps
        self.joint_update_count = 0
        self.critic_update_count = 0
        self.optimizer_count = 1
        self.last_actor_sample_count = 0
        self.last_critic_sample_count = 0

    def record_transition(self, **kwargs) -> None:
        _assert_matching_tensor_dict("centralized states", kwargs["states"], self.possible_agents)
        _assert_matching_tensor_dict("next centralized states", kwargs["next_states"], self.possible_agents)
        _assert_matching_tensor_dict("team rewards", kwargs["rewards"], self.possible_agents)
        _assert_matching_tensor_dict("terminated flags", kwargs["terminated"], self.possible_agents)
        _assert_matching_tensor_dict("truncated flags", kwargs["truncated"], self.possible_agents)
        super().record_transition(**kwargs)

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update_joint(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)
        MultiAgent.post_interaction(self, timestep=timestep, timesteps=timesteps)

    @staticmethod
    def _flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, tensor.shape[-1])

    def _validate_memory_consistency(self) -> None:
        first_memory = self.memories[self.possible_agents[0]]
        for name in ("states", "rewards", "terminated", "truncated", "values"):
            reference = first_memory.get_tensor_by_name(name)
            for uid in self.possible_agents[1:]:
                candidate = self.memories[uid].get_tensor_by_name(name)
                if reference.shape != candidate.shape or not torch.equal(reference, candidate):
                    raise RuntimeError(
                        f"SharedPolicyMAPPO requires identical per-agent memory tensor '{name}'."
                    )

    def update_joint(self, *, timestep: int, timesteps: int) -> None:
        self._validate_memory_consistency()
        first = self.possible_agents[0]
        first_memory = self.memories[first]

        with torch.no_grad(), torch.autocast(
            device_type=self._device_type,
            enabled=self.cfg.mixed_precision,
        ):
            next_inputs = {
                "observations": self._observation_preprocessor[first](
                    self._current_next_observations[first]
                ),
                "states": self._state_preprocessor[first](self._current_next_states[first]),
            }
            self.shared_value.enable_training_mode(False)
            last_values, _ = self.shared_value.act(next_inputs, role="value")
            self.shared_value.enable_training_mode(True)
            last_values = self._value_preprocessor[first](last_values, inverse=True)

        values = first_memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=first_memory.get_tensor_by_name("rewards"),
            terminated=first_memory.get_tensor_by_name("terminated"),
            truncated=first_memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[first],
            lambda_coefficient=self.cfg.gae_lambda[first],
            time_limit_bootstrap=self.cfg.time_limit_bootstrap[first],
        )

        actor_observations = torch.cat(
            [
                self._flat(self.memories[uid].get_tensor_by_name("observations"))
                for uid in self.possible_agents
            ],
            dim=0,
        )
        actor_actions = torch.cat(
            [
                self._flat(self.memories[uid].get_tensor_by_name("actions"))
                for uid in self.possible_agents
            ],
            dim=0,
        )
        actor_log_probs = torch.cat(
            [
                self._flat(self.memories[uid].get_tensor_by_name("log_prob"))
                for uid in self.possible_agents
            ],
            dim=0,
        )
        actor_advantages = self._flat(advantages).repeat(len(self.possible_agents), 1)
        critic_states = self._flat(first_memory.get_tensor_by_name("states"))
        critic_values = self._flat(values)
        critic_returns = self._flat(returns)
        self.last_actor_sample_count = actor_observations.shape[0]
        self.last_critic_sample_count = critic_states.shape[0]

        entropy_scale = scheduled_entropy_scale(
            self.entropy_loss_scale_start,
            self.entropy_loss_scale_end,
            timestep,
            timesteps,
            self.entropy_schedule_timesteps,
        )
        cumulative_policy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_entropy_loss = 0.0
        optimization_steps = 0

        for epoch in range(self.cfg.learning_epochs[first]):
            actor_batches = torch.tensor_split(
                torch.randperm(actor_observations.shape[0], device=self.device),
                self.cfg.mini_batches[first],
            )
            critic_batches = torch.tensor_split(
                torch.randperm(critic_states.shape[0], device=self.device),
                self.cfg.mini_batches[first],
            )
            for actor_index, critic_index in zip(actor_batches, critic_batches, strict=True):
                actor_inputs = {
                    "observations": self._observation_preprocessor[first](
                        actor_observations[actor_index],
                        train=not epoch,
                    ),
                    "states": critic_states[critic_index],
                }
                critic_inputs = {
                    "observations": actor_observations[actor_index[: critic_index.numel()]],
                    "states": self._state_preprocessor[first](
                        critic_states[critic_index],
                        train=not epoch,
                    ),
                }
                with torch.autocast(
                    device_type=self._device_type,
                    enabled=self.cfg.mixed_precision,
                ):
                    _, outputs = self.shared_policy.act(
                        {
                            **actor_inputs,
                            "taken_actions": actor_actions[actor_index],
                        },
                        role="policy",
                    )
                    next_log_prob = outputs["log_prob"]
                    ratio = torch.exp(next_log_prob - actor_log_probs[actor_index])
                    surrogate = actor_advantages[actor_index] * ratio
                    clipped = actor_advantages[actor_index] * torch.clip(
                        ratio,
                        1.0 - self.cfg.ratio_clip[first],
                        1.0 + self.cfg.ratio_clip[first],
                    )
                    policy_loss = -torch.min(surrogate, clipped).mean()
                    entropy_loss = (
                        -entropy_scale
                        * self.shared_policy.get_entropy(role="policy").mean()
                    )

                    predicted_values, _ = self.shared_value.act(critic_inputs, role="value")
                    if self.cfg.value_clip[first] > 0:
                        predicted_values = critic_values[critic_index] + torch.clip(
                            predicted_values - critic_values[critic_index],
                            min=-self.cfg.value_clip[first],
                            max=self.cfg.value_clip[first],
                        )
                    value_loss = self.cfg.value_loss_scale[first] * F.mse_loss(
                        critic_returns[critic_index],
                        predicted_values,
                    )

                self.shared_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
                if self.cfg.grad_norm_clip[first] > 0:
                    self.scaler.unscale_(self.shared_optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.shared_policy.parameters(),
                            self.shared_value.parameters(),
                        ),
                        self.cfg.grad_norm_clip[first],
                    )
                self.scaler.step(self.shared_optimizer)
                self.scaler.update()

                cumulative_policy_loss += float(policy_loss.detach())
                cumulative_value_loss += float(value_loss.detach())
                cumulative_entropy_loss += float(entropy_loss.detach())
                optimization_steps += 1

        self.joint_update_count += 1
        self.critic_update_count += 1
        denominator = max(optimization_steps, 1)
        self.track_data("Loss / Policy loss (shared)", cumulative_policy_loss / denominator)
        self.track_data("Loss / Value loss (shared)", cumulative_value_loss / denominator)
        self.track_data("Loss / Entropy loss (shared)", cumulative_entropy_loss / denominator)
        self.track_data("Policy / Entropy scale (shared)", entropy_scale)
        self.track_data(
            "Policy / Standard deviation (shared)",
            self.shared_policy.distribution(role="policy").stddev.mean().item(),
        )
