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


def episode_equivalent_collision_rate(
    collision_costs: torch.Tensor,
    *,
    episode_steps: int,
) -> float:
    """Convert per-step collision termination cost to an episode-rate estimate."""

    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive.")
    if collision_costs.numel() == 0:
        raise ValueError("collision_costs must be non-empty.")
    return float((collision_costs.detach().float().mean() * episode_steps).cpu())


def lagrangian_multiplier_update(
    multiplier: float,
    collision_rate: float,
    *,
    collision_limit: float,
    learning_rate: float,
    maximum: float,
) -> float:
    """Apply the fixed projected dual-ascent update used by exp142."""

    if multiplier < 0.0 or collision_limit < 0.0:
        raise ValueError("multiplier and collision_limit must be non-negative.")
    if learning_rate <= 0.0 or maximum <= 0.0:
        raise ValueError("learning_rate and maximum must be positive.")
    updated = multiplier + learning_rate * (collision_rate - collision_limit)
    return min(max(updated, 0.0), maximum)


def collision_termination_cost(done: object, *, device: torch.device) -> torch.Tensor:
    """Return the sole exp142 cost signal from real collision termination flags."""

    collision = getattr(done, "collision", None)
    if not isinstance(collision, torch.Tensor):
        raise RuntimeError("infos.done.collision is missing.")
    return collision.to(device=device, dtype=torch.float32)[:, None]


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


def normalized_agent_credit_traces(
    credits: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    discount_factor: float,
    trace_lambda: float,
    time_limit_bootstrap: bool,
) -> torch.Tensor:
    """Compute one jointly normalized temporal trace per agent.

    ``credits`` has shape ``[agents, rollout, environments, 1]``. All agents
    share termination flags. Normalization is joint across agents rather than
    per agent, preserving their relative allocation while allowing either
    zero-sum or agent-local, non-zero-sum credit assignments.
    """

    if credits.ndim != 4:
        raise ValueError("credits must have shape [agents, rollout, environments, 1].")
    if not (0.0 <= trace_lambda <= 1.0):
        raise ValueError("trace_lambda must be in [0, 1].")
    not_done = (
        (terminated | truncated) if time_limit_bootstrap else terminated
    ).logical_not()
    traces = torch.zeros_like(credits)
    running = torch.zeros_like(credits[:, 0])
    for index in reversed(range(credits.shape[1])):
        running = credits[:, index] + (
            discount_factor
            * trace_lambda
            * not_done[index].unsqueeze(0)
            * running
        )
        traces[:, index] = running
    return (traces - traces.mean()) / (traces.std() + 1.0e-8)


def normalized_centered_credit_traces(
    credits: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    discount_factor: float,
    trace_lambda: float,
    time_limit_bootstrap: bool,
) -> torch.Tensor:
    """Backward-compatible entry point for historical centered credits."""

    return normalized_agent_credit_traces(
        credits,
        terminated,
        truncated,
        discount_factor=discount_factor,
        trace_lambda=trace_lambda,
        time_limit_bootstrap=time_limit_bootstrap,
    )


def primary_preserving_gradient_merge(
    primary_gradients: tuple[torch.Tensor | None, ...],
    auxiliary_gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    auxiliary_scale: float,
) -> tuple[tuple[torch.Tensor, ...], dict[str, float]]:
    """Project a conflicting auxiliary gradient and cap it at the primary norm."""

    if len(primary_gradients) != len(parameters) or len(auxiliary_gradients) != len(parameters):
        raise ValueError("Gradient and parameter tuples must have matching lengths.")
    primary = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(primary_gradients, parameters, strict=True)
    )
    auxiliary = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(auxiliary_gradients, parameters, strict=True)
    )
    primary_dot_auxiliary = sum(
        (left * right).sum() for left, right in zip(primary, auxiliary, strict=True)
    )
    primary_norm_sq = sum(value.square().sum() for value in primary)
    auxiliary_norm = torch.sqrt(
        sum(value.square().sum() for value in auxiliary).clamp_min(0.0)
    )
    conflict = primary_dot_auxiliary < 0.0
    coefficient = torch.where(
        conflict,
        primary_dot_auxiliary / primary_norm_sq.clamp_min(1.0e-12),
        torch.zeros_like(primary_dot_auxiliary),
    )
    projected = tuple(
        auxiliary_value - coefficient * primary_value
        for primary_value, auxiliary_value in zip(primary, auxiliary, strict=True)
    )
    # Float32 cancellation can leave a small negative residual after the
    # analytic projection. Remove that residual once more instead of relaxing
    # the primary-preservation invariant.
    residual_dot = sum(
        (left * right).sum() for left, right in zip(primary, projected, strict=True)
    )
    numerical_margin = torch.where(
        conflict,
        torch.full_like(residual_dot, 1.0e-6),
        torch.zeros_like(residual_dot),
    )
    residual_coefficient = torch.clamp(
        residual_dot / primary_norm_sq.clamp_min(1.0e-12) - numerical_margin,
        max=0.0,
    )
    projected = tuple(
        projected_value - residual_coefficient * primary_value
        for primary_value, projected_value in zip(primary, projected, strict=True)
    )
    primary_norm = torch.sqrt(primary_norm_sq.clamp_min(0.0))
    projected_norm = torch.sqrt(
        sum(value.square().sum() for value in projected).clamp_min(0.0)
    )
    norm_scale = torch.clamp(
        primary_norm / projected_norm.clamp_min(1.0e-12),
        max=1.0,
    )
    capped = tuple(value * norm_scale for value in projected)
    merged = tuple(
        primary_value + float(auxiliary_scale) * auxiliary_value
        for primary_value, auxiliary_value in zip(primary, capped, strict=True)
    )
    projected_dot = sum(
        (left * right).sum() for left, right in zip(primary, capped, strict=True)
    )
    merged_norm = torch.sqrt(
        sum(value.square().sum() for value in merged).clamp_min(0.0)
    )
    merged_dot = sum(
        (left * right).sum() for left, right in zip(primary, merged, strict=True)
    )
    alignment = merged_dot / (primary_norm * merged_norm).clamp_min(1.0e-12)
    return merged, {
        "conflict": float(conflict.float().detach().cpu()),
        "cosine": float(
            primary_dot_auxiliary
            .div((primary_norm * auxiliary_norm).clamp_min(1.0e-12))
            .detach()
            .cpu()
        ),
        "auxiliary_primary_norm_ratio": float(
            auxiliary_norm.div(primary_norm.clamp_min(1.0e-12)).detach().cpu()
        ),
        "projected_primary_dot": float(projected_dot.detach().cpu()),
        "combined_primary_cosine": float(alignment.detach().cpu()),
        "auxiliary_norm_cap_scale": float(norm_scale.detach().cpu()),
    }


class SharedPolicyMAPPO(MAPPO):
    """MAPPO variant that jointly updates one shared actor and one shared critic."""

    def __init__(
        self,
        *,
        entropy_loss_scale_end: float | None = None,
        entropy_schedule_timesteps: int | None = None,
        actor_credit_scale: float = 0.0,
        actor_credit_trace_lambda: float = 0.95,
        actor_credit_gradient_mode: str = "additive_advantage",
        collision_cost_value: nn.Module | None = None,
        collision_constraint_enabled: bool = False,
        collision_cost_discount_factor: float = 0.99,
        collision_cost_gae_lambda: float = 0.95,
        collision_cost_limit: float = 0.02,
        collision_episode_steps: int = 480,
        lagrangian_init: float = 0.0,
        lagrangian_learning_rate: float = 0.1,
        lagrangian_max: float = 2.0,
        collision_cost_value_learning_rate: float = 3.0e-4,
        collision_cost_value_loss_scale: float = 0.5,
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
        if actor_credit_scale < 0.0:
            raise ValueError("actor_credit_scale must be non-negative.")
        if not (0.0 <= actor_credit_trace_lambda <= 1.0):
            raise ValueError("actor_credit_trace_lambda must be in [0, 1].")
        self.actor_credit_scale = float(actor_credit_scale)
        self.actor_credit_trace_lambda = float(actor_credit_trace_lambda)
        if actor_credit_gradient_mode not in {
            "additive_advantage",
            "primary_projected_norm_cap",
        }:
            raise ValueError(
                "actor_credit_gradient_mode must be additive_advantage or "
                "primary_projected_norm_cap."
            )
        if actor_credit_gradient_mode == "primary_projected_norm_cap" and self.cfg.mixed_precision:
            raise ValueError("Primary-projected Actor credit requires mixed_precision=false.")
        self.actor_credit_gradient_mode = actor_credit_gradient_mode
        self.collision_constraint_enabled = bool(collision_constraint_enabled)
        if self.collision_constraint_enabled and self.actor_credit_scale > 0.0:
            raise ValueError(
                "The collision constraint component cannot be combined with Actor credit."
            )
        if self.collision_constraint_enabled and collision_cost_value is None:
            raise ValueError(
                "collision_cost_value is required when collision_constraint_enabled=true."
            )
        if not (0.0 <= collision_cost_discount_factor <= 1.0):
            raise ValueError("collision_cost_discount_factor must be in [0, 1].")
        if not (0.0 <= collision_cost_gae_lambda <= 1.0):
            raise ValueError("collision_cost_gae_lambda must be in [0, 1].")
        if collision_cost_limit < 0.0 or collision_episode_steps <= 0:
            raise ValueError("Collision cost limit and episode steps are invalid.")
        if not (0.0 <= lagrangian_init <= lagrangian_max):
            raise ValueError("lagrangian_init must be in [0, lagrangian_max].")
        if lagrangian_learning_rate <= 0.0 or lagrangian_max <= 0.0:
            raise ValueError("Lagrangian learning rate and maximum must be positive.")
        if collision_cost_value_learning_rate <= 0.0:
            raise ValueError("Collision cost value learning rate must be positive.")
        if collision_cost_value_loss_scale <= 0.0:
            raise ValueError("Collision cost value loss scale must be positive.")
        self.collision_cost_value = collision_cost_value
        self.collision_cost_discount_factor = float(collision_cost_discount_factor)
        self.collision_cost_gae_lambda = float(collision_cost_gae_lambda)
        self.collision_cost_limit = float(collision_cost_limit)
        self.collision_episode_steps = int(collision_episode_steps)
        self.lagrangian_multiplier = float(lagrangian_init)
        self.lagrangian_learning_rate = float(lagrangian_learning_rate)
        self.lagrangian_max = float(lagrangian_max)
        self.collision_cost_value_loss_scale = float(collision_cost_value_loss_scale)
        self.collision_cost_optimizer = None
        if self.collision_constraint_enabled:
            self.collision_cost_value.to(self.device)
            self.collision_cost_optimizer = torch.optim.Adam(
                self.collision_cost_value.parameters(),
                lr=float(collision_cost_value_learning_rate),
            )
            first_memory = self.memories[first]
            first_memory.create_tensor(name="collision_cost", size=1)
            first_memory.create_tensor(name="collision_cost_values", size=1)
            self.checkpoint_modules[first]["collision_cost_value"] = self.collision_cost_value
            self.checkpoint_modules[first]["collision_cost_optimizer"] = (
                self.collision_cost_optimizer
            )
        if self.actor_credit_scale > 0.0:
            for memory in self.memories.values():
                memory.create_tensor(name="actor_credit", size=1)
        self.joint_update_count = 0
        self.critic_update_count = 0
        self.optimizer_count = 2 if self.collision_constraint_enabled else 1
        self.last_actor_sample_count = 0
        self.last_critic_sample_count = 0
        self.last_actor_credit_abs_mean = 0.0
        self.last_actor_credit_std = 0.0
        self.last_team_advantage_std = 0.0
        self.last_actor_gradient_conflict_fraction = 0.0
        self.last_actor_gradient_cosine_mean = 0.0
        self.last_actor_gradient_projected_dot_min = 0.0
        self.last_actor_gradient_combined_cosine_min = 1.0
        self.last_actor_gradient_norm_cap_scale_mean = 1.0
        self.collision_cost_critic_update_count = 0
        self.last_collision_cost_value_loss = 0.0
        self.last_collision_episode_equivalent_rate = 0.0
        self.last_lagrangian_multiplier_applied = self.lagrangian_multiplier
        self.collision_constraint_history: list[dict[str, float]] = []

    def record_transition(self, **kwargs) -> None:
        _assert_matching_tensor_dict("centralized states", kwargs["states"], self.possible_agents)
        _assert_matching_tensor_dict("next centralized states", kwargs["next_states"], self.possible_agents)
        _assert_matching_tensor_dict("team rewards", kwargs["rewards"], self.possible_agents)
        _assert_matching_tensor_dict("terminated flags", kwargs["terminated"], self.possible_agents)
        _assert_matching_tensor_dict("truncated flags", kwargs["truncated"], self.possible_agents)
        memory_indices = {
            uid: int(self.memories[uid].memory_index) for uid in self.possible_agents
        }
        actor_credit = None
        collision_cost = None
        collision_cost_values = None
        if self.collision_constraint_enabled and self.training:
            done = kwargs["infos"].get("done")
            collision_cost = collision_termination_cost(done, device=self.device)
            first = self.possible_agents[0]
            with torch.no_grad():
                cost_inputs = {
                    "observations": self._observation_preprocessor[first](
                        kwargs["observations"][first]
                    ),
                    "states": self._state_preprocessor[first](kwargs["states"][first]),
                }
                self.collision_cost_value.enable_training_mode(False)
                collision_cost_values, _ = self.collision_cost_value.act(
                    cost_inputs, role="value"
                )
                self.collision_cost_value.enable_training_mode(True)
        if self.actor_credit_scale > 0.0:
            actor_credit_info = kwargs["infos"].get("actor_credit", {})
            actor_credit = actor_credit_info.get(
                "policy",
                actor_credit_info.get("centered"),
            )
            if not isinstance(actor_credit, torch.Tensor):
                raise RuntimeError("Actor credit is enabled but infos contain no policy credit.")
            expected_shape = (actor_credit.shape[0], len(self.possible_agents))
            if actor_credit.shape != expected_shape:
                raise RuntimeError(
                    "Actor policy credit must have shape [environments, agents]."
                )
        super().record_transition(**kwargs)
        if collision_cost is not None and collision_cost_values is not None:
            first = self.possible_agents[0]
            memory = self.memories[first]
            memory.tensors["collision_cost"][memory_indices[first]].copy_(collision_cost)
            memory.tensors["collision_cost_values"][memory_indices[first]].copy_(
                collision_cost_values
            )
        if actor_credit is not None:
            for index, uid in enumerate(self.possible_agents):
                self.memories[uid].tensors["actor_credit"][memory_indices[uid]].copy_(
                    actor_credit[:, index, None]
                )

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
        names = ["states", "rewards", "terminated", "truncated", "values"]
        for name in names:
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

            last_collision_cost_values = None
            if self.collision_constraint_enabled:
                self.collision_cost_value.enable_training_mode(False)
                last_collision_cost_values, _ = self.collision_cost_value.act(
                    next_inputs, role="value"
                )
                self.collision_cost_value.enable_training_mode(True)

        values = first_memory.get_tensor_by_name("values")
        team_rewards = first_memory.get_tensor_by_name("rewards")
        returns, advantages = compute_gae(
            rewards=team_rewards,
            terminated=first_memory.get_tensor_by_name("terminated"),
            truncated=first_memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[first],
            lambda_coefficient=self.cfg.gae_lambda[first],
            time_limit_bootstrap=self.cfg.time_limit_bootstrap[first],
        )
        collision_costs = None
        collision_cost_returns = None
        collision_cost_advantages = None
        lagrangian_multiplier_applied = self.lagrangian_multiplier
        if self.collision_constraint_enabled:
            collision_costs = first_memory.get_tensor_by_name("collision_cost")
            collision_cost_values = first_memory.get_tensor_by_name(
                "collision_cost_values"
            )
            collision_cost_returns, collision_cost_advantages = compute_gae(
                rewards=collision_costs,
                terminated=first_memory.get_tensor_by_name("terminated"),
                truncated=first_memory.get_tensor_by_name("truncated"),
                values=collision_cost_values,
                last_values=last_collision_cost_values,
                discount_factor=self.collision_cost_discount_factor,
                lambda_coefficient=self.collision_cost_gae_lambda,
                # Both termination and timeout are trajectory boundaries for
                # collision cost. This prevents bootstrap across environment reset.
                time_limit_bootstrap=True,
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
        if collision_cost_advantages is not None:
            actor_advantages = actor_advantages - lagrangian_multiplier_applied * (
                self._flat(collision_cost_advantages).repeat(
                    len(self.possible_agents), 1
                )
            )
        actor_credit_advantages = None
        if self.actor_credit_scale > 0.0:
            policy_credits = torch.stack(
                [
                    self.memories[uid].get_tensor_by_name("actor_credit")
                    for uid in self.possible_agents
                ],
                dim=0,
            )
            credit_traces = normalized_agent_credit_traces(
                policy_credits,
                first_memory.get_tensor_by_name("terminated"),
                first_memory.get_tensor_by_name("truncated"),
                discount_factor=self.cfg.discount_factor[first],
                trace_lambda=self.actor_credit_trace_lambda,
                time_limit_bootstrap=self.cfg.time_limit_bootstrap[first],
            )
            flat_credit = torch.cat(
                [self._flat(credit_traces[index]) for index in range(len(self.possible_agents))],
                dim=0,
            )
            actor_credit_advantages = flat_credit
            if self.actor_credit_gradient_mode == "additive_advantage":
                actor_advantages = actor_advantages + self.actor_credit_scale * flat_credit
            self.last_actor_credit_abs_mean = float(flat_credit.abs().mean().cpu())
            self.last_actor_credit_std = float(flat_credit.std().cpu())
        else:
            self.last_actor_credit_abs_mean = 0.0
            self.last_actor_credit_std = 0.0
        self.last_team_advantage_std = float(advantages.std().cpu())
        critic_states = self._flat(first_memory.get_tensor_by_name("states"))
        critic_values = self._flat(values)
        critic_returns = self._flat(returns)
        flat_collision_cost_returns = (
            self._flat(collision_cost_returns)
            if collision_cost_returns is not None
            else None
        )
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
        cumulative_collision_cost_value_loss = 0.0
        optimization_steps = 0
        projection_records: list[dict[str, float]] = []

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
                    terrain_policy_loss = None
                    if self.actor_credit_gradient_mode == "primary_projected_norm_cap":
                        if actor_credit_advantages is None:
                            raise RuntimeError("Projected Actor credit has no credit advantages.")
                        terrain_surrogate = actor_credit_advantages[actor_index] * ratio
                        terrain_clipped = actor_credit_advantages[actor_index] * torch.clip(
                            ratio,
                            1.0 - self.cfg.ratio_clip[first],
                            1.0 + self.cfg.ratio_clip[first],
                        )
                        terrain_policy_loss = -torch.min(
                            terrain_surrogate, terrain_clipped
                        ).mean()
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

                    collision_cost_value_loss = None
                    if flat_collision_cost_returns is not None:
                        predicted_collision_cost_values, _ = (
                            self.collision_cost_value.act(critic_inputs, role="value")
                        )
                        collision_cost_value_loss = (
                            self.collision_cost_value_loss_scale
                            * F.mse_loss(
                                flat_collision_cost_returns[critic_index],
                                predicted_collision_cost_values,
                            )
                        )

                self.shared_optimizer.zero_grad(set_to_none=True)
                if self.actor_credit_gradient_mode == "primary_projected_norm_cap":
                    if terrain_policy_loss is None:
                        raise RuntimeError("Projected Actor credit loss was not computed.")
                    policy_parameters = tuple(self.shared_policy.parameters())
                    primary_gradients = torch.autograd.grad(
                        policy_loss + entropy_loss,
                        policy_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    auxiliary_gradients = torch.autograd.grad(
                        terrain_policy_loss,
                        policy_parameters,
                        allow_unused=True,
                    )
                    merged_gradients, projection_metrics = primary_preserving_gradient_merge(
                        primary_gradients,
                        auxiliary_gradients,
                        policy_parameters,
                        auxiliary_scale=self.actor_credit_scale,
                    )
                    value_loss.backward()
                    for parameter, gradient in zip(
                        policy_parameters, merged_gradients, strict=True
                    ):
                        parameter.grad = gradient.detach().clone()
                    projection_records.append(projection_metrics)
                else:
                    self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
                if self.cfg.grad_norm_clip[first] > 0:
                    if self.actor_credit_gradient_mode != "primary_projected_norm_cap":
                        self.scaler.unscale_(self.shared_optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.shared_policy.parameters(),
                            self.shared_value.parameters(),
                        ),
                        self.cfg.grad_norm_clip[first],
                    )
                if self.actor_credit_gradient_mode == "primary_projected_norm_cap":
                    self.shared_optimizer.step()
                else:
                    self.scaler.step(self.shared_optimizer)
                    self.scaler.update()
                if collision_cost_value_loss is not None:
                    self.collision_cost_optimizer.zero_grad(set_to_none=True)
                    collision_cost_value_loss.backward()
                    if self.cfg.grad_norm_clip[first] > 0:
                        nn.utils.clip_grad_norm_(
                            self.collision_cost_value.parameters(),
                            self.cfg.grad_norm_clip[first],
                        )
                    self.collision_cost_optimizer.step()
                    cumulative_collision_cost_value_loss += float(
                        collision_cost_value_loss.detach()
                    )

                cumulative_policy_loss += float(policy_loss.detach())
                cumulative_value_loss += float(value_loss.detach())
                cumulative_entropy_loss += float(entropy_loss.detach())
                optimization_steps += 1

        self.joint_update_count += 1
        self.critic_update_count += 1
        if collision_costs is not None:
            self.collision_cost_critic_update_count += 1
            collision_rate = episode_equivalent_collision_rate(
                collision_costs,
                episode_steps=self.collision_episode_steps,
            )
            next_multiplier = lagrangian_multiplier_update(
                self.lagrangian_multiplier,
                collision_rate,
                collision_limit=self.collision_cost_limit,
                learning_rate=self.lagrangian_learning_rate,
                maximum=self.lagrangian_max,
            )
            self.last_collision_episode_equivalent_rate = collision_rate
            self.last_lagrangian_multiplier_applied = lagrangian_multiplier_applied
            self.lagrangian_multiplier = next_multiplier
        if projection_records:
            self.last_actor_gradient_conflict_fraction = sum(
                item["conflict"] for item in projection_records
            ) / len(projection_records)
            self.last_actor_gradient_cosine_mean = sum(
                item["cosine"] for item in projection_records
            ) / len(projection_records)
            self.last_actor_gradient_projected_dot_min = min(
                item["projected_primary_dot"] for item in projection_records
            )
            self.last_actor_gradient_combined_cosine_min = min(
                item["combined_primary_cosine"] for item in projection_records
            )
            self.last_actor_gradient_norm_cap_scale_mean = sum(
                item["auxiliary_norm_cap_scale"] for item in projection_records
            ) / len(projection_records)
        else:
            self.last_actor_gradient_conflict_fraction = 0.0
            self.last_actor_gradient_cosine_mean = 0.0
            self.last_actor_gradient_projected_dot_min = 0.0
            self.last_actor_gradient_combined_cosine_min = 1.0
            self.last_actor_gradient_norm_cap_scale_mean = 1.0
        denominator = max(optimization_steps, 1)
        self.last_collision_cost_value_loss = (
            cumulative_collision_cost_value_loss / denominator
        )
        if collision_costs is not None:
            self.collision_constraint_history.append(
                {
                    "update": float(self.joint_update_count),
                    "episode_equivalent_collision_rate": (
                        self.last_collision_episode_equivalent_rate
                    ),
                    "lagrangian_multiplier_applied": (
                        self.last_lagrangian_multiplier_applied
                    ),
                    "lagrangian_multiplier": self.lagrangian_multiplier,
                    "cost_value_loss": self.last_collision_cost_value_loss,
                }
            )
        self.track_data("Loss / Policy loss (shared)", cumulative_policy_loss / denominator)
        self.track_data("Loss / Value loss (shared)", cumulative_value_loss / denominator)
        self.track_data("Loss / Entropy loss (shared)", cumulative_entropy_loss / denominator)
        self.track_data("Policy / Entropy scale (shared)", entropy_scale)
        self.track_data("Policy / Actor credit scale", self.actor_credit_scale)
        self.track_data(
            "Policy / Actor gradient conflict fraction",
            self.last_actor_gradient_conflict_fraction,
        )
        self.track_data(
            "Policy / Actor projected-primary dot minimum",
            self.last_actor_gradient_projected_dot_min,
        )
        self.track_data(
            "Policy / Actor combined-primary cosine minimum",
            self.last_actor_gradient_combined_cosine_min,
        )
        self.track_data(
            "Policy / Actor credit absolute mean",
            self.last_actor_credit_abs_mean,
        )
        self.track_data("Policy / Actor credit std", self.last_actor_credit_std)
        if self.collision_constraint_enabled:
            self.track_data(
                "Constraint / Episode-equivalent collision rate",
                self.last_collision_episode_equivalent_rate,
            )
            self.track_data(
                "Constraint / Lagrangian multiplier",
                self.lagrangian_multiplier,
            )
            self.track_data(
                "Loss / Collision cost value loss",
                self.last_collision_cost_value_loss,
            )
        self.track_data(
            "Policy / Standard deviation (shared)",
            self.shared_policy.distribution(role="policy").stddev.mean().item(),
        )
