#!/usr/bin/env python3
"""Run the mandatory frozen exp158 DAE identifiability and causal audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_exp125_credit_assignment import _load_critic, _value, rank_correlation
from analyze_reward_component_identifiability import (
    component_metrics,
    fit_multi_output_regressor,
)
from dae_credit import (
    CounterfactualRewardModel,
    MultiScaleRewardStateEncoder,
    RunningRewardNormalizer,
    factual_reward_model_loss,
    reward_validation_metrics,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    DIFFERENTIAL_REVERSE_ENDPOINTS,
    DIFFERENTIAL_SPIN_YAW_DELTAS,
    DIFFERENTIAL_YIELD_ENDPOINTS,
    SPATIOTEMPORAL_ENDPOINTS,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from play import _load_policy_players


EXPERIMENT_ID = "exp158_dae_validation"
DEFAULT_CONFIG = ROOT / "configs/experiment/exp158_h1_gae.yaml"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/runs/exp157_h1_site_belief_n1/"
    "h1_n1_seed23_full_2400iter/checkpoints/ppo_timestep_134400.pt"
)
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/runs/exp156_differential_multiscale_ablation/"
    "_suite/scenario_manifest.json"
)
DEFAULT_RUN_DIR = ROOT / "outputs/runs/exp158_dae_validation/offline_credit_audit"
REWARD_COMPONENT_NAMES = ("gather", "near_safety", "path_risk")


@dataclass(slots=True)
class FactualDataset:
    states: torch.Tensor
    actor_observations: torch.Tensor
    actions: torch.Tensor
    probabilities: torch.Tensor
    rewards: torch.Tensor
    components: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    conflict_participants: torch.Tensor


@dataclass(slots=True)
class CounterfactualDataset:
    states: torch.Tensor
    actor_observations: torch.Tensor
    joint_actions: torch.Tensor
    probabilities: torch.Tensor
    rewards: torch.Tensor
    base_rewards: torch.Tensor
    conflict_participants: torch.Tensor
    shared_advantages: torch.Tensor
    cells: list[str]
    seeds: list[int]


class StateOnlyRewardModel(nn.Module):
    """Matched state encoder and reward trunk without joint-action inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = MultiScaleRewardStateEncoder()
        self.reward_trunk = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.normalizer = RunningRewardNormalizer()

    def forward(self, states: torch.Tensor, *, denormalize: bool = True) -> torch.Tensor:
        normalized = self.reward_trunk(self.state_encoder(states))
        return self.normalizer.denormalize(normalized) if denormalize else normalized


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _core_digest(core: MultiRoverGatheringCore) -> str:
    digest = hashlib.sha256()
    tensors = {
        "positions": core.positions,
        "velocities_xy": core.velocities_xy,
        "yaws": core.yaws,
        "angular_velocities": core.angular_velocities,
        "step_count": core.step_count,
        "success_hold_count": core.success_hold_count,
        "previous_physical_action": core.previous_physical_action,
        "communication_features": core.communication_cache.features,
        "communication_age": core.communication_cache.age,
        "conflict_consecutive": core.trajectory_conflicts.consecutive_steps,
    }
    for name, tensor in tensors.items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(core.generator.get_state().cpu().numpy().tobytes())
    return digest.hexdigest()


def capture_core_state(core: MultiRoverGatheringCore) -> dict[str, Any]:
    """Capture every mutable core field for exact diagnostic branching."""

    state: dict[str, Any] = {}
    for name, value in vars(core).items():
        if name in {"cfg", "device", "generator"}:
            continue
        state[name] = copy.deepcopy(value)
    state["__generator_state__"] = core.generator.get_state().clone()
    state["__digest__"] = _core_digest(core)
    return state


def restore_core_state(core: MultiRoverGatheringCore, state: dict[str, Any]) -> None:
    for name, value in state.items():
        if name.startswith("__"):
            continue
        setattr(core, name, copy.deepcopy(value))
    core.generator.set_state(state["__generator_state__"])
    if _core_digest(core) != state["__digest__"]:
        raise RuntimeError("Frozen environment state digest changed after restore")


def _balanced_family_actions(shape: tuple[int, int], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    families = torch.randint(0, 5, shape, generator=generator, device=device)
    actions = torch.zeros(shape, dtype=torch.long, device=device)
    ranges = ((0, 1), (1, 40), (40, 43), (43, 45), (45, 47))
    for family, (start, end) in enumerate(ranges):
        mask = families == family
        count = int(mask.sum())
        if count:
            actions[mask] = torch.randint(
                start,
                end,
                (count,),
                generator=generator,
                device=device,
            )
    return actions


def _sample_behavior_actions(
    probabilities: torch.Tensor,
    *,
    exploration_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    policy_actions = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        1,
        generator=generator,
    ).reshape(*probabilities.shape[:-1])
    exploratory = _balanced_family_actions(
        tuple(policy_actions.shape), generator, probabilities.device
    )
    mask = torch.rand(
        policy_actions.shape,
        generator=generator,
        device=probabilities.device,
    ) < float(exploration_fraction)
    return torch.where(mask, exploratory, policy_actions)


def _weighted_components(output: Any, core: MultiRoverGatheringCore, nearest_before: torch.Tensor) -> torch.Tensor:
    terms = output.info["reward_terms"]
    gather = terms.gather * float(core.cfg.reward_weights.gather)
    near_safety = (
        output.info["metrics"].nearest_neighbor_distance - nearest_before
    ).mean(dim=-1)
    relative_risk = (output.info.get("path_terrain") or {}).get("relative_risk_mean")
    if not isinstance(relative_risk, torch.Tensor):
        relative_risk = (output.info.get("path_terrain") or {}).get("risk_mean")
    if not isinstance(relative_risk, torch.Tensor):
        raise RuntimeError("exp158 audit requires quintic path risk")
    path_risk = -relative_risk.mean(dim=-1)
    return torch.stack((gather, near_safety, path_risk), dim=-1)


def collect_factual_dataset(
    *,
    config: Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    exploration_fraction: float,
) -> FactualDataset:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    cfg.initial_state.progress_timestep_override = int(
        (checkpoint_data.get("metadata") or {}).get("timesteps", 134400)
    )
    core = MultiRoverGatheringCore(cfg)
    act, _ = _load_policy_players(checkpoint_data, cfg, core.device, raw_cfg=raw_cfg)
    critic = _load_critic(checkpoint_data, cfg, core.device)
    actor_obs, critic_state = core.get_observations()
    generator = torch.Generator(device=core.device)
    generator.manual_seed(seed + 158_000)
    records: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "states",
            "actor_observations",
            "actions",
            "probabilities",
            "rewards",
            "components",
            "terminated",
            "truncated",
            "values",
            "next_values",
            "conflict_participants",
        )
    }
    for _ in range(steps):
        with torch.no_grad():
            probabilities = act.probabilities(actor_obs)
            actions = _sample_behavior_actions(
                probabilities,
                exploration_fraction=exploration_fraction,
                generator=generator,
            )
            values = _value(critic, critic_state)
        nearest_before = core.metrics.nearest_neighbor_distance.clone()
        records["states"].append(critic_state.detach().cpu())
        records["actor_observations"].append(actor_obs.detach().cpu())
        records["actions"].append(actions.detach().cpu())
        records["probabilities"].append(probabilities.detach().cpu())
        records["values"].append(values.detach().cpu())
        output = core.step(actions)
        conflict = output.info["trajectory_conflicts"]["active"]
        conflict = (conflict | conflict.transpose(1, 2)).any(dim=-1)
        records["rewards"].append(output.rewards[:, :1].detach().cpu())
        records["components"].append(
            _weighted_components(output, core, nearest_before).detach().cpu()
        )
        records["terminated"].append(output.terminated[:, None].detach().cpu())
        records["truncated"].append(output.truncated[:, None].detach().cpu())
        records["next_values"].append(
            _value(critic, output.critic_state).detach().cpu()
        )
        records["conflict_participants"].append(conflict.detach().cpu())
        actor_obs = output.actor_obs
        critic_state = output.critic_state

    stacked = {name: torch.stack(values) for name, values in records.items()}
    return FactualDataset(
        states=stacked["states"].reshape(-1, 950),
        actor_observations=stacked["actor_observations"].reshape(-1, 4, cfg.actor_obs_dim),
        actions=stacked["actions"].reshape(-1, 4),
        probabilities=stacked["probabilities"].reshape(-1, 4, 47),
        rewards=stacked["rewards"].reshape(-1, 1),
        components=stacked["components"].reshape(-1, len(REWARD_COMPONENT_NAMES)),
        terminated=stacked["terminated"].reshape(-1, 1),
        truncated=stacked["truncated"].reshape(-1, 1),
        values=stacked["values"].reshape(-1, 1),
        next_values=stacked["next_values"].reshape(-1, 1),
        conflict_participants=stacked["conflict_participants"].reshape(-1, 4),
    )


def _fit_reward_models(
    train: FactualDataset,
    validations: dict[int, FactualDataset],
    *,
    device: torch.device,
    model_seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[CounterfactualRewardModel, StateOnlyRewardModel, dict[str, Any]]:
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(model_seed)
        action_model = CounterfactualRewardModel().to(device)
        torch.manual_seed(model_seed + 1000)
        state_model = StateOnlyRewardModel().to(device)
    action_optimizer = torch.optim.Adam(action_model.parameters(), lr=3.0e-4)
    state_optimizer = torch.optim.Adam(state_model.parameters(), lr=3.0e-4)
    action_model.normalizer.update(train.rewards.to(device))
    state_model.normalizer.update(train.rewards.to(device))
    generator = torch.Generator()
    generator.manual_seed(model_seed + 158)
    for _ in range(epochs):
        permutation = torch.randperm(train.states.shape[0], generator=generator)
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            states = train.states[index].to(device)
            actions = train.actions[index].to(device)
            rewards = train.rewards[index].to(device)
            action_loss = factual_reward_model_loss(
                action_model, states, actions, rewards
            )
            action_optimizer.zero_grad(set_to_none=True)
            action_loss.backward()
            nn.utils.clip_grad_norm_(action_model.parameters(), 5.0)
            action_optimizer.step()

            target = state_model.normalizer.normalize(rewards)
            state_loss = (state_model(states, denormalize=False) - target).square().mean()
            state_optimizer.zero_grad(set_to_none=True)
            state_loss.backward()
            nn.utils.clip_grad_norm_(state_model.parameters(), 5.0)
            state_optimizer.step()

    validation_metrics: dict[str, Any] = {}
    action_model.eval()
    state_model.eval()
    for seed, dataset in validations.items():
        action_metrics = reward_validation_metrics(
            action_model,
            dataset.states.to(device),
            dataset.actions.to(device),
            dataset.rewards.to(device),
        )
        state_predictions = []
        with torch.no_grad():
            for start in range(0, dataset.states.shape[0], batch_size):
                state_predictions.append(
                    state_model(dataset.states[start : start + batch_size].to(device)).cpu()
                )
        state_prediction = torch.cat(state_predictions).reshape(-1)
        state_metrics = component_metrics(state_prediction, dataset.rewards.reshape(-1))
        improvement = (state_metrics["mse"] - action_metrics["mse"]) / max(
            state_metrics["mse"], 1.0e-12
        )
        validation_metrics[str(seed)] = {
            "action_conditioned": action_metrics,
            "state_only": state_metrics,
            "mse_improvement_fraction": improvement,
        }
    return action_model, state_model, validation_metrics


def _component_identifiability(
    train: FactualDataset,
    validations: dict[int, FactualDataset],
    *,
    device: torch.device,
    model_seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    train_state = train.states
    train_action = torch.nn.functional.one_hot(train.actions, num_classes=47).reshape(
        train.actions.shape[0], -1
    ).float()
    aggregate: dict[str, list[float]] = {name: [] for name in REWARD_COMPONENT_NAMES}
    by_validation: dict[str, Any] = {}
    for model_seed in model_seeds:
        state_model = fit_multi_output_regressor(
            train_state,
            train.components,
            device=device,
            seed=model_seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=3.0e-4,
            hidden_dim=128,
        )
        action_model = fit_multi_output_regressor(
            torch.cat((train_state, train_action), dim=-1),
            train.components,
            device=device,
            seed=model_seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=3.0e-4,
            hidden_dim=128,
        )
        for seed, dataset in validations.items():
            one_hot = torch.nn.functional.one_hot(
                dataset.actions, num_classes=47
            ).reshape(dataset.actions.shape[0], -1).float()
            state_prediction = state_model.predict(dataset.states.to(device)).cpu()
            action_prediction = action_model.predict(
                torch.cat((dataset.states, one_hot), dim=-1).to(device)
            ).cpu()
            seed_record = by_validation.setdefault(str(seed), {})
            for index, name in enumerate(REWARD_COMPONENT_NAMES):
                state_metrics = component_metrics(
                    state_prediction[:, index], dataset.components[:, index]
                )
                action_metrics = component_metrics(
                    action_prediction[:, index], dataset.components[:, index]
                )
                improvement = (state_metrics["mse"] - action_metrics["mse"]) / max(
                    state_metrics["mse"], 1.0e-12
                )
                seed_record.setdefault(name, []).append(improvement)
                aggregate[name].append(improvement)
    return {
        "by_validation_seed": by_validation,
        "aggregate": {
            name: {
                "mean_mse_improvement_fraction": sum(values) / len(values),
                "minimum_mse_improvement_fraction": min(values),
            }
            for name, values in aggregate.items()
        },
    }


def _rollout_branch(
    core: MultiRoverGatheringCore,
    snapshot: dict[str, Any],
    first_actions: torch.Tensor,
    act: Any,
    critic: nn.Module,
    *,
    horizon: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict[str, torch.Tensor]:
    restore_core_state(core, snapshot)
    _, initial_state = core.get_observations()
    initial_value = _value(critic, initial_state).detach()
    actions = first_actions
    rewards = []
    values = []
    terminated = []
    truncated = []
    active = torch.ones(core.num_envs, dtype=torch.bool, device=core.device)
    value_64 = None
    alive_64 = None
    for step in range(horizon):
        _, state = core.get_observations()
        values.append(_value(critic, state).detach())
        output = core.step(actions)
        rewards.append(output.rewards[:, 0] * active)
        terminated.append(output.terminated & active)
        truncated.append(output.truncated & active)
        done = (output.terminated | output.truncated) & active
        active = active & ~done
        if step == 63:
            value_64 = _value(critic, output.critic_state).detach()
            alive_64 = active.clone()
        with torch.no_grad():
            actions = act(output.actor_obs)
    last_value = _value(critic, output.critic_state).detach()
    rewards_t = torch.stack(rewards)
    terminated_t = torch.stack(terminated)[:, :, None]
    truncated_t = torch.stack(truncated)[:, :, None]
    values_t = torch.stack(values)[:, :, None]
    running = torch.zeros(core.num_envs, device=core.device)
    gae = torch.zeros_like(running)
    for step in reversed(range(horizon)):
        next_value = values_t[step + 1, :, 0] if step + 1 < horizon else last_value
        not_done = ~terminated_t[step, :, 0]
        running = rewards_t[step] - values_t[step, :, 0] + gamma * not_done * (
            next_value + gae_lambda * running
        )
        gae = running
    return {
        "rewards": rewards_t,
        "initial_value": initial_value,
        "gae_start": gae,
        "value_64": value_64 if value_64 is not None else last_value,
        "alive_64": alive_64 if alive_64 is not None else active,
        "last_value": last_value,
        "alive_last": active,
    }


def _enumerate_counterfactuals(
    core: MultiRoverGatheringCore,
    snapshot: dict[str, Any],
    base_actions: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    restore_core_state(core, snapshot)
    baseline = core.step(base_actions)
    baseline_rewards = baseline.rewards[:, 0].detach().clone()
    restore_core_state(core, snapshot)
    rewards = torch.empty(
        core.num_envs, core.n_agents, 47, device=core.device
    )
    for agent in range(core.n_agents):
        for action in range(47):
            candidate = base_actions.clone()
            candidate[:, agent] = action
            output = core.step(candidate)
            rewards[:, agent, action] = output.rewards[:, 0]
            restore_core_state(core, snapshot)
    factual = rewards.gather(
        2, base_actions.long().unsqueeze(-1)
    ).squeeze(-1)
    if not torch.allclose(
        factual,
        baseline_rewards[:, None].expand_as(factual),
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise RuntimeError("Counterfactual enumeration does not reconstruct factual reward")
    return rewards, baseline


def _cfg_for_manifest_cell(base: dict[str, Any], cell: dict[str, Any], *, device: str, num_envs: int) -> tuple[Any, dict[str, Any]]:
    raw = copy.deepcopy(base)
    raw.setdefault("experiment", {}).update(
        seed=int(cell["seed"]), num_envs=num_envs, device=device
    )
    raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
    raw["initial_state"]["curriculum_enabled"] = False
    raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
    raw.setdefault("safety", {})["collision_termination_enabled"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(raw, stream)
        path = Path(stream.name)
    try:
        cfg = cfg_from_experiment(path)
    finally:
        path.unlink(missing_ok=True)
    return cfg, raw


def collect_counterfactual_dataset(
    *,
    config: Path,
    checkpoint_data: dict[str, Any],
    manifest_path: Path,
    device: str,
    num_envs_per_cell: int,
    skip_long_horizon: bool,
) -> tuple[CounterfactualDataset, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = load_yaml(config)
    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "states",
            "actor_observations",
            "joint_actions",
            "probabilities",
            "rewards",
            "base_rewards",
            "conflict_participants",
            "shared_advantages",
        )
    }
    cells: list[str] = []
    seeds: list[int] = []
    restore_checks = 0
    long_records = []
    for cell in manifest["cells"]:
        cfg, raw = _cfg_for_manifest_cell(
            base, cell, device=device, num_envs=num_envs_per_cell
        )
        core = MultiRoverGatheringCore(cfg)
        act, _ = _load_policy_players(
            checkpoint_data, cfg, core.device, raw_cfg=raw
        )
        critic = _load_critic(checkpoint_data, cfg, core.device)
        actor_obs, _ = core.get_observations()
        for step in range(1, 97):
            with torch.no_grad():
                base_actions = act(actor_obs)
                probabilities = act.probabilities(actor_obs)
            if step in {32, 64, 96}:
                snapshot = capture_core_state(core)
                state_before = core.get_observations()[1].detach().clone()
                rewards, baseline = _enumerate_counterfactuals(
                    core, snapshot, base_actions
                )
                restore_checks += 1
                conflict = baseline.info["trajectory_conflicts"]["active"]
                participants = (conflict | conflict.transpose(1, 2)).any(dim=-1)
                if step == 32:
                    selected = torch.arange(
                        min(32, num_envs_per_cell), device=core.device
                    )
                    category = "random"
                elif step == 64:
                    score = participants.float().sum(dim=-1)
                    selected = score.topk(min(16, num_envs_per_cell)).indices
                    category = "conflict"
                else:
                    selected = core.metrics.dmax.topk(
                        min(16, num_envs_per_cell), largest=False
                    ).indices
                    category = "low_dmax"
                baseline_branch = _rollout_branch(
                    core,
                    snapshot,
                    base_actions,
                    act,
                    critic,
                    horizon=128 if step == 32 and not skip_long_horizon else 64,
                )
                collected["states"].append(state_before[selected].cpu())
                collected["actor_observations"].append(actor_obs[selected].cpu())
                collected["joint_actions"].append(base_actions[selected].cpu())
                collected["probabilities"].append(probabilities[selected].cpu())
                collected["rewards"].append(rewards[selected].cpu())
                collected["base_rewards"].append(
                    baseline.rewards[selected, :1].cpu()
                )
                collected["conflict_participants"].append(participants[selected].cpu())
                collected["shared_advantages"].append(
                    baseline_branch["gae_start"][selected, None].cpu()
                )
                cells.extend([str(cell["cell"])] * int(selected.numel()))
                seeds.extend([int(cell["seed"])] * int(selected.numel()))
                if step == 32 and not skip_long_horizon:
                    query = torch.arange(core.num_envs, device=core.device) % core.n_agents
                    rows = torch.arange(core.num_envs, device=core.device)
                    best = rewards[rows, query].argmax(dim=-1)
                    intervention = base_actions.clone()
                    intervention[rows, query] = best
                    altered = _rollout_branch(
                        core,
                        snapshot,
                        intervention,
                        act,
                        critic,
                        horizon=128,
                    )
                    delta = altered["rewards"] - baseline_branch["rewards"]
                    weights = 0.99 ** torch.arange(128, device=core.device)[:, None]
                    total = (weights * delta.abs()).sum(dim=0)
                    after = (weights[65:] * delta.abs()[65:]).sum(dim=0)
                    impact = after / total.clamp_min(1.0e-8)

                    def target(branch: dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
                        local_weights = 0.99 ** torch.arange(horizon, device=core.device)[:, None]
                        result = (local_weights * branch["rewards"][:horizon]).sum(dim=0)
                        if horizon == 64:
                            result = result + 0.99**64 * branch["value_64"] * branch["alive_64"]
                        else:
                            result = result + 0.99**horizon * branch["last_value"] * branch["alive_last"]
                        return result

                    error64 = (
                        baseline_branch["initial_value"] - target(baseline_branch, 64)
                    ).abs().mean()
                    error128 = (
                        baseline_branch["initial_value"] - target(baseline_branch, 128)
                    ).abs().mean()
                    improvement = float(
                        ((error64 - error128) / error64.clamp_min(1.0e-8)).cpu()
                    )
                    long_records.append(
                        {
                            "cell": cell["cell"],
                            "category": category,
                            "m_gt_64": float(impact.mean().cpu()),
                            "bootstrap_error_64": float(error64.cpu()),
                            "bootstrap_error_128": float(error128.cpu()),
                            "bootstrap_error_improvement": improvement,
                        }
                    )
                restore_core_state(core, snapshot)
            output = core.step(base_actions)
            actor_obs = output.actor_obs

    data = CounterfactualDataset(
        states=torch.cat(collected["states"]),
        actor_observations=torch.cat(collected["actor_observations"]),
        joint_actions=torch.cat(collected["joint_actions"]),
        probabilities=torch.cat(collected["probabilities"]),
        rewards=torch.cat(collected["rewards"]),
        base_rewards=torch.cat(collected["base_rewards"]),
        conflict_participants=torch.cat(collected["conflict_participants"]),
        shared_advantages=torch.cat(collected["shared_advantages"]),
        cells=cells,
        seeds=seeds,
    )
    long_horizon = {
        "skipped": skip_long_horizon,
        "records": long_records,
        "m_gt_64": (
            sum(record["m_gt_64"] for record in long_records) / len(long_records)
            if long_records
            else None
        ),
        "bootstrap_error_improvement": (
            sum(record["bootstrap_error_improvement"] for record in long_records)
            / len(long_records)
            if long_records
            else None
        ),
        "restore_digest_checks": restore_checks,
    }
    return data, long_horizon


def _action_lateral_signs() -> torch.Tensor:
    signs = [0.0]
    for endpoint in SPATIOTEMPORAL_ENDPOINTS:
        signs.extend([float(torch.sign(torch.tensor(endpoint[1])))] * 3)
    signs.extend(float(torch.sign(torch.tensor(item[1]))) for item in DIFFERENTIAL_REVERSE_ENDPOINTS)
    signs.extend(float(torch.sign(torch.tensor(item))) for item in DIFFERENTIAL_SPIN_YAW_DELTAS)
    signs.extend(float(torch.sign(torch.tensor(item[1]))) for item in DIFFERENTIAL_YIELD_ENDPOINTS)
    return torch.tensor(signs)


def _observation_aliasing_rate(dataset: CounterfactualDataset) -> float:
    observations = dataset.actor_observations.reshape(-1, dataset.actor_observations.shape[-1]).float()
    best = dataset.rewards.argmax(dim=-1).reshape(-1)
    mean = observations.mean(dim=0)
    std = observations.std(dim=0).clamp_min(1.0e-5)
    normalized = (observations - mean) / std
    distances = torch.cdist(normalized, normalized) / normalized.shape[-1] ** 0.5
    distances.fill_diagonal_(float("inf"))
    nearest_distance, nearest = distances.min(dim=1)
    threshold = torch.quantile(nearest_distance, 0.10)
    close = nearest_distance <= threshold
    signs = _action_lateral_signs()
    incompatible = signs[best] * signs[best[nearest]] < 0.0
    return float(incompatible[close].float().mean()) if close.any() else 0.0


def _counterfactual_model_metrics(
    models: list[CounterfactualRewardModel],
    dataset: CounterfactualDataset,
    *,
    device: torch.device,
) -> dict[str, Any]:
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions.append(
                model.predict_all_actions(
                    dataset.states.to(device),
                    dataset.joint_actions.to(device),
                    chunk_size=65_536,
                ).cpu()
            )
    prediction = torch.stack(predictions).mean(dim=0)
    true = dataset.rewards
    seed_metrics = {}
    for seed in sorted(set(dataset.seeds)):
        index = torch.tensor([value == seed for value in dataset.seeds])
        correlations = []
        for row in range(int(index.sum())):
            true_row = true[index][row]
            prediction_row = prediction[index][row]
            for agent in range(4):
                value = rank_correlation(prediction_row[agent], true_row[agent])
                if value is not None:
                    correlations.append(float(value))
        seed_metrics[str(seed)] = {
            "mean_action_spearman": sum(correlations) / max(len(correlations), 1),
            "samples": int(index.sum()),
        }
    probabilities = dataset.probabilities
    predicted_expectation = (prediction * probabilities).sum(dim=-1)
    true_expectation = (true * probabilities).sum(dim=-1)
    expectation_error = float(
        (predicted_expectation - true_expectation).abs().mean()
        / true.std().clamp_min(1.0e-8)
    )
    std_by_agent = prediction.std(dim=-1).mean(dim=0)
    rows = torch.arange(dataset.joint_actions.shape[0])
    factual = true.gather(
        2, dataset.joint_actions[:, :, None]
    ).squeeze(-1)
    true_delta = factual - true_expectation
    participants = dataset.conflict_participants
    participant_values = true_delta.abs()[participants]
    nonparticipant_values = true_delta.abs()[~participants]
    ratio = (
        float(participant_values.median() / nonparticipant_values.median().clamp_min(1.0e-8))
        if participant_values.numel() and nonparticipant_values.numel()
        else 0.0
    )
    repeated_advantage = dataset.shared_advantages.expand(-1, 4)
    shared_correlation = rank_correlation(repeated_advantage, true_delta)
    return {
        "prediction_std_by_agent": std_by_agent.tolist(),
        "minimum_prediction_std": float(std_by_agent.min()),
        "policy_weighted_expectation_error_std_fraction": expectation_error,
        "by_terrain_seed": seed_metrics,
        "minimum_seed_action_spearman": min(
            item["mean_action_spearman"] for item in seed_metrics.values()
        ),
        "participant_nonparticipant_abs_delta_median_ratio": ratio,
        "shared_advantage_delta_spearman": float(shared_correlation or 0.0),
        "observation_aliasing_rate": _observation_aliasing_rate(dataset),
    }


def run_audit(
    *,
    config: Path,
    checkpoint: Path,
    manifest: Path,
    run_dir: Path,
    device: str,
    train_num_envs: int,
    validation_num_envs: int,
    factual_steps: int,
    counterfactual_num_envs: int,
    model_seeds: tuple[int, ...],
    model_epochs: int,
    batch_size: int,
    skip_long_horizon: bool,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device)
    actor_digest_before = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in checkpoint_data["rover_0"]["policy"].values()
        )
    ).hexdigest()
    train = collect_factual_dataset(
        config=config,
        checkpoint_data=checkpoint_data,
        device=device,
        num_envs=train_num_envs,
        steps=factual_steps,
        seed=14023,
        exploration_fraction=0.30,
    )
    validations = {
        seed: collect_factual_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=validation_num_envs,
            steps=factual_steps,
            seed=seed,
            exploration_fraction=0.30,
        )
        for seed in (15023, 16023)
    }
    action_models = []
    model_validation = []
    model_artifacts = []
    for seed in model_seeds:
        action_model, state_model, validation = _fit_reward_models(
            train,
            validations,
            device=torch_device,
            model_seed=seed,
            epochs=model_epochs,
            batch_size=batch_size,
        )
        action_models.append(action_model)
        model_validation.append({"model_seed": seed, "validation": validation})
        model_artifacts.append(
            {
                "model_seed": seed,
                "action_conditioned": {
                    key: value.detach().cpu()
                    for key, value in action_model.state_dict().items()
                },
                "state_only": {
                    key: value.detach().cpu()
                    for key, value in state_model.state_dict().items()
                },
            }
        )
    total_improvements = {
        str(seed): min(
            record["validation"][str(seed)]["mse_improvement_fraction"]
            for record in model_validation
        )
        for seed in validations
    }
    components = _component_identifiability(
        train,
        validations,
        device=torch_device,
        model_seeds=model_seeds,
        epochs=model_epochs,
        batch_size=batch_size,
    )
    counterfactual, long_horizon = collect_counterfactual_dataset(
        config=config,
        checkpoint_data=checkpoint_data,
        manifest_path=manifest,
        device=device,
        num_envs_per_cell=counterfactual_num_envs,
        skip_long_horizon=skip_long_horizon,
    )
    counterfactual_metrics = _counterfactual_model_metrics(
        action_models,
        counterfactual,
        device=torch_device,
    )
    temporal_pass = bool(skip_long_horizon) or not (
        float(long_horizon["m_gt_64"] or 0.0) >= 0.30
        and float(long_horizon["bootstrap_error_improvement"] or 0.0) >= 0.15
    )
    component_passes = {
        name: components["aggregate"][name]["minimum_mse_improvement_fraction"]
        >= 0.15
        for name in REWARD_COMPONENT_NAMES
    }
    checks = {
        "temporal_credit_not_dominant": temporal_pass,
        "participant_margin_ratio_ge_2": counterfactual_metrics[
            "participant_nonparticipant_abs_delta_median_ratio"
        ]
        >= 2.0,
        "shared_advantage_delta_spearman_lt_0_20": abs(
            counterfactual_metrics["shared_advantage_delta_spearman"]
        )
        < 0.20,
        "observation_aliasing_rate_le_0_20": counterfactual_metrics[
            "observation_aliasing_rate"
        ]
        <= 0.20,
        "total_reward_action_mse_improvement_ge_0_15_each_seed": all(
            value >= 0.15 for value in total_improvements.values()
        ),
        "two_of_three_component_improvements_ge_0_15": sum(component_passes.values())
        >= 2,
        "counterfactual_prediction_std_gt_1e_4": counterfactual_metrics[
            "minimum_prediction_std"
        ]
        > 1.0e-4,
        "counterfactual_action_spearman_ge_0_30_each_seed": counterfactual_metrics[
            "minimum_seed_action_spearman"
        ]
        >= 0.30,
        "policy_weighted_expectation_error_le_0_25_std": counterfactual_metrics[
            "policy_weighted_expectation_error_std_fraction"
        ]
        <= 0.25,
    }
    actor_digest_after = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in checkpoint_data["rover_0"]["policy"].values()
        )
    ).hexdigest()
    checks["frozen_actor_unchanged"] = actor_digest_before == actor_digest_after
    passed = all(checks.values()) and not skip_long_horizon
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "exp158_offline_audit_v1",
        },
        "experiment": EXPERIMENT_ID,
        "status": "dae_offline_gate_passed" if passed else "dae_offline_gate_failed",
        "passed": passed,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "collection": {
            "train_samples": int(train.states.shape[0]),
            "validation_samples_by_seed": {
                str(seed): int(dataset.states.shape[0])
                for seed, dataset in validations.items()
            },
            "counterfactual_states": int(counterfactual.states.shape[0]),
            "counterfactual_labels": int(counterfactual.rewards.numel()),
            "model_seeds": list(model_seeds),
            "model_epochs": model_epochs,
        },
        "total_reward_models": model_validation,
        "minimum_total_mse_improvement_by_validation_seed": total_improvements,
        "component_identifiability": components,
        "component_passes": component_passes,
        "counterfactual_metrics": counterfactual_metrics,
        "long_horizon": long_horizon,
        "checks": checks,
        "decision": (
            "allow_h1_seed23_paired_training"
            if passed
            else "stop_before_dae_training"
        ),
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
        },
    }
    metrics_dir = run_dir / "metrics"
    artifacts_dir = run_dir / "artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "offline_gate.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {
            "diagnostic_only": True,
            "deployable": False,
            "source_checkpoint": str(checkpoint),
            "models": model_artifacts,
        },
        artifacts_dir / "reward_models.pt",
    )
    manifest_data = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "run": "offline_credit_audit",
        "producer": "scripts/audit_exp158_dae.py",
        "summary": {"status": result["status"], "passed": passed},
        "artifacts": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "diagnostic_models": str(
                (artifacts_dir / "reward_models.pt").relative_to(ROOT)
            ),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT.relative_to(ROOT)))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR.relative_to(ROOT)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-num-envs", type=int, default=128)
    parser.add_argument("--validation-num-envs", type=int, default=64)
    parser.add_argument("--factual-steps", type=int, default=480)
    parser.add_argument("--counterfactual-num-envs", type=int, default=64)
    parser.add_argument("--model-seeds", type=_parse_seeds, default=(7, 17, 29))
    parser.add_argument("--model-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-long-horizon", action="store_true")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    for required in (resolve(args.config), resolve(args.checkpoint), resolve(args.manifest)):
        if not required.is_file():
            raise SystemExit(f"Required exp158 audit input is missing: {required}")
    result = run_audit(
        config=resolve(args.config),
        checkpoint=resolve(args.checkpoint),
        manifest=resolve(args.manifest),
        run_dir=resolve(args.run_dir),
        device=args.device,
        train_num_envs=args.train_num_envs,
        validation_num_envs=args.validation_num_envs,
        factual_steps=args.factual_steps,
        counterfactual_num_envs=args.counterfactual_num_envs,
        model_seeds=args.model_seeds,
        model_epochs=args.model_epochs,
        batch_size=args.batch_size,
        skip_long_horizon=args.skip_long_horizon,
    )
    print(json.dumps({"status": result["status"], "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
