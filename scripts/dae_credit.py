"""Training-only Difference Advantage Estimation utilities for exp158.

Nothing in this module is used by the deployed Actor.  The counterfactual
reward model consumes centralized training state and joint actions only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


CRITIC_STATE_DIM = 950
N_AGENTS = 4
ACTION_COUNT = 47
CRITIC_SLICES = {
    "agents": (0, 32),
    "team": (32, 40),
    "terrain": (40, 45),
    "oracle": (45, 54),
    "agent_terrain": (54, 950),
}
TERRAIN_SLICES = {
    "fine": (0, 126, 7, 9),
    "medium": (126, 168, 3, 7),
    "coarse": (168, 224, 4, 7),
}


def dae_beta_schedule(
    update: int,
    *,
    target: float = 0.3,
    warmup_updates: int = 128,
    ramp_updates: int = 128,
) -> float:
    """Return the pre-registered exp158 beta for a one-indexed policy update."""

    if update <= 0:
        raise ValueError("update must be one-indexed and positive")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target beta must be in [0, 1]")
    if warmup_updates < 0 or ramp_updates < 0:
        raise ValueError("warmup and ramp updates must be non-negative")
    if update <= warmup_updates:
        return 0.0
    if ramp_updates == 0:
        return float(target)
    ramp_position = min(update - warmup_updates, ramp_updates)
    return float(target) * float(ramp_position) / float(ramp_updates)


def compute_raw_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float,
    lambda_coefficient: float,
    time_limit_bootstrap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unnormalized GAE while preserving the project's SKRL semantics."""

    not_done = (
        (terminated | truncated) if time_limit_bootstrap else terminated
    ).logical_not()
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for index in reversed(range(rewards.shape[0])):
        next_values = values[index + 1] if index + 1 < rewards.shape[0] else last_values
        running = rewards[index] - values[index] + discount_factor * not_done[index] * (
            next_values + lambda_coefficient * running
        )
        advantages[index] = running
    return advantages + values, advantages


def compute_dae_advantages(
    *,
    team_raw_advantages: torch.Tensor,
    expected_counterfactual_rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    beta: float,
    discount_factor: float,
    lambda_coefficient: float,
) -> torch.Tensor:
    """Return normalized per-agent DAE advantages.

    Args:
        team_raw_advantages: ``[T, E, 1]`` unnormalized team GAE.
        expected_counterfactual_rewards: ``[T, E, A, 1]`` policy-weighted
            counterfactual immediate reward baselines.

    Returns:
        Tensor shaped ``[A, T, E, 1]``.  Beta zero deliberately normalizes
        before replication, matching the existing shared-team GAE path exactly.
    """

    if team_raw_advantages.ndim != 3 or team_raw_advantages.shape[-1] != 1:
        raise ValueError("team_raw_advantages must have shape [T, E, 1]")
    if expected_counterfactual_rewards.ndim != 4:
        raise ValueError(
            "expected_counterfactual_rewards must have shape [T, E, A, 1]"
        )
    if expected_counterfactual_rewards.shape[:2] != team_raw_advantages.shape[:2]:
        raise ValueError("DAE reward and team-advantage horizons must match")
    if expected_counterfactual_rewards.shape[-1] != 1:
        raise ValueError("DAE reward baseline must have a singleton final dimension")
    if terminated.shape != team_raw_advantages.shape or truncated.shape != team_raw_advantages.shape:
        raise ValueError("termination tensors must match team_raw_advantages")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    n_agents = expected_counterfactual_rewards.shape[2]
    if beta == 0.0:
        normalized = (team_raw_advantages - team_raw_advantages.mean()) / (
            team_raw_advantages.std() + 1.0e-8
        )
        return normalized.unsqueeze(0).repeat(n_agents, 1, 1, 1)

    expected = expected_counterfactual_rewards.permute(2, 0, 1, 3)
    traces = torch.zeros_like(expected)
    running = torch.zeros_like(expected[:, 0])
    episode_not_done = (terminated | truncated).logical_not()
    trace_discount = float(discount_factor) * float(lambda_coefficient) * float(beta)
    for index in reversed(range(team_raw_advantages.shape[0])):
        running = expected[:, index] + trace_discount * episode_not_done[index].unsqueeze(0) * running
        traces[:, index] = running
    dae_raw = team_raw_advantages.unsqueeze(0) - float(beta) * traces
    return (dae_raw - dae_raw.mean()) / (dae_raw.std() + 1.0e-8)


class MultiScaleRewardStateEncoder(nn.Module):
    """Independent 950-D state encoder matching structured_multiscale_v3 slices."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_encoder = nn.Sequential(nn.Linear(8, 32), nn.ELU())
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.terrain_projection = nn.Sequential(nn.Linear(3 * 32, 32), nn.ELU())
        self.agent_fusion = nn.Sequential(nn.Linear(64, 32), nn.ELU())
        self.team_encoder = nn.Sequential(nn.Linear(8, 32), nn.ELU())
        self.summary_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
        self.oracle_encoder = nn.Sequential(nn.Linear(9, 32), nn.ELU())
        self.projection = nn.Sequential(nn.Linear(144, 128), nn.ELU())

    @staticmethod
    def _terrain_grids(flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        grids = []
        for start, end, x_size, y_size in TERRAIN_SLICES.values():
            values = flat[..., start:end]
            grid = values.reshape(-1, x_size, y_size, 2).permute(0, 3, 1, 2)
            grids.append(grid.contiguous())
        return tuple(grids)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != CRITIC_STATE_DIM:
            raise ValueError(
                f"DAE reward state encoder expects {CRITIC_STATE_DIM} values, "
                f"got {states.shape[-1]}"
            )
        leading = states.shape[:-1]
        agents = states[..., 0:32].reshape(-1, N_AGENTS, 8)
        encoded_agents = self.agent_encoder(agents)
        terrain = states[..., 54:950].reshape(-1, N_AGENTS, 224)
        terrain_flat = terrain.reshape(-1, 224)
        pooled = [self.terrain_cnn(grid).mean(dim=(-2, -1)) for grid in self._terrain_grids(terrain_flat)]
        terrain_encoded = self.terrain_projection(torch.cat(pooled, dim=-1)).reshape(
            -1, N_AGENTS, 32
        )
        fused = self.agent_fusion(torch.cat((encoded_agents, terrain_encoded), dim=-1))
        encoded = torch.cat(
            (
                fused.mean(dim=1),
                fused.amax(dim=1),
                self.team_encoder(states[..., 32:40].reshape(-1, 8)),
                self.summary_encoder(states[..., 40:45].reshape(-1, 5)),
                self.oracle_encoder(states[..., 45:54].reshape(-1, 9)),
            ),
            dim=-1,
        )
        return self.projection(encoded).reshape(*leading, 128)


class RunningRewardNormalizer(nn.Module):
    """Numerically stable factual-reward normalization owned by the reward model."""

    def __init__(self, epsilon: float = 1.0e-3) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("m2", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double().reshape(-1)
        if values.numel() == 0:
            return
        batch_count = torch.as_tensor(float(values.numel()), device=values.device, dtype=torch.float64)
        batch_mean = values.mean()
        batch_m2 = (values - batch_mean).square().sum()
        if self.count.item() == 0.0:
            self.count.copy_(batch_count)
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean.add_(delta * batch_count / total)
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.count.copy_(total)

    @property
    def std(self) -> torch.Tensor:
        denominator = torch.clamp(self.count - 1.0, min=1.0)
        return torch.sqrt(self.m2 / denominator).clamp_min(self.epsilon)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.dtype)) / self.std.to(values.dtype)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.dtype) + self.mean.to(values.dtype)


class CounterfactualRewardModel(nn.Module):
    """Shared queried-agent model for factual and all-action reward prediction."""

    def __init__(
        self,
        *,
        action_count: int = ACTION_COUNT,
        n_agents: int = N_AGENTS,
        action_embedding_dim: int = 16,
        agent_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if action_count <= 1 or n_agents <= 1:
            raise ValueError("DAE requires at least two actions and two agents")
        self.action_count = int(action_count)
        self.n_agents = int(n_agents)
        self.action_embedding_dim = int(action_embedding_dim)
        self.state_encoder = MultiScaleRewardStateEncoder()
        self.action_embedding = nn.Embedding(self.action_count, self.action_embedding_dim)
        self.agent_embedding = nn.Embedding(self.n_agents, int(agent_embedding_dim))
        self.mask_embedding = nn.Parameter(torch.zeros(self.action_embedding_dim))
        input_dim = (
            128
            + self.n_agents * self.action_embedding_dim
            + self.action_embedding_dim
            + int(agent_embedding_dim)
        )
        if input_dim != 216:
            raise AssertionError(f"exp158 reward trunk input must be 216, got {input_dim}")
        self.reward_trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.normalizer = RunningRewardNormalizer()

    def _predict_from_latent(
        self,
        state_latent: torch.Tensor,
        joint_actions: torch.Tensor,
        queried_agents: torch.Tensor,
        candidate_actions: torch.Tensor,
        *,
        denormalize: bool = True,
    ) -> torch.Tensor:
        if state_latent.ndim != 2 or state_latent.shape[-1] != 128:
            raise ValueError("state_latent must have shape [B, 128]")
        if joint_actions.shape != (state_latent.shape[0], self.n_agents):
            raise ValueError("joint_actions must have shape [B, 4]")
        queried_agents = queried_agents.long().reshape(-1)
        candidate_actions = candidate_actions.long().reshape(-1)
        if (
            queried_agents.shape[0] != state_latent.shape[0]
            or candidate_actions.shape[0] != state_latent.shape[0]
        ):
            raise ValueError("query and candidate batches must match state_latent")
        if queried_agents.min() < 0 or queried_agents.max() >= self.n_agents:
            raise ValueError("queried agent index is out of range")
        if joint_actions.min() < 0 or joint_actions.max() >= self.action_count:
            raise ValueError("joint action is out of range")
        if candidate_actions.min() < 0 or candidate_actions.max() >= self.action_count:
            raise ValueError("candidate action is out of range")

        action_latent = self.action_embedding(joint_actions.long()).clone()
        rows = torch.arange(state_latent.shape[0], device=state_latent.device)
        action_latent[rows, queried_agents] = self.mask_embedding
        features = torch.cat(
            (
                state_latent,
                action_latent.flatten(start_dim=1),
                self.action_embedding(candidate_actions),
                self.agent_embedding(queried_agents),
            ),
            dim=-1,
        )
        normalized = self.reward_trunk(features)
        return self.normalizer.denormalize(normalized) if denormalize else normalized

    def forward(
        self,
        states: torch.Tensor,
        joint_actions: torch.Tensor,
        queried_agents: torch.Tensor,
        candidate_actions: torch.Tensor,
        *,
        denormalize: bool = True,
    ) -> torch.Tensor:
        if states.ndim != 2 or states.shape[-1] != CRITIC_STATE_DIM:
            raise ValueError("states must have shape [B, 950]")
        return self._predict_from_latent(
            self.state_encoder(states),
            joint_actions,
            queried_agents,
            candidate_actions,
            denormalize=denormalize,
        )

    def factual_predictions(
        self,
        states: torch.Tensor,
        joint_actions: torch.Tensor,
        *,
        denormalize: bool = True,
    ) -> torch.Tensor:
        batch = states.shape[0]
        queries = torch.arange(self.n_agents, device=states.device).repeat(batch)
        state_latent = self.state_encoder(states)
        expanded_state_latent = state_latent[:, None, :].expand(
            -1, self.n_agents, -1
        ).reshape(batch * self.n_agents, -1)
        expanded_joint = joint_actions[:, None, :].expand(-1, self.n_agents, -1).reshape(
            batch * self.n_agents, self.n_agents
        )
        candidates = joint_actions.long().reshape(-1)
        predictions = self._predict_from_latent(
            expanded_state_latent,
            expanded_joint,
            queries,
            candidates,
            denormalize=denormalize,
        )
        return predictions.reshape(batch, self.n_agents, 1)

    @torch.no_grad()
    def predict_all_actions(
        self,
        states: torch.Tensor,
        joint_actions: torch.Tensor,
        *,
        chunk_size: int = 65_536,
    ) -> torch.Tensor:
        """Return ``[B, 4, action_count]`` predictions in bounded chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        batch = states.shape[0]
        state_latent = self.state_encoder(states)
        total = batch * self.n_agents * self.action_count
        flat_index = torch.arange(total, device=states.device)
        outputs = []
        for start in range(0, total, chunk_size):
            index = flat_index[start : start + chunk_size]
            sample = torch.div(
                index,
                self.n_agents * self.action_count,
                rounding_mode="floor",
            )
            remainder = index % (self.n_agents * self.action_count)
            query = torch.div(remainder, self.action_count, rounding_mode="floor")
            candidate = remainder % self.action_count
            outputs.append(
                self._predict_from_latent(
                    state_latent[sample],
                    joint_actions[sample],
                    query,
                    candidate,
                    denormalize=True,
                ).squeeze(-1)
            )
        return torch.cat(outputs).reshape(batch, self.n_agents, self.action_count)


def factual_reward_model_loss(
    model: CounterfactualRewardModel,
    states: torch.Tensor,
    joint_actions: torch.Tensor,
    team_rewards: torch.Tensor,
) -> torch.Tensor:
    """MSE on normalized factual team rewards, averaged over queried agents."""

    targets = model.normalizer.normalize(team_rewards).reshape(-1, 1, 1)
    predictions = model.factual_predictions(
        states,
        joint_actions,
        denormalize=False,
    )
    return (predictions - targets.expand_as(predictions)).square().mean()


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.numel() < 2 or left.std() <= 1.0e-12 or right.std() <= 1.0e-12:
        return 0.0
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1].cpu())


def spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.numel() < 2:
        return 0.0
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    return pearson_correlation(left_rank, right_rank)


def reward_validation_metrics(
    model: CounterfactualRewardModel,
    states: torch.Tensor,
    joint_actions: torch.Tensor,
    team_rewards: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        prediction = model.factual_predictions(states, joint_actions).mean(dim=1).squeeze(-1)
    target = team_rewards.reshape(-1)
    mse = float((prediction - target).square().mean().cpu())
    variance = float((target - target.mean()).square().mean().cpu())
    return {
        "mse": mse,
        "r2": 1.0 - mse / max(variance, 1.0e-12),
        "pearson": pearson_correlation(prediction, target),
        "target_std": float(target.std().cpu()),
    }
