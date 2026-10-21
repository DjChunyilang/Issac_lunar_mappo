"""Shared recurrent actor / centralized critic MAPPO for REQUEST actions."""

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


from .actions import (
    ACTION_VERSION,
    ROLLOUT_VERSION,
    HybridSample,
    PolarHoldDistribution,
    bounded_log_prob,
    validate_bounds,
)


class RecurrentPolicy(nn.Module):
    def __init__(
        self,
        critic_dim,
        *,
        terrain_channels=2,
        observation_version="local_goal_v1",
        action_version="local_goal_request_v1",
        rho_max=1.6,
        beta_max=3.141592653589793,
    ):
        super().__init__()
        if terrain_channels not in (2, 3):
            raise ValueError("Expected legacy or validity-aware terrain channels")
        if observation_version not in (
            "local_goal_v1",
            "local_goal_clock_v2",
            "local_intent_clock_v3",
        ):
            raise ValueError("Unsupported local-goal observation version")
        if action_version not in ("local_goal_request_v1", ACTION_VERSION):
            raise ValueError("Unsupported action version")
        self.hybrid = action_version == ACTION_VERSION
        if self.hybrid != (observation_version == "local_intent_clock_v3"):
            raise ValueError("Policy action/observation version mismatch")
        validate_bounds(rho_max, beta_max)
        self.action_version = action_version
        self.rho_max, self.beta_max = rho_max, beta_max
        self.observation_version = observation_version
        self.terrain_channels = terrain_channels
        # Same shared two-convolution N1 multiscale encoder as the legacy route.
        self.terrain = nn.Sequential(
            nn.Conv2d(terrain_channels, 16, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ELU(),
        )
        self.terrain_projection = nn.Sequential(nn.Linear(864, 64), nn.ELU())
        self.ego = nn.Sequential(nn.Linear(15, 32), nn.ELU())
        self.neighbors = nn.Sequential(nn.Linear(51, 48), nn.ELU())
        self.aggregation = nn.Sequential(nn.Linear(5, 16), nn.ELU())
        self.feedback = nn.Sequential(
            nn.Linear(115 + int(observation_version != "local_goal_v1"), 64), nn.ELU()
        )
        self.trunk = nn.Sequential(nn.Linear(224, 128), nn.ELU())
        self.gru = nn.GRUCell(128, 128)
        self.mean = nn.Linear(128, 2 if self.hybrid else 4)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((2 if self.hybrid else 4,), -0.5))
        if self.hybrid:
            self.hold_logit = nn.Linear(128, 1)
            nn.init.orthogonal_(self.hold_logit.weight, gain=0.01)
            nn.init.zeros_(self.hold_logit.bias)
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
        )

    def encode(self, obs):
        shape = obs.shape[:-1]
        obs = obs.reshape(-1, obs.shape[-1])
        encoded = []
        start = 66
        for nx, ny in [(7, 9), (3, 7), (4, 7)]:
            end = start + nx * ny * self.terrain_channels
            grid = obs[:, start:end].reshape(-1, nx, ny, self.terrain_channels).permute(0, 3, 1, 2)
            start = end
            encoded.append(F.adaptive_avg_pool2d(self.terrain(grid), (3, 3)).flatten(1))
        features = torch.cat(
            [
                self.ego(obs[:, :15]),
                self.neighbors(obs[:, 15:66]),
                self.aggregation(obs[:, start : start + 5]),
                self.terrain_projection(torch.cat(encoded, -1)),
                self.feedback(obs[:, start + 5 :]),
            ],
            -1,
        )
        return self.trunk(features).reshape(*shape, 128)

    def distribution(self, obs, hidden):
        hidden = self.gru(self.encode(obs).reshape(-1, 128), hidden.reshape(-1, 128)).reshape_as(
            hidden
        )
        return self.action_distribution(hidden), hidden

    def action_distribution(self, hidden):
        mean, std = self.mean(hidden), self.log_std.clamp(-5, 1).exp()
        if self.hybrid:
            return PolarHoldDistribution(
                mean, std, self.hold_logit(hidden).squeeze(-1), self.rho_max, self.beta_max
            )
        return Normal(mean, std)

    def sample(self, obs, hidden, deterministic=False):
        normal, hidden = self.distribution(obs, hidden)
        if self.hybrid:
            action, latent, log = normal.sample(deterministic=deterministic)
            return HybridSample(action, latent, log, hidden)
        latent = normal.mean if deterministic else normal.sample()
        return latent.tanh(), latent, bounded_log_prob(normal, latent), hidden

    def value(self, state):
        # No hard-clipping of predicted values or targets.
        return self.critic(state).squeeze(-1)


def bootstrap_values(policy, states, allowed):
    """Never evaluate a terminal/invalid successor (masking NaN after V is too late)."""
    result = torch.zeros(states.shape[:-1], device=states.device, dtype=states.dtype)
    if allowed.any():
        result[allowed] = policy.value(states[allowed])
    return result


def duration_gae(
    rewards,
    values,
    next_values,
    durations,
    terminated,
    truncated,
    gamma=0.99,
    lam=0.95,
    *,
    bootstrap_allowed=None,
    learning_valid=None,
):
    valid = torch.ones_like(terminated) if learning_valid is None else learning_valid
    bootstrap = ~terminated if bootstrap_allowed is None else bootstrap_allowed
    bootstrap = bootstrap & valid & ~terminated
    safe_values = torch.where(valid, values, 0.0)
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = (
            torch.where(valid[t], rewards[t], 0.0)
            + gamma ** durations[t] * torch.where(bootstrap[t], next_values[t], 0.0)
            - safe_values[t]
        )
        carry = (
            delta
            + (gamma * lam) ** durations[t] * (~(terminated[t] | truncated[t]) & valid[t]) * carry
        )
        carry = torch.where(valid[t], carry, 0.0)
        advantages[t] = carry
    return advantages, advantages + safe_values


def ppo_update(policy, optimizer, batch, *, epochs=5, clip=0.15, entropy_weight=0.0009):
    if getattr(policy, "diagnostic_only", False):
        raise ValueError("Action ablation preparation cannot enter legacy PPO training")
    hybrid = getattr(policy, "hybrid", False)
    if hybrid and batch.get("rollout_version") != ROLLOUT_VERSION:
        raise ValueError("Hybrid rollout version mismatch")
    advantage, returns = duration_gae(
        batch["reward"],
        batch["value"],
        batch["next_value"],
        batch["duration"],
        batch["terminated"],
        batch["truncated"],
        bootstrap_allowed=batch.get("bootstrap_allowed"),
        learning_valid=batch.get("learning_valid"),
    )
    valid = batch.get("learning_valid", torch.ones_like(batch["terminated"]))
    if not valid.any():
        return {"actor_loss": 0.0, "critic_mse": 0.0, "bounded_entropy": 0.0}
    selected = advantage[valid]
    advantage = torch.where(
        valid, (advantage - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6), 0.0
    )
    # Exclude invalid inputs before any network/distribution operation, not only from losses.
    batch = dict(batch)
    keys = ("obs", "state", "hidden", "latent", "log_prob")
    if hybrid:
        if batch["hold"].dtype != torch.bool or batch["motion_active"].dtype != torch.bool:
            raise ValueError("Hybrid rollout branch masks must be boolean")
        keys += ("action", "hold", "motion_active")
    for key in keys:
        mask = valid.reshape(*valid.shape, *([1] * (batch[key].ndim - valid.ndim)))
        fill = torch.ones_like(batch[key]) if key == "hold" else torch.zeros_like(batch[key])
        batch[key] = torch.where(mask, batch[key], fill)
    if hybrid:
        shape = batch["obs"].shape[:-1]
        if (
            batch["latent"].shape != (*shape, 2)
            or batch["action"].shape != (*shape, 2)
            or batch["hold"].shape != shape
            or batch["motion_active"].shape != shape
            or not torch.equal(batch["motion_active"], ~batch["hold"])
        ):
            raise ValueError("Hybrid rollout action/mask shape mismatch")
        active = batch["motion_active"]
        latent, action = batch["latent"], batch["action"]
        if not torch.isfinite(latent).all() or not torch.isfinite(action).all():
            raise ValueError("Nonfinite valid rollout action")
        if (latent[~active] != 0).any() or (action[~active] != 0).any():
            raise ValueError("HOLD rollout placeholders must be zero")
        expected = torch.stack(
            (
                (latent[..., 0].tanh() + 1) * (policy.rho_max / 2),
                latent[..., 1].tanh() * policy.beta_max,
            ),
            -1,
        )
        if not torch.allclose(action[active], expected[active], atol=1e-6, rtol=1e-6):
            raise ValueError("Rollout action does not match original latent")
    length = len(advantage)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        for start in range(0, length, 16):
            end = min(start + 16, length)
            selected_valid = valid[start:end]
            if not selected_valid.any():
                continue
            # Sample-time initial GRU state at each sequence boundary, never zeros by convenience.
            hidden = batch["hidden"][start].detach()
            encoded = policy.encode(batch["obs"][start:end])
            logs, entropies = [], []
            for j, t in enumerate(range(start, end)):
                if j:
                    done = batch["terminated"][t - 1] | batch["truncated"][t - 1] | ~valid[t - 1]
                    hidden = hidden * (~done)[:, None, None]
                hidden = policy.gru(
                    encoded[j].reshape(-1, 128), hidden.reshape(-1, 128)
                ).reshape_as(hidden)
                if hybrid:
                    distribution = policy.action_distribution(hidden)
                    logs.append(distribution.log_prob(batch["latent"][t], batch["hold"][t]))
                    entropies.append(distribution.entropy())
                else:
                    normal = Normal(policy.mean(hidden), policy.log_std.clamp(-5, 1).exp())
                    logs.append(bounded_log_prob(normal, batch["latent"][t]))
                    entropies.append(-bounded_log_prob(normal, normal.rsample()))
            log = torch.stack(logs)
            ratio = (log - batch["log_prob"][start:end]).exp()
            adv = advantage[start:end, :, None]
            actor = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)[
                selected_valid
            ].mean()
            value = policy.value(batch["state"][start:end])
            critic = F.mse_loss(value[selected_valid], returns[start:end][selected_valid])
            entropy = torch.stack(entropies)[selected_valid].mean()
            loss = (
                (actor + 0.5 * critic - entropy_weight * entropy)
                * selected_valid.sum()
                / valid.sum()
            )
            loss.backward()
            losses.append([actor.item(), critic.item(), entropy.item()])
        norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite PPO gradients")
        optimizer.step()
    return {
        "actor_loss": sum(x[0] for x in losses) / len(losses),
        "critic_mse": sum(x[1] for x in losses) / len(losses),
        "bounded_entropy": sum(x[2] for x in losses) / len(losses),
    }
