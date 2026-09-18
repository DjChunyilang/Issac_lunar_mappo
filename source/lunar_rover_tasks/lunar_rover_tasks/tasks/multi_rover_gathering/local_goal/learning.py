"""Shared recurrent actor / centralized critic MAPPO for REQUEST actions."""

import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def bounded_log_prob(normal, latent):
    # Stable exact change of variables for tanh; never infer actions from controls.
    log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    return (normal.log_prob(latent) - log_jacobian).sum(-1)


class RecurrentPolicy(nn.Module):
    def __init__(self, critic_dim):
        super().__init__()
        # Same shared two-convolution N1 multiscale encoder as the legacy route.
        self.terrain = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ELU(), nn.Conv2d(16, 32, 3, padding=1), nn.ELU()
        )
        self.terrain_projection = nn.Sequential(nn.Linear(864, 64), nn.ELU())
        self.ego = nn.Sequential(nn.Linear(15, 32), nn.ELU())
        self.neighbors = nn.Sequential(nn.Linear(51, 48), nn.ELU())
        self.aggregation = nn.Sequential(nn.Linear(5, 16), nn.ELU())
        self.feedback = nn.Sequential(nn.Linear(115, 64), nn.ELU())
        self.trunk = nn.Sequential(nn.Linear(224, 128), nn.ELU())
        self.gru = nn.GRUCell(128, 128)
        self.mean = nn.Linear(128, 4)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((4,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
        )

    def encode(self, obs):
        shape = obs.shape[:-1]
        obs = obs.reshape(-1, obs.shape[-1])
        encoded = []
        for start, end, nx, ny in [(66, 192, 7, 9), (192, 234, 3, 7), (234, 290, 4, 7)]:
            grid = obs[:, start:end].reshape(-1, nx, ny, 2).permute(0, 3, 1, 2)
            encoded.append(F.adaptive_avg_pool2d(self.terrain(grid), (3, 3)).flatten(1))
        features = torch.cat(
            [
                self.ego(obs[:, :15]),
                self.neighbors(obs[:, 15:66]),
                self.aggregation(obs[:, 290:295]),
                self.terrain_projection(torch.cat(encoded, -1)),
                self.feedback(obs[:, 295:]),
            ],
            -1,
        )
        return self.trunk(features).reshape(*shape, 128)

    def distribution(self, obs, hidden):
        hidden = self.gru(self.encode(obs).reshape(-1, 128), hidden.reshape(-1, 128)).reshape_as(
            hidden
        )
        return Normal(self.mean(hidden), self.log_std.clamp(-5, 1).exp()), hidden

    def sample(self, obs, hidden, deterministic=False):
        normal, hidden = self.distribution(obs, hidden)
        latent = normal.mean if deterministic else normal.sample()
        return latent.tanh(), latent, bounded_log_prob(normal, latent), hidden

    def value(self, state):
        # No hard-clipping of predicted values or targets.
        return self.critic(state).squeeze(-1)


def duration_gae(
    rewards, values, next_values, durations, terminated, truncated, gamma=0.99, lam=0.95
):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma ** durations[t] * (~terminated[t]) * next_values[t] - values[t]
        carry = delta + (gamma * lam) ** durations[t] * (~(terminated[t] | truncated[t])) * carry
        advantages[t] = carry
    return advantages, advantages + values


def ppo_update(policy, optimizer, batch, *, epochs=5, clip=0.15, entropy_weight=0.0009):
    advantage, returns = duration_gae(
        batch["reward"],
        batch["value"],
        batch["next_value"],
        batch["duration"],
        batch["terminated"],
        batch["truncated"],
    )
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    length = len(advantage)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        for start in range(0, length, 16):
            end = min(start + 16, length)
            # Sample-time initial GRU state at each sequence boundary, never zeros by convenience.
            hidden = batch["hidden"][start].detach()
            encoded = policy.encode(batch["obs"][start:end])
            logs, entropies = [], []
            for j, t in enumerate(range(start, end)):
                if j:
                    done = batch["terminated"][t - 1] | batch["truncated"][t - 1]
                    hidden = hidden * (~done)[:, None, None]
                hidden = policy.gru(
                    encoded[j].reshape(-1, 128), hidden.reshape(-1, 128)
                ).reshape_as(hidden)
                normal = Normal(policy.mean(hidden), policy.log_std.clamp(-5, 1).exp())
                logs.append(bounded_log_prob(normal, batch["latent"][t]))
                entropies.append(-bounded_log_prob(normal, normal.rsample()))
            log = torch.stack(logs)
            ratio = (log - batch["log_prob"][start:end]).exp()
            adv = advantage[start:end, :, None]
            actor = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            value = policy.value(batch["state"][start:end])
            critic = F.mse_loss(value, returns[start:end])
            entropy = torch.stack(entropies).mean()
            loss = (actor + 0.5 * critic - entropy_weight * entropy) * (end - start) / length
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
