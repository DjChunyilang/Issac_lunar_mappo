"""Conditional MOVE/HOLD policy contract, independent of planner controls."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.distributions import Bernoulli, Normal
from torch.nn import functional as F

from .contracts import rotation
from .protocol import LocalIntent

ACTION_VERSION = "polar_hold_v2"
ROLLOUT_VERSION = "polar_hold_rollout_v1"


def validate_bounds(rho_max, beta_max):
    if not np.isfinite([rho_max, beta_max]).all() or rho_max <= 0 or not 0 < beta_max <= math.pi:
        raise ValueError("Expected positive finite rho_max and beta_max in (0, pi]")


def bounded_log_prob(normal, latent):
    jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    return (normal.log_prob(latent) - jacobian).sum(-1)


@dataclass(frozen=True)
class PolicyAction:
    physical: torch.Tensor | np.ndarray  # [..., 2]: rho, beta; zero placeholder for HOLD
    hold: torch.Tensor | np.ndarray  # [...], bool
    version: str = ACTION_VERSION

    def numpy(self):
        def convert(value):
            return (
                value.detach().cpu().numpy().copy()
                if torch.is_tensor(value)
                else np.array(value, copy=True)
            )

        return PolicyAction(convert(self.physical), convert(self.hold), self.version)

    def validate(self, shape, rho_max, beta_max):
        validate_bounds(rho_max, beta_max)
        action = self.numpy()
        if action.version != ACTION_VERSION or action.physical.shape != (*shape, 2):
            raise ValueError("Invalid polar action version/shape")
        if action.hold.shape != shape or action.hold.dtype != np.bool_:
            raise ValueError("HOLD must be a boolean array matching agent dimensions")
        if not np.isfinite(action.physical).all():
            raise ValueError("Nonfinite polar action")
        rho, beta = action.physical[..., 0], action.physical[..., 1]
        if (rho < 0).any() or (rho > rho_max).any() or (np.abs(beta) > beta_max).any():
            raise ValueError("Polar action out of bounds")
        if np.any(action.physical[action.hold] != 0):
            raise ValueError("HOLD continuous placeholders must be zero")
        return action

    def decode(self, poses, times, rho_max, beta_max):
        poses = np.asarray(poses)
        if poses.ndim != 3 or poses.shape[-1] != 3 or not np.isfinite(poses).all():
            raise ValueError("Expected finite [env, agent, 3] poses")
        e, a = poses.shape[:2]
        times = np.asarray(times)
        if times.shape != (e,) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Expected per-environment episode time")
        action = self.validate((e, a), rho_max, beta_max)
        intents = []
        for env in range(e):
            for agent in range(a):
                pose, now = poses[env, agent], float(times[env])
                if action.hold[env, agent]:
                    intents.append(LocalIntent.for_hold(pose, now))
                else:
                    rho, beta = action.physical[env, agent]
                    local = rho * np.array([np.cos(beta), np.sin(beta)])
                    intents.append(
                        LocalIntent(pose[:2] + rotation(pose[2]) @ local, False, now, now + 1)
                    )
        return intents


@dataclass(frozen=True)
class HybridSample:
    action: PolicyAction
    latent: torch.Tensor
    log_prob: torch.Tensor
    hidden: torch.Tensor
    rollout_version: str = ROLLOUT_VERSION

    @property
    def motion_active(self):
        return ~self.action.hold

    def rollout_fields(self):
        return {
            "action": self.action.physical,
            "hold": self.action.hold,
            "motion_active": self.motion_active,
            "latent": self.latent,
            "log_prob": self.log_prob,
        }


class PolarHoldDistribution:
    """An atom at HOLD plus a two-dimensional physical polar density for MOVE."""

    def __init__(self, mean, std, hold_logits, rho_max=1.6, beta_max=math.pi):
        validate_bounds(rho_max, beta_max)
        if mean.shape[-1] != 2 or hold_logits.shape != mean.shape[:-1]:
            raise ValueError("Conditional distribution shape mismatch")
        self.normal = Normal(mean, std)
        self.branch = Bernoulli(logits=hold_logits)
        self.logits = hold_logits
        self.rho_max, self.beta_max = rho_max, beta_max
        self.log_scale = math.log(rho_max / 2) + math.log(beta_max)

    def transform(self, latent):
        u = latent.tanh()
        return torch.stack(((u[..., 0] + 1) * (self.rho_max / 2), u[..., 1] * self.beta_max), -1)

    def log_prob(self, latent, hold):
        if (
            hold.dtype != torch.bool
            or hold.shape != self.logits.shape
            or latent.shape != self.normal.loc.shape
        ):
            raise ValueError("Conditional sample shape/dtype mismatch")
        # Do not evaluate continuous density on HOLD rows (including unused NaN placeholders).
        result = torch.where(hold, F.logsigmoid(self.logits), F.logsigmoid(-self.logits))
        active = ~hold
        if active.any():
            normal = Normal(self.normal.loc[active], self.normal.scale[active])
            result = result.clone()
            result[active] += bounded_log_prob(normal, latent[active]) - self.log_scale
        return result

    def entropy(self):
        # Current-policy expectation, independent of the rollout's sampled branch/latent.
        latent = self.normal.rsample()
        conditional = -bounded_log_prob(self.normal, latent) + self.log_scale
        return self.branch.entropy() + torch.sigmoid(-self.logits) * conditional

    def sample(self, *, deterministic=False):
        hold = self.logits > 0 if deterministic else self.branch.sample().bool()
        active = ~hold
        latent = torch.zeros_like(self.normal.loc)
        if active.any():
            normal = Normal(self.normal.loc[active], self.normal.scale[active])
            latent[active] = normal.mean if deterministic else normal.sample()
        physical = torch.where(active[..., None], self.transform(latent), 0.0)
        return PolicyAction(physical, hold), latent, self.log_prob(latent, hold)
