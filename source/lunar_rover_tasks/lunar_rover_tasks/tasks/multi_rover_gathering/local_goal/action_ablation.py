"""T11 action contracts only: shared T04 backbone, no environment/training entry.

The prepared payload is deliberately not a PlannerInput. New-scale S02/S03 and
their full acceptance evidence must exist before binding it to an executor.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .actions import PolarHoldDistribution, bounded_log_prob, validate_bounds
from .baselines import PRIMITIVE_VERSION, primitive_fingerprint
from .contracts import rotation, wrap
from .learning import RecurrentPolicy

REPRESENTATIONS = {
    "A0": "t11_polar_move_contract_v1",
    "A1": "t11_polar_hold_contract_v1",
    "A2": "t11_cartesian_request_contract_v1",
    "A3": "t11_primitive47_contract_v1",
}
ROLLOUT_VERSION = "t11_action_replay_v1"
STATE_VERSION = "t11_action_diagnostic_state_v1"


@dataclass(frozen=True)
class ActionContract:
    representation: str
    profile_fingerprint: str
    rho_max: float
    speed_cap_max: float
    beta_max: float = math.pi
    request_seconds: float = 1.0

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS:
            raise ValueError("Unknown T11 representation")
        if (not isinstance(self.profile_fingerprint, str)
                or len(self.profile_fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in self.profile_fingerprint)):
            raise ValueError("Expected immutable profile fingerprint")
        validate_bounds(self.rho_max, self.beta_max)
        if not np.isfinite([self.speed_cap_max, self.request_seconds]).all() or min(
            self.speed_cap_max, self.request_seconds
        ) <= 0:
            raise ValueError("Invalid speed/request bound")

    @classmethod
    def from_spec(cls, representation, spec):
        return cls(representation, spec.fingerprint, spec.rho_max,
                   spec.vehicle.forward_limit, request_seconds=spec.grid.request_validity)

    @property
    def version(self):
        return REPRESENTATIONS[self.representation]

    @property
    def continuous_dim(self):
        return {"A0": 2, "A1": 2, "A2": 4, "A3": 0}[self.representation]

    @property
    def metadata(self):
        return {
            **asdict(self), "action_version": self.version,
            "rollout_version": ROLLOUT_VERSION,
            "scope": "probability_contract_only", "training_eligible": False,
            "probability_measure": {"A0": "physical_rho_beta", "A1": "hold_atom_and_physical_rho_beta",
                                    "A2": "normalized_cartesian_yaw_speed", "A3": "native_index"}[self.representation],
            "primitive_version": PRIMITIVE_VERSION if self.representation == "A3" else None,
            "primitive_fingerprint": primitive_fingerprint() if self.representation == "A3" else None,
        }


@dataclass(frozen=True)
class ActionPayload:
    """Original policy output; continuous tensor is empty for native categorical."""
    continuous: torch.Tensor
    hold: torch.Tensor
    primitive_id: torch.Tensor
    action_version: str
    primitive_version: str | None = None
    primitive_fingerprint: str | None = None

    def validate(self, contract, shape):
        if self.action_version != contract.version:
            raise ValueError("Action payload version mismatch")
        if (self.continuous.shape != (*shape, contract.continuous_dim)
                or self.hold.shape != shape or self.hold.dtype != torch.bool
                or self.primitive_id.shape != shape or self.primitive_id.dtype != torch.int64):
            raise ValueError("Action payload shape/dtype mismatch")
        if not torch.is_floating_point(self.continuous) or not torch.isfinite(self.continuous).all():
            raise ValueError("Nonfinite/nonfloating continuous action")
        if not (self.continuous.device == self.hold.device == self.primitive_id.device):
            raise ValueError("Mixed action devices")
        if contract.representation == "A3":
            if (self.primitive_version != PRIMITIVE_VERSION
                    or self.primitive_fingerprint != primitive_fingerprint()
                    or self.hold.any() or ((self.primitive_id < 0) | (self.primitive_id >= 47)).any()):
                raise ValueError("Invalid native primitive payload")
        else:
            if self.primitive_version is not None or self.primitive_fingerprint is not None:
                raise ValueError("Unexpected primitive library identity")
            if (self.primitive_id != -1).any():
                raise ValueError("Unexpected primitive payload")
            if contract.representation != "A1" and self.hold.any():
                raise ValueError("This representation has no explicit HOLD")
            if (self.continuous[self.hold] != 0).any():
                raise ValueError("HOLD placeholders must be zero")
            if contract.representation == "A2":
                if (self.continuous.abs() > 1).any():
                    raise ValueError("Normalized Cartesian action out of bounds")
            elif ((self.continuous[..., 0] < 0).any()
                  or (self.continuous[..., 0] > contract.rho_max).any()
                  or (self.continuous[..., 1].abs() > contract.beta_max).any()):
                raise ValueError("Polar action out of bounds")


@dataclass(frozen=True)
class PreparedRequest:
    """Frozen decision semantics; not yet an executable planning protocol."""
    goal_xy: tuple[float, float] | None
    hold: bool
    terminal_yaw: float | None
    speed_cap: float | None
    primitive_id: int | None
    timestamp: float
    expires_at: float
    action_version: str
    primitive_version: str | None = None
    primitive_fingerprint: str | None = None


def decode_payload(payload, contract, poses, times):
    poses, times = np.asarray(poses, dtype=float), np.asarray(times, dtype=float)
    if (poses.ndim != 3 or poses.shape[-1] != 3 or not np.isfinite(poses).all()
            or times.shape != poses.shape[:1] or not np.isfinite(times).all() or (times < 0).any()):
        raise ValueError("Expected finite [env,agent,3] poses and per-env times")
    payload.validate(contract, poses.shape[:2])
    continuous = payload.continuous.detach().cpu().numpy()
    hold = payload.hold.detach().cpu().numpy()
    ids = payload.primitive_id.detach().cpu().numpy()
    result = []
    for env in range(poses.shape[0]):
        for agent in range(poses.shape[1]):
            pose, now = poses[env, agent], float(times[env])
            xy, yaw, speed, primitive = None, None, None, None
            if contract.representation == "A3":
                # ID 0 retains the library's HOLD semantics, without polar conversion.
                primitive = int(ids[env, agent])
            elif hold[env, agent]:
                xy, yaw, speed = tuple(pose[:2]), float(pose[2]), 0.0
            else:
                action = continuous[env, agent]
                if contract.representation == "A2":
                    local = contract.rho_max * action[:2]
                    yaw = float(wrap(pose[2] + math.pi * action[2]))
                    speed = float(contract.speed_cap_max * (action[3] + 1) / 2)
                else:
                    rho, beta = action
                    local = rho * np.array([np.cos(beta), np.sin(beta)])
                xy = tuple(pose[:2] + rotation(pose[2]) @ local)
            result.append(PreparedRequest(xy, bool(hold[env, agent]), yaw, speed, primitive,
                                          now, now + contract.request_seconds, contract.version,
                                          payload.primitive_version, payload.primitive_fingerprint))
    return result


class BoundedMoveDistribution:
    def __init__(self, mean, std, contract):
        if mean.shape[-1] != contract.continuous_dim or contract.representation not in ("A0", "A2"):
            raise ValueError("Bounded MOVE distribution shape/representation mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid continuous distribution parameters")
        self.normal, self.contract = Normal(mean, std), contract

    def transform(self, latent):
        action = latent.tanh()
        if self.contract.representation == "A0":
            action = torch.stack(((action[..., 0] + 1) * self.contract.rho_max / 2,
                                  action[..., 1] * self.contract.beta_max), -1)
        return action

    def log_prob(self, latent):
        if latent.shape != self.normal.loc.shape or not torch.isfinite(latent).all():
            raise ValueError("Invalid original MOVE latent")
        log = bounded_log_prob(self.normal, latent)
        if self.contract.representation == "A0":
            log = log - math.log(self.contract.rho_max / 2) - math.log(self.contract.beta_max)
        return log

    def entropy(self):
        return -self.log_prob(self.normal.rsample())


@dataclass(frozen=True)
class AblationSample:
    action: ActionPayload
    latent: torch.Tensor
    log_prob: torch.Tensor
    hidden: torch.Tensor
    rollout_version: str = ROLLOUT_VERSION


class AblationPolicy(RecurrentPolicy):
    """Same encoder/GRU/critic as A1; only action heads vary. CPU diagnostic use."""
    def __init__(self, critic_dim, contract, *, terrain_channels=2):
        super().__init__(critic_dim, terrain_channels=terrain_channels,
                         observation_version="local_intent_clock_v3", action_version="polar_hold_v2",
                         rho_max=contract.rho_max, beta_max=contract.beta_max)
        self.contract = contract
        self.diagnostic_only = True
        self.hybrid = False  # Deliberately incompatible with legacy trainer/ppo_update.
        self.action_version = contract.version
        self.critic_dim = critic_dim
        if contract.representation != "A1":
            del self.hold_logit
        if contract.representation == "A2":
            self.mean = nn.Linear(128, 4)
            nn.init.orthogonal_(self.mean.weight, gain=0.01)
            nn.init.zeros_(self.mean.bias)
            self.log_std = nn.Parameter(torch.full((4,), -0.5))
        elif contract.representation == "A3":
            del self.mean, self.log_std
            self.primitive_logits = nn.Linear(128, 47)
            nn.init.orthogonal_(self.primitive_logits.weight, gain=0.01)
            nn.init.zeros_(self.primitive_logits.bias)

    @property
    def actor_dim(self):
        return 66 + 112 * self.terrain_channels + 5 + 116

    def enable_value_normalization(self):
        from .upstream_value_normalization import RunningValueNormalization
        if hasattr(self, 'value_normalizer'):
            raise ValueError('Value normalization is already configured')
        self.value_normalizer = RunningValueNormalization().to(next(self.parameters()).device)

    def value(self, state):
        value = super().value(state)
        if hasattr(self, 'value_normalizer'):
            value = self.value_normalizer.denormalize(value)
        return value

    def distribution(self, obs, hidden):
        if (obs.shape[-1] != self.actor_dim or hidden.shape != (*obs.shape[:-1], 128)
                or not torch.isfinite(obs).all() or not torch.isfinite(hidden).all()):
            raise ValueError("Invalid T11 synthetic observation/hidden state")
        return super().distribution(obs, hidden)

    def action_distribution(self, hidden):
        if hidden.shape[-1] != 128 or not torch.isfinite(hidden).all():
            raise ValueError("Invalid T11 action-head hidden state")
        rep = self.contract.representation
        if rep == "A3":
            return Categorical(logits=self.primitive_logits(hidden))
        mean, std = self.mean(hidden), self.log_std.clamp(-5, 1).exp()
        if rep == "A1":
            return PolarHoldDistribution(mean, std, self.hold_logit(hidden).squeeze(-1),
                                         self.contract.rho_max, self.contract.beta_max)
        return BoundedMoveDistribution(mean, std, self.contract)

    def sample(self, obs, hidden, deterministic=False):
        distribution, following = self.distribution(obs, hidden)
        shape = following.shape[:-1]
        hold = torch.zeros(shape, dtype=torch.bool, device=following.device)
        ids = torch.full(shape, -1, dtype=torch.int64, device=following.device)
        if self.contract.representation == "A1":
            action, latent, log = distribution.sample(deterministic=deterministic)
            continuous, hold = action.physical, action.hold
        elif self.contract.representation == "A3":
            ids = distribution.logits.argmax(-1) if deterministic else distribution.sample()
            continuous = latent = following.new_empty((*shape, 0))
            log = distribution.log_prob(ids)
        else:
            latent = distribution.normal.mean if deterministic else distribution.normal.sample()
            continuous, log = distribution.transform(latent), distribution.log_prob(latent)
        native = self.contract.representation == "A3"
        return AblationSample(ActionPayload(continuous, hold, ids, self.contract.version,
                                           PRIMITIVE_VERSION if native else None,
                                           primitive_fingerprint() if native else None),
                              latent, log, following)

    def evaluate_sample(self, hidden, sample):
        if sample.rollout_version != ROLLOUT_VERSION:
            raise ValueError("T11 rollout version mismatch")
        sample.action.validate(self.contract, hidden.shape[:-1])
        if (sample.latent.shape != (*hidden.shape[:-1], self.contract.continuous_dim)
                or not torch.isfinite(sample.latent).all()):
            raise ValueError("T11 original latent shape/value mismatch")
        distribution = self.action_distribution(hidden)
        if self.contract.representation == "A3":
            return distribution.log_prob(sample.action.primitive_id), distribution.entropy()
        expected = distribution.transform(sample.latent)
        if self.contract.representation == "A1":
            if (sample.latent[sample.action.hold] != 0).any():
                raise ValueError("HOLD latent placeholders must be zero")
            expected = torch.where(sample.action.hold[..., None], 0.0, expected)
        if not torch.allclose(sample.action.continuous, expected, atol=1e-6, rtol=1e-6):
            raise ValueError("Action changed after original sampling")
        log = (distribution.log_prob(sample.latent, sample.action.hold)
               if self.contract.representation == "A1" else distribution.log_prob(sample.latent))
        return log, distribution.entropy()

    @property
    def diagnostic_metadata(self):
        result = {**self.contract.metadata, "state_version": STATE_VERSION,
                "encoder_version": "t04_clock_backbone_contract_v1", "critic_dim": self.critic_dim,
                "actor_dim": self.actor_dim,
                "terrain_channels": self.terrain_channels,
                "observation_scope": "synthetic_t04_layout_not_new_scale_environment"}
        if hasattr(self, 'value_normalizer'):
            from .upstream_value_normalization import VALUE_NORMALIZATION_VERSION
            result['value_normalization'] = VALUE_NORMALIZATION_VERSION
        return result

    def export_diagnostic_state(self):
        return {"metadata": self.diagnostic_metadata,
                "weights": {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}}

    def load_diagnostic_state(self, state):
        if not isinstance(state, dict) or state.get("metadata") != self.diagnostic_metadata:
            raise ValueError("T11 diagnostic contract/version mismatch")
        weights, expected = state.get("weights"), self.state_dict()
        if not isinstance(weights, dict) or set(weights) != set(expected):
            raise ValueError("T11 diagnostic state keys mismatch")
        if any(not torch.is_tensor(value) or value.shape != expected[key].shape
               or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
               for key, value in weights.items()):
            raise ValueError("T11 diagnostic state tensor mismatch")
        # All validation precedes mutation; do not import optimizer/RNG/checkpoint semantics.
        self.load_state_dict(deepcopy(weights), strict=True)


def build_ablation_policies(critic_dim, spec, *, terrain_channels=2):
    """Copy one initialized backbone/critic; do not rely on RNG consumption order."""
    policies = {rep: AblationPolicy(critic_dim, ActionContract.from_spec(rep, spec),
                                   terrain_channels=terrain_channels) for rep in REPRESENTATIONS}
    head_prefixes = ("mean.", "hold_logit.", "primitive_logits.", "log_std")
    common = {key: value for key, value in policies["A1"].state_dict().items()
              if not key.startswith(head_prefixes)}
    for policy in policies.values():
        policy.load_state_dict({**policy.state_dict(), **common})
    return policies


def replay_action_sequence(policy, observations, initial_hidden, samples, learning_valid, episode_ended):
    """Differentiable probability replay, without an optimizer or environment.

    Invalid environments are removed *before* encoding/distribution evaluation.
    Tail hidden state includes real episode resets, but no buffer-cut reset.
    Stored samples and the sequence's initial state are fixed rollout data;
    gradients follow only the current policy's reconstruction of the sequence.
    """
    if (observations.ndim != 4 or initial_hidden.shape != (*observations.shape[1:3], 128)
            or learning_valid.shape != observations.shape[:2]
            or episode_ended.shape != learning_valid.shape
            or learning_valid.dtype != torch.bool or episode_ended.dtype != torch.bool
            or len(samples) != len(observations)):
        raise ValueError("T11 recurrent replay shape/dtype mismatch")
    hidden = initial_hidden.detach()
    logs, ratios, entropies = [], [], []
    for obs, sample, valid, ended in zip(observations, samples, learning_valid, episode_ended):
        log, ratio, entropy = [obs.new_full(obs.shape[:2], x) for x in (0.0, 1.0, 0.0)]
        following = torch.zeros_like(hidden)
        if valid.any():
            if not torch.isfinite(obs[valid]).all() or not torch.isfinite(hidden[valid]).all():
                raise ValueError("Nonfinite valid replay input")
            if sample.rollout_version != ROLLOUT_VERSION:
                raise ValueError("T11 rollout version mismatch")
            action = sample.action
            selected_action = ActionPayload(action.continuous[valid].detach(), action.hold[valid],
                                           action.primitive_id[valid], action.action_version,
                                           action.primitive_version, action.primitive_fingerprint)
            selected = AblationSample(selected_action, sample.latent[valid].detach(),
                                      sample.log_prob[valid].detach(), sample.hidden[valid].detach(),
                                      sample.rollout_version)
            _, next_hidden = policy.distribution(obs[valid], hidden[valid])
            new_log, current_entropy = policy.evaluate_sample(next_hidden, selected)
            if selected.log_prob.shape != new_log.shape or not torch.isfinite(selected.log_prob).all():
                raise ValueError("Invalid original log probability")
            log[valid], entropy[valid] = new_log, current_entropy
            ratio[valid] = (new_log - selected.log_prob).exp()
            if not torch.isfinite(ratio[valid]).all():
                raise FloatingPointError("Nonfinite T11 PPO ratio")
            following[valid] = next_hidden
        hidden = torch.where((ended | ~valid)[:, None, None], 0.0, following)
        logs.append(log)
        ratios.append(ratio)
        entropies.append(entropy)
    shape = observations.shape[:3]
    return {"log_prob": torch.stack(logs) if logs else observations.new_empty(shape),
            "ratio": torch.stack(ratios) if ratios else observations.new_empty(shape),
            "entropy": torch.stack(entropies) if entropies else observations.new_empty(shape),
            "tail_hidden": hidden}


def clipped_actor_objective(ratios, advantages, learning_valid, clip=0.15):
    """Diagnostic PPO surrogate; no updating parameters or comparing raw entropy."""
    if (ratios.ndim != 3 or advantages.ndim != 2 or learning_valid.ndim != 2
            or ratios.shape[:2] != advantages.shape or learning_valid.shape != advantages.shape
            or learning_valid.dtype != torch.bool or not math.isfinite(clip) or not 0 < clip < 1):
        raise ValueError("Invalid T11 surrogate inputs")
    selected = ratios[learning_valid]
    adv = advantages[learning_valid].detach().unsqueeze(-1)
    if not torch.isfinite(selected).all() or not torch.isfinite(adv).all() or (selected < 0).any():
        raise ValueError("Nonfinite valid T11 surrogate")
    if not learning_valid.any():
        return ratios.new_zeros(())
    return -torch.minimum(selected * adv, selected.clamp(1 - clip, 1 + clip) * adv).mean()
