"""Optional local parking prior, inside the A1 sampling/replay distribution.

This is independent Bernoulli exploration conditioned on each rover's legal
observation, not a joint action override or a parking safety certificate.
"""
from dataclasses import asdict, dataclass
import math

import torch


VERSION = 'local_parking_logit_prior_v1'


@dataclass(frozen=True)
class ParkingExploration:
    full_radius: float
    dmax: float
    dispersion: float
    start_logit: float = 2.4
    continuation_logit: float = 1.6
    strength: float = 1.
    spatial_scaling: str = 'log1p'

    def __post_init__(self):
        values = (self.full_radius, self.dmax, self.dispersion,
                  self.start_logit, self.continuation_logit, self.strength)
        if (not all(math.isfinite(x) for x in values) or min(values[:3]) <= 0
                or min(values[3:5]) < 0 or not 0 <= self.strength <= 1
                or self.spatial_scaling != 'log1p'):
            raise ValueError('Invalid local parking exploration settings')

    @classmethod
    def from_spec(cls, spec, *, spatial_scaling):
        return cls(spec.communication.full_radius, spec.task.dmax,
                   spec.task.dispersion, spatial_scaling=spatial_scaling)

    @property
    def metadata(self):
        return {'version': VERSION, **asdict(self)}

    def compact_geometry(self, observation):
        if observation.shape[-1] != 411:
            raise ValueError('Parking exploration requires the 411-slot local observation')
        neighbors = observation[..., 15:66].reshape(*observation.shape[:-1], 3, 17)
        valid = ((observation[..., [381, 391, 401]] > .5)
                 & (neighbors[..., 11] > .5))
        # Mask missing/stale/SPARSE coordinates before the inverse log transform.
        scaled = torch.where(valid[..., None], neighbors[..., :2], 0.)
        relative = (scaled.sign() * self.dmax
                    * torch.expm1(scaled.abs() * math.log1p(self.full_radius / self.dmax)))
        points = torch.cat((torch.zeros_like(relative[..., :1, :]), relative), dim=-2)
        distances2 = (points[..., :, None, :] - points[..., None, :, :]).square().sum(-1)
        dispersion = (points - points.mean(-2, keepdim=True)).square().sum(-1).mean(-1)
        return (valid.all(-1) & (distances2.amax(dim=(-2, -1)) <= self.dmax ** 2)
                & (dispersion <= self.dispersion))

    def logit_bias(self, observation):
        compact = self.compact_geometry(observation)
        continuing = observation[..., 14] > .5  # Current, valid ego HOLD intent.
        return (compact.to(observation.dtype) * self.strength
                * (self.start_logit + self.continuation_logit * continuing.to(observation.dtype)))


def exploration_strength(actual_base_steps):
    """Frozen within each PPO batch: full through 3k, fade to zero at 8k."""
    if actual_base_steps < 0:
        raise ValueError('Negative physical step count')
    return min(1., max(0., (8000 - actual_base_steps) / 5000))
