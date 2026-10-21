"""Running value targets; public policy values remain in reward units.

Each valid team return enters the moments once per rollout, before PPO epochs.
This is target normalization, not PopArt or a modification of environment reward.
"""
import torch
from torch import nn

VALUE_NORMALIZATION_VERSION = 'running_team_return_moments_v1'


class RunningValueNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('count', torch.zeros((), dtype=torch.int64))
        self.register_buffer('mean', torch.zeros((), dtype=torch.float64))
        self.register_buffer('m2', torch.zeros((), dtype=torch.float64))

    @property
    def std(self):
        # Identity before the first rollout; bounded scale for constant targets.
        return torch.where(self.count > 0, (self.m2 / self.count.clamp_min(1)).clamp_min(.01).sqrt(),
                           torch.ones_like(self.mean))

    @torch.no_grad()
    def update(self, returns):
        values = returns.detach().to(device=self.mean.device, dtype=torch.float64).reshape(-1)
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError('Value moments require nonempty finite valid returns')
        batch_mean = values.mean()
        delta = batch_mean - self.mean
        total = self.count + values.numel()
        self.m2.add_((values - batch_mean).square().sum()
                     + delta.square() * self.count * values.numel() / total)
        self.mean.add_(delta * values.numel() / total)
        self.count.copy_(total)

    def normalize(self, values):
        return (values - self.mean.to(values)) / self.std.to(values)

    def denormalize(self, values):
        return values * self.std.to(values) + self.mean.to(values)
