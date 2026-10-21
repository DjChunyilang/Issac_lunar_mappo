"""Explicit S04 cost version; preserve v1/v2 and physical hard constraints."""
from dataclasses import dataclass
import math

import numpy as np
import torch

from .upstream_mppi import UpstreamMPPISettings,UpstreamMPPIAdapter
from .upstream_mppi_precision import UpstreamMPPIKernelV2,UpstreamMPPIAdapterV2


@dataclass(frozen=True)
class UpstreamMPPISettingsV3(UpstreamMPPISettings):
    goal_scale: float = 1.
    terrain_scale: float = 1.
    neighbor_scale: float = 1.
    smooth_scale: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in (
                self.goal_scale,self.terrain_scale,self.neighbor_scale,self.smooth_scale)):
            raise ValueError('Calibration cost scales must be positive finite values')


class UpstreamMPPIKernelV3(UpstreamMPPIKernelV2):
    version='upstream_mppi_kernel_cost_v3'
    cost_version='upstream_mppi_speed_normalized_cost_v2'

    @property
    def metadata(self):
        return {**super().metadata,'smooth_normalization':'physical_speed_limits_v2',
                'hard_constraints':'unchanged_actuator_footprint_v2',
                'initial_guide':'goal_distance_over_horizon_v1'}

    def seed_controls(self,requests):
        # The old initial mean always used the global speed cap, including
        # sub-meter goals beside a parked peer. Two small-noise iterations then
        # had no feasible complete sample. This is an initialization guide,
        # never a speed ceiling or an implicit terminal stopping constraint.
        command=super().seed_controls(requests)
        poses=self.tensor(np.stack([r.ego_pose for r in requests]))
        delta=self.tensor(np.stack([r.intent.goal_xy for r in requests]))-poses[:,:2]
        length=delta.norm(dim=-1);angle=torch.atan2(delta[:,1],delta[:,0])-poses[:,2]
        angle=torch.atan2(angle.sin(),angle.cos());reverse=angle.abs()>torch.pi/2
        turn=torch.where(reverse,torch.atan2((angle+torch.pi).sin(),(angle+torch.pi).cos()),angle)
        speed=length/self.spec.grid.horizon
        command[:,:,0]*=torch.minimum(torch.ones_like(speed),speed/command[:,0,0].abs().clamp_min(1e-12))[:,None]
        command[:,:,1]=(turn/self.spec.grid.horizon).clamp(-self.spec.vehicle.yaw_limit,self.spec.vehicle.yaw_limit)[:,None]
        command=torch.where((length<1e-8)[:,None,None],0.,command)
        return command

    def cost(self,requests,states,controls,terrain,neighbors):
        parts,safe=super().cost(requests,states,controls,terrain,neighbors)
        u=controls.double()
        previous=torch.as_tensor(np.stack([r.previous_control for r in requests]),device=self.device,dtype=torch.float64)
        change=torch.diff(torch.cat((previous[:,None,None].expand(-1,u.shape[1],1,-1),u),dim=2),dim=2)
        scale=u.new_tensor([self.spec.vehicle.forward_limit,self.spec.vehicle.yaw_limit])
        parts['smooth']=self.settings.smooth_scale*((change/scale).square()*u.new_tensor([.5,.2])).sum((-1,-2))
        parts['goal']*=self.settings.goal_scale;parts['terrain']*=self.settings.terrain_scale
        parts['neighbor']*=self.settings.neighbor_scale
        return parts,safe


class UpstreamMPPIAdapterV3(UpstreamMPPIAdapterV2):
    adapter_id='mppi_upstream_v3'
    state_version=3

    def __init__(self,cfg,kernel=None):
        if cfg.planner_adapter!=self.adapter_id:raise ValueError('MPPI cost v3 requires an explicit configuration')
        if kernel is not None and (type(kernel) is not UpstreamMPPIKernelV3 or kernel.spec.fingerprint!=cfg.spec.fingerprint):
            raise ValueError('MPPI v3 kernel/profile mismatch')
        UpstreamMPPIAdapter.__init__(self,cfg,kernel=kernel or UpstreamMPPIKernelV3(cfg,
            UpstreamMPPISettingsV3(samples=cfg.mppi_samples,iterations=cfg.mppi_iterations,device=cfg.planner_device)))


class UpstreamMPPIKernelV4(UpstreamMPPIKernelV3):
    version='upstream_mppi_kernel_target_v4'
    cost_version='upstream_mppi_target_normalized_cost_v3'

    @property
    def metadata(self):
        return {**super().metadata,'goal_normalization':'remaining_distance_with_capture_floor_v1',
                'goal_distance_floor':self.spec.task.capture_radius}

    def cost(self,requests,states,controls,terrain,neighbors):
        parts,safe=super().cost(requests,states,controls,terrain,neighbors)
        device=self.device
        goals=torch.as_tensor(np.stack([r.intent.goal_xy for r in requests]),device=device,dtype=torch.float64)
        poses=torch.as_tensor(np.stack([r.ego_pose for r in requests]),device=device,dtype=torch.float64)
        distance=(goals-poses[:,:2]).norm(dim=-1).clamp_min(self.spec.task.capture_radius)
        yaw=torch.as_tensor([r.ego_pose[2] if r.intent.terminal_yaw is None else r.intent.terminal_yaw for r in requests],device=device,dtype=torch.float64)
        active=torch.as_tensor([r.intent.terminal_yaw is not None for r in requests],device=device)
        yaw_cost=self.settings.goal_scale*active[:,None]*2*(1-(states.double()[:,:,-1,2]-yaw[:,None]).cos())
        parts['goal']=(parts['goal']-yaw_cost)*(self.spec.rho_max/distance[:,None]).square()+yaw_cost
        return parts,safe


class UpstreamMPPIAdapterV4(UpstreamMPPIAdapterV3):
    adapter_id='mppi_upstream_v4'
    state_version=4

    def __init__(self,cfg,kernel=None):
        if cfg.planner_adapter!=self.adapter_id:raise ValueError('MPPI target v4 requires an explicit configuration')
        if kernel is not None and (type(kernel) is not UpstreamMPPIKernelV4 or kernel.spec.fingerprint!=cfg.spec.fingerprint):
            raise ValueError('MPPI v4 kernel/profile mismatch')
        UpstreamMPPIAdapter.__init__(self,cfg,kernel=kernel or UpstreamMPPIKernelV4(cfg,
            UpstreamMPPISettingsV3(samples=cfg.mppi_samples,iterations=cfg.mppi_iterations,device=cfg.planner_device)))
