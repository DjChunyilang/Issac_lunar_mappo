"""Explicit S04 cost version; preserve v1/v2 and physical hard constraints."""
from copy import copy
from dataclasses import dataclass,replace
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
    footprint_margin_m: float | None = None
    footprint_cover_spacing_m: float | None = None
    large_crater_query: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.large_crater_query not in (None,'observed_grid_sat_v1'):
            raise ValueError('Unsupported MPPI large-crater query')
        if self.large_crater_query is not None and self.footprint_margin_m!=0.:
            raise ValueError('Observed-grid SAT uses the actual body with zero extra margin')
        if any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in (
                self.goal_scale,self.terrain_scale,self.neighbor_scale,self.smooth_scale)):
            raise ValueError('Calibration cost scales must be positive finite values')
        for name in ('footprint_margin_m','footprint_cover_spacing_m'):
            value=getattr(self,name)
            if value is not None and (isinstance(value,bool) or not math.isfinite(value)
                    or value<0 or (name=='footprint_cover_spacing_m' and value==0)):
                raise ValueError('Invalid MPPI footprint geometry parameter: '+name)


class UpstreamMPPIKernelV3(UpstreamMPPIKernelV2):
    version='upstream_mppi_kernel_cost_v3'
    cost_version='upstream_mppi_speed_normalized_cost_v2'

    @property
    def metadata(self):
        metadata=super().metadata
        # Unset tuning parameters preserve historical checkpoint identities.
        for name in ('footprint_margin_m','footprint_cover_spacing_m','large_crater_query'):
            if metadata['settings'][name] is None:metadata['settings'].pop(name)
        return {**metadata,'smooth_normalization':'physical_speed_limits_v2',
                'hard_constraints':('unchanged_actuator_footprint_v2' if self.cfg.predictive_footprint_checks
                    else 'actuator_center_terrain_neighbor_workspace_only_v1'),
                **({'terrain_risk':'slope_traversability_roughness_height_step_v1'}
                   if not self.cfg.predictive_footprint_checks else {}),
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
        constants=self._cost_constants(requests)
        previous=constants['previous']
        change=torch.diff(torch.cat((previous[:,None,None].expand(-1,u.shape[1],1,-1),u),dim=2),dim=2)
        scale=constants['limits']
        parts['smooth']=self.settings.smooth_scale*((change/scale).square()*constants['smooth_weight']).sum((-1,-2))
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

    def planning_terrain(self,view):
        """Planner-owned geometry; observed cells and physical vehicle stay intact."""
        margin=self.settings.footprint_margin_m;spacing=self.settings.footprint_cover_spacing_m
        if margin is None and spacing is None:return view
        if self.cfg.obstacle_input is None:
            raise ValueError('MPPI footprint tuning requires local obstacle grid input')
        from .upstream_grid import cover_points
        key=(view.profile_fingerprint,view.settings.fingerprint,margin,spacing)
        if getattr(self,'_cover_key',None)!=key:
            self._cover_geometry=cover_points(self.spec,view.settings,spacing=spacing,margin=margin,centered=True)
            self._cover_key=key
        local=copy(view);local.points,local.cover_radius=self._cover_geometry
        if self.settings.large_crater_query is not None:
            if not self.cfg.obstacle_input.graded_craters:
                raise ValueError('Observed-grid SAT requires graded crater labels')
            local.large_crater_query=self.settings.large_crater_query
            local._large_crater_rectangles=view.large_crater_rectangles()
        return local

    def pack(self,requests):
        if self.settings.footprint_margin_m is None and self.settings.footprint_cover_spacing_m is None:
            return super().pack(requests)
        return super().pack([replace(r,terrain=self.planning_terrain(r.terrain)) for r in requests])

    @property
    def metadata(self):
        graded=self.cfg.obstacle_input is not None and self.cfg.obstacle_input.graded_craters
        tuned=self.settings.footprint_margin_m is not None or self.settings.footprint_cover_spacing_m is not None
        return {**super().metadata,
                **({'planning_footprint_cover':'cell_center_disks_v1'} if tuned else {}),
                **({'obstacle_input':self.cfg.obstacle_input.metadata,'terrain_query':'fixed_grid_cuda_lookup_v1',
                    'terrain_risk':'crater_free_driving_ground_v2' if graded else 'conservative_coarse_ground_v1',
                    'cost':'upstream_grid_crater_classes_v2' if graded else 'upstream_grid_obstacle_soft_cost_v1'} if self.cfg.obstacle_input is not None else {}),
                'goal_normalization':'remaining_distance_with_capture_floor_v1',
                'goal_distance_floor':self.spec.task.capture_radius}

    def cost(self,requests,states,controls,terrain,neighbors):
        parts,safe=super().cost(requests,states,controls,terrain,neighbors)
        constants=self._cost_constants(requests)
        goals,poses=constants['goals'],constants['poses']
        distance=(goals-poses[:,:2]).norm(dim=-1).clamp_min(self.spec.task.capture_radius)
        yaw,active=constants['yaw'],constants['yaw_active']
        yaw_cost=self.settings.goal_scale*active[:,None]*2*(1-(states.double()[:,:,-1,2]-yaw[:,None]).cos())
        parts['goal']=(parts['goal']-yaw_cost)*(self.spec.rho_max/distance[:,None]).square()+yaw_cost
        if self.cfg.obstacle_input is not None:
            cost=self.cfg.obstacle_input.data['cost'];band=cost['soft_band_m']
            if self.cfg.obstacle_input.graded_craters:
                d,medium=terrain.obstacle_distances(states[:,:,:-1]).double().unbind(-1)
            else:
                d=terrain.obstacle_distance(states[:,:,:-1]).double()
            near=(1-d/band).clamp(0,1).square();overlap=(-d/band).clamp(0,1)
            parts['obstacle_near']=self.spec.grid.dt*cost['near_weight_per_s']*near.sum(-1)
            parts['obstacle_overlap']=self.spec.grid.dt*cost['overlap_weight_per_s']*overlap.sum(-1)
            if self.cfg.obstacle_input.graded_craters:
                classes=self.cfg.obstacle_input.data['crater_classes']
                parts['crater_medium']=self.spec.grid.dt*classes['medium_cost_per_s']*(-medium/band).clamp(0,1).sum(-1)
                # The explicit cover/SAT backend checks all nodes, including the
                # terminal state. Execution owns the separate body/circle test.
                large=terrain.obstacle_distance(states,kind='crater_large')
                safe&=(large>=0).all(-1)
        return parts,safe


class UpstreamMPPIAdapterV4(UpstreamMPPIAdapterV3):
    adapter_id='mppi_upstream_v4'
    state_version=4

    def _review(self,item,trajectory):
        # The regenerated trajectory must use the same geometry as MPPI sampling.
        return super()._review(replace(item,terrain=self.batch_kernel.planning_terrain(item.terrain)),trajectory)

    def __init__(self,cfg,kernel=None):
        if cfg.planner_adapter!=self.adapter_id:raise ValueError('MPPI target v4 requires an explicit configuration')
        if kernel is not None and (type(kernel) is not UpstreamMPPIKernelV4 or kernel.spec.fingerprint!=cfg.spec.fingerprint):
            raise ValueError('MPPI v4 kernel/profile mismatch')
        UpstreamMPPIAdapter.__init__(self,cfg,kernel=kernel or UpstreamMPPIKernelV4(cfg,
            UpstreamMPPISettingsV3(samples=cfg.mppi_samples,iterations=cfg.mppi_iterations,device=cfg.planner_device)))
