"""Versioned split precision: configured sampling, exact actuator feedback.

Physical control boundaries are discontinuous at the low-speed yaw threshold.
Casting actual feedback to sampling precision can turn a lawful actuator state
into an unlawful one. V2 therefore keeps feedback, actuator projection and
midpoint prediction in float64 while preserving configured raw sampling and
importance-update precision. The historical v1 kernel remains unchanged.
"""
from dataclasses import asdict

import numpy as np
import torch

from .upstream_execution import brake_sequence,limit_control_torch,midpoint_torch,terrain_gain_torch
from .upstream_mppi import UpstreamMPPIAdapter,UpstreamMPPIKernel

PRECISION_VERSION="sampling_configured_actuator_fp64_v2"


class UpstreamMPPIKernelV2(UpstreamMPPIKernel):
    version="upstream_mppi_kernel_actuator_v2"

    @property
    def metadata(self):
        return {**super().metadata,"precision_version":PRECISION_VERSION,
                "sampling_dtype":self.settings.dtype,"actuator_dtype":"float64",
                "prediction_dtype":"float64","zero_roundoff":1e-10,
                "physical_model":asdict(self.spec.vehicle)}

    def rollout(self,requests,raw,terrain):
        if raw.ndim!=4 or raw.shape[0]!=len(requests) or raw.shape[-2:]!=(40,2):
            raise ValueError("MPPI controls require [B,K,40,2]")
        if raw.device != terrain.device:
            raise ValueError("MPPI controls and local terrain require the same device")
        if self.device.type == 'cuda' and not terrain.empty and all(v._uniform_features for v in terrain._views):
            from .upstream_native import rollout_cuda
            result = rollout_cuda(requests, raw, terrain, self.spec)
            if result is not None:
                return result
        def physical(value):return torch.as_tensor(value,dtype=torch.float64,device=self.device)
        pose=physical(np.stack([r.ego_pose for r in requests]))
        previous=physical(np.stack([r.previous_control for r in requests]))
        state=pose[:,None].expand(-1,raw.shape[1],-1)
        last=previous[:,None].expand(-1,raw.shape[1],-1)
        caps=physical([self.spec.vehicle.forward_limit if r.intent.speed_cap is None else r.intent.speed_cap
                       for r in requests])[:,None]
        reverse=caps.clamp_max(self.spec.vehicle.reverse_limit)
        numeric=torch.isfinite(raw).all((-1,-2))
        clean=torch.where(torch.isfinite(raw),raw,0.)
        states,controls,velocities,known_nodes=[state],[],[],[]
        stopping=torch.zeros(raw.shape[:2],device=self.device,dtype=torch.bool)
        v=self.spec.vehicle;dt=self.spec.grid.dt
        for t in range(40):
            # Conversion occurs after raw perturbation generation; neither
            # clipped controls nor these double predictions replace epsilon.
            desired=clean[:,:,t].to(torch.float64).clone()
            desired[...,0]=torch.maximum(-reverse,torch.minimum(caps,desired[...,0]))
            bound=(torch.ceil(last[...,0].abs()/(v.linear_deceleration*dt)-1e-10)
                   +torch.ceil(last[...,1].abs()/(v.angular_deceleration*dt)-1e-10)+1)
            stopping|=bound>=40-t
            desired=torch.where(stopping[...,None],0.,desired)
            target,prior=desired.reshape(-1,2),last.reshape(-1,2)
            chunks=[limit_control_torch(target[start:start+self.settings.rollout_chunk],
                prior[start:start+self.settings.rollout_chunk],self.spec)
                for start in range(0,len(prior),self.settings.rollout_chunk)]
            last=torch.cat(chunks).reshape_as(last)
            last=torch.where(last.abs()<=1e-10,0.,last)
            features,known=terrain.lookup(state[...,:2])
            gain=terrain_gain_torch(state,features.to(torch.float64),self.spec)
            state,velocity=midpoint_torch(state,last,self.spec,gain=gain)
            states.append(state);controls.append(last);velocities.append(velocity);known_nodes.append(known)
        _,known=terrain.lookup(state[...,:2]);known_nodes.append(known)
        return torch.stack(states,2),torch.stack(controls,2),torch.stack(velocities,2),numeric,torch.stack(known_nodes,2)

    def _cap_curves(self,requests):
        cap=torch.as_tensor([self.spec.vehicle.forward_limit if r.intent.speed_cap is None else r.intent.speed_cap
                             for r in requests],dtype=torch.float64,device=self.device)
        if (all(r.intent.speed_cap is None for r in requests)
                or all(not np.any(r.previous_control) for r in requests)):
            return cap[:,None].expand(-1,40)
        previous=torch.as_tensor(np.stack([r.previous_control for r in requests]),dtype=torch.float64,device=self.device)
        limits=[]
        # Batch all environments/vehicles. Only the explicit control-time
        # dimension is iterated, matching the actual NumPy braking contract.
        for _ in range(40):
            previous=limit_control_torch(torch.zeros_like(previous),previous,self.spec)
            previous=torch.where(previous.abs()<=1e-10,0.,previous)
            limits.append(torch.maximum(cap,previous[:,0].abs()))
        return torch.stack(limits,1)

    def pack(self,requests):
        terrain,neighbors=super().pack(requests)
        neighbors.actuator_cap_curve_v2=self._cap_curves(requests)
        return terrain,neighbors

    def cost(self,requests,states,controls,terrain,neighbors):
        parts,safe=super().cost(requests,states,controls,terrain,neighbors)
        # A clipped desired command does not prove that actuator projection
        # satisfies a lowered local cap. Preserve the lawful braking transient
        # and exclude violating trajectories from the feasible-sample set.
        cap=getattr(neighbors,'actuator_cap_curve_v2',None)
        if cap is None:cap=self._cap_curves(requests)
        safe&=(controls[...,0].abs()<=cap[:,None]+1e-10).all(-1)
        return parts,safe


class UpstreamMPPIAdapterV2(UpstreamMPPIAdapter):
    adapter_id="mppi_upstream_v2"
    state_version=2

    def __init__(self,cfg,kernel=None):
        if cfg.planner_adapter!=self.adapter_id:
            raise ValueError("MPPI v2 requires an explicit versioned adapter configuration")
        if kernel is not None and (type(kernel) is not UpstreamMPPIKernelV2 or kernel.spec.fingerprint!=cfg.spec.fingerprint):
            raise ValueError("MPPI v2 cannot use a legacy or different-profile kernel")
        super().__init__(cfg,kernel=kernel or UpstreamMPPIKernelV2(cfg))

    def _review(self,item,trajectory):
        if item.intent.speed_cap is not None:
            brake=brake_sequence(item.previous_control,self.spec)
            cap=np.maximum(item.intent.speed_cap,np.pad(abs(brake[:,0]),(0,max(0,40-len(brake)))))[:40]
            if (abs(trajectory.controls[:,0])>cap+1e-10).any():return "intent_speed_cap"
        return super()._review(item,trajectory)
