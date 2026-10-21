"""Actual finite macro rollout with persistent GRU state and original actions.

Collection and restore do not start training or construct an optimizer. The
result uses the existing conditional-HOLD PPO/variable-duration GAE contract.
"""
from copy import deepcopy

import numpy as np
import torch

from .actions import ROLLOUT_VERSION
from .learning import RecurrentPolicy,bootstrap_values

ROLLOUT_COLLECTOR_VERSION='actual_recurrent_upstream_v1'


class UpstreamRollout:
    def __init__(self,env,policy):
        if (type(policy) is not RecurrentPolicy or not policy.hybrid or policy.rho_max!=env.spec.rho_max
                or policy.critic[0].in_features!=991):raise ValueError('Collector requires the matching A1 backbone and action scale')
        self.env,self.policy=env,policy
        self.device=next(policy.parameters()).device
        self.hidden=torch.zeros((env.e,4,128),device=self.device)
        self.transitions=0

    @property
    def metadata(self):
        return {'collector_version':ROLLOUT_COLLECTOR_VERSION,'environment':self.env.metadata,
            'action_version':self.policy.action_version,'rollout_version':ROLLOUT_VERSION,
            'actor_dim':411,'critic_dim':991,'rho_max':self.policy.rho_max,'beta_max':self.policy.beta_max}

    @torch.no_grad()
    def collect(self,length,*,deterministic=False):
        if type(length) is not int or length<1:raise ValueError('Invalid rollout length')
        fields={};episodes=[];records=[]
        for _ in range(length):
            # Reset only actual ended environments. Ordinary collection cuts
            # preserve both environment queues/warm starts and the GRU state.
            ids=np.flatnonzero(self.env._ended)
            if len(ids):self.env.reset(ids);self.hidden[ids]=0.
            actor,state=self.env.observe();actor=actor.to(self.device);state=state.to(self.device)
            initial=self.hidden.clone();sample=self.policy.sample(actor,initial,deterministic=deterministic)
            original=sample.action.numpy()
            requests=self.env.polar_requests(original.physical[...,0],original.physical[...,1],original.hold)
            transition=self.env.step_intents(requests)
            allowed=transition.bootstrap_allowed.to(self.device)
            valid=transition.learning_valid.to(self.device)
            next_value=bootstrap_values(self.policy,transition.terminal_critic.to(self.device),allowed)
            row={'obs':actor,'state':state,'hidden':initial,'value':self.policy.value(state),
                 'next_value':next_value,'reward':transition.reward.to(self.device),
                 'duration':transition.duration.to(self.device),'terminated':transition.terminated.to(self.device),
                 'truncated':transition.truncated.to(self.device),'bootstrap_allowed':allowed,'learning_valid':valid,
                 'terminal_reason':transition.terminal_reason.to(self.device),**sample.rollout_fields()}
            for name,value in row.items():
                if name=='rollout_version':continue
                fields.setdefault(name,[]).append(value.detach().clone())
            done=transition.terminated.to(self.device)|~valid
            self.hidden=torch.where(done[:,None,None],0.,sample.hidden).detach()
            self.transitions+=int(valid.sum())
            episodes.extend(transition.episodes);records.extend(transition.records)
        batch={k:torch.stack(v) for k,v in fields.items()};batch['rollout_version']=ROLLOUT_VERSION
        return batch,episodes,records

    def state_dict(self):
        return {'metadata':deepcopy(self.metadata),'environment':self.env.state_dict(),
                'hidden':self.hidden.detach().cpu().clone(),'transitions':self.transitions,
                'policy':{k:v.detach().cpu().clone() for k,v in self.policy.state_dict().items()},
                'torch_rng':torch.get_rng_state().clone(),
                'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None}

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('metadata')!=self.metadata:raise ValueError('Collector snapshot contract mismatch')
        h=state.get('hidden');weights=state.get('policy');expected=self.policy.state_dict()
        if (not torch.is_tensor(h) or h.shape!=self.hidden.shape or h.dtype!=self.hidden.dtype or not torch.isfinite(h).all()
                or type(state.get('transitions')) is not int or state['transitions']<0 or not isinstance(weights,dict)
                or set(weights)!=set(expected) or any(not torch.is_tensor(v) or v.shape!=expected[k].shape
                    or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in weights.items())):
            raise ValueError('Invalid collector recurrent/weight state')
        rng=state.get('torch_rng')
        generator=torch.Generator()
        try:generator.set_state(rng)
        except (TypeError,RuntimeError) as error:raise ValueError('Invalid collector RNG state') from error
        if self.device.type=='cuda':
            if state.get('cuda_rng') is None or len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG/device mismatch')
        elif state.get('cuda_rng') is not None:raise ValueError('Cross-device collector restore is unsupported')
        # Environment restore is itself atomic; all policy checks precede it.
        self.env.load_state_dict(state['environment']);self.policy.load_state_dict(weights)
        self.hidden=h.to(self.device).clone();self.transitions=state['transitions'];torch.set_rng_state(rng)
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
