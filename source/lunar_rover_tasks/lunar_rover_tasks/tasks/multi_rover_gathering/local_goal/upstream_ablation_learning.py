"""Real-environment recurrent T11 preparation; formal training remains gated.

All four heads retain original sampled payloads and probability measures.
Execution uses the versioned bridge, including native library identity. The
bounded diagnostic update is a preparation check, not a training launcher.
"""
from copy import deepcopy

import numpy as np
import torch

from .action_ablation import AblationPolicy,replay_action_sequence,clipped_actor_objective
from .learning import bootstrap_values,duration_gae
from .upstream_action_bridge import UpstreamActionBridge
from .upstream_decision import step_decision, action_can_affect_transition

COLLECTOR_VERSION='t11_actual_ablation_rollout_v1'
LEARNED_BOUND_VERSION='learned_horizon_braking_lease_v1'


def renew_legal_bounds(env):
    """Action-independent promises covering the full reachable prediction.

    A braking-distance bound alone does not bound displacement over the eight
    second review horizon. All learned vehicles can choose MOVE concurrently;
    also budget that horizon so their certificates do not make even a static
    stopping tail infeasible. Scripted leases keep their independent behavior.
    """
    for e in range(env.e):
        if env._ended[e]:continue
        now=float(env.core.times[e])
        for a in range(4):
            p=env.local_pose(e,a);snapshot=env.radio.snapshot(e,a,now,p)
            gaps=[np.linalg.norm(p[:2]-message.pose[:2])-2*env.spec.vehicle.effective_radius
                  for i,message in enumerate(snapshot.messages) if snapshot.state_valid[i]]
            requested=min(env.spec.vehicle.forward_limit,.8*max(0.,min(gaps,default=np.inf))
                /(env.spec.grid.horizon+env.spec.grid.request_validity+env.cfg.simulated_latency))
            if env.cfg.motion_lease_version=='braking_clearance_lease_v3':
                from .upstream_speed_lease import braking_clearance_limit
                requested=min(requested,braking_clearance_limit(
                    env.spec,min(gaps,default=np.inf),env.cfg.simulated_latency))
            cap=max(float(abs(env.core.previous_control[e,a,0])),float(requested))
            env.pledge_speed(e,a,cap,now+env.spec.grid.horizon+env.cfg.simulated_latency+2*env.spec.grid.high_level_period)


class UpstreamAblationRollout:
    def __init__(self,env,policy,learning_time=None):
        if type(policy) is not AblationPolicy or policy.critic_dim!=991:
            raise ValueError('Actual T11 collector requires the prepared shared backbone')
        expected=(.99,.95) if learning_time is None else learning_time.discounts(env.spec.grid.dt)
        if (env.reward_tracker.settings.gamma,env.reward_tracker.settings.lam)!=expected:
            raise ValueError('Reward and learning time discounts disagree')
        if learning_time is not None and policy.contract.representation not in ('A0','A1'):
            raise ValueError('Persistent decision learning currently supports A0/A1')
        self.learning_time=learning_time
        self.gamma_step,self.lambda_step=expected
        self.base_steps=0;self.decisions=0
        self.env,self.policy=env,policy;self.bridge=UpstreamActionBridge(env,policy.contract)
        self.device=next(policy.parameters()).device
        self.hidden=torch.zeros((env.e,4,128),device=self.device);self.transitions=0
        self.agent_sampler=None

    @property
    def metadata(self):
        result={'collector_version':COLLECTOR_VERSION,'bridge':self.bridge.metadata,
            'policy':self.policy.diagnostic_metadata,'observation_scope':'actual_dense_private_sensor',
            'device':str(self.device),'gamma':self.gamma_step,'lambda':self.lambda_step,
            'execution_bound_policy':(LEARNED_BOUND_VERSION if self.env.cfg.motion_lease_version=='braking_clearance_lease_v3'
                                      else 'legal_clearance_lease_upstream_v1'),'formal_training_ready':False}
        if self.env.cfg.learning_runtime:
            result.pop('formal_training_ready')  # Readiness belongs to the audited launcher.
            result['training_session_version']=self.env.cfg.training_session_version
        if self.learning_time is not None:
            result['learning_time']=self.learning_time.metadata(self.env.spec.grid.dt)
        return result

    @torch.no_grad()
    def collect(self,length,*,deterministic=False):
        if type(length) is not int or not 1<=length<=16:raise ValueError('Preparation collection length must be 1..16')
        rows={};samples=[];episodes=[];records=[]
        for _ in range(length):
            ids=np.flatnonzero(self.env._ended)
            if len(ids):self.env.reset(ids);self.hidden[ids]=0.
            renew_legal_bounds(self.env)
            obs,state=self.env.observe();obs=obs.to(self.device);state=state.to(self.device)
            initial=self.hidden.clone()
            sample=(self.policy.sample(obs,initial,deterministic=deterministic) if self.agent_sampler is None
                    else self.agent_sampler(obs,state,initial,deterministic=deterministic))
            transition=step_decision(self.bridge,sample.action,self.learning_time,renew_legal_bounds)
            self.env.discard_delivered_history()
            # One activation per vehicle per macro; store outcome without altering execution.
            for record in transition.records:
                if record.get('event')=='request_activation' and 'planner_outcome' not in record:
                    output=self.env.plans[record['env_id']][record['agent_id']]
                    if output is not None:
                        record['planner_outcome']={'valid':bool(output.valid),'fallback_reason':output.fallback_reason}
            valid=transition.learning_valid.to(self.device);ended=transition.terminated.to(self.device)
            allowed=transition.bootstrap_allowed.to(self.device)
            values={'obs':obs,'state':state,'hidden':initial,'value':self.policy.value(state),
                'next_value':bootstrap_values(self.policy,transition.terminal_critic.to(self.device),allowed),
                'reward':transition.reward.to(self.device),'duration':transition.duration.to(self.device),
                'action_effective':action_can_affect_transition(ended,transition.duration.to(self.device),
                    self.env.cfg.simulated_latency,self.env.spec.grid.dt),
                'terminated':ended,'truncated':transition.truncated.to(self.device),
                'learning_valid':valid,'bootstrap_allowed':allowed,
                'terminal_reason':transition.terminal_reason.to(self.device)}
            for name,value in values.items():rows.setdefault(name,[]).append(value.detach().clone())
            samples.append(deepcopy(sample))
            self.hidden=torch.where((ended|~valid)[:,None,None],0.,sample.hidden).detach()
            self.transitions+=int(valid.sum())
            self.base_steps+=int(transition.duration.sum());self.decisions+=int((transition.duration>0).sum())
            episodes.extend(transition.episodes);records.extend(transition.records)
        batch={k:torch.stack(v) for k,v in rows.items()}
        batch.update(samples=samples,metadata=deepcopy(self.metadata))
        return batch,episodes,records

    def state_dict(self, *, include_environment=True):
        return {'metadata':deepcopy(self.metadata),
            **({'base_steps':self.base_steps,'decisions':self.decisions} if self.learning_time is not None else {}),
            **({'environment':self.env.state_dict()} if include_environment else {}),
            'policy':self.policy.export_diagnostic_state(),'hidden':self.hidden.detach().cpu().clone(),
            'transitions':self.transitions,'torch_rng':torch.get_rng_state().clone(),
            'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None}

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('metadata')!=self.metadata:raise ValueError('Actual T11 restore contract mismatch')
        if self.learning_time is not None and any(type(state.get(k)) is not int or state[k]<0 for k in ('base_steps','decisions')):
            raise ValueError('Invalid physical sampling counters')
        hidden=state.get('hidden');count=state.get('transitions');rng=state.get('torch_rng')
        if (not torch.is_tensor(hidden) or hidden.shape!=self.hidden.shape or hidden.dtype!=self.hidden.dtype
                or not torch.isfinite(hidden).all() or type(count) is not int or count<0):raise ValueError('Invalid recurrent state')
        try:torch.Generator().set_state(rng)
        except (TypeError,RuntimeError) as error:raise ValueError('Invalid collector RNG') from error
        cuda=state.get('cuda_rng')
        if self.device.type=='cuda':
            if not isinstance(cuda,list) or len(cuda)!=torch.cuda.device_count():raise ValueError('CUDA RNG/device identity mismatch')
            for i,value in enumerate(cuda):
                try:torch.Generator(device=f'cuda:{i}').set_state(value)
                except (TypeError,RuntimeError) as error:raise ValueError('Invalid CUDA RNG') from error
        elif cuda is not None:raise ValueError('Cross-device hot recovery is unsupported')
        trial=deepcopy(self.policy);trial.load_diagnostic_state(state['policy'])
        self.env.load_state_dict(state['environment'])
        self.policy.load_diagnostic_state(state['policy']);self.hidden=hidden.to(self.device).clone()
        self.transitions=count
        self.base_steps=state.get('base_steps',0);self.decisions=state.get('decisions',0)
        torch.set_rng_state(rng)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def diagnostic_ablation_update(collector,optimizer,batch,*,clip=.15,entropy_weight=.0009):
    """One bounded update on actual transitions, without saving a checkpoint."""
    if type(collector) is not UpstreamAblationRollout or batch.get('metadata')!=collector.metadata:
        raise ValueError('Diagnostic update requires the matching actual rollout contract')
    if not 1<=len(batch['reward'])<=16:raise ValueError('Formal budgets cannot use the diagnostic update')
    policy=collector.policy;valid=batch['learning_valid']
    if not valid.any():return {'actor_loss':0.,'critic_mse':0.,'entropy':0.,'gradient_norm':0.,'optimizer_updates':0}
    owned={id(p) for group in optimizer.param_groups for p in group['params']}
    if owned!={id(p) for p in policy.parameters()}:raise ValueError('Optimizer does not own this policy')
    advantages,returns=duration_gae(batch['reward'],batch['value'],batch['next_value'],batch['duration'],
        batch['terminated'],batch['truncated'],gamma=collector.gamma_step,lam=collector.lambda_step,
        bootstrap_allowed=batch['bootstrap_allowed'],learning_valid=valid)
    selected=advantages[valid];advantages=torch.where(valid,
        (advantages-selected.mean())/selected.std(unbiased=False).clamp_min(1e-6),0.)
    optimizer.zero_grad()
    replay=replay_action_sequence(policy,batch['obs'],batch['hidden'][0],batch['samples'],valid,batch['terminated'])
    actor=clipped_actor_objective(replay['ratio'],advantages,valid,clip)
    values=policy.value(batch['state'][valid])
    critic=torch.nn.functional.mse_loss(values,returns[valid].detach())
    entropy=replay['entropy'][valid].mean();loss=actor+.5*critic-entropy_weight*entropy
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actual diagnostic loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite actual diagnostic gradient')
    optimizer.step()
    return {'actor_loss':float(actor.detach()),'critic_mse':float(critic.detach()),'entropy':float(entropy.detach()),
        'gradient_norm':float(norm),'optimizer_updates':1}
