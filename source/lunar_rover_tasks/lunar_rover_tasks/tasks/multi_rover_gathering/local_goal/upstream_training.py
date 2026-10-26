"""Prepared recurrent MAPPO engine; execution requires verified prerequisites.

This module does not release the project's training pause. The caller must be
an explicitly authorized training entry after the independent readiness audit.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict

import torch

from .action_ablation import replay_action_sequence,clipped_actor_objective
from .learning import duration_gae
from .upstream_ablation_learning import UpstreamAblationRollout

TRAINER_VERSION='upstream_recurrent_ablation_trainer_v2'


def long_training_budget_valid(config, budget):
    if type(budget) is not int:return False
    if config['version']=='rendezvous_fixed_lunar_1km_v1':return 0<budget<=2000000
    return budget in (1000000,2000000)


@dataclass(frozen=True)
class TrainingOptions:
    rollout_macro_steps: int = 128
    epochs: int = 5
    learning_rate: float = 1e-4
    clip: float = .15
    entropy_weight: float = .0009
    value_weight: float = .5
    gradient_limit: float = 1.
    gamma: float = .99
    lam: float = .95
    gradient_clipping: str = 'global'
    terminal_actor_credit: str = 'all'
    value_normalization: str = 'none'

    def __post_init__(self):
        if (type(self.rollout_macro_steps) is not int or self.rollout_macro_steps<1 or self.rollout_macro_steps>512
                or type(self.epochs) is not int or not 1<=self.epochs<=10):raise ValueError('Invalid recurrent training batch')
        if self.gradient_clipping not in {'global','actor_critic'}:
            raise ValueError('Unknown policy/value gradient clipping semantics')
        if self.terminal_actor_credit not in {'all','effective'}:
            raise ValueError('Unknown terminal actor credit semantics')
        if self.value_normalization not in {'none','running'}:
            raise ValueError('Unknown value normalization semantics')
        if not all(torch.isfinite(torch.tensor(v)) and v>0 for k,v in asdict(self).items() if k not in {'rollout_macro_steps','epochs','gradient_clipping','terminal_actor_credit','value_normalization'}):
            raise ValueError('Invalid finite training coefficients')
        if not 0<self.gamma<=1 or not 0<self.lam<=1:raise ValueError('Invalid base-step discounts')

    @property
    def metadata(self):
        result=asdict(self)
        # Keep the original protocol's checkpoint identity byte-for-byte in
        # meaning. The new optimizer semantics are always explicit.
        if self.gradient_clipping=='global':result.pop('gradient_clipping')
        if self.terminal_actor_credit=='all':result.pop('terminal_actor_credit')
        if self.value_normalization=='none':result.pop('value_normalization')
        return result


class PreparedUpstreamTrainer:
    def __init__(self,collector,options=TrainingOptions()):
        if not isinstance(collector,UpstreamAblationRollout):raise ValueError('Expected actual four-head collector')
        if (options.gamma,options.lam)!=(collector.gamma_step,collector.lambda_step):
            raise ValueError('PPO and collector discounts disagree')
        if options.value_normalization=='running':
            if not hasattr(collector.policy,'value_normalizer'):
                collector.policy.enable_value_normalization()
        elif hasattr(collector.policy,'value_normalizer'):
            raise ValueError('Value normalization does not match training options')
        self.collector=collector;self.options=options;self.updates=0
        self.optimizer=None
        self.diagnostic_batches=0

    @property
    def metadata(self):
        return {'trainer_version':TRAINER_VERSION,'options':self.options.metadata,'collector':self.collector.metadata}

    def collect(self,macro_steps=None,*,on_chunk=None,chunk_decisions=16):
        remaining=self.options.rollout_macro_steps if macro_steps is None else macro_steps
        if type(remaining) is not int or not 1<=remaining<=self.options.rollout_macro_steps:
            raise ValueError('Requested collection length exceeds the prepared rollout')
        if type(chunk_decisions) is not int or not 1 <= chunk_decisions <= 16:
            raise ValueError('Collection logging chunk must be 1 to 16 decisions')
        parts=[];episodes=[];records=[]
        while remaining:
            length=min(chunk_decisions,remaining);batch,ends,trace=self.collector.collect(length)
            parts.append(batch);episodes.extend(ends);records.extend(trace);remaining-=length
            if on_chunk is not None:on_chunk(batch,ends,trace,remaining)
        batch={k:torch.cat([p[k] for p in parts]) for k in parts[0] if torch.is_tensor(parts[0][k])}
        batch['samples']=[s for p in parts for s in p['samples']];batch['metadata']=deepcopy(self.collector.metadata)
        return batch,episodes,records

    def update(self,batch,*,readiness,training_authorized=False):
        # Readiness is computed from owned immutable evidence by the launcher;
        # it is separate from explicit release of the current project pause.
        signal=(readiness.get('schema_version')=='upstream_t11_signal_readiness_v1'
                and readiness.get('scope')=='signal100k' and readiness.get('signal_training_ready') is True
                and readiness.get('maximum_budget')==100000
                and self.collector.transitions<=100000)
        decision_credit=(readiness.get('schema_version')=='t11_decision_credit_readiness_v1'
                and readiness.get('scope')=='decision_credit'
                and readiness.get('maximum_base_steps')==200000
                and self.collector.learning_time is not None
                and readiness.get('learning_time')==self.collector.learning_time.metadata(self.collector.env.spec.grid.dt)
                and self.collector.policy.contract.representation=='A1'
                and self.collector.env.e==16
                and 0<self.collector.base_steps<=200000)
        repair_validation=(readiness.get('schema_version')=='t11_learning_repair_validation_v1'
                and readiness.get('scope')=='learning_repair_validation'
                and readiness.get('maximum_base_steps')==20000
                and self.collector.learning_time is not None
                and readiness.get('learning_time')==self.collector.learning_time.metadata(self.collector.env.spec.grid.dt)
                and self.collector.metadata.get('initial_cohort')=='near_far_balanced_v1'
                and self.collector.policy.contract.representation=='A1'
                and self.collector.env.e==16 and 0<self.collector.base_steps<=20000)
        curriculum=(readiness.get('schema_version')=='t11_curriculum_readiness_v1'
                and readiness.get('scope')=='curriculum_1km'
                and readiness.get('curriculum_config') is not None
                and readiness.get('curriculum_config')==self.collector.metadata.get('curriculum_config')
                and long_training_budget_valid(readiness['curriculum_config'],readiness.get('maximum_base_steps'))
                and self.collector.learning_time is not None
                and readiness.get('learning_time')==self.collector.learning_time.metadata(self.collector.env.spec.grid.dt)
                and self.collector.env.spec.world_width==1000. and self.collector.env.spec.world_height==1000.
                and self.collector.policy.contract.representation=='A1' and self.collector.env.e==16
                and 0<self.collector.base_steps<=readiness['maximum_base_steps'])
        if training_authorized is not True or not (readiness.get('formal_training_ready') is True or signal or decision_credit or repair_validation or curriculum):
            raise ValueError('Formal training remains paused or prerequisites incomplete')
        if readiness.get('profile_fingerprint')!=self.collector.env.spec.fingerprint:
            raise ValueError('Training readiness belongs to another vehicle/profile')
        if self.diagnostic_batches:
            raise ValueError('Start formal training from its own initialization, not a diagnostic learner')
        if not self.collector.env.cfg.learning_runtime:
            raise ValueError('Formal update requires an explicit learning runtime; this configuration is paused')
        return self._perform_update(batch)

    def diagnostic_update(self,batch):
        """At most two bounded batches through the same complete PPO update path."""
        if not self.collector.env.cfg.training_paused:
            raise ValueError('Diagnostic updates require a paused diagnostic runtime')
        if self.diagnostic_batches>=2 or not 1<=len(batch['reward'])<=128:
            raise ValueError('Diagnostic preparation is limited to two batches of at most128 macro steps')
        result=self._perform_update(batch)
        self.diagnostic_batches+=1
        return {**result,'diagnostic_only':True,'formal_training_interactions':0}

    def _perform_update(self,batch):
        if batch.get('metadata')!=self.collector.metadata:raise ValueError('Training batch contract mismatch')
        if not batch['learning_valid'].any():return {'updates':0,'valid_transitions':0}
        if self.optimizer is None:
            self.optimizer=torch.optim.Adam(self.collector.policy.parameters(),lr=self.options.learning_rate)
        result=recurrent_mappo_update(self.collector.policy,self.optimizer,batch,self.options)
        self.updates+=result['updates']
        return result

    def release_rollout(self):
        """Release framework-owned rollout storage after the caller archives it."""

    def state_dict(self, *, collector_state=None):
        return {'metadata':deepcopy(self.metadata),'collector':self.collector.state_dict() if collector_state is None else collector_state,'updates':self.updates,
            'diagnostic_batches':self.diagnostic_batches,
            'optimizer':None if self.optimizer is None else deepcopy(self.optimizer.state_dict())}

    def load_state_dict(self,state):
        if state.get('metadata')!=self.metadata or type(state.get('updates')) is not int or state['updates']<0:
            raise ValueError('Trainer resume identity/counter mismatch')
        if type(state.get('diagnostic_batches')) is not int or not 0<=state['diagnostic_batches']<=2:
            raise ValueError('Invalid diagnostic preparation counter')
        optimizer=None
        if state.get('optimizer') is not None:
            # Validate structure against a copy before mutating the live learner.
            trial=deepcopy(self.collector.policy);check=torch.optim.Adam(trial.parameters(),lr=self.options.learning_rate)
            check.load_state_dict(state['optimizer'])
            for slots in check.state.values():
                if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in slots.values()):raise ValueError('Nonfinite optimizer snapshot')
            optimizer=torch.optim.Adam(self.collector.policy.parameters(),lr=self.options.learning_rate)
            optimizer.load_state_dict(state['optimizer'])
        self.collector.load_state_dict(state['collector']);self.optimizer=optimizer;self.updates=state['updates']
        self.diagnostic_batches=state['diagnostic_batches']


def clip_learning_gradients(policy,options):
    """Keep independent Actor/Critic gradients from sharing the clipping cap."""
    actor=[p for name,p in policy.named_parameters() if not name.startswith('critic.')]
    critic=list(policy.critic.parameters())
    actor_gradients=[p.grad.detach().norm() for p in actor if p.grad is not None]
    actor_norm=(torch.linalg.vector_norm(torch.stack(actor_gradients)) if actor_gradients
                else next(policy.parameters()).new_zeros(()))
    critic_norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in critic if p.grad is not None]))
    if options.gradient_clipping=='global':
        total=torch.nn.utils.clip_grad_norm_(policy.parameters(),options.gradient_limit)
        actor_scale=critic_scale=min(1.,options.gradient_limit/(float(total)+1e-6))
    else:
        torch.nn.utils.clip_grad_norm_(actor,options.gradient_limit)
        torch.nn.utils.clip_grad_norm_(critic,options.gradient_limit)
        total=torch.hypot(actor_norm,critic_norm)
        actor_scale=min(1.,options.gradient_limit/(float(actor_norm)+1e-6))
        critic_scale=min(1.,options.gradient_limit/(float(critic_norm)+1e-6))
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite formal recurrent gradient')
    return {'gradient_norm':float(total),'actor_gradient_norm':float(actor_norm),
            'critic_gradient_norm':float(critic_norm),'actor_gradient_scale':actor_scale,
            'critic_gradient_scale':critic_scale}


def recurrent_mappo_update(policy,optimizer,batch,options):
    """Shared probability/sequence contract used by legacy and skrl MAPPO.

    Framework ownership and storage are supplied by the caller. Keeping this
    objective shared makes an explicit checkpoint migration numerically testable.
    """
    valid=batch['learning_valid']
    if not valid.any():return {'updates':0,'valid_transitions':0}
    actor_valid=(valid & batch['action_effective'] if options.terminal_actor_credit=='effective' else valid)
    advantages,returns=duration_gae(batch['reward'],batch['value'],batch['next_value'],batch['duration'],
        batch['terminated'],batch['truncated'],gamma=options.gamma,lam=options.lam,
        bootstrap_allowed=batch['bootstrap_allowed'],learning_valid=valid)
    selected=advantages[valid]
    target=returns[valid].detach();old_value=batch['value'][valid].detach()
    target_variance=target.var(unbiased=False)
    diagnostics={'version':'t11_learning_diagnostics_v1',
        'actor_transitions':int(actor_valid.sum()),'actor_excluded_terminals':int((valid & ~actor_valid).sum()),
        'advantage_mean':float(selected.mean()),'advantage_std':float(selected.std(unbiased=False)),
        'return_mean':float(target.mean()),'return_std':float(target.std(unbiased=False)),
        'value_mean':float(old_value.mean()),
        'explained_variance_before_update':None if target_variance<=1e-12 else
            float(1-(target-old_value).var(unbiased=False)/target_variance)}
    # Team advantages grouped by sampled rover action are correlations, not
    # counterfactual per-rover credit assignments.
    holds=torch.stack([sample.action.hold for sample in batch['samples']])
    team_advantage=advantages[...,None].expand_as(holds)
    for name,mask in (('hold',holds),('move',~holds)):
        selected_actions=mask & valid[...,None]
        diagnostics[name+'_team_advantage_mean']=(float(team_advantage[selected_actions].mean())
            if selected_actions.any() else None)
    actor_selected=advantages[actor_valid]
    advantages=(torch.where(actor_valid,(advantages-actor_selected.mean())/actor_selected.std(unbiased=False).clamp_min(1e-6),0.)
                if actor_valid.any() else torch.zeros_like(advantages))
    normalizer=policy.value_normalizer if options.value_normalization=='running' else None
    if normalizer is not None:
        # GAE above uses rollout-time, denormalized V. Freeze these targets and
        # update moments once, not once per PPO epoch or once per rover.
        normalizer.update(target)
        critic_target=normalizer.normalize(target)
        diagnostics['value_normalization']={'count':int(normalizer.count),
            'mean':float(normalizer.mean),'std':float(normalizer.std)}
        diagnostics['critic_mse_units']='normalized_value_targets'
    else:
        critic_target=target
    metrics=[]
    for _ in range(options.epochs):
        optimizer.zero_grad()
        replay=replay_action_sequence(policy,batch['obs'],batch['hidden'][0],batch['samples'],valid,batch['terminated'])
        actor=clipped_actor_objective(replay['ratio'],advantages,actor_valid,options.clip)
        predicted=(policy.critic(batch['state'][valid]).squeeze(-1) if normalizer is not None
                   else policy.value(batch['state'][valid]))
        critic=torch.nn.functional.mse_loss(predicted,critic_target)
        entropy=replay['entropy'][actor_valid].mean() if actor_valid.any() else critic.new_zeros(())
        loss=actor+options.value_weight*critic-options.entropy_weight*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite formal recurrent loss')
        loss.backward();gradient_metrics=clip_learning_gradients(policy,options)
        optimizer.step()
        ratio=replay['ratio'][actor_valid].detach()
        old_log=torch.stack([sample.log_prob for sample in batch['samples']])[actor_valid]
        log_ratio=replay['log_prob'][actor_valid].detach()-old_log
        metrics.append({'actor_loss':float(actor.detach()),'critic_mse':float(critic.detach()),
            'entropy':float(entropy.detach()),**gradient_metrics,
            'ratio_mean':float(ratio.mean()) if ratio.numel() else None,
            'ratio_min':float(ratio.min()) if ratio.numel() else None,'ratio_max':float(ratio.max()) if ratio.numel() else None,
            'clip_fraction':float(((ratio-1).abs()>options.clip).float().mean()) if ratio.numel() else None,
            'approx_kl':float((ratio-1-log_ratio).mean()) if ratio.numel() else None})
        if normalizer is not None:
            with torch.no_grad():
                residual=policy.value(batch['state'][valid])-target
                metrics[-1]['critic_raw_mse_after_epoch']=float(residual.square().mean())
                metrics[-1]['critic_raw_bias_after_epoch']=float(residual.mean())
                metrics[-1]['explained_variance_after_epoch']=(None if target_variance<=1e-12 else
                    float(1-residual.var(unbiased=False)/target_variance))
    return {'epochs':metrics,'updates':len(metrics),'valid_transitions':int(valid.sum()),'diagnostics':diagnostics}
