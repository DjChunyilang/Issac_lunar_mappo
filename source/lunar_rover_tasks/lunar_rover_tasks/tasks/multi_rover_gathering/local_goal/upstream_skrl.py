"""skrl MAPPO with the versioned T11 shared-team recurrent objective.

Four rovers share one actor and one centralized critic/optimizer. A skrl agent
therefore represents the team; its observation retains the rover axis. Stock
MAPPO's shuffled feed-forward GAE/update cannot represent duration-weighted
returns, masked recurrent replay or the HOLD atom. Those parts are explicit
extensions, while models, inference, memory, optimizer and agent checkpoints
belong to skrl. No generic Gym wrapper or duplicate environment is allocated.
"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import skrl
import torch
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.multi_agents.torch.mappo import MAPPO

from .upstream_training import (PreparedUpstreamTrainer, TrainingOptions,
                                TRAINER_VERSION, recurrent_mappo_update)

SKRL_TRAINER_VERSION='upstream_skrl_recurrent_mappo_v1'


class TeamModel(Model):
    def __init__(self,policy):
        device=next(policy.parameters()).device
        super().__init__(observation_space=gym.spaces.Box(-np.inf,np.inf,(4,policy.actor_dim),dtype=np.float32),
            state_space=gym.spaces.Box(-np.inf,np.inf,(policy.critic_dim,),dtype=np.float32),
            action_space=gym.spaces.Box(-np.inf,np.inf,(4,policy.contract.continuous_dim+2),dtype=np.float32),
            device=device)
        self.policy=policy
        self.hidden=None
        self.deterministic=False

    def compute(self,inputs,*,role=''):
        return self.act(inputs,role=role)

    def act(self,inputs,*,role=''):
        if role=='value':return self.policy.value(inputs['states']),{}
        if role!='policy':raise ValueError('Unknown T11 model role')
        sample=self.policy.sample(inputs['observations'],self.hidden,deterministic=self.deterministic)
        # Packed actions are only the skrl transport. The original probability
        # payload and latent are retained, including categorical/native heads.
        action=torch.cat((sample.action.continuous,sample.action.hold[...,None],
                          sample.action.primitive_id[...,None]),dim=-1)
        return action,{'log_prob':sample.log_prob,'sample':sample}


class RecurrentTeamMAPPO(MAPPO):
    def __init__(self,collector,options):
        self.options=options
        model=TeamModel(collector.policy)
        memory=RandomMemory(memory_size=options.rollout_macro_steps,num_envs=collector.env.e,device=collector.device)
        super().__init__(possible_agents=['team'],models={'team':{'policy':model,'value':model}},
            memories={'team':memory},observation_spaces={'team':model.observation_space},
            state_spaces={'team':model.state_space},action_spaces={'team':model.action_space},
            device=collector.device,cfg={'rollouts':options.rollout_macro_steps,'learning_epochs':options.epochs,
                'mini_batches':1,'learning_rate':options.learning_rate,'discount_factor':options.gamma,
                'gae_lambda':options.lam,'ratio_clip':options.clip,'grad_norm_clip':options.gradient_limit,
                'entropy_loss_scale':options.entropy_weight,'value_loss_scale':options.value_weight,
                'learning_rate_scheduler_kwargs':{'team':{}},
                'observation_preprocessor_kwargs':{'team':{}},'state_preprocessor_kwargs':{'team':{}},
                'value_preprocessor_kwargs':{'team':{}},
                'experiment':{'write_interval':0,'checkpoint_interval':0}})
        self.epoch_updates=0
        self._batch=None

    @torch.no_grad()
    def sample(self,obs,state,hidden,*,deterministic=False):
        model=self.policies['team'];model.hidden=hidden;model.deterministic=deterministic
        _,outputs=self.act({'team':obs},{'team':state},timestep=0,timesteps=0)
        return outputs['team']['sample']

    def record_batch(self,batch):
        memory=self.memories['team'];length=len(batch['reward'])
        if not 1<=length<=memory.memory_size:raise ValueError('skrl recurrent memory capacity exceeded')
        tensors={k:v for k,v in batch.items() if torch.is_tensor(v)}
        # Allocate every tensor before filling: skrl create_tensor initializes all
        # floating tensors with NaN. Preserve [time,env,rover,...], never shuffle.
        for key,value in tensors.items():
            if key not in memory.tensors:
                memory.create_tensor(key,size=tuple(value.shape[2:]),dtype=value.dtype,keep_dimensions=True)
        memory.reset()
        for key,value in tensors.items():memory.get_tensor_by_name(key)[:length].copy_(value)
        memory.memory_index=length%memory.memory_size;memory.filled=length==memory.memory_size
        self._batch={k:memory.get_tensor_by_name(k)[:length] for k in tensors}
        self._batch.update(samples=deepcopy(batch['samples']),metadata=deepcopy(batch['metadata']))

    def update(self,*,timestep,timesteps,uid):
        if uid!='team' or self._batch is None:raise ValueError('No recorded T11 team rollout')
        result=recurrent_mappo_update(self.policies[uid].policy,self.optimizers[uid],self._batch,self.options)
        self.epoch_updates+=result['updates']
        for epoch in result.get('epochs',[]):
            for name,value in epoch.items():
                if value is not None:self.track_data('T11 / '+name,value)
        self._batch=None
        return result


class SkrlUpstreamTrainer(PreparedUpstreamTrainer):
    """Existing admission/session contract, backed by an actual skrl MAPPO agent."""
    def __init__(self,collector,options=TrainingOptions()):
        super().__init__(collector,options)
        self.agent=RecurrentTeamMAPPO(collector,options)
        self.optimizer=self.agent.optimizers['team']
        collector.agent_sampler=self.agent.sample

    @property
    def metadata(self):
        return {**super().metadata,'trainer_version':SKRL_TRAINER_VERSION,
                'framework':'skrl','framework_version':skrl.__version__,
                'algorithm':'MAPPO','objective':'duration_gae_shared_team_recurrent_v1'}

    def _perform_update(self,batch):
        if batch.get('metadata')!=self.collector.metadata:raise ValueError('Training batch contract mismatch')
        self.agent.record_batch(batch)
        result=self.agent.update(timestep=self.collector.transitions,timesteps=self.collector.transitions,uid='team')
        self.updates=self.agent.epoch_updates
        return result

    def release_rollout(self):
        self.agent._batch=None
        memory=self.agent.memories['team']
        self.agent.memories['team']=RandomMemory(memory_size=memory.memory_size,
            num_envs=memory.num_envs,device=memory.device)

    def load_state_dict(self,state):
        super().load_state_dict(state)
        if self.optimizer is None:
            self.optimizer=torch.optim.Adam(self.collector.policy.parameters(),lr=self.options.learning_rate)
        self.agent.optimizers['team']=self.optimizer
        self.agent.checkpoint_modules['team']['optimizer']=self.optimizer
        self.agent.epoch_updates=self.updates
        self.agent._batch=None

    def migrate_legacy_state(self,state):
        """Explicit, same-device update-boundary import; never relabel source files."""
        expected={k:v for k,v in self.metadata.items() if k not in
                  {'framework','framework_version','algorithm','objective'}}
        expected['trainer_version']=TRAINER_VERSION
        if state.get('metadata')!=expected:raise ValueError('Legacy migration contract mismatch')
        converted=deepcopy(state);converted['metadata']=deepcopy(self.metadata)
        self.load_state_dict(converted)

    def import_learner_state(self,state):
        """Explicit new-cohort warm start: retain learning, reset environments/GRU."""
        if (state['metadata']['trainer_version']!=SKRL_TRAINER_VERSION
                or state['metadata']['options']!=self.metadata['options']
                or state['diagnostic_batches']!=0 or self.collector.transitions!=0
                or torch.count_nonzero(self.collector.hidden)):
            raise ValueError('Learner import requires compatible learning options and a fresh cohort')
        if state['collector']['metadata'].get('learning_time')!=self.collector.metadata.get('learning_time'):
            raise ValueError('Learner import cannot change decision/discount semantics')
        trial=deepcopy(self.collector.policy)
        trial.load_diagnostic_state(state['collector']['policy'])
        optimizer=torch.optim.Adam(self.collector.policy.parameters(),lr=self.options.learning_rate)
        if state['optimizer'] is not None:optimizer.load_state_dict(deepcopy(state['optimizer']))
        for slots in optimizer.state.values():
            if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in slots.values()):
                raise ValueError('Nonfinite optimizer snapshot')
        self.collector.policy.load_state_dict(trial.state_dict())
        self.optimizer=optimizer
        self.agent.optimizers['team']=optimizer
        self.agent.checkpoint_modules['team']['optimizer']=optimizer
        self.updates=state['updates'];self.agent.epoch_updates=self.updates
        self.collector.transitions=state['collector']['transitions']
        torch.set_rng_state(state['collector']['torch_rng'])
