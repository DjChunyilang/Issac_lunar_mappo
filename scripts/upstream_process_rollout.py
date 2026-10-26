"""Persistent environment owners with centralized, synchronous policy sampling.

An explicit new sampling protocol: owners retain complete independent environment
cohorts and GPU planner caches. Only actions, observations and required execution
records cross the macro-step boundary. Terrain/checkpoints travel only on save
or restore. No next-policy rollout is prefetched during PPO updates.
"""
from copy import deepcopy
from dataclasses import fields
import multiprocessing as mp
import atexit
from multiprocessing.connection import wait
from types import SimpleNamespace
import traceback
from pathlib import Path

import numpy as np
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies,ActionPayload
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import bootstrap_values
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout,renew_legal_bounds
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import step_decision, action_can_affect_transition

PROTOCOL='t11_persistent_owner_central_policy_v1'


def _owner(pipe,options,cpu_workers,service=None):
    try:
        from run_upstream_t11 import make_collector
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
        torch.set_num_threads(1);UpstreamEnvironment.cpu_workers=cpu_workers
        if service is not None:
            from upstream_gpu_service import install_client
            install_client(service)
        collector=make_collector(**options);env=collector.env
        pipe.send(('ok',(env.cfg,collector.metadata)))
        while True:
            command,value=pipe.recv()
            if command=='close':break
            if command=='prepare':
                reset=np.flatnonzero(env._ended)
                if len(reset):env.reset(reset)
                renew_legal_bounds(env)
                obs,state=env.observe();result=(obs.numpy(),state.numpy(),reset)
            elif command=='step':
                continuous,hold,ids,metadata=value
                payload=ActionPayload(torch.from_numpy(continuous),torch.from_numpy(hold),torch.from_numpy(ids),**metadata)
                transition=step_decision(collector.bridge,payload,collector.learning_time,renew_legal_bounds)
                env.discard_delivered_history()
                for record in transition.records:
                    if record.get('event')=='request_activation' and 'planner_outcome' not in record:
                        output=env.plans[record['env_id']][record['agent_id']]
                        if output is not None:record['planner_outcome']={'valid':bool(output.valid),'fallback_reason':output.fallback_reason}
                result={f.name:getattr(transition,f.name) for f in fields(transition)}
                result={k:v.numpy() if torch.is_tensor(v) else v for k,v in result.items()}
                result['poses']=env.core.poses.copy();result['task']=env.last_task
            elif command=='gpu_stats':result=[getattr(kernel,'service_stats',{}) for kernel in env.pool._kernels.values()]
            elif command=='perception_stats':
                result={'terrain_query_points':env.sensor.queried_points,
                        'physical_query_points':env.physical_query_points,
                        'memory':[[getattr(m,'stats',{}) for m in row] for row in env.memory],
                        'grid_upload_bytes':[getattr(k,'last_grid_upload_bytes',0) for k in env.pool._kernels.values()],
                        'spawn_attempts':None if env.curriculum is None else env.curriculum.spawn_attempts,
                        'poses':env.core.poses.copy()}
            elif command=='curriculum_level':env.curriculum.set_level(value);result=None
            elif command=='state':result=env.state_dict()
            elif command=='restore':env.load_state_dict(value);result=None
            elif command=='save_shard':
                from upstream_checkpoint import save_tensor_file
                result=save_tensor_file(value,env.state_dict())
            elif command=='restore_shard':
                state=torch.load(value,map_location='cpu',weights_only=False)
                env.load_state_dict(state)
                del state
                result=None
            else:raise ValueError('Unknown owner command: '+command)
            pipe.send(('ok',result))
            del result,value
    except BaseException:
        try:pipe.send(('error',traceback.format_exc()))
        finally:pipe.close()
    finally:pipe.close()


class ProcessUpstreamRollout(UpstreamAblationRollout):
    def __init__(self,base,representation,seed,*,owners=2,envs_per_owner=2,samples=512,
                 episode_seconds=360.,mode='diagnostic',cpu_workers=2,device='cpu',cache_dir=None,gpu_service=False,learning_time=None,
                 initial_cohort=None,curriculum_config=None):
        if owners<1 or envs_per_owner<1:raise ValueError('Positive owner/environment counts required')
        self.connections=[];self.processes=[];self.serial_dispatch=False
        self.learning_time=learning_time;self.base_steps=0;self.decisions=0
        self.initial_cohort=initial_cohort
        self.curriculum_config=deepcopy(curriculum_config)
        level = (curriculum_config['initial_level'] if curriculum_config is not None
                 and curriculum_config['version'] == 'rendezvous_curriculum_1km_v2' else 0)
        self.curriculum_progress={'level':level,'stage_start_base_steps':0,'outcomes':[],'history':[]}
        atexit.register(self.close)
        context=mp.get_context('spawn')
        self.owner_options=[];self.gpu_service=gpu_service;self.service_process=None
        services=[None]*owners
        if gpu_service:
            from _common import cfg_from_experiment
            if cfg_from_experiment(base).obstacle_input is not None:
                raise ValueError('Incremental grids use owner-local CUDA caches; shared point-service transport is unsupported')
            from upstream_gpu_service import serve
            pairs=[context.Pipe() for _ in range(owners)]
            services=[pair[0] for pair in pairs]
            self.service_process=context.Process(target=serve,args=([pair[1] for pair in pairs],),name='rover-gpu-service',daemon=True)
            self.service_process.start()
            for pair in pairs:pair[1].close()
        try:
            for owner in range(owners):
                options=dict(base=base,representation=representation,seed=seed+owner*envs_per_owner,
                    samples=samples,num_envs=envs_per_owner,episode_seconds=episode_seconds,mode=mode,
                    cache_dir=None if cache_dir is None else str(cache_dir/f'owner_{owner}'),learning_time=learning_time,
                    initial_cohort=initial_cohort,curriculum_config=curriculum_config)
                self.owner_options.append(options)
                parent,child=context.Pipe()
                process=context.Process(target=_owner,args=(child,options,cpu_workers,services[owner]),name=f'rover-owner-{owner}')
                process.start();child.close();self.connections.append(parent);self.processes.append(process)
                if services[owner] is not None:services[owner].close()
            initialized=self._receive_all()
        except BaseException:
            self.close();raise
        cfg=initialized[0][0]
        self.owner_metadata=[item[1] for item in initialized]
        self.gamma_step,self.lambda_step=((.99,.95) if learning_time is None else learning_time.discounts(cfg.spec.grid.dt))
        self.env=SimpleNamespace(e=owners*envs_per_owner,spec=cfg.spec,cfg=cfg,
            core=SimpleNamespace(poses=None),last_task=None)
        self.device=torch.device(device);self.envs_per_owner=envs_per_owner
        torch.manual_seed(seed)
        self.policy=build_ablation_policies(cfg.critic_dim,cfg.spec,terrain_channels=cfg.terrain_channels)[representation].to(self.device)
        self.hidden=torch.zeros((self.env.e,4,128),device=self.device)
        self.agent_sampler=None;self.transitions=0

    @property
    def metadata(self):
        return {**({'curriculum_config':self.curriculum_config} if self.curriculum_config is not None else {}),
            **({'initial_cohort':self.initial_cohort} if self.initial_cohort is not None else {}),
            **({'learning_time':self.learning_time.metadata(self.env.spec.grid.dt)} if self.learning_time is not None else {}),
            'collector_version':PROTOCOL,'owners':len(self.connections),'envs_per_owner':self.envs_per_owner,
            'owner_metadata':self.owner_metadata,'policy':self.policy.diagnostic_metadata,'device':str(self.device),
            'gamma':self.gamma_step,'lambda':self.lambda_step,'gpu_service':self.gpu_service,'sampling':'central_shared_actor_original_full_batch_order',
            'synchronization':'complete_rollout_then_update'}

    def _receive_all(self):
        pending={pipe:index for index,pipe in enumerate(self.connections)};result=[None]*len(pending)
        while pending:
            for pipe in wait(pending):
                status,value=pipe.recv()
                if status!='ok':raise RuntimeError('Environment owner failed:\n'+value)
                result[pending.pop(pipe)]=value
        return result

    def _request(self,command,values):
        if self.serial_dispatch:
            result=[]
            for pipe,value in zip(self.connections,values,strict=True):
                pipe.send((command,value));status,reply=pipe.recv()
                if status!='ok':raise RuntimeError('Environment owner failed:\n'+reply)
                result.append(reply)
            return result
        for pipe,value in zip(self.connections,values,strict=True):pipe.send((command,value))
        return self._receive_all()

    @torch.no_grad()
    def collect(self,length,*,deterministic=False):
        if type(length) is not int or not 1<=length<=16:raise ValueError('Owner collection length must be 1..16')
        rows={};samples=[];episodes=[];records=[]
        for _ in range(length):
            prepared=self._request('prepare',[None]*len(self.connections))
            for owner,(_,_,reset) in enumerate(prepared):
                if len(reset):self.hidden[reset+owner*self.envs_per_owner]=0.
            obs=torch.from_numpy(np.concatenate([p[0] for p in prepared])).to(self.device)
            state=torch.from_numpy(np.concatenate([p[1] for p in prepared])).to(self.device)
            initial=self.hidden.clone()
            sample=(self.policy.sample(obs,initial,deterministic=deterministic) if self.agent_sampler is None else
                    self.agent_sampler(obs,state,initial,deterministic=deterministic))
            action=sample.action
            arrays=[v.detach().cpu().numpy() for v in (action.continuous,action.hold,action.primitive_id)]
            metadata={k:getattr(action,k) for k in ('action_version','primitive_version','primitive_fingerprint')}
            commands=[tuple(v[i*self.envs_per_owner:(i+1)*self.envs_per_owner] for v in arrays)+(metadata,)
                      for i in range(len(self.connections))]
            transitions=self._request('step',commands)
            def tensor(name):return torch.from_numpy(np.concatenate([t[name] for t in transitions])).to(self.device)
            valid=tensor('learning_valid');ended=tensor('terminated');allowed=tensor('bootstrap_allowed')
            values={'obs':obs,'state':state,'hidden':initial,'value':self.policy.value(state),
                'action_effective':action_can_affect_transition(ended,tensor('duration'),
                    self.env.cfg.simulated_latency,self.env.spec.grid.dt),
                'next_value':bootstrap_values(self.policy,tensor('terminal_critic'),allowed),
                **{k:tensor(k) for k in ('reward','duration','terminated','truncated','learning_valid','bootstrap_allowed','terminal_reason')}}
            for key,value in values.items():rows.setdefault(key,[]).append(value.detach().clone())
            samples.append(deepcopy(sample))
            self.hidden=torch.where((ended|~valid)[:,None,None],0.,sample.hidden).detach()
            self.transitions+=int(valid.sum())
            self.base_steps+=int(tensor('duration').sum());self.decisions+=int((tensor('duration')>0).sum())
            for owner,transition in enumerate(transitions):
                offset=owner*self.envs_per_owner
                for episode in transition['episodes']:
                    episode=deepcopy(episode);episode['env_id']+=offset;episodes.append(episode)
                # Nested execution arrays retain cohort dimensions. Explicit
                # owner/global offset prevents ambiguous local environment IDs.
                for record in transition['records']:
                    record['owner_id']=owner;record['env_offset']=offset;records.append(record)
            self.env.core.poses=np.concatenate([t['poses'] for t in transitions])
            self.env.last_task=[t['task'] for t in transitions]
        batch={k:torch.stack(v) for k,v in rows.items()}
        batch.update(samples=samples,metadata=deepcopy(self.metadata))
        return batch,episodes,records

    def set_curriculum_level(self,level):
        self._request('curriculum_level',[level]*len(self.connections))

    def state_dict(self, *, include_environment=True):
        return {**({'curriculum_progress':deepcopy(self.curriculum_progress)} if self.curriculum_config is not None else {}),
            'metadata':deepcopy(self.metadata),
            **({'base_steps':self.base_steps,'decisions':self.decisions} if self.learning_time is not None else {}),
            **({'owners':self._request('state',[None]*len(self.connections))} if include_environment else {}),
            'policy':self.policy.export_diagnostic_state(),'hidden':self.hidden.detach().cpu().clone(),
            'transitions':self.transitions,'torch_rng':torch.get_rng_state().clone(),
            'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None}

    def save_owner_shards(self,directory):
        shards=[]
        for owner,pipe in enumerate(self.connections):
            pipe.send(('save_shard',str((Path(directory)/f'owner_{owner:03d}.pt').resolve())))
            status,result=pipe.recv()
            if status!='ok':raise RuntimeError('Environment owner failed:\n'+result)
            shards.append(dict(result,owner=owner))
        return shards

    def load_state_dict(self,state):
        if state.get('metadata')!=self.metadata:raise ValueError('Owner sampling protocol mismatch; legacy hot import forbidden')
        if self.learning_time is not None and any(type(state.get(k)) is not int or state[k]<0 for k in ('base_steps','decisions')):
            raise ValueError('Invalid physical sampling counters')
        hidden=state['hidden']
        if hidden.shape!=self.hidden.shape or not torch.isfinite(hidden).all():raise ValueError('Invalid owner recurrent state')
        torch.Generator().set_state(state['torch_rng'])
        trial=deepcopy(self.policy);trial.load_diagnostic_state(state['policy'])
        if 'owner_shards' in state:
            for pipe,shard in zip(self.connections,state['owner_shards'],strict=True):
                pipe.send(('restore_shard',shard['file']));status,result=pipe.recv()
                if status!='ok':raise RuntimeError('Environment owner failed:\n'+result)
        else:
            for pipe,owner in zip(self.connections,state['owners'],strict=True):
                pipe.send(('restore',owner));status,result=pipe.recv()
                if status!='ok':raise RuntimeError('Environment owner failed:\n'+result)
        self.policy.load_diagnostic_state(state['policy']);self.hidden=hidden.to(self.device).clone()
        if self.curriculum_config is not None:self.curriculum_progress=deepcopy(state['curriculum_progress'])
        self.transitions=state['transitions']
        self.base_steps=state.get('base_steps',0);self.decisions=state.get('decisions',0)
        torch.set_rng_state(state['torch_rng'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])

    def close(self):
        atexit.unregister(self.close)
        for pipe in self.connections:
            try:pipe.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join()
        for pipe in self.connections:pipe.close()
        if self.service_process is not None:
            if self.service_process.is_alive():self.service_process.terminate()
            self.service_process.join();self.service_process=None
        self.connections=[];self.processes=[]
