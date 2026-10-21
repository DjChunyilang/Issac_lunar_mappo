"""Bounded experiment entry, panel accounting and selection regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_decision_experiment as experiment
from upstream_decision_evaluation import near_collector, panel, compare_results
from run_upstream_t11 import make_collector
from upstream_process_rollout import ProcessUpstreamRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions, PreparedUpstreamTrainer

BASE='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'


def request(**kwargs):
    result=dict(decision_credit_arm='all',representation='A1',seed=23,budget=200000,samples=512,
        env_owners=4,num_envs=16,owner_cpu_workers=2,trainer_backend='skrl',policy_device='cpu',
        gpu_service=False,import_checkpoint=None,import_learner=None,authorize_signal100k=False,
        resume_state=None,run_id='decision_contract',describe_decision_credit=False,
        release_training_pause=False,far_checkpoint=None,gradient_clipping='global',initial_continuous_std_scale=1.,
        terminal_actor_credit='all')
    return SimpleNamespace(**(result|kwargs))


def test_entry_description_does_not_release_pause_or_allocate(monkeypatch,capsys):
    monkeypatch.setattr(experiment,'run_arm',lambda *_:pytest.fail('must not allocate'))
    experiment.run_experiment(request(describe_decision_credit=True))
    result=json.loads(capsys.readouterr().out)
    assert result['evaluation_base_step_reservation_suite']==18000
    with pytest.raises(ValueError,match='paused'):
        experiment.run_experiment(request())
    for key,value in [('seed',24),('budget',500000),('num_envs',48),('representation','A0'),
                      ('import_checkpoint','old.pt')]:
        with pytest.raises(ValueError):experiment.validate_request(request(**{key:value}))


def test_evaluation_reservations_survive_failure(monkeypatch,tmp_path):
    (tmp_path/'metrics').mkdir()
    c=SimpleNamespace(base_steps=0,policy=object(),learning_time=LearningTimeConfig())
    args=SimpleNamespace(config='test',far_checkpoint='test')
    def fail(*args,**kwargs):raise RuntimeError('panel failed after consuming steps')
    monkeypatch.setattr(experiment,'evaluate_panels',fail)
    with pytest.raises(RuntimeError):experiment._evaluation(tmp_path,'initial',c,args)
    reservation=json.loads((tmp_path/'metrics/evaluation_initial_reservation.json').read_text())
    assert reservation['reserved_base_steps']==1600
    with pytest.raises(ValueError,match='Interrupted'):experiment._evaluation(tmp_path,'initial',c,args)


def test_screen_uses_geometry_hold_and_parking_not_reward():
    def arm(progress,hold=.5,parking=1,spread=1.):
        result={}
        for mode in ('sample','mean'):
            result[f'far_{mode}']={'dmax_reduction':[progress]*4,'dispersion_reduction':[spread]*4,
                                  'hold_probability':[hold]*4,'discounted_reward':999999.}
            result[f'near_{mode}']={'successes':parking}
        return result
    results={'A':arm(0.),'B':arm(2.,.4),'C':arm(3.,.3)}
    assert compare_results(results)['candidate']=='C'
    results['C']=arm(3.,.3,parking=0)
    assert compare_results(results)['candidate']=='B'
    results['B']=arm(0.)
    report=compare_results(results)
    assert report['candidate'] is None and report['next_step']=='legal_local_direction_guidance'


def test_near_panel_uses_existing_stop_contract_and_never_resets():
    torch.set_num_threads(1)
    timing=LearningTimeConfig(5.,1.)
    source=make_collector(BASE,'A1',23,samples=8,num_envs=1,learning_time=timing)
    c=near_collector(BASE,source.policy,timing,samples=8)
    np.testing.assert_allclose(c.env.core.previous_control[...,0],.1)
    np.testing.assert_allclose(c.env.core.actual_velocity[...,0],.1)
    # Force a policy HOLD for this contract check, then let the real task decide.
    with torch.no_grad():c.policy.hold_logit.weight.zero_();c.policy.hold_logit.bias.fill_(10.)
    episodes=c.env.core.episodes.copy()
    result=panel(c,deterministic=True,seconds=10)
    assert 0<result['actual_base_steps']<=200
    assert result['optimizer_updates']==0
    assert result['hold_fraction'].tolist()==[1.]*4
    np.testing.assert_array_equal(c.env.core.episodes,episodes)
    assert np.isfinite(result['critic_residual']).all()
    assert result['successes']==4


def test_four_owners_persistent_admission_and_checkpoint_counters(tmp_path):
    from upstream_checkpoint import save_checkpoint,read_checkpoint
    torch.set_num_threads(1)
    timing=LearningTimeConfig(5.,1.)
    c=ProcessUpstreamRollout(BASE,'A1',23,owners=4,envs_per_owner=4,samples=8,
        episode_seconds=.2,cpu_workers=1,learning_time=timing,mode='signal')
    try:
        t=PreparedUpstreamTrainer(c,TrainingOptions(rollout_macro_steps=1,epochs=1,
            gamma=c.gamma_step,lam=c.lambda_step))
        batch,_,_=t.collect()
        assert c.base_steps==c.decisions==16
        audit={'schema_version':'t11_decision_credit_readiness_v1','scope':'decision_credit',
            'maximum_base_steps':200000,'profile_fingerprint':c.env.spec.fingerprint,
            'learning_time':timing.metadata(.2),'formal_training_ready':False}
        # Validate authorization independently without starting research learning.
        with pytest.raises(ValueError,match='paused'):t.update(batch,readiness=audit)
        invalid=deepcopy(audit);invalid['learning_time']['discount_reference_period_s']=.2
        with pytest.raises(ValueError,match='paused'):t.update(batch,readiness=invalid,training_authorized=True)
        # This test only inspects admitted routing; numerical PPO is tested on
        # isolated diagnostic collectors in test_upstream_decision_learning.
        t._perform_update=lambda batch:{'admitted':True}
        assert t.update(batch,readiness=audit,training_authorized=True)=={'admitted':True}
        checkpoint=tmp_path/'update_000000.pt'
        save_checkpoint(t,{'test':True},0,checkpoint,scope='decision_credit_contract_only')
        header=json.loads(checkpoint.with_suffix('.json').read_text())
        assert header['actual_base_steps']==header['policy_decisions']==16
        c.collect(1)
        assert c.base_steps==32
        saved=read_checkpoint(checkpoint)
        t.load_state_dict(saved['trainer'])
        assert c.base_steps==c.decisions==16
    finally:c.close()


def test_far_panel_restores_physics_but_resets_reward_and_recurrent_accounting(tmp_path,monkeypatch):
    import upstream_decision_evaluation as evaluation
    torch.set_num_threads(1)
    source=make_collector(BASE,'A1',23,samples=8,num_envs=4,mode='signal')
    saved=source.env.state_dict()
    file=tmp_path/'owned_test_state.pt';torch.save(saved,file)
    monkeypatch.setattr(evaluation,'read_checkpoint',lambda _: {'trainer':{'collector':{
        'metadata':{'policy':{'representation':'A1'}},'owner_shards':[{'file':str(file)}]}}})
    timing=LearningTimeConfig(5.,1.)
    c=evaluation.far_collector(BASE,'test',source.policy,timing,samples=8)
    np.testing.assert_array_equal(c.env.core.poses,source.env.core.poses)
    np.testing.assert_array_equal(c.env.core.times,source.env.core.times)
    assert c.env.reward_tracker.settings.gamma==timing.discounts(.2)[0]
    assert not c.env.reward_tracker.steps.any() and not c.hidden.any()
    assert experiment.parameter_digest(c.policy)==experiment.parameter_digest(source.policy)


def test_driver_counts_actual_steps_and_stops_before_partial_commitment(tmp_path,monkeypatch):
    # Exercise the real session loop, files, evaluation phases and reservation
    # logic with cheap synthetic physics/PPO. No research training is launched.
    class Collector:
        def __init__(self,*args,learning_time,**kwargs):
            self.learning_time=learning_time;self.base_steps=0;self.decisions=0;self.transitions=0
            self.policy=torch.nn.Linear(1,1)
            self.env=SimpleNamespace(spec=SimpleNamespace(grid=SimpleNamespace(dt=.2),fingerprint='test'),
                                     cfg=SimpleNamespace(configuration_json='{}'))
        def close(self):pass
    class Trainer:
        def __init__(self,c,options):self.collector=c;self.options=options;self.updates=0
        def collect(self,count,on_chunk=None):
            duration=torch.full((count,16),25,dtype=torch.int64)
            duration[0,0]=1  # Early termination consumes less than the reservation.
            batch={k:torch.zeros((count,16)) for k in ('reward','value','next_value')}
            batch.update(obs=torch.zeros((count,16,4,1)),state=torch.zeros((count,16,1)),
                hidden=torch.zeros((count,16,4,128)),duration=duration,
                terminated=torch.zeros((count,16),dtype=torch.bool),truncated=torch.zeros((count,16),dtype=torch.bool),
                bootstrap_allowed=torch.ones((count,16),dtype=torch.bool),learning_valid=torch.ones((count,16),dtype=torch.bool),
                action_effective=torch.ones((count,16),dtype=torch.bool),
                terminal_reason=torch.zeros((count,16),dtype=torch.int64),
                samples=[SimpleNamespace(action=SimpleNamespace(hold=torch.zeros((16,4),dtype=torch.bool)),
                    log_prob=torch.zeros((16,4))) for _ in range(count)])
            self.collector.base_steps+=int(duration.sum())
            self.collector.decisions+=count*16;self.collector.transitions+=count*16
            on_chunk(batch,[],[],0)
            return batch,[],[]
        def update(self,batch,**kwargs):self.updates+=5;return {'updates':5}
        def release_rollout(self):pass
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(experiment,'source_snapshot',lambda:{})
    monkeypatch.setattr(experiment,'ProcessUpstreamRollout',Collector)
    monkeypatch.setattr(experiment,'SkrlUpstreamTrainer',Trainer)
    monkeypatch.setattr(experiment,'save_checkpoint',lambda t,i,u,p,**kw: p.write_text(str(t.collector.base_steps)))
    archived=[]
    monkeypatch.setattr(experiment,'save_state',lambda run,name,data:archived.append(data))
    phases=[]
    def evaluate(path,phase,c,args,final=False):
        phases.append((phase,c.base_steps))
        return {'results':{}}
    monkeypatch.setattr(experiment,'_evaluation',evaluate)
    cfg=tmp_path/'base.yaml';cfg.write_text('test')
    checkpoint=tmp_path/'historical.pt';checkpoint.write_text('test')
    args=request(decision_credit_arm='C',config=str(cfg),far_checkpoint=str(checkpoint),updates_this_invocation=None)
    summary,_=experiment.run_arm(args)
    assert summary['budget_complete'] and 199600<summary['actual_base_steps']<=200000
    assert 0<=summary['unused_base_steps']<400
    assert summary['sampling_updates']==21  # Actual terminal durations, not 20 nominal batches.
    assert phases[0]==('initial',0) and phases[-1][0]=='final'
    assert all(steps>=100000 for name,steps in phases if name=='midpoint')
    assert len(archived)==summary['sampling_updates']
    assert {'obs','state','hidden'} <= archived[0]['learning'].keys()
    torch.Generator().set_state(archived[0]['update_rng'])
