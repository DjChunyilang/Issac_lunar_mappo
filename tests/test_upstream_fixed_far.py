"""Fixed far distribution has no hidden curriculum, and a complete eval budget."""
import sys
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from upstream_curriculum_experiment import load_curriculum,timing_for,collection_decisions,evaluation_panels,evaluation_reservation,evaluation_budget_upper_bound,convergence_report
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import SpawnCurriculum,update_progress
from training_tensorboard import TrainingTensorBoard,FULL_FAR_PINNED_SCALARS


def test_replacement_run_uses_remaining_fixed_lunar_budget():
    import pytest
    from upstream_curriculum_experiment import validate_long_training_budget
    fixed = load_curriculum(CONFIG)
    validate_long_training_budget(fixed, 1490000)
    for budget in (0, -1, 2000001, True):
        with pytest.raises(ValueError):validate_long_training_budget(fixed, budget)
    curriculum = load_curriculum('configs/training/upstream_1km_curriculum_v2.yaml')
    with pytest.raises(ValueError):validate_long_training_budget(curriculum, 1490000)


def test_obstacle_prelaunch_audits_the_actual_reward_identity():
    from audit_upstream_curriculum import reward_audit
    cfg = cfg_from_experiment('configs/experiment/upstream_1km_obstacle_input_v1.yaml')
    config = load_curriculum('configs/training/upstream_1km_far_obstacle_input_v1.yaml')
    audited = reward_audit(cfg.spec, config, obstacle_input=cfg.obstacle_input)
    assert audited['settings']['obstacle_input'] == cfg.obstacle_input.metadata
    assert audited['settings']['obstacle_reward_version'] == 'executed_obstacle_penalty_v1'
    assert 'obstacle_input' not in reward_audit(cfg.spec, config)['settings']


def test_cli_dispatches_remaining_budget_without_relaxing_legacy(monkeypatch):
    import pytest
    import run_upstream_t11
    import upstream_curriculum_experiment
    received = []
    monkeypatch.setattr(upstream_curriculum_experiment, 'run_experiment', received.append)
    args = ['run_upstream_t11.py', '--config', BASE, '--run-id', 'remaining_budget_contract',
            '--representation', 'A1', '--mode', 'train', '--budget', '1497631']
    monkeypatch.setattr(sys, 'argv', args + ['--training-config', CONFIG])
    run_upstream_t11.main()
    assert len(received) == 1 and received[0].budget == 1497631
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit):run_upstream_t11.main()

CONFIG='configs/training/upstream_1km_far_fixed_v1.yaml'
BASE='configs/experiment/upstream_1km_curriculum_v1.yaml'

def test_fixed_spawn_resume_no_promotion_or_stage_cap():
    cfg=load_curriculum(CONFIG);spec=cfg_from_experiment(BASE).spec
    sampler=SpawnCurriculum(spec,cfg,23,64)
    poses=sampler.sample(range(64))
    pairs=np.linalg.norm(poses[:,:,None,:2]-poses[:,None,:,:2],axis=-1)
    assert np.all(pairs.max(axis=(1,2))>350) and pairs.max()<=600
    assert np.array_equal(sampler.deadlines,np.full(64,1500.))
    before=sampler.state_dict();expected=sampler.sample([2,5]);sampler.load_state_dict(before)
    np.testing.assert_array_equal(expected,sampler.sample([2,5]))
    p={'level':0,'outcomes':[],'history':[],'stage_start_base_steps':0}
    after,promoted=update_progress(cfg,p,[{'curriculum_level':0,'terminal_reason':'success'}]*100,300000)
    assert after==p and not promoted
    assert collection_decisions(cfg,p,2000000,300000,16,25)==32
    assert collection_decisions(cfg,p,2000000,1999600,16,25)==1
    assert collection_decisions(cfg,p,2000000,1999601,16,25)==0
    # Rejection consumes sampler RNG only, not physical steps; counters restore.
    sampler._birth_sensor=object();sampler._legal_birth=lambda poses: sampler.spawn_attempts % 2 == 0
    before=sampler.state_dict();expected=sampler.sample([2,5]);counts=(sampler.spawn_attempts,sampler.spawn_rejections)
    sampler.load_state_dict(before)
    np.testing.assert_array_equal(expected,sampler.sample([2,5]))
    assert counts==(sampler.spawn_attempts,sampler.spawn_rejections)==(4,2)


def test_1500s_discount_and_complete_eval_budget():
    c=load_curriculum(CONFIG);timing=timing_for(c);g,l=timing.discounts(.2)
    assert np.isclose(g**7500,.997**1200)
    assert np.isclose((g*l)**25,(c['gamma']*.99)**5)
    assert timing.steps(.2)==25
    assert evaluation_reservation(evaluation_panels(c,0,'initial'))==60000
    assert evaluation_reservation(evaluation_panels(c,0,'final',final=True))==120000
    assert evaluation_budget_upper_bound(c,2000000)==360000<=c['evaluation_max_base_steps']


def test_full_success_pinned_and_unseen_required(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    writer=TrainingTensorBoard(tmp_path,pinned='full_far')
    row={'training_base_steps':500000,'actual_base_steps':60000,'reserved_base_steps':60000,
         'results':{f'far_{mode}':{'full_episode_evaluation':True,'initial':{'dmax':[500]*4},'successes':n,
                                'dmax_reduction':[100]*4} for mode,n in [('sample',3),('mean',2)]}}
    writer.evaluation(row,'evaluation_step_0500000');writer.close()
    acc=EventAccumulator(str(tmp_path/'tensorboard')).Reload()
    assert acc.Scalars('Main/SR_S')[-1].value==.75 and acc.Scalars('Main/SR_M')[-1].value==.5
    assert len(FULL_FAR_PINNED_SCALARS)==10
    (tmp_path/'metrics').mkdir()
    for i in range(3):
        row['training_base_steps']=i*500000
        for r in row['results'].values():r['successes']=4
        (tmp_path/'metrics'/f'evaluation_{i}.json').write_text(json.dumps(row))
    c=load_curriculum(CONFIG)
    assert not convergence_report(tmp_path,c)['stable_convergence_screen_passed']
    for mode in ('sample','mean'):row['results'][f'unseen_{mode}']=deepcopy(row['results'][f'far_{mode}'])
    (tmp_path/'metrics'/'evaluation_2.json').write_text(json.dumps(row))
    assert convergence_report(tmp_path,c)['stable_convergence_screen_passed']


def test_observed_slope_failure_is_terminal_credit_not_missing_data():
    import pytest
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_task import UpstreamTask
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReport
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward, UpstreamRewardSettings, actual_facts
    spec=cfg_from_experiment(BASE).spec
    task=UpstreamTask(spec,3,1500.,terrain_failure_terminal=True)
    poses=np.zeros((3,4,3));poses[...,:2]=spec.task.slots
    speed=np.zeros((3,4,2))
    reports=[[FootprintReport(True,minimum_body_clearance=.3,checked_poses=1) for _ in range(4)] for _ in range(3)]
    # Measured offending report from the 36-step real lunar reproduction.
    reports[0][2]=FootprintReport(False,'driving_slope',.1766033268436753,.0030085801235016185,.07444121218021138,.2727877922886563,1)
    reports[1][2]=FootprintReport(False,'unknown_footprint',checked_poses=1)
    before=actual_facts(spec,poses,speed,reports,np.zeros(3))
    row=task.advance(poses,speed,reports,np.full(3,.2),environment_valid=np.array([True,False,True]),terrain_failed=np.array([True,False,False]))
    assert row['terminal_reason'].tolist()==['terrain_failure','environment_invalid','none']
    assert row['learning_valid'].tolist()==[True,False,True]
    assert row['bootstrap_allowed'].tolist()==[False,False,True]
    g,l=timing_for(load_curriculum(CONFIG)).discounts(.2)
    reward=UpstreamReward(spec,3,UpstreamRewardSettings(gamma=g,lam=l,gather_scaling='log1p'))
    features=np.zeros((3,4,5));features[...,4]=1
    value,terms=reward.advance(before,before,row,speed,features,np.ones((3,4)),np.zeros((3,4)))
    assert np.isfinite(value).all() and value[1]==0
    assert terms['terminal'].tolist()==[-55.,0.,0.]
    assert terms['success_hold'][0]==12.  # True terminal successor potential is zero.
    saved=task.state_dict();restored=UpstreamTask(spec,3,1500.,terrain_failure_terminal=True)
    restored.load_state_dict(saved)
    assert restored.state_dict()['terminal_reason'][0]=='terrain_failure'
    with pytest.raises(ValueError,match='mismatch'):
        UpstreamTask(spec,3,1500.).load_state_dict(saved)
    restored.reset(np.array([0]));assert restored._reason.tolist()==['none','environment_invalid','none']


def test_live_collection_chunks_preserve_rng_and_async_terminal_samples():
    from run_upstream_t11 import make_collector
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer,TrainingOptions
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig
    torch.set_num_threads(1)
    results=[]
    for chunk in (16,1):
        c=make_collector('configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml','A1',23,
                         samples=8,num_envs=2,episode_seconds=.4,learning_time=LearningTimeConfig(5.,1.))
        t=PreparedUpstreamTrainer(c,TrainingOptions(rollout_macro_steps=3,gamma=c.gamma_step,lam=c.lambda_step))
        calls=[];torch.manual_seed(2027)
        batch,episodes,_=t.collect(3,chunk_decisions=chunk,on_chunk=lambda b,e,r,left:calls.append((c.base_steps,left)))
        results.append((batch,episodes,c.base_steps,calls))
    a,b=results
    assert a[1]==b[1] and a[2]==b[2]==12
    assert a[3]==[(12,0)] and b[3]==[(4,2),(8,1),(12,0)]
    for key in a[0]:
        if torch.is_tensor(a[0][key]):torch.testing.assert_close(a[0][key],b[0][key],atol=0,rtol=0)
    for sa,sb in zip(a[0]['samples'],b[0]['samples']):
        torch.testing.assert_close(sa.log_prob,sb.log_prob,atol=0,rtol=0)
        torch.testing.assert_close(sa.action.hold,sb.action.hold)


def test_bound_spawn_sampler_can_be_copied_for_environment_restore():
    from copy import deepcopy
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import SpawnCurriculum
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import create_reviewer
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor
    from upstream_curriculum_experiment import load_curriculum
    config=load_curriculum('configs/training/upstream_1km_far_fixed_v1.yaml')
    spec=cfg_from_experiment('configs/experiment/upstream_1km_curriculum_v1.yaml').spec
    sampler=SpawnCurriculum(spec,config,23,1)
    sampler.bind_terrain(DenseSensor(spec,None),create_reviewer(spec,'flat_frame_pose_cache_v3'))
    sampler.sample([0]);restored=deepcopy(sampler)
    np.testing.assert_array_equal(sampler.sample([0]),restored.sample([0]))
    assert sampler.spawn_attempts==restored.spawn_attempts


def test_concurrent_evaluation_freezes_policy_and_resume_uses_original_snapshot(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import upstream_curriculum_experiment as runner
    for name in ('state','metrics','tensorboard'):(tmp_path/name).mkdir()
    class Process:
        def __init__(self,**kwargs):self.kwargs=kwargs;self.alive=False;self.exitcode=None
        def start(self):self.alive=True
        def is_alive(self):return self.alive
        def join(self,timeout=None):pass
    monkeypatch.setattr(runner.mp,'get_context',lambda _:SimpleNamespace(Process=Process))
    config=load_curriculum(CONFIG)
    collector=SimpleNamespace(policy=torch.nn.Linear(2,1),base_steps=0,learning_time=timing_for(config),curriculum_progress={'level':0})
    state=torch.get_rng_state().clone();expected=deepcopy(collector.policy.state_dict())
    worker=runner.ConcurrentFixedEvaluation(tmp_path,SimpleNamespace(config=BASE),config)
    worker.start('initial',collector)
    assert worker.process.is_alive() and torch.equal(state,torch.get_rng_state())
    with torch.no_grad():collector.policy.weight.add_(100.)
    collector.base_steps=12800
    resumed=runner.ConcurrentFixedEvaluation(tmp_path,SimpleNamespace(config=BASE),config)
    resumed.resume_pending()
    saved=torch.load(tmp_path/'state/evaluation_initial.pt',weights_only=False)
    assert saved['collector'].base_steps==0
    for name,value in expected.items():torch.testing.assert_close(saved['collector'].policy.state_dict()[name],value,atol=0,rtol=0)
    def evaluated(path,phase,frozen,args,cfg,**kwargs):
        assert phase=='initial' and frozen.base_steps==0
        kwargs['tensorboard'].scalars('evaluation_progress/panel_actual_base_steps',25,0)
        (path/'metrics/evaluation_initial.json').write_text('{}')
    monkeypatch.setattr(runner,'evaluate',evaluated)
    runner._evaluate_frozen(tmp_path,tmp_path/'state/evaluation_initial.pt',None,config,False,False)
    assert list((tmp_path/'tensorboard').glob('*tfevents*'))
    resumed.process.alive=False;resumed.process.exitcode=0
    assert resumed.poll()


def test_evaluator_failure_is_not_silently_ignored(tmp_path):
    import pytest
    from types import SimpleNamespace
    from upstream_curriculum_experiment import ConcurrentFixedEvaluation
    worker=ConcurrentFixedEvaluation(tmp_path,None,None);worker.phase='initial'
    worker.process=SimpleNamespace(is_alive=lambda:False,join=lambda:None,exitcode=7)
    with pytest.raises(RuntimeError,match='code 7'):worker.poll()


def test_interrupted_evaluation_keeps_old_reservation_and_charges_retry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import run_upstream_t11,upstream_decision_evaluation
    from upstream_curriculum_experiment import evaluate
    (tmp_path/'metrics').mkdir()
    (tmp_path/'metrics/evaluation_initial_reservation.json').write_text(json.dumps({'reserved_base_steps':60000}))
    config=load_curriculum(CONFIG)
    collector=SimpleNamespace(policy=torch.nn.Linear(2,1),base_steps=0,learning_time=timing_for(config),curriculum_progress={'level':0})
    def make_fake(*args,**kwargs):
        curriculum=SimpleNamespace(set_level=lambda _:None,sample=lambda *a,**k:np.zeros((4,4,3)),deadlines=np.full(4,1500.))
        return SimpleNamespace(env=SimpleNamespace(curriculum=curriculum,task=SimpleNamespace(deadlines=np.zeros(4)),reset=lambda *a:None))
    monkeypatch.setattr(run_upstream_t11,'make_collector',make_fake)
    monkeypatch.setattr(upstream_decision_evaluation,'panel',lambda *a,**k:{'episodes':[{}]*4,'actual_base_steps':0,'trace':[]})
    evaluate(tmp_path,'initial',collector,SimpleNamespace(config=BASE),config)
    reservations=list((tmp_path/'metrics').glob('evaluation_*_reservation.json'))
    assert len(reservations)==2
    assert sum(json.loads(p.read_text())['reserved_base_steps'] for p in reservations)==120000
