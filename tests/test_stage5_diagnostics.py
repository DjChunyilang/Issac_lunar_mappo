"""Freeze, selection, restart and legal script contracts; no training."""
from dataclasses import replace
from copy import deepcopy
import json
import numpy as np
import pytest

from test_planner_baselines import low_input
from _common import cfg_from_experiment
from planner_evidence import EvidenceRun,json_write
from benchmark_planners import Runner,frozen_cases,single_case,local_loop
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import to_input,FAMILIES,digest,make_case
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import (
    LeaderSlotPolicy,ScriptObservation,ScriptMessage,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import LocalGoalEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.safety import SafetyInput
from test_cuda_evidence_gates import cuda_proof


@pytest.mark.parametrize('family',FAMILIES)
def test_case_shapes_fingerprint_and_common_information(family):
    case=make_case(family,0)
    item=to_input(case);item.validate()
    assert item.terrain.snapshot().features.shape[1]==5
    assert make_case(family,0)==case
    bad=deepcopy(case);bad['goal'][0]+=1
    with pytest.raises(ValueError):to_input(bad)


def test_thousand_cases_stratified_and_no_result_filtering():
    cases=frozen_cases()
    assert len(cases)==1000 and len({c['case_id'] for c in cases})==1000
    for family in FAMILIES:assert sum(c['case_id'].startswith(family+'_') for c in cases)==110
    assert sum(c['solvability']=='blocked_by_design' for c in cases)>=110


@pytest.mark.parametrize('variant',['line','quintic','primitive47','mppi512'])
def test_runner_native_and_safety_outputs_restore(tmp_path,variant):
    cfg=cfg_from_experiment('configs/experiment/t09_planner_benchmark.yaml')
    runner=Runner(cfg,variant,tmp_path)
    row,p,state,item=single_case(runner,make_case('flat',0))
    p.load_state_dict(state);again,_=runner.plan(p,item)
    np.testing.assert_allclose(again.trajectory.controls,row['native']['controls'],atol=1e-5)
    assert row['safety_off']['safety_status']=='projection_disabled'
    assert row['safety_on']['safety_status'] in ('unchanged','projected','verified_brake','unverified_brake')
    if variant=='primitive47':
        assert len(row['primitive_selection']['candidates'])==47
        chosen=row['primitive_selection']['chosen_id']
        valid=[v for v in row['primitive_selection']['candidates'] if v['valid']]
        assert (row['metrics']['common_cost'],chosen)==min((v['cost'],v['primitive_id']) for v in valid)


def observation(agent=1, *, now=0., full=True, expiry=1.):
    item=low_input()
    leader=ScriptMessage(0,np.array([0,0]),np.array([0,0]),0.,full,0.,expiry,True)
    return ScriptObservation(0,agent,np.array([1.6,0,0]),np.zeros(2),now,item.terrain,(leader,),())


def test_script_slots_hysteresis_missing_and_snapshot():
    policy=LeaderSlotPolicy();obs=observation()
    action,rows=policy.action([obs],(1,4))
    assert not action.hold[0,1] and action.physical[0,1,0]==pytest.approx(.8)
    saved=policy.state_dict();restored=LeaderSlotPolicy();restored.load_state_dict(saved)
    np.testing.assert_array_equal(action.physical,restored.action([obs],(1,4))[0].physical)
    atslot=replace(obs,pose=np.array([.8,0,0]))
    assert policy.action([atslot],(1,4))[0].hold[0,1]
    assert policy.action([replace(atslot,pose=np.array([1.01,0,0]))],(1,4))[0].hold[0,1]
    assert not policy.action([replace(atslot,pose=np.array([1.11,0,0]))],(1,4))[0].hold[0,1]
    assert policy.action([observation(now=2.)],(1,4))[0].hold[0,1]
    assert policy.action([observation(full=False)],(1,4))[0].hold[0,1]
    policy.reset([0]);assert policy.states=={}


def test_evidence_immutable_and_strict_resume(tmp_path):
    config='configs/experiment/t09_planner_benchmark.yaml'
    run=EvidenceRun(tmp_path/'run',config,settings={'test':True})
    with pytest.raises(FileExistsError):EvidenceRun(run.path,config)
    EvidenceRun(run.path,config,resume=True,settings={'test':True})
    with pytest.raises(ValueError):EvidenceRun(run.path,config,resume=True,settings={'test':False})
    run.snapshot('state.pkl',{'x':np.ones(2)})
    np.testing.assert_array_equal(run.restore('state.pkl')['x'],np.ones(2))
    with pytest.raises(FileExistsError):run.snapshot('state.pkl',{})
    json_write(run.path/'metrics/failure.json',{'failed':True})
    with pytest.raises(FileExistsError):json_write(run.path/'metrics/failure.json',{'failed':False})


def test_actual_environment_safety_restore_isolated_and_hidden_information(tmp_path):
    cfg=cfg_from_experiment('configs/experiment/t10_gathering_no_learning.yaml')
    cfg.terrain.type='flat_proxy';cfg.terrain.amplitude=0
    cfg.local_goal_contract.mppi_samples=64
    env=LocalGoalEnvironment(cfg,tmp_path)
    try:
        policy=LeaderSlotPolicy()
        action,_=policy.action(env.script_observations(),(2,4))
        first=env.step(action,max_low_steps=1)
        assert len(first.records[0]['planner_execution'])==8
        saved=env.state_dict();policy_state=policy.state_dict()
        observations=env.script_observations();before=LeaderSlotPolicy();before.load_state_dict(policy_state)
        before_action=before.action(observations,(2,4))[0]
        env.core.oracle_point.add_(1000);env.core.positions[1,:,0].add_(100)
        after=LeaderSlotPolicy();after.load_state_dict(policy_state)
        after_action=after.action(env.script_observations(),(2,4))[0]
        np.testing.assert_array_equal(before_action.physical[0],after_action.physical[0])
        env.load_state_dict(saved)
        next_action,_=policy.action(env.script_observations(),(2,4))
        x=env.step(next_action,max_low_steps=1)
        env.load_state_dict(saved);y=env.step(next_action,max_low_steps=1)
        np.testing.assert_allclose(x.records[0]['commands'],y.records[0]['commands'],atol=1e-5)
        second=env.projectors[4].state_dict()
        env.reset([0])
        np.testing.assert_array_equal(second['last_control'],env.projectors[4].last_control)
        assert env.projectors[0].last_control is None
        assert 'safety_states' in saved
        for row in x.records[0]['planner_execution']:
            assert row['safety']['applied_control']==row['applied_control']
        # Changing a projector never rewrites the nominal PlannerOutput.
        assert all(p is not None for p in env.plans[4:])
    finally:env.close()


def test_tensor_batch_panel_uses_real_kernel_and_order(tmp_path):
    from benchmark_planners import batch_panel
    cfg=cfg_from_experiment('configs/experiment/t09_planner_benchmark.yaml')
    runner=Runner(cfg,'mppi512',tmp_path)
    # Keep the integration test small; the frozen benchmark still runs K512/1024/2048.
    runner.cfg.local_goal_contract.mppi_samples=64
    panel=batch_panel(runner,[make_case('flat',i) for i in range(4)],4)
    assert panel['scheduler']=='tensor_batch' and len(panel['outputs'])==4
    assert panel['throughput_plans_s']>0
    assert all(out['valid'] for out in panel['outputs'])


def test_formal_t10_refuses_missing_gate(tmp_path,monkeypatch):
    import sys
    from evaluate_gathering_no_learning import main
    monkeypatch.setattr(sys,'argv',['evaluate','--run-dir',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='requires --gate-b'):main()
    assert not (tmp_path/'run').exists()


def test_interrupted_snapshot_preserved_and_not_restored(tmp_path):
    run=EvidenceRun(tmp_path/'run','configs/experiment/t09_planner_benchmark.yaml')
    interrupted=run.path/'state/x.pkl';interrupted.write_bytes(b'partial')
    run.snapshot('x.pkl',{'valid':True})
    assert run.restore('x.pkl')['valid']
    assert len(list((run.path/'state').glob('*.orphan_*')))==1


@pytest.mark.parametrize('variant',['nmpc','mppi512'])
def test_real_planners_share_safety_and_actual_control_feedback(tmp_path,variant,monkeypatch):
    import os
    if variant=='nmpc':assert os.environ.get('ACADOS_SOURCE_DIR'), 'Real solver required, no skip'
    cfg=cfg_from_experiment('configs/experiment/t10_gathering_no_learning.yaml')
    cfg.simulation.num_envs=1;cfg.terrain.type='flat_proxy';cfg.terrain.amplitude=0
    cfg.local_goal_contract.planner_adapter='nmpc_polar_v1' if variant=='nmpc' else 'mppi_terrain_neighbor_v1'
    cfg.local_goal_contract.planner_revision='prefix_consistency_v2' if variant=='nmpc' else 'mppi_terrain_neighbor_v1'
    cfg.local_goal_contract.mppi_samples=64
    env=LocalGoalEnvironment(cfg,tmp_path)
    try:
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import PolicyAction
        action=PolicyAction(np.tile([.8,0.],(1,4,1)),np.zeros((1,4),bool))
        transition=env.step(action,max_low_steps=1)
        record=transition.records[0]
        assert len(record['planner_execution'])==4
        import lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment as environment
        original=environment.plan_batch;captured=[]
        def inspect_inputs(planners,inputs,**kwargs):
            captured.extend(inputs)
            return original(planners,inputs,**kwargs)
        predictions=deepcopy(env.execution_plans)
        monkeypatch.setattr(environment,'plan_batch',inspect_inputs)
        env.step(action,max_low_steps=1)
        for row in record['planner_execution']:
            a=row['agent_id'];i=a
            assert row['safety']['applied_control']==record['commands'][0][a]
            request=captured[a]
            np.testing.assert_allclose(request.previous_control,row['applied_control'],atol=1e-6)
            prediction=predictions[i]
            assert row['execution_prediction_published']==(prediction is not None)
            if prediction is not None:np.testing.assert_array_equal(prediction.controls[0],row['applied_control'])
    finally:env.close()


def test_first_move_sampling_after_hold_and_public_resume(tmp_path):
    from evaluate_gathering_no_learning import execute_pair,scene_cases
    run=EvidenceRun(tmp_path/'run','configs/experiment/t10_gathering_no_learning.yaml',settings={'bounded_test':True})
    cases=scene_cases()[:2]
    execute_pair(run,cases,'mppi512',True,180,max_macros=1)
    execute_pair(run,cases,'mppi512',True,180,max_macros=2)
    samples=list((run.path/'samples').glob('*.npz'))
    assert samples, 'First actual MOVE after HOLD must be recorded before step 25'
    first_step=min(int(p.stem.split('_')[-3]) for p in samples)
    assert first_step<25
    traces=[json.loads(p.read_text()) for p in (run.path/'metrics').glob('trace_*.json')]
    assert len(traces)==2
    for trace in traces:
        for record in trace['records']:
            for row in record['planner_execution']:
                i=row['env_id']*4+row['agent_id'];prediction=record['execution_predictions'][i]
                if prediction is not None:np.testing.assert_allclose(prediction['controls'][0],row['applied_control'])
                assert 'diagnostics' in row
    assert len(list((run.path/'state').glob('*.pkl')))==2


def test_local_benchmark_expiry_uses_environment_conservative_rule():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import shifted_input
    item=to_input(make_case('cross',0))
    source={peer.sender_id:peer.prediction for peer in item.neighbors}
    for _ in range(4):item=shifted_input(item,np.zeros(2),neighbor_messages=source)
    prediction=item.neighbors[0].prediction
    np.testing.assert_array_equal(prediction.positions,np.tile(source[2].positions[0],(16,1)))
    np.testing.assert_allclose(prediction.radii,.46+1.15*(item.now+np.arange(16)*.2))
    assert prediction.expires_at==.6


def test_cuda_variants_preserve_configured_revision_and_keep_four_baselines_on_cpu(tmp_path):
    from evaluate_gathering_no_learning import config_for_family
    from _common import ROOT
    path=tmp_path/'cuda.yaml'
    path.write_text(f'extends: {ROOT / "configs/experiment/t10_gathering_no_learning.yaml"}\n'
                    'local_goal:\n  planner_device: cuda\n'
                    '  planner_adapter: mppi_terrain_neighbor_stable_v2\n'
                    '  planner_revision: mppi_terrain_neighbor_stable_v2\n')
    cfg=cfg_from_experiment(path)
    for variant in ('mppi512','mppi1024','mppi2048'):
        runner=Runner(cfg,variant,tmp_path/variant)
        assert runner.cfg.local_goal_contract.planner_adapter=='mppi_terrain_neighbor_stable_v2'
        assert runner.meta['mppi']['numerics_version']=='fp32_fp64_accum_v2'
        assert runner.cfg.local_goal_contract.mppi_samples==int(variant[4:])
        assert runner.device_evidence()['planner_device']=='cuda'
    for variant in ('line','quintic','primitive47','nmpc'):
        assert Runner(cfg,variant,tmp_path/variant).device_evidence()['planner_device']=='cpu'
    for variant in ('nmpc','mppi512'):
        effective=config_for_family(path,'flat_compact',variant,True,180)
        assert effective.simulation.device=='cpu'
        assert effective.local_goal_contract.planner_device==('cpu' if variant=='nmpc' else 'cuda')
        if variant=='mppi512':assert effective.local_goal_contract.planner_revision=='mppi_terrain_neighbor_stable_v2'


def test_cuda_regression_does_not_inherit_cpu_cuda_skip_exception(tmp_path):
    import xml.etree.ElementTree as ET
    from benchmark_planners import regression_evidence
    from planner_evidence import source_snapshot
    baseline=ET.parse('outputs/runs/t07_mppi_terrain_neighbor/acceptance_seed23_cpu_20261010_01/metrics/contracts_acceptance_compat_final.xml')
    root=ET.Element('testsuite')
    real=[deepcopy(t) for t in baseline.iter('testcase') if 'real_acados' in t.attrib['name']]
    for testcase in real:root.append(testcase)
    for i in range(600-len(real)):
        ET.SubElement(root,'testcase',classname='contract',name=f'passed_{i}')
    skipped=ET.SubElement(root,'testcase',classname='contract',name='test_cuda_optional_cpu_only')
    ET.SubElement(skipped,'skipped')
    path=tmp_path/'regression.xml';ET.ElementTree(root).write(path,encoding='unicode')
    path.with_suffix('.source.json').write_text(json.dumps(source_snapshot()))
    assert regression_evidence(path)['passed']
    assert not regression_evidence(path,device='cuda')['passed']


def test_cuda_gate_b_needs_full_current_device_coverage_and_strict_acceptance(cuda_proof,monkeypatch):
    import benchmark_planners as benchmark
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import VARIANTS
    cfg,run,junit,acceptance,_=cuda_proof
    cases=[{'case_id':f'case_{i:04d}','family':FAMILIES[min(i//110,8)],'sha256':f'input_{i}'} for i in range(1000)]
    json_write(run.path/'config/cases.json',{'cases':cases})
    locals_=[cases[110*i+j] for i in range(9) for j in range(2)]
    repeats_=[cases[110*i+j] for i in range(9) for j in range(3)]+cases[990:993]
    first_row=None
    for variant in VARIANTS:
        metadata=Runner(cfg,variant,run.path/variant).device_evidence()
        rows=[{'case_id':case['case_id'],'variant':variant,'seed':23,'input_sha256':case['sha256'],
               'native':{'valid':True,'reason':None},'metrics':{'progress':.8},
               'timings':{'planning_s':.1},'rss_mib':1.,'safety_on':{'safety_status':'unchanged'},
               **metadata} for case in cases]
        json_write(run.path/f'metrics/single_{variant}_0000.json',{'rows':rows})
        if variant=='mppi512':first_row=deepcopy(rows[0])
        for case in locals_:
            for safety in (False,True):
                json_write(run.path/f'metrics/loop_{variant}_{case["case_id"]}_{int(safety)}.json',
                           {'case_id':case['case_id'],'family':case['family'],'variant':variant,
                            'safety_enabled':safety,'success':True,'collision':False,'limit_violation':False,**metadata})
        for case in repeats_:
            for seed in (23,31,47):
                json_write(run.path/f'metrics/repeat_{variant}_{case["case_id"]}_{seed}.json',
                           {'passed':True,'row':{'case_id':case['case_id'],'variant':variant,'seed':seed,**metadata}})
        for batch in (4,16):
            json_write(run.path/f'metrics/panel_{variant}_{batch}.json',{'variant':variant,'batch_size':batch,**metadata})
    monkeypatch.setattr(benchmark,'regression_evidence',lambda *args,**kwargs: {'passed':True})
    monkeypatch.setattr(benchmark,'source_snapshot',lambda:deepcopy(run.identity['source_snapshot']))
    passed=benchmark.aggregate(run,cuda_junit=junit,cuda_acceptance=acceptance)
    assert passed['gate_b_cuda'] and not passed['gate_b_cpu'],passed
    assert passed['required_counts']=={'single':7000,'local':252,'repeat':630,'panel':14}
    assert not benchmark.aggregate(run,cuda_junit=junit)['gate_b_cuda']
    # A complete count with a copied CPU MPPI row cannot certify GPU production.
    path=run.path/'metrics/single_mppi512_0000.json'
    value=json.loads(path.read_text());value['rows'][0]['planner_device']='cpu';path.write_text(json.dumps(value))
    mismatched=benchmark.aggregate(run,cuda_junit=junit,cuda_acceptance=acceptance)
    assert not mismatched['gate_b_cuda'] and not mismatched['device_records_match']
    value['rows'][0]=first_row;path.write_text(json.dumps(value))
    # Duplicate panel publications cannot be used to satisfy an exact 14-panel suite.
    json_write(run.path/'metrics/panel_duplicate_4.json',json.loads((run.path/'metrics/panel_line_4.json').read_text()))
    assert not benchmark.aggregate(run,cuda_junit=junit,cuda_acceptance=acceptance)['gate_b_cuda']
