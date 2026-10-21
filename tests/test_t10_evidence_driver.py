"""Strict T10 evidence coverage, family configuration and runtime map boundaries."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import sys

import pytest

import evaluate_gathering_no_learning as driver
from _common import cfg_from_experiment
from test_cuda_evidence_gates import cuda_proof


def test_interrupted_episode_replay_preserves_result_and_rejects_divergence(tmp_path):
    path = tmp_path / 'episode.json'
    result = {'episode': {'terminal_reason': 'success', 'hold': 25}, 'seconds_batch': 10.}
    driver.publish_episode(path, result)
    original = path.read_bytes()
    driver.publish_episode(path, {**result, 'seconds_batch': 20.})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='Episode replay mismatch'):
        driver.publish_episode(path, {**result, 'episode': {'terminal_reason': 'task_deadline', 'hold': 0}})
    assert path.read_bytes() == original


def test_map_fingerprint_checks_runtime_array_content(tmp_path):
    config = tmp_path / 'terrain.yaml'
    config.write_text('terrain: {}\n')
    array = tmp_path / 'features.npy'
    array.write_bytes(b'original runtime array')
    scene = tmp_path / 'scene.json'
    scene.write_text(json.dumps({'files': {'features': {
        'file': array.name, 'sha256': hashlib.sha256(array.read_bytes()).hexdigest(),
    }}}))
    sources = driver.map_sources(config, scene)
    assert sources[str(array)] == hashlib.sha256(array.read_bytes()).hexdigest()
    array.write_bytes(b'changed without a manifest update')
    with pytest.raises(ValueError, match='runtime array fingerprint mismatch'):
        driver.map_sources(config, scene)


def test_effective_family_config_rejected_before_environment_construction(tmp_path, monkeypatch):
    run = SimpleNamespace(path=tmp_path, config='configs/experiment/t10_gathering_no_learning.yaml')
    (tmp_path / 'config').mkdir()
    cfg = cfg_from_experiment(run.config)
    monkeypatch.setattr(driver, 'config_for_family', lambda *args: cfg)
    scenes = driver.scene_cases()[:2]
    tag = 'flat_compact_23_31_mppi512_s1_t180'
    saved = asdict(cfg)
    saved['simulation']['episode_length_s'] += 1
    (tmp_path / f'config/{tag}.json').write_text(json.dumps(saved))
    monkeypatch.setattr(driver, 'LocalGoalEnvironment', lambda *args: pytest.fail('Constructed before config validation'))
    with pytest.raises(ValueError, match='effective config mismatch'):
        driver.execute_pair(run, scenes, 'mppi512', True, 180)


@pytest.mark.parametrize('change', ['consumer', 'planning', 'communication'])
def test_gate_b_consumer_revision_preserves_all_producer_dependencies(tmp_path, monkeypatch, change):
    import planner_evidence
    root = tmp_path / 't09'
    (root / 'metrics').mkdir(parents=True)
    gate = root / 'metrics/summary.json'
    gate.write_text(json.dumps({'gate_b_cpu': True}))
    consumer = 'scripts/evaluate_gathering_no_learning.py'
    paths = {'consumer': consumer, 'planning': 'source/planner.py', 'communication': 'source/communication.py'}
    producer = {path: 'before' for path in paths.values()}
    (root / 'run_manifest.json').write_text(json.dumps({'identity': {'source_snapshot': producer}}))
    current = dict(producer)
    current[paths[change]] = 'after'
    monkeypatch.setattr(planner_evidence, 'source_snapshot', lambda: current)
    monkeypatch.setattr(driver, 'map_sources', lambda: {})
    class ReachedEvidenceRun(Exception):
        pass
    def capture_run(*args, **kwargs):
        assert kwargs['settings']['consumer_version'] == 't10_evidence_driver_v3'
        raise ReachedEvidenceRun
    monkeypatch.setattr(driver, 'EvidenceRun', capture_run)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--run-dir', str(tmp_path / 't10'), '--gate-b', str(gate)])
    if change == 'consumer':
        with pytest.raises(ReachedEvidenceRun):
            driver.main()
    else:
        with pytest.raises(ValueError, match='dependencies differ'):
            driver.main()


def matrix_rows():
    rows = []
    for scene in driver.scene_cases():
        for variant in ('nmpc', 'mppi512'):
            for safety in (False, True):
                for duration in (96, 180):
                    rows.append({'scenario': scene, 'variant': variant, 'safety_enabled': safety,
                                 'duration_s': duration, 'samples': 512 if variant == 'mppi512' else None,
                                 'episode': {'success': True, 'collision': False},
                                 'execution_limit_violations': 0, 'proxy_swept_collisions': 0,
                                 'category': 'success'})
    return rows


@pytest.mark.parametrize('corruption', [None, 'duplicate', 'wrong_scene', 'lower_k', 'swept_collision'])
def test_summary_checks_exact_matrix_and_actual_simple_gate(tmp_path, corruption):
    rows = deepcopy(matrix_rows())
    if corruption == 'duplicate':
        rows[-1] = deepcopy(rows[-2])
    elif corruption == 'wrong_scene':
        rows[-1]['scenario']['positions'][0][0] += 1
    elif corruption == 'lower_k':
        rows[-1]['samples'] = 128
    elif corruption == 'swept_collision':
        simple = next(r for r in rows if r['variant'] == 'mppi512' and r['safety_enabled'] and r['duration_s'] == 180)
        simple['proxy_swept_collisions'] = 1
    (tmp_path / 'metrics').mkdir()
    for i, row in enumerate(rows):
        (tmp_path / f'metrics/episode_{i:03d}.json').write_text(json.dumps(row))
    summary = driver.summary(SimpleNamespace(path=tmp_path))
    assert summary['completed_episodes'] == 144
    assert summary['t10_functional_pass'] is (corruption is None)
    if corruption == 'swept_collision':
        assert summary['coverage_complete'] and summary['simple_passed'] == 17
    elif corruption is not None:
        assert not summary['coverage_complete']


@pytest.mark.parametrize('counts,expected', [
    ({'missing_or_expired_leader': 3, 'move_to_slot': 285, 'planner:solver_failure': 1425}, 'planning_rejection'),
    ({'missing_or_expired_leader': 270, 'move_to_slot': 18}, 'communication_missing'),
    ({'priority_yield': 260, 'move_to_slot': 28}, 'high_level_wait'),
    ({'move_to_slot': 288, 'safety_infeasible': 600}, 'safety_infeasible'),
])
def test_failure_classification_distinguishes_transient_and_persistent_symptoms(counts, expected):
    from collections import Counter
    episode = {'success': False, 'collision': False, 'terminal_reason': 'task_deadline',
               'duration_low_steps': 480, 'flatness_ok': True}
    assert driver.failure_category(episode, Counter(counts)) == expected


def test_failure_classification_preserves_collision_priority():
    from collections import Counter
    episode = {'success': True, 'collision': True, 'terminal_reason': 'collision',
               'duration_low_steps': 20, 'flatness_ok': True}
    assert driver.failure_category(episode, Counter()) == 'collision'


def cuda_gate(cuda_proof):
    import planner_evidence as evidence
    cfg,run,junit,acceptance,_=cuda_proof
    proof=evidence.cuda_acceptance_evidence(acceptance,cfg,junit)
    gate={
        'evidence_protocol':evidence.CUDA_EVIDENCE_VERSION,'planner_device':'cuda',
        'planner_metadata':evidence.planner_metadata(cfg),'gate_b_cpu':False,'gate_b_cuda':True,
        'coverage_complete':True,'simple_loops_pass':True,'numeric_repeat_pass':True,
        'device_records_match':True,'production_source_matches':True,
        'single_records':7000,'local_episodes':252,'repeat_records':630,'batch_panels':14,
        'regression_evidence':{'passed':True,'skipped':[],'tests':600,
                               'expected_real_acados':18,'real_acados_passed':18},
        'cuda_acceptance_evidence':proof,
    }
    path=run.path/'metrics/gate_b.json';path.write_text(json.dumps(gate))
    return cfg,run,junit,acceptance,path,gate


@pytest.mark.parametrize('corruption',[None,'cpu_gate','missing_numeric','missing_runtime','dtype','numerics','coverage','skip','short_regression'])
def test_formal_cuda_t10_requires_current_matching_cuda_proofs(cuda_proof,corruption):
    cfg,_,junit,acceptance,path,gate=cuda_gate(cuda_proof)
    if corruption=='cpu_gate':gate.update(gate_b_cuda=False,gate_b_cpu=True)
    elif corruption=='missing_numeric':junit=None
    elif corruption=='missing_runtime':acceptance=None
    elif corruption=='dtype':gate['planner_metadata']['dtype']='float64'
    elif corruption=='numerics':gate['planner_metadata']['numerics_version']='old'
    elif corruption=='coverage':gate['single_records']=6999
    elif corruption=='skip':gate['regression_evidence']['skipped']=['test_cuda_required']
    elif corruption=='short_regression':gate['regression_evidence']['tests']=599
    path.write_text(json.dumps(gate))
    if corruption is None:
        result=driver.load_gate_b(path,cfg,cuda_junit=junit,cuda_acceptance=acceptance)
        assert result['t10_cuda_acceptance']['passed']
    else:
        with pytest.raises(ValueError):
            driver.load_gate_b(path,cfg,cuda_junit=junit,cuda_acceptance=acceptance)


@pytest.mark.parametrize('corruption',[None,'cpu_record','pilot','missing_gate','gate_changed'])
def test_cuda_full_matrix_is_distinct_from_cpu_records_pilot_and_unbound_results(cuda_proof,corruption):
    import planner_evidence as evidence
    cfg,run,junit,acceptance,path,gate=cuda_gate(cuda_proof)
    run.identity['settings']['pilot']=corruption=='pilot'
    if corruption!='missing_gate':
        evidence.json_write(run.path/'config/gate_b.json',{'path':str(path.resolve()),
                            'sha256':evidence.file_sha256(path),'result':gate})
    rows=matrix_rows()
    metadata=evidence.planner_metadata(cfg)
    for row in rows:
        row.update(evidence_protocol=evidence.CUDA_EVIDENCE_VERSION,suite_planner_device='cuda',
                   planner_device='cuda' if row['variant']=='mppi512' else 'cpu',
                   planner_metadata=metadata if row['variant']=='mppi512' else None,
                   observation_source='actual_sensor_transport',training_interactions=0,optimizer_updates=0)
    if corruption=='cpu_record':next(row for row in rows if row['variant']=='mppi512')['planner_device']='cpu'
    if corruption=='gate_changed':path.write_text(json.dumps({**gate,'gate_b_cuda':False}))
    for i,row in enumerate(rows):
        evidence.json_write(run.path/f'metrics/episode_{i:03d}.json',row)
    value=driver.summary(run)
    assert value['completed_episodes']==144
    assert value['t10_cuda_functional_pass'] is (corruption is None)
    assert value['t10_functional_pass'] is (corruption is None)
    assert value['planner_device']=='cuda'
    if corruption=='pilot':assert value['full_task_validation']=='pilot'
    elif corruption is None:assert value['cuda_validation']=='passed'
    else:assert value['cuda_validation']=='incomplete'


@pytest.mark.parametrize('change',['consumer','planning'])
def test_cuda_gate_preserves_only_original_consumer_exception(cuda_proof,monkeypatch,change):
    import planner_evidence as evidence
    cfg,run,junit,acceptance,path,_=cuda_gate(cuda_proof)
    producer=run.identity['source_snapshot']
    current=deepcopy(producer)
    current['scripts/evaluate_gathering_no_learning.py' if change=='consumer' else 'source/planning.py']='changed'
    monkeypatch.setattr(evidence,'source_snapshot',lambda:deepcopy(current))
    if change=='consumer':
        gate=driver.load_gate_b(path,cfg,cuda_junit=junit,cuda_acceptance=acceptance)
        assert gate['t10_dependency_check']['consumer_changed']
    else:
        with pytest.raises(ValueError,match='dependencies differ'):
            driver.load_gate_b(path,cfg,cuda_junit=junit,cuda_acceptance=acceptance)
