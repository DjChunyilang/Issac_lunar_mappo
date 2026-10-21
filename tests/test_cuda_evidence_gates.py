"""CPU-only tests of CUDA evidence completeness and provenance; no GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
import planner_evidence as evidence
from _common import ROOT, cfg_from_experiment


@pytest.fixture
def cuda_proof(tmp_path, monkeypatch):
    snapshot = {'scripts/evaluate_gathering_no_learning.py': 'consumer', 'source/planning.py': 'planner'}
    tests = {'tests/test_mppi_cuda.py': 'required-test-source'}
    monkeypatch.setattr(evidence, 'source_snapshot', lambda: deepcopy(snapshot))
    monkeypatch.setattr(evidence, 'cuda_test_sources', lambda: deepcopy(tests))
    monkeypatch.setattr(evidence, 'cuda_required_tests', lambda: {('tests.test_mppi_cuda', 'required_cuda_contract')})
    config = tmp_path / 'cuda.yaml'
    config.write_text(f'extends: {ROOT / "configs/experiment/t09_planner_benchmark.yaml"}\n'
                      'local_goal:\n  planner_device: cuda\n'
                      '  planner_adapter: mppi_terrain_neighbor_stable_v2\n'
                      '  planner_revision: mppi_terrain_neighbor_stable_v2\n'
                      '  mppi_query_chunk: 1024\n')
    cfg = cfg_from_experiment(config)
    run = evidence.EvidenceRun(tmp_path / 'runtime', config, settings={'runtime': True})
    suite = ET.Element('testsuite')
    ET.SubElement(suite, 'testcase', classname='tests.test_mppi_cuda', name='required_cuda_contract')
    junit = tmp_path / 'cuda.xml'
    ET.ElementTree(suite).write(junit, encoding='unicode')
    evidence.json_write(junit.with_suffix('.source.json'), {'source_snapshot': snapshot, 'test_sources': tests})
    artifact = run.path / 'metrics/runtime.json'
    evidence.json_write(artifact, {'actual_panel_records': 'unit-test-fixture'})
    manifest = run.path / 'run_manifest.json'
    acceptance = run.path / 'metrics/strict_acceptance.json'
    value = {
        'schema_version': evidence.CUDA_ACCEPTANCE_VERSION, 'passed': True,
        'source_snapshot': snapshot, 'test_sources': tests,
        'planner_metadata': evidence.planner_metadata(cfg),
        'contract_junit': {'path': str(junit.resolve()), 'sha256': evidence.file_sha256(junit)},
        'manifest': {'path': str(manifest.resolve()), 'sha256': evidence.file_sha256(manifest)},
        'checks': {name: True for name in ('numeric_test_pass', 'no_skip', 'matrix_complete',
                                          'complex_inputs', 'calibration', 'snapshot', 'device', 'precision')},
        'panels': [{'samples': k, 'batch_size': b, 'passed': True}
                   for k in (512, 1024, 2048) for b in (1, 4, 16, 64)],
        'complex_families': ['flat', 'slope', 'crater', 'narrow', 'head_on', 'cross', 'hold', 'reverse', 'unknown'],
        'artifacts': [{'path': str(artifact.resolve()), 'sha256': evidence.file_sha256(artifact)}],
    }
    evidence.json_write(acceptance, value)
    return cfg, run, junit, acceptance, value


def test_cuda_evidence_accepts_bound_complete_proof_without_running_gpu(cuda_proof):
    cfg, run, junit, acceptance, _ = cuda_proof
    assert evidence.cuda_junit_evidence(junit)['passed']
    result = evidence.cuda_acceptance_evidence(acceptance, cfg, junit)
    assert result['passed'] and result['completed_panels'] == 12
    manifest = json.loads((run.path / 'run_manifest.json').read_text())
    assert manifest['cuda_validation'] == 'pending'
    assert manifest['identity']['evidence_protocol'] == evidence.CUDA_EVIDENCE_VERSION
    assert manifest['planner_metadata']['numerics_version'] == 'fp32_fp64_accum_v2'


@pytest.mark.parametrize('corruption', ['bare_boolean', 'device', 'numerics', 'matrix_duplicate',
                                      'matrix_missing', 'complex_missing', 'test_sources', 'source',
                                      'check', 'junit_sha', 'manifest_sha', 'artifact_sha', 'no_artifacts'])
def test_cuda_acceptance_rejects_incomplete_or_unbound_proof(cuda_proof, corruption):
    cfg, _, junit, acceptance, original = cuda_proof
    value = deepcopy(original)
    if corruption == 'bare_boolean': value = {'passed': True}
    elif corruption == 'device': value['planner_metadata']['device'] = 'cpu'
    elif corruption == 'numerics': value['planner_metadata']['numerics_version'] = 'other'
    elif corruption == 'matrix_duplicate': value['panels'][-1] = value['panels'][-2]
    elif corruption == 'matrix_missing': value['panels'].pop()
    elif corruption == 'complex_missing': value['complex_families'].pop()
    elif corruption == 'test_sources': value['test_sources']['tests/test_mppi_cuda.py'] = 'different'
    elif corruption == 'source': value['source_snapshot']['source/planning.py'] = 'different'
    elif corruption == 'check': value['checks']['precision'] = False
    elif corruption == 'junit_sha': value['contract_junit']['sha256'] = 'different'
    elif corruption == 'manifest_sha': value['manifest']['sha256'] = 'different'
    elif corruption == 'artifact_sha': value['artifacts'][0]['sha256'] = 'different'
    elif corruption == 'no_artifacts': value['artifacts'] = []
    acceptance.write_text(json.dumps(value))
    assert not evidence.cuda_acceptance_evidence(acceptance, cfg, junit)['passed']


@pytest.mark.parametrize('corruption', ['failure', 'error', 'skip', 'missing', 'duplicate', 'source', 'test_source'])
def test_cuda_junit_requires_all_current_tests_no_skip_and_exact_sources(cuda_proof, corruption):
    cfg, _, junit, acceptance, _ = cuda_proof
    tree = ET.parse(junit)
    root = tree.getroot()
    testcase = root.find('testcase')
    if corruption in ('failure', 'error'): ET.SubElement(testcase, corruption)
    elif corruption == 'skip': ET.SubElement(testcase, 'skipped')
    elif corruption == 'missing': root.remove(testcase)
    elif corruption == 'duplicate': root.append(deepcopy(testcase))
    else:
        sidecar = junit.with_suffix('.source.json')
        value = json.loads(sidecar.read_text())
        if corruption == 'source': value['source_snapshot']['source/planning.py'] = 'different'
        else: value['test_sources']['tests/test_mppi_cuda.py'] = 'different'
        sidecar.write_text(json.dumps(value))
    tree.write(junit, encoding='unicode')
    assert not evidence.cuda_junit_evidence(junit)['passed']
    assert not evidence.cuda_acceptance_evidence(acceptance, cfg, junit)['passed']


def test_old_cpu_manifest_semantics_and_resume_remain_compatible(tmp_path):
    config = 'configs/experiment/t09_planner_benchmark.yaml'
    run = evidence.EvidenceRun(tmp_path / 'cpu', config)
    manifest = json.loads((run.path / 'run_manifest.json').read_text())
    assert manifest['cuda_validation'] == 'deferred'
    assert 'evidence_protocol' not in manifest['identity']
    assert 'planner_metadata' not in manifest
    evidence.EvidenceRun(run.path, config, resume=True)


def test_actual_cuda_collection_contains_every_required_contract():
    required = evidence.cuda_required_tests()
    assert required and all(classname == 'tests.test_mppi_cuda' for classname, _ in required)
    assert any(name.startswith('test_cpu_cuda_same_injected_noise') for _, name in required)
