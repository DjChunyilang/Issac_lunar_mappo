"""Append-only no-learning run evidence, strict fingerprinted resume."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import os
import pickle
import tempfile
import subprocess
import importlib.metadata
import sys
import xml.etree.ElementTree as ET
from _common import ROOT,load_yaml,cfg_from_experiment

CUDA_EVIDENCE_VERSION = 'planner_cuda_evidence_v1'
CUDA_ACCEPTANCE_VERSION = 'mppi_cuda_acceptance_v1'
CUDA_TEST_FILES = ('tests/test_mppi_cuda.py',)


def planner_device(cfg):
    """The host simulation remains CPU; this is the configured MPPI device."""
    if getattr(cfg, 'runtime_family', None) == 'upstream_scale_v1':
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import UpstreamRuntimeConfig
        if not isinstance(cfg, UpstreamRuntimeConfig):
            raise ValueError('Invalid upstream evidence configuration')
        device = str(cfg.planner_device).split(':', 1)[0]
    else:
        device = str(cfg.local_goal_contract.planner_device).split(':', 1)[0]
    if device not in ('cpu', 'cuda'):
        raise ValueError('Unsupported planner evidence device')
    return device


def planner_metadata(cfg):
    if getattr(cfg, 'runtime_family', None) == 'upstream_scale_v1':
        return cfg.metadata
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.mppi import MPPIConfig
    return MPPIConfig.from_contract(cfg.local_goal_contract, cfg.terrain.slope_speed_scale).metadata()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cuda_test_sources():
    return {name: file_sha256(ROOT / name) for name in CUDA_TEST_FILES}


def cuda_required_tests():
    """Collect identities without running fixtures or querying a GPU."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-o', 'addopts=', *CUDA_TEST_FILES],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f'CUDA contract collection failed: {result.stderr or result.stdout}')
    identities = set()
    for line in result.stdout.splitlines():
        if not any(line.startswith(name + '::') for name in CUDA_TEST_FILES):
            continue
        parts = line.strip().split('::')
        classname = parts[0].removesuffix('.py').replace('/', '.')
        if len(parts) > 2:
            classname += '.' + '.'.join(parts[1:-1])
        identities.add((classname, parts[-1]))
    if not identities:
        raise ValueError('No required CUDA contracts collected')
    return identities


def cuda_junit_evidence(path, *, reference_snapshot=None):
    """Require the full current GPU contract suite and its executed test sources."""
    if path is None:
        return {'passed': False, 'reason': 'missing_cuda_junit'}
    try:
        path = Path(path)
        tests = list(ET.parse(path).iter('testcase'))
        expected = cuda_required_tests()
        actual = [(t.get('classname', ''), t.get('name', '')) for t in tests]
        failures = [name for t, (_, name) in zip(tests, actual)
                    if t.find('failure') is not None or t.find('error') is not None]
        skipped = [name for t, (_, name) in zip(tests, actual) if t.find('skipped') is not None]
        provenance_path = path.with_suffix('.source.json')
        provenance = json.loads(provenance_path.read_text())
        expected_source = source_snapshot() if reference_snapshot is None else reference_snapshot
        source_matches = provenance.get('source_snapshot') == expected_source
        test_sources = cuda_test_sources()
        tests_match = provenance.get('test_sources') == test_sources
        missing = sorted(expected - set(actual))
        duplicate = len(actual) != len(set(actual))
        return {
            'passed': not failures and not skipped and not missing and not duplicate
                      and source_matches and tests_match,
            'junit': str(path.resolve()), 'sha256': file_sha256(path),
            'source_sha256': file_sha256(provenance_path), 'test_sources': test_sources,
            'tests': len(tests), 'required_tests': len(expected), 'missing_tests': missing,
            'failures': failures, 'skipped': skipped, 'duplicate_tests': duplicate,
            'source_matches': source_matches, 'test_sources_match': tests_match,
        }
    except (OSError, ValueError, ET.ParseError, TypeError, KeyError) as error:
        return {'passed': False, 'reason': f'invalid_cuda_junit:{error}'}


def cuda_acceptance_evidence(path, cfg, junit=None, *, reference_snapshot=None):
    """Bind runtime coverage to immutable files, actual metadata and strict JUnit."""
    if path is None:
        return {'passed': False, 'reason': 'missing_cuda_acceptance'}
    try:
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import FAMILIES
        path = Path(path)
        value = json.loads(path.read_text())
        checks = ('numeric_test_pass', 'no_skip', 'matrix_complete', 'complex_inputs',
                  'calibration', 'snapshot', 'device', 'precision')
        expected_panels = {(k, b) for k in (512, 1024, 2048) for b in (1, 4, 16, 64)}
        panels = value.get('panels', [])
        panel_ids = [(p['samples'], p['batch_size']) for p in panels]
        numeric = cuda_junit_evidence(junit, reference_snapshot=reference_snapshot)
        junit_record = value.get('contract_junit', {})
        junit_matches = (numeric.get('passed', False)
                         and junit_record.get('path') == numeric.get('junit')
                         and junit_record.get('sha256') == numeric.get('sha256'))
        manifest_record = value['manifest']
        manifest_path = Path(manifest_record['path'])
        manifest = json.loads(manifest_path.read_text())
        recorded_config_path = manifest_path.parent / 'config/experiment.json'
        recorded_config = json.loads(recorded_config_path.read_text())
        recorded_cfg = cfg_from_experiment(recorded_config_path)
        expected_source = source_snapshot() if reference_snapshot is None else reference_snapshot
        source_matches = (value.get('source_snapshot') == expected_source
                          and manifest['identity']['source_snapshot'] == expected_source)
        manifest_matches = (manifest_record['sha256'] == file_sha256(manifest_path)
                            and manifest.get('training_interactions') == 0
                            and manifest.get('optimizer_updates') == 0)
        artifacts = value.get('artifacts', [])
        artifacts_match = bool(artifacts) and all(
            Path(record['path']).resolve().is_relative_to(manifest_path.resolve().parent)
            and record['sha256'] == file_sha256(record['path']) for record in artifacts
        )
        metadata_matches = (planner_device(cfg) == 'cuda'
                            and value.get('planner_metadata') == planner_metadata(cfg)
                            and recorded_config == manifest['identity']['resolved_config']
                            and planner_device(recorded_cfg) == 'cuda'
                            and planner_metadata(recorded_cfg) == planner_metadata(cfg)
                            and recorded_cfg.simulation.device == 'cpu'
                            and recorded_cfg.local_goal_contract.training_paused)
        passed = (value.get('schema_version') == CUDA_ACCEPTANCE_VERSION
                  and value.get('passed') is True and source_matches and manifest_matches
                  and value.get('test_sources') == cuda_test_sources()
                  and metadata_matches and junit_matches and artifacts_match
                  and all(value.get('checks', {}).get(name) is True for name in checks)
                  and len(panels) == len(expected_panels) and set(panel_ids) == expected_panels
                  and all(p.get('passed') is True for p in panels)
                  and set(value.get('complex_families', [])) == set(FAMILIES))
        return {'passed': bool(passed), 'path': str(path.resolve()), 'sha256': file_sha256(path),
                'schema_version': value.get('schema_version'), 'source_matches': source_matches,
                'manifest_matches': manifest_matches, 'artifacts_match': artifacts_match,
                'planner_metadata_matches': metadata_matches, 'planner_metadata': value.get('planner_metadata'),
                'numeric_evidence': numeric, 'junit_matches': junit_matches,
                'checks': value.get('checks', {}), 'completed_panels': len(panels)}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'passed': False, 'reason': f'invalid_cuda_acceptance:{error}'}


def source_snapshot():
    directory=ROOT/'source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering'
    paths=[p for p in directory.rglob('*') if p.suffix in {'.py','.cu','.cpp','.h'}]
    paths+=list((ROOT/'scripts').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def json_write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
    text=json.dumps(value,indent=2,allow_nan=False)
    # Same-directory atomic publication; incomplete temp is never a completed case.
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        f.write(text);temp=Path(f.name)
    try:
        os.link(temp,path)
    finally:
        temp.unlink()


class EvidenceRun:
    def __init__(self,path,config,*,resume=False,settings=None):
        self.path=Path(path);self.config=config
        snapshot=source_snapshot();resolved=load_yaml(config)
        self.cfg=cfg_from_experiment(config);self.planner_device=planner_device(self.cfg)
        dependencies={'python':sys.version}
        for package in ('torch','numpy','scipy','osqp','casadi','acados-template'):
            try:dependencies[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:dependencies[package]=None
        self.identity={'source_snapshot':snapshot,'resolved_config':resolved,'settings':settings or {},'dependencies':dependencies}
        upstream = getattr(self.cfg, 'runtime_family', None) == 'upstream_scale_v1'
        if upstream:
            self.identity['contracts'] = self.cfg.metadata
        if self.planner_device == 'cuda':
            self.identity['evidence_protocol'] = CUDA_EVIDENCE_VERSION
        manifest=self.path/'run_manifest.json'
        if manifest.exists():
            if not resume: raise FileExistsError('Existing run requires explicit --resume')
            old=json.loads(manifest.read_text())
            if old['identity'] != self.identity: raise ValueError('Run source/config/settings mismatch')
        else:
            if resume: raise ValueError('Cannot resume a missing run')
            for d in ('metrics','logs','config','state','samples'): (self.path/d).mkdir(parents=True,exist_ok=True)
            metadata = {'cuda_validation': 'deferred'}
            if upstream:
                metadata.update(runtime_family=self.cfg.runtime_family,
                                planner_metadata=self.cfg.metadata, formal_training='paused',
                                runtime_execution='not_verified_by_manifest')
            if self.planner_device == 'cuda':
                metadata.update(cuda_validation='pending', planner_device='cuda', simulation_device='cpu',
                                planner_metadata=planner_metadata(self.cfg), formal_training='paused')
            json_write(manifest,{'identity':self.identity,'platform':platform.platform(),
                'cpu':platform.processor(),'affinity':sorted(os.sched_getaffinity(0)),
                'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'training_interactions':0,'optimizer_updates':0, **metadata})
            json_write(self.path/'config/experiment.json',resolved)

    def resolved(self,cfg):
        p=self.path/'config/effective_env_config.json'
        if not p.exists(): json_write(p,asdict(cfg))

    def snapshot(self,name,state):
        p=self.path/'state'/name
        if p.exists():
            if p.with_suffix(p.suffix+'.json').exists():raise FileExistsError(p)
            # Preserve an interrupted publication; it was never a committed checkpoint.
            checksum=hashlib.sha256(p.read_bytes()).hexdigest()
            orphan=p.with_name(p.name+'.orphan_'+checksum)
            if orphan.exists():p.unlink()
            else:p.rename(orphan)
        data=pickle.dumps(state,protocol=5)
        with tempfile.NamedTemporaryFile('wb',dir=p.parent,delete=False) as stream:
            stream.write(data);temporary=Path(stream.name)
        try:os.link(temporary,p)
        finally:temporary.unlink()
        json_write(p.with_suffix(p.suffix+'.json'),{'sha256':hashlib.sha256(data).hexdigest(),
            'purpose':'local trusted diagnostic state; no solver object or training checkpoint'})
        return str(p)

    def restore(self,name):
        p=self.path/'state'/name
        data=p.read_bytes();meta=json.loads(p.with_suffix(p.suffix+'.json').read_text())
        if hashlib.sha256(data).hexdigest()!=meta['sha256']: raise ValueError('Snapshot checksum mismatch')
        return pickle.loads(data)
