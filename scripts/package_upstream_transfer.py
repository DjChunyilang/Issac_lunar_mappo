#!/usr/bin/env python3
"""Build a local source/data handoff, excluding weights, solver caches and venvs."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import tarfile

from _common import ROOT
from evaluate_gathering_no_learning import map_sources
from planner_evidence import json_write,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);a=p.parse_args()
    target=ROOT/'outputs/runs/upstream_scale_t11'/a.run_id;target.mkdir(parents=True,exist_ok=False)
    for name in ('config','metrics','artifacts'):(target/name).mkdir()
    files=set()
    for dirname,pattern in [('scripts','*.py'),('scripts','*.sh'),('source','*.py'),('source','*.cu'),
                            ('source','*.cpp'),('source','*.h'),('configs','*.yaml'),('tests','*.py')]:
        files.update(p for p in (ROOT/dirname).rglob(pattern) if '__pycache__' not in p.parts)
    files.update(ROOT.glob('requirements*.txt'));files.update([ROOT/'pyproject.toml',ROOT/'AGENTS.md'])
    for name in ('docs/runbooks/t11_training_handoff.md','docs/t11_ablation_protocol.md','docs/t11_evidence_map.md'):
        files.add(ROOT/name)
    data=map_sources()
    for path in data:files.add((ROOT/path).resolve())
    records={str(p.relative_to(ROOT)):{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in sorted(files)}
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    acados=ROOT/'.venv_isaaclab/acados'
    acados_commit=subprocess.check_output(['git','-C',str(acados),'rev-parse','HEAD'],text=True).strip()
    manifest={'schema_version':'upstream_transfer_v1','files':records,'python':platform.python_version(),
        'producer_packages':packages,'acados_commit':acados_commit,'training_paused':True,
        'scope':'Source plus accepted NPB development-window runtime data. No learned weights, optimizer state, generated solver binary, venv, or permission to train.'}
    json_write(target/'config/transfer_manifest.json',manifest)
    archive=target/'artifacts/source_and_runtime_data.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for relative in records:tar.add(ROOT/relative,arcname=relative,recursive=False)
        tar.add(target/'config/transfer_manifest.json',arcname='transfer_manifest.json')
    json_write(target/'run_manifest.json',{'kind':'local_transfer_package','script_sha256':file_sha256(__file__),
        'manifest_sha256':file_sha256(target/'config/transfer_manifest.json'),'archive_sha256':file_sha256(archive),
        'bytes':archive.stat().st_size,'file_count':len(records),'formal_training':False})
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(records)}))


if __name__=='__main__':main()
