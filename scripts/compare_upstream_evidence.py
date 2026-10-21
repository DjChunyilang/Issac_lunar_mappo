#!/usr/bin/env python3
"""Inventory source differences; never certify cross-version evidence reuse."""
import argparse
from collections import Counter
import json
from pathlib import Path

from _common import ROOT
from planner_evidence import source_snapshot,json_write,file_sha256


def category(path):
    name=Path(path).name
    if name in {'upstream_training.py','upstream_ablation_learning.py','action_ablation.py','upstream_action_bridge.py','run_upstream_t11.py'}:
        return 'learning_or_action_path'
    if path.startswith('source/'):
        return 'shared_runtime_review_required'
    if name in {'analyze_upstream_t11.py','profile_upstream_t11.py','compare_upstream_evidence.py','package_upstream_transfer.py','verify_upstream_transfer.py','report_upstream_preparation.py'}:
        return 'offline_tool_not_runtime'
    return 'driver_or_other_review_required'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();target=ROOT/'outputs/runs/upstream_scale_t11'/a.run_id;current=source_snapshot();rows=[]
    for name in a.runs:
        run=Path(name).resolve()
        if not run.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Expected owned evidence')
        manifest=run/'run_manifest.json';raw=json.loads(manifest.read_text());identity=raw.get('identity',raw)
        old=identity['source_snapshot'];changes=[]
        for path in sorted(old.keys()|current.keys()):
            if old.get(path)!=current.get(path):changes.append({'path':path,'producer_sha256':old.get(path),
                'current_sha256':current.get(path),'category':category(path)})
        rows.append({'run':str(run),'manifest_sha256':file_sha256(manifest),'changes':changes,
            'category_counts':dict(Counter(c['category'] for c in changes)),
            'profile_fingerprint':identity.get('contracts',{}).get('profile_fingerprint'),
            'scope':'File inventory only. Reuse requires producer dependency review; added offline scripts alone do not change vehicle dynamics.'})
    json_write(target/'metrics/source_comparison.json',{'schema_version':'upstream_source_inventory_v1',
        'current_source_snapshot':current,'runs':rows,'dependency_audit_passed':False,'formal_training_ready':False})
    json_write(target/'run_manifest.json',{'kind':'read_only_dependency_inventory','script_sha256':file_sha256(__file__),
        'sources':{r['run']:r['manifest_sha256'] for r in rows}})
    for r in rows:print(Path(r['run']).name,r['category_counts'])


if __name__=='__main__':main()
