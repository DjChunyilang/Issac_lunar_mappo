#!/usr/bin/env python3
"""Publish an immutable readiness audit; missing/failed evidence never passes."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cfg_from_experiment
from planner_evidence import file_sha256, json_write, source_snapshot

REQUIRED=('gate_b','gathering','contracts','cuda','training_entry')


def audit_signal_readiness(cfg, validation_path, *, authorized=False):
    """Bounded learning permission is distinct from full system acceptance."""
    p=Path(validation_path).resolve()
    if not p.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Validation must be owned evidence')
    validation=json.loads(p.read_text())
    import xml.etree.ElementTree as ET
    checks=[]
    for record in validation.get('junit',[]):
        target=Path(record['path']).resolve()
        if not target.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('JUnit must be owned evidence')
        cases=list(ET.parse(target).iter('testcase'))
        checks.append(file_sha256(target)==record['sha256'] and bool(cases)
            and all(not any(c.find(tag) is not None for tag in ('failure','error','skipped')) for c in cases))
    passed=bool(checks) and all(checks) and validation.get('failures')==0 and validation.get('skips')==0
    bounded=(cfg.mppi_samples==512 and cfg.mppi_iterations==2 and cfg.clock_mode=='offline_simulated_latency')
    return {'schema_version':'upstream_t11_signal_readiness_v1','scope':'signal100k',
            'signal_training_ready':authorized is True and passed and bounded,
            'formal_training_ready':False,'system_acceptance_complete':False,
            'profile_fingerprint':cfg.spec.fingerprint,'source_snapshot':source_snapshot(),
            'base_configuration':json.loads(cfg.configuration_json),
            'allowed_representations':['A1','A0','A2','A3'],'allowed_seed':23,'maximum_budget':100000,
            'checks':{'explicit_signal_authorization':authorized is True,'targeted_validation':passed,'bounded_configuration':bounded},
            'validation':{'path':str(p),'sha256':file_sha256(p)},
            'deferred_system_evidence':['complete_144','primitive47_remaining','full_gate_b'],
            'scope_note':'User-authorized learning signal check; no full-system acceptance or budget escalation.'}


def audit_readiness(cfg, evidence_paths, *, accepted_source=None):
    current=source_snapshot();checks={};evidence={};issues=[]
    # Final reconciliation is explicit and binds the exact current source and
    # prerequisite files. A diagnostic summary alone cannot grant admission.
    accepted=None
    if accepted_source:
        p=Path(accepted_source).resolve()
        if not p.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Source reconciliation must be owned evidence')
        accepted=json.loads(p.read_text())
        checks['source_reconciled']=(accepted.get('schema_version')=='upstream_t11_dependency_audit_v1'
            and accepted.get('passed') is True and accepted.get('source_snapshot')==current
            and accepted.get('profile_fingerprint')==cfg.spec.fingerprint)
    else:checks['source_reconciled']=False
    for name in REQUIRED:
        value=evidence_paths.get(name)
        if value is None:
            checks[name]=False;issues.append(name+':missing');continue
        p=Path(value).resolve()
        if not p.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Only owned prerequisite evidence is accepted')
        result=json.loads(p.read_text());sha=file_sha256(p)
        evidence[name]={'path':str(p),'sha256':sha}
        if name=='gate_b':
            ok=(result.get('gate_b_functional') is True and result.get('matrix_complete') is True
                and result.get('counts')=={'plan':7000,'repeat':630,'local':252,'batch':14})
        elif name=='gathering':
            ok=(result.get('s05_gathering_passed') is True and result.get('count')==144
                and result.get('simple_gate_18_passed') is True and result.get('matrix_complete') is True)
        else:ok=result.get('passed') is True
        checks[name]=ok
        if not ok:issues.append(name+':incomplete_or_failed')
        if accepted is not None and accepted.get('evidence',{}).get(name)!=evidence[name]:
            checks['source_reconciled']=False;issues.append(name+':reconciliation_checksum_mismatch')
    if not checks['source_reconciled']:issues.append('dependency_reconciliation:missing_or_stale')
    return {'schema_version':'upstream_t11_readiness_v1','profile_fingerprint':cfg.spec.fingerprint,
            'source_snapshot':current,'evidence':evidence,'checks':checks,'issues':issues,
            'formal_training_ready':all(checks.values()),'pause_released':False,
            'dependency_audit':None if accepted_source is None else
                {'path':str(Path(accepted_source).resolve()),'sha256':file_sha256(accepted_source)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True)
    for name in REQUIRED:parser.add_argument('--'+name.replace('_','-'))
    parser.add_argument('--dependency-audit')
    parser.add_argument('--scope',choices=('system','signal100k'),default='system')
    parser.add_argument('--authorize-signal100k',action='store_true')
    parser.add_argument('--signal-validation')
    args=parser.parse_args();cfg=cfg_from_experiment(args.config)
    if args.scope=='signal100k':
        if not args.signal_validation:parser.error('--signal-validation is required')
        result=audit_signal_readiness(cfg,args.signal_validation,authorized=args.authorize_signal100k)
    else:result=audit_readiness(cfg,{name:getattr(args,name) for name in REQUIRED},accepted_source=args.dependency_audit)
    path=ROOT/'outputs/runs/upstream_scale_t11'/args.run_id/'metrics/readiness.json'
    json_write(path,result)
    print(json.dumps({'path':str(path),'formal_training_ready':result['formal_training_ready'],
        'signal_training_ready':result.get('signal_training_ready',False),'issues':result.get('issues',[])}))


if __name__=='__main__':main()
