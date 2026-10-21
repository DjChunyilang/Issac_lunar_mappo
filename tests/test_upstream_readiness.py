"""Incomplete or changed evidence cannot release a frozen training prerequisite."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_upstream_t11_readiness as audit
from planner_evidence import file_sha256


def test_missing_evidence_remains_unready():
    result=audit.audit_readiness(SimpleNamespace(spec=SimpleNamespace(fingerprint='p')), {})
    assert not result['formal_training_ready'] and not result['pause_released']
    assert all(not value for value in result['checks'].values())


def test_complete_counts_require_exact_dependency_reconciliation(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    monkeypatch.setattr(audit,'source_snapshot',lambda:{'runtime.py':'current'})
    root=tmp_path/'outputs/runs/test';root.mkdir(parents=True)
    cfg=SimpleNamespace(spec=SimpleNamespace(fingerprint='p'))
    values={'gate_b':{'gate_b_functional':True,'matrix_complete':True,
                     'counts':{'plan':7000,'repeat':630,'local':252,'batch':14}},
            'gathering':{'s05_gathering_passed':True,'count':144,'simple_gate_18_passed':True,'matrix_complete':True},
            'contracts':{'passed':True},'cuda':{'passed':True},'training_entry':{'passed':True}}
    paths={}
    for name,value in values.items():
        p=root/(name+'.json');p.write_text(json.dumps(value));paths[name]=str(p)
    assert not audit.audit_readiness(cfg,paths)['formal_training_ready']
    reconciliation=root/'dependency_audit.json'
    reconciliation.write_text(json.dumps({'schema_version':'upstream_t11_dependency_audit_v1','passed':True,
        'source_snapshot':{'runtime.py':'current'},'profile_fingerprint':'p',
        'evidence':{k:{'path':v,'sha256':file_sha256(v)} for k,v in paths.items()}}))
    result=audit.audit_readiness(cfg,paths,accepted_source=reconciliation)
    assert result['formal_training_ready'] and not result['pause_released']
    Path(paths['gathering']).write_text(json.dumps({**values['gathering'],'count':143}))
    changed=audit.audit_readiness(cfg,paths,accepted_source=reconciliation)
    assert not changed['formal_training_ready'] and not changed['checks']['source_reconciled']
