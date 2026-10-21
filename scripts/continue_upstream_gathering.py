#!/usr/bin/env python3
"""Resume the 144-case matrix, prioritizing all 18 mandatory simple cases.

Workers claim disjoint scene/seed-pair jobs. Run from an archived producer; completed results
remain immutable and interruptions resume through execute_pair's public state.
This command never starts learning or changes admission requirements.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import time

from _common import ROOT
from evaluate_upstream_gathering import FAMILIES, execute_pair
from planner_evidence import EvidenceRun, json_write


def jobs(variant):
    if variant not in ('nmpc', 'mppi512'):
        raise ValueError('Unknown gathering planner')
    required = [(f, seeds, variant, True, 360)
                for f in FAMILIES[:3] for seeds in ((23, 31), (47,))]
    remaining = [(f, seeds, variant, safety, seconds)
                 for f in FAMILIES for seconds in (360, 180)
                 for safety in (True, False) for seeds in ((23, 31), (47,))
                 if not (f in FAMILIES[:3] and safety and seconds == 360)]
    return required + remaining


def summarize(rows):
    expected = {(f, s, v, a, d) for f in FAMILIES for s in (23, 31, 47)
                for v in ('nmpc', 'mppi512') for a in (False, True) for d in (180, 360)}
    keys = [(r['family'], r['seed'], r['variant'], r['safety'], r['deadline']) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Duplicate or unexpected episode identity')
    simple = [r for r in rows if r['family'] in FAMILIES[:3] and r['safety'] and r['deadline'] == 360]
    simple_passed = len(simple) == 18 and all(r['successful'] and not r['violations'] for r in simple)
    complete = set(keys) == expected
    return {'count': len(rows), 'expected': 144, 'matrix_complete': complete,
            'simple_completed': len(simple), 'simple_expected': 18,
            'simple_gate_18_passed': simple_passed,
            'successful': sum(r['successful'] for r in rows),
            'failure_categories': dict(Counter(r['failure_category'] for r in rows)),
            'violations': sum(len(r['violations']) for r in rows),
            'unverified_execution': sum(r.get('counts', {}).get('unverified_execution', 0) for r in rows),
            'missing': [list(k) for k in sorted(expected - set(keys))],
            's05_gathering_passed': complete and simple_passed,
            'formal_training_ready': False}


def report(run, label):
    paths = sorted((run.path / 'metrics/episodes').glob('*.json'))
    result = summarize([json.loads(p.read_text()) for p in paths])
    result['generated_unix'] = time.time()
    result['producer_manifest'] = str(run.path / 'run_manifest.json')
    json_write(run.path / 'metrics' / f'coverage_{len(paths):03d}_{label}_{time.time_ns()}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', choices=('nmpc', 'mppi512'))
    parser.add_argument('--worker-id', type=int, default=0,
                        help='Unique per-planner worker identity; jobs use process locks')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'outputs/runs/upstream_scale_s05' / args.run_id
    manifest = json.loads((path / 'run_manifest.json').read_text())
    run = EvidenceRun(path, args.config, resume=True, settings=manifest['identity']['settings'])
    if args.report_only:
        print(json.dumps(report(run, 'manual')))
        return
    if args.variant is None:
        parser.error('--variant is required for a worker')
    if args.worker_id < 0:
        parser.error('--worker-id must be nonnegative')
    import torch
    torch.set_num_threads(1)
    label = f'{args.variant}_worker{args.worker_id}'
    with (run.path / 'logs' / f'{label}.lock').open('a') as lock:
        # Process lifetime lock prevents duplicate execution on resume.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(f'{os.getpid()} {time.time()}\n')
        lock.flush()
        for family, seeds, variant, safety, seconds in jobs(args.variant):
            tag = f'{family}_{"_".join(map(str, seeds))}_{variant}_safe{int(safety)}_t{seconds}'
            with (run.path / 'logs' / f'job_{tag}.lock').open('a') as job_lock:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                print('START', family, seeds, variant, safety, seconds, flush=True)
                execute_pair(run, family, list(seeds), variant, safety, seconds)
                result = report(run, label)
                print('COVERAGE', result['count'], 'SIMPLE', result['simple_completed'], flush=True)
        json_write(run.path / 'logs' / f'{label}_complete_{time.time_ns()}.json',
                   {'returncode': 0, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
