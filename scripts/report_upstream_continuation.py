#!/usr/bin/env python3
"""Append-only progress and failure symptoms, with immutable source links.

Failure categories here are observable symptoms, not inferred root causes.
No completion count or report from this tool grants training admission.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from _common import ROOT
from continue_upstream_gathering import summarize
from planner_evidence import file_sha256, json_write


def local_symptom(row, capture_radius):
    if row['completed']:
        return 'completed'
    if row['violations']:
        return 'execution_violation'
    if row['terminal_reason'] == 'environment_invalid':
        return 'environment_invalid'
    if row['goal_error'] <= capture_radius + 1e-10:
        return 'within_capture_without_completed_stop_and_dwell'
    return 'outside_capture_at_end'


def snapshot(benchmark, gathering):
    expected = {'plan': 7000, 'repeat': 630, 'local': 252, 'batch': 14}
    counts = {phase: len(list((benchmark / 'metrics' / phase).glob('*.json'))) for phase in expected}
    profile = json.loads((benchmark / 'config/design_profile.json').read_text())
    # The actual parsed runtime spec is recorded independently from result rows.
    effective = json.loads((benchmark / 'config/effective_env_config.json').read_text())
    capture = effective['spec']['task']['capture_radius']
    rows = []
    for path in sorted((benchmark / 'metrics/local').glob('*.json')):
        row = json.loads(path.read_text())
        keys = ('case_id', 'variant', 'safety', 'completed', 'terminal_reason', 'goal_error',
                'dwell_steps', 'simulation_seconds', 'recovery_count', 'violations')
        rows.append({**{k: row[k] for k in keys}, 'symptom': local_symptom(row, capture),
                     'path': str(path), 'sha256': file_sha256(path)})
    episodes = []
    for path in sorted((gathering / 'metrics/episodes').glob('*.json')):
        row = json.loads(path.read_text())
        episodes.append({**row, 'path': str(path), 'sha256': file_sha256(path)})
    by_variant = {}
    for row in rows:
        item = by_variant.setdefault(row['variant'], {'completed_cases': 0, 'goal_successes': 0, 'expected': 36})
        item['completed_cases'] += 1
        item['goal_successes'] += int(row['completed'])
    return {'schema_version': 'upstream_continuation_progress_v1', 'generated_unix': time.time(),
            'sources': [{'path': str(p / 'run_manifest.json'), 'sha256': file_sha256(p / 'run_manifest.json')}
                        for p in (benchmark, gathering)],
            'benchmark': {'counts': counts, 'expected': expected,
                          'remaining': {k: expected[k] - v for k, v in counts.items()},
                          'by_variant': by_variant, 'local_successes': sum(r['completed'] for r in rows),
                          'failure_symptoms': dict(Counter(r['symptom'] for r in rows if not r['completed'])),
                          'capture_radius': capture, 'profile_present': bool(profile),
                          'rows': rows},
            'gathering': summarize(episodes),
            'episode_sources': [{k: r[k] for k in ('family', 'seed', 'variant', 'safety', 'deadline',
                                                 'successful', 'simulation_seconds', 'path', 'sha256')}
                                for r in episodes],
            'scope': 'Progress only. Stop/terrain/waiting causes require trace/state diagnosis. '
                     'Original and v5 benchmark dependencies remain distinct.',
            'formal_training_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-run', required=True)
    parser.add_argument('--gathering-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    base = ROOT / 'outputs/runs'
    benchmark = base / 'upstream_scale_s05' / args.benchmark_run
    gathering = base / 'upstream_scale_s05' / args.gathering_run
    target = base / 'upstream_scale_t11' / args.run_id
    if not (target / 'run_manifest.json').exists():
        json_write(target / 'run_manifest.json', {'kind': 'read_only_continuation_report',
                   'script_sha256': file_sha256(__file__), 'benchmark': str(benchmark), 'gathering': str(gathering)})
    previous = None
    while True:
        result = snapshot(benchmark, gathering)
        current = (tuple(result['benchmark']['counts'].values()), result['gathering']['count'])
        if current != previous:
            path = target / 'metrics' / f'progress_{time.time_ns()}.json'
            json_write(path, result)
            print(json.dumps({'path': str(path), 'counts': result['benchmark']['counts'],
                              'episodes': result['gathering']['count'],
                              'simple': result['gathering']['simple_completed']}), flush=True)
            previous = current
        if not args.watch or (result['benchmark']['counts'] == result['benchmark']['expected']
                              and result['gathering']['matrix_complete']):
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
