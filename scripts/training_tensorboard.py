"""TensorBoard projection of immutable training JSON; no environment or RNG use.

Called synchronously by the learner at collection/update/evaluation boundaries.
JSON remains authoritative. Reopening a run rebuilds its curves and purges the
previous projection (not the files), including recovery after a writer crash.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlencode

from torch.utils.tensorboard import SummaryWriter

TENSORBOARD_METADATA = {'directory': 'tensorboard', 'step_unit': 'actual_training_base_environment_steps',
                        'source': 'metrics JSON', 'required': True}

# The native Time Series Pinned section supports at most ten scalar cards.
# These are aliases of existing values, not new estimates or pooled cohorts.
PINNED_SCALARS = {
    'episode/raw_return_mean': 'Main/Return',
    'episode/termination_fraction/success': 'Main/Success',
    'train/behavior/parking_hold_fraction': 'Main/HOLD_park',
    'train/behavior/approach_hold_fraction': 'Main/HOLD_move',
    'ppo/diagnostics/explained_variance_before_update': 'Main/Value_EV',
    'ppo/epoch_last/approx_kl': 'Main/KL',
    'eval/short/parking_sample/success_rate': 'Main/Park_S',
    'eval/short/parking_mean/success_rate': 'Main/Park_M',
    'eval/short/approach_sample/dmax_reduction/mean': 'Main/Approach_S',
    'eval/short/approach_mean/dmax_reduction/mean': 'Main/Approach_M',
}


FULL_FAR_PINNED_SCALARS = {
    'eval/full/far_sample/success_rate': 'Main/SR_S',
    'eval/full/far_mean/success_rate': 'Main/SR_M',
    'episode/termination_fraction/success': 'Main/Train_SR',
    'episode/raw_return_mean': 'Main/Return',
    'ppo/diagnostics/explained_variance_before_update': 'Main/Value_EV',
    'ppo/epoch_last/approx_kl': 'Main/KL',
    'ppo/epoch_last/clip_fraction': 'Main/Clip',
    'ppo/epoch_last/entropy': 'Main/Entropy',
    'train/hold_fraction': 'Main/HOLD',
    'eval/full/far_sample/dmax_reduction/mean': 'Main/Dmax_S',
}


BOUNDED_FAR_PINNED_SCALARS = {
    'eval/short/far_sample/mean_observed_seconds': 'Main/ObservedTime_S',
    'eval/short/far_mean/mean_observed_seconds': 'Main/ObservedTime_M',
    'eval/short/far_sample/window_failure_fraction': 'Main/WindowFailure_S',
    'eval/short/far_mean/window_failure_fraction': 'Main/WindowFailure_M',
    'eval/short/far_sample/dmax_reduction/mean': 'Main/Dmax_S',
    'eval/short/far_mean/dmax_reduction/mean': 'Main/Dmax_M',
    'episode/raw_return_mean': 'Main/Return',
    'train/hold_fraction': 'Main/HOLD',
    'ppo/diagnostics/explained_variance_before_update': 'Main/Value_EV',
    'ppo/epoch_last/approx_kl': 'Main/KL',
}


def pinned_url(base_url='http://127.0.0.1:6006', *, scalars=None):
    """Native TensorBoard deep link; pin preferences belong to the browser."""
    cards = [{'plugin': 'scalars', 'tag': tag} for tag in (PINNED_SCALARS if scalars is None else scalars).values()]
    return base_url.rstrip('/') + '/?' + urlencode({
        'pinnedCards': json.dumps(cards, separators=(',', ':')),
        'tagFilter': '^Main/',
    }) + '#timeseries'


class TrainingTensorBoard:
    def __init__(self, run_dir, *, subdir='tensorboard', pinned='diagnostic'):
        if pinned not in ('diagnostic', 'full_far', 'bounded_far'):
            raise ValueError('Unknown TensorBoard pinned view')
        self.pinned_scalars = (BOUNDED_FAR_PINNED_SCALARS if pinned == 'bounded_far'
                              else FULL_FAR_PINNED_SCALARS if pinned == 'full_far' else PINNED_SCALARS)
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / subdir
        self.writer = SummaryWriter(str(self.log_dir), purge_step=0, flush_secs=10)
        self.seen = set()
        self.sources = {}
        self.writer.add_text('run/Pinned', '[Open the 10 key charts](' + pinned_url(scalars=self.pinned_scalars) + ')', 0)
        (self.log_dir / 'pinned_url.txt').write_text(pinned_url(scalars=self.pinned_scalars) + '\n')
        self.writer.add_text('run/units',
            'T11 x axis: actual training base environment steps (0.2 s per environment; '
            'not multiplied by rovers). Evaluation steps are separate. '
            'Episode statistics cover completed episodes in each batch; absent episodes '
            'produce no return/success point. PPO epoch mean and last are separate. '
            'Arrays have /mean and /case_N tags. JSON is authoritative.', 0)
        self.writer.flush()
        try:
            self.sync()
        except BaseException:
            self.close()
            raise

    def scalars(self, prefix, value, step):
        if isinstance(value, dict):
            for key, item in value.items():
                self.scalars(f'{prefix}/{key}', item, step)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(x, (int, float)) for x in value):
                self.scalars(prefix + '/mean', sum(value) / len(value), step)
            for i, item in enumerate(value):
                # Lists of episode/history dictionaries require explicit aggregation.
                if not isinstance(item, dict):
                    self.scalars(f'{prefix}/case_{i}', item, step)
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite TensorBoard metric: {prefix} at {step}')
            self.writer.add_scalar(prefix, value, step)
            if prefix in self.pinned_scalars:
                self.writer.add_scalar(self.pinned_scalars[prefix], value, step)

    def update(self, row):
        step = row['actual_base_steps']
        self.scalars('train', {k: v for k, v in row.items() if k not in ('learning', 'episodes')}, step)
        learning = row.get('learning', {})
        self.scalars('ppo', {k: v for k, v in learning.items() if k != 'epochs'}, step)
        epochs = learning.get('epochs', [])
        if epochs:
            for key in epochs[0]:
                values = [e[key] for e in epochs if isinstance(e.get(key), (int, float))]
                if values:
                    self.scalars('ppo/epoch_mean/' + key, sum(values) / len(values), step)
            self.scalars('ppo/epoch_last', epochs[-1], step)
        episodes = row.get('episodes', [])
        self.scalars('episode/completed_count', len(episodes), step)
        if episodes:
            for key in ('raw_return', 'discounted_return', 'steps', 'hold_count'):
                values = [e[key] for e in episodes if key in e]
                if values:
                    self.scalars('episode/' + key + '_mean', sum(values) / len(values), step)
            for reason in sorted({'success', 'task_deadline'} | {e['terminal_reason'] for e in episodes}):
                self.scalars('episode/termination_fraction/' + reason,
                             sum(e.get('terminal_reason') == reason for e in episodes) / len(episodes), step)
            components = sorted({k for e in episodes for k in e.get('reward_components', {})})
            for key in components:
                values = [e['reward_components'][key] for e in episodes if key in e.get('reward_components', {})]
                self.scalars('episode/reward_components/' + key + '_mean', sum(values) / len(values), step)

    def evaluation(self, row, name):
        step = row['training_base_steps']
        # The exploration probe evaluates different policies at the same step.
        variant = '/no_prior' if name.endswith('_no_prior') else '/prior' if name.endswith('_prior') else ''
        for panel, result in row.get('results', {}).items():
            scope = 'full' if result.get('full_episode_evaluation') else 'short'
            prefix = f'eval{variant}/{scope}/{panel}'
            self.scalars(prefix, {k: v for k, v in result.items() if k not in ('episodes', 'trace')}, step)
            cases = len(result.get('initial', {}).get('dmax', []))
            if cases:
                self.scalars(prefix + '/case_count', cases, step)
                self.scalars(prefix + '/success_rate', result['successes'] / cases, step)
        self.scalars('evaluation_budget' + variant, {k: row[k] for k in
                     ('actual_base_steps', 'reserved_base_steps') if k in row}, step)
        if 'cases' in row:  # Historical T11 independent18 protocol.
            for i, result in enumerate(row['cases']):
                self.scalars(f'eval/independent/case_{i}', result, step)

    def sync(self):
        """Project only newly completed JSON files; flush for live monitoring."""
        for path in (self.run_dir / 'run_manifest.json', self.run_dir / 'config/effective_configuration.json'):
            if path.exists() and path not in self.seen:
                self.writer.add_text('run/' + path.stem, '```json\n' + path.read_text() + '\n```', 0)
                self.seen.add(path)
        pending = []
        for pattern in ('update_*.json', 'evaluation_*.json', 'collection_*.json', '*summary*.json'):
            for path in (self.run_dir / 'metrics').glob(pattern):
                if path in self.seen:
                    continue
                row = json.loads(path.read_text())
                self.seen.add(path)
                if path.stem.startswith('evaluation_'):
                    if not ('results' in row or 'cases' in row) or 'training_base_steps' not in row:
                        continue
                    step = row['training_base_steps']
                else:
                    if 'actual_base_steps' not in row:
                        continue  # Historical transition-count-only records have no physical x coordinate.
                    step = row['actual_base_steps']
                if type(step) is not int or step < 0:
                    raise ValueError('TensorBoard requires a nonnegative integer physical-step count')
                pending.append((step, path, row))
        for step, path, row in sorted(pending):
            if path.stem.startswith('update_'):
                self.update(row)
            elif path.stem.startswith('evaluation_'):
                self.evaluation(row, path.stem)
            elif path.stem.startswith('collection_'):
                self.scalars('sampling', {k: row[k] for k in
                             ('actual_base_steps', 'policy_decisions', 'valid_transitions', 'hold_fraction', 'reward_mean', 'base_steps_per_second') if k in row}, step)
            else:
                self.scalars('summary', row, step)
                self.writer.add_text('run/stop_reason', str(row.get('stop_reason', '')), step)
            self.sources[str(path.relative_to(self.run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description='Rebuild T11 TensorBoard curves from JSON without learning')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_dir / 'run_manifest.json').is_file():
        raise ValueError('Expected an existing run manifest')
    # Keep offline replay separate from live events and immutable original evidence.
    with TrainingTensorBoard(args.run_dir, subdir='tensorboard_replay') as log:
        log.sync()
        provenance = {'command': ['training_tensorboard.py', '--run-dir', str(args.run_dir)],
                      'source_sha256': log.sources, 'environment_steps_added': 0, 'optimizer_updates_added': 0}
        (log.log_dir / 'replay_sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(json.dumps({'tensorboard_dir': str(log.log_dir), 'source_files': len(log.sources)}))


if __name__ == '__main__':
    main()
