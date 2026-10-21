"""Sequential owner-local checkpoints; publish a learner backup first."""
import json
import os
from pathlib import Path
import tempfile

import torch

from planner_evidence import file_sha256

OWNER_FORMAT = 't11_owner_shards_v1'
LEARNER_FORMAT = 't11_learner_only_v1'


def _publish(path, writer):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink()
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_tensor_file(path, state):
    _publish(path, lambda stream: torch.save(state, stream))
    return {'file': Path(path).name, 'sha256': file_sha256(path),
            'bytes': Path(path).stat().st_size}


def _save_header(path, state, *, scope):
    metadata = save_tensor_file(path, state)
    metadata.update(team_transitions=state['trainer']['collector']['transitions'],
                    scope=scope, checkpoint_format=state.get('checkpoint_format', 'monolithic'),
                    system_acceptance_complete=False)
    collector=state['trainer']['collector']
    if 'learning_time' in collector['metadata']:
        metadata.update(actual_base_steps=collector['base_steps'],policy_decisions=collector['decisions'],
                        learning_time=collector['metadata']['learning_time'])
    _publish(Path(path).with_suffix('.json'),
             lambda stream: stream.write(json.dumps(metadata, indent=2).encode()))


def save_checkpoint(trainer, identity, updates, target, *, scope):
    """Call only with collection/PPO stopped at an update boundary."""
    target = Path(target)
    collector = trainer.collector
    central = collector.state_dict(include_environment=False)
    state = {'identity': identity, 'trainer': trainer.state_dict(collector_state=central),
             'session_updates': updates}
    learner = target.with_name(target.stem + '.learner.pt')
    _save_header(learner, {**state, 'checkpoint_format': LEARNER_FORMAT}, scope='learner_only')
    if hasattr(collector, 'save_owner_shards'):
        directory = target.with_suffix('.owners')
        directory.mkdir()
        shards = collector.save_owner_shards(directory)
        state['checkpoint_format'] = OWNER_FORMAT
        state['owner_shards'] = [dict(shard, file=directory.name + '/' + shard['file']) for shard in shards]
    else:
        state['trainer']['collector'] = collector.state_dict()
    _save_header(target, state, scope=scope)
    return target


def read_checkpoint(path, *, learner_only=False):
    """Verify all shards before restoration; legacy monolithic files still load."""
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix('.json').read_text())
    if file_sha256(path) != metadata['sha256']:
        raise ValueError('Training checkpoint checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    kind = state.get('checkpoint_format')
    if kind == LEARNER_FORMAT and not learner_only:
        raise ValueError('Learner-only checkpoint requires explicit learner import; no environment resume')
    if kind == OWNER_FORMAT and not learner_only:
        shards = state['owner_shards']
        if len(shards) != state['trainer']['collector']['metadata']['owners']:
            raise ValueError('Checkpoint owner shard count mismatch')
        resolved = []
        for owner, shard in enumerate(shards):
            file = (path.parent / shard['file']).resolve()
            if shard['owner'] != owner or not file.is_relative_to(path.parent):
                raise ValueError('Checkpoint owner shard identity/path mismatch')
            if file.stat().st_size != shard['bytes'] or file_sha256(file) != shard['sha256']:
                raise ValueError('Checkpoint owner shard checksum mismatch')
            resolved.append({**shard, 'file': str(file)})
        state['trainer']['collector']['owner_shards'] = resolved
    return state
