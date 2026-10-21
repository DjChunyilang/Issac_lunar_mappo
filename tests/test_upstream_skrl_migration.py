"""Actual skrl ownership, probability/update equivalence and boundary recovery."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from skrl.multi_agents.torch.mappo import MAPPO

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_upstream_t11 import make_collector
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer,TrainingOptions
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer

BASE='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'


def test_skrl_real_update_migration_and_resume(tmp_path):
    torch.set_num_threads(1)
    collector=make_collector(BASE,'A1',23,samples=16,num_envs=1,episode_seconds=2.)
    options=TrainingOptions(rollout_macro_steps=2,epochs=2)
    legacy=PreparedUpstreamTrainer(collector,options)
    batch,_,_=legacy.collect()
    before=legacy.state_dict()
    reference=legacy.diagnostic_update(batch)
    expected=deepcopy(collector.policy.state_dict())
    migrated=SkrlUpstreamTrainer(collector,options)
    assert isinstance(migrated.agent,MAPPO)
    migrated.migrate_legacy_state(before)
    assert migrated.optimizer is migrated.agent.optimizers['team']
    assert migrated.agent.policies['team'].policy is collector.policy
    rng=torch.get_rng_state()
    direct=collector.policy.sample(batch['obs'][0],batch['hidden'][0])
    torch.set_rng_state(rng)
    sampled=migrated.agent.sample(batch['obs'][0],batch['state'][0],batch['hidden'][0])
    assert torch.equal(direct.log_prob,sampled.log_prob)
    assert torch.equal(direct.action.hold,sampled.action.hold)
    assert torch.equal(direct.action.continuous,sampled.action.continuous)
    torch.set_rng_state(rng)  # Entropy uses Monte Carlo samples, just like the legacy objective.
    result=migrated.diagnostic_update(batch)
    assert result==reference
    for key,value in collector.policy.state_dict().items():assert torch.equal(value,expected[key]),key
    assert len(migrated.agent.memories['team'])==2
    assert torch.equal(migrated.agent.memories['team'].get_tensor_by_name('duration'),batch['duration'])
    boundary=migrated.state_dict()
    migrated.agent.save(str(tmp_path/'agent.pt'))
    assert (tmp_path/'agent.pt').is_file()
    again=migrated.diagnostic_update(batch)
    expected=deepcopy(collector.policy.state_dict())
    migrated.load_state_dict(boundary)
    assert migrated.diagnostic_update(batch)==again
    for key,value in collector.policy.state_dict().items():assert torch.equal(value,expected[key]),key
    with pytest.raises(ValueError,match='limited'):migrated.diagnostic_update(batch)
    with pytest.raises(ValueError,match='identity'):migrated.load_state_dict(before)

    migrated.migrate_legacy_state(before)
    assert len(migrated.optimizer.state)==0
    assert migrated.updates==0 and migrated.diagnostic_batches==0


@pytest.mark.parametrize('representation',['A0','A1','A2','A3'])
def test_skrl_samples_original_four_action_contracts(representation):
    from types import SimpleNamespace
    from _common import cfg_from_experiment
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import RecurrentTeamMAPPO
    torch.set_num_threads(1);torch.manual_seed(73)
    spec=cfg_from_experiment(BASE).spec
    policy=build_ablation_policies(991,spec)[representation]
    agent=RecurrentTeamMAPPO(SimpleNamespace(policy=policy,device=torch.device('cpu'),env=SimpleNamespace(e=2)),TrainingOptions(rollout_macro_steps=2))
    obs=torch.zeros(2,4,policy.actor_dim);state=torch.zeros(2,991);hidden=torch.zeros(2,4,128)
    rng=torch.get_rng_state();reference=policy.sample(obs,hidden)
    torch.set_rng_state(rng);actual=agent.sample(obs,state,hidden)
    assert torch.equal(actual.log_prob,reference.log_prob)
    assert torch.equal(actual.latent,reference.latent)
    assert torch.equal(actual.action.hold,reference.action.hold)
    assert torch.equal(actual.action.primitive_id,reference.action.primitive_id)
    assert torch.equal(actual.action.continuous,reference.action.continuous)
    actual.action.validate(policy.contract,(2,4))
