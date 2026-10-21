"""Public cold recovery clears only the selected planner's hot/RNG state."""
import numpy as np
import pytest
import torch

from test_upstream_factory import cfg as cfg_fixture,terrain as terrain_fixture,inputs
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import UpstreamPlannerPool


def test_per_agent_recovery_keeps_same_environment_peer_and_other_environment_rng():
    torch.set_num_threads(1);cfg=cfg_fixture.__wrapped__();terrain=terrain_fixture.__wrapped__(cfg)
    pool=UpstreamPlannerPool(cfg);pool.plan_batch(inputs(cfg,terrain,((0,1),(0,2),(1,1))))
    before={r['identity']:r['state'] for r in pool.state_dict()['planners']}
    pool.reset_agent(0,1)
    after={r['identity']:r['state'] for r in pool.state_dict()['planners']}
    assert after[0,1]['identity'] is None
    for identity in ((0,2),(1,1)):
        assert before[identity]['identity']==after[identity]['identity']
        np.testing.assert_array_equal(before[identity]['adapter_state']['warm'],after[identity]['adapter_state']['warm'])
        torch.testing.assert_close(before[identity]['adapter_state']['generator'],after[identity]['adapter_state']['generator'],rtol=0,atol=0)
    restored=UpstreamPlannerPool(cfg);restored.load_state_dict(pool.state_dict())
    assert restored.state_dict()['planners'][0]['state']['identity'] is None


@pytest.mark.parametrize('identity',[(0,4),(-1,1),(True,1)])
def test_invalid_recovery_never_constructs_a_planner(identity):
    cfg=cfg_fixture.__wrapped__();pool=UpstreamPlannerPool(cfg)
    with pytest.raises(ValueError):pool.reset_agent(*identity)
    assert pool.state_dict()['planners']==[]
