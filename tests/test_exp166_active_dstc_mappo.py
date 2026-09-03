from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (  # noqa: E402
    build_multiscale_site_belief_observation,
)


def _cfg(num_envs: int = 2):
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp166_active_dstc_mappo_smoke.yaml"
    )
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = num_envs
    return cfg


def test_active_dstc_actor_interface_is_407_by_47_and_oracle_invariant() -> None:
    core = MultiRoverGatheringCore(_cfg())
    actor_before, critic_before = core.get_observations()
    assert actor_before.shape == (2, 4, 407)
    assert critic_before.shape == (2, 950)
    core.oracle_point.uniform_(-10.0, 10.0)
    core.oracle_search_feasible[:] = True
    actor_after, critic_after = core.get_observations()
    assert torch.equal(actor_before, actor_after)
    assert torch.equal(critic_before, critic_after)


def test_invalid_site_belief_has_exactly_zero_potential_channel() -> None:
    core = MultiRoverGatheringCore(_cfg(num_envs=1))
    invalid = build_multiscale_site_belief_observation(
        core.positions,
        core.yaws,
        torch.zeros_like(core.positions),
        core.cfg.terrain,
        core.terrain_runtime,
        site_valid=torch.zeros(1, 4, dtype=torch.bool),
    )
    fine = invalid[..., :189].reshape(1, 4, 7, 9, 3)
    medium = invalid[..., 189:252].reshape(1, 4, 3, 7, 3)
    coarse = invalid[..., 252:336].reshape(1, 4, 4, 7, 3)
    assert torch.equal(fine[..., 2], torch.zeros_like(fine[..., 2]))
    assert torch.equal(medium[..., 2], torch.zeros_like(medium[..., 2]))
    assert torch.equal(coarse[..., 2], torch.zeros_like(coarse[..., 2]))


def test_actor_action_is_not_replaced_by_active_dstc_or_r4() -> None:
    hold_core = MultiRoverGatheringCore(_cfg(num_envs=1))
    move_core = MultiRoverGatheringCore(_cfg(num_envs=1))
    assert torch.equal(hold_core.positions, move_core.positions)
    hold = torch.zeros(1, 4, dtype=torch.long)
    forward = torch.full((1, 4), 32, dtype=torch.long)
    hold_output = hold_core.step(hold)
    move_output = move_core.step(forward)
    hold_motion = torch.linalg.vector_norm(
        hold_output.info["positions"][..., :2] - hold_core.active_dstc_runtime.initial_positions[..., :2],
        dim=-1,
    )
    move_motion = torch.linalg.vector_norm(
        move_output.info["positions"][..., :2] - move_core.active_dstc_runtime.initial_positions[..., :2],
        dim=-1,
    )
    assert float(hold_motion.max()) == 0.0
    assert float(move_motion.mean()) > 0.0


def test_active_dstc_reward_is_in_team_total_and_reset_isolated() -> None:
    core = MultiRoverGatheringCore(_cfg())
    output = core.step(torch.full((2, 4), 32, dtype=torch.long))
    terms = output.info["reward_terms"]
    weights = core.cfg.reward_weights
    expected = (
        weights.gather * terms.gather
        + weights.oracle * terms.oracle
        + weights.energy * terms.energy
        + weights.safety * terms.safety
        + weights.terrain * terms.terrain
        + weights.flatness * terms.flatness
        + weights.motion * terms.motion
        + weights.consistency * terms.consistency
        + terms.success_hold
        + weights.terminal * terms.terminal
        + weights.active_dstc * terms.active_dstc
    )
    assert torch.allclose(terms.total, expected)
    assert torch.allclose(output.rewards, terms.total[:, None].expand(-1, 4))

    runtime = core.active_dstc_runtime
    assert runtime is not None
    other_target = runtime.target_points[1].clone()
    core.reset(torch.tensor([0]))
    assert torch.equal(runtime.target_points[1], other_target)
    assert int(runtime.scan_versions[0]) == 1
