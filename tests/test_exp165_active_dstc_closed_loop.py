from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.decentralized_primitive_optimizer import (  # noqa: E402
    select_decentralized_primitives,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)


def _core(num_envs: int = 2):
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp165_active_dstc_closed_loop.yaml"
    )
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = num_envs
    return cfg, MultiRoverGatheringCore(cfg)


def test_r4_returns_finite_47_primitive_choices_and_terminal_hold() -> None:
    cfg, core = _core()
    centers = core.positions[..., :2].mean(dim=1)
    previous = torch.zeros(core.num_envs, core.n_agents, dtype=torch.long)
    result = select_decentralized_primitives(
        core.positions,
        core.yaws,
        centers,
        previous,
        torch.ones(core.num_envs, dtype=torch.bool),
        cfg,
        core.terrain_runtime,
    )
    assert result.actions.shape == (core.num_envs, core.n_agents)
    assert int(result.actions.min()) >= 0
    assert int(result.actions.max()) <= 46
    assert torch.isfinite(result.selected_cost).all()

    held = select_decentralized_primitives(
        core.positions,
        core.yaws,
        centers,
        previous,
        torch.ones(core.num_envs, dtype=torch.bool),
        cfg,
        core.terrain_runtime,
        terminal_hold=torch.ones(core.num_envs, dtype=torch.bool),
    )
    assert torch.equal(held.actions, torch.zeros_like(held.actions))


def test_r4_agent_zero_is_invariant_to_non_neighbour_state() -> None:
    cfg, core = _core(num_envs=1)
    positions = core.positions.clone()
    positions[0, 0, :2] = torch.tensor((0.0, 0.0))
    positions[0, 1:, :2] = torch.tensor(
        ((20.0, 0.0), (0.0, 20.0), (20.0, 20.0))
    )
    centers = torch.zeros(1, 2)
    previous = torch.zeros(1, core.n_agents, dtype=torch.long)
    first = select_decentralized_primitives(
        positions,
        core.yaws,
        centers,
        previous,
        torch.ones(1, dtype=torch.bool),
        cfg,
        core.terrain_runtime,
    )
    changed = positions.clone()
    changed[0, 1:, :2] += 5.0
    second = select_decentralized_primitives(
        changed,
        core.yaws,
        centers,
        previous,
        torch.ones(1, dtype=torch.bool),
        cfg,
        core.terrain_runtime,
    )
    assert int(first.actions[0, 0]) == int(second.actions[0, 0])
    assert float(first.selected_cost[0, 0]) == float(second.selected_cost[0, 0])
