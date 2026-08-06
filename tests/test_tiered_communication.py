from __future__ import annotations

import math

import pytest
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.communication import (
    TieredCommunicationCache,
    build_cached_aggregation_features,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg


def _state(distance: float) -> tuple[torch.Tensor, ...]:
    positions = torch.tensor([[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]])
    velocities = torch.tensor([[[0.0, 0.0], [1.5, -0.5]]])
    yaws = torch.tensor([[0.0, 0.5]])
    terrain = torch.tensor(
        [[
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
        ]]
    )
    return positions, velocities, yaws, terrain


def _cache() -> TieredCommunicationCache:
    return TieredCommunicationCache(
        num_envs=1,
        n_agents=2,
        max_neighbors=1,
        device="cpu",
    )


@pytest.mark.parametrize("distance", [11.9, 12.0])
def test_full_message_at_or_inside_radius(distance: float) -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(distance))
    message = cache.snapshot().features.reshape(1, 2, 1, 12)[0, 0, 0]
    assert cache.full[0, 0, 1]
    assert message[11] == pytest.approx(1.0)
    assert torch.any(message[2:4] != 0.0)
    assert torch.allclose(message[6:11], _state(distance)[3][0, 1])


def test_sparse_message_outside_radius_hides_velocity_and_terrain() -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(12.1))
    message = cache.snapshot().features.reshape(1, 2, 1, 12)[0, 0, 0]
    assert not cache.full[0, 0, 1]
    assert message[11] == pytest.approx(0.5)
    assert torch.count_nonzero(message[2:4]) == 0
    assert torch.count_nonzero(message[6:11]) == 0


def test_maximum_map_distance_uses_four_second_period() -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(25.0 * math.sqrt(2.0)))
    assert cache.update_period[0, 0, 1] == pytest.approx(4.0)


def test_sparse_cache_does_not_leak_untransmitted_pose_changes() -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(12.1))
    before = cache.snapshot()
    positions, velocities, yaws, terrain = _state(20.0)
    cache.advance(
        dt=0.2,
        positions=positions,
        velocities_xy=velocities,
        yaws=yaws,
        terrain_summary=terrain,
    )
    after = cache.snapshot()
    before_message = before.features.reshape(1, 2, 1, 12)[0, 0, 0]
    after_message = after.features.reshape(1, 2, 1, 12)[0, 0, 0]
    assert torch.allclose(before_message[:11], after_message[:11])
    assert after_message[11] < before_message[11]


def test_repeated_snapshot_is_pure_and_entering_range_restores_full_message() -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(12.1))
    first = cache.snapshot()
    second = cache.snapshot()
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.ages, second.ages)
    positions, velocities, yaws, terrain = _state(11.9)
    cache.advance(
        dt=0.2,
        positions=positions,
        velocities_xy=velocities,
        yaws=yaws,
        terrain_summary=terrain,
    )
    restored = cache.snapshot().features.reshape(1, 2, 1, 12)[0, 0, 0]
    assert cache.full[0, 0, 1]
    assert restored[11] == pytest.approx(1.0)
    assert torch.any(restored[2:4] != 0.0)


def test_leaving_range_immediately_clears_restricted_fields() -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(11.9))
    positions, velocities, yaws, terrain = _state(12.1)
    cache.advance(
        dt=0.2,
        positions=positions,
        velocities_xy=velocities,
        yaws=yaws,
        terrain_summary=terrain,
    )
    message = cache.snapshot().features.reshape(1, 2, 1, 12)[0, 0, 0]
    assert torch.count_nonzero(message[2:4]) == 0
    assert torch.count_nonzero(message[6:11]) == 0


def test_cached_aggregation_uses_only_snapshot_values() -> None:
    cache = _cache()
    cache.reset(torch.tensor([0]), *_state(12.1))
    snapshot = cache.snapshot()
    aggregation = build_cached_aggregation_features(
        snapshot,
        map_max_distance_m=25.0 * math.sqrt(2.0),
    )
    assert aggregation.shape == (1, 2, 5)
    assert aggregation[0, 0, 0] == pytest.approx(1.0)
    assert aggregation[0, 0, 3] == pytest.approx(0.5)


def test_actor_observation_does_not_read_uncommunicated_neighbor_or_oracle_state() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.observation.schema_version = "ego_v8_decentralized_tiered"
    cfg.observation.communication_radius = 12.0
    cfg.safety.world_xy_limit = 12.5
    env = MultiRoverGatheringCore(cfg)
    env.positions[0, 0, :2] = torch.tensor([0.0, 0.0])
    env.positions[0, 1, :2] = torch.tensor([12.1, 0.0])
    env._reset_communication(torch.tensor([0]))
    original, _ = env.get_observations()
    original_receiver = original[0, 0].clone()

    env.positions[0, 1, :2] = torch.tensor([20.0, 4.0])
    env.velocities_xy[0, 1] = torch.tensor([7.0, -3.0])
    env.oracle_point.add_(5.0)
    env.gather_slot_points.add_(7.0)
    unchanged, _ = env.get_observations()
    assert torch.equal(original_receiver, unchanged[0, 0])
