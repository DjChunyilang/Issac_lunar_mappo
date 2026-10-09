"""Guard the Gymnasium registration that static import searches cannot see."""

from __future__ import annotations

import gymnasium as gym

import lunar_rover_tasks


def test_proxy_task_registration_entry_point() -> None:
    task_id = lunar_rover_tasks.TASK_ID
    assert task_id == "Isaac-MultiRover-Gathering-Direct-v0"
    spec = gym.spec(task_id)
    assert spec.entry_point == (
        "lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env:"
        "MultiRoverGatheringGymEnv"
    )
