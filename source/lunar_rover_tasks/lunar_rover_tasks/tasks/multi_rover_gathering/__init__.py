"""First-stage multi-rover gathering task registration."""

from __future__ import annotations

TASK_ID = "Isaac-MultiRover-Gathering-Direct-v0"


def register_task() -> None:
    """Register the first-stage proxy task with gymnasium if available."""
    try:
        import gymnasium as gym
        from gymnasium.envs.registration import registry
    except Exception:
        return

    if TASK_ID in registry:
        return

    gym.register(
        id=TASK_ID,
        entry_point=(
            "lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env:"
            "MultiRoverGatheringGymEnv"
        ),
    )


register_task()

__all__ = ["TASK_ID", "register_task"]

