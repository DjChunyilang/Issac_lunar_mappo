"""Lunar rover task extension package."""

from lunar_rover_tasks.tasks.multi_rover_gathering import TASK_ID, register_task


register_task()

__all__ = ["TASK_ID", "register_task"]

