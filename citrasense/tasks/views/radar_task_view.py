"""Typed accessor view for radar-modality tasks (placeholder for #307)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrasense.tasks.task import Task


class RadarTaskView:
    """Narrow view over a Task whose sensor_type is ``"radar"``."""

    __slots__ = ("_task",)

    def __init__(self, task: Task) -> None:
        if task.sensor_type != "radar":
            raise ValueError(f"Expected radar task, got sensor_type={task.sensor_type!r}")
        self._task = task

    @property
    def task(self) -> Task:
        return self._task
