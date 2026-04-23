"""Typed accessor view for RF/antenna-modality tasks (placeholder for #307)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrasense.tasks.task import Task


class RfTaskView:
    """Narrow view over a Task whose sensor_type is ``"rf"``."""

    __slots__ = ("_task",)

    def __init__(self, task: Task) -> None:
        if task.sensor_type != "rf":
            raise ValueError(f"Expected rf task, got sensor_type={task.sensor_type!r}")
        self._task = task

    @property
    def task(self) -> Task:
        return self._task

    @property
    def antenna_id(self) -> str:
        return self._task.sensor_id
