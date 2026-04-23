"""Typed accessor view for telescope-modality tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrasense.tasks.task import Task


class TelescopeTaskView:
    """Narrow, typed view over a Task whose sensor_type is ``"telescope"``.

    Exposes telescope-specific fields with snake_case names and proxies
    generic Task attributes (id, type, status, status helpers) so callers
    rarely need to unwrap ``.task`` directly.
    """

    __slots__ = ("_task",)

    def __init__(self, task: Task) -> None:
        sensor_type = getattr(task, "sensor_type", "telescope")
        if sensor_type != "telescope":
            raise ValueError(f"Expected telescope task, got sensor_type={sensor_type!r}")
        self._task = task

    @property
    def task(self) -> Task:
        return self._task

    # -- Generic task fields proxied for convenience --------------------------

    @property
    def id(self) -> str:
        return self._task.id

    @property
    def type(self) -> str:
        return self._task.type

    @property
    def status(self) -> str:
        return self._task.status

    @property
    def task_start(self) -> str:
        return self._task.taskStart

    @property
    def task_stop(self) -> str:
        return self._task.taskStop

    def set_status_msg(self, msg: str | None) -> None:
        self._task.set_status_msg(msg)

    def get_status_msg(self) -> str | None:
        return self._task.get_status_msg()

    # -- Telescope-specific fields --------------------------------------------

    @property
    def satellite_id(self) -> str:
        return self._task.satelliteId

    @property
    def satellite_name(self) -> str:
        return self._task.satelliteName

    @property
    def telescope_id(self) -> str:
        return self._task.telescopeId

    @property
    def telescope_name(self) -> str:
        return self._task.telescopeName

    @property
    def ground_station_id(self) -> str:
        return self._task.groundStationId

    @property
    def ground_station_name(self) -> str:
        return self._task.groundStationName

    @property
    def assigned_filter_name(self) -> str | None:
        return self._task.assigned_filter_name

    def __repr__(self) -> str:
        return f"<TelescopeTaskView {self._task.id[:8]} '{self._task.satelliteName}'>"
