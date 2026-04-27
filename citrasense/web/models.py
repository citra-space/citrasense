"""Pydantic models used by the web layer."""

from typing import Any

from pydantic import BaseModel


class SystemStatus(BaseModel):
    """Current system status.

    Per-sensor fields (telescope, camera, focuser, mount, autofocus,
    alignment, session, etc.) live exclusively in ``sensors[sensor_id]``
    dicts.  Only true site-level aggregates remain at the top level.
    """

    # Site-level task orchestration.
    #
    # ``current_task_ids`` maps ``sensor_id -> currently-executing task id``
    # for every sensor actively running a task.  There is intentionally no
    # scalar ``current_task`` field: the legacy "first non-None wins" shape
    # was ambiguous in multi-sensor deployments.  The UI uses the keys of
    # this dict to drive per-row ``isActive`` styling and cancel-button
    # gating.
    current_task_ids: dict[str, str] = {}
    tasks_pending: int = 0
    processing_active: bool = True
    system_busy: bool = False
    system_busy_reason: str = ""
    tasks_by_stage: dict[str, list[dict]] | None = None
    pipeline_stats: dict[str, Any] | None = None

    # Ground station / location
    ground_station_id: str | None = None
    ground_station_name: str | None = None
    ground_station_url: str | None = None
    ground_station_latitude: float | None = None
    ground_station_longitude: float | None = None
    ground_station_altitude: float | None = None
    location_source: str | None = None  # "gps" or "ground_station"
    location_latitude: float | None = None
    location_longitude: float | None = None
    location_altitude: float | None = None

    # Time / GPS health
    time_health: dict[str, Any] | None = None
    gpsd_fix: dict[str, Any] | None = None
    adapter_gps: dict[str, Any] | None = None

    # Dependencies & processors
    missing_dependencies: list[dict[str, str]] = []
    active_processors: list[str] = []

    # Safety
    safety_status: dict[str, Any] | None = None
    elset_health: dict[str, Any] | None = None

    # Misc site-level.
    #
    # The legacy scalar ``latest_task_image_url`` was removed: in a
    # multi-sensor deployment "latest" is per sensor and lives at
    # ``sensors[sid]['latest_task_image_url']`` (populated by
    # :class:`StatusCollector`).
    last_update: str = ""
    status_collection_ms: float | None = None
    status_collection_breakdown: dict[str, float] | None = None

    # Per-sensor status: sensors[sensor_id] -> { type, connected, ... }
    sensors: dict[str, dict] = {}
