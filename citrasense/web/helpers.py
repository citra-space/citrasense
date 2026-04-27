"""Small formatting helpers shared across the web layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from citrasense.constants import AUTOFOCUS_TARGET_PRESETS
from citrasense.location.gps_fix import GPSFix

if TYPE_CHECKING:
    from citrasense.sensors.sensor_runtime import SensorRuntime


def get_sensor_context(ctx: Any, sensor_id: str) -> tuple[Any, SensorRuntime]:
    """Resolve *sensor_id* to ``(sensor, runtime)`` or raise 404.

    The sensor is typed as ``Any`` because callers typically need
    modality-specific attributes (e.g. ``TelescopeSensor.adapter``)
    that are not on the abstract base class.
    """
    sm = getattr(ctx.daemon, "sensor_manager", None)
    if sm is None:
        raise HTTPException(503, "Sensor manager not initialized")
    sensor = sm.get_sensor(sensor_id)
    if sensor is None:
        raise HTTPException(404, f"Unknown sensor: {sensor_id}")
    td = getattr(ctx.daemon, "task_dispatcher", None)
    runtime = td.get_runtime(sensor_id) if td else None
    if runtime is None:
        raise HTTPException(503, f"No runtime for sensor: {sensor_id}")
    return sensor, runtime


# Standard astronomical filter names for the editable filter name dropdown.
# Mirrors the canonical names from the Citra API's filter library so that
# task assignment matching works without typos.
FILTER_NAME_OPTIONS = [
    {"group": "Broadband", "names": ["Luminance", "Red", "Green", "Blue", "Clear"]},
    {"group": "Johnson-Cousins", "names": ["U", "B", "V", "R", "I"]},
    {"group": "Sloan", "names": ["sloan_u", "sloan_g", "sloan_r", "sloan_i", "sloan_z"]},
    {"group": "Narrowband", "names": ["Ha", "Hb", "OIII", "SII"]},
]


def _gps_fix_to_dict(fix: GPSFix) -> dict[str, Any]:
    """Convert a GPSFix into the dict shape broadcast to the web UI."""
    return {
        "latitude": fix.latitude,
        "longitude": fix.longitude,
        "altitude": fix.altitude,
        "fix_mode": fix.fix_mode,
        "satellites": fix.satellites,
        "is_strong": fix.is_strong_fix,
        "eph": fix.eph,
        "sep": fix.sep,
        "gpsd_version": fix.gpsd_version,
        "device_path": fix.device_path,
        "device_driver": fix.device_driver,
    }


def _task_to_dict(task: Any) -> dict:
    """Format a Task object into the dict shape the web layer expects.

    ``satelliteId`` is included so the Scheduled Tasks card on the monitoring
    page can deep-link the target name to the Citra app's satellite page
    (``{app_url}/satellites/{satelliteId}``).  ``app_url`` is exposed
    separately via ``GET /api/config``.
    """
    d: dict[str, Any] = {
        "id": task.id,
        "start_time": task.taskStart,
        "stop_time": task.taskStop or None,
        "status": task.status,
        "sensor_type": getattr(task, "sensor_type", "telescope"),
        "sensor_id": getattr(task, "sensor_id", None),
    }
    if getattr(task, "sensor_type", "telescope") == "telescope":
        d["satelliteId"] = task.satelliteId
        d["target"] = task.satelliteName
        d["filter"] = task.assigned_filter_name
    else:
        d["target"] = getattr(task, "sensor_id", "") or task.id[:8]
    return d


def _resolve_autofocus_target_name(settings: Any) -> str:
    """Return a human-readable name for the active autofocus target."""
    preset_key = settings.autofocus_target_preset or "mirach"

    if preset_key == "current":
        return "Current position"

    if preset_key == "custom":
        ra = settings.autofocus_target_custom_ra
        dec = settings.autofocus_target_custom_dec
        if ra is not None and dec is not None:
            return f"Custom (RA={ra:.4f}, Dec={dec:.4f})"
        return "Mirach (custom missing coords)"

    preset = AUTOFOCUS_TARGET_PRESETS.get(preset_key)
    if not preset:
        return f"Mirach (unknown preset '{preset_key}')"

    return f"{preset['name']} ({preset['designation']})"
