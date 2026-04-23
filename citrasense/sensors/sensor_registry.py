"""Sensor type registry.

Maps ``sensor_type`` (``"telescope"`` today; ``"passive_radar"`` / ``"rf"``
tomorrow) to a concrete :class:`~citrasense.sensors.abstract_sensor.AbstractSensor`
class. This is the top-level wiring point the daemon uses; the telescope
sensor internally still consults
:mod:`citrasense.hardware.adapter_registry` to pick the actual
NINA/KStars/INDI/Direct/Dummy adapter.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrasense.sensors.abstract_sensor import AbstractSensor

REGISTERED_SENSORS: dict[str, dict[str, str]] = {
    "telescope": {
        "module": "citrasense.sensors.telescope.telescope_sensor",
        "class_name": "TelescopeSensor",
        "description": "Optical telescope (wraps an AbstractAstroHardwareAdapter)",
    },
}


def get_sensor_class(sensor_type: str) -> type[AbstractSensor]:
    """Resolve ``sensor_type`` to its concrete sensor class.

    Args:
        sensor_type: Short registry key such as ``"telescope"``.

    Returns:
        The sensor class (an :class:`AbstractSensor` subclass).

    Raises:
        ValueError: If ``sensor_type`` is not registered.
        ImportError: If the sensor module cannot be imported (missing
            optional dependency, etc.).
    """
    if sensor_type not in REGISTERED_SENSORS:
        available = ", ".join(f"'{name}'" for name in REGISTERED_SENSORS)
        raise ValueError(f"Unknown sensor type: '{sensor_type}'. Valid options are: {available}")

    info = REGISTERED_SENSORS[sensor_type]
    module = importlib.import_module(info["module"])
    return getattr(module, info["class_name"])


def list_sensors() -> dict[str, dict[str, str]]:
    """All registered sensor types keyed by ``sensor_type``."""
    return {
        name: {
            "description": info["description"],
            "module": info["module"],
            "class_name": info["class_name"],
        }
        for name, info in REGISTERED_SENSORS.items()
    }
