"""Sensor abstraction — phase 1 of the multi-sensor migration.

This package introduces the ``Sensor`` contract that will eventually front every
modality (telescope, passive radar, RF, ...) behind a single interface. Phase 1
only ships the contract, an in-process event bus, and a thin
:class:`~citrasense.sensors.telescope.telescope_sensor.TelescopeSensor` wrapper
around the existing :class:`~citrasense.hardware.abstract_astro_hardware_adapter.AbstractAstroHardwareAdapter`.

See issue citra-space/citrasense#306 for scope and follow-ups.
"""

from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    AcquisitionEvent,
    AcquisitionResult,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.bus import InProcessBus, SensorBus, Subscription
from citrasense.sensors.sensor_manager import SensorManager
from citrasense.sensors.sensor_registry import (
    REGISTERED_SENSORS,
    get_sensor_class,
    list_sensors,
)
from citrasense.sensors.sensor_runtime import SensorRuntime

__all__ = [
    "REGISTERED_SENSORS",
    "AbstractSensor",
    "AcquisitionContext",
    "AcquisitionEvent",
    "AcquisitionResult",
    "InProcessBus",
    "SensorAcquisitionMode",
    "SensorBus",
    "SensorCapabilities",
    "SensorManager",
    "SensorRuntime",
    "Subscription",
    "get_sensor_class",
    "list_sensors",
]
