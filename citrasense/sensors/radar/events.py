"""Pydantic event types published by :class:`PassiveRadarSensor`.

:class:`~citrasense.sensors.bus.SensorBus` requires pydantic ``BaseModel``
events on the wire.  ``AcquisitionEvent`` on the abstract sensor module
is a plain dataclass — radar sensors wrap each observation in a
:class:`RadarObservationEvent` (a ``BaseModel``) before publishing on
``sensors.{sensor_id}.events.acquisition``.

Keeping the event schema narrow and validated here means downstream
consumers (``SensorRuntime``, the processing queue, tests) can rely on
shape without touching the raw NATS JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RadarObservationEvent(BaseModel):
    """An enriched ``pr_sensor`` observation published to the SensorBus.

    ``payload`` is the full JSON dict from ``radar.sensor.{id}.observations``.
    We deliberately don't narrow it further here: ``pr_sensor`` owns the
    schema, and downstream consumers read the fields they need.  If the
    schema drifts, the formatter / filter surface the breakage with a
    clear error rather than a cryptic pydantic failure here.
    """

    model_config = ConfigDict(frozen=True)

    sensor_id: str
    modality: str = "radar"
    timestamp: datetime
    payload: dict[str, Any]
