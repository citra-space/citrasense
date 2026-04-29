"""Processing context for the radar pipeline.

Mirrors the role :class:`~citrasense.pipelines.optical.optical_processing_context.OpticalProcessingContext`
plays for image pipelines: a narrow, typed bundle of everything a
radar processor needs to do its work.  Unlike the optical context
there's no FITS file — the payload is the enriched ``Observation``
JSON that ``pr_sensor`` published on ``radar.sensor.{id}.observations``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

    from citrasense.api.abstract_api_client import AbstractCitraApiClient
    from citrasense.sensors.radar.events import RadarObservationEvent

    LoggerLike = logging.Logger | logging.LoggerAdapter


@dataclass
class RadarProcessingContext:
    """Narrow dataclass passed into every radar processor.

    Parameters
    ----------
    sensor_id:
        Local citrasense sensor id (``cfg.id``).
    event:
        The :class:`RadarObservationEvent` that triggered processing.
        ``event.payload`` is the full ``pr_sensor`` ``Observation`` dict.
    antenna_id:
        Backend-registered radar antenna UUID used as ``antenna_id`` on
        upload.  Empty string means "unconfigured" — the formatter will
        refuse to upload observations in that case.
    api_client:
        The site's API client (passed to the upload queue).
    artifact_dir:
        Site-wide per-sensor artifact directory
        (``<analysis_dir>/radar/<sensor_id>``).  Created by the
        runtime; processors write sanitised JSON into it.  ``None``
        means "don't persist artifacts" — the writer becomes a no-op
        rather than falling back to CWD.
    detection_min_snr_db:
        SNR floor gate — observations with ``quality.snr_db`` below
        this threshold are dropped before upload formatting.
    forward_only_tasked_satellites:
        If ``True``, only observations whose ``target.citra_uuid``
        matches a satellite currently tasked at this site are
        uploaded.
    task_index:
        Lightweight tasked-sats lookup (the daemon's ``TaskIndex``
        instance) used for the tasked-satellite gate.  ``None`` means
        the gate degrades to "allow everything" — matching v1 intent.
    logger:
        Sensor-scoped logger.
    received_at:
        Monotonic timestamp of when the event was picked off the bus
        (optional; used for latency telemetry).
    """

    sensor_id: str
    event: RadarObservationEvent
    antenna_id: str = ""
    api_client: AbstractCitraApiClient | None = None
    artifact_dir: Path | None = None
    detection_min_snr_db: float = 0.0
    forward_only_tasked_satellites: bool = False
    task_index: Any = None
    logger: LoggerLike | None = None
    received_at: datetime | None = None
    # Populated by the formatter; read by the upload queue.
    upload_payload: dict[str, Any] | None = None
    # Short reason why the observation was dropped (if any).
    drop_reason: str | None = None
