"""Runtime collection of live sensors owned by the daemon.

``SensorManager`` is the daemon-facing facade. It knows how to build sensors
from :class:`~citrasense.settings.citrasense_settings.SensorConfig` (via
:mod:`citrasense.sensors.sensor_registry`) and drives bulk lifecycle
operations (connect all, disconnect all). The daemon talks to this object
instead of a single ``hardware_adapter`` attribute.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING

from citrasense.sensors.sensor_registry import get_sensor_class

if TYPE_CHECKING:
    from citrasense.sensors.abstract_sensor import AbstractSensor, AcquisitionContext
    from citrasense.settings.citrasense_settings import SensorConfig


class SensorManager:
    """Owns the live sensor instances for the current daemon lifecycle."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._sensors: dict[str, AbstractSensor] = {}

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        configs: Iterable[SensorConfig],
        *,
        logger: Logger,
        images_dir: Path,
    ) -> SensorManager:
        """Build a ``SensorManager`` from a list of :class:`SensorConfig`.

        Each config's ``type`` is resolved through the sensor registry; the
        concrete class is expected to expose a
        ``from_config(cfg, *, logger, images_dir)`` classmethod that handles
        modality-specific wiring (for the telescope, that means looking up
        the NINA/KStars/INDI/Direct adapter).
        """
        mgr = cls(logger=logger)
        for cfg in configs:
            sensor_cls = get_sensor_class(cfg.type)
            sensor = sensor_cls.from_config(cfg, logger=logger, images_dir=images_dir)  # type: ignore[attr-defined]
            mgr.register(sensor)
        return mgr

    def register(self, sensor: AbstractSensor) -> None:
        """Add ``sensor`` to the manager. Raises if the id is already taken."""
        if sensor.sensor_id in self._sensors:
            raise ValueError(f"Duplicate sensor id: {sensor.sensor_id!r}")
        self._sensors[sensor.sensor_id] = sensor

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, sensor_id: str) -> AbstractSensor:
        """Return the sensor registered under ``sensor_id``."""
        return self._sensors[sensor_id]

    def get_sensor(self, sensor_id: str) -> AbstractSensor | None:
        """Return the sensor registered under *sensor_id*, or ``None``."""
        return self._sensors.get(sensor_id)

    # Kept for internal callers that prefer the shorter name.
    get_or_none = get_sensor

    def all(self) -> list[AbstractSensor]:
        """All registered sensors, in registration order."""
        return list(self._sensors.values())

    def iter_by_type(self, sensor_type: str) -> Iterator[AbstractSensor]:
        """Iterate sensors whose ``sensor_type`` matches."""
        for s in self._sensors.values():
            if s.sensor_type == sensor_type:
                yield s

    # NOTE: ``first_of_type`` used to live here as a convenience helper, but it
    # was a classic "first sensor wins" footgun in multi-sensor deployments —
    # callers that needed *a* telescope silently got the first one registered
    # and hid bugs where the actual sensor should have been selected by id.
    # Use ``iter_by_type`` + explicit selection, or resolve via
    # ``TaskDispatcher._runtime_for_task`` / ``SensorManager.get``.

    # ── Bulk lifecycle ────────────────────────────────────────────────

    def connect_all(self, ctx: AcquisitionContext | None = None) -> dict[str, bool]:
        """Connect every sensor. Returns ``{sensor_id: success}``.

        Connection failures are logged but do not raise; callers decide
        whether a single-sensor failure is fatal.
        """
        results: dict[str, bool] = {}
        for sid, sensor in self._sensors.items():
            try:
                results[sid] = bool(sensor.connect(ctx))
            except Exception as e:
                self._logger.error("Sensor %r failed to connect: %s", sid, e, exc_info=True)
                results[sid] = False
        return results

    def disconnect_all(self) -> None:
        """Disconnect every sensor, swallowing per-sensor exceptions."""
        for sid, sensor in self._sensors.items():
            try:
                sensor.disconnect()
            except Exception as e:
                self._logger.warning("Sensor %r failed to disconnect cleanly: %s", sid, e)

    # ── Diagnostics ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sensors)

    def __contains__(self, sensor_id: object) -> bool:
        return sensor_id in self._sensors

    def __iter__(self) -> Iterator[AbstractSensor]:
        return iter(self._sensors.values())
