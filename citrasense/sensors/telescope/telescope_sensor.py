"""Telescope sensor — phase-1 wrapper around ``AbstractAstroHardwareAdapter``.

This class adapts the existing telescope hardware stack (NINA, KStars, INDI,
Direct, Dummy) to the new :class:`~citrasense.sensors.abstract_sensor.AbstractSensor`
contract. It intentionally stays **thin** in phase 1: the adapter is the
authoritative controller and still owns mount/camera/filter/focuser. Consumers
who need telescope-specific verbs reach through ``sensor.adapter`` — the
compatibility surface the daemon's ``hardware_adapter`` property relies on.

Phase 1 contract
----------------

* ``acquisition_mode = ON_DEMAND``, modality ``"optical"``.
* ``connect`` / ``disconnect`` / ``is_connected`` forward to the adapter.
* ``get_settings_schema`` forwards to the adapter class's classmethod (same
  API the web config form uses today).
* :meth:`acquire` is wired in phase 4 (task-flow refactor). In phase 1 the
  daemon still drives acquisition through ``TaskManager`` → telescope tasks
  → ``self.hardware_adapter.*``, so calling ``acquire`` is a programming
  error and raises ``NotImplementedError``.
* Streaming verbs raise ``NotImplementedError`` (this sensor is on-demand).
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from citrasense.hardware.adapter_registry import get_adapter_class
from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    AcquisitionResult,
    SensorAcquisitionMode,
    SensorCapabilities,
)

if TYPE_CHECKING:
    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.settings.citrasense_settings import SensorConfig
    from citrasense.tasks.task import Task


class TelescopeSensor(AbstractSensor):
    """Thin ``AbstractSensor`` wrapper around a telescope hardware adapter."""

    sensor_type: ClassVar[str] = "telescope"

    _CAPABILITIES: ClassVar[SensorCapabilities] = SensorCapabilities(
        acquisition_mode=SensorAcquisitionMode.ON_DEMAND,
        modalities=("optical",),
    )

    def __init__(
        self,
        sensor_id: str,
        adapter: AbstractAstroHardwareAdapter,
        *,
        adapter_key: str = "",
    ) -> None:
        """Build a telescope sensor around an already-constructed adapter.

        Args:
            sensor_id: Unique sensor id (e.g. ``"telescope-0"``).
            adapter: The concrete hardware adapter that does the real work.
            adapter_key: The short registry key (``"nina"``, ``"direct"``,
                ...) the adapter was built from. Kept around so the web UI
                can look up the settings schema by key even after the
                adapter has been instantiated.
        """
        super().__init__(sensor_id=sensor_id)
        self.adapter = adapter
        self.adapter_key = adapter_key

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: SensorConfig,
        *,
        logger: Logger,
        images_dir: Path,
    ) -> TelescopeSensor:
        """Build a :class:`TelescopeSensor` from a :class:`SensorConfig`.

        Resolves ``cfg.adapter`` through
        :func:`citrasense.hardware.adapter_registry.get_adapter_class` and
        instantiates it with the same kwargs the daemon's legacy
        ``_create_hardware_adapter`` used: ``logger``, ``images_dir``, and
        the adapter-specific settings dict splat.
        """
        adapter_class = get_adapter_class(cfg.adapter)
        adapter = adapter_class(logger=logger, images_dir=images_dir, **cfg.adapter_settings)
        return cls(sensor_id=cfg.id, adapter=adapter, adapter_key=cfg.adapter)

    # ── AbstractSensor surface ────────────────────────────────────────

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        """Forward to :meth:`AbstractAstroHardwareAdapter.connect`.

        ``ctx`` is accepted for protocol compatibility but not consumed here;
        the telescope adapter gets its site services wired by the daemon
        separately (``set_location_service``, ``set_safety_monitor``, ...)
        during ``_initialize_telescope``. Phase-4 follow-ups will consolidate
        that wiring through the context.
        """
        del ctx
        return bool(self.adapter.connect())

    def disconnect(self) -> None:
        """Forward to :meth:`AbstractAstroHardwareAdapter.disconnect`."""
        self.adapter.disconnect()

    def is_connected(self) -> bool:
        """True iff the adapter reports both telescope and camera connected."""
        return bool(self.adapter.is_telescope_connected() and self.adapter.is_camera_connected())

    def get_capabilities(self) -> SensorCapabilities:
        return self._CAPABILITIES

    def get_settings_schema(self) -> list[dict[str, Any]]:
        """Forward to the adapter class's ``get_settings_schema`` classmethod.

        Uses the adapter class (not the instance) to match the existing
        web-config behavior: the schema is looked up from the class so the
        UI can ask for it before the adapter is connected.
        """
        adapter_cls = type(self.adapter)
        return [dict(entry) for entry in adapter_cls.get_settings_schema()]

    # ── Acquisition verbs ─────────────────────────────────────────────

    def acquire(self, task: Task, ctx: AcquisitionContext) -> AcquisitionResult:
        """Wired in phase 4.

        Today the daemon still routes through ``TaskManager`` →
        ``AbstractBaseTelescopeTask`` → ``self.hardware_adapter.*``. Calling
        ``TelescopeSensor.acquire`` in phase 1 is a programming error.
        """
        raise NotImplementedError(
            "TelescopeSensor.acquire() is wired in phase 4 of the multi-sensor "
            "migration; today the daemon still drives acquisition via "
            "TaskManager and the underlying hardware adapter."
        )
