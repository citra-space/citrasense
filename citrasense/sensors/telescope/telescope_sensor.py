"""Telescope sensor — wrapper around ``AbstractAstroHardwareAdapter``.

This class adapts the telescope hardware stack (NINA, KStars, INDI,
Direct, Dummy) to the :class:`~citrasense.sensors.abstract_sensor.AbstractSensor`
contract. It stays **thin**: the adapter is the authoritative controller
and still owns mount/camera/filter/focuser. Consumers who need
telescope-specific verbs reach through ``sensor.adapter`` — the daemon
no longer exposes a top-level ``hardware_adapter`` property; hardware
is resolved per-sensor via the ``SensorManager`` + ``SensorRuntime``.

Phase 1 contract
----------------

* ``acquisition_mode = ON_DEMAND``, modality ``"optical"``.
* ``connect`` / ``disconnect`` / ``is_connected`` forward to the adapter.
* ``get_settings_schema`` forwards to the adapter class's classmethod (same
  API the web config form uses today).
* :meth:`acquire` is wired in phase 4 (task-flow refactor). In phase 1 the
  daemon still drives acquisition through ``TaskDispatcher`` → telescope tasks
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
    import logging

    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.safety.safety_monitor import SafetyMonitor
    from citrasense.sensors.telescope.safety.cable_wrap_check import CableWrapCheck
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
        self.citra_record: dict[str, Any] | None = None
        self._cable_wrap_check: CableWrapCheck | None = None
        # Let the adapter namespace its per-sensor state files
        # (e.g. pointing_model_<sensor_id>.json) via self.sensor_id.
        self.adapter.sensor_id = sensor_id

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: SensorConfig,
        *,
        logger: Logger,
        images_dir: Path,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> TelescopeSensor:
        """Build a :class:`TelescopeSensor` from a :class:`SensorConfig`.

        Resolves ``cfg.adapter`` through
        :func:`citrasense.hardware.adapter_registry.get_adapter_class` and
        instantiates it with the same kwargs the daemon's legacy
        ``_create_hardware_adapter`` used: ``logger``, ``images_dir``, and
        the adapter-specific settings dict splat.

        The logger handed to the adapter is wrapped in a sensor-scoped
        :class:`SensorLoggerAdapter` so every record emitted by INDI /
        NINA / KStars / Direct flows carries ``extra={'sensor_id': ...}``
        for the web UI's per-sensor log filter.
        """
        from citrasense.logging.sensor_logger import get_sensor_logger

        adapter_class = get_adapter_class(cfg.adapter)
        adapter_logger = get_sensor_logger(
            logger.getChild(f"{adapter_class.__name__}[{cfg.id}]"),
            cfg.id,
        )
        extra_kwargs: dict = {}
        if data_dir is not None:
            extra_kwargs["data_dir"] = data_dir
        if cache_dir is not None:
            extra_kwargs["cache_dir"] = cache_dir
        adapter = adapter_class(logger=adapter_logger, images_dir=images_dir, **extra_kwargs, **cfg.adapter_settings)
        return cls(sensor_id=cfg.id, adapter=adapter, adapter_key=cfg.adapter)

    # ── AbstractSensor surface ────────────────────────────────────────

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        """Forward to :meth:`AbstractAstroHardwareAdapter.connect`.

        ``ctx`` is accepted for protocol compatibility but not consumed here;
        the telescope adapter gets its site services wired by the daemon
        separately (``set_location_service``, ``set_safety_monitor``, ...)
        during ``_initialize_telescopes``. Phase-4 follow-ups will consolidate
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

        Today the daemon still routes through ``TaskDispatcher`` →
        ``AbstractBaseTelescopeTask`` → ``self.hardware_adapter.*``. Calling
        ``TelescopeSensor.acquire`` in phase 1 is a programming error.
        """
        raise NotImplementedError(
            "TelescopeSensor.acquire() is wired in phase 4 of the multi-sensor "
            "migration; today the daemon still drives acquisition via "
            "TaskDispatcher and the underlying hardware adapter."
        )

    # ── Safety checks ─────────────────────────────────────────────────

    def register_safety_checks(
        self,
        safety_monitor: SafetyMonitor,
        *,
        logger: logging.Logger,
        state_file: Path,
    ) -> None:
        """Create and register telescope-specific safety checks.

        Creates a :class:`CableWrapCheck` for the mount (if one exists),
        registers it with ``safety_monitor``, wires the operator-stop gate,
        and performs a deferred startup unwind if the persisted cable state
        exceeds the hard limit.
        """
        from citrasense.sensors.telescope.safety.cable_wrap_check import CableWrapCheck as _CableWrapCheck

        mount = self.adapter.mount
        if mount is None:
            return

        self._cable_wrap_check = _CableWrapCheck(logger, mount, state_file=state_file)
        self._cable_wrap_check.start()
        mount.register_sync_listener(self._cable_wrap_check.notify_sync)

        op_stop = safety_monitor.operator_stop
        self._cable_wrap_check.safety_gate = lambda: not op_stop.is_active

        safety_monitor.register_sensor_check(self.sensor_id, self._cable_wrap_check)

        if self._cable_wrap_check.needs_startup_unwind():
            logger.warning(
                "Persisted cable wrap at %.1f° exceeds hard limit — starting deferred unwind",
                self._cable_wrap_check.cumulative_deg,
            )
            self._cable_wrap_check.execute_action()
            if self._cable_wrap_check.did_last_unwind_fail():
                self._cable_wrap_check.mark_intervention_required()
                logger.critical(
                    "Startup unwind did not converge (%.1f° remaining) — "
                    "manual intervention required before the system can "
                    "operate. Use web UI to reset after physically verifying cables.",
                    self._cable_wrap_check.cumulative_deg,
                )

    def unregister_safety_checks(self, safety_monitor: SafetyMonitor) -> None:
        """Unregister and stop telescope-specific safety checks."""
        if self._cable_wrap_check is not None:
            self._cable_wrap_check.join_unwind(timeout=10.0)
            self._cable_wrap_check.stop()
            safety_monitor.unregister_sensor_check(self.sensor_id, "cable_wrap")
            self._cable_wrap_check = None
