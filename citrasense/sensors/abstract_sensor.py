"""Core contracts for the Sensor abstraction.

The :class:`AbstractSensor` base class is deliberately narrow. Modality-specific
verbs (``point_telescope``, ``take_image``, filter management, autofocus, ...)
stay on :class:`~citrasense.hardware.abstract_astro_hardware_adapter.AbstractAstroHardwareAdapter`
and are reachable via ``TelescopeSensor.adapter``. Only the lifecycle and the
two acquisition verb-sets belong here.

Acquisition modes
-----------------

* ``ON_DEMAND`` sensors (e.g. the telescope today) get a Citra task and produce
  an :class:`AcquisitionResult`. They implement :meth:`AbstractSensor.acquire`.
* ``STREAMING`` sensors (e.g. the upcoming passive-radar service) run
  continuously and push :class:`AcquisitionEvent` objects onto a
  :class:`~citrasense.sensors.bus.SensorBus`. They implement
  :meth:`AbstractSensor.start_stream` / :meth:`AbstractSensor.stop_stream`.

A sensor implements exactly one set of verbs. The other raises
``NotImplementedError`` and the dispatcher never calls it (it checks
``capabilities.acquisition_mode`` first).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from citrasense.sensors.bus import SensorBus
    from citrasense.tasks.task import Task


class SensorAcquisitionMode(Enum):
    """How a sensor produces data.

    See module docstring for the semantics of each mode.
    """

    ON_DEMAND = "on_demand"
    STREAMING = "streaming"


@dataclass(frozen=True)
class SensorCapabilities:
    """What a concrete sensor can do — advertised to the daemon at boot time.

    Kept intentionally minimal in phase 1. Additional capability bits (FOV,
    supported filters, native streaming rate, ...) are follow-ups.
    """

    acquisition_mode: SensorAcquisitionMode
    modalities: tuple[str, ...]  # e.g. ("optical",), ("radar",), ("rf",)


@dataclass
class AcquisitionEvent:
    """An event emitted by a streaming sensor.

    ``payload`` is modality-specific and intentionally untyped at this layer —
    the concrete sensor's own pydantic models should be used for validation
    before publishing.
    """

    timestamp: datetime
    sensor_id: str
    modality: str
    payload: dict[str, Any]


@dataclass
class AcquisitionResult:
    """The result of a one-shot, on-demand acquisition.

    Shaped after today's telescope output (image paths + metadata) so the
    existing processing/upload pipeline can keep consuming it unchanged when
    the task-flow refactor wires sensors into the dispatcher in phase 4.
    """

    image_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class AcquisitionContext:
    """Site-wide services a sensor may need during ``connect``/``acquire``/``start_stream``.

    Every field is optional so phase-1 sensors can construct a minimal context
    in tests, and so follow-up phases can extend this without breaking
    existing sensors. The daemon will populate the full set before calling
    into any sensor at runtime.

    The telescope's existing data flow continues to run through
    :class:`~citrasense.tasks.runner.TaskManager`; this context is the seam
    that will carry those services into sensors in phase 4/5.
    """

    # Narrow services; typed loosely to avoid upward imports from sensors/
    # into api/, location/, etc. (see CLAUDE.md "Dependency direction").
    api_client: Any = None
    location_service: Any = None
    preview_bus: Any = None
    elset_cache: Any = None
    apass_catalog: Any = None
    safety_monitor: Any = None
    settings: Any = None
    telescope_record: dict[str, Any] | None = None
    ground_station: dict[str, Any] | None = None
    processor_registry: Any = None
    task_index: Any = None


class AbstractSensor(ABC):
    """Narrow contract every sensor must implement.

    Subclasses advertise their modality and mode via :meth:`get_capabilities`
    and implement either :meth:`acquire` (ON_DEMAND) or
    :meth:`start_stream` / :meth:`stop_stream` (STREAMING). The unused verbs
    stay as the default ``NotImplementedError`` stubs.
    """

    #: Short registry key — ``"telescope"``, future ``"passive_radar"``, ``"rf"``.
    sensor_type: ClassVar[str] = ""

    def __init__(self, sensor_id: str) -> None:
        self.sensor_id = sensor_id

    @abstractmethod
    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        """Bring the sensor online (serial handshake, SDK init, etc.)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release any resources held by the sensor."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True iff the sensor is currently reachable."""

    @abstractmethod
    def get_capabilities(self) -> SensorCapabilities:
        """Describe what this sensor can do."""

    @abstractmethod
    def get_settings_schema(self) -> list[dict[str, Any]]:
        """Return the settings schema exposed to the web UI."""

    # ── On-demand acquisition ─────────────────────────────────────────────
    def acquire(self, task: Task, ctx: AcquisitionContext) -> AcquisitionResult:
        """Perform a one-shot acquisition for the given task.

        Default implementation raises ``NotImplementedError``. ``ON_DEMAND``
        sensors must override this; ``STREAMING`` sensors must not be called
        here (the dispatcher checks ``capabilities.acquisition_mode``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement acquire(); "
            "check capabilities.acquisition_mode before calling."
        )

    # ── Streaming acquisition ─────────────────────────────────────────────
    def start_stream(self, bus: SensorBus, ctx: AcquisitionContext) -> None:
        """Begin publishing :class:`AcquisitionEvent`s to ``bus``.

        The sensor MUST publish events under
        ``sensors.{self.sensor_id}.events.acquisition`` (see
        :mod:`citrasense.sensors.bus` for the full convention).

        Default implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement start_stream(); "
            "check capabilities.acquisition_mode before calling."
        )

    def stop_stream(self) -> None:
        """Stop any in-flight streaming work.

        Must be idempotent. Default implementation raises
        ``NotImplementedError`` to match :meth:`start_stream`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stop_stream(); "
            "check capabilities.acquisition_mode before calling."
        )
