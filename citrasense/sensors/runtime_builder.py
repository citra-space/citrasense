"""Per-modality runtime builder protocol and shared build context.

The sensor init pipeline has two phases per sensor:

1. **Synchronous build** — construct a :class:`SensorRuntime` in
   ``init_state="pending"``, register it with the dispatcher, and
   wire any session managers / libraries that don't depend on a live
   adapter.  No ``adapter.connect()`` here.
2. **Asynchronous post-connect wiring** — run on the orchestrator's
   worker thread *after* ``adapter.connect()`` succeeds: filter
   discovery, pointing model installation, calibration manager
   wiring, safety check registration.

This module defines the abstractions both phases share:

* :class:`BuildContext` — bundles the daemon-owned dependencies the
  builders need.  Passing this dataclass instead of the daemon itself
  keeps the dependency surface narrow and explicit (per the
  *Module boundaries and encapsulation* section in CLAUDE.md).
* :class:`RuntimeBuilder` — the small Protocol the per-modality
  builders implement.
* :func:`build_for_sensor` — dispatches by ``sensor.sensor_type`` to
  the right modality builder so the daemon can register every sensor
  in a single loop without knowing the modality details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

    from citrasense.analysis.task_index import TaskIndex
    from citrasense.api.citra_api_client import AbstractCitraApiClient
    from citrasense.astro.elset_cache import ElsetCache
    from citrasense.catalogs.apass_catalog import ApassCatalog
    from citrasense.location import LocationService
    from citrasense.safety.safety_monitor import SafetyMonitor
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.sensors.bus import InProcessBus
    from citrasense.sensors.preview_bus import PreviewBus
    from citrasense.sensors.sensor_runtime import SensorRuntime
    from citrasense.settings.citrasense_settings import CitraSenseSettings
    from citrasense.tasks.task_dispatcher import TaskDispatcher
    from citrasense.web.server import CitraSenseWebServer


@dataclass
class BuildContext:
    """Daemon-owned dependencies passed to per-modality runtime builders.

    Each field is a narrow capability (an API client, a single
    service, a callback) instead of the daemon object as a whole.
    Builders read from these fields but never mutate the daemon's
    state directly — any state the daemon needs back from a builder
    (e.g. a freshly-built runtime) flows out as a return value.
    """

    api_client: AbstractCitraApiClient
    task_dispatcher: TaskDispatcher
    safety_monitor: SafetyMonitor
    settings: CitraSenseSettings
    location_service: LocationService | None
    elset_cache: ElsetCache
    apass_catalog: ApassCatalog
    ground_station: dict[str, Any] | None
    preview_bus: PreviewBus
    task_index: TaskIndex | None
    sensor_bus: InProcessBus
    web_server: CitraSenseWebServer | None
    on_annotated_image: Callable[[str, str], None]
    # Filter-config helpers live on the daemon (web routes call them
    # too).  Pass them in as callbacks so the telescope builder
    # doesn't need to import the daemon.
    save_filter_config: Callable[[Any], None]
    sync_filters_to_backend: Callable[[Any], None]
    logger: logging.Logger
    # Optional callback installed on every freshly-built runtime so
    # init-state transitions surface as toasts.  Provided by the
    # init orchestrator; the daemon itself shouldn't need to import
    # the toast wiring.
    init_state_callback_factory: Callable[[str], Callable[[str, str | None], None] | None] | None = None


class RuntimeBuilder(Protocol):
    """Protocol implemented by each modality's runtime builder.

    ``build`` is synchronous and side-effect-light; ``connect_post_wiring``
    runs on the init executor's worker thread after a successful
    ``adapter.connect()``.  Modalities with no post-connect work
    (radar, allsky today) can implement ``connect_post_wiring`` as a
    no-op.
    """

    def build(self, sensor: AbstractSensor) -> SensorRuntime:
        """Construct a SensorRuntime in ``init_state='pending'`` and register it.

        Must register the runtime with the dispatcher before returning so
        the dispatcher can immediately gate task dispatch on
        ``runtime.is_ready`` while the connect is still in flight.
        """
        ...

    def connect_post_wiring(self, runtime: SensorRuntime) -> None:
        """Wire post-connect dependencies (filter sync, pointing model, etc.).

        Errors raised here propagate to the orchestrator which marks
        the runtime ``failed`` — so a transient problem in a wiring
        step doesn't silently leave the runtime in an inconsistent
        ``connected`` state.
        """
        ...


def build_for_sensor(sensor: AbstractSensor, ctx: BuildContext) -> tuple[SensorRuntime, RuntimeBuilder]:
    """Dispatch to the right modality builder by ``sensor.sensor_type``.

    Returns the freshly-built runtime *and* the builder instance so the
    init orchestrator can call ``connect_post_wiring`` on the same
    builder later, after ``adapter.connect()`` succeeds.  Builders are
    cheap (no per-sensor caches), so constructing a fresh one per
    sensor keeps each sensor's wiring independent.
    """
    sensor_type = sensor.sensor_type
    if sensor_type == "telescope":
        from citrasense.sensors.telescope.runtime_builder import TelescopeRuntimeBuilder

        builder: RuntimeBuilder = TelescopeRuntimeBuilder(ctx)
    elif sensor_type == "passive_radar":
        from citrasense.sensors.radar.runtime_builder import RadarRuntimeBuilder

        builder = RadarRuntimeBuilder(ctx)
    elif sensor_type == "allsky":
        from citrasense.sensors.allsky.runtime_builder import AllskyRuntimeBuilder

        builder = AllskyRuntimeBuilder(ctx)
    else:
        raise ValueError(f"No RuntimeBuilder registered for sensor_type={sensor_type!r}")

    runtime = builder.build(sensor)
    return runtime, builder
