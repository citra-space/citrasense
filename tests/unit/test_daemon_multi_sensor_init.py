"""Daemon-init coverage for the parallel / async sensor-init pipeline (#339).

These tests pin down the behaviour of ``CitraSenseDaemon._initialize_telescopes``
after the refactor that extracted the orchestration plumbing into
:mod:`citrasense.sensors.init_orchestrator` and the per-modality
runtime builders:

- :func:`resolve_canonical_ground_station` runs *before* any
  ``adapter.connect()`` call, stamps ``citra_record`` on every telescope
  sensor, and rejects divergent ``groundStationId`` values up front.
- ``_initialize_telescopes`` registers every runtime in
  ``init_state="pending"`` and submits per-sensor connects to the
  orchestrator's executor, returning quickly even if some adapters block.
- A failing or timing-out connect on one sensor never aborts the others.
- ``request_sensor_reconnect`` re-uses the orchestrator's worker pool.

The tests stub out heavy daemon dependencies (sensor manager, API
client, web server) and patch the per-modality runtime factory so we
can read ``init_state`` transitions without spinning up real queues
or touching real hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

from citrasense.citrasense_daemon import CitraSenseDaemon
from citrasense.sensors.init_orchestrator import resolve_canonical_ground_station


def _make_telescope_sensor(sensor_id: str, *, connect: Any = True) -> MagicMock:
    """Build a fake telescope sensor whose ``adapter.connect()`` is configurable.

    ``connect`` may be ``True`` / ``False`` for fast paths, a callable
    that returns a bool (called when ``adapter.connect()`` runs), or an
    exception instance/class to raise.
    """
    s = MagicMock()
    s.sensor_id = sensor_id
    s.sensor_type = "telescope"
    s.name = sensor_id
    s.is_connected.return_value = False
    s.adapter = MagicMock()

    def _connect_side_effect() -> bool:
        if isinstance(connect, BaseException) or (isinstance(connect, type) and issubclass(connect, BaseException)):
            raise connect  # type: ignore[misc]
        if callable(connect):
            return bool(connect())
        return bool(connect)

    s.adapter.connect.side_effect = _connect_side_effect
    s.connect.side_effect = _connect_side_effect
    s.adapter.is_mount_homed.return_value = True
    s.adapter.supports_direct_camera_control.return_value = False
    s.adapter.supports_flat_automation.return_value = False
    s.adapter.supports_park.return_value = False
    s.adapter.get_filter_config.return_value = {}
    s.adapter.get_gps_location.return_value = None
    s.adapter.pointing_model = None
    s.adapter.camera = None
    s.adapter.scope_slew_rate_degrees_per_second = 0.0
    s.adapter.telescope_record = None
    s.adapter.elset_cache = None
    s.citra_record = None
    return s


def _make_daemon(sensors: list[MagicMock]) -> CitraSenseDaemon:
    """Build a CitraSenseDaemon with the smallest viable test fixtures.

    Uses ``__new__`` to skip ``__init__`` so we don't drag in the full
    daemon dependency graph; we then plug in the few attributes the
    code under test reaches into.
    """
    daemon = CitraSenseDaemon.__new__(CitraSenseDaemon)

    daemon.settings = MagicMock()
    configs = []
    for s in sensors:
        sc = MagicMock()
        sc.id = s.sensor_id
        sc.citra_sensor_id = f"api-{s.sensor_id}"
        sc.connect_timeout_seconds = 5.0
        configs.append(sc)
    daemon.settings.sensors = configs
    daemon.settings.get_sensor_config = lambda sid: next((c for c in configs if c.id == sid), None)

    daemon.api_client = MagicMock()
    daemon.api_client.does_api_server_accept_key.return_value = True

    def _get_telescope(api_id: str) -> dict[str, Any]:
        return {
            "id": api_id,
            "groundStationId": "gs-1",
            "maxSlewRate": 3.0,
            "name": f"Scope {api_id}",
        }

    daemon.api_client.get_telescope.side_effect = _get_telescope
    daemon.api_client.get_ground_station.return_value = {"id": "gs-1", "name": "TestGS"}

    sm = MagicMock()
    sm.iter_by_type.side_effect = lambda t: iter(sensors if t == "telescope" else [])
    sm.__iter__ = lambda self: iter(sensors)
    sm.get_sensor.side_effect = lambda sid: next((s for s in sensors if s.sensor_id == sid), None)
    daemon.sensor_manager = sm

    daemon.location_service = MagicMock()
    daemon.time_monitor = MagicMock()
    daemon.web_server = MagicMock()
    daemon.web_server.send_toast = MagicMock()
    daemon.elset_cache = MagicMock()
    daemon.apass_catalog = MagicMock()
    daemon.processor_registry = MagicMock()
    daemon.task_dispatcher = None
    daemon.safety_monitor = MagicMock()
    daemon.ground_station = None
    daemon.latest_annotated_image_paths = {}
    daemon.preview_bus = MagicMock()
    daemon.task_index = MagicMock()
    daemon.sensor_bus = MagicMock()
    daemon._retention_timer = None
    daemon._stop_requested = False

    # The async-init plumbing is now owned by SensorInitOrchestrator,
    # constructed inside _initialize_telescopes once the dispatcher
    # exists.  Tests that call _initialize_telescopes() get an
    # orchestrator wired automatically; tests that exercise the
    # reconnect path can also seed daemon._init_orchestrator manually.
    daemon._init_orchestrator = None

    return daemon


class _RuntimeStub:
    """Just enough of SensorRuntime to exercise init_state and connect-callback."""

    def __init__(self, sensor: MagicMock, **_kwargs: Any) -> None:
        self.sensor = sensor
        self.sensor_id = sensor.sensor_id
        self.sensor_type = sensor.sensor_type
        self.hardware_adapter = sensor.adapter
        self.init_state = "pending"
        self.init_error: str | None = None
        self.on_init_state_change: Any = None
        self._transitions: list[tuple[str, str | None]] = []
        self.alignment_manager = MagicMock()
        self.alignment_manager.set_pointing_model = MagicMock()
        self.autofocus_manager = MagicMock()
        self.calibration_library = MagicMock()
        self.calibration_manager = None
        self.observing_session_manager = None
        self.self_tasking_manager = None
        self.acquisition_queue = MagicMock()
        self.logger = MagicMock()
        # Methods touched by TelescopeRuntimeBuilder.build
        self.are_queues_idle = MagicMock()
        self.attach_calibration_library = MagicMock(side_effect=self._set_calibration_library)

    def _set_calibration_library(self, library: Any) -> None:
        self.calibration_library = library

    @property
    def is_ready(self) -> bool:
        return self.init_state == "connected"

    def _transition(self, state: str, error: str | None) -> None:
        self.init_state = state
        self.init_error = error if state in ("failed", "timed_out") else None
        self._transitions.append((state, error))
        cb = self.on_init_state_change
        if cb:
            cb(state, error)

    def mark_connecting(self) -> None:
        self._transition("connecting", None)

    def mark_connected(self) -> None:
        self._transition("connected", None)

    def mark_failed(self, error: str) -> None:
        self._transition("failed", error)

    def mark_timed_out(self, error: str) -> None:
        self._transition("timed_out", error)

    def mark_disconnected(self) -> None:
        self._transition("pending", None)


def _patched_runtimes() -> tuple[Any, dict[str, _RuntimeStub]]:
    """Patch ``SensorRuntime`` so each constructed runtime is a stub we can inspect."""
    stubs: dict[str, _RuntimeStub] = {}

    def _factory(sensor: MagicMock, **kwargs: Any) -> _RuntimeStub:
        stub = _RuntimeStub(sensor, **kwargs)
        stubs[stub.sensor_id] = stub
        return stub

    return _factory, stubs


class TestGroundStationPreflight:
    """Exercise :func:`resolve_canonical_ground_station`."""

    def test_canonical_chosen_in_lexical_order(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        daemon = _make_daemon([scope_b, scope_a])

        # Both scopes report gs-1 from the default mock; the canonical
        # decision must not depend on the iteration order of the
        # sensor manager — sorting by sensor_id is the documented rule.
        ok, err, gs = resolve_canonical_ground_station(
            sensor_manager=daemon.sensor_manager,  # type: ignore[arg-type]
            settings=daemon.settings,
            api_client=daemon.api_client,  # type: ignore[arg-type]
            logger=MagicMock(),
            location_service=daemon.location_service,
        )
        assert ok is True
        assert err is None
        assert gs is not None
        assert gs["id"] == "gs-1"
        # Both sensors must have citra_record stamped, regardless of order.
        assert scope_a.citra_record is not None
        assert scope_b.citra_record is not None

    def test_divergent_ground_stations_rejected(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        daemon = _make_daemon([scope_a, scope_b])

        def _get_gs(gs_id: str) -> dict[str, Any]:
            return {"id": gs_id, "name": f"GS-{gs_id}"}

        def _get_telescope(api_id: str) -> dict[str, Any]:
            return {
                "id": api_id,
                "groundStationId": "gs-1" if api_id == "api-scope-a" else "gs-2",
                "maxSlewRate": 3.0,
                "name": api_id,
            }

        daemon.api_client.get_telescope.side_effect = _get_telescope  # type: ignore[union-attr]
        daemon.api_client.get_ground_station.side_effect = _get_gs  # type: ignore[union-attr]

        ok, err, gs = resolve_canonical_ground_station(
            sensor_manager=daemon.sensor_manager,  # type: ignore[arg-type]
            settings=daemon.settings,
            api_client=daemon.api_client,  # type: ignore[arg-type]
            logger=MagicMock(),
            location_service=daemon.location_service,
        )
        assert ok is False
        assert err is not None
        assert "Multi-ground-station deployments are not supported" in err
        assert gs is None


class TestParallelInit:
    """Exercise the parallel/async sensor init fan-out via _initialize_telescopes."""

    def _setup_dispatcher(self, daemon: CitraSenseDaemon) -> MagicMock:
        td = MagicMock()
        td.task_dict = {}
        td.imaging_tasks = {}
        td.processing_tasks = {}
        td.uploading_tasks = {}
        td._runtimes = {}

        def _register(rt: Any) -> None:
            td._runtimes[rt.sensor_id] = rt

        def _iter_runtimes() -> list[Any]:
            return list(td._runtimes.values())

        def _get_runtime(sid: str) -> Any:
            return td._runtimes.get(sid)

        td.register_runtime.side_effect = _register
        td.iter_runtimes.side_effect = _iter_runtimes
        td.get_runtime.side_effect = _get_runtime
        return td

    def test_one_failed_does_not_abort_others(self):
        scope_a = _make_telescope_sensor("scope-a", connect=True)
        # scope-b's adapter returns False from connect(); the old serial
        # flow aborted everything when this happened.  The new flow
        # must mark scope-b as failed and still flip scope-a to connected.
        scope_b = _make_telescope_sensor("scope-b", connect=False)
        daemon = _make_daemon([scope_a, scope_b])

        td = self._setup_dispatcher(daemon)
        runtime_factory, stubs = _patched_runtimes()

        with (
            patch(
                "citrasense.sensors.telescope.runtime_builder.SensorRuntime",
                side_effect=runtime_factory,
            ),
            patch("citrasense.citrasense_daemon.TaskDispatcher", return_value=td),
            patch.object(daemon, "_initialize_safety_monitor"),
            patch("citrasense.sensors.telescope.runtime_builder.TelescopeRuntimeBuilder.connect_post_wiring"),
            patch.object(daemon, "_start_retention_timer"),
        ):
            success, error = daemon._initialize_telescopes()

        assert success is True
        assert error is None

        assert daemon._init_orchestrator is not None
        if daemon._init_orchestrator._init_thread is not None:
            daemon._init_orchestrator._init_thread.join(timeout=10)
            assert not daemon._init_orchestrator._init_thread.is_alive()

        assert stubs["scope-a"].init_state == "connected"
        assert stubs["scope-b"].init_state == "failed"
        assert stubs["scope-b"].init_error is not None

    def test_throwing_connect_marks_failed(self):
        scope_a = _make_telescope_sensor("scope-a", connect=True)
        scope_b = _make_telescope_sensor("scope-b", connect=RuntimeError("USB device missing"))
        daemon = _make_daemon([scope_a, scope_b])

        td = self._setup_dispatcher(daemon)
        runtime_factory, stubs = _patched_runtimes()

        with (
            patch(
                "citrasense.sensors.telescope.runtime_builder.SensorRuntime",
                side_effect=runtime_factory,
            ),
            patch("citrasense.citrasense_daemon.TaskDispatcher", return_value=td),
            patch.object(daemon, "_initialize_safety_monitor"),
            patch("citrasense.sensors.telescope.runtime_builder.TelescopeRuntimeBuilder.connect_post_wiring"),
            patch.object(daemon, "_start_retention_timer"),
        ):
            success, _ = daemon._initialize_telescopes()
        assert success is True

        assert daemon._init_orchestrator is not None
        if daemon._init_orchestrator._init_thread is not None:
            daemon._init_orchestrator._init_thread.join(timeout=10)

        assert stubs["scope-a"].init_state == "connected"
        assert stubs["scope-b"].init_state == "failed"
        assert "USB device missing" in (stubs["scope-b"].init_error or "")

    def test_hung_connect_times_out(self):
        # scope-b's connect blocks longer than the watchdog deadline.  We
        # set ``connect_timeout_seconds=0.5`` on its config so the
        # watchdog blows quickly without making the test slow.
        hang = threading.Event()

        def _hang() -> bool:
            hang.wait(timeout=5.0)
            return True

        scope_a = _make_telescope_sensor("scope-a", connect=True)
        scope_b = _make_telescope_sensor("scope-b", connect=_hang)
        daemon = _make_daemon([scope_a, scope_b])
        for sc in daemon.settings.sensors:
            if sc.id == "scope-b":
                sc.connect_timeout_seconds = 0.5

        td = self._setup_dispatcher(daemon)
        runtime_factory, stubs = _patched_runtimes()

        try:
            with (
                patch(
                    "citrasense.sensors.telescope.runtime_builder.SensorRuntime",
                    side_effect=runtime_factory,
                ),
                patch("citrasense.citrasense_daemon.TaskDispatcher", return_value=td),
                patch.object(daemon, "_initialize_safety_monitor"),
                patch("citrasense.sensors.telescope.runtime_builder.TelescopeRuntimeBuilder.connect_post_wiring"),
                patch.object(daemon, "_start_retention_timer"),
            ):
                start = time.monotonic()
                success, _ = daemon._initialize_telescopes()
                elapsed = time.monotonic() - start

            # _initialize_telescopes must return quickly even when one
            # sensor's connect is hung.  1.5s is generous for a 0.5s
            # watchdog plus thread-startup overhead.
            assert success is True
            assert elapsed < 1.5, f"_initialize_telescopes blocked for {elapsed:.2f}s"

            assert daemon._init_orchestrator is not None
            if daemon._init_orchestrator._init_thread is not None:
                daemon._init_orchestrator._init_thread.join(timeout=5.0)

            assert stubs["scope-a"].init_state == "connected"
            assert stubs["scope-b"].init_state == "timed_out"
            assert "timed out" in (stubs["scope-b"].init_error or "")
        finally:
            hang.set()

    def test_all_runtimes_registered_in_pending_before_connect(self):
        scope_a = _make_telescope_sensor("scope-a", connect=True)
        scope_b = _make_telescope_sensor("scope-b", connect=True)
        daemon = _make_daemon([scope_a, scope_b])

        td = self._setup_dispatcher(daemon)
        runtime_factory, _stubs = _patched_runtimes()

        # Capture the init_state at the moment register_runtime is
        # called.  The contract is that runtimes are registered before
        # the connect fan-out submits any work to the executor.
        captured: list[tuple[str, str]] = []

        def _capture_register(rt: Any) -> None:
            captured.append((rt.sensor_id, rt.init_state))
            td._runtimes[rt.sensor_id] = rt

        td.register_runtime.side_effect = _capture_register

        with (
            patch(
                "citrasense.sensors.telescope.runtime_builder.SensorRuntime",
                side_effect=runtime_factory,
            ),
            patch("citrasense.citrasense_daemon.TaskDispatcher", return_value=td),
            patch.object(daemon, "_initialize_safety_monitor"),
            patch("citrasense.sensors.telescope.runtime_builder.TelescopeRuntimeBuilder.connect_post_wiring"),
            patch.object(daemon, "_start_retention_timer"),
        ):
            daemon._initialize_telescopes()

        assert daemon._init_orchestrator is not None
        if daemon._init_orchestrator._init_thread is not None:
            daemon._init_orchestrator._init_thread.join(timeout=10)

        # Every sensor must have been registered while still in
        # ``pending`` — the dispatcher's poll/runner threads gate on
        # ``runtime.is_ready`` so a runtime not yet connected is just
        # silently skipped.
        assert {sid for sid, _ in captured} == {"scope-a", "scope-b"}
        assert all(state == "pending" for _, state in captured)


class TestPerSensorReconnect:
    """Exercise ``request_sensor_reconnect`` which delegates to the orchestrator."""

    def test_unknown_sensor_rejected_when_no_orchestrator(self):
        daemon = _make_daemon([])
        # No _init_orchestrator wired yet — the daemon facade should
        # reject the request cleanly instead of raising.
        ok, err = daemon.request_sensor_reconnect("ghost")
        assert ok is False
        assert err == "Daemon not initialized"

    def test_unknown_sensor_returns_404_payload(self):
        daemon = _make_daemon([])
        td = MagicMock()
        td.get_runtime.return_value = None
        daemon.task_dispatcher = td

        from citrasense.sensors.init_orchestrator import SensorInitOrchestrator

        daemon._init_orchestrator = SensorInitOrchestrator(
            logger=MagicMock(),
            web_server=daemon.web_server,
            sensor_manager=daemon.sensor_manager,  # type: ignore[arg-type]
            settings=daemon.settings,
            task_dispatcher=td,
        )

        ok, err = daemon.request_sensor_reconnect("ghost")
        assert ok is False
        assert err is not None
        assert err.startswith("Unknown sensor")

    def test_reconnect_already_in_flight_rejected(self):
        scope = _make_telescope_sensor("scope-a", connect=True)
        daemon = _make_daemon([scope])
        runtime = _RuntimeStub(scope)

        td = MagicMock()
        td.get_runtime.return_value = runtime
        daemon.task_dispatcher = td

        from citrasense.sensors.init_orchestrator import SensorInitOrchestrator

        orch = SensorInitOrchestrator(
            logger=MagicMock(),
            web_server=daemon.web_server,
            sensor_manager=daemon.sensor_manager,  # type: ignore[arg-type]
            settings=daemon.settings,
            task_dispatcher=td,
        )
        # Pretend the telescope builder already ran so reconnect can
        # find a builder for "scope-a".
        orch._builders["scope-a"] = MagicMock()

        # Pre-seed an unfinished future so the second request is
        # rejected without racing two adapter inits.
        pending = MagicMock()
        pending.done.return_value = False
        orch._futures["scope-a"] = pending

        daemon._init_orchestrator = orch

        ok, err = daemon.request_sensor_reconnect("scope-a")
        assert ok is False
        assert err is not None
        assert "already in flight" in err
