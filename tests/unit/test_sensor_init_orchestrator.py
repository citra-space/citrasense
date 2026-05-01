"""Unit tests for :class:`SensorInitOrchestrator`.

These tests exercise the orchestrator in isolation — no daemon, no
real builders — so failures point straight at the orchestration
layer instead of the daemon glue.

Coverage targets:

* Watchdog timeout flips a hung connect to ``timed_out`` quickly.
* ``adapter.connect()`` returning False marks the runtime ``failed``.
* A raising ``connect()`` also marks ``failed`` (with the exception
  message).
* A successful connect runs the builder's ``connect_post_wiring`` and
  flips the runtime to ``connected``.
* Reconnect requests for a sensor with an in-flight future are
  rejected with a 409-shaped ``(False, "...")`` payload.
* The toast callback factory returns ``None`` when no web server is
  configured and a real callback otherwise.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import MagicMock

from citrasense.sensors.init_orchestrator import SensorInitOrchestrator


class _FakeRuntime:
    """Minimal SensorRuntime double — mark_* + sensor + sensor_id."""

    def __init__(self, sensor_id: str, *, sensor: Any) -> None:
        self.sensor = sensor
        self.sensor_id = sensor_id
        self.init_state = "pending"
        self.init_error: str | None = None
        self.transitions: list[tuple[str, str | None]] = []
        self.on_init_state_change: Any = None

    @property
    def is_ready(self) -> bool:
        return self.init_state == "connected"

    def _set(self, state: str, err: str | None) -> None:
        self.init_state = state
        self.init_error = err
        self.transitions.append((state, err))

    def mark_connecting(self) -> None:
        self._set("connecting", None)

    def mark_connected(self) -> None:
        self._set("connected", None)

    def mark_failed(self, err: str) -> None:
        self._set("failed", err)

    def mark_timed_out(self, err: str) -> None:
        self._set("timed_out", err)

    def mark_disconnected(self) -> None:
        self._set("pending", None)


class _FakeBuilder:
    """RuntimeBuilder double for orchestrator tests."""

    def __init__(self, *, post_wiring_raises: Exception | None = None) -> None:
        self.post_wiring_raises = post_wiring_raises
        self.post_wiring_calls: list[Any] = []

    def build(self, sensor: Any) -> Any:  # pragma: no cover - orchestrator never calls build
        raise NotImplementedError

    def connect_post_wiring(self, runtime: Any) -> None:
        self.post_wiring_calls.append(runtime)
        if self.post_wiring_raises is not None:
            raise self.post_wiring_raises


def _make_orchestrator(*, web_server: Any = None, timeout: float = 5.0) -> SensorInitOrchestrator:
    settings = MagicMock()
    settings.get_sensor_config.return_value = MagicMock(connect_timeout_seconds=timeout)
    sensor_manager = MagicMock()
    sensor_manager.get_sensor.return_value = MagicMock(name="display-name")
    task_dispatcher = MagicMock()
    task_dispatcher.iter_runtimes.return_value = []
    task_dispatcher.get_runtime.return_value = None
    return SensorInitOrchestrator(
        logger=logging.getLogger("test.orchestrator"),
        web_server=web_server,
        sensor_manager=sensor_manager,
        settings=settings,
        task_dispatcher=task_dispatcher,
    )


def _fake_sensor(*, connect: Any = True) -> MagicMock:
    s = MagicMock()
    s.adapter = MagicMock()

    def _connect_side_effect() -> bool:
        if isinstance(connect, BaseException) or (isinstance(connect, type) and issubclass(connect, BaseException)):
            raise connect  # type: ignore[misc]
        if callable(connect):
            return bool(connect())
        return bool(connect)

    s.connect.side_effect = _connect_side_effect
    return s


class TestRunSensorInit:
    def test_successful_connect_runs_post_wiring_and_marks_connected(self):
        orch = _make_orchestrator()
        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=True))
        builder = _FakeBuilder()

        orch._run_sensor_init(runtime, builder)  # type: ignore[arg-type]

        assert runtime.init_state == "connected"
        assert builder.post_wiring_calls == [runtime]
        assert ("connecting", None) in runtime.transitions
        assert ("connected", None) in runtime.transitions

    def test_connect_returns_false_marks_failed(self):
        orch = _make_orchestrator()
        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=False))
        builder = _FakeBuilder()

        orch._run_sensor_init(runtime, builder)  # type: ignore[arg-type]

        assert runtime.init_state == "failed"
        assert runtime.init_error is not None
        assert builder.post_wiring_calls == []

    def test_connect_raises_marks_failed_with_message(self):
        orch = _make_orchestrator()
        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=RuntimeError("nope")))
        builder = _FakeBuilder()

        orch._run_sensor_init(runtime, builder)  # type: ignore[arg-type]

        assert runtime.init_state == "failed"
        assert "nope" in (runtime.init_error or "")

    def test_hung_connect_marks_timed_out(self):
        # 0.3s watchdog so the test is fast.  The hung worker stays
        # alive on a daemon thread until the event is set.
        orch = _make_orchestrator(timeout=0.3)
        hang = threading.Event()

        def _hang() -> bool:
            hang.wait(timeout=5.0)
            return True

        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=_hang))
        builder = _FakeBuilder()

        try:
            orch._run_sensor_init(runtime, builder)  # type: ignore[arg-type]
        finally:
            hang.set()

        assert runtime.init_state == "timed_out"
        assert "timed out" in (runtime.init_error or "")
        assert builder.post_wiring_calls == []

    def test_post_wiring_failure_marks_failed(self):
        orch = _make_orchestrator()
        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=True))
        builder = _FakeBuilder(post_wiring_raises=RuntimeError("filter sync exploded"))

        orch._run_sensor_init(runtime, builder)  # type: ignore[arg-type]

        assert runtime.init_state == "failed"
        assert "filter sync exploded" in (runtime.init_error or "")


class TestRequestReconnect:
    def test_unknown_sensor_returns_404_payload(self):
        orch = _make_orchestrator()
        # No runtime registered for this id.
        ok, err = orch.request_reconnect("ghost")
        assert ok is False
        assert err is not None
        assert err.startswith("Unknown sensor")

    def test_in_flight_future_rejects_second_request(self):
        orch = _make_orchestrator()
        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=True))
        orch.task_dispatcher.get_runtime.return_value = runtime  # type: ignore[union-attr]

        builder = _FakeBuilder()
        orch._builders["scope-a"] = builder  # type: ignore[assignment]

        pending = MagicMock()
        pending.done.return_value = False
        orch._futures["scope-a"] = pending

        ok, err = orch.request_reconnect("scope-a")
        assert ok is False
        assert err is not None
        assert "already in flight" in err


class TestToastCallbackFactory:
    def test_returns_none_without_web_server(self):
        orch = _make_orchestrator(web_server=None)
        assert orch.make_init_state_toast_callback("scope-a") is None

    def test_emits_success_and_danger_toasts(self):
        ws = MagicMock()
        orch = _make_orchestrator(web_server=ws)
        cb = orch.make_init_state_toast_callback("scope-a")
        assert cb is not None

        cb("connecting", None)
        cb("connected", None)
        cb("failed", "boom")
        cb("timed_out", "stuck")

        kinds = [call.args[1] for call in ws.send_toast.call_args_list]
        assert kinds == ["info", "success", "danger", "danger"]
        # Toast id is shared so the UI dedupes successive transitions
        # for the same sensor.
        ids = {call.args[2] for call in ws.send_toast.call_args_list}
        assert ids == {"sensor-init-scope-a"}
