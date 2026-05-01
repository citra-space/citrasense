"""Unit tests for :class:`SensorInitOrchestrator`.

These tests exercise the orchestrator in isolation — no daemon, no
real builders — so failures point straight at the orchestration
layer instead of the daemon glue.

Coverage targets:

* Watchdog timeout flips a hung connect to ``timed_out`` quickly,
  and crucially returns *quickly* — without the bug fix, the
  ``with ThreadPoolExecutor`` exit would ``join()`` the hung connect
  thread and silently undo the watchdog.
* ``adapter.connect()`` returning False marks the runtime ``failed``.
* A raising ``connect()`` also marks ``failed`` (with the exception
  message).
* A successful connect runs the builder's ``connect_post_wiring`` and
  flips the runtime to ``connected``.
* Reconnect requests for a sensor with an in-flight future are
  rejected with a 409-shaped ``(False, "...")`` payload.
* A slow ``disconnect()`` on one sensor doesn't block reconnect
  requests for unrelated sensors (lock-scope regression).
* ``shutdown()`` mid-init prevents workers from resurrecting torn-down
  runtimes via ``mark_connected``.
* The toast callback factory returns ``None`` when no web server is
  configured and a real callback otherwise.
"""

from __future__ import annotations

import logging
import threading
import time
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

    def test_watchdog_returns_quickly_when_connect_hangs(self):
        """Regression: the inner executor must not block ``_run_sensor_init`` past the timeout.

        Earlier code wrapped the inner executor in
        ``with ThreadPoolExecutor(...) as inner:``, whose ``__exit__``
        calls ``shutdown(wait=True)`` and ``join()``s the worker
        thread.  That join silently waits for the hung ``connect()``
        to return, defeating the watchdog timeout — operators reported
        a "timed out" status only minutes after the deadline because
        the connect thread happened to give up eventually.

        Pin the contract: the watchdog must surface ``timed_out`` and
        return well before the simulated hang would naturally end.
        We use a 0.2s timeout against a connect that "hangs" for 30s
        (effectively forever for the test) and assert the worker
        function returns in under 2s.
        """
        orch = _make_orchestrator(timeout=0.2)
        hang = threading.Event()

        def _hang() -> bool:
            hang.wait(timeout=30.0)
            return True

        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=_hang))
        builder = _FakeBuilder()

        start = time.monotonic()
        try:
            orch._run_sensor_init(runtime, builder)  # type: ignore[arg-type]
        finally:
            hang.set()
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"watchdog returned in {elapsed:.2f}s — timeout was bypassed"
        assert runtime.init_state == "timed_out"

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

    def test_slow_disconnect_on_sensor_a_does_not_block_sensor_b(self):
        """Regression: a hung disconnect on one sensor must not starve others.

        Earlier code held ``_futures_lock`` across
        ``runtime.sensor.disconnect()``.  A USB driver deadlock on
        sensor A would therefore freeze every other sensor's
        reconnect request — exactly the failure mode #339 was trying
        to fix.  This test makes sensor A's disconnect block, then
        fires sensor B's reconnect from the main thread and asserts
        it returns promptly.
        """
        orch = _make_orchestrator(timeout=5.0)

        a_started = threading.Event()
        a_release = threading.Event()

        def _a_disconnect() -> None:
            a_started.set()
            a_release.wait(timeout=10.0)

        sensor_a = _fake_sensor(connect=True)
        sensor_a.disconnect.side_effect = _a_disconnect
        runtime_a = _FakeRuntime("scope-a", sensor=sensor_a)
        runtime_b = _FakeRuntime("scope-b", sensor=_fake_sensor(connect=True))

        runtimes = {"scope-a": runtime_a, "scope-b": runtime_b}
        orch.task_dispatcher.get_runtime.side_effect = lambda sid: runtimes.get(sid)  # type: ignore[union-attr]
        orch._builders["scope-a"] = _FakeBuilder()  # type: ignore[assignment]
        orch._builders["scope-b"] = _FakeBuilder()  # type: ignore[assignment]

        a_thread = threading.Thread(
            target=orch.request_reconnect,
            args=("scope-a",),
            name="test-reconnect-a",
            daemon=True,
        )
        a_thread.start()
        try:
            assert a_started.wait(timeout=2.0), "sensor-a disconnect never started"

            start = time.monotonic()
            ok, err = orch.request_reconnect("scope-b")
            elapsed = time.monotonic() - start

            assert ok is True, f"sensor-b reconnect rejected: {err}"
            assert err is None
            assert elapsed < 0.5, (
                f"sensor-b reconnect blocked for {elapsed:.2f}s while sensor-a "
                "disconnect was hung — _futures_lock is still held during disconnect"
            )
        finally:
            a_release.set()
            a_thread.join(timeout=3.0)
            orch.shutdown()


class TestShutdownGate:
    """Coverage for the shutdown-event checkpoints in ``_run_sensor_init``.

    Without these gates, a slow ``connect()`` that completed *after*
    ``TaskDispatcher.stop()`` had already torn down a runtime would
    still call ``runtime.mark_connected()`` — which restarts the queue
    trio, the bus subscription, and the streaming producer on a
    runtime nobody is supposed to be using anymore.  That manifested
    as zombie capture loops surviving a daemon reload.
    """

    def test_shutdown_during_init_prevents_mark_connected(self):
        """``shutdown()`` mid-connect must stop the worker before mark_connected fires.

        Drive ``_run_sensor_init`` on a thread.  The fake sensor's
        ``connect()`` blocks on an event — that lets us slip
        ``shutdown()`` in between gate 1 (``mark_connecting`` already
        fired) and gate 3 (right before ``mark_connected``).  After
        we release the connect, the worker must observe the shutdown
        flag and return without flipping the runtime to ``connected``.
        """
        orch = _make_orchestrator(timeout=10.0)

        connect_can_proceed = threading.Event()

        def _slow_connect() -> bool:
            # Simulate a slow connect that finishes only after we've
            # already called shutdown().  In production this is a
            # USB or HTTP handshake that takes several seconds.
            connect_can_proceed.wait(timeout=5.0)
            return True

        runtime = _FakeRuntime("scope-a", sensor=_fake_sensor(connect=_slow_connect))
        builder = _FakeBuilder()

        worker_thread = threading.Thread(
            target=orch._run_sensor_init,
            args=(runtime, builder),
            name="test-init-worker",
            daemon=True,
        )
        worker_thread.start()
        try:
            # Wait until the worker has flipped the runtime to
            # ``connecting`` so we know the watchdog has already
            # submitted the inner connect future.
            for _ in range(50):
                if runtime.init_state == "connecting":
                    break
                time.sleep(0.01)
            assert (
                runtime.init_state == "connecting"
            ), f"worker never reached 'connecting' state (was {runtime.init_state})"

            # Tear-down arrives *while the connect is still running*.
            orch.shutdown()
        finally:
            connect_can_proceed.set()
            worker_thread.join(timeout=3.0)

        assert not worker_thread.is_alive(), "init worker did not exit after shutdown"
        # The crucial assertion: even though connect() ultimately
        # returned True, the worker must NOT have run mark_connected
        # (which resurrects queues/streams on a torn-down runtime).
        assert (
            runtime.init_state != "connected"
        ), f"runtime was resurrected to 'connected' after shutdown (transitions: {runtime.transitions})"
        states = [state for state, _ in runtime.transitions]
        assert "connected" not in states
        # Post-wiring is also gated, so a successful connect followed
        # by shutdown should leave post_wiring untouched.
        assert builder.post_wiring_calls == []

    def test_fan_out_clears_shutdown_event(self):
        """A fresh fan_out after shutdown must un-stick the gate.

        Daemon reloads call ``shutdown()`` then ``fan_out()`` again
        with the new runtime set.  If the shutdown event stayed set,
        every worker submitted on the new executor would bail at
        gate 1 and no sensor would ever connect.
        """
        orch = _make_orchestrator()
        orch._shutdown_event.set()

        # Build one (runtime, builder) pair so fan_out has work to do.
        runtime = MagicMock()
        runtime.sensor_id = "scope-a"
        builder = _FakeBuilder()
        # Make _run_sensor_init a no-op so the test doesn't actually
        # exercise the connect path — we only care that fan_out
        # cleared the event before submitting.
        orch._run_sensor_init = MagicMock()  # type: ignore[method-assign]

        try:
            orch.fan_out([(runtime, builder)])  # type: ignore[list-item]
            assert not orch._shutdown_event.is_set()
        finally:
            orch.shutdown()


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
