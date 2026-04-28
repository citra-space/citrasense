"""NATS-backed :class:`DetectionSource` for ``pr_sensor``.

Owns an asyncio loop running in a dedicated daemon thread so the
synchronous, thread-driven :class:`PassiveRadarSensor` can hand off
message delivery without inheriting the asyncio world itself.

Subject hierarchy (see ``passive_radar/docs/NATS_ARCHITECTURE.md`` and
``passive_radar/README.md`` for the authoritative list — only the
subjects implemented in ``pr_sensor``/``pr_nats_client`` are used here):

Subscribes under ``radar.sensor.{id}.*``::

    observations  status  health  stations  error

Plus the fleet registry subjects ``radar.registry.announce`` /
``radar.registry.depart`` — both filtered by ``sensor_id`` inside the
dispatcher so this source only surfaces announces / departs for its
configured sensor.

Publishes (request-reply) under ``radar.control.{id}.*``::

    start  stop  ping  config.set  config.get

``nats-py`` handles reconnection natively; callers observe transport
state via :meth:`is_connected`.  Malformed JSON is logged at warning
and dropped rather than blowing up the dispatcher thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from nats.aio.client import Client as NatsClient  # type: ignore[reportMissingImports]
from nats.aio.msg import Msg as NatsMsg  # type: ignore[reportMissingImports]
from nats.aio.subscription import Subscription as NatsSubscription  # type: ignore[reportMissingImports]
from nats.errors import NoRespondersError  # type: ignore[reportMissingImports]
from nats.errors import TimeoutError as NatsTimeoutError  # type: ignore[reportMissingImports]

from citrasense.sensors.radar.detection_source import MessageHandler

_SENTINEL: Any = object()


class NatsDetectionSource:
    """NATS subscriber for a single ``pr_sensor`` instance.

    Parameters
    ----------
    nats_url:
        URL of the NATS server (``nats://host:4222``).
    sensor_id:
        The ``--sensor-id`` the target ``pr_sensor`` was started with.
        Used to build the ``radar.sensor.{id}.*`` and
        ``radar.control.{id}.*`` subject namespaces.
    logger:
        Logger to attribute parse / reconnect / dispatch warnings to.
        Should be a sensor-scoped logger so records carry
        ``sensor_id`` in their ``extra`` payload for the web log
        filter.
    connect_timeout:
        Seconds to wait for the initial NATS connection inside
        :meth:`start` before giving up.  Reconnect attempts after the
        initial connect happen in the background and are unbounded.
    """

    #: Default reconnect backoff lower bound (seconds).  ``nats-py``
    #: stubs type this as ``int``; we keep the constant as an ``int`` to
    #: satisfy the type checker.
    _RECONNECT_TIME_WAIT: int = 2
    #: Default maximum reconnect attempts (-1 = forever).  ``pr_sensor``
    #: outages shouldn't take citrasense down, so we reconnect forever.
    _MAX_RECONNECT_ATTEMPTS: int = -1

    def __init__(
        self,
        *,
        nats_url: str,
        sensor_id: str,
        logger: logging.Logger | logging.LoggerAdapter,
        connect_timeout: float = 10.0,
    ) -> None:
        self._nats_url = nats_url
        self._sensor_id = sensor_id
        # ``LoggerAdapter`` forwards ``getChild`` at runtime but pyright
        # only sees it on ``Logger`` — the cast keeps the type checker
        # from complaining without narrowing away adapter use.
        child_name = f"NatsDetectionSource[{sensor_id}]"
        if isinstance(logger, logging.Logger):
            self._logger: logging.Logger | logging.LoggerAdapter = logger.getChild(child_name)
        else:
            self._logger = logger
        self._connect_timeout = connect_timeout

        # Asyncio plumbing
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._nc: NatsClient | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._running = False
        self._subscriptions: list[NatsSubscription] = []

        # Handlers (mutable across restarts)
        self._handlers_lock = threading.RLock()
        self._handlers: dict[str, MessageHandler | None] = {
            "on_observation": None,
            "on_detection": None,
            "on_status": None,
            "on_health": None,
            "on_stations": None,
            "on_error": None,
            "on_announce": None,
            "on_depart": None,
        }

        # Staleness tracking — monotonic seconds since the last status
        # message.  Written on the asyncio thread, read by any thread
        # through :meth:`is_stream_stale`.
        self._last_status_monotonic: float | None = None

    # ── Subject names ──────────────────────────────────────────────────

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    def _sensor_subject(self, suffix: str) -> str:
        return f"radar.sensor.{self._sensor_id}.{suffix}"

    def _control_subject(self, suffix: str) -> str:
        return f"radar.control.{self._sensor_id}.{suffix}"

    # ── Handler management ────────────────────────────────────────────

    def set_handlers(
        self,
        *,
        on_observation: MessageHandler | None = _SENTINEL,
        on_detection: MessageHandler | None = _SENTINEL,
        on_status: MessageHandler | None = _SENTINEL,
        on_health: MessageHandler | None = _SENTINEL,
        on_stations: MessageHandler | None = _SENTINEL,
        on_error: MessageHandler | None = _SENTINEL,
        on_announce: MessageHandler | None = _SENTINEL,
        on_depart: MessageHandler | None = _SENTINEL,
    ) -> None:
        """Replace zero or more handlers at runtime.

        Pass a handler (or ``None`` to clear one) to mutate it; leave an
        argument at its sentinel default to leave it untouched.
        """
        updates = {
            "on_observation": on_observation,
            "on_detection": on_detection,
            "on_status": on_status,
            "on_health": on_health,
            "on_stations": on_stations,
            "on_error": on_error,
            "on_announce": on_announce,
            "on_depart": on_depart,
        }
        with self._handlers_lock:
            for name, value in updates.items():
                if value is _SENTINEL:
                    continue
                self._handlers[name] = value

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(
        self,
        *,
        on_observation: MessageHandler | None = None,
        on_detection: MessageHandler | None = None,
        on_status: MessageHandler | None = None,
        on_health: MessageHandler | None = None,
        on_stations: MessageHandler | None = None,
        on_error: MessageHandler | None = None,
        on_announce: MessageHandler | None = None,
        on_depart: MessageHandler | None = None,
    ) -> None:
        if self._running:
            self.set_handlers(
                on_observation=on_observation,
                on_detection=on_detection,
                on_status=on_status,
                on_health=on_health,
                on_stations=on_stations,
                on_error=on_error,
                on_announce=on_announce,
                on_depart=on_depart,
            )
            return

        with self._handlers_lock:
            self._handlers["on_observation"] = on_observation
            self._handlers["on_detection"] = on_detection
            self._handlers["on_status"] = on_status
            self._handlers["on_health"] = on_health
            self._handlers["on_stations"] = on_stations
            self._handlers["on_error"] = on_error
            self._handlers["on_announce"] = on_announce
            self._handlers["on_depart"] = on_depart

        self._ready.clear()
        self._start_error = None
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"nats-radar[{self._sensor_id}]",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=self._connect_timeout):
            self._running = False
            raise ConnectionError(f"NATS connect to {self._nats_url} timed out after {self._connect_timeout:.1f}s")
        if self._start_error is not None:
            self._running = False
            err = self._start_error
            self._start_error = None
            raise err
        self._logger.info("Connected to NATS at %s", self._nats_url)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        loop = self._loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            try:
                future.result(timeout=5.0)
            except Exception as exc:
                self._logger.warning("Shutdown coroutine raised: %s", exc)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._nc = None
        self._subscriptions.clear()
        self._logger.info("NATS detection source stopped")

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def is_connected(self) -> bool:
        nc = self._nc
        return bool(nc and nc.is_connected)

    # ── Staleness ─────────────────────────────────────────────────────

    def is_stream_stale(self, max_age_s: float) -> bool:
        ts = self._last_status_monotonic
        if ts is None:
            return True
        return (time.monotonic() - ts) > max_age_s

    def seconds_since_status(self) -> float | None:
        ts = self._last_status_monotonic
        if ts is None:
            return None
        return time.monotonic() - ts

    # ── Request / reply ───────────────────────────────────────────────

    def send_command(
        self,
        suffix: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if not self.is_running():
            raise ConnectionError("Detection source is not running")
        loop = self._loop
        if loop is None:
            raise ConnectionError("Detection source event loop not available")
        subject = self._control_subject(suffix)
        body = json.dumps(payload or {}).encode("utf-8")

        async def _do() -> dict[str, Any]:
            assert self._nc is not None
            msg = await self._nc.request(subject, body, timeout=timeout)
            if not msg.data:
                return {}
            try:
                result = json.loads(msg.data.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Non-JSON reply from {subject!r}: {exc}; body={msg.data!r}") from exc
            if not isinstance(result, dict):
                raise ValueError(f"Unexpected reply shape from {subject!r}: {result!r}")
            return result

        future = asyncio.run_coroutine_threadsafe(_do(), loop)
        try:
            return future.result(timeout=timeout + 1.0)
        except NatsTimeoutError as exc:
            raise TimeoutError(f"NATS request {subject!r} timed out") from exc
        except NoRespondersError as exc:
            raise ConnectionError(f"No responders for {subject!r} — pr_sensor not reachable") from exc

    # ── Internal event loop ───────────────────────────────────────────

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._main())
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()
            self._logger.error("NATS loop terminated: %s", exc, exc_info=True)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _main(self) -> None:
        nc = NatsClient()
        self._nc = nc
        try:
            await nc.connect(
                servers=[self._nats_url],
                reconnect_time_wait=self._RECONNECT_TIME_WAIT,
                max_reconnect_attempts=self._MAX_RECONNECT_ATTEMPTS,
                disconnected_cb=self._on_disconnected,
                reconnected_cb=self._on_reconnected,
                error_cb=self._on_error_cb,
                # ``nats-py`` stubs type ``connect_timeout`` as ``int``;
                # fractional seconds aren't useful here so round up.
                connect_timeout=int(max(1, round(self._connect_timeout))),
            )
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            return

        try:
            await self._subscribe_all()
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            await self._shutdown()
            return

        self._ready.set()
        while self._running:
            await asyncio.sleep(0.25)

    async def _subscribe_all(self) -> None:
        assert self._nc is not None
        patterns: list[tuple[str, Callable[[NatsMsg], Awaitable[None]]]] = [
            (self._sensor_subject("observations"), self._make_dispatch("on_observation")),
            (self._sensor_subject("detections"), self._make_dispatch("on_detection")),
            (self._sensor_subject("status"), self._make_status_dispatch()),
            (self._sensor_subject("health"), self._make_dispatch("on_health")),
            (self._sensor_subject("stations"), self._make_dispatch("on_stations")),
            (self._sensor_subject("error"), self._make_dispatch("on_error")),
            ("radar.registry.announce", self._make_registry_dispatch("on_announce")),
            ("radar.registry.depart", self._make_registry_dispatch("on_depart")),
        ]
        for subject, cb in patterns:
            sub = await self._nc.subscribe(subject, cb=cb)
            self._subscriptions.append(sub)
        self._logger.debug("Subscribed to %d subjects for sensor %s", len(patterns), self._sensor_id)

    async def _shutdown(self) -> None:
        # Best-effort unsubscribe first so the client stops delivering
        # callbacks while we close the connection.
        for sub in self._subscriptions:
            try:
                await sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()
        nc = self._nc
        if nc is None:
            return
        try:
            if not nc.is_closed:
                await nc.drain()
        except Exception as exc:
            self._logger.debug("NATS drain raised: %s", exc)
            try:
                await nc.close()
            except Exception:
                pass

    # ── Dispatch helpers ──────────────────────────────────────────────

    def _make_dispatch(self, handler_name: str) -> Callable[[NatsMsg], Awaitable[None]]:
        async def _cb(msg: NatsMsg) -> None:
            self._deliver(handler_name, msg)

        return _cb

    def _make_status_dispatch(self) -> Callable[[NatsMsg], Awaitable[None]]:
        async def _cb(msg: NatsMsg) -> None:
            # Stamp staleness regardless of handler decode success —
            # we still know the sensor produced a heartbeat.
            self._last_status_monotonic = time.monotonic()
            self._deliver("on_status", msg)

        return _cb

    def _make_registry_dispatch(self, handler_name: str) -> Callable[[NatsMsg], Awaitable[None]]:
        sensor_id = self._sensor_id

        async def _cb(msg: NatsMsg) -> None:
            payload = self._decode(msg)
            if payload is None:
                return
            if payload.get("sensor_id") != sensor_id:
                return
            with self._handlers_lock:
                handler = self._handlers.get(handler_name)
            if handler is None:
                return
            try:
                handler(payload)
            except Exception as exc:
                self._logger.warning("Handler %s raised on %s: %s", handler_name, msg.subject, exc, exc_info=True)

        return _cb

    def _deliver(self, handler_name: str, msg: NatsMsg) -> None:
        payload = self._decode(msg)
        if payload is None:
            return
        with self._handlers_lock:
            handler = self._handlers.get(handler_name)
        if handler is None:
            return
        try:
            handler(payload)
        except Exception as exc:
            self._logger.warning("Handler %s raised on %s: %s", handler_name, msg.subject, exc, exc_info=True)

    def _decode(self, msg: NatsMsg) -> dict[str, Any] | None:
        try:
            parsed = json.loads(msg.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._logger.warning("Malformed message on %s: %s", msg.subject, exc)
            return None
        if not isinstance(parsed, dict):
            self._logger.warning("Unexpected payload shape on %s: %s (not a dict)", msg.subject, type(parsed).__name__)
            return None
        return parsed

    # ── NATS client callbacks ────────────────────────────────────────

    async def _on_disconnected(self) -> None:
        self._logger.warning("Disconnected from NATS at %s", self._nats_url)

    async def _on_reconnected(self) -> None:
        self._logger.info("Reconnected to NATS at %s", self._nats_url)

    async def _on_error_cb(self, err: Exception) -> None:
        self._logger.debug("NATS client error: %s", err)
