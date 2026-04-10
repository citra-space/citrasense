"""WebSocket event listener for NINA Advanced API.

Maintains a persistent connection to ws://<host>:1888/v2/socket and
dispatches incoming events to threading.Event signals and optional callbacks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect


def derive_ws_url(api_url: str) -> str:
    """Convert a NINA REST API URL to the WebSocket /socket endpoint.

    Example: 'http://nina:1888/v2/api' -> 'ws://nina:1888/v2/socket'
    """
    parsed = urlparse(api_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    # Strip trailing /api (or /api/) and append /socket
    path = parsed.path.rstrip("/")
    if path.endswith("/api"):
        path = path[: -len("/api")]
    return f"{scheme}://{parsed.hostname}:{parsed.port}{path}/socket"


class NinaEventListener:
    """Background thread that listens to NINA WebSocket events.

    Provides threading.Event signals that adapter methods can .clear() before
    issuing a command, then .wait(timeout=...) on for instant reaction.
    """

    RECONNECT_BASE_SECONDS = 1.0
    RECONNECT_MAX_SECONDS = 30.0
    RECV_TIMEOUT_SECONDS = 5.0

    def __init__(self, ws_url: str, logger: logging.Logger):
        self._ws_url = ws_url
        self._logger = logger
        self._thread: threading.Thread | None = None
        self._running = False

        # Event signals — adapter clears before command, waits after
        self.sequence_finished = threading.Event()
        self.sequence_failed = threading.Event()
        self.autofocus_finished = threading.Event()
        self.autofocus_error = threading.Event()
        self.filter_changed = threading.Event()
        self.image_saved = threading.Event()

        # Last-event payloads (guarded by _data_lock)
        self._data_lock = threading.Lock()
        self._last_filter_change: dict[str, Any] | None = None
        self._last_image_save: dict[str, Any] | None = None
        self._last_sequence_error: dict[str, Any] | None = None
        self._last_af_point_time: float = 0.0

        # Optional callbacks set by the adapter
        self.on_af_point: Callable[[int, float], None] | None = None
        self.on_image_save: Callable[[dict[str, Any]], None] | None = None

    # -- public data accessors (thread-safe) --

    @property
    def last_filter_change(self) -> dict[str, Any] | None:
        with self._data_lock:
            return dict(self._last_filter_change) if self._last_filter_change else None

    @property
    def last_image_save(self) -> dict[str, Any] | None:
        with self._data_lock:
            return dict(self._last_image_save) if self._last_image_save else None

    @property
    def last_sequence_error(self) -> dict[str, Any] | None:
        with self._data_lock:
            return dict(self._last_sequence_error) if self._last_sequence_error else None

    @property
    def last_af_point_time(self) -> float:
        with self._data_lock:
            return self._last_af_point_time

    @last_af_point_time.setter
    def last_af_point_time(self, value: float) -> None:
        with self._data_lock:
            self._last_af_point_time = value

    # -- lifecycle --

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="nina-ws-listener", daemon=True)
        self._thread.start()
        self._logger.info(f"NINA WebSocket listener started ({self._ws_url})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        self._logger.info("NINA WebSocket listener stopped")

    @property
    def connected(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # -- internal listener loop --

    def _run(self):
        backoff = self.RECONNECT_BASE_SECONDS
        while self._running:
            try:
                self._logger.debug(f"Connecting to NINA WebSocket at {self._ws_url} ...")
                with ws_connect(self._ws_url, open_timeout=10, close_timeout=5) as ws:
                    self._logger.info("NINA WebSocket connected")
                    backoff = self.RECONNECT_BASE_SECONDS
                    while self._running:
                        try:
                            raw = ws.recv(timeout=self.RECV_TIMEOUT_SECONDS)
                        except TimeoutError:
                            continue
                        try:
                            msg = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            self._logger.debug(f"Non-JSON WS message: {raw!r:.200}")
                            continue
                        self._dispatch(msg)
            except ConnectionClosed as e:
                if self._running:
                    self._logger.warning(f"NINA WebSocket closed: {e}. Reconnecting in {backoff:.0f}s ...")
            except Exception as e:
                if self._running:
                    self._logger.warning(f"NINA WebSocket error: {e}. Reconnecting in {backoff:.0f}s ...")
            if self._running:
                time.sleep(backoff)
                backoff = min(backoff * 2, self.RECONNECT_MAX_SECONDS)

    def _dispatch(self, msg: dict):
        """Route an incoming WebSocket message to the right signal/callback."""
        response = msg.get("Response")
        if not isinstance(response, dict):
            return

        event = response.get("Event")
        if not event:
            return

        self._logger.debug(f"NINA WS event: {event}")

        if event == "SEQUENCE-FINISHED":
            self.sequence_finished.set()

        elif event == "SEQUENCE-ENTITY-FAILED":
            with self._data_lock:
                self._last_sequence_error = response
            self.sequence_failed.set()

        elif event == "AUTOFOCUS-FINISHED":
            self.autofocus_finished.set()

        elif event in ("ERROR-AF", "AUTOFOCUS-ERROR"):
            self.autofocus_error.set()

        elif event == "AUTOFOCUS-POINT-ADDED":
            with self._data_lock:
                self._last_af_point_time = time.time()
            stats = response.get("ImageStatistics") or response
            position = stats.get("Position")
            hfr = stats.get("HFR")
            if self.on_af_point and position is not None and hfr is not None:
                try:
                    self.on_af_point(int(position), float(hfr))
                except Exception as e:
                    self._logger.debug(f"on_af_point callback error: {e}")

        elif event == "FILTERWHEEL-CHANGED":
            with self._data_lock:
                self._last_filter_change = response
            self.filter_changed.set()

        elif event == "IMAGE-SAVE":
            stats = response.get("ImageStatistics", {})
            with self._data_lock:
                self._last_image_save = stats
            self.image_saved.set()
            if self.on_image_save:
                try:
                    self.on_image_save(stats)
                except Exception as e:
                    self._logger.debug(f"on_image_save callback error: {e}")
