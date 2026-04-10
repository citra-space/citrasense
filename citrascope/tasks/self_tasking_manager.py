"""Self-tasking manager: periodically requests work from the server.

Calls ``POST /collection-requests/batch`` every ``_REQUEST_INTERVAL_SECONDS``
while the observing session is in the ``OBSERVING`` state.  The server
discovers visible satellites and creates tasks; the normal ``poll_tasks``
loop picks them up on the next cycle.

No queue-depth gating — we request on a fixed interval regardless of
pending task count.  The server handles deduplication.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from citrascope.location.twilight import ObservingWindow
from citrascope.tasks.observing_session import SessionState

if TYPE_CHECKING:
    from citrascope.api.abstract_api_client import AbstractCitraApiClient
    from citrascope.settings.citrascope_settings import CitraScopeSettings

_REQUEST_INTERVAL_SECONDS = 300  # 5 minutes


class SelfTaskingManager:
    """Requests batch collection requests from the server on a timer.

    Gating (all must be true):
    1. ``self_tasking_enabled`` in settings
    2. Session state is ``OBSERVING``
    3. At least ``_REQUEST_INTERVAL_SECONDS`` since last request
    4. Observing window is available with a valid ``dark_end``
    """

    def __init__(
        self,
        api_client: AbstractCitraApiClient,
        settings: CitraScopeSettings,
        logger: logging.Logger,
        ground_station_id: str,
        sensor_id: str,
        get_session_state: Callable[[], SessionState],
        get_observing_window: Callable[[], ObservingWindow | None],
    ):
        self._api_client = api_client
        self._settings = settings
        self._logger = logger
        self._ground_station_id = ground_station_id
        self._sensor_id = sensor_id
        self._get_session_state = get_session_state
        self._get_observing_window = get_observing_window

        self._last_request_time: float = 0.0  # monotonic, for throttle
        self._last_request_epoch: float | None = None  # wall-clock, for display
        self._last_request_created: int | None = None

    def check_and_request(self) -> None:
        """Evaluate conditions and request work if met.  Call from poll loop."""
        if not self._settings.self_tasking_enabled:
            return

        if self._get_session_state() != SessionState.OBSERVING:
            return

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _REQUEST_INTERVAL_SECONDS and self._last_request_time > 0:
            return

        window = self._get_observing_window()
        if window is None or window.dark_end is None:
            return

        self._request_work(window, window.dark_end)

    def _request_work(self, window: ObservingWindow, dark_end: str) -> None:
        """Build and send the batch collection request."""
        settings = self._settings

        group_ids = settings.self_tasking_satellite_group_ids or None
        exclude_types = settings.self_tasking_exclude_object_types or None
        orbit_regimes = settings.self_tasking_include_orbit_regimes or None
        collection_type = settings.self_tasking_collection_type or "Track"

        self._logger.info(
            "Self-tasking: requesting batch work (type=%s, window %s → %s, gs=%s)",
            collection_type,
            window.dark_start,
            dark_end,
            self._ground_station_id,
        )

        try:
            result = self._api_client.create_batch_collection_requests(
                window_start=window.dark_start or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                window_stop=dark_end,
                ground_station_id=self._ground_station_id,
                sensor_id=self._sensor_id,
                discover_visible=not bool(group_ids),
                satellite_group_ids=group_ids,
                request_type=collection_type,
                exclude_types=exclude_types,
                include_orbit_regimes=orbit_regimes,
            )
            self._last_request_time = time.monotonic()
            self._last_request_epoch = time.time()
            if result is not None:
                created = result.get("created", "?")
                self._last_request_created = created if isinstance(created, int) else None
                self._logger.info("Self-tasking: batch request succeeded (created=%s)", created)
            else:
                self._logger.warning("Self-tasking: batch request returned None — will retry next interval")
        except Exception:
            self._last_request_time = time.monotonic()
            self._last_request_epoch = time.time()
            self._logger.error("Self-tasking: batch request failed", exc_info=True)

    def status_dict(self) -> dict[str, Any]:
        """Build a dict for the web status broadcast."""
        next_request_seconds: float | None = None
        if self._last_request_time > 0:
            elapsed = time.monotonic() - self._last_request_time
            remaining = _REQUEST_INTERVAL_SECONDS - elapsed
            next_request_seconds = max(0.0, remaining)
        return {
            "last_batch_request": self._last_request_epoch,
            "last_batch_created": self._last_request_created,
            "next_request_seconds": next_request_seconds,
        }
