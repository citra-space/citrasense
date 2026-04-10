"""Observing session state machine for autonomous night operations.

Tracks sun altitude to determine when it is dark enough to observe, and
orchestrates startup actions (unpark, autofocus) and shutdown actions (drain
queues, park) based on configurable ``do_*`` switches.

The ``update()`` method is called from the ``TaskManager.poll_tasks`` loop
every 15 seconds.  It recomputes the session state and triggers actions
as needed — no additional threads.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from citrascope.location.twilight import ObservingWindow, compute_observing_window

_SHUTDOWN_TIMEOUT_SECONDS = 300  # 5 minutes — force-park if imaging hasn't drained

if TYPE_CHECKING:
    from citrascope.settings.citrascope_settings import CitraScopeSettings


class SessionState(Enum):
    DAYTIME = "daytime"
    NIGHT_STARTUP = "night_startup"
    OBSERVING = "observing"
    NIGHT_SHUTDOWN = "night_shutdown"


class ObservingSessionManager:
    """State machine that drives autonomous night operations.

    State transitions::

        DAYTIME → NIGHT_STARTUP    when observing_session_enabled and sun below threshold
        NIGHT_STARTUP → OBSERVING  when all enabled startup actions complete
        OBSERVING → NIGHT_SHUTDOWN when sun rises above threshold
        NIGHT_SHUTDOWN → DAYTIME   when imaging idle + park done, OR timeout (force-park)

    During NIGHT_SHUTDOWN the manager only waits for the **imaging queue** to
    drain (processing and upload don't use the mount).  A hard timeout
    (``_SHUTDOWN_TIMEOUT_SECONDS``) force-parks if imaging is stuck.

    The manager does not own any threads — it is driven by external calls to
    ``update()`` on the poll loop's cadence.
    """

    def __init__(
        self,
        settings: CitraScopeSettings,
        logger: logging.Logger,
        get_location: Callable[[], tuple[float, float] | None],
        request_autofocus: Callable[[], Any],
        is_autofocus_running: Callable[[], bool],
        is_imaging_idle: Callable[[], bool],
        are_queues_idle: Callable[[], bool],
        park_mount: Callable[[], bool] | None,
        unpark_mount: Callable[[], bool] | None,
        request_pointing_calibration: Callable[[], Any] | None = None,
        is_pointing_calibration_running: Callable[[], bool] | None = None,
    ):
        self._settings = settings
        self._logger = logger
        self._get_location = get_location
        self._request_autofocus = request_autofocus
        self._is_autofocus_running = is_autofocus_running
        self._is_imaging_idle = is_imaging_idle
        self._are_queues_idle = are_queues_idle
        self._park_mount = park_mount
        self._unpark_mount = unpark_mount
        self._request_pointing_calibration = request_pointing_calibration
        self._is_pointing_calibration_running = is_pointing_calibration_running

        self._state = SessionState.DAYTIME
        self._observing_window: ObservingWindow | None = None

        # Track which startup actions have been initiated/completed
        self._unpark_done = False
        self._pointing_calibration_requested = False
        self._autofocus_requested = False
        self._park_done = False
        self._shutdown_entered_at: float | None = None

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def observing_window(self) -> ObservingWindow | None:
        return self._observing_window

    def update(self) -> SessionState:
        """Recompute session state and trigger actions.  Call from poll loop."""
        if not self._settings.observing_session_enabled:
            if self._state != SessionState.DAYTIME:
                self._logger.info("Observing session disabled — resetting to DAYTIME")
                self._reset_to_daytime()
            return self._state

        self._refresh_observing_window()
        is_dark = self._observing_window.is_dark if self._observing_window else False

        if self._state == SessionState.DAYTIME:
            if is_dark:
                self._logger.info(
                    "Sun below threshold (%.1f°) — transitioning to NIGHT_STARTUP",
                    self._observing_window.current_sun_altitude if self._observing_window else 0,
                )
                self._state = SessionState.NIGHT_STARTUP
                self._unpark_done = False
                self._pointing_calibration_requested = False
                self._autofocus_requested = False

        elif self._state == SessionState.NIGHT_STARTUP:
            if not is_dark:
                self._logger.info("Sun rose during NIGHT_STARTUP — transitioning to NIGHT_SHUTDOWN")
                self._state = SessionState.NIGHT_SHUTDOWN
                self._park_done = False
                self._shutdown_entered_at = time.monotonic()
            else:
                self._run_startup_actions()

        elif self._state == SessionState.OBSERVING:
            if not is_dark:
                self._logger.info("Sun rising above threshold — transitioning to NIGHT_SHUTDOWN")
                self._state = SessionState.NIGHT_SHUTDOWN
                self._park_done = False
                self._shutdown_entered_at = time.monotonic()

        elif self._state == SessionState.NIGHT_SHUTDOWN:
            self._run_shutdown_actions()

        return self._state

    def _refresh_observing_window(self) -> None:
        location = self._get_location()
        if location is None:
            self._observing_window = None
            return
        lat, lon = location
        threshold = self._settings.observing_session_sun_altitude_threshold
        try:
            self._observing_window = compute_observing_window(lat, lon, threshold)
        except Exception:
            self._logger.warning("Failed to compute observing window", exc_info=True)
            self._observing_window = None

    def _run_startup_actions(self) -> None:
        """Execute enabled startup actions in order: unpark → pointing calibration → autofocus → done."""
        # Step 1: Unpark
        if self._settings.observing_session_do_park and not self._unpark_done:
            if self._unpark_mount is not None:
                self._logger.info("NIGHT_STARTUP: Unparking mount")
                try:
                    if not self._unpark_mount():
                        self._logger.warning("Unpark returned failure — will retry next cycle")
                        return
                except Exception:
                    self._logger.warning("Unpark failed — will retry next cycle", exc_info=True)
                    return
            self._unpark_done = True
            return

        # Step 2: Pointing calibration (before autofocus — plate solving tolerates mild defocus)
        if (
            self._settings.observing_session_do_pointing_calibration
            and self._request_pointing_calibration is not None
            and not self._pointing_calibration_requested
        ):
            self._logger.info("NIGHT_STARTUP: Running pointing calibration")
            try:
                self._request_pointing_calibration()
            except Exception:
                self._logger.warning("Pointing calibration request failed", exc_info=True)
            self._pointing_calibration_requested = True
            return

        if (
            self._settings.observing_session_do_pointing_calibration
            and self._is_pointing_calibration_running is not None
            and self._is_pointing_calibration_running()
        ):
            return  # Still calibrating

        # Step 3: Autofocus
        if self._settings.observing_session_do_autofocus and not self._autofocus_requested:
            self._logger.info("NIGHT_STARTUP: Requesting autofocus")
            try:
                self._request_autofocus()
            except Exception:
                self._logger.warning("Autofocus request failed", exc_info=True)
            self._autofocus_requested = True
            return

        if self._settings.observing_session_do_autofocus and self._is_autofocus_running():
            return  # Still waiting for autofocus to finish

        # All startup actions complete
        self._logger.info("NIGHT_STARTUP complete — transitioning to OBSERVING")
        self._state = SessionState.OBSERVING

    def _run_shutdown_actions(self) -> None:
        """Wait for imaging to finish, then park.  Force-park on timeout.

        Only the imaging queue needs to drain before parking — processing
        and upload don't use the mount and can continue after park.  A hard
        timeout force-parks even if imaging is stuck, protecting hardware
        from sunrise exposure.
        """
        timed_out = False
        if self._shutdown_entered_at is not None:
            elapsed = time.monotonic() - self._shutdown_entered_at
            if elapsed >= _SHUTDOWN_TIMEOUT_SECONDS:
                timed_out = True
                self._logger.warning("NIGHT_SHUTDOWN: timeout after %.0fs — force-parking to protect hardware", elapsed)

        if not timed_out and not self._is_imaging_idle():
            return  # Wait for current exposure to finish

        if self._settings.observing_session_do_park and not self._park_done:
            if self._park_mount is not None:
                self._logger.info("NIGHT_SHUTDOWN: Parking mount")
                try:
                    if not self._park_mount():
                        if not timed_out:
                            self._logger.warning("Park returned failure — will retry next cycle")
                            return
                        self._logger.error("Park failed even on timeout — proceeding to DAYTIME anyway")
                except Exception:
                    if not timed_out:
                        self._logger.warning("Park failed — will retry next cycle", exc_info=True)
                        return
                    self._logger.error("Park failed even on timeout — proceeding to DAYTIME anyway", exc_info=True)
            self._park_done = True

        self._logger.info("NIGHT_SHUTDOWN complete — transitioning to DAYTIME")
        self._reset_to_daytime()

    def _reset_to_daytime(self) -> None:
        self._state = SessionState.DAYTIME
        self._unpark_done = False
        self._pointing_calibration_requested = False
        self._autofocus_requested = False
        self._park_done = False
        self._shutdown_entered_at = None

    def is_winding_down(self) -> bool:
        """Return True when the session is shutting down and no new imaging should start."""
        return self._state == SessionState.NIGHT_SHUTDOWN

    def _get_session_activity(self) -> str | None:
        """Return a human-readable label for the current startup/shutdown sub-step."""
        if self._state == SessionState.NIGHT_STARTUP:
            if self._settings.observing_session_do_park and not self._unpark_done:
                return "Unparking mount"
            if (
                self._settings.observing_session_do_pointing_calibration
                and self._request_pointing_calibration is not None
            ):
                if not self._pointing_calibration_requested:
                    return "Requesting pointing calibration"
                if self._is_pointing_calibration_running and self._is_pointing_calibration_running():
                    return "Calibrating pointing model"
            if self._settings.observing_session_do_autofocus:
                if not self._autofocus_requested:
                    return "Requesting autofocus"
                if self._is_autofocus_running():
                    return "Autofocusing"
            return "Finishing startup"
        if self._state == SessionState.NIGHT_SHUTDOWN:
            if not self._is_imaging_idle():
                return "Waiting for imaging to finish"
            if self._settings.observing_session_do_park and not self._park_done:
                return "Parking mount"
            return "Finishing shutdown"
        return None

    def status_dict(self) -> dict[str, Any]:
        """Build a dict for the web status broadcast."""
        window = self._observing_window
        return {
            "observing_session_state": self._state.value,
            "session_activity": self._get_session_activity(),
            "observing_session_threshold": self._settings.observing_session_sun_altitude_threshold,
            "sun_altitude": window.current_sun_altitude if window else None,
            "dark_window_start": window.dark_start if window else None,
            "dark_window_end": window.dark_end if window else None,
        }
