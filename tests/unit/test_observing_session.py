"""Tests for ObservingSessionManager state machine."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

from citrasense.location.twilight import ObservingWindow
from citrasense.sensors.telescope.observing_session import (
    _SHUTDOWN_TIMEOUT_SECONDS,
    ObservingSessionManager,
    SessionState,
)


def _make_sensor_config(**overrides):
    """Return a mock SensorConfig with observing session defaults."""
    s = MagicMock()
    s.observing_session_enabled = True
    s.observing_session_sun_altitude_threshold = -12.0
    s.observing_session_do_pointing_calibration = False
    s.observing_session_do_autofocus = True
    s.observing_session_do_park = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_manager(
    settings=None,
    location=(38.0, -105.0),
    autofocus_running=False,
    imaging_idle=True,
    queues_idle=True,
):
    settings = settings or _make_sensor_config()
    logger = logging.getLogger("test_session")

    request_autofocus = MagicMock()
    is_autofocus_running = MagicMock(return_value=autofocus_running)
    is_imaging_idle = MagicMock(return_value=imaging_idle)
    are_queues_idle = MagicMock(return_value=queues_idle)
    park_mount = MagicMock(return_value=True)
    unpark_mount = MagicMock(return_value=True)

    mgr = ObservingSessionManager(
        sensor_config=settings,
        logger=logger,
        get_location=lambda: location,
        request_autofocus=request_autofocus,
        is_autofocus_running=is_autofocus_running,
        is_imaging_idle=is_imaging_idle,
        are_queues_idle=are_queues_idle,
        park_mount=park_mount,
        unpark_mount=unpark_mount,
    )
    return mgr, request_autofocus, is_autofocus_running, is_imaging_idle, are_queues_idle, park_mount, unpark_mount


_DARK_WINDOW = ObservingWindow(
    is_dark=True, current_sun_altitude=-20.0, dark_start="2025-01-01T00:00:00Z", dark_end="2025-01-01T06:00:00Z"
)
_LIGHT_WINDOW = ObservingWindow(is_dark=False, current_sun_altitude=10.0)


def test_initial_state_is_daytime():
    mgr, *_ = _make_manager()
    assert mgr.state == SessionState.DAYTIME


def test_stays_daytime_when_disabled():
    settings = _make_sensor_config(observing_session_enabled=False)
    mgr, *_ = _make_manager(settings=settings)
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW
        mgr.update()
    assert mgr.state == SessionState.DAYTIME


def test_daytime_to_night_startup_when_dark():
    mgr, *_ = _make_manager()
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW
        result = mgr.update()
    assert result == SessionState.NIGHT_STARTUP


def test_night_startup_unpark_then_autofocus():
    mgr, request_af, is_af_running, _, _, _park, unpark = _make_manager()

    # Force into NIGHT_STARTUP
    mgr._state = SessionState.NIGHT_STARTUP
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW

        # First update: unpark
        mgr.update()
        unpark.assert_called_once()
        request_af.assert_not_called()

        # Second update: request autofocus
        mgr.update()
        request_af.assert_called_once()

        # Third update: autofocus still running
        is_af_running.return_value = True
        mgr.update()
        assert mgr.state == SessionState.NIGHT_STARTUP

        # Fourth update: autofocus done
        is_af_running.return_value = False
        mgr.update()
        assert mgr.state == SessionState.OBSERVING


def test_night_startup_unpark_retries_on_failure():
    mgr, _, _, _, _, _park, unpark = _make_manager()
    unpark.return_value = False

    mgr._state = SessionState.NIGHT_STARTUP
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW

        # First call: unpark fails, stays in NIGHT_STARTUP, does not advance
        mgr.update()
        unpark.assert_called_once()
        assert mgr._unpark_done is False
        assert mgr.state == SessionState.NIGHT_STARTUP

        # Second call: unpark succeeds, advances
        unpark.return_value = True
        mgr.update()
        assert mgr._unpark_done is True


def test_night_startup_unpark_retries_on_exception():
    mgr, _, _, _, _, _park, unpark = _make_manager()
    unpark.side_effect = RuntimeError("connection lost")

    mgr._state = SessionState.NIGHT_STARTUP
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW

        mgr.update()
        assert mgr._unpark_done is False
        assert mgr.state == SessionState.NIGHT_STARTUP

        unpark.side_effect = None
        unpark.return_value = True
        mgr.update()
        assert mgr._unpark_done is True


def test_night_startup_aborts_if_sun_rises():
    """If sun rises during NIGHT_STARTUP, skip to NIGHT_SHUTDOWN."""
    mgr, *_ = _make_manager()
    mgr._state = SessionState.NIGHT_STARTUP

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    assert mgr.state == SessionState.NIGHT_SHUTDOWN
    assert mgr._shutdown_entered_at is not None


def test_night_startup_skips_unpark_when_disabled():
    settings = _make_sensor_config(observing_session_do_park=False)
    mgr, request_af, *_ = _make_manager(settings=settings)

    mgr._state = SessionState.NIGHT_STARTUP
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW

        # First update: skips unpark, requests autofocus
        mgr.update()
        request_af.assert_called_once()


def test_night_startup_skips_autofocus_when_disabled():
    settings = _make_sensor_config(observing_session_do_autofocus=False, observing_session_do_park=False)
    mgr, request_af, *_ = _make_manager(settings=settings)

    mgr._state = SessionState.NIGHT_STARTUP
    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _DARK_WINDOW

        # Should go straight to OBSERVING
        mgr.update()
        assert mgr.state == SessionState.OBSERVING
        request_af.assert_not_called()


def test_observing_to_night_shutdown():
    mgr, *_ = _make_manager()
    mgr._state = SessionState.OBSERVING

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    assert mgr.state == SessionState.NIGHT_SHUTDOWN


def test_night_shutdown_waits_for_imaging():
    mgr, _, _, _is_imaging_idle, _, park, _ = _make_manager(imaging_idle=False)
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic()

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    assert mgr.state == SessionState.NIGHT_SHUTDOWN
    park.assert_not_called()


def test_night_shutdown_parks_when_imaging_idle():
    mgr, _, _, _is_imaging_idle, _, park, _ = _make_manager(imaging_idle=True)
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic()

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    assert mgr.state == SessionState.DAYTIME
    park.assert_called_once()


def test_night_shutdown_park_retries_on_failure():
    mgr, _, _, _is_imaging_idle, _, park, _ = _make_manager(imaging_idle=True)
    park.return_value = False
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic()

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW

        # Park fails — stays in NIGHT_SHUTDOWN
        mgr.update()
        assert mgr.state == SessionState.NIGHT_SHUTDOWN
        park.assert_called_once()

        # Park succeeds on retry
        park.return_value = True
        mgr.update()
        assert mgr.state == SessionState.DAYTIME


def test_night_shutdown_park_retries_on_exception():
    mgr, _, _, _is_imaging_idle, _, park, _ = _make_manager(imaging_idle=True)
    park.side_effect = RuntimeError("mount error")
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic()

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW

        mgr.update()
        assert mgr.state == SessionState.NIGHT_SHUTDOWN

        park.side_effect = None
        park.return_value = True
        mgr.update()
        assert mgr.state == SessionState.DAYTIME


def test_night_shutdown_force_parks_even_if_park_fails():
    """On timeout, proceed to DAYTIME even if park itself fails."""
    mgr, _, _, _is_imaging_idle, _, park, _ = _make_manager(imaging_idle=False)
    park.return_value = False
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic() - _SHUTDOWN_TIMEOUT_SECONDS - 1

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    # Even though park failed, timeout forces transition to DAYTIME
    assert mgr.state == SessionState.DAYTIME
    park.assert_called_once()


def test_night_shutdown_does_not_wait_for_processing_or_upload():
    """Shutdown should park even if processing/upload queues still have work."""
    mgr, _, _, _is_imaging_idle, _are_queues_idle, park, _ = _make_manager(imaging_idle=True, queues_idle=False)
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic()

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    assert mgr.state == SessionState.DAYTIME
    park.assert_called_once()


def test_night_shutdown_force_parks_on_timeout():
    """After timeout, park even if imaging is still busy."""
    mgr, _, _, _is_imaging_idle, _, park, _ = _make_manager(imaging_idle=False)
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._shutdown_entered_at = time.monotonic() - _SHUTDOWN_TIMEOUT_SECONDS - 1

    with patch.object(mgr, "_refresh_observing_window"):
        mgr._observing_window = _LIGHT_WINDOW
        mgr.update()

    assert mgr.state == SessionState.DAYTIME
    park.assert_called_once()


def test_is_winding_down():
    mgr, *_ = _make_manager()
    assert mgr.is_winding_down() is False

    mgr._state = SessionState.NIGHT_SHUTDOWN
    assert mgr.is_winding_down() is True

    mgr._state = SessionState.OBSERVING
    assert mgr.is_winding_down() is False


def test_status_dict():
    mgr, *_ = _make_manager()
    mgr._state = SessionState.OBSERVING
    mgr._observing_window = _DARK_WINDOW

    sd = mgr.status_dict()
    assert sd["observing_session_state"] == "observing"
    assert sd["sun_altitude"] == -20.0
    assert sd["dark_window_start"] is not None
    assert sd["dark_window_end"] is not None
    assert sd["observing_session_threshold"] == -12.0
    assert sd["session_activity"] is None


def test_status_dict_activity_during_startup():
    mgr, *_ = _make_manager()
    mgr._state = SessionState.NIGHT_STARTUP
    mgr._observing_window = _DARK_WINDOW

    sd = mgr.status_dict()
    assert sd["session_activity"] == "Unparking mount"

    mgr._unpark_done = True
    sd = mgr.status_dict()
    assert sd["session_activity"] == "Requesting autofocus"

    mgr._autofocus_requested = True
    # Autofocus is running
    mgr._is_autofocus_running = MagicMock(return_value=True)
    sd = mgr.status_dict()
    assert sd["session_activity"] == "Autofocusing"


def test_status_dict_activity_during_shutdown():
    mgr, _, _, is_imaging_idle, _, _, _ = _make_manager(imaging_idle=False)
    mgr._state = SessionState.NIGHT_SHUTDOWN
    mgr._observing_window = _LIGHT_WINDOW

    sd = mgr.status_dict()
    assert sd["session_activity"] == "Waiting for imaging to finish"

    is_imaging_idle.return_value = True
    sd = mgr.status_dict()
    assert sd["session_activity"] == "Parking mount"


def test_no_location_stays_in_state():
    mgr, *_ = _make_manager(location=None)
    with patch.object(mgr, "_refresh_observing_window", wraps=mgr._refresh_observing_window):
        mgr.update()
    assert mgr.state == SessionState.DAYTIME
    assert mgr.observing_window is None
