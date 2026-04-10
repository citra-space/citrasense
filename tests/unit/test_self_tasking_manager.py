"""Tests for SelfTaskingManager gating logic and API call construction."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

from citrascope.location.twilight import ObservingWindow
from citrascope.tasks.observing_session import SessionState
from citrascope.tasks.self_tasking_manager import _REQUEST_INTERVAL_SECONDS, SelfTaskingManager


def _make_manager(
    self_tasking_enabled=True,
    session_state=SessionState.OBSERVING,
    window=None,
    exclude_types=None,
    orbit_regimes=None,
    group_ids=None,
    collection_type="Track",
):
    api_client = MagicMock()
    api_client.create_batch_collection_requests.return_value = {"status": "ok", "created": 5}

    settings = MagicMock()
    settings.self_tasking_enabled = self_tasking_enabled
    settings.self_tasking_satellite_group_ids = group_ids or []
    settings.self_tasking_exclude_object_types = exclude_types or []
    settings.self_tasking_include_orbit_regimes = orbit_regimes or []
    settings.self_tasking_collection_type = collection_type

    logger = logging.getLogger("test_self_tasking")

    if window is None:
        window = ObservingWindow(
            is_dark=True,
            current_sun_altitude=-20.0,
            dark_start="2025-01-01T00:00:00Z",
            dark_end="2025-01-01T06:00:00Z",
        )

    mgr = SelfTaskingManager(
        api_client=api_client,
        settings=settings,
        logger=logger,
        ground_station_id="gs-001",
        sensor_id="sensor-001",
        get_session_state=lambda: session_state,
        get_observing_window=lambda: window,
    )
    return mgr, api_client, settings


def test_check_and_request_fires_when_conditions_met():
    mgr, api_client, _ = _make_manager()
    mgr.check_and_request()
    api_client.create_batch_collection_requests.assert_called_once()


def test_skips_when_disabled():
    mgr, api_client, _ = _make_manager(self_tasking_enabled=False)
    mgr.check_and_request()
    api_client.create_batch_collection_requests.assert_not_called()


def test_skips_when_not_observing():
    mgr, api_client, _ = _make_manager(session_state=SessionState.DAYTIME)
    mgr.check_and_request()
    api_client.create_batch_collection_requests.assert_not_called()


def test_skips_when_no_window():
    window = ObservingWindow(is_dark=True, current_sun_altitude=-20.0, dark_start=None, dark_end=None)
    mgr, api_client, _ = _make_manager(window=window)
    mgr.check_and_request()
    api_client.create_batch_collection_requests.assert_not_called()


def test_respects_interval():
    mgr, api_client, _ = _make_manager()

    # First call should fire
    mgr.check_and_request()
    assert api_client.create_batch_collection_requests.call_count == 1

    # Immediate second call should be throttled
    mgr.check_and_request()
    assert api_client.create_batch_collection_requests.call_count == 1


def test_fires_again_after_interval():
    mgr, api_client, _ = _make_manager()

    mgr.check_and_request()
    assert api_client.create_batch_collection_requests.call_count == 1

    # Fast-forward past the interval
    mgr._last_request_time = time.monotonic() - _REQUEST_INTERVAL_SECONDS - 1
    mgr.check_and_request()
    assert api_client.create_batch_collection_requests.call_count == 2


def test_passes_targeting_filters():
    mgr, api_client, _ = _make_manager(
        exclude_types=["Debris", "Unknown"],
        orbit_regimes=["LEO", "GEO"],
        group_ids=["group-1"],
    )
    mgr.check_and_request()
    call_kwargs = api_client.create_batch_collection_requests.call_args
    assert call_kwargs.kwargs["exclude_types"] == ["Debris", "Unknown"]
    assert call_kwargs.kwargs["include_orbit_regimes"] == ["LEO", "GEO"]
    assert call_kwargs.kwargs["satellite_group_ids"] == ["group-1"]
    # When group_ids are set, discover_visible should be False
    assert call_kwargs.kwargs["discover_visible"] is False


def test_discover_visible_when_no_group_ids():
    mgr, api_client, _ = _make_manager(group_ids=[])
    mgr.check_and_request()
    call_kwargs = api_client.create_batch_collection_requests.call_args
    assert call_kwargs.kwargs["discover_visible"] is True


def test_handles_api_failure_gracefully():
    mgr, api_client, _ = _make_manager()
    api_client.create_batch_collection_requests.return_value = None
    mgr.check_and_request()
    # Should not raise; interval should still be recorded
    assert mgr._last_request_time > 0


def test_handles_api_exception_gracefully():
    mgr, api_client, _ = _make_manager()
    api_client.create_batch_collection_requests.side_effect = Exception("network error")
    mgr.check_and_request()
    # Should not raise; interval should still be recorded
    assert mgr._last_request_time > 0


def test_status_dict_before_any_request():
    mgr, _, _ = _make_manager()
    sd = mgr.status_dict()
    assert sd["last_batch_request"] is None
    assert sd["last_batch_created"] is None
    assert sd["next_request_seconds"] is None


def test_status_dict_after_request():
    mgr, _, _ = _make_manager()
    before = time.time()
    mgr.check_and_request()
    after = time.time()
    sd = mgr.status_dict()
    assert sd["last_batch_request"] is not None
    # Must be a wall-clock epoch, not a monotonic timestamp
    assert before <= sd["last_batch_request"] <= after
    assert sd["last_batch_created"] == 5
    assert sd["next_request_seconds"] is not None
    assert sd["next_request_seconds"] > 0


def test_next_request_seconds_decreases_over_time():
    mgr, _, _ = _make_manager()
    mgr.check_and_request()

    sd1 = mgr.status_dict()
    # Simulate time passing by shifting _last_request_time back
    mgr._last_request_time -= 60
    sd2 = mgr.status_dict()

    assert sd2["next_request_seconds"] < sd1["next_request_seconds"]


def test_default_collection_type_is_track():
    mgr, api_client, _ = _make_manager()
    mgr.check_and_request()
    call_kwargs = api_client.create_batch_collection_requests.call_args
    assert call_kwargs.kwargs["request_type"] == "Track"


def test_passes_characterization_collection_type():
    mgr, api_client, _ = _make_manager(collection_type="Characterization")
    mgr.check_and_request()
    call_kwargs = api_client.create_batch_collection_requests.call_args
    assert call_kwargs.kwargs["request_type"] == "Characterization"
