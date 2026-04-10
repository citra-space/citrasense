"""Unit tests for AutofocusManager."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from citrascope.constants import AUTOFOCUS_TARGET_PRESETS
from citrascope.tasks.autofocus_manager import AutofocusManager


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_hardware_adapter():
    adapter = MagicMock()
    adapter.supports_autofocus.return_value = True
    adapter.supports_filter_management.return_value = False
    return adapter


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.scheduled_autofocus_enabled = False
    settings.autofocus_schedule_mode = "interval"
    settings.autofocus_interval_minutes = 60
    settings.autofocus_after_sunset_offset_minutes = 60
    settings.last_autofocus_timestamp = None
    settings.autofocus_target_preset = "mirach"
    settings.autofocus_target_custom_ra = None
    settings.autofocus_target_custom_dec = None
    settings.adapter_settings = {}
    return settings


@pytest.fixture
def mock_imaging_queue():
    queue = MagicMock()
    queue.is_idle.return_value = True
    return queue


@pytest.fixture
def autofocus_manager(mock_logger, mock_hardware_adapter, mock_settings, mock_imaging_queue):
    return AutofocusManager(mock_logger, mock_hardware_adapter, mock_settings, mock_imaging_queue)


# ---------------------------------------------------------------------------
# Request / Cancel / Query lifecycle
# ---------------------------------------------------------------------------


def test_request_sets_flag(autofocus_manager):
    autofocus_manager.request()
    assert autofocus_manager.is_requested() is True


def test_request_returns_true(autofocus_manager):
    assert autofocus_manager.request() is True


def test_cancel_clears_flag(autofocus_manager):
    autofocus_manager.request()
    result = autofocus_manager.cancel()
    assert result is True
    assert autofocus_manager.is_requested() is False


def test_cancel_when_not_requested(autofocus_manager):
    assert autofocus_manager.cancel() is False


# ---------------------------------------------------------------------------
# check_and_execute -- manual request
# ---------------------------------------------------------------------------


def test_check_and_execute_runs_on_request(autofocus_manager, mock_hardware_adapter):
    autofocus_manager.request()
    result = autofocus_manager.check_and_execute()
    assert result is True
    mock_hardware_adapter.do_autofocus.assert_called_once()


def test_check_and_execute_clears_request(autofocus_manager):
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    assert autofocus_manager.is_requested() is False


def test_check_and_execute_noop_when_idle(autofocus_manager, mock_hardware_adapter):
    result = autofocus_manager.check_and_execute()
    assert result is False
    mock_hardware_adapter.do_autofocus.assert_not_called()


# ---------------------------------------------------------------------------
# check_and_execute -- imaging queue gating
# ---------------------------------------------------------------------------


def test_defers_when_imaging_queue_busy(autofocus_manager, mock_imaging_queue, mock_hardware_adapter):
    mock_imaging_queue.is_idle.return_value = False
    autofocus_manager.request()
    result = autofocus_manager.check_and_execute()
    assert result is False
    mock_hardware_adapter.do_autofocus.assert_not_called()
    assert autofocus_manager.is_requested() is True


def test_runs_when_imaging_queue_idle(autofocus_manager, mock_imaging_queue, mock_hardware_adapter):
    mock_imaging_queue.is_idle.return_value = True
    autofocus_manager.request()
    result = autofocus_manager.check_and_execute()
    assert result is True
    mock_hardware_adapter.do_autofocus.assert_called_once()


# ---------------------------------------------------------------------------
# check_and_execute -- scheduled autofocus
# ---------------------------------------------------------------------------


def test_scheduled_autofocus_fires_when_overdue(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.scheduled_autofocus_enabled = True
    mock_settings.autofocus_interval_minutes = 60
    mock_settings.last_autofocus_timestamp = int(time.time()) - 3700
    result = autofocus_manager.check_and_execute()
    assert result is True
    mock_hardware_adapter.do_autofocus.assert_called_once()


def test_scheduled_autofocus_skips_when_not_due(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.scheduled_autofocus_enabled = True
    mock_settings.autofocus_interval_minutes = 60
    mock_settings.last_autofocus_timestamp = int(time.time()) - 10
    result = autofocus_manager.check_and_execute()
    assert result is False
    mock_hardware_adapter.do_autofocus.assert_not_called()


def test_scheduled_autofocus_skips_when_disabled(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.scheduled_autofocus_enabled = False
    mock_settings.last_autofocus_timestamp = int(time.time()) - 99999
    result = autofocus_manager.check_and_execute()
    assert result is False
    mock_hardware_adapter.do_autofocus.assert_not_called()


def test_scheduled_autofocus_skips_when_no_adapter_support(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.scheduled_autofocus_enabled = True
    mock_hardware_adapter.supports_autofocus.return_value = False
    result = autofocus_manager.check_and_execute()
    assert result is False
    mock_hardware_adapter.do_autofocus.assert_not_called()


def test_scheduled_autofocus_fires_when_never_run(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.scheduled_autofocus_enabled = True
    mock_settings.last_autofocus_timestamp = None
    result = autofocus_manager.check_and_execute()
    assert result is True
    mock_hardware_adapter.do_autofocus.assert_called_once()


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def test_resolve_target_preset(autofocus_manager, mock_settings):
    mock_settings.autofocus_target_preset = "vega"
    ra, dec = autofocus_manager._resolve_target()
    expected = AUTOFOCUS_TARGET_PRESETS["vega"]
    assert ra == expected["ra"]
    assert dec == expected["dec"]


def test_resolve_target_custom(autofocus_manager, mock_settings):
    mock_settings.autofocus_target_preset = "custom"
    mock_settings.autofocus_target_custom_ra = 123.45
    mock_settings.autofocus_target_custom_dec = -67.89
    ra, dec = autofocus_manager._resolve_target()
    assert ra == 123.45
    assert dec == -67.89


def test_resolve_target_custom_missing_dec(autofocus_manager, mock_settings):
    mock_settings.autofocus_target_preset = "custom"
    mock_settings.autofocus_target_custom_ra = 123.45
    mock_settings.autofocus_target_custom_dec = None
    ra, dec = autofocus_manager._resolve_target()
    mirach = AUTOFOCUS_TARGET_PRESETS["mirach"]
    assert ra == mirach["ra"]
    assert dec == mirach["dec"]


def test_resolve_target_unknown_preset(autofocus_manager, mock_settings):
    mock_settings.autofocus_target_preset = "nonexistent_star"
    ra, dec = autofocus_manager._resolve_target()
    mirach = AUTOFOCUS_TARGET_PRESETS["mirach"]
    assert ra == mirach["ra"]
    assert dec == mirach["dec"]


def test_resolve_target_current_position(autofocus_manager, mock_settings):
    mock_settings.autofocus_target_preset = "current"
    ra, dec = autofocus_manager._resolve_target()
    assert ra is None
    assert dec is None


def test_resolve_target_no_settings(autofocus_manager):
    autofocus_manager.settings = None
    ra, dec = autofocus_manager._resolve_target()
    assert ra is None
    assert dec is None


# ---------------------------------------------------------------------------
# Target kwargs passed to adapter
# ---------------------------------------------------------------------------


def test_adapter_receives_named_preset_coords(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.autofocus_target_preset = "vega"
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    _, kwargs = mock_hardware_adapter.do_autofocus.call_args
    expected = AUTOFOCUS_TARGET_PRESETS["vega"]
    assert kwargs["target_ra"] == expected["ra"]
    assert kwargs["target_dec"] == expected["dec"]


def test_adapter_receives_none_for_current_position(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.autofocus_target_preset = "current"
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    _, kwargs = mock_hardware_adapter.do_autofocus.call_args
    assert kwargs["target_ra"] is None
    assert kwargs["target_dec"] is None


def test_adapter_receives_custom_coords(autofocus_manager, mock_settings, mock_hardware_adapter):
    mock_settings.autofocus_target_preset = "custom"
    mock_settings.autofocus_target_custom_ra = 200.0
    mock_settings.autofocus_target_custom_dec = 45.0
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    _, kwargs = mock_hardware_adapter.do_autofocus.call_args
    assert kwargs["target_ra"] == 200.0
    assert kwargs["target_dec"] == 45.0


# ---------------------------------------------------------------------------
# Running state and progress
# ---------------------------------------------------------------------------


def test_is_running_during_execution(autofocus_manager, mock_hardware_adapter):
    captured_running = []

    def capture_state(**kwargs):
        captured_running.append(autofocus_manager.is_running())

    mock_hardware_adapter.do_autofocus.side_effect = capture_state
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    assert captured_running == [True]


def test_is_not_running_after_completion(autofocus_manager):
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    assert autofocus_manager.is_running() is False


def test_progress_callback_passed(autofocus_manager, mock_hardware_adapter):
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    _, kwargs = mock_hardware_adapter.do_autofocus.call_args
    assert "on_progress" in kwargs
    assert callable(kwargs["on_progress"])


def test_progress_cleared_after_execution(autofocus_manager):
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    assert autofocus_manager.progress == ""


# ---------------------------------------------------------------------------
# Timestamp and settings persistence
# ---------------------------------------------------------------------------


def test_timestamp_saved_on_success(autofocus_manager, mock_settings):
    autofocus_manager.request()
    before = int(time.time())
    autofocus_manager.check_and_execute()
    assert mock_settings.last_autofocus_timestamp >= before
    mock_settings.save.assert_called_once()


def test_timestamp_saved_on_failure(autofocus_manager, mock_hardware_adapter, mock_settings):
    mock_hardware_adapter.do_autofocus.side_effect = RuntimeError("focus motor stuck")
    autofocus_manager.request()
    before = int(time.time())
    autofocus_manager.check_and_execute()
    assert mock_settings.last_autofocus_timestamp >= before
    mock_settings.save.assert_called_once()


# ---------------------------------------------------------------------------
# Filter config save
# ---------------------------------------------------------------------------


def test_filter_config_saved_after_autofocus(autofocus_manager, mock_hardware_adapter, mock_settings):
    mock_hardware_adapter.supports_filter_management.return_value = True
    mock_hardware_adapter.get_filter_config.return_value = {"0": {"name": "Red"}, "1": {"name": "Green"}}
    autofocus_manager.request()
    autofocus_manager.check_and_execute()
    assert mock_settings.adapter_settings["filters"] == {"0": {"name": "Red"}, "1": {"name": "Green"}}


# ---------------------------------------------------------------------------
# Autofocus target name resolution (status display)
# ---------------------------------------------------------------------------


class TestResolveAutofocusTargetName:
    """Tests for _resolve_autofocus_target_name used by the status API."""

    def test_named_preset(self):
        from citrascope.web.app import _resolve_autofocus_target_name

        settings = MagicMock()
        settings.autofocus_target_preset = "vega"
        result = _resolve_autofocus_target_name(settings)
        assert result == "Vega (Alpha Lyrae)"

    def test_current_position(self):
        from citrascope.web.app import _resolve_autofocus_target_name

        settings = MagicMock()
        settings.autofocus_target_preset = "current"
        result = _resolve_autofocus_target_name(settings)
        assert result == "Current position"

    def test_custom_with_coords(self):
        from citrascope.web.app import _resolve_autofocus_target_name

        settings = MagicMock()
        settings.autofocus_target_preset = "custom"
        settings.autofocus_target_custom_ra = 123.4567
        settings.autofocus_target_custom_dec = -45.6789
        result = _resolve_autofocus_target_name(settings)
        assert "123.4567" in result
        assert "-45.6789" in result
        assert "Custom" in result

    def test_custom_missing_coords(self):
        from citrascope.web.app import _resolve_autofocus_target_name

        settings = MagicMock()
        settings.autofocus_target_preset = "custom"
        settings.autofocus_target_custom_ra = None
        settings.autofocus_target_custom_dec = None
        result = _resolve_autofocus_target_name(settings)
        assert "Mirach" in result

    def test_unknown_preset(self):
        from citrascope.web.app import _resolve_autofocus_target_name

        settings = MagicMock()
        settings.autofocus_target_preset = "nonexistent"
        result = _resolve_autofocus_target_name(settings)
        assert "Mirach" in result
        assert "nonexistent" in result

    def test_falsy_preset_defaults_to_mirach(self):
        from citrascope.web.app import _resolve_autofocus_target_name

        settings = MagicMock()
        settings.autofocus_target_preset = ""
        result = _resolve_autofocus_target_name(settings)
        assert "Mirach" in result


# ---------------------------------------------------------------------------
# Dummy adapter autofocus targeting
# ---------------------------------------------------------------------------


def _has_apass_catalog() -> bool:
    """Return True only if the APASS catalog DB exists and is queryable."""
    try:
        from citrascope.catalogs.apass_catalog import ApassCatalog

        cat = ApassCatalog()
        if not cat.is_available():
            return False
        df = cat.cone_search(180.0, 30.0, 1.0)
        return len(df) > 0
    except Exception:
        return False


class TestDummyAdapterAutofocusTargeting:
    """Verify DummyAdapter handles None/real coords correctly in do_autofocus."""

    @pytest.mark.slow
    @pytest.mark.skipif(not _has_apass_catalog(), reason="APASS catalog not available")
    def test_skips_slew_when_both_none(self, tmp_path: Path):
        from citrascope.hardware.dummy_adapter import DummyAdapter

        adapter = DummyAdapter(MagicMock(), tmp_path)
        adapter.connect()
        adapter.do_autofocus(target_ra=None, target_dec=None)

    @pytest.mark.slow
    @pytest.mark.skipif(not _has_apass_catalog(), reason="APASS catalog not available")
    def test_uses_provided_coords(self, tmp_path: Path):
        from citrascope.hardware.dummy_adapter import DummyAdapter

        adapter = DummyAdapter(MagicMock(), tmp_path)
        adapter.connect()
        adapter.do_autofocus(target_ra=100.0, target_dec=30.0)

    def test_raises_on_partial_none(self, tmp_path: Path):
        from citrascope.hardware.dummy_adapter import DummyAdapter

        adapter = DummyAdapter(MagicMock(), tmp_path)
        adapter.connect()
        with pytest.raises(ValueError, match="both be set or both be None"):
            adapter.do_autofocus(target_ra=100.0, target_dec=None)
        with pytest.raises(ValueError, match="both be set or both be None"):
            adapter.do_autofocus(target_ra=None, target_dec=30.0)


# ---------------------------------------------------------------------------
# After-sunset scheduling
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_location_service():
    svc = MagicMock()
    svc.get_current_location.return_value = {
        "latitude": 33.0,
        "longitude": -105.0,
        "altitude": 2200.0,
        "source": "ground_station",
    }
    return svc


@pytest.fixture
def sunset_manager(mock_logger, mock_hardware_adapter, mock_settings, mock_imaging_queue, mock_location_service):
    mock_settings.autofocus_schedule_mode = "after_sunset"
    mock_settings.scheduled_autofocus_enabled = True
    mock_settings.autofocus_after_sunset_offset_minutes = 60
    return AutofocusManager(
        mock_logger,
        mock_hardware_adapter,
        mock_settings,
        imaging_queue=mock_imaging_queue,
        location_service=mock_location_service,
    )


class TestAfterSunsetScheduling:
    """Tests for the after-sunset autofocus scheduling mode."""

    def test_fires_when_past_trigger_and_never_run(self, sunset_manager):
        sunset_two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = sunset_two_hours_ago + timedelta(minutes=60)
            assert sunset_manager._should_run_after_sunset() is True

    def test_skips_when_before_trigger_time(self, sunset_manager):
        future_sunset = datetime.now(timezone.utc) + timedelta(hours=2)
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = future_sunset + timedelta(minutes=60)
            assert sunset_manager._should_run_after_sunset() is False

    def test_skips_when_already_run_after_trigger(self, sunset_manager, mock_settings):
        trigger_time = datetime.now(timezone.utc) - timedelta(hours=1)
        ran_after_trigger = trigger_time + timedelta(minutes=10)
        mock_settings.last_autofocus_timestamp = int(ran_after_trigger.timestamp())
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = trigger_time
            assert sunset_manager._should_run_after_sunset() is False

    def test_fires_when_last_run_was_before_trigger(self, sunset_manager, mock_settings):
        trigger_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        ran_before_trigger = trigger_time - timedelta(hours=12)
        mock_settings.last_autofocus_timestamp = int(ran_before_trigger.timestamp())
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = trigger_time
            assert sunset_manager._should_run_after_sunset() is True

    def test_skips_when_no_location(self, sunset_manager, mock_location_service):
        mock_location_service.get_current_location.return_value = None
        assert sunset_manager._should_run_after_sunset() is False

    def test_skips_when_no_location_service(
        self, mock_logger, mock_hardware_adapter, mock_settings, mock_imaging_queue
    ):
        mock_settings.autofocus_schedule_mode = "after_sunset"
        mock_settings.scheduled_autofocus_enabled = True
        mgr = AutofocusManager(mock_logger, mock_hardware_adapter, mock_settings, imaging_queue=mock_imaging_queue)
        assert mgr._should_run_after_sunset() is False

    def test_skips_when_sunset_computation_fails(self, sunset_manager):
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = None
            assert sunset_manager._should_run_after_sunset() is False

    def test_check_and_execute_uses_sunset_mode(self, sunset_manager, mock_hardware_adapter):
        trigger_time = datetime.now(timezone.utc) - timedelta(hours=1)
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = trigger_time
            result = sunset_manager.check_and_execute()
            assert result is True
            mock_hardware_adapter.do_autofocus.assert_called_once()


class TestGetNextAutofocusMinutes:
    """Tests for the get_next_autofocus_minutes status helper."""

    def test_interval_mode_with_recent_run(self, autofocus_manager, mock_settings):
        mock_settings.scheduled_autofocus_enabled = True
        mock_settings.autofocus_schedule_mode = "interval"
        mock_settings.autofocus_interval_minutes = 60
        mock_settings.last_autofocus_timestamp = int(time.time()) - 600
        result = autofocus_manager.get_next_autofocus_minutes()
        assert result is not None
        assert 49 <= result <= 51

    def test_interval_mode_never_run(self, autofocus_manager, mock_settings):
        mock_settings.scheduled_autofocus_enabled = True
        mock_settings.autofocus_schedule_mode = "interval"
        mock_settings.last_autofocus_timestamp = None
        assert autofocus_manager.get_next_autofocus_minutes() == 0

    def test_returns_none_when_disabled(self, autofocus_manager, mock_settings):
        mock_settings.scheduled_autofocus_enabled = False
        assert autofocus_manager.get_next_autofocus_minutes() is None

    def test_sunset_mode_returns_none_without_location(self, mock_logger, mock_hardware_adapter, mock_settings):
        mock_settings.scheduled_autofocus_enabled = True
        mock_settings.autofocus_schedule_mode = "after_sunset"
        mock_hardware_adapter.supports_autofocus.return_value = True
        mgr = AutofocusManager(mock_logger, mock_hardware_adapter, mock_settings)
        assert mgr.get_next_autofocus_minutes() is None

    def test_sunset_mode_future_trigger(self, sunset_manager):
        future_trigger = datetime.now(timezone.utc) + timedelta(minutes=90)
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = future_trigger
            result = sunset_manager.get_next_autofocus_minutes()
            assert result is not None
            assert 88 <= result <= 91

    def test_sunset_mode_overdue(self, sunset_manager):
        past_trigger = datetime.now(timezone.utc) - timedelta(minutes=30)
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = past_trigger
            assert sunset_manager.get_next_autofocus_minutes() == 0

    def test_sunset_mode_done_for_tonight(self, sunset_manager, mock_settings):
        trigger_time = datetime.now(timezone.utc) - timedelta(hours=1)
        ran_after = trigger_time + timedelta(minutes=5)
        mock_settings.last_autofocus_timestamp = int(ran_after.timestamp())
        with patch("citrascope.tasks.autofocus_manager.AutofocusManager._compute_sunset_trigger_time") as mock_trigger:
            mock_trigger.return_value = trigger_time
            assert sunset_manager.get_next_autofocus_minutes() is None
