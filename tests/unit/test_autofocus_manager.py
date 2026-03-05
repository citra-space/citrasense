"""Unit tests for AutofocusManager."""

import time
from pathlib import Path
from unittest.mock import MagicMock

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
    settings.autofocus_interval_minutes = 60
    settings.last_autofocus_timestamp = None
    settings.autofocus_target_preset = "mirach"
    settings.autofocus_target_custom_ra = None
    settings.autofocus_target_custom_dec = None
    settings.adapter_settings = {}
    return settings


@pytest.fixture
def mock_daemon(mock_settings):
    daemon = MagicMock()
    daemon.settings = mock_settings
    return daemon


@pytest.fixture
def mock_imaging_queue():
    queue = MagicMock()
    queue.is_idle.return_value = True
    return queue


@pytest.fixture
def autofocus_manager(mock_logger, mock_hardware_adapter, mock_daemon, mock_imaging_queue):
    return AutofocusManager(mock_logger, mock_hardware_adapter, mock_daemon, mock_imaging_queue)


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


def test_resolve_target_no_settings(autofocus_manager, mock_daemon):
    mock_daemon.settings = None
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


class TestDummyAdapterAutofocusTargeting:
    """Verify DummyAdapter handles None/real coords correctly in do_autofocus."""

    def test_skips_slew_when_both_none(self, tmp_path: Path):
        from citrascope.hardware.dummy_adapter import DummyAdapter

        adapter = DummyAdapter(MagicMock(), tmp_path)
        adapter.connect()
        adapter.do_autofocus(target_ra=None, target_dec=None)

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
