"""Unit tests for NINA adapter autofocus hardening (issues #204, #237)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from citrasense.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter


@pytest.fixture
def adapter():
    """Create a NinaAdvancedHttpAdapter with mocked internals."""
    a = NinaAdvancedHttpAdapter(
        logger=MagicMock(),
        images_dir=Path("/tmp"),
        nina_api_path="http://nina:1888/v2/api",
    )
    el = MagicMock()
    el.filter_changed = threading.Event()
    el.autofocus_finished = threading.Event()
    el.autofocus_error = threading.Event()
    el.on_af_point = None
    el.last_af_point_time = 0.0
    a._event_listener = el

    focuser = MagicMock()
    focuser.move_absolute.return_value = True
    focuser.get_position.return_value = 9000
    focuser.is_moving.return_value = False
    a._focuser = focuser
    return a


def _mock_response(json_data):
    """Build a mock requests.Response that returns *json_data* from .json()."""
    m = MagicMock()
    m.json.return_value = json_data
    m.raise_for_status.return_value = None
    return m


# ---------- _get_current_filter_id ----------


class TestGetCurrentFilterId:
    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_returns_id_from_selected_filter(self, mock_get, adapter):
        mock_get.return_value = _mock_response(
            {
                "Success": True,
                "Response": {
                    "SelectedFilter": {"Name": "Blue", "Id": 2},
                    "AvailableFilters": [],
                },
            }
        )
        assert adapter._get_current_filter_id() == 2

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_returns_none_on_failure(self, mock_get, adapter):
        mock_get.return_value = _mock_response({"Success": False, "Error": "Not connected"})
        assert adapter._get_current_filter_id() is None

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_returns_none_on_missing_field(self, mock_get, adapter):
        mock_get.return_value = _mock_response({"Success": True, "Response": {}})
        assert adapter._get_current_filter_id() is None

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_returns_none_on_network_error(self, mock_get, adapter):
        mock_get.side_effect = ConnectionError("refused")
        assert adapter._get_current_filter_id() is None


# ---------- _auto_focus_one_filter — filter skip (the actual bug fix) ----------


class TestAutoFocusFilterSkip:
    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_skips_change_when_already_on_filter(self, mock_get, adapter):
        """When _get_current_filter_id returns the target, skip the WS wait."""
        fw_info = _mock_response({"Success": True, "Response": {"SelectedFilter": {"Name": "Clear", "Id": 0}}})
        focuser_move = _mock_response({"Success": True, "Response": "Focuser move started"})
        focuser_info = _mock_response({"Success": True, "Response": {"Position": 9000}})
        af_trigger = _mock_response({"Success": True, "Response": "Autofocus started"})
        last_af = _mock_response(
            {
                "Success": True,
                "Response": {
                    "CalculatedFocusPoint": {"Position": 8500, "Value": 1.75},
                },
            }
        )

        def route_get(url, **kwargs):
            if "filterwheel/info" in url:
                return fw_info
            if "change-filter" in url:
                pytest.fail("Should not call change-filter when already on target filter")
            if "focuser/move" in url:
                return focuser_move
            if "focuser/info" in url:
                return focuser_info
            if "focuser/auto-focus" in url:
                adapter._event_listener.autofocus_finished.set()
                return af_trigger
            if "focuser/last-af" in url:
                return last_af
            return _mock_response({"Success": True})

        mock_get.side_effect = route_get

        result = adapter._auto_focus_one_filter(0, "Clear", 9000)
        assert result == 8500

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_changes_filter_when_different(self, mock_get, adapter):
        """When current filter differs from target, do the change+wait dance."""
        fw_info = _mock_response({"Success": True, "Response": {"SelectedFilter": {"Name": "Red", "Id": 0}}})
        change_resp = _mock_response({"Success": True, "Response": "Filter changed"})
        focuser_move = _mock_response({"Success": True, "Response": "Focuser move started"})
        focuser_info = _mock_response({"Success": True, "Response": {"Position": 9000}})
        af_trigger = _mock_response({"Success": True, "Response": "Autofocus started"})
        last_af = _mock_response(
            {
                "Success": True,
                "Response": {
                    "CalculatedFocusPoint": {"Position": 8200, "Value": 1.65},
                },
            }
        )

        change_called = []

        def route_get(url, **kwargs):
            if "filterwheel/info" in url:
                return fw_info
            if "change-filter" in url:
                change_called.append(url)
                adapter._event_listener.filter_changed.set()
                return change_resp
            if "focuser/move" in url:
                return focuser_move
            if "focuser/info" in url:
                return focuser_info
            if "focuser/auto-focus" in url:
                adapter._event_listener.autofocus_finished.set()
                return af_trigger
            if "focuser/last-af" in url:
                return last_af
            return _mock_response({"Success": True})

        mock_get.side_effect = route_get

        result = adapter._auto_focus_one_filter(2, "Blue", 9000)
        assert result == 8200
        assert len(change_called) == 1
        assert "filterId=2" in change_called[0]


# ---------- _auto_focus_one_filter — silent failure detection (issue #237) ----------


class TestAutoFocusSilentFailure:
    """When NINA AF fails without sending WS events, the adapter should detect
    stale AF activity + idle focuser and exit early."""

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_detects_silent_failure_via_activity_timeout(self, mock_get, adapter):
        """AF points stop arriving and focuser is idle → silent failure detected."""
        adapter.AF_ACTIVITY_TIMEOUT = 0
        fw_info = _mock_response({"Success": True, "Response": {"SelectedFilter": {"Name": "Clear", "Id": 0}}})
        focuser_move = _mock_response({"Success": True, "Response": "Focuser move started"})
        focuser_info_idle = _mock_response({"Success": True, "Response": {"Position": 9000, "IsMoving": False}})
        af_trigger = _mock_response({"Success": True, "Response": "Autofocus started"})

        def route_get(url, **kwargs):
            if "filterwheel/info" in url:
                return fw_info
            if "focuser/move" in url:
                return focuser_move
            if "focuser/auto-focus" in url:
                return af_trigger
            if "focuser/info" in url:
                return focuser_info_idle
            return _mock_response({"Success": True})

        mock_get.side_effect = route_get

        result = adapter._auto_focus_one_filter(0, "Clear", 9000)
        assert result == 9000
        adapter.logger.warning.assert_any_call(
            f"Autofocus for filter Clear appears to have failed silently — "
            f"no AF points for {adapter.AF_ACTIVITY_TIMEOUT}s and focuser is idle"
        )

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_detects_silent_failure_when_no_points_ever_arrive(self, mock_get, adapter):
        """NINA fails before sending any AF points — activity timeout still fires."""
        adapter.AF_ACTIVITY_TIMEOUT = 0
        fw_info = _mock_response({"Success": True, "Response": {"SelectedFilter": {"Name": "Clear", "Id": 0}}})
        focuser_move = _mock_response({"Success": True, "Response": "Focuser move started"})
        focuser_info_idle = _mock_response({"Success": True, "Response": {"Position": 9000, "IsMoving": False}})
        af_trigger = _mock_response({"Success": True, "Response": "Autofocus started"})

        def route_get(url, **kwargs):
            if "filterwheel/info" in url:
                return fw_info
            if "focuser/move" in url:
                return focuser_move
            if "focuser/auto-focus" in url:
                return af_trigger
            if "focuser/info" in url:
                return focuser_info_idle
            return _mock_response({"Success": True})

        mock_get.side_effect = route_get

        adapter._event_listener.last_af_point_time = 0.0
        result = adapter._auto_focus_one_filter(0, "Clear", 9000)
        assert result == 9000
        assert adapter._event_listener.last_af_point_time > 0.0

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_no_false_positive_while_focuser_moving(self, mock_get, adapter):
        """AF points are stale but focuser is still moving → don't declare failure yet.
        Instead, the WS event fires and the normal success path is taken."""
        adapter.AF_ACTIVITY_TIMEOUT = 0
        fw_info = _mock_response({"Success": True, "Response": {"SelectedFilter": {"Name": "Clear", "Id": 0}}})
        af_trigger = _mock_response({"Success": True, "Response": "Autofocus started"})
        last_af = _mock_response(
            {"Success": True, "Response": {"CalculatedFocusPoint": {"Position": 8500, "Value": 1.75}}}
        )

        is_moving_call_count = 0

        def _is_moving_side_effect():
            nonlocal is_moving_call_count
            is_moving_call_count += 1
            if is_moving_call_count >= 2:
                adapter._event_listener.autofocus_finished.set()
            return True

        adapter._focuser.is_moving.side_effect = _is_moving_side_effect

        def route_get(url, **kwargs):
            if "filterwheel/info" in url:
                return fw_info
            if "focuser/auto-focus" in url:
                return af_trigger
            if "focuser/last-af" in url:
                return last_af
            return _mock_response({"Success": True})

        mock_get.side_effect = route_get

        result = adapter._auto_focus_one_filter(0, "Clear", 9000)
        assert result == 8500
