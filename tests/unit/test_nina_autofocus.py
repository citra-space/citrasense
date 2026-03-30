"""Unit tests for NINA adapter autofocus hardening (issue #204)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from citrascope.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter


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
    a._event_listener = el
    return a


def _mock_response(json_data):
    """Build a mock requests.Response that returns *json_data* from .json()."""
    m = MagicMock()
    m.json.return_value = json_data
    m.raise_for_status.return_value = None
    return m


# ---------- _get_current_filter_id ----------


class TestGetCurrentFilterId:
    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
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

    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
    def test_returns_none_on_failure(self, mock_get, adapter):
        mock_get.return_value = _mock_response({"Success": False, "Error": "Not connected"})
        assert adapter._get_current_filter_id() is None

    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
    def test_returns_none_on_missing_field(self, mock_get, adapter):
        mock_get.return_value = _mock_response({"Success": True, "Response": {}})
        assert adapter._get_current_filter_id() is None

    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
    def test_returns_none_on_network_error(self, mock_get, adapter):
        mock_get.side_effect = ConnectionError("refused")
        assert adapter._get_current_filter_id() is None


# ---------- _auto_focus_one_filter — filter skip (the actual bug fix) ----------


class TestAutoFocusFilterSkip:
    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
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

    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
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
