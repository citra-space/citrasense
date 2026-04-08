"""Unit tests for NINA adapter connect() — safety monitor connection (issue #228)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from citrascope.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter

API = "http://nina:1888/v2/api"


def _ok(response=None):
    """Mock a successful NINA API response."""
    m = MagicMock()
    m.json.return_value = {"Success": True, "Response": response or {}}
    return m


def _fail(error="Not available"):
    """Mock a failed NINA API response."""
    m = MagicMock()
    m.json.return_value = {"Success": False, "Error": error}
    return m


def _filterwheel_info():
    """Mock filterwheel info response with one filter."""
    m = MagicMock()
    m.json.return_value = {
        "Success": True,
        "Response": {"AvailableFilters": [{"Id": 0, "Name": "Clear"}]},
    }
    return m


def _all_succeed_responses():
    """Return the ordered mock responses for a fully successful connect()."""
    return [
        _ok(),  # camera connect
        _ok(),  # camera cool
        _ok(),  # filterwheel connect
        _ok(),  # focuser connect
        _ok(),  # safety monitor connect
        _ok(),  # mount connect
        _ok(),  # mount unpark
        _filterwheel_info(),  # discover_filters -> filterwheel info
    ]


@pytest.fixture
def adapter():
    return NinaAdvancedHttpAdapter(
        logger=MagicMock(),
        images_dir=Path("/tmp"),
        nina_api_path=API,
    )


class TestConnectSafetyMonitor:
    @patch("citrascope.hardware.nina.nina_adapter.NinaEventListener")
    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
    def test_safety_monitor_connect_called(self, mock_get, mock_ws, adapter):
        """Safety monitor connect is attempted during connect()."""
        mock_get.side_effect = _all_succeed_responses()

        result = adapter.connect()

        assert result is True
        urls = [c.args[0] for c in mock_get.call_args_list]
        assert f"{API}/equipment/safetymonitor/connect" in urls

    @patch("citrascope.hardware.nina.nina_adapter.NinaEventListener")
    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
    def test_safety_monitor_failure_does_not_block_connect(self, mock_get, mock_ws, adapter):
        """A failed safety monitor connect warns but still returns True."""
        responses = _all_succeed_responses()
        responses[4] = _fail("No device selected")  # safety monitor fails
        mock_get.side_effect = responses

        result = adapter.connect()

        assert result is True
        adapter.logger.warning.assert_any_call("Failed to connect safety monitor: No device selected")

    @patch("citrascope.hardware.nina.nina_adapter.NinaEventListener")
    @patch("citrascope.hardware.nina.nina_adapter.requests.get")
    def test_safety_monitor_called_after_focuser_before_mount(self, mock_get, mock_ws, adapter):
        """Safety monitor connect is sequenced between focuser and mount."""
        mock_get.side_effect = _all_succeed_responses()

        adapter.connect()

        urls = [c.args[0] for c in mock_get.call_args_list]
        focuser_idx = urls.index(f"{API}/equipment/focuser/connect")
        safety_idx = urls.index(f"{API}/equipment/safetymonitor/connect")
        mount_idx = urls.index(f"{API}/equipment/mount/connect")
        assert focuser_idx < safety_idx < mount_idx
