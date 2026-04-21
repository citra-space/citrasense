"""Unit tests for NinaFocuser, capture_preview, and set_filter on NinaAdvancedHttpAdapter."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from citrasense.hardware.nina.nina_focuser import NinaFocuser

API = "http://nina:1888/v2/api"


def _info_response(
    position: int = 5000,
    is_moving: bool = False,
    temperature: float = 20.5,
    max_step: int = 50000,
    connected: bool = True,
):
    m = MagicMock()
    m.json.return_value = {
        "Success": True,
        "Response": {
            "Position": position,
            "IsMoving": is_moving,
            "Temperature": temperature,
            "MaxStep": max_step,
            "Connected": connected,
        },
    }
    return m


def _ok(response=None):
    m = MagicMock()
    m.json.return_value = {"Success": True, "Response": response or {}}
    return m


def _fail(error="Command failed"):
    m = MagicMock()
    m.json.return_value = {"Success": False, "Error": error}
    return m


@pytest.fixture
def focuser():
    return NinaFocuser(logger=MagicMock(), nina_api_path=API)


# ──────────────────────────────────────────────────────────────────────
# NinaFocuser: connection lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestNinaFocuserConnect:
    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_connect_success(self, mock_get, focuser):
        mock_get.return_value = _ok()
        assert focuser.connect() is True
        assert focuser._connected is True
        mock_get.assert_called_once()
        assert "focuser/connect" in mock_get.call_args.args[0]

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_connect_failure(self, mock_get, focuser):
        mock_get.return_value = _fail()
        assert focuser.connect() is False
        assert focuser._connected is False

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_connect_network_error(self, mock_get, focuser):
        mock_get.side_effect = requests.ConnectionError("refused")
        assert focuser.connect() is False

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_is_connected_queries_info(self, mock_get, focuser):
        mock_get.return_value = _info_response(connected=True)
        assert focuser.is_connected() is True

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_is_connected_false_when_disconnected(self, mock_get, focuser):
        mock_get.return_value = _info_response(connected=False)
        assert focuser.is_connected() is False

    def test_disconnect(self, focuser):
        focuser._connected = True
        focuser.disconnect()
        assert focuser._connected is False


# ──────────────────────────────────────────────────────────────────────
# NinaFocuser: move operations
# ──────────────────────────────────────────────────────────────────────


class TestNinaFocuserMove:
    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_move_absolute_success(self, mock_get, focuser):
        mock_get.return_value = _ok()
        assert focuser.move_absolute(12345) is True
        url = mock_get.call_args.args[0]
        assert "focuser/move?" in url
        assert "position=12345" in url

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_move_absolute_failure(self, mock_get, focuser):
        mock_get.return_value = _fail()
        assert focuser.move_absolute(12345) is False

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_move_relative_computes_absolute_target(self, mock_get, focuser):
        """NINA has no relative endpoint — move_relative reads position then issues absolute move."""
        mock_get.side_effect = [
            _info_response(position=5000, max_step=50000),  # get_position
            _info_response(position=5000, max_step=50000),  # get_max_position
            _ok(),  # move_absolute
        ]
        assert focuser.move_relative(-200) is True
        move_url = mock_get.call_args_list[-1].args[0]
        assert "focuser/move?" in move_url
        assert "position=4800" in move_url

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_move_relative_clamps_to_zero(self, mock_get, focuser):
        mock_get.side_effect = [
            _info_response(position=100, max_step=50000),
            _info_response(position=100, max_step=50000),
            _ok(),
        ]
        assert focuser.move_relative(-500) is True
        move_url = mock_get.call_args_list[-1].args[0]
        assert "position=0" in move_url

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_move_relative_clamps_to_max(self, mock_get, focuser):
        mock_get.side_effect = [
            _info_response(position=49900, max_step=50000),
            _info_response(position=49900, max_step=50000),
            _ok(),
        ]
        assert focuser.move_relative(500) is True
        move_url = mock_get.call_args_list[-1].args[0]
        assert "position=50000" in move_url

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_move_relative_fails_when_position_unknown(self, mock_get, focuser):
        mock_get.return_value = _fail()
        assert focuser.move_relative(100) is False

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_abort_move(self, mock_get, focuser):
        mock_get.return_value = _ok()
        focuser.abort_move()
        url = mock_get.call_args.args[0]
        assert "stop-move" in url

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_abort_move_tolerates_error(self, mock_get, focuser):
        mock_get.side_effect = requests.Timeout("timeout")
        focuser.abort_move()  # should not raise


# ──────────────────────────────────────────────────────────────────────
# NinaFocuser: info queries
# ──────────────────────────────────────────────────────────────────────


class TestNinaFocuserInfo:
    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_get_position(self, mock_get, focuser):
        mock_get.return_value = _info_response(position=7777)
        assert focuser.get_position() == 7777

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_get_position_returns_none_on_failure(self, mock_get, focuser):
        mock_get.return_value = _fail()
        assert focuser.get_position() is None

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_is_moving(self, mock_get, focuser):
        mock_get.return_value = _info_response(is_moving=True)
        assert focuser.is_moving() is True

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_is_not_moving(self, mock_get, focuser):
        mock_get.return_value = _info_response(is_moving=False)
        assert focuser.is_moving() is False

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_is_moving_returns_false_on_error(self, mock_get, focuser):
        mock_get.side_effect = requests.ConnectionError("refused")
        assert focuser.is_moving() is False

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_get_max_position(self, mock_get, focuser):
        mock_get.return_value = _info_response(max_step=65000)
        assert focuser.get_max_position() == 65000

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_get_temperature(self, mock_get, focuser):
        mock_get.return_value = _info_response(temperature=18.3)
        assert focuser.get_temperature() == pytest.approx(18.3)

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_get_temperature_nan_returns_none(self, mock_get, focuser):
        mock_get.return_value = _info_response(temperature=float("nan"))
        assert focuser.get_temperature() is None

    @patch("citrasense.hardware.nina.nina_focuser.requests.get")
    def test_get_temperature_none_returns_none(self, mock_get, focuser):
        resp = MagicMock()
        resp.json.return_value = {
            "Success": True,
            "Response": {"Position": 5000, "IsMoving": False, "Temperature": None, "MaxStep": 50000},
        }
        mock_get.return_value = resp
        assert focuser.get_temperature() is None


# ──────────────────────────────────────────────────────────────────────
# NinaAdvancedHttpAdapter: capture_preview
# ──────────────────────────────────────────────────────────────────────


def _make_adapter():
    from citrasense.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter

    return NinaAdvancedHttpAdapter(
        logger=MagicMock(),
        images_dir=Path("/tmp"),
        nina_api_path=API,
    )


class TestCapturePreview:
    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_returns_jpeg_data_url(self, mock_get):
        adapter = _make_adapter()
        jpeg_bytes = b"\xff\xd8\xff\xe0fake-jpeg-data"
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "image/jpeg"}
        resp.content = jpeg_bytes
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = adapter.capture_preview(1.0)

        assert result.startswith("data:image/jpeg;base64,")
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert "stream" in call_kwargs.kwargs.get("params", {}) or "stream" in str(call_kwargs)

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_returns_png_data_url(self, mock_get):
        adapter = _make_adapter()
        png_bytes = b"\x89PNGfake-png-data"
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "image/png"}
        resp.content = png_bytes
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = adapter.capture_preview(2.0)

        assert result.startswith("data:image/png;base64,")

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_concurrent_capture_raises(self, mock_get):
        adapter = _make_adapter()
        adapter._preview_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="already in progress"):
                adapter.capture_preview(1.0)
        finally:
            adapter._preview_lock.release()

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_network_error_propagates(self, mock_get):
        adapter = _make_adapter()
        mock_get.side_effect = requests.ConnectionError("refused")
        with pytest.raises(requests.ConnectionError):
            adapter.capture_preview(1.0)

    def test_supports_direct_camera_control_delegates_to_is_camera_connected(self):
        adapter = _make_adapter()
        with patch.object(adapter, "is_camera_connected", return_value=True):
            assert adapter.supports_direct_camera_control() is True
        with patch.object(adapter, "is_camera_connected", return_value=False):
            assert adapter.supports_direct_camera_control() is False


# ──────────────────────────────────────────────────────────────────────
# NinaAdvancedHttpAdapter: set_filter
# ──────────────────────────────────────────────────────────────────────


class TestSetFilter:
    def _adapter_with_listener(self):
        adapter = _make_adapter()
        listener = MagicMock()
        listener.filter_changed = threading.Event()
        adapter._event_listener = listener
        adapter.filter_map = {
            0: {"name": "Luminance", "focus_position": 9000, "enabled": True},
            1: {"name": "Red", "focus_position": 9200, "enabled": True},
        }
        return adapter, listener

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_set_filter_already_on_target(self, mock_get):
        adapter, _listener = self._adapter_with_listener()
        info_resp = MagicMock()
        info_resp.json.return_value = {
            "Success": True,
            "Response": {"SelectedFilter": {"Id": 1}},
        }
        mock_get.return_value = info_resp

        assert adapter.set_filter(1) is True
        # Should not have sent a change-filter command
        urls = [c.args[0] for c in mock_get.call_args_list]
        assert not any("change-filter" in u for u in urls)

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_set_filter_changes_and_moves_focus(self, mock_get):
        adapter, listener = self._adapter_with_listener()
        focuser_mock = MagicMock()
        adapter._focuser = focuser_mock

        filter_info = MagicMock()
        filter_info.json.return_value = {
            "Success": True,
            "Response": {"SelectedFilter": {"Id": 0}},
        }
        change_resp = MagicMock()
        change_resp.json.return_value = {"Success": True, "Response": {}}

        call_count = {"n": 0}

        def _get_side_effect(*args, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx == 0:
                return filter_info
            # Second call is change-filter — fire the WS event
            listener.filter_changed.set()
            return change_resp

        mock_get.side_effect = _get_side_effect

        assert adapter.set_filter(1) is True
        focuser_mock.move_absolute.assert_called_once_with(9200)

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_set_filter_nina_rejects(self, mock_get):
        adapter, _listener = self._adapter_with_listener()

        filter_info = MagicMock()
        filter_info.json.return_value = {
            "Success": True,
            "Response": {"SelectedFilter": {"Id": 0}},
        }
        change_resp = MagicMock()
        change_resp.json.return_value = {"Success": False, "Error": "Device busy"}
        mock_get.side_effect = [filter_info, change_resp]

        assert adapter.set_filter(1) is False

    def test_set_filter_no_listener_returns_false(self):
        adapter = _make_adapter()
        adapter._event_listener = None
        assert adapter.set_filter(0) is False

    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_set_filter_timeout_returns_false(self, mock_get):
        adapter, _listener = self._adapter_with_listener()
        adapter.HARDWARE_MOVE_TIMEOUT = 0.01

        filter_info = MagicMock()
        filter_info.json.return_value = {
            "Success": True,
            "Response": {"SelectedFilter": {"Id": 0}},
        }
        change_resp = MagicMock()
        change_resp.json.return_value = {"Success": True, "Response": {}}
        mock_get.side_effect = [filter_info, change_resp]

        # Event never fires — simulates WS timeout
        assert adapter.set_filter(1) is False


# ──────────────────────────────────────────────────────────────────────
# NinaAdvancedHttpAdapter: focuser wiring in connect()
# ──────────────────────────────────────────────────────────────────────


def _filterwheel_info():
    m = MagicMock()
    m.json.return_value = {
        "Success": True,
        "Response": {"AvailableFilters": [{"Id": 0, "Name": "Clear"}]},
    }
    return m


def _all_succeed_responses():
    return [
        _ok(),  # camera connect
        _ok(),  # camera cool
        _ok(),  # filterwheel connect
        _ok(),  # focuser connect
        _ok(),  # safety monitor connect
        _ok(),  # mount connect
        _ok(),  # mount unpark
        _filterwheel_info(),  # discover_filters
    ]


class TestFocuserWiring:
    @patch("citrasense.hardware.nina.nina_adapter.NinaEventListener")
    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_focuser_created_on_connect(self, mock_get, mock_ws):
        mock_get.side_effect = _all_succeed_responses()
        adapter = _make_adapter()

        assert adapter.connect() is True
        assert adapter.focuser is not None
        assert isinstance(adapter.focuser, NinaFocuser)

    @patch("citrasense.hardware.nina.nina_adapter.NinaEventListener")
    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_focuser_none_when_focuser_connect_fails(self, mock_get, mock_ws):
        responses = _all_succeed_responses()
        responses[3] = _fail("No focuser")  # focuser connect fails
        mock_get.side_effect = responses
        adapter = _make_adapter()

        assert adapter.connect() is True
        assert adapter.focuser is None

    @patch("citrasense.hardware.nina.nina_adapter.NinaEventListener")
    @patch("citrasense.hardware.nina.nina_adapter.requests.get")
    def test_focuser_cleared_on_disconnect(self, mock_get, mock_ws):
        mock_get.side_effect = _all_succeed_responses()
        adapter = _make_adapter()
        adapter.connect()

        assert adapter.focuser is not None
        adapter.disconnect()
        assert adapter.focuser is None
