"""Tests for ZWO EAF (Electronic Automatic Focuser) driver.

These tests mock the native libEAFFocuser library since it won't be installed in CI.
Focuses on: error code constants, connect/disconnect lifecycle, movement, temperature
handling, backlash/reverse settings, and device registry integration.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestEafBindingsConstants:
    """Verify our error code constants match the SDK header values."""

    def test_error_code_values(self):
        from citrasense.hardware.devices.focuser.zwo_eaf_bindings import (
            EAF_ERROR_CLOSED,
            EAF_ERROR_ERROR_STATE,
            EAF_ERROR_GENERAL_ERROR,
            EAF_ERROR_INVALID_ID,
            EAF_ERROR_INVALID_INDEX,
            EAF_ERROR_INVALID_VALUE,
            EAF_ERROR_MOVING,
            EAF_ERROR_NOT_SUPPORTED,
            EAF_ERROR_REMOVED,
            EAF_SUCCESS,
        )

        assert EAF_SUCCESS == 0
        assert EAF_ERROR_INVALID_INDEX == 1
        assert EAF_ERROR_INVALID_ID == 2
        assert EAF_ERROR_INVALID_VALUE == 3
        assert EAF_ERROR_REMOVED == 4
        assert EAF_ERROR_MOVING == 5
        assert EAF_ERROR_ERROR_STATE == 6
        assert EAF_ERROR_GENERAL_ERROR == 7
        assert EAF_ERROR_NOT_SUPPORTED == 8
        assert EAF_ERROR_CLOSED == 9

    def test_error_names_dict(self):
        from citrasense.hardware.devices.focuser.zwo_eaf_bindings import _ERROR_NAMES

        assert _ERROR_NAMES[0] == "EAF_SUCCESS"
        assert _ERROR_NAMES[4] == "EAF_ERROR_REMOVED"

    def test_eaf_error_message(self):
        from citrasense.hardware.devices.focuser.zwo_eaf_bindings import EafError

        err = EafError("EAFMove", 5)
        assert err.error_code == 5
        assert "EAFMove" in str(err)
        assert "EAF_ERROR_MOVING" in str(err)


class TestEafLibraryLoading:
    """Test library loading behavior when native library is missing."""

    def test_library_not_found_raises(self):
        from citrasense.hardware.devices.focuser.zwo_eaf_bindings import EafLibraryNotFound

        with patch("citrasense.hardware.devices.focuser.zwo_eaf_bindings._lib", None):
            with patch("citrasense.hardware.devices.focuser.zwo_eaf_bindings.ctypes") as mock_ctypes:
                mock_ctypes.util.find_library.return_value = None
                mock_ctypes.cdll = MagicMock()
                with patch("os.environ.get", return_value=None):
                    with patch("os.path.isfile", return_value=False):
                        import citrasense.hardware.devices.focuser.zwo_eaf_bindings as mod

                        mod._lib = None
                        with pytest.raises(EafLibraryNotFound):
                            mod._load_library()


class TestZwoEafFocuserConnect:
    """Test ZwoEafFocuser connect/disconnect with mocked native library."""

    def _make_focuser(self, **kwargs):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        logger = logging.getLogger("test")
        return ZwoEafFocuser(logger, **kwargs)

    def _mock_eaf_instance(self):
        mock = MagicMock()
        mock.is_open = True
        mock.get_sdk_version.return_value = "1.6.0"
        mock.get_num.return_value = 1
        mock.get_id.return_value = 0
        info = MagicMock()
        info.Name = b"ZWO EAF"
        info.MaxStep = 100000
        info.ID = 0
        mock.get_property.return_value = info
        mock.get_firmware_version.return_value = (2, 1, 0)
        mock.get_serial_number.return_value = "0011aabb"
        mock.get_max_step.return_value = 100000
        mock.get_position.return_value = 50000
        mock.get_temperature.return_value = 22.5
        return mock

    def test_connect_success(self):
        mock_eaf = self._mock_eaf_instance()
        MockEafDriver = MagicMock(return_value=mock_eaf)

        mock_bindings = MagicMock(EafFocuser=MockEafDriver)
        with patch.dict(
            "sys.modules",
            {"citrasense.hardware.devices.focuser.zwo_eaf_bindings": mock_bindings},
        ):
            focuser = self._make_focuser(focuser_id=-1, backlash=10, reverse=False, beep=True)
            result = focuser.connect()

        assert result is True
        assert focuser._sdk_id == 0
        mock_eaf.open.assert_called_once_with(0)
        mock_eaf.set_backlash.assert_called_once_with(10)
        mock_eaf.set_reverse.assert_called_once_with(False)
        mock_eaf.set_beep.assert_called_once_with(True)

    def test_connect_no_focusers(self):
        mock_eaf = self._mock_eaf_instance()
        mock_eaf.get_num.return_value = 0
        MockEafDriver = MagicMock(return_value=mock_eaf)

        with patch.dict(
            "sys.modules",
            {"citrasense.hardware.devices.focuser.zwo_eaf_bindings": MagicMock(EafFocuser=MockEafDriver)},
        ):
            focuser = self._make_focuser()
            result = focuser.connect()

        assert result is False

    def test_connect_import_error(self):
        """Returns False when the SDK library is not installed."""
        with patch.dict("sys.modules", {"citrasense.hardware.devices.focuser.zwo_eaf_bindings": None}):
            focuser = self._make_focuser()
            result = focuser.connect()

        assert result is False

    def test_disconnect(self):
        mock_eaf = self._mock_eaf_instance()
        MockEafDriver = MagicMock(return_value=mock_eaf)

        with patch.dict(
            "sys.modules",
            {"citrasense.hardware.devices.focuser.zwo_eaf_bindings": MagicMock(EafFocuser=MockEafDriver)},
        ):
            focuser = self._make_focuser()
            focuser.connect()

        focuser.disconnect()
        mock_eaf.close.assert_called_once()
        assert focuser._eaf is None
        assert focuser._sdk_id is None


class TestZwoEafFocuserMovement:
    """Test focuser movement operations."""

    def _connected_focuser(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        focuser = ZwoEafFocuser(logging.getLogger("test"))
        mock_eaf = MagicMock()
        mock_eaf.is_open = True
        mock_eaf.get_position.return_value = 50000
        mock_eaf.get_max_step.return_value = 100000
        mock_eaf.is_moving.return_value = (False, False)
        focuser._eaf = mock_eaf
        focuser._sdk_id = 0
        focuser._max_step = 100000
        return focuser, mock_eaf

    def test_move_absolute(self):
        focuser, mock_eaf = self._connected_focuser()
        result = focuser.move_absolute(75000)
        assert result is True
        mock_eaf.move.assert_called_once_with(75000)

    def test_move_absolute_out_of_range(self):
        focuser, mock_eaf = self._connected_focuser()
        result = focuser.move_absolute(200000)
        assert result is False
        mock_eaf.move.assert_not_called()

    def test_move_absolute_negative(self):
        focuser, mock_eaf = self._connected_focuser()
        result = focuser.move_absolute(-1)
        assert result is False
        mock_eaf.move.assert_not_called()

    def test_move_relative(self):
        focuser, mock_eaf = self._connected_focuser()
        result = focuser.move_relative(1000)
        assert result is True
        mock_eaf.move.assert_called_once_with(51000)

    def test_move_relative_negative(self):
        focuser, mock_eaf = self._connected_focuser()
        result = focuser.move_relative(-5000)
        assert result is True
        mock_eaf.move.assert_called_once_with(45000)

    def test_move_relative_exceeds_max(self):
        focuser, _mock_eaf = self._connected_focuser()
        result = focuser.move_relative(60000)
        assert result is False

    def test_get_position(self):
        focuser, _mock_eaf = self._connected_focuser()
        assert focuser.get_position() == 50000

    def test_is_moving_true(self):
        focuser, mock_eaf = self._connected_focuser()
        mock_eaf.is_moving.return_value = (True, False)
        assert focuser.is_moving() is True

    def test_is_moving_false(self):
        focuser, _mock_eaf = self._connected_focuser()
        assert focuser.is_moving() is False

    def test_abort_move(self):
        focuser, mock_eaf = self._connected_focuser()
        focuser.abort_move()
        mock_eaf.stop.assert_called_once()

    def test_get_max_position(self):
        focuser, _mock_eaf = self._connected_focuser()
        assert focuser.get_max_position() == 100000

    def test_not_connected_returns_none_or_false(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        focuser = ZwoEafFocuser(logging.getLogger("test"))
        assert focuser.move_absolute(100) is False
        assert focuser.move_relative(100) is False
        assert focuser.get_position() is None
        assert focuser.is_moving() is False
        assert focuser.get_max_position() is None
        assert focuser.get_temperature() is None


class TestZwoEafTemperature:
    """Test temperature reading and the -273 sentinel."""

    def _connected_focuser(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        focuser = ZwoEafFocuser(logging.getLogger("test"))
        mock_eaf = MagicMock()
        mock_eaf.is_open = True
        focuser._eaf = mock_eaf
        focuser._sdk_id = 0
        return focuser, mock_eaf

    def test_temperature_normal(self):
        focuser, mock_eaf = self._connected_focuser()
        mock_eaf.get_temperature.return_value = 22.5
        assert focuser.get_temperature() == 22.5

    def test_temperature_sentinel_returns_none(self):
        focuser, mock_eaf = self._connected_focuser()
        mock_eaf.get_temperature.return_value = -273.0
        assert focuser.get_temperature() is None

    def test_temperature_error_returns_none(self):
        focuser, mock_eaf = self._connected_focuser()
        mock_eaf.get_temperature.side_effect = Exception("sensor failure")
        assert focuser.get_temperature() is None


class TestZwoEafSettingsSchema:
    """Test the settings schema generation."""

    def test_schema_has_required_fields(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        ZwoEafFocuser._focuser_cache = None
        schema = ZwoEafFocuser.get_settings_schema()

        names = [s["name"] for s in schema]
        assert "focuser_id" in names
        assert "backlash" in names
        assert "reverse" in names
        assert "beep" in names

    def test_focuser_id_has_auto_option(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        ZwoEafFocuser._focuser_cache = None
        schema = ZwoEafFocuser.get_settings_schema()

        focuser_id_entry = next(s for s in schema if s["name"] == "focuser_id")
        options = focuser_id_entry["options"]
        assert any(o["value"] == -1 for o in options)


class TestZwoEafDeviceRegistry:
    """Test that ZWO EAF is properly registered."""

    def test_zwo_eaf_in_focuser_registry(self):
        from citrasense.hardware.devices.device_registry import FOCUSER_DEVICES

        assert "zwo_eaf" in FOCUSER_DEVICES
        assert FOCUSER_DEVICES["zwo_eaf"]["class_name"] == "ZwoEafFocuser"

    def test_zwo_eaf_importable_via_registry(self):
        from citrasense.hardware.devices.device_registry import get_focuser_class

        cls = get_focuser_class("zwo_eaf")
        assert cls.__name__ == "ZwoEafFocuser"

    def test_zwo_eaf_in_init_exports(self):
        from citrasense.hardware.devices.focuser import ZwoEafFocuser

        assert ZwoEafFocuser.get_friendly_name() == "ZWO EAF"


class TestZwoEafIsConnected:
    """Test the is_connected property."""

    def test_not_connected_initially(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        focuser = ZwoEafFocuser(logging.getLogger("test"))
        assert focuser.is_connected() is False

    def test_connected_after_open(self):
        from citrasense.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        focuser = ZwoEafFocuser(logging.getLogger("test"))
        mock_eaf = MagicMock()
        mock_eaf.is_open = True
        focuser._eaf = mock_eaf
        focuser._sdk_id = 0
        assert focuser.is_connected() is True
