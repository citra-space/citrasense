"""Tests for Moravian Instruments camera and filter wheel drivers.

These tests mock the native gxccd library since it won't be installed in CI.
Focuses on: enum constants correctness, error handling, integrated filter wheel
auto-detection, filter map population in DirectHardwareAdapter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMoravianBindingsConstants:
    """Verify our enum constants match the gxccd.h header values."""

    def test_boolean_parameter_constants(self):
        from citrascope.hardware.devices.moravian_bindings import GBP_COOLER, GBP_FILTERS

        assert GBP_COOLER == 4
        assert GBP_FILTERS == 6

    def test_integer_parameter_constants(self):
        from citrascope.hardware.devices.moravian_bindings import (
            GIP_CHIP_D,
            GIP_CHIP_W,
            GIP_DEFAULT_READ_MODE,
            GIP_FILTERS,
            GIP_MAX_GAIN,
            GIP_PIXEL_D,
            GIP_PIXEL_W,
        )

        assert GIP_CHIP_W == 1
        assert GIP_CHIP_D == 2
        assert GIP_PIXEL_W == 3
        assert GIP_PIXEL_D == 4
        assert GIP_FILTERS == 8
        assert GIP_DEFAULT_READ_MODE == 12
        assert GIP_MAX_GAIN == 16

    def test_string_parameter_constants(self):
        from citrascope.hardware.devices.moravian_bindings import (
            GSP_CAMERA_DESCRIPTION,
            GSP_CAMERA_SERIAL,
            GSP_CHIP_DESCRIPTION,
            GSP_MANUFACTURER,
        )

        assert GSP_CAMERA_DESCRIPTION == 0
        assert GSP_MANUFACTURER == 1
        assert GSP_CAMERA_SERIAL == 2
        assert GSP_CHIP_DESCRIPTION == 3

    def test_value_constants(self):
        from citrascope.hardware.devices.moravian_bindings import GV_CHIP_TEMPERATURE

        assert GV_CHIP_TEMPERATURE == 0

    def test_filter_wheel_constants(self):
        from citrascope.hardware.devices.moravian_bindings import (
            FW_GIP_FILTERS,
            FW_GSP_DESCRIPTION,
            FW_GSP_SERIAL_NUMBER,
        )

        assert FW_GIP_FILTERS == 5
        assert FW_GSP_DESCRIPTION == 0
        assert FW_GSP_SERIAL_NUMBER == 2


class TestGxccdCameraWithoutLibrary:
    """Test GxccdCamera behavior when native library is missing."""

    def test_library_not_found_raises(self):
        from citrascope.hardware.devices.moravian_bindings import GxccdLibraryNotFound

        with patch("citrascope.hardware.devices.moravian_bindings._lib", None):
            with patch("citrascope.hardware.devices.moravian_bindings.ctypes") as mock_ctypes:
                mock_ctypes.util.find_library.return_value = None
                mock_ctypes.cdll = MagicMock()
                with patch("os.environ.get", return_value=None):
                    with patch("os.path.isfile", return_value=False):
                        import citrascope.hardware.devices.moravian_bindings as mod
                        from citrascope.hardware.devices.moravian_bindings import _load_library

                        mod._lib = None
                        with pytest.raises(GxccdLibraryNotFound):
                            _load_library()


class TestMoravianCameraConnect:
    """Test MoravianCamera connect/disconnect with mocked native library."""

    def _make_camera(self, **kwargs):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera

        logger = logging.getLogger("test")
        return MoravianCamera(logger, **kwargs)

    def test_connect_usb_success(self):
        """Camera connects via USB, reads info, detects integrated filter wheel."""
        mock_cam_instance = MagicMock()
        mock_cam_instance.is_initialized = True
        mock_cam_instance.get_string_parameter.side_effect = lambda idx: {
            0: "C2-12000",
            1: "MI",
            2: "SN001",
            3: "KAI-12000",
        }[idx]
        mock_cam_instance.get_integer_parameter.side_effect = lambda idx: {
            1: 4242,
            2: 2838,
            3: 7400,
            4: 7400,
            8: 5,
            12: 0,
            13: 5,  # GIP_MAX_WINDOW_HEATING
            14: 3,  # GIP_MAX_FAN
            16: 255,
            17: 65535,  # GIP_MAX_PIXEL_VALUE
        }[idx]
        mock_cam_instance.get_boolean_parameter.side_effect = lambda idx: {
            3: True,  # GBP_SHUTTER
            4: True,  # GBP_COOLER
            5: True,  # GBP_FAN
            6: True,  # GBP_FILTERS
            8: True,  # GBP_WINDOW_HEATING
            14: False,  # GBP_ELECTRONIC_SHUTTER
            15: False,  # GBP_GPS
        }[idx]
        mock_cam_instance.enumerate_read_modes.return_value = ["Low Noise", "Fast Preview"]
        mock_cam_instance.get_value.return_value = -15.2
        mock_cam_instance.enumerate_filters.return_value = [
            ("Luminance", 0xFFFFFF, 0),
            ("Red", 0xFF0000, 100),
            ("Green", 0x00FF00, 95),
            ("Blue", 0x0000FF, 110),
            ("Clear", 0x000000, 0),
        ]

        MockGxccdCamera = MagicMock(return_value=mock_cam_instance)

        import citrascope.hardware.devices.moravian_bindings as bindings_mod

        with patch.object(bindings_mod, "GxccdCamera", MockGxccdCamera):
            camera = self._make_camera(camera_id=-1, connection_type="usb")
            result = camera.connect()

        assert result is True
        assert camera._camera_info["width"] == 4242
        assert camera._camera_info["height"] == 2838
        assert camera._camera_info["pixel_size_um"] == 7.4
        assert camera._has_cooler is True

        fw = camera.get_integrated_filter_wheel()
        assert fw is not None
        assert fw.get_filter_count() == 5
        assert fw.get_filter_names() == ["Luminance", "Red", "Green", "Blue", "Clear"]

    def test_connect_returns_false_on_error(self):
        """Camera returns False when initialization fails."""
        import citrascope.hardware.devices.moravian_bindings as bindings_mod

        mock_cam_instance = MagicMock()
        mock_cam_instance.initialize_usb.side_effect = bindings_mod.GxccdError("USB error")
        MockGxccdCamera = MagicMock(return_value=mock_cam_instance)

        with patch.object(bindings_mod, "GxccdCamera", MockGxccdCamera):
            camera = self._make_camera(camera_id=-1)
            result = camera.connect()

        assert result is False


class TestMoravianIntegratedFilterWheel:
    """Test the integrated filter wheel that shares the camera handle."""

    def test_set_filter_delegates_to_camera(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianIntegratedFilterWheel

        mock_cam = MagicMock()
        mock_cam.is_initialized = True
        mock_cam.enumerate_filters.return_value = [("Lum", 0, 0), ("Red", 0, 0)]
        logger = logging.getLogger("test")

        fw = MoravianIntegratedFilterWheel(mock_cam, 2, logger)
        fw.connect()

        assert fw.set_filter_position(1) is True
        mock_cam.set_filter.assert_called_once_with(1)
        assert fw.get_filter_position() == 1

    def test_disconnect_does_not_release_handle(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianIntegratedFilterWheel

        mock_cam = MagicMock()
        mock_cam.is_initialized = True
        logger = logging.getLogger("test")

        fw = MoravianIntegratedFilterWheel(mock_cam, 3, logger)
        fw.disconnect()
        mock_cam.release.assert_not_called()

    def test_set_filter_names(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianIntegratedFilterWheel

        mock_cam = MagicMock()
        mock_cam.is_initialized = True
        mock_cam.enumerate_filters.return_value = []
        logger = logging.getLogger("test")

        fw = MoravianIntegratedFilterWheel(mock_cam, 3, logger)
        fw.connect()

        assert fw.set_filter_names(["A", "B"]) is False  # Wrong count
        assert fw.set_filter_names(["A", "B", "C"]) is True
        assert fw.get_filter_names() == ["A", "B", "C"]


class TestMoravianStandaloneFilterWheel:
    """Test the standalone external filter wheel driver."""

    def test_connect_returns_false_on_error(self):
        import citrascope.hardware.devices.moravian_bindings as bindings_mod
        from citrascope.hardware.devices.filter_wheel.moravian_filter_wheel import MoravianFilterWheel

        mock_fw_instance = MagicMock()
        mock_fw_instance.initialize_usb.side_effect = bindings_mod.GxccdError("USB error")
        MockGxccdFW = MagicMock(return_value=mock_fw_instance)

        logger = logging.getLogger("test")

        with patch.object(bindings_mod, "GxccdFilterWheel", MockGxccdFW):
            fw = MoravianFilterWheel(logger, wheel_id=-1)
            result = fw.connect()

        assert result is False


class TestDirectAdapterFilterAutoDetect:
    """Test DirectHardwareAdapter's integrated filter wheel auto-detection."""

    def test_populate_filter_map_from_hardware(self):
        """When camera has integrated FW, filter_map is populated on connect."""
        from citrascope.hardware.direct.direct_adapter import DirectHardwareAdapter

        mock_camera = MagicMock()
        mock_camera.connect.return_value = True
        mock_camera.is_connected.return_value = True
        mock_camera.get_preferred_file_extension.return_value = "fits"

        mock_fw = MagicMock()
        mock_fw.connect.return_value = True
        mock_fw.get_filter_names.return_value = ["Luminance", "Red", "Green", "Blue"]
        mock_fw.get_filter_count.return_value = 4
        mock_camera.get_integrated_filter_wheel.return_value = mock_fw

        logger = logging.getLogger("test")
        images_dir = Path("/tmp/test_images")

        with patch("citrascope.hardware.direct.direct_adapter.get_camera_class") as mock_get_cam:
            with patch("citrascope.hardware.direct.direct_adapter.check_dependencies") as mock_check:
                mock_check.return_value = {"available": True, "missing": [], "install_cmd": ""}
                mock_cam_class = MagicMock(return_value=mock_camera)
                mock_cam_class.get_friendly_name.return_value = "Test Camera"
                mock_get_cam.return_value = mock_cam_class

                adapter = DirectHardwareAdapter(logger, images_dir, camera_type="moravian")

        # Override the camera with our mock (private attr backing the property)
        adapter._camera = mock_camera
        adapter.connect()

        assert adapter.filter_wheel is mock_fw
        assert adapter.supports_filter_management() is True
        assert 0 in adapter.filter_map
        assert adapter.filter_map[0]["name"] == "Luminance"
        assert adapter.filter_map[3]["name"] == "Blue"

    def test_update_filter_name(self):
        """DirectHardwareAdapter.update_filter_name() updates both map and hardware."""
        from citrascope.hardware.direct.direct_adapter import DirectHardwareAdapter

        logger = logging.getLogger("test")
        images_dir = Path("/tmp/test_images")

        with patch("citrascope.hardware.direct.direct_adapter.get_camera_class") as mock_get_cam:
            with patch("citrascope.hardware.direct.direct_adapter.check_dependencies") as mock_check:
                mock_check.return_value = {"available": True, "missing": [], "install_cmd": ""}
                mock_cam_class = MagicMock()
                mock_cam_class.get_friendly_name.return_value = "Test Camera"
                mock_get_cam.return_value = mock_cam_class

                adapter = DirectHardwareAdapter(logger, images_dir, camera_type="moravian")

        adapter.filter_map = {0: {"name": "Luminance", "focus_position": 0, "enabled": True}}

        mock_fw = MagicMock()
        mock_fw.get_filter_names.return_value = ["Luminance"]
        mock_fw.set_filter_names.return_value = True
        adapter.filter_wheel = mock_fw

        assert adapter.update_filter_name("0", "L-filter") is True
        assert adapter.filter_map[0]["name"] == "L-filter"
        mock_fw.set_filter_names.assert_called_once_with(["L-filter"])

    def test_update_filter_name_invalid_id(self):
        from citrascope.hardware.direct.direct_adapter import DirectHardwareAdapter

        logger = logging.getLogger("test")
        images_dir = Path("/tmp/test_images")

        with patch("citrascope.hardware.direct.direct_adapter.get_camera_class") as mock_get_cam:
            with patch("citrascope.hardware.direct.direct_adapter.check_dependencies") as mock_check:
                mock_check.return_value = {"available": True, "missing": [], "install_cmd": ""}
                mock_cam_class = MagicMock()
                mock_cam_class.get_friendly_name.return_value = "Test Camera"
                mock_get_cam.return_value = mock_cam_class

                adapter = DirectHardwareAdapter(logger, images_dir, camera_type="moravian")

        adapter.filter_map = {}
        assert adapter.update_filter_name("99", "Nope") is False


class TestAbstractCameraIntegratedFilterWheel:
    """Test the default get_integrated_filter_wheel on AbstractCamera."""

    def test_default_returns_none(self):
        from citrascope.hardware.devices.camera import AbstractCamera

        class DummyCamera(AbstractCamera):
            @classmethod
            def get_friendly_name(cls):
                return "Dummy"

            @classmethod
            def get_dependencies(cls):
                return {"packages": [], "install_extra": ""}

            @classmethod
            def get_settings_schema(cls):
                return []

            def connect(self):
                return True

            def disconnect(self):
                pass

            def is_connected(self):
                return True

            def capture_array(self, duration, gain=None, offset=None, binning=1):
                import numpy as np

                return np.zeros((10, 10), dtype=np.uint16)

            def take_exposure(self, duration, gain=None, offset=None, binning=1, save_path=None):
                return Path("/tmp/test.fits")

            def abort_exposure(self):
                pass

            def get_temperature(self):
                return None

            def set_temperature(self, temperature):
                return False

            def start_cooling(self):
                return False

            def stop_cooling(self):
                return False

            def get_camera_info(self):
                return {}

        cam = DummyCamera(logging.getLogger("test"))
        assert cam.get_integrated_filter_wheel() is None


class TestMoravianDeviceRegistry:
    """Test that Moravian devices are properly registered."""

    def test_moravian_camera_in_registry(self):
        from citrascope.hardware.devices.device_registry import CAMERA_DEVICES

        assert "moravian" in CAMERA_DEVICES
        assert CAMERA_DEVICES["moravian"]["class_name"] == "MoravianCamera"

    def test_moravian_filter_wheel_in_registry(self):
        from citrascope.hardware.devices.device_registry import FILTER_WHEEL_DEVICES

        assert "moravian" in FILTER_WHEEL_DEVICES
        assert FILTER_WHEEL_DEVICES["moravian"]["class_name"] == "MoravianFilterWheel"
