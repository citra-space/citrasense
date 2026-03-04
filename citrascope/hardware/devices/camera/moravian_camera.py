"""Moravian Instruments camera driver and integrated filter wheel.

Supports Gx (CCD) and Cx (CMOS) cameras via the native gxccd library.
The integrated filter wheel (if present) is auto-detected on connect and
exposed via get_integrated_filter_wheel() for DirectHardwareAdapter.
"""

from __future__ import annotations

import logging
import time
from ctypes import create_string_buffer
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from citrascope.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrascope.hardware.devices.camera import AbstractCamera
from citrascope.hardware.devices.filter_wheel import AbstractFilterWheel

if TYPE_CHECKING:
    from citrascope.hardware.devices.moravian_bindings import GxccdCamera


def _probe_moravian_cameras() -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Enumerate Moravian cameras via the native SDK.

    Standalone module-level function so it can be pickled and sent to a
    subprocess by :func:`run_hardware_probe`.  Returns ``(options, read_modes)``.
    """
    from citrascope.hardware.devices.moravian_bindings import (
        GSP_CAMERA_DESCRIPTION,
        GSP_CAMERA_SERIAL,
    )
    from citrascope.hardware.devices.moravian_bindings import (
        GxccdCamera as GxccdCam,
    )

    options: list[dict[str, str | int]] = [{"value": -1, "label": "Auto (single camera)"}]
    read_modes: list[dict[str, str | int]] = [{"value": -1, "label": "Camera default"}]

    probe = GxccdCam()
    ids = probe.enumerate_usb()
    for cid in ids:
        try:
            cam = GxccdCam()
            cam.initialize_usb(cid)
            try:
                desc = cam.get_string_parameter(GSP_CAMERA_DESCRIPTION).strip()
                serial = cam.get_string_parameter(GSP_CAMERA_SERIAL).strip()
                if len(read_modes) == 1:
                    try:
                        modes = cam.enumerate_read_modes()
                        for i, name in enumerate(modes):
                            read_modes.append({"value": i, "label": name})
                    except Exception:
                        pass
                label = f"{desc} (SN: {serial})" if serial else desc
                options.append({"value": cid, "label": label})
            finally:
                cam.release()
        except Exception:
            options.append({"value": cid, "label": f"Camera {cid}"})

    return options, read_modes


_CAMERA_FALLBACK: list[dict[str, str | int]] = [{"value": -1, "label": "Auto (single camera)"}]
_READ_MODE_FALLBACK: list[dict[str, str | int]] = [{"value": -1, "label": "Camera default"}]


class MoravianCamera(AbstractCamera):
    """Driver for Moravian Instruments Gx/Cx cameras via gxccd native library."""

    _read_mode_cache: list[dict[str, str | int]] | None = None

    @classmethod
    def get_friendly_name(cls) -> str:
        return "Moravian Instruments Camera (Gx/Cx)"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {"packages": [], "install_extra": ""}

    @classmethod
    def _detect_available_cameras(cls) -> list[dict[str, str | int]]:
        """Probe connected Moravian cameras and return {value, label} options.

        Uses :meth:`_cached_hardware_probe` for subprocess isolation and
        caching so a hung native call cannot freeze the web server.
        """
        options, read_modes = cls._cached_hardware_probe(
            _probe_moravian_cameras,
            fallback=(_CAMERA_FALLBACK, _READ_MODE_FALLBACK),
            cache_ttl=30.0,
            timeout=10.0,
        )

        if len(read_modes) > 1:
            cls._read_mode_cache = read_modes
        return options

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        camera_options = cls._detect_available_cameras()
        read_mode_options = cls._read_mode_cache or [{"value": -1, "label": "Camera default"}]

        return cast(
            list[SettingSchemaEntry],
            [
                {
                    "name": "camera_id",
                    "friendly_name": "Camera",
                    "type": "int",
                    "default": -1,
                    "description": "Select which camera to use",
                    "required": False,
                    "options": camera_options,
                    "group": "Camera",
                },
                {
                    "name": "connection_type",
                    "friendly_name": "Connection Type",
                    "type": "str",
                    "default": "usb",
                    "description": "USB or Ethernet connection",
                    "required": False,
                    "options": ["usb", "eth"],
                    "group": "Camera",
                },
                {
                    "name": "eth_ip",
                    "friendly_name": "Ethernet Adapter IP",
                    "type": "str",
                    "default": "192.168.0.5",
                    "description": "IP address of the Moravian Camera Ethernet Adapter",
                    "required": False,
                    "visible_when": {"field": "connection_type", "value": "eth"},
                    "group": "Camera",
                },
                {
                    "name": "eth_port",
                    "friendly_name": "Ethernet Adapter Port",
                    "type": "int",
                    "default": 48899,
                    "description": "Port of the Moravian Camera Ethernet Adapter",
                    "required": False,
                    "visible_when": {"field": "connection_type", "value": "eth"},
                    "group": "Camera",
                },
                {
                    "name": "default_read_mode",
                    "friendly_name": "Read Mode",
                    "type": "int",
                    "default": -1,
                    "description": "Sensor read mode",
                    "required": False,
                    "options": read_mode_options,
                    "group": "Camera",
                },
                {
                    "name": "default_gain",
                    "friendly_name": "Default Gain",
                    "type": "int",
                    "default": 0,
                    "description": "Default gain register value (0 = minimum, max depends on camera model)",
                    "required": False,
                    "min": 0,
                    "group": "Camera",
                },
                {
                    "name": "default_binning",
                    "friendly_name": "Default Binning",
                    "type": "int",
                    "default": 1,
                    "description": "Pixel binning factor (1x1, 2x2, etc.)",
                    "required": False,
                    "options": [
                        {"value": 1, "label": "1x1 (no binning)"},
                        {"value": 2, "label": "2x2"},
                        {"value": 3, "label": "3x3"},
                        {"value": 4, "label": "4x4"},
                    ],
                    "group": "Camera",
                },
                {
                    "name": "cooling_target_temp",
                    "friendly_name": "Cooling Target (°C)",
                    "type": "float",
                    "default": -10.0,
                    "description": "Target sensor temperature. Set to 20 to disable cooling.",
                    "required": False,
                    "min": -50,
                    "max": 20,
                    "group": "Camera",
                },
            ],
        )

    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        self._camera_id: int = int(kwargs.get("camera_id", -1))
        self._connection_type: str = kwargs.get("connection_type", "usb")
        self._eth_ip: str = kwargs.get("eth_ip", "192.168.0.5")
        self._eth_port: int = int(kwargs.get("eth_port", 48899))
        self._default_read_mode: int = int(kwargs.get("default_read_mode", -1))
        self._default_gain: int = int(kwargs.get("default_gain", 0))
        self._default_binning: int = int(kwargs.get("default_binning", 1))
        self._cooling_target_temp: float = float(kwargs.get("cooling_target_temp", -10.0))

        self._gxccd: GxccdCamera | None = None
        self._has_cooler = False
        self._has_fan = False
        self._has_window_heating = False
        self._max_fan: int = 0
        self._max_window_heating: int = 0
        self._cooling_active = False
        self._target_temp: float | None = None
        self._read_modes: list[str] = []
        self._camera_info: dict = {}
        self._integrated_fw: MoravianIntegratedFilterWheel | None = None

    def connect(self) -> bool:
        try:
            from citrascope.hardware.devices.moravian_bindings import (
                GBP_COOLER,
                GBP_FAN,
                GBP_FILTERS,
                GBP_WINDOW_HEATING,
                GIP_CHIP_D,
                GIP_CHIP_W,
                GIP_DEFAULT_READ_MODE,
                GIP_FILTERS,
                GIP_MAX_FAN,
                GIP_MAX_GAIN,
                GIP_MAX_WINDOW_HEATING,
                GIP_PIXEL_D,
                GIP_PIXEL_W,
                GSP_CAMERA_DESCRIPTION,
                GSP_CAMERA_SERIAL,
                GSP_CHIP_DESCRIPTION,
                GV_CHIP_TEMPERATURE,
                GxccdCamera,
                GxccdError,
            )
        except ImportError as e:
            self.logger.error(f"Moravian native library not available: {e}")
            return False

        try:
            cam = GxccdCamera()
            if self._connection_type == "eth":
                cam.configure_eth(self._eth_ip, self._eth_port)
                cam.initialize_eth(self._camera_id)
            else:
                cam.initialize_usb(self._camera_id)

            self._gxccd = cam

            # Read camera info
            desc = cam.get_string_parameter(GSP_CAMERA_DESCRIPTION)
            serial = cam.get_string_parameter(GSP_CAMERA_SERIAL)
            chip = cam.get_string_parameter(GSP_CHIP_DESCRIPTION)
            w = cam.get_integer_parameter(GIP_CHIP_W)
            h = cam.get_integer_parameter(GIP_CHIP_D)
            pixel_w_nm = cam.get_integer_parameter(GIP_PIXEL_W)
            pixel_h_nm = cam.get_integer_parameter(GIP_PIXEL_D)
            max_gain = cam.get_integer_parameter(GIP_MAX_GAIN)

            self._camera_info = {
                "model": desc.strip(),
                "serial_number": serial.strip(),
                "chip": chip.strip(),
                "width": w,
                "height": h,
                "pixel_size_um": round(pixel_w_nm / 1000.0, 3),
                "pixel_size_h_um": round(pixel_h_nm / 1000.0, 3),
                "bit_depth": 16,
                "max_gain": max_gain,
            }

            # Cooling and thermal capabilities
            self._has_cooler = cam.get_boolean_parameter(GBP_COOLER)
            self._has_fan = cam.get_boolean_parameter(GBP_FAN)
            self._has_window_heating = cam.get_boolean_parameter(GBP_WINDOW_HEATING)
            if self._has_fan:
                self._max_fan = cam.get_integer_parameter(GIP_MAX_FAN)
            if self._has_window_heating:
                self._max_window_heating = cam.get_integer_parameter(GIP_MAX_WINDOW_HEATING)

            # Enumerate read modes and push to class-level cache for the settings schema
            try:
                self._read_modes = cam.enumerate_read_modes()
                if self._read_modes:
                    for i, mode_name in enumerate(self._read_modes):
                        self.logger.info(f"  Read mode {i}: {mode_name}")
                    MoravianCamera._read_mode_cache = [{"value": -1, "label": "Camera default"}] + [
                        {"value": i, "label": name} for i, name in enumerate(self._read_modes)
                    ]
            except Exception as e:
                self.logger.debug(f"Could not enumerate read modes: {e}")

            # Apply default read mode
            if self._default_read_mode >= 0:
                cam.set_read_mode(self._default_read_mode)
                mode_label = (
                    self._read_modes[self._default_read_mode]
                    if self._default_read_mode < len(self._read_modes)
                    else str(self._default_read_mode)
                )
                self.logger.info(f"Read mode set to {mode_label}")
            else:
                default = cam.get_integer_parameter(GIP_DEFAULT_READ_MODE)
                cam.set_read_mode(default)

            # Integrated filter wheel
            has_filters = cam.get_boolean_parameter(GBP_FILTERS)
            if has_filters:
                num_filters = cam.get_integer_parameter(GIP_FILTERS)
                self._integrated_fw = MoravianIntegratedFilterWheel(cam, num_filters, self.logger)
                self._integrated_fw.connect()
                self.logger.info(f"Detected integrated filter wheel with {num_filters} positions")

            temp = cam.get_value(GV_CHIP_TEMPERATURE)
            self.logger.info(
                f"Connected to {desc.strip()} (SN: {serial.strip()}) "
                f"{w}x{h} px, {temp:.1f}°C"
                + (f", fan: max={self._max_fan}" if self._has_fan else "")
                + (f", heating: max={self._max_window_heating}" if self._has_window_heating else "")
            )

            # Auto-start cooling if cooler present and target is below ambient
            if self._has_cooler and self._cooling_target_temp < 20.0:
                self.start_cooling()

            return True

        except (GxccdError, OSError) as e:
            self.logger.error(f"Failed to connect to Moravian camera: {e}")
            self._gxccd = None
            return False

    def disconnect(self):
        if self._gxccd and self._cooling_active:
            try:
                self.stop_cooling()
            except Exception as e:
                self.logger.warning(f"Error stopping cooling during disconnect: {e}")
        self._integrated_fw = None
        if self._gxccd:
            self._gxccd.release()
            self._gxccd = None
            self.logger.info("Moravian camera disconnected")

    def is_connected(self) -> bool:
        return self._gxccd is not None and self._gxccd.is_initialized

    def get_integrated_filter_wheel(self) -> AbstractFilterWheel | None:
        return self._integrated_fw

    # -- Exposure --

    def capture_array(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
    ) -> np.ndarray:
        if not self.is_connected():
            raise RuntimeError("Camera not connected")
        assert self._gxccd is not None

        cam = self._gxccd
        w = self._camera_info["width"]
        h = self._camera_info["height"]

        gain_val = gain if gain is not None else self._default_gain
        cam.set_gain(gain_val)
        cam.set_binning(binning, binning)

        self.logger.debug(f"Starting {duration}s exposure (gain={gain_val}, binning={binning}x{binning})")
        cam.start_exposure(duration, True, 0, 0, w, h)

        # Sleep for most of the exposure, then poll (per gxccd.h recommendation)
        if duration > 0.5:
            time.sleep(duration - 0.2)
        while not cam.image_ready():
            time.sleep(0.05)

        binned_w = w // binning
        binned_h = h // binning
        buf_size = binned_w * 2 * binned_h
        buf = create_string_buffer(buf_size)
        cam.read_image(buf, buf_size)

        return np.frombuffer(bytes(buf), dtype=np.uint16).reshape((binned_h, binned_w))

    def take_exposure(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        save_path: Path | None = None,
    ) -> Path:
        data = self.capture_array(duration, gain, offset, binning)

        if save_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"moravian_{ts}.fits")

        gain_val = gain if gain is not None else self._default_gain
        self._save_fits(data.tobytes(), data.shape[1], data.shape[0], duration, gain_val, binning, save_path)
        self.logger.info(f"Image saved to {save_path}")
        return save_path

    def _save_fits(
        self,
        buf: bytes,
        width: int,
        height: int,
        exp_time: float,
        gain: int,
        binning: int,
        save_path: Path,
    ) -> None:
        import numpy as np
        from astropy.io import fits

        # gxccd returns 16-bit LE, bottom-up (Cartesian origin)
        data = np.frombuffer(buf, dtype=np.uint16).reshape((height, width))
        # FITS convention: first pixel is bottom-left, same as gxccd -- no flip needed

        hdu = fits.PrimaryHDU(data)
        hdr = hdu.header
        hdr["EXPTIME"] = (exp_time, "Exposure time in seconds")
        hdr["GAIN"] = (gain, "Camera gain register value")
        hdr["XBINNING"] = (binning, "Horizontal binning")
        hdr["YBINNING"] = (binning, "Vertical binning")
        hdr["DATE-OBS"] = (datetime.now(timezone.utc).isoformat(), "UTC observation time")
        hdr["INSTRUME"] = (self._camera_info.get("model", ""), "Camera model")

        serial = self._camera_info.get("serial_number", "")
        if serial:
            hdr["CAMSER"] = (serial, "Camera serial number")

        temp = self.get_temperature()
        if temp is not None:
            hdr["CCD-TEMP"] = (temp, "Sensor temperature in C")

        hdu.writeto(save_path, overwrite=True)

    def abort_exposure(self):
        if self.is_connected() and self._gxccd:
            try:
                self._gxccd.abort_exposure(download=False)
                self.logger.info("Exposure aborted")
            except Exception as e:
                self.logger.error(f"Error aborting exposure: {e}")

    # -- Temperature --

    def get_temperature(self) -> float | None:
        if not self.is_connected():
            return None
        assert self._gxccd is not None
        try:
            from citrascope.hardware.devices.moravian_bindings import GV_CHIP_TEMPERATURE

            return self._gxccd.get_value(GV_CHIP_TEMPERATURE)
        except Exception:
            return None

    def set_temperature(self, temperature: float) -> bool:
        if not self.is_connected() or not self._has_cooler:
            return False
        assert self._gxccd is not None
        try:
            self._gxccd.set_temperature(temperature)
            self._target_temp = temperature
            self._cooling_active = True
            self.logger.info(f"Target temperature set to {temperature}°C")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set temperature: {e}")
            return False

    def start_cooling(self) -> bool:
        """Start the full thermal lifecycle: fan, window heating, ramp, and target temp."""
        if not self._has_cooler or not self.is_connected():
            return False
        assert self._gxccd is not None

        try:
            if self._has_fan:
                self._gxccd.set_fan(self._max_fan)
                self.logger.info(f"Fan enabled at speed {self._max_fan}")

            if self._has_window_heating:
                self._gxccd.set_window_heating(self._max_window_heating)
                self.logger.info(f"Window heating enabled at intensity {self._max_window_heating}")

            self._gxccd.set_temperature_ramp(3.0)
            self.logger.info("Temperature ramp set to 3 °C/min")

            target = self._cooling_target_temp
            self._gxccd.set_temperature(target)
            self._target_temp = target
            self._cooling_active = True
            self.logger.info(f"Cooling started, target {target}°C")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start cooling: {e}")
            return False

    def stop_cooling(self) -> bool:
        """Ramp to ambient and disable fan/heating."""
        if not self.is_connected() or not self._has_cooler:
            return False
        assert self._gxccd is not None

        try:
            self._gxccd.set_temperature_ramp(3.0)
            self._gxccd.set_temperature(20.0)
            self.logger.info("Cooling ramp-down to +20°C started")

            if self._has_fan:
                self._gxccd.set_fan(0)
                self.logger.info("Fan disabled")

            if self._has_window_heating:
                self._gxccd.set_window_heating(0)
                self.logger.info("Window heating disabled")

            self._cooling_active = False
            self._target_temp = None
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop cooling: {e}")
            return False

    def get_default_binning(self) -> int:
        return self._default_binning

    def get_camera_info(self) -> dict:
        return self._camera_info.copy()


# ---------------------------------------------------------------------------
# Integrated filter wheel (shares the camera_t handle)
# ---------------------------------------------------------------------------


class MoravianIntegratedFilterWheel(AbstractFilterWheel):
    """Filter wheel built into a Moravian camera, controlled via gxccd_set_filter().

    Not registered in the device registry -- created by MoravianCamera when
    GBP_FILTERS is true, and returned via get_integrated_filter_wheel().
    """

    def __init__(self, gxccd_camera: GxccdCamera, num_filters: int, logger: logging.Logger):
        # Skip AbstractHardwareDevice.__init__ kwargs since we're not registry-created
        self.logger = logger
        self._cam = gxccd_camera
        self._num_filters = num_filters
        self._filter_names: list[str] = []
        self._position: int = 0

    # -- Registry stubs (never called via registry) --

    @classmethod
    def get_friendly_name(cls) -> str:
        return "Moravian Integrated Filter Wheel"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {"packages": [], "install_extra": ""}

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        return []

    # -- Lifecycle --

    def connect(self) -> bool:
        try:
            filters = self._cam.enumerate_filters()
            self._filter_names = [name for name, _color, _offset in filters]
            self._num_filters = len(filters) if filters else self._num_filters
            # Pad with default names if .ini not configured
            while len(self._filter_names) < self._num_filters:
                self._filter_names.append(f"Filter {len(self._filter_names) + 1}")
            self.logger.info(f"Filter wheel: {self._filter_names}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not enumerate filters: {e}")
            self._filter_names = [f"Filter {i + 1}" for i in range(self._num_filters)]
            return True

    def disconnect(self):
        pass  # Camera owns the handle

    def is_connected(self) -> bool:
        return self._cam.is_initialized

    # -- Filter operations --

    def set_filter_position(self, position: int) -> bool:
        try:
            self._cam.set_filter(position)
            self._position = position
            return True
        except Exception as e:
            self.logger.error(f"Failed to set filter position {position}: {e}")
            return False

    def get_filter_position(self) -> int | None:
        return self._position

    def is_moving(self) -> bool:
        return False  # gxccd_set_filter blocks until complete

    def get_filter_count(self) -> int:
        return self._num_filters

    def get_filter_names(self) -> list[str]:
        return list(self._filter_names)

    def set_filter_names(self, names: list[str]) -> bool:
        if len(names) != self._num_filters:
            return False
        self._filter_names = list(names)
        return True
