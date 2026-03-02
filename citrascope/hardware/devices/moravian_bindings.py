"""ctypes bindings for the Moravian Instruments gxccd native library.

Wraps the camera_t (gxccd_*) and standalone filter wheel (gxfw_*) C APIs.
Only functions actually used by MoravianCamera and MoravianFilterWheel are
bound here -- extend as needed.

The native library must be installed separately:
  https://www.gxccd.com/cat?id=156&lang=409

macOS note: Gatekeeper will quarantine the downloaded dylib. Remove the
quarantine attribute before first use::

    sudo xattr -d com.apple.quarantine /path/to/libgxccd.dylib

Alternatively, build libgxccd from source -- locally compiled libraries
are not quarantined. This is a development-only concern; Linux deployments
are unaffected.

Based on the official Python example in libgxccd-*/example/test.py and the
header at libgxccd-*/include/gxccd.h.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from ctypes import (
    CFUNCTYPE,
    byref,
    c_bool,
    c_double,
    c_float,
    c_int,
    c_size_t,
    c_uint16,
    c_uint32,
    c_void_p,
    create_string_buffer,
    sizeof,
)

# ---------------------------------------------------------------------------
# C enum constants we actually use (from gxccd.h)
# ---------------------------------------------------------------------------

# gxccd_get_boolean_parameter indexes
GBP_COOLER = 4
GBP_FAN = 5
GBP_FILTERS = 6
GBP_WINDOW_HEATING = 8

# gxccd_get_integer_parameter indexes
GIP_CHIP_W = 1
GIP_CHIP_D = 2
GIP_PIXEL_W = 3
GIP_PIXEL_D = 4
GIP_MAX_BINNING_X = 5
GIP_MAX_BINNING_Y = 6
GIP_READ_MODES = 7
GIP_FILTERS = 8
GIP_DEFAULT_READ_MODE = 12
GIP_MAX_WINDOW_HEATING = 13
GIP_MAX_FAN = 14
GIP_MAX_GAIN = 16

# gxccd_get_string_parameter indexes
GSP_CAMERA_DESCRIPTION = 0
GSP_MANUFACTURER = 1
GSP_CAMERA_SERIAL = 2
GSP_CHIP_DESCRIPTION = 3

# gxccd_get_value indexes
GV_CHIP_TEMPERATURE = 0

# gxfw_get_integer_parameter indexes
FW_GIP_FILTERS = 5

# gxfw_get_string_parameter indexes
FW_GSP_DESCRIPTION = 0
FW_GSP_SERIAL_NUMBER = 2

# Callback type for enumeration
ENUM_CALLBACK = CFUNCTYPE(None, c_int)

_BUF_SIZE = 256


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GxccdLibraryNotFound(ImportError):
    """The native libgxccd shared library could not be found."""


class GxccdError(RuntimeError):
    """A gxccd/gxfw C function returned -1."""


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    lib_name = "libgxccd.dylib" if sys.platform == "darwin" else "libgxccd.so"

    # 1. Explicit env path
    env_path = os.environ.get("GXCCD_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        _lib = ctypes.cdll.LoadLibrary(env_path)
        return _lib

    # 2. System search (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, etc.)
    found = ctypes.util.find_library("gxccd")
    if found:
        _lib = ctypes.cdll.LoadLibrary(found)
        return _lib

    # 3. Well-known paths
    for d in ("/usr/local/lib", "/usr/lib"):
        candidate = os.path.join(d, lib_name)
        if os.path.isfile(candidate):
            _lib = ctypes.cdll.LoadLibrary(candidate)
            return _lib

    system = platform.system()
    hint = (
        f"Download the {'macOS' if system == 'Darwin' else 'Linux'} library from\n"
        "  https://www.gxccd.com/cat?id=156&lang=409\n"
        f"and copy {lib_name} to /usr/local/lib/, or set GXCCD_LIB_PATH."
    )
    raise GxccdLibraryNotFound(f"Could not find {lib_name}.\n{hint}")


def _init_camera_api(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for camera functions we use."""
    lib.gxccd_enumerate_usb.argtypes = [ENUM_CALLBACK]
    lib.gxccd_enumerate_eth.argtypes = [ENUM_CALLBACK]
    lib.gxccd_configure.argtypes = [ctypes.c_char_p]
    lib.gxccd_configure_eth.argtypes = [ctypes.c_char_p, c_uint16]

    lib.gxccd_initialize_usb.argtypes = [c_int]
    lib.gxccd_initialize_usb.restype = c_void_p
    lib.gxccd_initialize_eth.argtypes = [c_int]
    lib.gxccd_initialize_eth.restype = c_void_p
    lib.gxccd_release.argtypes = [c_void_p]

    lib.gxccd_get_boolean_parameter.argtypes = [c_void_p, c_int, c_void_p]
    lib.gxccd_get_integer_parameter.argtypes = [c_void_p, c_int, c_void_p]
    lib.gxccd_get_string_parameter.argtypes = [c_void_p, c_int, c_void_p, c_size_t]
    lib.gxccd_get_value.argtypes = [c_void_p, c_int, c_void_p]

    lib.gxccd_set_temperature.argtypes = [c_void_p, c_float]
    lib.gxccd_set_temperature_ramp.argtypes = [c_void_p, c_float]
    lib.gxccd_set_binning.argtypes = [c_void_p, c_int, c_int]
    lib.gxccd_set_read_mode.argtypes = [c_void_p, c_int]
    lib.gxccd_set_gain.argtypes = [c_void_p, c_uint16]

    lib.gxccd_start_exposure.argtypes = [c_void_p, c_double, c_bool, c_int, c_int, c_int, c_int]
    lib.gxccd_start_exposure.restype = c_int
    lib.gxccd_abort_exposure.argtypes = [c_void_p, c_bool]
    lib.gxccd_image_ready.argtypes = [c_void_p, c_void_p]
    lib.gxccd_read_image.argtypes = [c_void_p, c_void_p, c_size_t]

    lib.gxccd_enumerate_filters.argtypes = [c_void_p, c_int, c_void_p, c_size_t, c_void_p, c_void_p]
    lib.gxccd_set_filter.argtypes = [c_void_p, c_int]
    lib.gxccd_set_fan.argtypes = [c_void_p, ctypes.c_uint8]
    lib.gxccd_set_window_heating.argtypes = [c_void_p, ctypes.c_uint8]

    lib.gxccd_enumerate_read_modes.argtypes = [c_void_p, c_int, c_void_p, c_size_t]
    lib.gxccd_enumerate_read_modes.restype = c_int

    lib.gxccd_get_last_error.argtypes = [c_void_p, c_void_p, c_size_t]


def _init_fw_api(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for standalone filter wheel functions we use."""
    lib.gxfw_enumerate_usb.argtypes = [ENUM_CALLBACK]
    lib.gxfw_enumerate_eth.argtypes = [ENUM_CALLBACK]

    lib.gxfw_initialize_usb.argtypes = [c_int]
    lib.gxfw_initialize_usb.restype = c_void_p
    lib.gxfw_initialize_eth.argtypes = [c_int]
    lib.gxfw_initialize_eth.restype = c_void_p
    lib.gxfw_release.argtypes = [c_void_p]

    lib.gxfw_get_integer_parameter.argtypes = [c_void_p, c_int, c_void_p]
    lib.gxfw_get_string_parameter.argtypes = [c_void_p, c_int, c_void_p, c_size_t]

    lib.gxfw_enumerate_filters.argtypes = [c_void_p, c_int, c_void_p, c_size_t, c_void_p, c_void_p]
    lib.gxfw_set_filter.argtypes = [c_void_p, c_int]
    lib.gxfw_reinit_filter_wheel.argtypes = [c_void_p, c_void_p]

    lib.gxfw_get_last_error.argtypes = [c_void_p, c_void_p, c_size_t]


def get_library() -> ctypes.CDLL:
    """Load and configure the gxccd library (cached after first call)."""
    lib = _load_library()
    _init_camera_api(lib)
    _init_fw_api(lib)
    return lib


# ---------------------------------------------------------------------------
# Helper: check return value and raise on error
# ---------------------------------------------------------------------------


def _last_error(lib: ctypes.CDLL, handle: c_void_p | None, getter_name: str) -> str:
    buf = create_string_buffer(_BUF_SIZE)
    getattr(lib, getter_name)(handle, buf, c_size_t(_BUF_SIZE))
    return buf.value.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# GxccdCamera
# ---------------------------------------------------------------------------


class GxccdCamera:
    """Thin wrapper around a single camera_t* handle.

    Follows the pattern in libgxccd's own test.py with added error checking.
    """

    def __init__(self) -> None:
        self._lib = get_library()
        self._handle: c_void_p | None = None

    @property
    def is_initialized(self) -> bool:
        return self._handle is not None

    def _check(self, ret: int) -> None:
        if ret == -1:
            raise GxccdError(_last_error(self._lib, self._handle, "gxccd_get_last_error"))

    # -- enumerate / connect / disconnect --

    def enumerate_usb(self) -> list[int]:
        ids: list[int] = []
        self._lib.gxccd_enumerate_usb(ENUM_CALLBACK(lambda cid: ids.append(cid)))
        return ids

    def enumerate_eth(self) -> list[int]:
        ids: list[int] = []
        self._lib.gxccd_enumerate_eth(ENUM_CALLBACK(lambda cid: ids.append(cid)))
        return ids

    def configure(self, ini_path: str | None = None) -> None:
        self._lib.gxccd_configure(ini_path.encode() if ini_path else None)

    def configure_eth(self, ip: str | None = None, port: int = 0) -> None:
        self._lib.gxccd_configure_eth(ip.encode() if ip else None, c_uint16(port))

    def initialize_usb(self, camera_id: int = -1) -> None:
        self._handle = self._lib.gxccd_initialize_usb(c_int(camera_id))
        if not self._handle:
            raise GxccdError(f"gxccd_initialize_usb failed (id={camera_id})")

    def initialize_eth(self, camera_id: int = -1) -> None:
        self._handle = self._lib.gxccd_initialize_eth(c_int(camera_id))
        if not self._handle:
            raise GxccdError(f"gxccd_initialize_eth failed (id={camera_id})")

    def release(self) -> None:
        if self._handle:
            self._lib.gxccd_release(self._handle)
            self._handle = None

    # -- parameter getters (mirror test.py pattern) --

    def get_boolean_parameter(self, index: int) -> bool:
        assert self._handle is not None
        v = c_bool()
        self._check(self._lib.gxccd_get_boolean_parameter(self._handle, index, byref(v)))
        return v.value

    def get_integer_parameter(self, index: int) -> int:
        assert self._handle is not None
        v = c_int()
        self._check(self._lib.gxccd_get_integer_parameter(self._handle, index, byref(v)))
        return v.value

    def get_string_parameter(self, index: int) -> str:
        assert self._handle is not None
        buf = create_string_buffer(_BUF_SIZE)
        self._check(self._lib.gxccd_get_string_parameter(self._handle, index, buf, sizeof(buf) - 1))
        return buf.value.decode("utf-8", errors="replace")

    def get_value(self, index: int) -> float:
        assert self._handle is not None
        v = c_float()
        self._check(self._lib.gxccd_get_value(self._handle, index, byref(v)))
        return v.value

    # -- temperature / thermal --

    def set_temperature(self, temp_celsius: float) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_temperature(self._handle, c_float(temp_celsius)))

    def set_temperature_ramp(self, degrees_per_min: float) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_temperature_ramp(self._handle, c_float(degrees_per_min)))

    def set_fan(self, speed: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_fan(self._handle, ctypes.c_uint8(speed)))

    def set_window_heating(self, intensity: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_window_heating(self._handle, ctypes.c_uint8(intensity)))

    # -- imaging --

    def set_binning(self, x: int, y: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_binning(self._handle, x, y))

    def set_read_mode(self, mode: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_read_mode(self._handle, mode))

    def set_gain(self, gain: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_gain(self._handle, c_uint16(gain)))

    def start_exposure(self, exp_time: float, use_shutter: bool, x: int, y: int, w: int, h: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_start_exposure(self._handle, c_double(exp_time), c_bool(use_shutter), x, y, w, h))

    def abort_exposure(self, download: bool = False) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_abort_exposure(self._handle, c_bool(download)))

    def image_ready(self) -> bool:
        assert self._handle is not None
        ready = c_bool()
        self._check(self._lib.gxccd_image_ready(self._handle, byref(ready)))
        return ready.value

    def read_image(self, buf: ctypes.Array, size: int) -> None:  # type: ignore[type-arg]
        assert self._handle is not None
        self._check(self._lib.gxccd_read_image(self._handle, buf, c_size_t(size)))

    # -- read modes --

    def enumerate_read_modes(self) -> list[str]:
        """Returns list of read mode names, indexed by position."""
        assert self._handle is not None
        modes: list[str] = []
        idx = 0
        while True:
            buf = create_string_buffer(_BUF_SIZE)
            ret = self._lib.gxccd_enumerate_read_modes(self._handle, idx, buf, sizeof(buf) - 1)
            if ret == -1:
                break
            modes.append(buf.value.decode("utf-8", errors="replace"))
            idx += 1
        return modes

    # -- integrated filter wheel --

    def enumerate_filters(self) -> list[tuple[str, int, int]]:
        """Returns list of (name, color_rgb, focus_offset)."""
        assert self._handle is not None
        filters: list[tuple[str, int, int]] = []
        idx = 0
        while True:
            buf = create_string_buffer(_BUF_SIZE)
            color = c_uint32()
            offset = c_int()
            ret = self._lib.gxccd_enumerate_filters(
                self._handle, idx, buf, sizeof(buf) - 1, byref(color), byref(offset)
            )
            if ret == -1:
                break
            filters.append((buf.value.decode("utf-8", errors="replace"), color.value, offset.value))
            idx += 1
        return filters

    def set_filter(self, index: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxccd_set_filter(self._handle, index))


# ---------------------------------------------------------------------------
# GxccdFilterWheel (standalone external)
# ---------------------------------------------------------------------------


class GxccdFilterWheel:
    """Thin wrapper around a single fwheel_t* handle."""

    def __init__(self) -> None:
        self._lib = get_library()
        self._handle: c_void_p | None = None

    @property
    def is_initialized(self) -> bool:
        return self._handle is not None

    def _check(self, ret: int) -> None:
        if ret == -1:
            raise GxccdError(_last_error(self._lib, self._handle, "gxfw_get_last_error"))

    # -- enumerate / connect / disconnect --

    def enumerate_usb(self) -> list[int]:
        ids: list[int] = []
        self._lib.gxfw_enumerate_usb(ENUM_CALLBACK(lambda cid: ids.append(cid)))
        return ids

    def enumerate_eth(self) -> list[int]:
        ids: list[int] = []
        self._lib.gxfw_enumerate_eth(ENUM_CALLBACK(lambda cid: ids.append(cid)))
        return ids

    def initialize_usb(self, wheel_id: int = -1) -> None:
        self._handle = self._lib.gxfw_initialize_usb(c_int(wheel_id))
        if not self._handle:
            raise GxccdError(f"gxfw_initialize_usb failed (id={wheel_id})")

    def initialize_eth(self, wheel_id: int = -1) -> None:
        self._handle = self._lib.gxfw_initialize_eth(c_int(wheel_id))
        if not self._handle:
            raise GxccdError(f"gxfw_initialize_eth failed (id={wheel_id})")

    def release(self) -> None:
        if self._handle:
            self._lib.gxfw_release(self._handle)
            self._handle = None

    # -- parameter getters --

    def get_integer_parameter(self, index: int) -> int:
        assert self._handle is not None
        v = c_int()
        self._check(self._lib.gxfw_get_integer_parameter(self._handle, index, byref(v)))
        return v.value

    def get_string_parameter(self, index: int) -> str:
        assert self._handle is not None
        buf = create_string_buffer(_BUF_SIZE)
        self._check(self._lib.gxfw_get_string_parameter(self._handle, index, buf, sizeof(buf) - 1))
        return buf.value.decode("utf-8", errors="replace")

    # -- filters --

    def enumerate_filters(self) -> list[tuple[str, int, int]]:
        """Returns list of (name, color_rgb, focus_offset)."""
        assert self._handle is not None
        filters: list[tuple[str, int, int]] = []
        idx = 0
        while True:
            buf = create_string_buffer(_BUF_SIZE)
            color = c_uint32()
            offset = c_int()
            ret = self._lib.gxfw_enumerate_filters(self._handle, idx, buf, sizeof(buf) - 1, byref(color), byref(offset))
            if ret == -1:
                break
            filters.append((buf.value.decode("utf-8", errors="replace"), color.value, offset.value))
            idx += 1
        return filters

    def set_filter(self, index: int) -> None:
        assert self._handle is not None
        self._check(self._lib.gxfw_set_filter(self._handle, index))

    def reinit_filter_wheel(self) -> int:
        """Returns number of detected filters."""
        assert self._handle is not None
        num = c_int()
        self._check(self._lib.gxfw_reinit_filter_wheel(self._handle, byref(num)))
        return num.value
