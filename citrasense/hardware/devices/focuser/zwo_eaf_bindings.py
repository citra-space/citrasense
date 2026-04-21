"""ctypes bindings for the ZWO EAF (Electronic Automatic Focuser) native SDK.

Wraps the EAF_focuser.h C API. Only functions actually used by ZwoEafFocuser
are bound here -- extend as needed.

The native library must be installed separately. Options:
  - Linux: ``sudo apt install libasi`` (from INDI PPA)
  - macOS/manual: download from https://astronomy-imaging-camera.com/software-drivers
    (Developers tab) and copy libEAFFocuser.dylib to /usr/local/lib/

macOS note: Gatekeeper will quarantine the downloaded dylib. Remove the
quarantine attribute before first use::

    sudo xattr -d com.apple.quarantine /path/to/libEAFFocuser.dylib

Based on the official SDK header EAF_focuser.h from
https://github.com/indilib/indi-3rdparty/blob/master/libasi/EAF_focuser.h
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from ctypes import (
    Structure,
    byref,
    c_bool,
    c_char,
    c_float,
    c_int,
    c_ubyte,
)

# ---------------------------------------------------------------------------
# C enum: EAF_ERROR_CODE
# ---------------------------------------------------------------------------

EAF_SUCCESS = 0
EAF_ERROR_INVALID_INDEX = 1
EAF_ERROR_INVALID_ID = 2
EAF_ERROR_INVALID_VALUE = 3
EAF_ERROR_REMOVED = 4
EAF_ERROR_MOVING = 5
EAF_ERROR_ERROR_STATE = 6
EAF_ERROR_GENERAL_ERROR = 7
EAF_ERROR_NOT_SUPPORTED = 8
EAF_ERROR_CLOSED = 9

_ERROR_NAMES: dict[int, str] = {
    EAF_SUCCESS: "EAF_SUCCESS",
    EAF_ERROR_INVALID_INDEX: "EAF_ERROR_INVALID_INDEX",
    EAF_ERROR_INVALID_ID: "EAF_ERROR_INVALID_ID",
    EAF_ERROR_INVALID_VALUE: "EAF_ERROR_INVALID_VALUE",
    EAF_ERROR_REMOVED: "EAF_ERROR_REMOVED",
    EAF_ERROR_MOVING: "EAF_ERROR_MOVING",
    EAF_ERROR_ERROR_STATE: "EAF_ERROR_ERROR_STATE",
    EAF_ERROR_GENERAL_ERROR: "EAF_ERROR_GENERAL_ERROR",
    EAF_ERROR_NOT_SUPPORTED: "EAF_ERROR_NOT_SUPPORTED",
    EAF_ERROR_CLOSED: "EAF_ERROR_CLOSED",
}


# ---------------------------------------------------------------------------
# C struct: EAF_INFO
# ---------------------------------------------------------------------------


class EAF_INFO(Structure):
    _fields_ = [
        ("ID", c_int),
        ("Name", c_char * 64),
        ("MaxStep", c_int),
    ]


# ---------------------------------------------------------------------------
# C struct: EAF_ID (also used as EAF_SN)
# ---------------------------------------------------------------------------


class EAF_ID(Structure):
    _fields_ = [
        ("id", c_ubyte * 8),
    ]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EafLibraryNotFound(ImportError):
    """The native libEAFFocuser shared library could not be found."""


class EafError(RuntimeError):
    """An EAF SDK function returned an error code."""

    def __init__(self, func_name: str, error_code: int) -> None:
        self.error_code = error_code
        name = _ERROR_NAMES.get(error_code, f"UNKNOWN({error_code})")
        super().__init__(f"{func_name} failed: {name}")


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    lib_name = "libEAFFocuser.dylib" if sys.platform == "darwin" else "libEAFFocuser.so"

    # 1. Explicit env path
    env_path = os.environ.get("EAF_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        _lib = ctypes.cdll.LoadLibrary(env_path)
        return _lib

    # 2. System search (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, etc.)
    found = ctypes.util.find_library("EAFFocuser")
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
        f"Download the {'macOS' if system == 'Darwin' else 'Linux'} SDK from\n"
        "  https://astronomy-imaging-camera.com/software-drivers\n"
        f"and copy {lib_name} to /usr/local/lib/, or set EAF_LIB_PATH.\n"
        "On Linux you can also install via: sudo apt install libasi"
    )
    raise EafLibraryNotFound(f"Could not find {lib_name}.\n{hint}")


def _init_api(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for all EAF functions we use."""
    # -- Lifecycle --
    lib.EAFGetNum.argtypes = []
    lib.EAFGetNum.restype = c_int

    lib.EAFGetID.argtypes = [c_int, ctypes.POINTER(c_int)]
    lib.EAFGetID.restype = c_int

    lib.EAFGetProperty.argtypes = [c_int, ctypes.POINTER(EAF_INFO)]
    lib.EAFGetProperty.restype = c_int

    lib.EAFOpen.argtypes = [c_int]
    lib.EAFOpen.restype = c_int

    lib.EAFClose.argtypes = [c_int]
    lib.EAFClose.restype = c_int

    lib.EAFGetSDKVersion.argtypes = []
    lib.EAFGetSDKVersion.restype = ctypes.c_char_p

    # -- Movement --
    lib.EAFMove.argtypes = [c_int, c_int]
    lib.EAFMove.restype = c_int

    lib.EAFStop.argtypes = [c_int]
    lib.EAFStop.restype = c_int

    lib.EAFIsMoving.argtypes = [c_int, ctypes.POINTER(c_bool), ctypes.POINTER(c_bool)]
    lib.EAFIsMoving.restype = c_int

    lib.EAFGetPosition.argtypes = [c_int, ctypes.POINTER(c_int)]
    lib.EAFGetPosition.restype = c_int

    # Note: the SDK has a typo -- "Postion" not "Position"
    lib.EAFResetPostion.argtypes = [c_int, c_int]
    lib.EAFResetPostion.restype = c_int

    # -- Configuration --
    lib.EAFGetMaxStep.argtypes = [c_int, ctypes.POINTER(c_int)]
    lib.EAFGetMaxStep.restype = c_int

    lib.EAFSetMaxStep.argtypes = [c_int, c_int]
    lib.EAFSetMaxStep.restype = c_int

    lib.EAFStepRange.argtypes = [c_int, ctypes.POINTER(c_int)]
    lib.EAFStepRange.restype = c_int

    lib.EAFGetReverse.argtypes = [c_int, ctypes.POINTER(c_bool)]
    lib.EAFGetReverse.restype = c_int

    lib.EAFSetReverse.argtypes = [c_int, c_bool]
    lib.EAFSetReverse.restype = c_int

    lib.EAFGetBacklash.argtypes = [c_int, ctypes.POINTER(c_int)]
    lib.EAFGetBacklash.restype = c_int

    lib.EAFSetBacklash.argtypes = [c_int, c_int]
    lib.EAFSetBacklash.restype = c_int

    lib.EAFSetBeep.argtypes = [c_int, c_bool]
    lib.EAFSetBeep.restype = c_int

    lib.EAFGetBeep.argtypes = [c_int, ctypes.POINTER(c_bool)]
    lib.EAFGetBeep.restype = c_int

    # -- Info --
    lib.EAFGetTemp.argtypes = [c_int, ctypes.POINTER(c_float)]
    lib.EAFGetTemp.restype = c_int

    lib.EAFGetFirmwareVersion.argtypes = [
        c_int,
        ctypes.POINTER(c_ubyte),
        ctypes.POINTER(c_ubyte),
        ctypes.POINTER(c_ubyte),
    ]
    lib.EAFGetFirmwareVersion.restype = c_int

    lib.EAFGetSerialNumber.argtypes = [c_int, ctypes.POINTER(EAF_ID)]
    lib.EAFGetSerialNumber.restype = c_int


def get_library() -> ctypes.CDLL:
    """Load and configure the EAF library (cached after first call)."""
    lib = _load_library()
    _init_api(lib)
    return lib


# ---------------------------------------------------------------------------
# Helper: check return code
# ---------------------------------------------------------------------------


def _check(func_name: str, ret: int) -> None:
    if ret != EAF_SUCCESS:
        raise EafError(func_name, ret)


# ---------------------------------------------------------------------------
# EafFocuser -- Pythonic wrapper around a single EAF device
# ---------------------------------------------------------------------------


class EafFocuser:
    """Thin wrapper around a single EAF focuser, identified by SDK ID.

    Usage::

        eaf = EafFocuser()
        count = eaf.get_num()
        eaf_id = eaf.get_id(0)
        info = eaf.get_property(eaf_id)
        eaf.open(eaf_id)
        eaf.move(50000)
        pos = eaf.get_position()
        eaf.close()
    """

    def __init__(self) -> None:
        self._lib = get_library()
        self._id: int | None = None

    @property
    def is_open(self) -> bool:
        return self._id is not None

    # -- Lifecycle --

    def get_num(self) -> int:
        """Get number of connected EAF focusers (refreshes device list)."""
        return self._lib.EAFGetNum()

    def get_id(self, index: int) -> int:
        """Get the SDK ID for the focuser at the given index."""
        eaf_id = c_int()
        _check("EAFGetID", self._lib.EAFGetID(index, byref(eaf_id)))
        return eaf_id.value

    def get_property(self, eaf_id: int) -> EAF_INFO:
        """Get property struct for a focuser (works before open)."""
        info = EAF_INFO()
        _check("EAFGetProperty", self._lib.EAFGetProperty(eaf_id, byref(info)))
        return info

    def open(self, eaf_id: int) -> None:
        """Open a focuser by its SDK ID."""
        _check("EAFOpen", self._lib.EAFOpen(eaf_id))
        self._id = eaf_id

    def close(self) -> None:
        """Close the currently open focuser."""
        if self._id is not None:
            _check("EAFClose", self._lib.EAFClose(self._id))
            self._id = None

    def get_sdk_version(self) -> str:
        raw = self._lib.EAFGetSDKVersion()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    # -- Movement --

    def move(self, step: int) -> None:
        """Move to an absolute step position."""
        assert self._id is not None
        _check("EAFMove", self._lib.EAFMove(self._id, step))

    def stop(self) -> None:
        """Stop any current movement."""
        assert self._id is not None
        _check("EAFStop", self._lib.EAFStop(self._id))

    def is_moving(self) -> tuple[bool, bool]:
        """Check movement state. Returns (is_moving, is_hand_control)."""
        assert self._id is not None
        moving = c_bool()
        hand_ctrl = c_bool()
        _check("EAFIsMoving", self._lib.EAFIsMoving(self._id, byref(moving), byref(hand_ctrl)))
        return moving.value, hand_ctrl.value

    def get_position(self) -> int:
        """Get current focuser position in steps."""
        assert self._id is not None
        pos = c_int()
        _check("EAFGetPosition", self._lib.EAFGetPosition(self._id, byref(pos)))
        return pos.value

    def reset_position(self, step: int) -> None:
        """Set current position label without moving the motor."""
        assert self._id is not None
        _check("EAFResetPostion", self._lib.EAFResetPostion(self._id, step))

    # -- Configuration --

    def get_max_step(self) -> int:
        """Get the user-configured maximum step limit."""
        assert self._id is not None
        val = c_int()
        _check("EAFGetMaxStep", self._lib.EAFGetMaxStep(self._id, byref(val)))
        return val.value

    def set_max_step(self, max_step: int) -> None:
        """Set the user-configured maximum step limit."""
        assert self._id is not None
        _check("EAFSetMaxStep", self._lib.EAFSetMaxStep(self._id, max_step))

    def get_step_range(self) -> int:
        """Get the hardware-supported maximum step (cannot be exceeded by set_max_step)."""
        assert self._id is not None
        val = c_int()
        _check("EAFStepRange", self._lib.EAFStepRange(self._id, byref(val)))
        return val.value

    def get_reverse(self) -> bool:
        """Get whether direction is reversed."""
        assert self._id is not None
        val = c_bool()
        _check("EAFGetReverse", self._lib.EAFGetReverse(self._id, byref(val)))
        return val.value

    def set_reverse(self, reverse: bool) -> None:
        """Set direction reversal."""
        assert self._id is not None
        _check("EAFSetReverse", self._lib.EAFSetReverse(self._id, c_bool(reverse)))

    def get_backlash(self) -> int:
        """Get backlash compensation (0-255 steps)."""
        assert self._id is not None
        val = c_int()
        _check("EAFGetBacklash", self._lib.EAFGetBacklash(self._id, byref(val)))
        return val.value

    def set_backlash(self, steps: int) -> None:
        """Set backlash compensation (0-255 steps)."""
        assert self._id is not None
        _check("EAFSetBacklash", self._lib.EAFSetBacklash(self._id, steps))

    def set_beep(self, enabled: bool) -> None:
        """Enable/disable the movement beep."""
        assert self._id is not None
        _check("EAFSetBeep", self._lib.EAFSetBeep(self._id, c_bool(enabled)))

    def get_beep(self) -> bool:
        """Get whether the movement beep is enabled."""
        assert self._id is not None
        val = c_bool()
        _check("EAFGetBeep", self._lib.EAFGetBeep(self._id, byref(val)))
        return val.value

    # -- Info --

    def get_temperature(self) -> float:
        """Get temperature in Celsius. Returns -273.0 if sensor is invalid."""
        assert self._id is not None
        temp = c_float()
        _check("EAFGetTemp", self._lib.EAFGetTemp(self._id, byref(temp)))
        return temp.value

    def get_firmware_version(self) -> tuple[int, int, int]:
        """Get firmware version as (major, minor, build)."""
        assert self._id is not None
        major = c_ubyte()
        minor = c_ubyte()
        build = c_ubyte()
        _check(
            "EAFGetFirmwareVersion",
            self._lib.EAFGetFirmwareVersion(self._id, byref(major), byref(minor), byref(build)),
        )
        return major.value, minor.value, build.value

    def get_serial_number(self) -> str:
        """Get the device serial number as a hex string."""
        assert self._id is not None
        sn = EAF_ID()
        ret = self._lib.EAFGetSerialNumber(self._id, byref(sn))
        if ret != EAF_SUCCESS:
            return "unknown"
        return "".join(f"{b:02x}" for b in sn.id)
