"""ZWO EAF (Electronic Automatic Focuser) driver.

Wraps the native libEAFFocuser SDK via ctypes bindings to provide
focuser control as an AbstractFocuser device for the DirectHardwareAdapter.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, cast

from citrasense.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrasense.hardware.devices.focuser import AbstractFocuser

if TYPE_CHECKING:
    from citrasense.hardware.devices.focuser.zwo_eaf_bindings import EafFocuser


def _probe_eaf_focusers() -> list[dict[str, str | int]]:
    """Enumerate ZWO EAF focusers via the native SDK.

    Standalone module-level function so it can be pickled and sent to a
    subprocess by :func:`run_hardware_probe`.
    """
    from citrasense.hardware.devices.focuser.zwo_eaf_bindings import EafFocuser as EafProbe

    options: list[dict[str, str | int]] = [{"value": -1, "label": "Auto (first available)"}]
    probe = EafProbe()
    count = probe.get_num()
    for idx in range(count):
        try:
            eaf_id = probe.get_id(idx)
        except Exception:
            continue
        try:
            info = probe.get_property(eaf_id)
            name = info.Name.decode("utf-8", errors="replace").strip()
            label = f"{name} (ID: {eaf_id}, max: {info.MaxStep})" if name else f"EAF {eaf_id}"
            options.append({"value": eaf_id, "label": label})
        except Exception:
            options.append({"value": eaf_id, "label": f"EAF (ID: {eaf_id})"})
    return options


_FOCUSER_FALLBACK: list[dict[str, str | int]] = [{"value": -1, "label": "Auto (first available)"}]


class ZwoEafFocuser(AbstractFocuser):
    """Driver for the ZWO Electronic Automatic Focuser via the native SDK."""

    # -- AbstractHardwareDevice class methods --

    @classmethod
    def get_friendly_name(cls) -> str:
        return "ZWO EAF"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {
            "packages": [],
            "install_extra": "",
        }

    @classmethod
    def _detect_available_focusers(cls) -> list[dict[str, str | int]]:
        """Probe connected ZWO EAF focusers and return {value, label} options.

        Uses :meth:`_cached_hardware_probe` for subprocess isolation and
        caching so a hung native call cannot freeze the web server.
        """
        return cls._cached_hardware_probe(
            _probe_eaf_focusers,
            fallback=_FOCUSER_FALLBACK,
            timeout=10.0,
        )

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        focuser_options = cls._detect_available_focusers()

        return cast(
            list[SettingSchemaEntry],
            [
                {
                    "name": "focuser_id",
                    "friendly_name": "Focuser",
                    "type": "int",
                    "default": -1,
                    "description": "Select which EAF to use",
                    "required": False,
                    "options": focuser_options,
                    "group": "Focuser",
                },
                {
                    "name": "backlash",
                    "friendly_name": "Backlash Compensation",
                    "type": "int",
                    "default": 0,
                    "description": "Backlash compensation in steps (0-255). 0 disables.",
                    "required": False,
                    "min": 0,
                    "max": 255,
                    "group": "Focuser",
                },
                {
                    "name": "reverse",
                    "friendly_name": "Reverse Direction",
                    "type": "bool",
                    "default": False,
                    "description": "Reverse the focuser motor direction",
                    "required": False,
                    "group": "Focuser",
                },
                {
                    "name": "beep",
                    "friendly_name": "Beep on Move",
                    "type": "bool",
                    "default": False,
                    "description": "Beep when the focuser starts moving",
                    "required": False,
                    "group": "Focuser",
                },
            ],
        )

    # -- Instance --

    def __init__(self, logger: logging.Logger, **kwargs: Any) -> None:
        super().__init__(logger=logger, **kwargs)
        self._focuser_id: int = int(kwargs.get("focuser_id", -1))
        self._backlash: int = int(kwargs.get("backlash", 0))
        self._reverse: bool = bool(kwargs.get("reverse", False))
        self._beep: bool = bool(kwargs.get("beep", False))
        self._eaf: EafFocuser | None = None
        self._sdk_id: int | None = None
        self._max_step: int = 0
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            from citrasense.hardware.devices.focuser.zwo_eaf_bindings import (
                EafFocuser as EafDriver,
            )
        except ImportError:
            self.logger.error("ZWO EAF SDK library not found. See zwo_eaf_bindings.py for installation instructions.")
            return False

        with self._lock:
            try:
                eaf = EafDriver()
                sdk_ver = eaf.get_sdk_version()
                self.logger.info(f"ZWO EAF SDK version: {sdk_ver}")

                count = eaf.get_num()
                if count == 0:
                    self.logger.error("No ZWO EAF focusers found")
                    return False
                self.logger.info(f"Found {count} ZWO EAF focuser(s)")

                # Resolve which focuser to open
                target_id: int
                if self._focuser_id < 0:
                    target_id = eaf.get_id(0)
                    self.logger.info(f"Auto-selecting first focuser (ID: {target_id})")
                else:
                    target_id = self._focuser_id

                info = eaf.get_property(target_id)
                name = info.Name.decode("utf-8", errors="replace").strip()
                self.logger.info(f"Opening EAF: {name} (ID: {target_id}, hardware max step: {info.MaxStep})")

                eaf.open(target_id)

                # Read firmware version
                try:
                    major, minor, build = eaf.get_firmware_version()
                    self.logger.info(f"EAF firmware: {major}.{minor}.{build}")
                except Exception as e:
                    self.logger.debug(f"Could not read firmware version: {e}")

                # Read serial number
                try:
                    sn = eaf.get_serial_number()
                    self.logger.info(f"EAF serial: {sn}")
                except Exception as e:
                    self.logger.debug(f"Could not read serial number: {e}")

                # Apply settings
                try:
                    eaf.set_backlash(self._backlash)
                    self.logger.info(f"Backlash compensation: {self._backlash} steps")
                except Exception as e:
                    self.logger.warning(f"Could not set backlash: {e}")

                try:
                    eaf.set_reverse(self._reverse)
                    self.logger.info(f"Reverse direction: {self._reverse}")
                except Exception as e:
                    self.logger.warning(f"Could not set reverse: {e}")

                try:
                    eaf.set_beep(self._beep)
                    self.logger.info(f"Beep on move: {self._beep}")
                except Exception as e:
                    self.logger.warning(f"Could not set beep: {e}")

                # Read max step
                try:
                    self._max_step = eaf.get_max_step()
                    self.logger.info(f"Max step (user-configured): {self._max_step}")
                except Exception as e:
                    self._max_step = info.MaxStep
                    self.logger.debug(f"Using hardware max step ({self._max_step}): {e}")

                # Read current position and temperature
                try:
                    pos = eaf.get_position()
                    self.logger.info(f"Current position: {pos}")
                except Exception as e:
                    self.logger.debug(f"Could not read position: {e}")

                try:
                    temp = eaf.get_temperature()
                    if temp > -273:
                        self.logger.info(f"Temperature: {temp:.1f} C")
                except Exception as e:
                    self.logger.debug(f"Could not read temperature: {e}")

                self._eaf = eaf
                self._sdk_id = target_id
                self.logger.info("ZWO EAF connected successfully")
                return True

            except Exception as e:
                self.logger.error(f"Failed to connect to ZWO EAF: {e}")
                return False

    def disconnect(self) -> None:
        with self._lock:
            if self._eaf is not None:
                try:
                    self._eaf.close()
                except Exception as e:
                    self.logger.warning(f"Error closing EAF: {e}")
                self._eaf = None
                self._sdk_id = None
        self.logger.info("ZWO EAF disconnected")

    def is_connected(self) -> bool:
        return self._eaf is not None and self._eaf.is_open

    # -- AbstractFocuser --

    def move_absolute(self, position: int) -> bool:
        if position < 0 or (self._max_step > 0 and position > self._max_step):
            self.logger.error(f"Position {position} out of range (0-{self._max_step})")
            return False

        with self._lock:
            if not self._eaf:
                self.logger.warning("EAF not connected")
                return False
            try:
                self._eaf.move(position)
                return True
            except Exception as e:
                self.logger.error(f"Failed to move EAF to {position}: {e}")
                return False

    def move_relative(self, steps: int) -> bool:
        with self._lock:
            if not self._eaf:
                self.logger.warning("EAF not connected")
                return False

            try:
                current = self._eaf.get_position()
            except Exception as e:
                self.logger.error(f"Cannot move relative: failed to read position: {e}")
                return False

            target = current + steps
            if target < 0 or (self._max_step > 0 and target > self._max_step):
                self.logger.error(f"Target position {target} out of range (0-{self._max_step})")
                return False

            try:
                self._eaf.move(target)
                return True
            except Exception as e:
                self.logger.error(f"Failed to move EAF to {target}: {e}")
                return False

    def get_position(self) -> int | None:
        with self._lock:
            if not self._eaf:
                return None
            try:
                return self._eaf.get_position()
            except Exception as e:
                self.logger.debug(f"Failed to read position: {e}")
                return None

    def is_moving(self) -> bool:
        with self._lock:
            if not self._eaf:
                return False
            try:
                moving, _hand_ctrl = self._eaf.is_moving()
                return moving
            except Exception:
                return False

    def abort_move(self) -> None:
        with self._lock:
            if not self._eaf:
                return
            try:
                self._eaf.stop()
            except Exception as e:
                self.logger.warning(f"Failed to stop EAF: {e}")

    def get_max_position(self) -> int | None:
        with self._lock:
            if not self._eaf:
                return None
            try:
                return self._eaf.get_max_step()
            except Exception:
                return self._max_step if self._max_step > 0 else None

    def get_temperature(self) -> float | None:
        with self._lock:
            if not self._eaf:
                return None
            try:
                temp = self._eaf.get_temperature()
                if temp <= -273:
                    return None
                return temp
            except Exception:
                return None
