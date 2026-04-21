"""NINA focuser device — proxies NINA Advanced API focuser REST endpoints.

Implements the AbstractFocuser contract so the web UI manual focus
controls, status polling, and focuser move/abort routes all work
transparently when NINA is the active adapter.
"""

from __future__ import annotations

import logging

import requests

from citrasense.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrasense.hardware.devices.focuser import AbstractFocuser


class NinaFocuser(AbstractFocuser):
    """Focuser device backed by the NINA Advanced API REST endpoints.

    Not registered in the device registry — instantiated internally by
    :class:`NinaAdvancedHttpAdapter` during ``connect()``.
    """

    FOCUSER_URL = "/equipment/focuser/"

    def __init__(
        self,
        logger: logging.Logger,
        nina_api_path: str,
        *,
        info_timeout: float = 10.0,
        command_timeout: float = 30.0,
        connect_timeout: float = 5.0,
    ) -> None:
        super().__init__(logger=logger)
        self._api = nina_api_path
        self._info_timeout = info_timeout
        self._command_timeout = command_timeout
        self._connect_timeout = connect_timeout
        self._connected = False

    # -- AbstractHardwareDevice classmethods (stubs — never used from registry) --

    @classmethod
    def get_friendly_name(cls) -> str:
        return "NINA Focuser"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {"packages": [], "install_extra": ""}

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        return []

    # -- Connection lifecycle --

    def connect(self) -> bool:
        try:
            resp = requests.get(
                self._api + self.FOCUSER_URL + "connect",
                timeout=self._connect_timeout,
            ).json()
            self._connected = resp.get("Success", False)
            return self._connected
        except Exception as e:
            self.logger.warning(f"NINA focuser connect failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        try:
            info = self._get_info()
            if info is None:
                self._connected = False
                return False
            self._connected = info.get("Connected", False)
            return self._connected
        except Exception:
            self._connected = False
            return False

    # -- AbstractFocuser --

    def move_absolute(self, position: int) -> bool:
        try:
            resp = requests.get(
                f"{self._api}{self.FOCUSER_URL}move?position={position}",
                timeout=self._command_timeout,
            ).json()
            return resp.get("Success", False)
        except Exception as e:
            self.logger.error(f"NINA focuser move_absolute failed: {e}")
            return False

    def move_relative(self, steps: int) -> bool:
        """Move focuser by a relative number of steps.

        NINA only exposes an absolute move endpoint, so this reads the
        current position and issues an absolute move to ``current + steps``.
        """
        current = self.get_position()
        if current is None:
            self.logger.error("Cannot move relative: current position unknown")
            return False
        target = current + steps
        if target < 0:
            target = 0
        max_pos = self.get_max_position()
        if max_pos is not None and target > max_pos:
            target = max_pos
        return self.move_absolute(target)

    def get_position(self) -> int | None:
        info = self._get_info()
        if info is None:
            return None
        pos = info.get("Position")
        return int(pos) if pos is not None else None

    def is_moving(self) -> bool:
        info = self._get_info()
        if info is None:
            return False
        return bool(info.get("IsMoving", False))

    def abort_move(self) -> None:
        try:
            requests.get(
                self._api + self.FOCUSER_URL + "stop-move",
                timeout=self._command_timeout,
            )
        except Exception as e:
            self.logger.error(f"NINA focuser abort_move failed: {e}")

    def get_max_position(self) -> int | None:
        info = self._get_info()
        if info is None:
            return None
        val = info.get("MaxStep")
        return int(val) if val is not None else None

    def get_temperature(self) -> float | None:
        info = self._get_info()
        if info is None:
            return None
        temp = info.get("Temperature")
        if temp is None:
            return None
        temp_f = float(temp)
        if temp_f != temp_f:  # NaN check — NINA returns NaN when sensor absent
            return None
        return temp_f

    # -- Internal helpers --

    def _get_info(self) -> dict | None:
        """Query ``GET .../focuser/info`` and return the ``Response`` dict, or None on failure."""
        try:
            resp = requests.get(
                self._api + self.FOCUSER_URL + "info",
                timeout=self._info_timeout,
            ).json()
            if resp.get("Success"):
                return resp.get("Response")
            return None
        except Exception:
            return None
