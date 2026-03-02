"""Moravian Instruments standalone external filter wheel driver.

Uses the gxfw_* C API for filter wheels that are separate USB/ETH devices,
as opposed to the integrated wheels built into Moravian cameras (which are
handled by MoravianIntegratedFilterWheel in moravian_camera.py).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from citrascope.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrascope.hardware.devices.filter_wheel import AbstractFilterWheel

if TYPE_CHECKING:
    from citrascope.hardware.devices.moravian_bindings import GxccdFilterWheel as GxccdFW


class MoravianFilterWheel(AbstractFilterWheel):
    """Driver for standalone Moravian Instruments external filter wheels."""

    @classmethod
    def get_friendly_name(cls) -> str:
        return "Moravian Instruments Filter Wheel (External)"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {"packages": [], "install_extra": ""}

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        return cast(
            list[SettingSchemaEntry],
            [
                {
                    "name": "wheel_id",
                    "friendly_name": "Wheel ID",
                    "type": "int",
                    "default": -1,
                    "description": "Filter wheel identifier (-1 for single wheel, or ID from enumeration)",
                    "required": False,
                    "group": "Filter Wheel",
                },
                {
                    "name": "connection_type",
                    "friendly_name": "Connection Type",
                    "type": "str",
                    "default": "usb",
                    "description": "USB or Ethernet connection",
                    "required": False,
                    "options": ["usb", "eth"],
                    "group": "Filter Wheel",
                },
                {
                    "name": "eth_ip",
                    "friendly_name": "Ethernet Adapter IP",
                    "type": "str",
                    "default": "192.168.0.5",
                    "description": "IP address of the Ethernet Adapter",
                    "required": False,
                    "visible_when": {"field": "connection_type", "value": "eth"},
                    "group": "Filter Wheel",
                },
                {
                    "name": "eth_port",
                    "friendly_name": "Ethernet Adapter Port",
                    "type": "int",
                    "default": 48899,
                    "description": "Port of the Ethernet Adapter",
                    "required": False,
                    "visible_when": {"field": "connection_type", "value": "eth"},
                    "group": "Filter Wheel",
                },
            ],
        )

    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        self._wheel_id: int = int(kwargs.get("wheel_id", -1))
        self._connection_type: str = kwargs.get("connection_type", "usb")
        self._eth_ip: str = kwargs.get("eth_ip", "192.168.0.5")
        self._eth_port: int = int(kwargs.get("eth_port", 48899))

        self._gxfw: GxccdFW | None = None
        self._filter_names: list[str] = []
        self._num_filters: int = 0
        self._position: int = 0

    def connect(self) -> bool:
        try:
            from citrascope.hardware.devices.moravian_bindings import (
                FW_GIP_FILTERS,
                FW_GSP_DESCRIPTION,
                FW_GSP_SERIAL_NUMBER,
                GxccdError,
                GxccdFilterWheel,
            )
        except ImportError as e:
            self.logger.error(f"Moravian native library not available: {e}")
            return False

        try:
            fw = GxccdFilterWheel()
            if self._connection_type == "eth":
                fw.initialize_eth(self._wheel_id)
            else:
                fw.initialize_usb(self._wheel_id)

            self._gxfw = fw

            desc = fw.get_string_parameter(FW_GSP_DESCRIPTION)
            serial = fw.get_string_parameter(FW_GSP_SERIAL_NUMBER)
            self._num_filters = fw.get_integer_parameter(FW_GIP_FILTERS)

            # Read filter names from .ini config
            filters = fw.enumerate_filters()
            self._filter_names = [name for name, _color, _offset in filters]
            while len(self._filter_names) < self._num_filters:
                self._filter_names.append(f"Filter {len(self._filter_names) + 1}")

            self.logger.info(
                f"Connected to {desc.strip()} (SN: {serial.strip()}) "
                f"with {self._num_filters} filters: {self._filter_names}"
            )
            return True

        except (GxccdError, OSError) as e:
            self.logger.error(f"Failed to connect to Moravian filter wheel: {e}")
            self._gxfw = None
            return False

    def disconnect(self):
        if self._gxfw:
            self._gxfw.release()
            self._gxfw = None
            self.logger.info("Moravian filter wheel disconnected")

    def is_connected(self) -> bool:
        return self._gxfw is not None and self._gxfw.is_initialized

    # -- Filter operations --

    def set_filter_position(self, position: int) -> bool:
        if not self.is_connected():
            return False
        assert self._gxfw is not None
        try:
            self._gxfw.set_filter(position)
            self._position = position
            return True
        except Exception as e:
            self.logger.error(f"Failed to set filter position {position}: {e}")
            return False

    def get_filter_position(self) -> int | None:
        if not self.is_connected():
            return None
        return self._position

    def is_moving(self) -> bool:
        return False  # gxfw_set_filter blocks until complete

    def get_filter_count(self) -> int:
        return self._num_filters

    def get_filter_names(self) -> list[str]:
        return list(self._filter_names)

    def set_filter_names(self, names: list[str]) -> bool:
        if len(names) != self._num_filters:
            return False
        self._filter_names = list(names)
        return True
