"""Concrete mount device for ZWO AM3/AM5/AM7 mounts.

Communicates over USB serial or WiFi TCP using the ZWO LX200-variant
protocol.  All public coordinate APIs use **degrees** (project convention);
the RA hours ↔ degrees conversion happens at this boundary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from citrascope.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrascope.hardware.devices.mount.abstract_mount import AbstractMount
from citrascope.hardware.devices.mount.zwo_am_protocol import (
    Direction,
    MountMode,
    TrackingRate,
    ZwoAmCommands,
    ZwoAmResponseParser,
)
from citrascope.hardware.devices.mount.zwo_am_transport import (
    DEFAULT_BAUD_RATE,
    DEFAULT_RETRY_COUNT,
    DEFAULT_TIMEOUT_S,
    SerialTransport,
    TcpTransport,
    ZwoAmTransport,
)

_TRACKING_RATE_MAP: dict[str, TrackingRate] = {
    "sidereal": TrackingRate.SIDEREAL,
    "lunar": TrackingRate.LUNAR,
    "solar": TrackingRate.SOLAR,
}

_ZWO_USB_VID = 0x03C3

_KNOWN_USB_VENDORS: dict[int, str] = {
    _ZWO_USB_VID: "ZWO",
    0x0403: "FTDI",
    0x067B: "Prolific",
    0x10C4: "Silicon Labs",
    0x1A86: "QinHeng",
}


class ZwoAmMount(AbstractMount):
    """ZWO AM3/AM5/AM7 mount controlled via serial (USB) or TCP (WiFi).

    Configuration kwargs:
        connection_type: ``"serial"`` (default) or ``"tcp"``
        port:            Serial device path, e.g. ``/dev/ttyUSB0`` or ``COM3``
        baud_rate:       Serial baud rate (default 9600)
        tcp_host:        WiFi hostname/IP when connection_type is ``tcp``
        tcp_port:        WiFi TCP port when connection_type is ``tcp``
        timeout:         Command timeout in seconds (default 2.0)
        retry_count:     Retries per command (default 3)
    """

    _port_cache: list[dict[str, str]] | None = None
    _port_cache_timestamp: float = 0
    _port_cache_ttl: float = 1.0

    @classmethod
    def _detect_serial_ports(cls) -> list[dict[str, str]]:
        """Enumerate serial ports with friendly labels.

        ZWO devices (VID 0x03C3) are labelled with their USB product name.
        Other common astronomy USB-serial adapters get vendor-tagged labels.
        Results are cached briefly to avoid repeated USB enumeration.
        """
        import time

        cache_age = time.time() - cls._port_cache_timestamp
        if cls._port_cache is not None and cache_age < cls._port_cache_ttl:
            return cls._port_cache

        ports: list[dict[str, str]] = []
        try:
            from serial.tools.list_ports import comports  # type: ignore[reportMissingImports]

            for info in sorted(comports()):
                vendor = _KNOWN_USB_VENDORS.get(info.vid) if info.vid else None
                product = info.product or ""

                if info.vid == _ZWO_USB_VID:
                    if product and not product.upper().startswith("ZWO"):
                        label = f"ZWO {product} ({info.device})"
                    elif product:
                        label = f"{product} ({info.device})"
                    else:
                        label = f"ZWO Device ({info.device})"
                elif vendor and product:
                    label = f"{product} ({info.device})"
                elif vendor:
                    label = f"{vendor} Adapter ({info.device})"
                elif info.description and info.description != "n/a":
                    label = f"{info.description} ({info.device})"
                else:
                    label = info.device

                ports.append({"value": info.device, "label": label})

        except ImportError:
            pass
        except Exception:
            pass

        if not ports:
            ports.append({"value": "/dev/ttyUSB0", "label": "/dev/ttyUSB0 (default)"})

        cls._port_cache = ports
        cls._port_cache_timestamp = time.time()
        return ports

    def __init__(self, logger: logging.Logger, **kwargs) -> None:
        super().__init__(logger=logger, **kwargs)

        conn_type = kwargs.get("connection_type", "serial")
        timeout = float(kwargs.get("timeout", DEFAULT_TIMEOUT_S))
        retries = int(kwargs.get("retry_count", DEFAULT_RETRY_COUNT))

        self._transport: ZwoAmTransport
        if conn_type == "tcp":
            host = str(kwargs.get("tcp_host", "10.0.0.1"))
            port = int(kwargs.get("tcp_port", 4030))
            self._transport = TcpTransport(host=host, port=port, timeout_s=timeout, retry_count=retries)
        else:
            serial_port = str(kwargs.get("port", "/dev/ttyUSB0"))
            baud = int(kwargs.get("baud_rate", DEFAULT_BAUD_RATE))
            self._transport = SerialTransport(port=serial_port, baud_rate=baud, timeout_s=timeout, retry_count=retries)

        self._model: str = ""
        self._firmware: str = ""

    # ------------------------------------------------------------------
    # AbstractHardwareDevice
    # ------------------------------------------------------------------

    @classmethod
    def get_friendly_name(cls) -> str:
        return "ZWO AM3/AM5/AM7 Mount"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {
            "packages": ["serial"],
            "install_extra": "zwo-mount",
        }

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        schema: list[Any] = [
            {
                "name": "connection_type",
                "friendly_name": "Connection Type",
                "type": "str",
                "default": "serial",
                "description": "How to connect to the mount",
                "required": True,
                "options": [
                    {"value": "serial", "label": "USB Serial"},
                    {"value": "tcp", "label": "WiFi (TCP)"},
                ],
                "group": "Mount",
            },
            {
                "name": "port",
                "friendly_name": "Serial Port",
                "type": "str",
                "default": cls._detect_serial_ports()[0]["value"],
                "description": "Serial port for the mount",
                "required": False,
                "options": cls._detect_serial_ports(),
                "group": "Mount",
                "visible_when": {"field": "connection_type", "value": "serial"},
            },
            {
                "name": "baud_rate",
                "friendly_name": "Baud Rate",
                "type": "int",
                "default": DEFAULT_BAUD_RATE,
                "description": "Serial baud rate (ZWO default is 9600)",
                "required": False,
                "group": "Mount",
                "visible_when": {"field": "connection_type", "value": "serial"},
            },
            {
                "name": "tcp_host",
                "friendly_name": "WiFi Host",
                "type": "str",
                "default": "10.0.0.1",
                "description": "Mount WiFi IP address or hostname",
                "required": False,
                "group": "Mount",
                "visible_when": {"field": "connection_type", "value": "tcp"},
            },
            {
                "name": "tcp_port",
                "friendly_name": "WiFi Port",
                "type": "int",
                "default": 4030,
                "description": "TCP port for WiFi serial bridge",
                "required": False,
                "group": "Mount",
                "visible_when": {"field": "connection_type", "value": "tcp"},
            },
            {
                "name": "timeout",
                "friendly_name": "Command Timeout",
                "type": "float",
                "default": DEFAULT_TIMEOUT_S,
                "description": "Seconds to wait for a command response",
                "required": False,
                "group": "Mount",
            },
            {
                "name": "retry_count",
                "friendly_name": "Retry Count",
                "type": "int",
                "default": DEFAULT_RETRY_COUNT,
                "description": "Number of retries for failed commands",
                "required": False,
                "group": "Mount",
            },
        ]
        return schema

    def connect(self) -> bool:
        try:
            self._transport.open()
        except Exception as exc:
            self.logger.error("Failed to open transport: %s", exc)
            return False

        try:
            self._model = self._transport.send_command_with_retry(ZwoAmCommands.get_mount_model()).rstrip("#")
            self.logger.info("Connected to mount: %s", self._model)
        except Exception as exc:
            self.logger.error("Mount handshake failed: %s", exc)
            self._transport.close()
            return False

        try:
            self._firmware = self._transport.send_command_with_retry(ZwoAmCommands.get_version()).rstrip("#")
            self.logger.info("Firmware version: %s", self._firmware)
        except Exception:
            self.logger.warning("Could not read firmware version")

        try:
            _, _, at_home, parked, mode = self._get_status_flags()
            self.logger.info("Mount mode: %s | At home: %s | Parked: %s", mode.value, at_home, parked)
        except Exception:
            self.logger.warning("Could not read mount status")

        try:
            limits_on = self.get_altitude_limits_enabled()
            lower, upper = self.get_limits()
            self.logger.info("Altitude limits: enabled=%s lower=%s° upper=%s°", limits_on, lower, upper)
        except Exception:
            self.logger.warning("Could not read altitude limits")

        try:
            flip, track, limit = self.get_meridian_flip_settings()
            self.logger.info("Meridian flip: enabled=%s track_after=%s limit=%s°", flip, track, limit)
        except Exception:
            self.logger.warning("Could not read meridian flip settings")

        return True

    def disconnect(self) -> None:
        self._transport.close()
        self.logger.info("Mount disconnected")

    def is_connected(self) -> bool:
        return self._transport.is_open()

    # ------------------------------------------------------------------
    # Core mount operations  (AbstractMount abstract methods)
    # ------------------------------------------------------------------

    def slew_to_radec(self, ra: float, dec: float) -> bool:
        ra_hours = ra / 15.0

        ra_cmd = ZwoAmCommands.set_target_ra_decimal(ra_hours)
        if not self._transport.send_command_bool_with_retry(ra_cmd):
            self.logger.error("Mount rejected RA target %.4f°", ra)
            return False

        dec_cmd = ZwoAmCommands.set_target_dec_decimal(dec)
        if not self._transport.send_command_bool_with_retry(dec_cmd):
            self.logger.error("Mount rejected Dec target %.4f°", dec)
            return False

        response = self._transport.send_goto_command(ZwoAmCommands.goto())
        error = ZwoAmResponseParser.parse_goto_response(response)
        if error is not None:
            self.logger.error("GoTo failed: %s (raw response: %r)", error, response)
            self._log_goto_diagnostics(ra, dec)
            return False

        self.logger.info("Slewing to RA=%.4f° Dec=%.4f°", ra, dec)
        return True

    def is_slewing(self) -> bool:
        _, slewing, _, _, _ = self._get_status_flags()
        return slewing

    def abort_slew(self) -> None:
        self._transport.send_command_no_response(ZwoAmCommands.stop_all())
        self.logger.info("Slew aborted")

    def get_radec(self) -> tuple[float, float]:
        ra_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_ra())
        parsed_ra = ZwoAmResponseParser.parse_ra(ra_resp)
        if parsed_ra is None:
            raise RuntimeError(f"Failed to parse RA response: {ra_resp!r}")
        ra_hours = ZwoAmResponseParser.hms_to_decimal_hours(*parsed_ra)
        ra_deg = ra_hours * 15.0

        dec_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_dec())
        parsed_dec = ZwoAmResponseParser.parse_dec(dec_resp)
        if parsed_dec is None:
            raise RuntimeError(f"Failed to parse Dec response: {dec_resp!r}")
        dec_deg = ZwoAmResponseParser.dms_to_decimal_degrees(*parsed_dec)

        return ra_deg, dec_deg

    def start_tracking(self, rate: str | None = "sidereal") -> bool:
        rate_str = rate or "sidereal"
        track_rate = _TRACKING_RATE_MAP.get(rate_str.lower())
        if track_rate is None:
            self.logger.warning("Unknown tracking rate %r, using sidereal", rate)
            track_rate = TrackingRate.SIDEREAL

        self._transport.send_command_no_response(ZwoAmCommands.set_tracking_rate(track_rate))
        self._transport.send_command_no_response(ZwoAmCommands.tracking_on())
        self.logger.info("Tracking started: %s", track_rate.value)
        return True

    def stop_tracking(self) -> bool:
        self._transport.send_command_no_response(ZwoAmCommands.tracking_off())
        self.logger.info("Tracking stopped")
        return True

    def is_tracking(self) -> bool:
        tracking, _, _, _, _ = self._get_status_flags()
        return tracking

    def park(self) -> bool:
        self._transport.send_command_no_response(ZwoAmCommands.goto_park())
        self.logger.info("Park initiated")
        return True

    def unpark(self) -> bool:
        self._transport.send_command_no_response(ZwoAmCommands.unpark())
        self.logger.info("Unparked")
        return True

    def is_parked(self) -> bool:
        _, _, _, parked, _ = self._get_status_flags()
        return parked

    def find_home(self) -> bool:
        self._transport.send_command_no_response(ZwoAmCommands.find_home())
        self.logger.info("Find-home initiated")
        return True

    def is_home(self) -> bool:
        _, _, at_home, _, _ = self._get_status_flags()
        return at_home

    def get_mount_info(self) -> dict:
        _, _, _, _, mode = self._get_status_flags()
        return {
            "model": self._model,
            "firmware": self._firmware,
            "mount_mode": mode.value,
            "supports_sync": True,
            "supports_guide_pulse": True,
            "supports_custom_tracking": False,
        }

    # ------------------------------------------------------------------
    # Optional capabilities (concrete overrides)
    # ------------------------------------------------------------------

    def sync_to_radec(self, ra: float, dec: float) -> bool:
        try:
            tracking, slewing, at_home, parked, mode = self._get_status_flags()
            self.logger.info(
                "Pre-sync status: mode=%s tracking=%s slewing=%s home=%s parked=%s",
                mode.value,
                tracking,
                slewing,
                at_home,
                parked,
            )
        except Exception:
            self.logger.warning("Could not read mount status before sync", exc_info=True)

        ra_hours = ra / 15.0

        ra_cmd = ZwoAmCommands.set_target_ra_decimal(ra_hours)
        if not self._transport.send_command_bool_with_retry(ra_cmd):
            self.logger.error("Mount rejected sync RA target %.4f°", ra)
            return False

        dec_cmd = ZwoAmCommands.set_target_dec_decimal(dec)
        if not self._transport.send_command_bool_with_retry(dec_cmd):
            self.logger.error("Mount rejected sync Dec target %.4f°", dec)
            return False

        sync_resp = self._transport.send_command_with_retry(ZwoAmCommands.sync())
        error = ZwoAmResponseParser.parse_goto_response(sync_resp)
        if error is not None:
            self.logger.error("Mount rejected sync: %s (raw response: %r)", error, sync_resp)
            return False

        self.logger.info("Sync accepted for RA=%.4f° Dec=%.4f° (response: %r)", ra, dec, sync_resp)

        try:
            readback_ra, readback_dec = self.get_radec()
            self.logger.info("Post-sync readback: RA=%.4f° Dec=%.4f°", readback_ra, readback_dec)
        except Exception:
            self.logger.warning("Could not read back position after sync")

        return True

    def guide_pulse(self, direction: str, duration_ms: int) -> bool:
        try:
            d = Direction(direction.lower())
        except ValueError:
            self.logger.error("Invalid guide direction: %s", direction)
            return False
        self._transport.send_command_no_response(ZwoAmCommands.guide_pulse(d, duration_ms))
        self.logger.debug("Guide pulse %s %dms", d.value, duration_ms)
        return True

    def get_meridian_auto_flip(self) -> bool | None:
        resp = self._transport.send_command_with_retry(ZwoAmCommands.get_meridian_flip_settings())
        self.logger.debug("Meridian flip raw response: %r", resp)
        flip, _, _ = ZwoAmResponseParser.parse_meridian_flip_settings(resp)
        return flip

    def get_meridian_flip_settings(self) -> tuple[bool, bool, int]:
        """Read full meridian flip settings: (enabled, track_after, limit_degrees)."""
        resp = self._transport.send_command_with_retry(ZwoAmCommands.get_meridian_flip_settings())
        return ZwoAmResponseParser.parse_meridian_flip_settings(resp)

    def set_meridian_auto_flip(self, enabled: bool) -> bool:
        """Enable/disable meridian auto-flip, preserving the other settings."""
        _, track, limit = self.get_meridian_flip_settings()
        cmd = ZwoAmCommands.set_meridian_flip_settings(enabled, track, limit)
        ok = self._transport.send_command_bool_with_retry(cmd)
        if ok:
            self.logger.info("Meridian auto-flip %s", "enabled" if enabled else "disabled")
        else:
            self.logger.warning("Mount rejected meridian flip command")
        return ok

    def set_meridian_flip_settings(self, enabled: bool, track_after: bool, limit: int) -> bool:
        """Set all meridian flip parameters at once."""
        cmd = ZwoAmCommands.set_meridian_flip_settings(enabled, track_after, limit)
        ok = self._transport.send_command_bool_with_retry(cmd)
        if ok:
            self.logger.info("Meridian flip: enabled=%s track_after=%s limit=%d°", enabled, track_after, limit)
        else:
            self.logger.warning("Mount rejected meridian flip settings")
        return ok

    def get_limits(self) -> tuple[int | None, int | None]:
        """Read the mount's lower (horizon) and upper (overhead) altitude limits."""
        lower_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_altitude_limit_lower())
        upper_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_altitude_limit_upper())
        return (
            ZwoAmResponseParser.parse_altitude_limit(lower_resp),
            ZwoAmResponseParser.parse_altitude_limit(upper_resp),
        )

    def get_altitude_limits_enabled(self) -> bool | None:
        resp = self._transport.send_command_with_retry(ZwoAmCommands.get_altitude_limit_enabled())
        return ZwoAmResponseParser.parse_bool(resp)

    def set_altitude_limits_enabled(self, enable: bool) -> None:
        self._transport.send_command_no_response(ZwoAmCommands.set_altitude_limit_enabled(enable))
        self.logger.info("Altitude limits %s", "enabled" if enable else "disabled")

    def set_horizon_limit(self, degrees: int) -> bool:
        ok = self._transport.send_command_bool_with_retry(ZwoAmCommands.set_altitude_limit_lower(degrees))
        if ok:
            self.logger.info("Lower altitude limit set to %d°", degrees)
        else:
            self.logger.debug("Mount rejected lower altitude limit %d° (firmware may not support :SLL#)", degrees)
        return ok

    def set_overhead_limit(self, degrees: int) -> bool:
        ok = self._transport.send_command_bool_with_retry(ZwoAmCommands.set_altitude_limit_upper(degrees))
        if ok:
            self.logger.info("Upper altitude limit set to %d°", degrees)
        else:
            self.logger.debug("Mount rejected upper altitude limit %d° (firmware may not support :SLH#)", degrees)
        return ok

    def set_equatorial_mode(self) -> bool:
        self._transport.send_command_no_response(ZwoAmCommands.set_polar_mode())
        self.logger.info("Sent :AP# (set equatorial mode) — requires mount restart to take effect")
        return True

    def get_mount_mode(self) -> str:
        _, _, _, _, mode = self._get_status_flags()
        return mode.value

    def get_azimuth(self) -> float | None:
        try:
            resp = self._transport.send_command_with_retry(ZwoAmCommands.get_azimuth())
            parsed = ZwoAmResponseParser.parse_azimuth(resp)
            if parsed is None:
                return None
            d, m, s = parsed
            return d + m / 60.0 + s / 3600.0
        except Exception:
            self.logger.debug("Could not read azimuth", exc_info=True)
            return None

    def get_altitude(self) -> float | None:
        try:
            resp = self._transport.send_command_with_retry(ZwoAmCommands.get_altitude())
            parsed = ZwoAmResponseParser.parse_azimuth(resp)
            if parsed is None:
                return None
            d, m, s = parsed
            sign = -1 if d < 0 else 1
            return sign * (abs(d) + m / 60.0 + s / 3600.0)
        except Exception:
            self.logger.debug("Could not read altitude", exc_info=True)
            return None

    def start_move(self, direction: str, rate: int = 7) -> bool:
        try:
            d = Direction(direction.lower())
        except ValueError:
            self.logger.error("Invalid move direction: %s", direction)
            return False
        self._transport.send_command_no_response(ZwoAmCommands.set_slew_rate(rate))
        self._transport.send_command_no_response(ZwoAmCommands.move_direction(d))
        self.logger.info("Started move %s at rate %d", d.value, rate)
        return True

    def stop_move(self, direction: str) -> bool:
        try:
            d = Direction(direction.lower())
        except ValueError:
            self.logger.error("Invalid stop direction: %s", direction)
            return False
        self._transport.send_command_no_response(ZwoAmCommands.stop_direction(d))
        self.logger.info("Stopped move %s", d.value)
        return True

    def set_site_location(self, latitude: float, longitude: float, altitude: float) -> bool:
        lat_cmd = ZwoAmCommands.set_latitude(latitude)
        if not self._transport.send_command_bool_with_retry(lat_cmd):
            self.logger.error("Mount rejected latitude %.4f", latitude)
            return False

        lon_cmd = ZwoAmCommands.set_longitude(longitude)
        if not self._transport.send_command_bool_with_retry(lon_cmd):
            self.logger.error("Mount rejected longitude %.4f", longitude)
            return False

        self.logger.info("Site location set: lat=%.4f° lon=%.4f° alt=%.0fm", latitude, longitude, altitude)

        try:
            lat_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_latitude())
            lon_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_longitude())
            readback_lat = ZwoAmResponseParser.parse_site_coordinate(lat_resp)
            readback_lon = ZwoAmResponseParser.parse_site_coordinate(lon_resp)
            if readback_lon is not None:
                readback_lon = -readback_lon  # Meade convention is west-positive
            self.logger.info(
                "Site location readback: lat=%s° (expected %.4f°) lon=%s° (expected %.4f°) | raw lat=%r lon=%r",
                f"{readback_lat:.4f}" if readback_lat is not None else "PARSE_FAIL",
                latitude,
                f"{readback_lon:.4f}" if readback_lon is not None else "PARSE_FAIL",
                longitude,
                lat_resp,
                lon_resp,
            )
        except Exception:
            self.logger.warning("Could not read back site location after write", exc_info=True)

        return True

    def sync_datetime(self) -> bool:
        now = datetime.now(timezone.utc)

        tz_ok = self._transport.send_command_bool_with_retry(ZwoAmCommands.set_timezone(0))
        time_ok = self._transport.send_command_bool_with_retry(ZwoAmCommands.set_time(now.hour, now.minute, now.second))
        # :SC (set date) goes last because some LX200 firmware sends extra
        # #-terminated strings after the ack, which corrupt subsequent reads.
        date_ok = self._transport.send_command_bool_with_retry(ZwoAmCommands.set_date(now.month, now.day, now.year))
        # Drain any extra :SC response data so it doesn't corrupt the next
        # command from any caller (web UI status polling, slew, etc.)
        self._transport._clear_input()

        if not (tz_ok and date_ok and time_ok):
            self.logger.warning("Mount time sync partially failed (tz=%s date=%s time=%s)", tz_ok, date_ok, time_ok)
            return False

        self.logger.info("Mount time synced to UTC: %s", now.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            lst_resp = self._transport.send_command_with_retry(ZwoAmCommands.get_sidereal_time())
            self.logger.info("Mount sidereal time readback: %s", lst_resp.rstrip("#"))
        except Exception:
            self.logger.warning("Could not read back sidereal time after time sync", exc_info=True)

        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_goto_diagnostics(self, ra: float, dec: float) -> None:
        """Log mount state to help diagnose GoTo rejections."""
        try:
            _, _, at_home, parked, mode = self._get_status_flags()
            limits_on = self.get_altitude_limits_enabled()
            lower, upper = self.get_limits()
            self.logger.warning(
                "GoTo diagnostic — target RA=%.4f° Dec=%.4f° | mode=%s home=%s parked=%s "
                "| limits_enabled=%s lower=%s° upper=%s°",
                ra,
                dec,
                mode.value,
                at_home,
                parked,
                limits_on,
                lower,
                upper,
            )
        except Exception:
            self.logger.debug("Could not read mount state for GoTo diagnostics", exc_info=True)

    def _get_status_flags(self) -> tuple[bool, bool, bool, bool, MountMode]:
        resp = self._transport.send_command_with_retry(ZwoAmCommands.get_status())
        return ZwoAmResponseParser.parse_status(resp)
