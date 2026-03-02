"""ZWO AM3/AM5/AM7 serial protocol — command generation and response parsing.

The protocol is a modified Meade LX200 text protocol.  Commands are sent as
``:XX#`` strings; responses (when expected) are ``#``-terminated.

This module contains **only** pure functions and enums — no I/O, no state.
It is safe to import and unit-test without any hardware present.

.. warning::

   The ZWO AM5 firmware is *not* standard LX200 and *not* standard OnStep.
   It is a heavily customised OnStepX fork that strips most ``:GX``/``:SX``
   extended commands and replaces limit/meridian commands with proprietary
   ones.  **Do not add standard LX200 limit commands** (``:Gh#``, ``:Go#``,
   ``:Sh#``, ``:So#``) or OnStep extended commands (``:GX95#``, ``:SX95#``)
   — they silently fail or return garbage on real ZWO hardware.

   The INDI ``lx200am5`` driver is the definitive reference for which
   commands actually work.  When in doubt, read the driver source, not the
   ZWO protocol PDF.

Authoritative references (in order):

  1. **INDI lx200am5 driver** (gold standard — written specifically for AM5):
     https://github.com/indilib/indi/blob/master/drivers/telescope/lx200am5.cpp
     (header: ``lx200am5.h``)

  2. **OnStepX firmware source** (useful for error codes and ``:GU#`` flags,
     but many commands are stripped from ZWO firmware):
     https://github.com/hjd1964/OnStepX/tree/main/src/telescope/mount

  3. **jmcguigs/zwo-control-rs** — Rust reference implementation.
   https://raw.githubusercontent.com/jmcguigs/zwo-control-rs/refs/heads/master/src/protocol.rs

  4. **ZWO Mount Serial Communication Protocol v2.1** — official but incomplete.

Commands that DO NOT work on ZWO AM5::

  :Gh# / :Go#        Standard LX200 get horizon/overhead limit → empty/garbage
  :Sh# / :So#        Standard LX200 set horizon/overhead limit → silently rejected
  :GX95#             OnStep get meridian auto-flip → timeout / no response
  :SX95,{0|1}#      OnStep set meridian auto-flip → rejected

ZWO-proprietary commands (use these instead)::

  Mount type (alt-az is the default for satellite observation work):
    :AA#               Set alt-az mode (fire-and-forget)
    :AP#               Set equatorial mode (fire-and-forget, requires mount restart)
    :GU#               Status flags: G=equatorial, Z=alt-az, H=home, P=parked,
                       n=not tracking, N=not slewing

  Altitude limits (INDI driver v1.3+ — may require newer firmware):
    :GLC#   → 0#/1#   Are limits enabled?
    :SLE#              Enable limits (fire-and-forget)
    :SLD#              Disable limits (fire-and-forget)
    :GLH#   → DD#     Get upper limit in degrees
    :GLL#   → DD#     Get lower limit in degrees
    :SLH{dd}# → 1/0  Set upper limit (60–90°)
    :SLL{dd}# → 1/0  Set lower limit (0–30°)

    WARNING: On firmware 1.1.2, these collide with standard LX200 :GL (Get
    Local Time) and :SL (Set Local Time).  :GLC# returns the local time
    string instead of a limit status, and :SLH#/:SLL# are rejected as
    malformed time values.  These commands likely require a firmware update.

  Meridian flip (compound get/set — equatorial mode only):
    :GTa#   → ET{s}DD#  e.g. 10+00# = flip=1, track-after=0, limit=+0°
    :STa{e}{t}{s}{dd}# → 1/0  e.g. :STa11+00# (flip on, track on, +0° limit)

  Other ZWO-specific:
    :GBu# / :SBu{n}#  Buzzer (0=off, 1=low, 2=high)
    :GRl# / :SRl{n}#  Heavy duty mode (1440=off, 720=on)
    :GAT#  → 0#/1#    Is tracking active?
    :Gm#   → W/E      Pier side (only meaningful in equatorial mode)
    :SOa#  → 1/0      Set current position as home
    :NSC#  → 1/0      Clear multi-star alignment data

GoTo (``:MS#``) and Sync (``:CM#``) error codes
(ZWO-specific — differs from standard OnStepX mapping)::

  0       Success (bare 0, NO # terminator — special case!)
  N/A     Sync success (``:CM#`` only — ``N/A#`` terminated)
  1 / e1  Below horizon
  2 / e2  Above overhead limit
  3 / e3  In standby (motors disabled)
  4 / e4  Parked
  5 / e5  Not aligned (coordinate transform not initialised)
  6 / e6  Outside limits (general)
  7 / e7  Pier side limit (needs meridian flip)
  8 / e8  In motion (already slewing or guiding)
"""

from __future__ import annotations

import math
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    """Cardinal direction for manual-motion and guide-pulse commands."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def opposite(self) -> Direction:
        _opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
        }
        return _opposites[self]


class TrackingRate(str, Enum):
    """Tracking-rate presets understood by the mount firmware."""

    SIDEREAL = "sidereal"
    LUNAR = "lunar"
    SOLAR = "solar"
    OFF = "off"


class MountMode(str, Enum):
    """Physical mount operating mode."""

    ALTAZ = "altaz"
    EQUATORIAL = "equatorial"
    UNKNOWN = "unknown"


class SlewRate:
    """Slew-speed preset on a 0-9 scale.

    Rate mappings (approximate):
      0       — guide rate  (~0.5× sidereal)
      1-3     — centering    (1-8× sidereal)
      4-6     — find         (16-64× sidereal)
      7-9     — slew         (up to 1440× sidereal)
    """

    GUIDE = 0
    CENTER = 3
    FIND = 6
    MAX = 9

    def __init__(self, value: int = 6) -> None:
        self.value = max(0, min(9, value))

    def __repr__(self) -> str:
        return f"SlewRate({self.value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SlewRate):
            return self.value == other.value
        return NotImplemented


# ---------------------------------------------------------------------------
# Command generation — returns the raw command string to send.
# ---------------------------------------------------------------------------


class ZwoAmCommands:
    """Pure-function command builders for the ZWO AM serial protocol."""

    # --- Getters (expect a response) ---

    @staticmethod
    def get_version() -> str:
        return ":GV#"

    @staticmethod
    def get_mount_model() -> str:
        return ":GVP#"

    @staticmethod
    def get_ra() -> str:
        """Response: ``HH:MM:SS#``"""
        return ":GR#"

    @staticmethod
    def get_dec() -> str:
        """Response: ``sDD*MM:SS#``"""
        return ":GD#"

    @staticmethod
    def get_azimuth() -> str:
        return ":GZ#"

    @staticmethod
    def get_altitude() -> str:
        return ":GA#"

    @staticmethod
    def get_sidereal_time() -> str:
        return ":GS#"

    @staticmethod
    def get_latitude() -> str:
        return ":Gt#"

    @staticmethod
    def get_longitude() -> str:
        return ":Gg#"

    @staticmethod
    def get_guide_rate() -> str:
        return ":Ggr#"

    @staticmethod
    def get_tracking_status() -> str:
        """Response: ``0#`` or ``1#``."""
        return ":GAT#"

    @staticmethod
    def get_status() -> str:
        """Response: status flags — ``n`` not tracking, ``N`` not slewing,
        ``H`` at home, ``P`` parked, ``G`` equatorial, ``Z`` altaz."""
        return ":GU#"

    @staticmethod
    def get_pier_side() -> str:
        return ":Gm#"

    @staticmethod
    def get_meridian_flip_settings() -> str:
        """Get compound meridian flip settings.

        Response: ``ET{s}DD#`` where E = flip enabled (0/1),
        T = track after meridian (0/1), {s}DD = signed degree limit.
        Example: ``10+00#`` = flip on, stop tracking, limit +0°.
        """
        return ":GTa#"

    @staticmethod
    def get_altitude_limit_enabled() -> str:
        """Get altitude-limit enabled status.  Response: ``0#`` or ``1#``."""
        return ":GLC#"

    @staticmethod
    def get_altitude_limit_upper() -> str:
        """Get upper altitude limit (ZWO proprietary).  Response: ``DD#``."""
        return ":GLH#"

    @staticmethod
    def get_altitude_limit_lower() -> str:
        """Get lower altitude limit (ZWO proprietary).  Response: ``DD#``."""
        return ":GLL#"

    # --- Target-coordinate setters (return ``1``/``0``) ---

    @staticmethod
    def set_target_ra(hours: int, minutes: int, seconds: int) -> str:
        return f":Sr{hours:02d}:{minutes:02d}:{seconds:02d}#"

    @staticmethod
    def set_target_ra_decimal(ra_hours: float) -> str:
        """Build ``:SrHH:MM:SS#`` from decimal hours."""
        total_seconds = round(ra_hours * 3600.0)
        h = (total_seconds // 3600) % 24
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f":Sr{h:02d}:{m:02d}:{s:02d}#"

    @staticmethod
    def set_target_dec(degrees: int, minutes: int, seconds: int) -> str:
        sign = "+" if degrees >= 0 else "-"
        return f":Sd{sign}{abs(degrees):02d}*{minutes:02d}:{seconds:02d}#"

    @staticmethod
    def set_target_dec_decimal(dec_degrees: float) -> str:
        """Build ``:SdsDD*MM:SS#`` from decimal degrees."""
        sign = "+" if dec_degrees >= 0.0 else "-"
        total_arcsec = round(abs(dec_degrees) * 3600.0)
        d = total_arcsec // 3600
        m = (total_arcsec % 3600) // 60
        s = total_arcsec % 60
        return f":Sd{sign}{d:02d}*{m:02d}:{s:02d}#"

    @staticmethod
    def set_target_azimuth_decimal(az_degrees: float) -> str:
        az = az_degrees % 360.0
        total_arcsec = round(az * 3600.0)
        d = (total_arcsec // 3600) % 360
        m = (total_arcsec % 3600) // 60
        s = total_arcsec % 60
        return f":Sz{d:03d}*{m:02d}:{s:02d}#"

    @staticmethod
    def set_target_altitude_decimal(alt_degrees: float) -> str:
        sign = "+" if alt_degrees >= 0.0 else "-"
        total_arcsec = round(abs(alt_degrees) * 3600.0)
        d = total_arcsec // 3600
        m = (total_arcsec % 3600) // 60
        s = total_arcsec % 60
        return f":Sa{sign}{d}*{m:02d}:{s:02d}#"

    # --- GoTo / Sync ---

    @staticmethod
    def goto() -> str:
        """Initiate slew to previously-set target.
        Response: ``0`` success, ``1``-``7`` various errors."""
        return ":MS#"

    @staticmethod
    def sync() -> str:
        """Sync mount model to previously-set target coordinates."""
        return ":CM#"

    @staticmethod
    def stop_all() -> str:
        return ":Q#"

    # --- Manual motion (fire-and-forget) ---

    @staticmethod
    def move_direction(direction: Direction) -> str:
        _cmds = {
            Direction.NORTH: ":Mn#",
            Direction.SOUTH: ":Ms#",
            Direction.EAST: ":Me#",
            Direction.WEST: ":Mw#",
        }
        return _cmds[direction]

    @staticmethod
    def stop_direction(direction: Direction) -> str:
        _cmds = {
            Direction.NORTH: ":Qn#",
            Direction.SOUTH: ":Qs#",
            Direction.EAST: ":Qe#",
            Direction.WEST: ":Qw#",
        }
        return _cmds[direction]

    @staticmethod
    def set_slew_rate(rate: SlewRate | int) -> str:
        v = rate.value if isinstance(rate, SlewRate) else max(0, min(9, rate))
        return f":R{v}#"

    # --- Guiding (fire-and-forget) ---

    @staticmethod
    def set_guide_rate(rate: float) -> str:
        """Rate is a fraction of sidereal, 0.1 – 0.9."""
        clamped = max(0.1, min(0.9, rate))
        return f":Rg{clamped:.1f}#"

    @staticmethod
    def guide_pulse(direction: Direction, duration_ms: int) -> str:
        suffix = {
            Direction.NORTH: "n",
            Direction.SOUTH: "s",
            Direction.EAST: "e",
            Direction.WEST: "w",
        }[direction]
        ms = max(0, min(9999, duration_ms))
        return f":Mg{suffix}{ms:04d}#"

    # --- Tracking (fire-and-forget) ---

    @staticmethod
    def tracking_on() -> str:
        return ":Te#"

    @staticmethod
    def tracking_off() -> str:
        return ":Td#"

    @staticmethod
    def set_tracking_rate(rate: TrackingRate) -> str:
        _cmds = {
            TrackingRate.SIDEREAL: ":TQ#",
            TrackingRate.LUNAR: ":TL#",
            TrackingRate.SOLAR: ":TS#",
            TrackingRate.OFF: ":Td#",
        }
        return _cmds[rate]

    # --- Home / Park (fire-and-forget) ---

    @staticmethod
    def find_home() -> str:
        return ":hC#"

    @staticmethod
    def goto_park() -> str:
        return ":hP#"

    @staticmethod
    def unpark() -> str:
        return ":hR#"

    @staticmethod
    def clear_alignment() -> str:
        return ":SRC#"

    # --- Mount mode (fire-and-forget) ---

    @staticmethod
    def set_altaz_mode() -> str:
        return ":AA#"

    @staticmethod
    def set_polar_mode() -> str:
        return ":AP#"

    # --- Site / time setters (return ``1``/``0``) ---

    @staticmethod
    def set_latitude(latitude: float) -> str:
        sign = "+" if latitude >= 0.0 else "-"
        total_arcsec = round(abs(latitude) * 3600.0)
        d = total_arcsec // 3600
        m = (total_arcsec % 3600) // 60
        s = total_arcsec % 60
        return f":St{sign}{d:02d}*{m:02d}:{s:02d}#"

    @staticmethod
    def set_longitude(longitude: float) -> str:
        """Build ``:Sg`` from east-positive decimal degrees.

        The AM5 (like all Meade LX200 variants) uses west-positive
        longitude: the input is negated and a ``+``/``-`` sign is
        prepended.  Matches the INDI ``lx200am5`` driver.
        """
        meade_lon = -longitude
        sign = "+" if meade_lon >= 0.0 else "-"
        total_arcsec = round(abs(meade_lon) * 3600.0)
        d = total_arcsec // 3600
        m = (total_arcsec % 3600) // 60
        s = total_arcsec % 60
        return f":Sg{sign}{d:03d}*{m:02d}:{s:02d}#"

    @staticmethod
    def set_date(month: int, day: int, year: int) -> str:
        return f":SC{month:02d}/{day:02d}/{year % 100:02d}#"

    @staticmethod
    def set_time(hour: int, minute: int, second: int) -> str:
        return f":SL{hour:02d}:{minute:02d}:{second:02d}#"

    @staticmethod
    def set_timezone(offset: int) -> str:
        sign = "+" if offset >= 0 else "-"
        return f":SG{sign}{abs(offset):02d}#"

    # --- Limits / meridian / buzzer ---

    @staticmethod
    def set_altitude_limit_enabled(enable: bool) -> str:
        """Enable or disable altitude limits (fire-and-forget, no response)."""
        return ":SLE#" if enable else ":SLD#"

    @staticmethod
    def set_altitude_limit_upper(degrees: int) -> str:
        """Set upper altitude limit (ZWO proprietary).  Response: ``1`` or ``0``."""
        clamped = max(60, min(90, degrees))
        return f":SLH{clamped:02d}#"

    @staticmethod
    def set_altitude_limit_lower(degrees: int) -> str:
        """Set lower altitude limit (ZWO proprietary).  Response: ``1`` or ``0``."""
        clamped = max(0, min(30, degrees))
        return f":SLL{clamped:02d}#"

    @staticmethod
    def set_meridian_flip_settings(enabled: bool, track_after: bool, limit: int) -> str:
        """Set compound meridian flip settings.  Response: ``1`` or ``0``.

        Args:
            enabled: Whether the mount should auto-flip at the meridian.
            track_after: Whether to continue tracking after a meridian flip.
            limit: Degrees past the meridian before triggering a flip (-15 to +15).
        """
        clamped = max(-15, min(15, limit))
        sign = "+" if clamped >= 0 else "-"
        return f":STa{1 if enabled else 0}{1 if track_after else 0}{sign}{abs(clamped):02d}#"

    @staticmethod
    def set_buzzer_volume(volume: int) -> str:
        return f":SBu{min(2, volume)}#"


# ---------------------------------------------------------------------------
# Response parsing — all methods accept the raw ``#``-terminated response.
# ---------------------------------------------------------------------------


class ZwoAmResponseParser:
    """Pure-function parsers for ZWO AM mount responses."""

    @staticmethod
    def parse_bool(response: str) -> bool | None:
        """Parse ``0#`` / ``1#`` (or without ``#``)."""
        trimmed = response.strip().rstrip("#")
        if trimmed == "1":
            return True
        if trimmed == "0":
            return False
        return None

    @staticmethod
    def parse_ra(response: str) -> tuple[int, int, float] | None:
        """Parse ``HH:MM:SS#`` → (hours, minutes, seconds)."""
        trimmed = response.strip().rstrip("#")
        parts = trimmed.split(":")
        try:
            if len(parts) >= 3:
                return int(parts[0]), int(parts[1]), float(parts[2])
            if len(parts) == 2:
                h = int(parts[0])
                min_frac = float(parts[1])
                m = int(min_frac)
                s = (min_frac - m) * 60.0
                return h, m, s
        except ValueError:
            pass
        return None

    @staticmethod
    def parse_dec(response: str) -> tuple[float, int, float] | None:
        """Parse Dec response → (signed_degrees, minutes, seconds).

        Accepts both ``sDD*MM:SS#`` (standard LX200) and ``sDD:MM:SS#``
        (colon-only variant seen on some ZWO firmware versions).

        Degrees is a float so the sign survives even when degrees == 0
        (e.g. ``-00*30:00`` → ``-0.0, 30, 0.0``).
        """
        trimmed = response.strip().rstrip("#")

        sign: float = 1.0
        rest = trimmed
        if trimmed.startswith("+"):
            rest = trimmed[1:]
        elif trimmed.startswith("-"):
            sign = -1.0
            rest = trimmed[1:]

        rest = rest.replace("°", "*")
        star_parts = rest.split("*")
        if len(star_parts) >= 2:
            try:
                degrees = int(star_parts[0])
                min_sec = star_parts[1].split(":")
                minutes = int(min_sec[0])
                seconds = float(min_sec[1]) if len(min_sec) > 1 else 0.0
                return sign * float(degrees), minutes, seconds
            except (ValueError, IndexError):
                return None

        colon_parts = rest.split(":")
        if len(colon_parts) >= 2:
            try:
                degrees = int(colon_parts[0])
                minutes = int(colon_parts[1])
                seconds = float(colon_parts[2]) if len(colon_parts) > 2 else 0.0
                return sign * float(degrees), minutes, seconds
            except (ValueError, IndexError):
                return None

        return None

    @staticmethod
    def parse_azimuth(response: str) -> tuple[int, int, float] | None:
        """Parse ``DDD*MM:SS#`` → (degrees, minutes, seconds)."""
        trimmed = response.strip().rstrip("#").replace("°", "*")
        star_parts = trimmed.split("*")
        if len(star_parts) < 2:
            return None
        try:
            degrees = int(star_parts[0])
            min_sec = star_parts[1].split(":")
            minutes = int(min_sec[0])
            seconds = float(min_sec[1]) if len(min_sec) > 1 else 0.0
            return degrees, minutes, seconds
        except (ValueError, IndexError):
            return None

    @staticmethod
    def parse_site_coordinate(response: str) -> float | None:
        """Parse a site latitude/longitude DMS response to decimal degrees.

        `:Gt#` returns ``sDD*MM:SS#``, `:Gg#` returns ``sDDD*MM:SS#``.
        Both use the same signed-DMS format as Dec responses.
        """
        parsed = ZwoAmResponseParser.parse_dec(response)
        if parsed is None:
            return None
        degrees, minutes, seconds = parsed
        sign = -1.0 if degrees < 0 or (degrees == 0 and str(response).lstrip().startswith("-")) else 1.0
        return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)

    @staticmethod
    def parse_goto_response(response: str) -> str | None:
        """Parse GoTo / Sync response.  Returns ``None`` on success, error string otherwise.

        ZWO AM5 error codes (differs from standard OnStepX mapping):

        ====  ==============================  ================================
        Code  ZWO meaning                     Standard OnStepX meaning
        ====  ==============================  ================================
        1     Below horizon                   Below horizon
        2     Above overhead limit            Above overhead limit
        3     In standby                      In standby
        4     Parked                          Parked
        5     **Not aligned**                 Goto in progress (CE_SLEW_IN_SLEW)
        6     Outside limits                  Outside limits
        7     Pier side limit                 Hardware fault
        8     In motion                       In motion
        ====  ==============================  ================================

        The ``e``-prefixed variants (``e5``, ``e6``, etc.) are equivalent to
        the bare digit forms.  ``:CM#`` returns ``N/A#`` on success.
        """
        trimmed = response.strip().rstrip("#")
        if trimmed == "N/A":
            return None
        _errors = {
            "0": None,
            "1": "Object below horizon",
            "2": "Above overhead limit",
            "3": "In standby (motors disabled)",
            "4": "Parked",
            "5": "Not aligned",
            "6": "Outside limits",
            "7": "Pier side limit",
            "8": "In motion",
            "e1": "Object below horizon",
            "e2": "Above overhead limit",
            "e3": "In standby (motors disabled)",
            "e4": "Parked",
            "e5": "Not aligned",
            "e6": "Outside limits",
            "e7": "Pier side limit",
            "e8": "In motion",
        }
        return _errors.get(trimmed, f"Unknown goto error: {trimmed}")

    @staticmethod
    def parse_status(response: str) -> tuple[bool, bool, bool, bool, MountMode]:
        """Parse ``:GU#`` flags → (tracking, slewing, at_home, parked, mount_mode).

        Flag reference (OnStep / ZWO AM variant):
          ``n`` not tracking, ``N`` not slewing,
          ``H`` at home, ``P`` parked, ``p``/``F`` park failed,
          ``G`` equatorial, ``Z`` altaz.
        """
        flags = response.strip().rstrip("#")
        tracking = "n" not in flags
        slewing = "N" not in flags
        at_home = "H" in flags
        parked = "P" in flags
        if "G" in flags:
            mode = MountMode.EQUATORIAL
        elif "Z" in flags:
            mode = MountMode.ALTAZ
        else:
            mode = MountMode.UNKNOWN
        return tracking, slewing, at_home, parked, mode

    # --- Limit / meridian parsers ---

    @staticmethod
    def parse_altitude_limit(response: str) -> int | None:
        """Parse ZWO altitude limit response ``DD#`` → integer degrees."""
        trimmed = response.strip().rstrip("#")
        try:
            return int(trimmed)
        except ValueError:
            return None

    @staticmethod
    def parse_meridian_flip_settings(response: str) -> tuple[bool, bool, int]:
        """Parse ``:GTa#`` compound response → (flip_enabled, track_after, limit_degrees).

        ZWO AM firmware returns 5+ chars like ``10+00`` where:
          [0]   flip enabled (``1``/``0``)
          [1]   track after meridian (``1``/``0``)
          [2]   sign (``+``/``-``)
          [3:]  degrees (e.g. ``00``, ``15``)
        """
        trimmed = response.strip().rstrip("#")
        if len(trimmed) < 4:
            return False, True, 0
        flip = trimmed[0] == "1"
        track = trimmed[1] == "1"
        sign_char = trimmed[2]
        try:
            degrees = int(trimmed[3:])
        except ValueError:
            degrees = 0
        if sign_char == "-":
            degrees = -degrees
        return flip, track, degrees

    # --- Coordinate helpers ---

    @staticmethod
    def hms_to_decimal_hours(hours: int, minutes: int, seconds: float) -> float:
        """Convert H:M:S to decimal hours."""
        return hours + minutes / 60.0 + seconds / 3600.0

    @staticmethod
    def dms_to_decimal_degrees(degrees: float, minutes: int, seconds: float) -> float:
        """Convert signed-D:M:S to decimal degrees.

        Uses ``math.copysign`` so that ``-0.0`` from the Dec parser
        correctly yields a negative result.
        """
        sign = math.copysign(1.0, degrees)
        return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)
