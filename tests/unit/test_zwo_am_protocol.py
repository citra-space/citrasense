"""Tests for ZWO AM mount protocol — command generation and response parsing.

Pure-function tests that need no hardware.  Command strings and response
formats are validated against the INDI lx200am5 driver source.

See ``zwo_am_protocol.py`` module docstring for the full protocol reference:
supported commands, broken commands, response formats, and error codes.
"""

import pytest

from citrascope.hardware.devices.mount.zwo_am_protocol import (
    Direction,
    MountMode,
    SlewRate,
    TrackingRate,
    ZwoAmCommands,
    ZwoAmResponseParser,
)

# ===================================================================
# Command generation
# ===================================================================


class TestZwoAmCommands:
    # --- target coordinate setters ---

    def test_set_target_ra(self):
        assert ZwoAmCommands.set_target_ra(12, 30, 45) == ":Sr12:30:45#"

    def test_set_target_ra_decimal(self):
        assert ZwoAmCommands.set_target_ra_decimal(12.5) == ":Sr12:30:00#"

    def test_set_target_ra_decimal_zero(self):
        assert ZwoAmCommands.set_target_ra_decimal(0.0) == ":Sr00:00:00#"

    def test_set_target_dec(self):
        assert ZwoAmCommands.set_target_dec(45, 30, 15) == ":Sd+45*30:15#"

    def test_set_target_dec_negative(self):
        assert ZwoAmCommands.set_target_dec(-23, 26, 21) == ":Sd-23*26:21#"

    def test_set_target_dec_decimal(self):
        assert ZwoAmCommands.set_target_dec_decimal(45.5) == ":Sd+45*30:00#"

    def test_set_target_dec_decimal_negative(self):
        assert ZwoAmCommands.set_target_dec_decimal(-23.5) == ":Sd-23*30:00#"

    def test_set_target_dec_decimal_zero(self):
        assert ZwoAmCommands.set_target_dec_decimal(0.0) == ":Sd+00*00:00#"

    # --- azimuth / altitude ---

    def test_set_target_azimuth_decimal(self):
        assert ZwoAmCommands.set_target_azimuth_decimal(180.0) == ":Sz180*00:00#"

    def test_set_target_azimuth_decimal_fraction(self):
        assert ZwoAmCommands.set_target_azimuth_decimal(90.5) == ":Sz090*30:00#"

    def test_set_target_azimuth_decimal_negative_wraps(self):
        assert ZwoAmCommands.set_target_azimuth_decimal(-90.0) == ":Sz270*00:00#"

    def test_set_target_altitude_decimal_positive(self):
        assert ZwoAmCommands.set_target_altitude_decimal(45.0) == ":Sa+45*00:00#"

    def test_set_target_altitude_decimal_negative(self):
        assert ZwoAmCommands.set_target_altitude_decimal(-10.5) == ":Sa-10*30:00#"

    # --- slew / motion ---

    def test_set_slew_rate_max(self):
        assert ZwoAmCommands.set_slew_rate(SlewRate(9)) == ":R9#"

    def test_set_slew_rate_guide(self):
        assert ZwoAmCommands.set_slew_rate(SlewRate(0)) == ":R0#"

    def test_set_slew_rate_int(self):
        assert ZwoAmCommands.set_slew_rate(5) == ":R5#"

    def test_set_slew_rate_clamps(self):
        assert ZwoAmCommands.set_slew_rate(99) == ":R9#"

    # --- guide pulse ---

    def test_guide_pulse_north(self):
        assert ZwoAmCommands.guide_pulse(Direction.NORTH, 500) == ":Mgn0500#"

    def test_guide_pulse_east(self):
        assert ZwoAmCommands.guide_pulse(Direction.EAST, 150) == ":Mge0150#"

    def test_guide_pulse_clamps_max(self):
        assert ZwoAmCommands.guide_pulse(Direction.WEST, 99999) == ":Mgw9999#"

    # --- tracking ---

    def test_tracking_sidereal(self):
        assert ZwoAmCommands.set_tracking_rate(TrackingRate.SIDEREAL) == ":TQ#"

    def test_tracking_lunar(self):
        assert ZwoAmCommands.set_tracking_rate(TrackingRate.LUNAR) == ":TL#"

    def test_tracking_solar(self):
        assert ZwoAmCommands.set_tracking_rate(TrackingRate.SOLAR) == ":TS#"

    def test_tracking_off(self):
        assert ZwoAmCommands.set_tracking_rate(TrackingRate.OFF) == ":Td#"

    def test_tracking_on(self):
        assert ZwoAmCommands.tracking_on() == ":Te#"

    def test_tracking_off_cmd(self):
        assert ZwoAmCommands.tracking_off() == ":Td#"

    # --- site location ---

    def test_set_latitude_positive(self):
        assert ZwoAmCommands.set_latitude(46.5) == ":St+46*30:00#"

    def test_set_latitude_negative(self):
        assert ZwoAmCommands.set_latitude(-33.75) == ":St-33*45:00#"

    def test_set_latitude_san_francisco(self):
        cmd = ZwoAmCommands.set_latitude(37.7749)
        assert cmd == ":St+37*46:30#"

    def test_set_longitude_east_positive(self):
        # East-positive input (e.g. Zurich at 6.25°E) becomes Meade west-negative
        assert ZwoAmCommands.set_longitude(6.25) == ":Sg-006*15:00#"

    def test_set_longitude_west_negative(self):
        # West longitude (e.g. Los Angeles at -118.5°) becomes Meade west-positive
        assert ZwoAmCommands.set_longitude(-118.5) == ":Sg+118*30:00#"

    def test_set_longitude_san_francisco(self):
        # Matches INDI lx200am5 driver output for SF
        cmd = ZwoAmCommands.set_longitude(-122.4194)
        assert cmd.startswith(":Sg+122*25:")

    def test_set_longitude_greenwich(self):
        assert ZwoAmCommands.set_longitude(0.0) == ":Sg+000*00:00#"

    # --- guide rate ---

    def test_set_guide_rate(self):
        assert ZwoAmCommands.set_guide_rate(0.5) == ":Rg0.5#"

    def test_set_guide_rate_clamps(self):
        assert ZwoAmCommands.set_guide_rate(1.5) == ":Rg0.9#"

    # --- goto / sync / stop ---

    def test_goto(self):
        assert ZwoAmCommands.goto() == ":MS#"

    def test_sync(self):
        assert ZwoAmCommands.sync() == ":CM#"

    def test_stop_all(self):
        assert ZwoAmCommands.stop_all() == ":Q#"

    # --- motion directions ---

    def test_move_directions(self):
        assert ZwoAmCommands.move_direction(Direction.NORTH) == ":Mn#"
        assert ZwoAmCommands.move_direction(Direction.SOUTH) == ":Ms#"
        assert ZwoAmCommands.move_direction(Direction.EAST) == ":Me#"
        assert ZwoAmCommands.move_direction(Direction.WEST) == ":Mw#"

    def test_stop_directions(self):
        assert ZwoAmCommands.stop_direction(Direction.NORTH) == ":Qn#"
        assert ZwoAmCommands.stop_direction(Direction.SOUTH) == ":Qs#"
        assert ZwoAmCommands.stop_direction(Direction.EAST) == ":Qe#"
        assert ZwoAmCommands.stop_direction(Direction.WEST) == ":Qw#"

    # --- home / park ---

    def test_park(self):
        assert ZwoAmCommands.goto_park() == ":hP#"

    def test_unpark(self):
        assert ZwoAmCommands.unpark() == ":hR#"

    def test_find_home(self):
        assert ZwoAmCommands.find_home() == ":hC#"

    # --- altitude limits (ZWO-specific :GL/:SL commands) ---

    def test_get_altitude_limit_enabled(self):
        assert ZwoAmCommands.get_altitude_limit_enabled() == ":GLC#"

    def test_set_altitude_limit_enabled_on(self):
        assert ZwoAmCommands.set_altitude_limit_enabled(True) == ":SLE#"

    def test_set_altitude_limit_enabled_off(self):
        assert ZwoAmCommands.set_altitude_limit_enabled(False) == ":SLD#"

    def test_get_altitude_limit_upper(self):
        assert ZwoAmCommands.get_altitude_limit_upper() == ":GLH#"

    def test_get_altitude_limit_lower(self):
        assert ZwoAmCommands.get_altitude_limit_lower() == ":GLL#"

    def test_set_altitude_limit_upper(self):
        assert ZwoAmCommands.set_altitude_limit_upper(90) == ":SLH90#"

    def test_set_altitude_limit_upper_clamps_low(self):
        assert ZwoAmCommands.set_altitude_limit_upper(30) == ":SLH60#"

    def test_set_altitude_limit_lower(self):
        assert ZwoAmCommands.set_altitude_limit_lower(10) == ":SLL10#"

    def test_set_altitude_limit_lower_zero(self):
        assert ZwoAmCommands.set_altitude_limit_lower(0) == ":SLL00#"

    def test_set_altitude_limit_lower_clamps(self):
        assert ZwoAmCommands.set_altitude_limit_lower(50) == ":SLL30#"

    # --- meridian flip (ZWO compound :GTa/:STa commands) ---

    def test_get_meridian_flip_settings(self):
        assert ZwoAmCommands.get_meridian_flip_settings() == ":GTa#"

    def test_set_meridian_flip_all_on(self):
        assert ZwoAmCommands.set_meridian_flip_settings(True, True, 0) == ":STa11+00#"

    def test_set_meridian_flip_disabled(self):
        assert ZwoAmCommands.set_meridian_flip_settings(False, True, 0) == ":STa01+00#"

    def test_set_meridian_flip_negative_limit(self):
        assert ZwoAmCommands.set_meridian_flip_settings(True, False, -5) == ":STa10-05#"

    def test_set_meridian_flip_positive_limit(self):
        assert ZwoAmCommands.set_meridian_flip_settings(True, True, 10) == ":STa11+10#"

    def test_set_meridian_flip_clamps_limit(self):
        assert ZwoAmCommands.set_meridian_flip_settings(True, True, 30) == ":STa11+15#"

    # --- mount mode ---

    def test_altaz_mode(self):
        assert ZwoAmCommands.set_altaz_mode() == ":AA#"

    def test_polar_mode(self):
        assert ZwoAmCommands.set_polar_mode() == ":AP#"

    # --- getters ---

    def test_getter_commands(self):
        assert ZwoAmCommands.get_ra() == ":GR#"
        assert ZwoAmCommands.get_dec() == ":GD#"
        assert ZwoAmCommands.get_status() == ":GU#"
        assert ZwoAmCommands.get_mount_model() == ":GVP#"
        assert ZwoAmCommands.get_version() == ":GV#"
        assert ZwoAmCommands.get_azimuth() == ":GZ#"
        assert ZwoAmCommands.get_altitude() == ":GA#"

    # --- date / time ---

    def test_set_date(self):
        assert ZwoAmCommands.set_date(3, 15, 2026) == ":SC03/15/26#"

    def test_set_time(self):
        assert ZwoAmCommands.set_time(22, 5, 30) == ":SL22:05:30#"

    def test_set_timezone(self):
        assert ZwoAmCommands.set_timezone(-5) == ":SG-05#"
        assert ZwoAmCommands.set_timezone(2) == ":SG+02#"


# ===================================================================
# Response parsing
# ===================================================================


class TestZwoAmResponseParser:
    # --- boolean ---

    def test_parse_bool_true(self):
        assert ZwoAmResponseParser.parse_bool("1#") is True

    def test_parse_bool_false(self):
        assert ZwoAmResponseParser.parse_bool("0#") is False

    def test_parse_bool_no_hash(self):
        assert ZwoAmResponseParser.parse_bool("1") is True

    def test_parse_bool_invalid(self):
        assert ZwoAmResponseParser.parse_bool("invalid") is None

    # --- RA ---

    def test_parse_ra_hms(self):
        result = ZwoAmResponseParser.parse_ra("12:30:45#")
        assert result is not None
        h, m, s = result
        assert h == 12
        assert m == 30
        assert abs(s - 45.0) < 0.001

    def test_parse_ra_hm_fraction(self):
        result = ZwoAmResponseParser.parse_ra("12:30.5#")
        assert result is not None
        h, m, s = result
        assert h == 12
        assert m == 30
        assert abs(s - 30.0) < 0.1

    def test_parse_ra_invalid(self):
        assert ZwoAmResponseParser.parse_ra("garbage") is None

    # --- Dec ---

    def test_parse_dec_positive(self):
        result = ZwoAmResponseParser.parse_dec("+45*30:15#")
        assert result is not None
        d, m, s = result
        assert d == 45
        assert m == 30
        assert abs(s - 15.0) < 0.001

    def test_parse_dec_negative(self):
        result = ZwoAmResponseParser.parse_dec("-23*26:21#")
        assert result is not None
        d, m, s = result
        assert d == -23
        assert m == 26
        assert abs(s - 21.0) < 0.001

    def test_parse_dec_no_seconds(self):
        result = ZwoAmResponseParser.parse_dec("+45*30#")
        assert result is not None
        d, m, s = result
        assert d == 45
        assert m == 30
        assert s == 0.0

    def test_parse_dec_degree_symbol(self):
        result = ZwoAmResponseParser.parse_dec("+45°30:15#")
        assert result is not None
        assert result[0] == 45

    def test_parse_dec_invalid(self):
        assert ZwoAmResponseParser.parse_dec("garbage") is None

    def test_parse_dec_colon_only_positive(self):
        result = ZwoAmResponseParser.parse_dec("2:45:40#")
        assert result is not None
        d, m, s = result
        assert d == 2.0
        assert m == 45
        assert abs(s - 40.0) < 0.001

    def test_parse_dec_colon_only_negative(self):
        result = ZwoAmResponseParser.parse_dec("-15:30:00#")
        assert result is not None
        d, m, s = result
        assert d == -15.0
        assert m == 30
        assert abs(s - 0.0) < 0.001

    def test_parse_dec_colon_only_negative_zero(self):
        result = ZwoAmResponseParser.parse_dec("-0:30:00#")
        assert result is not None
        d, m, s = result
        assert d == -0.0
        assert m == 30
        recovered = ZwoAmResponseParser.dms_to_decimal_degrees(d, m, s)
        assert recovered < 0

    def test_parse_dec_colon_only_with_sign_prefix(self):
        result = ZwoAmResponseParser.parse_dec("+45:15:30#")
        assert result is not None
        d, m, s = result
        assert d == 45.0
        assert m == 15
        assert abs(s - 30.0) < 0.001

    # --- azimuth ---

    def test_parse_azimuth(self):
        result = ZwoAmResponseParser.parse_azimuth("180*30:45#")
        assert result is not None
        d, m, s = result
        assert d == 180
        assert m == 30
        assert abs(s - 45.0) < 0.001

    # --- goto response ---

    def test_parse_goto_success(self):
        assert ZwoAmResponseParser.parse_goto_response("0#") is None

    def test_parse_goto_below_horizon(self):
        result = ZwoAmResponseParser.parse_goto_response("1#")
        assert result is not None
        assert "horizon" in result.lower()

    def test_parse_goto_pier_limit(self):
        result = ZwoAmResponseParser.parse_goto_response("7#")
        assert result is not None
        assert "pier" in result.lower()

    def test_parse_goto_e7(self):
        result = ZwoAmResponseParser.parse_goto_response("e7#")
        assert result is not None
        assert "pier" in result.lower()

    def test_parse_goto_above_overhead(self):
        result = ZwoAmResponseParser.parse_goto_response("2#")
        assert result is not None
        assert "overhead" in result.lower()

    def test_parse_goto_in_standby(self):
        result = ZwoAmResponseParser.parse_goto_response("3#")
        assert result is not None
        assert "standby" in result.lower()

    def test_parse_goto_e3_in_standby(self):
        result = ZwoAmResponseParser.parse_goto_response("e3#")
        assert result is not None
        assert "standby" in result.lower()

    def test_parse_goto_e6_outside_limits(self):
        result = ZwoAmResponseParser.parse_goto_response("e6#")
        assert result is not None
        assert "outside" in result.lower() or "limits" in result.lower()

    def test_parse_goto_e_prefixed_codes(self):
        assert ZwoAmResponseParser.parse_goto_response("e1#") is not None
        assert "horizon" in ZwoAmResponseParser.parse_goto_response("e1#").lower()  # type: ignore[union-attr]
        assert ZwoAmResponseParser.parse_goto_response("e2#") is not None
        assert "overhead" in ZwoAmResponseParser.parse_goto_response("e2#").lower()  # type: ignore[union-attr]
        assert ZwoAmResponseParser.parse_goto_response("e3#") is not None
        assert "standby" in ZwoAmResponseParser.parse_goto_response("e3#").lower()  # type: ignore[union-attr]
        assert ZwoAmResponseParser.parse_goto_response("e5#") is not None
        assert "aligned" in ZwoAmResponseParser.parse_goto_response("e5#").lower()  # type: ignore[union-attr]

    def test_parse_goto_sync_success(self):
        assert ZwoAmResponseParser.parse_goto_response("N/A#") is None

    def test_parse_goto_unknown(self):
        result = ZwoAmResponseParser.parse_goto_response("99#")
        assert result is not None
        assert "unknown" in result.lower()

    # --- site coordinate ---

    def test_parse_site_latitude(self):
        result = ZwoAmResponseParser.parse_site_coordinate("+38*50:27#")
        assert result is not None
        assert abs(result - 38.8408) < 0.001

    def test_parse_site_latitude_negative(self):
        result = ZwoAmResponseParser.parse_site_coordinate("-33*52:00#")
        assert result is not None
        assert result < 0
        assert abs(result - (-33.8667)) < 0.001

    def test_parse_site_longitude(self):
        result = ZwoAmResponseParser.parse_site_coordinate("+105*02:32#")
        assert result is not None
        assert abs(result - 105.0422) < 0.001

    def test_parse_site_coordinate_garbage(self):
        assert ZwoAmResponseParser.parse_site_coordinate("garbage") is None

    # --- status flags ---

    def test_parse_status_tracking_equatorial(self):
        tracking, slewing, at_home, parked, mode = ZwoAmResponseParser.parse_status("NG#")
        assert tracking is True
        assert slewing is False
        assert at_home is False
        assert parked is False
        assert mode == MountMode.EQUATORIAL

    def test_parse_status_idle_home_altaz(self):
        tracking, slewing, at_home, parked, mode = ZwoAmResponseParser.parse_status("nNHZ#")
        assert tracking is False
        assert slewing is False
        assert at_home is True
        assert parked is False
        assert mode == MountMode.ALTAZ

    def test_parse_status_slewing(self):
        tracking, slewing, at_home, parked, mode = ZwoAmResponseParser.parse_status("G#")
        assert tracking is True
        assert slewing is True
        assert at_home is False
        assert parked is False
        assert mode == MountMode.EQUATORIAL

    def test_parse_status_parked(self):
        tracking, slewing, at_home, parked, mode = ZwoAmResponseParser.parse_status("nNHPG#")
        assert tracking is False
        assert slewing is False
        assert at_home is True
        assert parked is True
        assert mode == MountMode.EQUATORIAL

    def test_parse_status_parked_without_home(self):
        tracking, slewing, at_home, parked, mode = ZwoAmResponseParser.parse_status("nNPZ#")
        assert tracking is False
        assert slewing is False
        assert at_home is False
        assert parked is True
        assert mode == MountMode.ALTAZ

    # --- altitude limit parser (ZWO :GLH/:GLL) ---

    def test_parse_altitude_limit_90(self):
        assert ZwoAmResponseParser.parse_altitude_limit("90#") == 90

    def test_parse_altitude_limit_zero(self):
        assert ZwoAmResponseParser.parse_altitude_limit("00#") == 0

    def test_parse_altitude_limit_15(self):
        assert ZwoAmResponseParser.parse_altitude_limit("15#") == 15

    def test_parse_altitude_limit_invalid(self):
        assert ZwoAmResponseParser.parse_altitude_limit("garbage#") is None

    # --- meridian flip compound parser (:GTa#) ---

    def test_parse_meridian_flip_enabled_track_zero(self):
        flip, track, limit = ZwoAmResponseParser.parse_meridian_flip_settings("10+00#")
        assert flip is True
        assert track is False
        assert limit == 0

    def test_parse_meridian_flip_disabled(self):
        flip, track, limit = ZwoAmResponseParser.parse_meridian_flip_settings("01+00#")
        assert flip is False
        assert track is True
        assert limit == 0

    def test_parse_meridian_flip_all_on_positive(self):
        flip, track, limit = ZwoAmResponseParser.parse_meridian_flip_settings("11+10#")
        assert flip is True
        assert track is True
        assert limit == 10

    def test_parse_meridian_flip_negative_limit(self):
        flip, track, limit = ZwoAmResponseParser.parse_meridian_flip_settings("10-05#")
        assert flip is True
        assert track is False
        assert limit == -5

    def test_parse_meridian_flip_short_response(self):
        flip, track, limit = ZwoAmResponseParser.parse_meridian_flip_settings("#")
        assert flip is False
        assert track is True
        assert limit == 0

    # --- coordinate helpers ---

    def test_hms_to_decimal_hours(self):
        result = ZwoAmResponseParser.hms_to_decimal_hours(12, 30, 0)
        assert abs(result - 12.5) < 0.0001

    def test_hms_to_decimal_hours_zero(self):
        assert ZwoAmResponseParser.hms_to_decimal_hours(0, 0, 0) == 0.0

    def test_dms_to_decimal_degrees_positive(self):
        result = ZwoAmResponseParser.dms_to_decimal_degrees(45, 30, 0)
        assert abs(result - 45.5) < 0.0001

    def test_dms_to_decimal_degrees_negative(self):
        result = ZwoAmResponseParser.dms_to_decimal_degrees(-23, 30, 0)
        assert abs(result - (-23.5)) < 0.0001


# ===================================================================
# Enums
# ===================================================================


class TestEnums:
    def test_direction_opposite(self):
        assert Direction.NORTH.opposite == Direction.SOUTH
        assert Direction.SOUTH.opposite == Direction.NORTH
        assert Direction.EAST.opposite == Direction.WEST
        assert Direction.WEST.opposite == Direction.EAST

    def test_slew_rate_clamp(self):
        assert SlewRate(10).value == 9
        assert SlewRate(-1).value == 0
        assert SlewRate(5).value == 5

    def test_slew_rate_equality(self):
        assert SlewRate(3) == SlewRate(3)
        assert SlewRate(3) != SlewRate(5)

    def test_slew_rate_constants(self):
        assert SlewRate.GUIDE == 0
        assert SlewRate.CENTER == 3
        assert SlewRate.FIND == 6
        assert SlewRate.MAX == 9


# ===================================================================
# RA degree ↔ hour conversion boundary
# ===================================================================


class TestRaConversionBoundary:
    """Validates the RA degrees → hours → command → parse → hours → degrees
    round-trip that happens at the ZwoAmMount boundary."""

    @pytest.mark.parametrize(
        "ra_deg",
        [0.0, 45.0, 90.0, 180.0, 270.0, 359.0, 187.5],
    )
    def test_ra_roundtrip(self, ra_deg: float):
        ra_hours = ra_deg / 15.0
        cmd = ZwoAmCommands.set_target_ra_decimal(ra_hours)
        # The command encodes hours as HH:MM:SS — parse that back
        # Strip the :Sr prefix and # suffix to get "HH:MM:SS"
        inner = cmd[3:-1]
        parts = inner.split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        recovered_hours = ZwoAmResponseParser.hms_to_decimal_hours(h, m, s)
        recovered_deg = recovered_hours * 15.0
        # 1-second resolution → worst case 15 arcsec = 0.00417°
        assert abs(recovered_deg - ra_deg) < 0.01, f"{ra_deg}° → {recovered_deg}°"

    @pytest.mark.parametrize(
        "dec_deg",
        [0.0, 45.5, -23.5, 90.0, -90.0, -0.5],
    )
    def test_dec_roundtrip(self, dec_deg: float):
        cmd = ZwoAmCommands.set_target_dec_decimal(dec_deg)
        inner = cmd[3:-1]  # strip :Sd and #
        parsed = ZwoAmResponseParser.parse_dec(inner + "#")
        assert parsed is not None
        recovered = ZwoAmResponseParser.dms_to_decimal_degrees(*parsed)
        assert abs(recovered - dec_deg) < 0.01, f"{dec_deg}° → {recovered}°"
