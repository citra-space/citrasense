"""Tests for telescope configuration health assessment."""

import pytest

from citrascope.hardware.config_health import (
    ConfigHealth,
    HardwareConfigCheck,
    _safe_float,
    _safe_int,
    assess_config_health,
)

# ---------------------------------------------------------------------------
# Helper: a realistic telescope record (matches Citra API shape)
# ---------------------------------------------------------------------------

TELESCOPE_RECORD = {
    "pixelSize": 3.76,
    "focalLength": 600,
    "focalRatio": 3.9,
    "horizontalPixelCount": 6248,
    "verticalPixelCount": 4176,
    "maxSlewRate": 5.0,
}

CAMERA_INFO = {
    "pixel_size_um": 3.76,
    "width": 6248,
    "height": 4176,
    "model": "ASI533MC Pro",
}


def _find_check(health: ConfigHealth, name: str) -> HardwareConfigCheck | None:
    return next((c for c in health.checks if c.name == name), None)


# ---------------------------------------------------------------------------
# Empty / missing input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_none_record_returns_empty(self):
        health = assess_config_health(telescope_record=None)
        assert health.checks == []
        assert health.has_warnings is False

    def test_empty_dict_returns_empty(self):
        health = assess_config_health(telescope_record={})
        assert health.checks == []

    def test_record_with_no_useful_keys(self):
        health = assess_config_health(telescope_record={"name": "My Scope"})
        assert health.checks == []


# ---------------------------------------------------------------------------
# Focal length (config-only, never has observed data)
# ---------------------------------------------------------------------------


class TestFocalLength:
    def test_focal_length_present(self):
        health = assess_config_health(telescope_record={"focalLength": 600, "focalRatio": 3.9})
        chk = _find_check(health, "focal_length")
        assert chk is not None
        assert chk.configured == 600
        assert "600 mm" in chk.configured_fmt
        assert "f/3.9" in chk.configured_fmt
        assert chk.status == "pending"

    def test_focal_length_without_ratio(self):
        health = assess_config_health(telescope_record={"focalLength": 1000})
        chk = _find_check(health, "focal_length")
        assert chk is not None
        assert "f/" not in chk.configured_fmt


# ---------------------------------------------------------------------------
# Pixel size (config vs camera hardware)
# ---------------------------------------------------------------------------


class TestPixelSize:
    def test_config_only_no_camera(self):
        health = assess_config_health(telescope_record={"pixelSize": 3.76})
        chk = _find_check(health, "pixel_size")
        assert chk is not None
        assert chk.status == "pending"
        assert chk.observed is None

    def test_matching_camera(self):
        health = assess_config_health(
            telescope_record={"pixelSize": 3.76},
            camera_info={"pixel_size_um": 3.76},
        )
        chk = _find_check(health, "pixel_size")
        assert chk is not None
        assert chk.status == "ok"
        assert chk.pct_diff == 0.0
        assert chk.source == "camera"

    def test_mismatched_camera(self):
        health = assess_config_health(
            telescope_record={"pixelSize": 3.76},
            camera_info={"pixel_size_um": 4.63},
        )
        chk = _find_check(health, "pixel_size")
        assert chk is not None
        assert chk.status == "warning"
        assert chk.pct_diff is not None
        assert chk.pct_diff > 10.0

    def test_camera_info_missing_pixel_size_key(self):
        health = assess_config_health(
            telescope_record={"pixelSize": 3.76},
            camera_info={"width": 6248},
        )
        chk = _find_check(health, "pixel_size")
        assert chk is not None
        assert chk.status == "pending"


# ---------------------------------------------------------------------------
# Pixel scale (config vs plate solve)
# ---------------------------------------------------------------------------


class TestPixelScale:
    def test_config_only(self):
        health = assess_config_health(telescope_record={"pixelSize": 3.76, "focalLength": 600})
        chk = _find_check(health, "pixel_scale")
        assert chk is not None
        assert chk.status == "pending"
        expected_scale = 3.76 / 600 * 206.265
        assert chk.configured == pytest.approx(expected_scale, abs=0.01)

    def test_matching_plate_solve(self):
        expected_scale = 3.76 / 600 * 206.265
        health = assess_config_health(
            telescope_record={"pixelSize": 3.76, "focalLength": 600},
            observed_pixel_scale=expected_scale,
        )
        chk = _find_check(health, "pixel_scale")
        assert chk is not None
        assert chk.status == "ok"
        assert chk.pct_diff == 0.0

    def test_mismatched_plate_solve(self):
        cfg_scale = 3.76 / 600 * 206.265
        health = assess_config_health(
            telescope_record={"pixelSize": 3.76, "focalLength": 600},
            observed_pixel_scale=cfg_scale * 1.20,
        )
        chk = _find_check(health, "pixel_scale")
        assert chk is not None
        assert chk.status == "warning"
        assert chk.pct_diff is not None
        assert chk.pct_diff > 10.0


# ---------------------------------------------------------------------------
# Sensor resolution (config vs camera hardware)
# ---------------------------------------------------------------------------


class TestSensorResolution:
    def test_exact_match(self):
        health = assess_config_health(
            telescope_record={"horizontalPixelCount": 6248, "verticalPixelCount": 4176},
            camera_info={"width": 6248, "height": 4176},
        )
        chk = _find_check(health, "sensor_resolution")
        assert chk is not None
        assert chk.status == "ok"
        assert chk.pct_diff == 0.0

    def test_dimension_mismatch(self):
        health = assess_config_health(
            telescope_record={"horizontalPixelCount": 6248, "verticalPixelCount": 4176},
            camera_info={"width": 4656, "height": 3520},
        )
        chk = _find_check(health, "sensor_resolution")
        assert chk is not None
        assert chk.status == "warning"

    def test_physical_size_appended_when_pixel_size_present(self):
        health = assess_config_health(
            telescope_record={
                "pixelSize": 3.76,
                "horizontalPixelCount": 6248,
                "verticalPixelCount": 4176,
            },
        )
        chk = _find_check(health, "sensor_resolution")
        assert chk is not None
        assert "mm" in chk.configured_fmt

    def test_no_sensor_check_without_both_dims(self):
        health = assess_config_health(telescope_record={"horizontalPixelCount": 6248})
        assert _find_check(health, "sensor_resolution") is None


# ---------------------------------------------------------------------------
# FOV (config vs plate solve)
# ---------------------------------------------------------------------------


class TestFOV:
    def test_config_only(self):
        health = assess_config_health(telescope_record=TELESCOPE_RECORD)
        chk = _find_check(health, "fov")
        assert chk is not None
        assert chk.status == "pending"

    def test_matching_plate_solve(self):
        cfg_scale = 3.76 / 600 * 206.265
        cfg_fov_w = cfg_scale * 6248 / 3600.0
        cfg_fov_h = cfg_scale * 4176 / 3600.0
        health = assess_config_health(
            telescope_record=TELESCOPE_RECORD,
            observed_fov_w=cfg_fov_w,
            observed_fov_h=cfg_fov_h,
        )
        chk = _find_check(health, "fov")
        assert chk is not None
        assert chk.status == "ok"

    def test_mismatched_plate_solve(self):
        health = assess_config_health(
            telescope_record=TELESCOPE_RECORD,
            observed_fov_w=5.0,
            observed_fov_h=3.5,
        )
        chk = _find_check(health, "fov")
        assert chk is not None
        assert chk.status == "warning"

    def test_only_one_fov_axis_not_enough(self):
        health = assess_config_health(
            telescope_record=TELESCOPE_RECORD,
            observed_fov_w=2.0,
        )
        chk = _find_check(health, "fov")
        assert chk is not None
        assert chk.status == "pending"


# ---------------------------------------------------------------------------
# Slew rate (config vs observed tracking)
# ---------------------------------------------------------------------------


class TestSlewRate:
    def test_config_only(self):
        health = assess_config_health(telescope_record={"maxSlewRate": 5.0})
        chk = _find_check(health, "slew_rate")
        assert chk is not None
        assert chk.group == "telescope"
        assert chk.status == "pending"

    def test_matching_observed(self):
        health = assess_config_health(
            telescope_record={"maxSlewRate": 5.0},
            observed_slew_rate=5.0,
        )
        chk = _find_check(health, "slew_rate")
        assert chk is not None
        assert chk.status == "ok"

    def test_mismatched_observed(self):
        health = assess_config_health(
            telescope_record={"maxSlewRate": 5.0},
            observed_slew_rate=2.0,
        )
        chk = _find_check(health, "slew_rate")
        assert chk is not None
        assert chk.status == "warning"
        assert chk.pct_diff is not None
        assert chk.pct_diff > 10.0


# ---------------------------------------------------------------------------
# has_warnings flag
# ---------------------------------------------------------------------------


class TestHasWarnings:
    def test_all_ok(self):
        health = assess_config_health(
            telescope_record=TELESCOPE_RECORD,
            camera_info=CAMERA_INFO,
        )
        assert health.has_warnings is False

    def test_one_warning_sets_flag(self):
        bad_camera = {**CAMERA_INFO, "pixel_size_um": 9.0}
        health = assess_config_health(
            telescope_record=TELESCOPE_RECORD,
            camera_info=bad_camera,
        )
        assert health.has_warnings is True

    def test_no_observed_data_no_warnings(self):
        health = assess_config_health(telescope_record=TELESCOPE_RECORD)
        assert health.has_warnings is False


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------


class TestToDict:
    def test_serialization_shape(self):
        health = assess_config_health(
            telescope_record=TELESCOPE_RECORD,
            camera_info=CAMERA_INFO,
        )
        d = health.to_dict()
        assert "has_warnings" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)
        for chk in d["checks"]:
            assert {"name", "label", "group", "status"} <= set(chk.keys())


# ---------------------------------------------------------------------------
# Edge cases: _safe_float / _safe_int
# ---------------------------------------------------------------------------


class TestSafeConversions:
    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (None, None),
            (0, None),
            (-1.5, None),
            ("garbage", None),
            (3.76, 3.76),
            ("3.76", 3.76),
            (1, 1.0),
        ],
    )
    def test_safe_float(self, val, expected):
        result = _safe_float(val)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (None, None),
            (0, None),
            (-5, None),
            ("garbage", None),
            (6248, 6248),
            ("6248", 6248),
        ],
    )
    def test_safe_int(self, val, expected):
        assert _safe_int(val) == expected
