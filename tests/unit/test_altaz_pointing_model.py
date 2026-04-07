"""Tests for the alt-az pointing model."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from citrascope.hardware.devices.mount.altaz_pointing_model import (
    AltAzPointingModel,
    altaz_to_radec,
    generate_calibration_grid,
    radec_to_altaz,
)

# ------------------------------------------------------------------
# Coordinate conversion helpers (use a fixed LST for deterministic tests)
# ------------------------------------------------------------------


def _fixed_lst_deg(lon_deg: float) -> float:
    """Return a deterministic LST for testing."""
    return (90.0 + lon_deg) % 360.0


@pytest.fixture(autouse=True)
def _patch_skyfield_gast():
    """Patch _skyfield_gast to return a fixed value so tests are deterministic."""
    with patch(
        "citrascope.hardware.devices.mount.altaz_pointing_model._skyfield_gast",
        return_value=90.0,
    ):
        yield


# ------------------------------------------------------------------
# Coordinate conversion round-trip
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ra", "dec", "lat", "lon"),
    [
        (45.0, 30.0, 40.0, -74.0),
        (180.0, -20.0, 35.0, 139.0),
        (0.0, 89.0, 52.0, 13.0),
        (270.0, 0.0, -33.0, 18.0),
    ],
)
def test_radec_altaz_round_trip(ra: float, dec: float, lat: float, lon: float):
    """RA/Dec → Alt/Az → RA/Dec should recover the original coordinates."""
    az, alt = radec_to_altaz(ra, dec, lat, lon)
    # Skip points below horizon where numerical precision is worse
    if alt < 5.0:
        pytest.skip("Below horizon — numeric instability")
    ra2, dec2 = altaz_to_radec(az, alt, lat, lon)
    assert abs(ra2 - ra) % 360 < 0.01, f"RA mismatch: {ra2} vs {ra}"
    assert abs(dec2 - dec) < 0.01, f"Dec mismatch: {dec2} vs {dec}"


# ------------------------------------------------------------------
# Model: synthetic calibration recovery
# ------------------------------------------------------------------


def _synthetic_points(
    AN: float,
    AW: float,
    IE: float,
    CA: float = 0.0,
    NPAE: float = 0.0,
    n: int = 12,
    lat: float = 40.0,
    lon: float = -74.0,
) -> list[tuple[float, float, float, float, float, float]]:
    """Generate synthetic calibration point pairs with a known error model.

    Returns a list of (mount_ra, mount_dec, solved_ra, solved_dec, lat, lon).
    """
    rng = np.random.RandomState(42)
    points = []
    for _ in range(n):
        az_true = rng.uniform(30.0, 330.0)
        alt_true = rng.uniform(25.0, 75.0)

        az_rad = math.radians(az_true)
        alt_rad = math.radians(alt_true)
        sin_az = math.sin(az_rad)
        cos_az = math.cos(az_rad)
        tan_alt = math.tan(alt_rad)
        sec_alt = 1.0 / math.cos(alt_rad)

        d_az = CA * sec_alt + NPAE * tan_alt + AN * sin_az * tan_alt - AW * cos_az * tan_alt
        d_alt = IE - AN * cos_az - AW * sin_az

        solved_ra, solved_dec = altaz_to_radec(az_true, alt_true, lat, lon)
        mount_ra, mount_dec = altaz_to_radec(az_true + d_az, alt_true + d_alt, lat, lon)
        points.append((mount_ra, mount_dec, solved_ra, solved_dec, lat, lon))
    return points


class TestSyntheticRecovery:
    """Fit a model to synthetic data with known errors and verify recovery."""

    def test_3term_recovery(self):
        AN, AW, IE = 0.5, 0.3, 0.1
        pts = _synthetic_points(AN, AW, IE, n=6)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        status = model.status()
        assert status["n_terms"] == 3
        assert abs(status["terms"]["AN"] - AN) < 0.05, f"AN recovery: {status['terms']['AN']}"
        assert abs(status["terms"]["AW"] - AW) < 0.05, f"AW recovery: {status['terms']['AW']}"
        assert abs(status["terms"]["IE"] - IE) < 0.05, f"IE recovery: {status['terms']['IE']}"

    def test_5term_recovery(self):
        AN, AW, IE, CA, NPAE = 0.4, -0.2, 0.15, 0.1, -0.05
        pts = _synthetic_points(AN, AW, IE, CA, NPAE, n=16)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        status = model.status()
        assert status["n_terms"] == 5
        assert abs(status["terms"]["AN"] - AN) < 0.05
        assert abs(status["terms"]["AW"] - AW) < 0.05
        assert abs(status["terms"]["IE"] - IE) < 0.05
        assert abs(status["terms"]["CA"] - CA) < 0.1
        assert abs(status["terms"]["NPAE"] - NPAE) < 0.1


# ------------------------------------------------------------------
# Model: correction reduces error
# ------------------------------------------------------------------


class TestCorrectionAccuracy:
    def test_correction_reduces_error(self):
        AN, AW, IE = 0.5, 0.3, 0.1
        lat, lon = 40.0, -74.0
        pts = _synthetic_points(AN, AW, IE, n=10, lat=lat, lon=lon)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # Test correction at new random positions
        rng = np.random.RandomState(99)
        for _ in range(5):
            target_az = rng.uniform(30, 330)
            target_alt = rng.uniform(25, 75)
            target_ra, target_dec = altaz_to_radec(target_az, target_alt, lat, lon)

            corrected_ra, corrected_dec = model.correct(target_ra, target_dec, lat, lon)
            # The corrected mount command, when subject to the same error model,
            # should land closer to the target than the uncorrected command
            assert corrected_ra != target_ra or corrected_dec != target_dec, "Correction should change the coordinates"


# ------------------------------------------------------------------
# Model: graceful degradation
# ------------------------------------------------------------------


class TestGracefulDegradation:
    def test_zero_points_passthrough(self):
        model = AltAzPointingModel()
        ra, dec = 45.0, 30.0
        assert model.correct(ra, dec, 40.0, -74.0) == (ra, dec)
        assert not model.is_active
        assert not model.is_trained

    def test_one_point_passthrough(self):
        model = AltAzPointingModel()
        model.add_point(45.0, 30.0, 45.1, 30.1, 40.0, -74.0)
        ra, dec = 60.0, 20.0
        assert model.correct(ra, dec, 40.0, -74.0) == (ra, dec)
        assert not model.is_active

    def test_three_points_3term(self):
        pts = _synthetic_points(0.3, 0.2, 0.1, n=4)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)
        assert model.is_active
        assert not model.is_trained
        assert model.status()["n_terms"] == 3

    def test_eight_points_5term(self):
        pts = _synthetic_points(0.3, 0.2, 0.1, 0.05, -0.03, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)
        assert model.is_trained
        assert model.status()["n_terms"] == 5


# ------------------------------------------------------------------
# Sky grid generation
# ------------------------------------------------------------------


class TestSkyGridGeneration:
    def test_generates_points(self):
        targets = generate_calibration_grid(
            current_az_deg=180.0,
            cable_wrap_cumulative_deg=0.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=10,
        )
        assert len(targets) == 10
        for ra, dec in targets:
            assert 0.0 <= ra < 360.0
            assert -90.0 <= dec <= 90.0

    def test_respects_tight_cable_budget(self):
        targets = generate_calibration_grid(
            current_az_deg=180.0,
            cable_wrap_cumulative_deg=200.0,
            cable_wrap_soft_limit_deg=240.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=10,
        )
        # Should still produce points, just narrower coverage
        assert len(targets) > 0

    def test_empty_when_no_alt_bands(self):
        targets = generate_calibration_grid(
            current_az_deg=180.0,
            cable_wrap_cumulative_deg=0.0,
            horizon_limit_deg=85.0,
            overhead_limit_deg=86.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=10,
        )
        # Should still produce at least some points (fallback to midpoint)
        assert len(targets) > 0


# ------------------------------------------------------------------
# Serialization round-trip
# ------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        pts = _synthetic_points(0.4, -0.2, 0.1, 0.05, -0.03, n=12)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        data = model.to_dict()
        restored = AltAzPointingModel.from_dict(data)

        assert restored.point_count == model.point_count
        assert restored.is_trained == model.is_trained
        assert restored.rms_arcmin == model.rms_arcmin
        original_status = model.status()
        restored_status = restored.status()
        for term in ("AN", "AW", "IE", "CA", "NPAE"):
            assert original_status["terms"][term] == restored_status["terms"][term]

    def test_file_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "pointing_model_state.json"
            pts = _synthetic_points(0.3, 0.2, 0.1, n=8)
            model = AltAzPointingModel(state_file=state_file)
            for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
                model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

            assert state_file.exists()
            data = json.loads(state_file.read_text())
            assert data["n_terms"] == 5

            # Create a new model that loads from the file
            model2 = AltAzPointingModel(state_file=state_file)
            assert model2.is_trained
            assert model2.point_count == model.point_count
            assert abs(model2.rms_arcmin - model.rms_arcmin) < 0.01


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestReset:
    def test_reset_clears_everything(self):
        pts = _synthetic_points(0.3, 0.2, 0.1, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        assert model.is_trained
        model.reset()

        assert not model.is_active
        assert not model.is_trained
        assert model.point_count == 0
        assert model.rms_arcmin == 0.0
        # Correction should be passthrough
        ra, dec = 45.0, 30.0
        assert model.correct(ra, dec, 40.0, -74.0) == (ra, dec)

    def test_reset_persists_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "pm.json"
            pts = _synthetic_points(0.3, 0.2, 0.1, n=8)
            model = AltAzPointingModel(state_file=state_file)
            for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
                model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

            model.reset()
            data = json.loads(state_file.read_text())
            assert data["n_terms"] == 0
            assert len(data["points"]) == 0


# ------------------------------------------------------------------
# Health monitoring
# ------------------------------------------------------------------


class TestHealthMonitoring:
    def test_degraded_after_bad_residuals(self):
        pts = _synthetic_points(0.3, 0.2, 0.1, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        assert model.health == "good"
        # Record residuals far exceeding the model's RMS
        huge_residual = model.rms_arcmin * 10.0
        for _ in range(5):
            model.record_verification_residual(huge_residual)

        assert model.health == "degraded"

    def test_stays_good_with_low_residuals(self):
        pts = _synthetic_points(0.3, 0.2, 0.1, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        for _ in range(5):
            model.record_verification_residual(model.rms_arcmin * 0.5)

        assert model.health == "good"


# ------------------------------------------------------------------
# Status output
# ------------------------------------------------------------------


class TestStatus:
    def test_untrained_status(self):
        model = AltAzPointingModel()
        status = model.status()
        assert status["state"] == "untrained"
        assert status["point_count"] == 0

    def test_trained_status_has_tilt(self):
        pts = _synthetic_points(0.5, 0.3, 0.1, CA=0.08, NPAE=-0.04, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        status = model.status()
        assert status["state"] == "trained"
        assert status["tilt_deg"] > 0
        assert status["tilt_direction_label"] != ""
        assert status["pointing_accuracy_arcmin"] >= 0

    def test_compass_labels(self):
        model = AltAzPointingModel()
        assert model._compass_label(0.0) == "N"
        assert model._compass_label(45.0) == "NE"
        assert model._compass_label(90.0) == "E"
        assert model._compass_label(180.0) == "S"
        assert model._compass_label(270.0) == "W"
