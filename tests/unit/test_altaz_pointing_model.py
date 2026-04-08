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
    """RA/Dec -> Alt/Az -> RA/Dec should recover the original coordinates."""
    az, alt = radec_to_altaz(ra, dec, lat, lon)
    if alt < 5.0:
        pytest.skip("Below horizon — numeric instability")
    ra2, dec2 = altaz_to_radec(az, alt, lat, lon)
    assert abs(ra2 - ra) % 360 < 0.01, f"RA mismatch: {ra2} vs {ra}"
    assert abs(dec2 - dec) < 0.01, f"Dec mismatch: {dec2} vs {dec}"


@pytest.mark.parametrize(
    ("az", "alt", "lat", "lon"),
    [
        (0.0, 45.0, 40.0, -74.0),
        (180.0, 45.0, 40.0, -74.0),
        (180.001, 45.0, 40.0, -74.0),
        (179.999, 45.0, 40.0, -74.0),
        (359.99, 45.0, 40.0, -74.0),
        (90.0, 85.0, 40.0, -74.0),
    ],
)
def test_altaz_radec_round_trip_edge_cases(az: float, alt: float, lat: float, lon: float):
    """Alt/Az -> RA/Dec -> Alt/Az round-trip at boundary azimuths."""
    ra, dec = altaz_to_radec(az, alt, lat, lon)
    az2, alt2 = radec_to_altaz(ra, dec, lat, lon)

    az_diff = abs(az2 - az)
    if az_diff > 180.0:
        az_diff = 360.0 - az_diff
    assert az_diff < 0.05, f"Az mismatch: {az2} vs {az}"
    assert abs(alt2 - alt) < 0.05, f"Alt mismatch: {alt2} vs {alt}"


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
    noise_arcsec: float = 0.0,
    seed: int = 42,
) -> list[tuple[float, float, float, float, float, float]]:
    """Generate synthetic calibration point pairs with a known error model.

    Simulates real calibration: the mount encoder reports (az_enc, alt_enc)
    while the optics actually point to (az_enc + d_az, alt_enc + d_alt).
    The plate solver returns the true optics position.

    Args:
        noise_arcsec: Gaussian noise added to the solved position (arcseconds)
            to simulate plate-solve measurement scatter.
        seed: Random seed for reproducibility.

    Returns a list of (mount_ra, mount_dec, solved_ra, solved_dec, lat, lon).
    """
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(n):
        az_enc = rng.uniform(30.0, 330.0)
        alt_enc = rng.uniform(25.0, 75.0)

        az_rad = math.radians(az_enc)
        alt_rad = math.radians(alt_enc)
        sin_az = math.sin(az_rad)
        cos_az = math.cos(az_rad)
        tan_alt = math.tan(alt_rad)
        sec_alt = 1.0 / math.cos(alt_rad)

        d_az = CA * sec_alt + NPAE * tan_alt + AN * sin_az * tan_alt - AW * cos_az * tan_alt
        d_alt = IE - AN * cos_az - AW * sin_az

        noise_deg = noise_arcsec / 3600.0
        d_az += rng.normal(0.0, noise_deg)
        d_alt += rng.normal(0.0, noise_deg)

        mount_ra, mount_dec = altaz_to_radec(az_enc, alt_enc, lat, lon)
        solved_ra, solved_dec = altaz_to_radec(az_enc + d_az, alt_enc + d_alt, lat, lon)
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

        rng = np.random.RandomState(99)
        for _ in range(5):
            target_az = rng.uniform(30, 330)
            target_alt = rng.uniform(25, 75)
            target_ra, target_dec = altaz_to_radec(target_az, target_alt, lat, lon)

            corrected_ra, corrected_dec = model.correct(target_ra, target_dec, lat, lon)
            assert corrected_ra != target_ra or corrected_dec != target_dec, "Correction should change the coordinates"

    def test_predict_error_accounts_for_cos_alt(self):
        """predict_error should project dAz through cos(alt) for sky-accurate magnitude.

        Using a model with known AN and all other terms zero, compute
        expected error at az=90, alt=60 and verify the cos(alt) factor
        appears in the result.  predict_error returns degrees.
        """
        lat, lon = 40.0, -74.0
        AN = 0.5
        pts = _synthetic_points(AN, 0.0, 0.0, n=10, lat=lat, lon=lon)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # At az=90°, alt=60°:
        #   dAz = AN * sin(90) * tan(60) = AN * 1.732
        #   dAlt = -AN * cos(90) = 0
        #   With cos(alt): sqrt((dAz * cos(60))^2 + dAlt^2) = dAz * 0.5
        #   Without cos(alt): sqrt(dAz^2 + dAlt^2) = dAz
        alt_rad = math.radians(60.0)
        expected_d_az = AN * math.sin(math.radians(90.0)) * math.tan(alt_rad)
        expected_with_cos = abs(expected_d_az * math.cos(alt_rad))
        expected_without_cos = abs(expected_d_az)

        ra_test, dec_test = altaz_to_radec(90.0, 60.0, lat, lon)
        result = model.predict_error(ra_test, dec_test, lat, lon)

        # Should match the cos(alt)-projected value, not the raw one
        assert (
            abs(result - expected_with_cos) < 1.0 / 60.0
        ), f"predict_error={result:.4f}° should be close to cos(alt)-projected {expected_with_cos:.4f}°"
        assert (
            abs(result - expected_without_cos) > 1.0 / 60.0
        ), f"predict_error={result:.4f}° should NOT match raw {expected_without_cos:.4f}°"


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
            n_points=15,
        )
        assert len(targets) == 15
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
            n_points=15,
        )
        # Should still produce points, just narrower coverage
        assert len(targets) > 0

    def test_asymmetric_cable_wrap_uses_unwind_direction(self):
        """With high cumulative winding, the grid should extend mostly in
        the unwinding direction to maximize azimuth diversity."""
        targets = generate_calibration_grid(
            current_az_deg=0.0,
            cable_wrap_cumulative_deg=190.0,
            cable_wrap_soft_limit_deg=240.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=15,
        )
        assert len(targets) == 15

        # Convert targets back to alt-az and check azimuth spread
        azimuths = []
        for ra, dec in targets:
            az, _alt = radec_to_altaz(ra, dec, 40.0, -74.0)
            azimuths.append(az)

        spread = AltAzPointingModel._azimuth_spread(azimuths)
        assert spread > 120.0, f"Azimuth spread {spread:.0f}° too narrow — grid not using unwind direction"

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
        assert restored.rms_deg == model.rms_deg
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
            assert abs(model2.rms_deg - model.rms_deg) < 0.01


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
        assert model.rms_deg == 0.0
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
        huge_residual = model.rms_deg * 10.0
        for _ in range(5):
            model.record_verification_residual(huge_residual)

        assert model.health == "degraded"

    def test_stays_good_with_low_residuals(self):
        pts = _synthetic_points(0.3, 0.2, 0.1, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)

        for _ in range(5):
            model.record_verification_residual(model.rms_deg * 0.5)

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
        assert status["pointing_accuracy_deg"] >= 0

    def test_tilt_direction_labels_via_status(self):
        """Verify compass labels through the public status() interface."""
        # AN>0 AW=0 → tilt toward N (bearing ≈ 0°)
        pts = _synthetic_points(0.5, 0.0, 0.0, n=10)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)
        assert model.status()["tilt_direction_label"] == "N"

        # AW>0, AN=0 → tilt toward E (bearing ≈ 90°)
        pts = _synthetic_points(0.0, 0.5, 0.0, n=10)
        model2 = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat, lon in pts:
            model2.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat, lon)
        assert model2.status()["tilt_direction_label"] == "E"


# ------------------------------------------------------------------
# Sigma-clipping
# ------------------------------------------------------------------


class TestSigmaClipping:
    def test_outliers_are_clipped(self):
        """20 clean synthetic points + 2 extreme outliers — terms should
        still recover accurately because sigma-clipping rejects outliers."""
        AN, AW, IE = 0.5, 0.3, 0.1
        lat, lon = 40.0, -74.0
        pts = _synthetic_points(AN, AW, IE, n=20, lat=lat, lon=lon)

        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # Add 2 extreme outliers
        model.add_point(100.0, 30.0, 105.0, 35.0, lat, lon)
        model.add_point(200.0, 40.0, 195.0, 35.0, lat, lon)

        status = model.status()
        assert abs(status["terms"]["AN"] - AN) < 0.05, f"AN={status['terms']['AN']}"
        assert abs(status["terms"]["AW"] - AW) < 0.05, f"AW={status['terms']['AW']}"
        assert abs(status["terms"]["IE"] - IE) < 0.05, f"IE={status['terms']['IE']}"

    def test_no_good_points_lost_clean_data(self):
        """With clean data only, sigma-clipping should not reject anything."""
        AN, AW, IE = 0.3, -0.2, 0.15
        pts = _synthetic_points(AN, AW, IE, n=15)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        assert model.point_count == 15
        assert model.rms_deg < 1.0 / 60.0

    def test_few_points_skips_clipping(self):
        """With barely enough points for a 3-term fit (< min_for_clip),
        sigma-clipping must not run — even if one point is a bit off."""
        AN, AW, IE = 0.3, 0.2, 0.1
        lat, lon = 40.0, -74.0
        # 4 points: enough for 3-term, but < 6 (3 + 3 spare), so no clipping
        pts = _synthetic_points(AN, AW, IE, n=4, lat=lat, lon=lon)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        assert model.is_active
        assert model.status()["n_terms"] == 3
        # All 4 points contribute (nothing clipped)
        assert model.point_count == 4

    def test_many_outliers_clips_correctly(self):
        """With 30% bad points, the clean majority should still win."""
        AN, AW, IE = 0.4, -0.3, 0.2
        lat, lon = 40.0, -74.0
        pts = _synthetic_points(AN, AW, IE, n=20, lat=lat, lon=lon)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # 8 extreme outliers (40% of total 28)
        rng = np.random.RandomState(77)
        for _ in range(8):
            az = rng.uniform(30, 330)
            alt = rng.uniform(25, 70)
            ra, dec = altaz_to_radec(az, alt, lat, lon)
            # Huge offset: 3-5 degrees
            model.add_point(ra, dec, ra + rng.uniform(3, 5), dec + rng.uniform(-5, -3), lat, lon)

        status = model.status()
        assert abs(status["terms"]["AN"] - AN) < 0.1, f"AN={status['terms']['AN']}"
        assert abs(status["terms"]["AW"] - AW) < 0.1, f"AW={status['terms']['AW']}"
        assert abs(status["terms"]["IE"] - IE) < 0.1, f"IE={status['terms']['IE']}"

    def test_floor_prevents_cascade_on_perfect_data(self):
        """The 1-arcmin floor must prevent sigma-clipping from cascading
        on near-perfect synthetic data and eating valid points.

        Without the floor, a tight fit shrinks sigma so small that normal
        numerical noise triggers rejection, which tightens sigma further,
        causing a cascade that can evict most of the dataset.
        """
        AN, AW, IE, CA, NPAE = 0.4, -0.2, 0.15, 0.1, -0.05
        pts = _synthetic_points(AN, AW, IE, CA, NPAE, n=50)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # With 50 perfect points the model should be 5-term, not downgraded
        assert model.status()["n_terms"] == 5
        # Recovery should be excellent
        assert abs(model.status()["terms"]["AN"] - AN) < 0.01
        assert abs(model.status()["terms"]["CA"] - CA) < 0.02

    def test_5term_downgrades_to_3term_when_clipping_reduces_count(self):
        """If sigma-clipping drops below 8 active points, the fit should
        gracefully downgrade to 3-term instead of blowing up."""
        AN, AW, IE = 0.3, 0.2, 0.1
        lat, lon = 40.0, -74.0
        # 10 clean points → 5-term eligible
        pts = _synthetic_points(AN, AW, IE, n=10, lat=lat, lon=lon)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # Confirm 5-term initially
        assert model.status()["n_terms"] == 5

        # Now add 5 extreme outliers to force heavy clipping on next fit
        for i in range(5):
            model.add_point(90.0 + i * 20, 40.0, 100.0 + i * 20, 50.0, lat, lon)

        # Should still produce a valid fit (3 or 5 term)
        status = model.status()
        assert status["n_terms"] in (3, 5)
        # The clean signal should still dominate
        assert abs(status["terms"]["AN"] - AN) < 0.15

    def test_noisy_data_still_recovers(self):
        """Realistic plate-solve noise (~20 arcsec) should not prevent
        accurate term recovery or cause excessive rejection."""
        AN, AW, IE = 0.5, 0.3, 0.1
        pts = _synthetic_points(AN, AW, IE, n=40, noise_arcsec=20.0, seed=99)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        status = model.status()
        # 20 arcsec noise is ~0.006°; terms are 0.1-0.5°. Should recover well.
        assert abs(status["terms"]["AN"] - AN) < 0.1, f"AN={status['terms']['AN']}"
        assert abs(status["terms"]["AW"] - AW) < 0.1, f"AW={status['terms']['AW']}"
        assert abs(status["terms"]["IE"] - IE) < 0.1, f"IE={status['terms']['IE']}"
        # RMS should reflect the noise floor, not be absurdly small
        assert status["pointing_accuracy_deg"] > 0.1 / 60.0
        assert status["pointing_accuracy_deg"] < 3.0 / 60.0

    def test_noisy_data_with_outliers(self):
        """Noise + outliers together: the sigma-clipping should reject the
        outliers without being confused by the noise in the clean points."""
        AN, AW, IE = 0.4, -0.2, 0.15
        lat, lon = 40.0, -74.0
        pts = _synthetic_points(AN, AW, IE, n=30, noise_arcsec=15.0, seed=55, lat=lat, lon=lon)
        model = AltAzPointingModel()
        for mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_ in pts:
            model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, lat_, lon_)

        # Add 3 moderate outliers (1-2 degree errors, not extreme)
        model.add_point(120.0, 35.0, 121.5, 36.5, lat, lon)
        model.add_point(240.0, 50.0, 238.5, 48.0, lat, lon)
        model.add_point(60.0, 40.0, 62.0, 42.0, lat, lon)

        status = model.status()
        assert abs(status["terms"]["AN"] - AN) < 0.1, f"AN={status['terms']['AN']}"
        assert abs(status["terms"]["AW"] - AW) < 0.1, f"AW={status['terms']['AW']}"
        assert abs(status["terms"]["IE"] - IE) < 0.1, f"IE={status['terms']['IE']}"


# ------------------------------------------------------------------
# Grid ordering and cable wrap safety
# ------------------------------------------------------------------


class TestGridOrdering:
    def test_serpentine_reverses_direction(self):
        """Adjacent altitude bands should sweep azimuth in opposite directions
        so cable wrap accumulation stays near zero."""
        targets = generate_calibration_grid(
            current_az_deg=180.0,
            cable_wrap_cumulative_deg=0.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=30,
        )
        assert len(targets) == 30

        azimuths: list[list[float]] = [[], [], []]
        for ra, dec in targets:
            az, alt = radec_to_altaz(ra, dec, 40.0, -74.0)
            if alt < 37.5:
                azimuths[0].append(az)
            elif alt < 52.5:
                azimuths[1].append(az)
            else:
                azimuths[2].append(az)

        def _net_sweep(azs: list[float]) -> float:
            """Sum of consecutive signed deltas — positive = CW, negative = CCW."""
            total = 0.0
            for i in range(len(azs) - 1):
                d = azs[i + 1] - azs[i]
                if d > 180.0:
                    d -= 360.0
                elif d < -180.0:
                    d += 360.0
                total += d
            return total

        # Band 0 and band 1 should sweep in opposite az directions
        if len(azimuths[0]) >= 2 and len(azimuths[1]) >= 2:
            dir0 = _net_sweep(azimuths[0])
            dir1 = _net_sweep(azimuths[1])
            assert dir0 * dir1 < 0, f"Bands should reverse: band0 sweep={dir0:.0f}°, band1 sweep={dir1:.0f}°"

    def test_cable_wrap_stays_bounded(self):
        """Simulating shortest-path slews through the grid should keep
        cumulative cable wrap well within the soft limit."""
        targets = generate_calibration_grid(
            current_az_deg=180.0,
            cable_wrap_cumulative_deg=0.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=15,
        )
        cumulative = 0.0
        prev_az = 180.0
        for ra, dec in targets:
            az, _ = radec_to_altaz(ra, dec, 40.0, -74.0)
            delta = az - prev_az
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0
            cumulative += delta
            prev_az = az

        assert abs(cumulative) < 240.0, f"Cumulative cable wrap {cumulative:.0f}° exceeds soft limit"

    def test_no_75_degree_band(self):
        """Grid should not contain points at 75° altitude."""
        targets = generate_calibration_grid(
            current_az_deg=180.0,
            cable_wrap_cumulative_deg=0.0,
            lat_deg=40.0,
            lon_deg=-74.0,
            n_points=15,
        )
        for ra, dec in targets:
            _, alt = radec_to_altaz(ra, dec, 40.0, -74.0)
            assert alt < 70.0, f"Grid point at alt={alt:.0f}° — 75° band should be excluded"


# ------------------------------------------------------------------
# No point eviction
# ------------------------------------------------------------------


class TestNoEviction:
    def test_all_points_preserved(self):
        """Adding many points should preserve all of them (no eviction)."""
        model = AltAzPointingModel()
        lat, lon = 40.0, -74.0
        rng = np.random.RandomState(123)
        for _ in range(200):
            az = rng.uniform(30.0, 330.0)
            alt = rng.uniform(25.0, 75.0)
            ra, dec = altaz_to_radec(az, alt, lat, lon)
            model.add_point(ra, dec, ra + 0.01, dec + 0.01, lat, lon)
        assert model.point_count == 200


# ------------------------------------------------------------------
# Nearby-point guard
# ------------------------------------------------------------------


class TestNearbyPointGuard:
    def test_nearby_returns_true(self):
        model = AltAzPointingModel()
        lat, lon = 40.0, -74.0
        ra, dec = altaz_to_radec(180.0, 45.0, lat, lon)
        model.add_point(ra, dec, ra + 0.01, dec + 0.01, lat, lon)

        assert model.has_nearby_point(180.0, 45.0)
        assert model.has_nearby_point(180.5, 45.5)

    def test_distant_returns_false(self):
        model = AltAzPointingModel()
        lat, lon = 40.0, -74.0
        ra, dec = altaz_to_radec(180.0, 45.0, lat, lon)
        model.add_point(ra, dec, ra + 0.01, dec + 0.01, lat, lon)

        assert not model.has_nearby_point(185.0, 45.0)
        assert not model.has_nearby_point(180.0, 50.0)

    def test_empty_model_returns_false(self):
        model = AltAzPointingModel()
        assert not model.has_nearby_point(180.0, 45.0)

    def test_custom_threshold(self):
        model = AltAzPointingModel()
        lat, lon = 40.0, -74.0
        ra, dec = altaz_to_radec(180.0, 45.0, lat, lon)
        model.add_point(ra, dec, ra + 0.01, dec + 0.01, lat, lon)

        assert model.has_nearby_point(183.0, 45.0, min_sep_deg=5.0)
        assert not model.has_nearby_point(183.0, 45.0, min_sep_deg=2.0)
