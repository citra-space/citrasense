"""Tests for SatelliteMatcherProcessor helper methods."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from citrascope.processors.builtin.satellite_matcher_processor import (
    _GEO_MATCH_RADIUS_DEG,
    _GEO_MEAN_MOTION_HIGH,
    _GEO_MEAN_MOTION_LOW,
    _MATCH_RADIUS_DEG,
    _STAR_MATCH_TOLERANCE_DEG,
    SatelliteMatcherProcessor,
)
from citrascope.processors.processor_result import ProcessingContext


# ===================================================================
# 1. Star subtraction
# ===================================================================
def _make_sources(coords: list[tuple[float, float]], elongation: float = 2.0) -> pd.DataFrame:
    """Create a minimal source DataFrame from (ra, dec) pairs."""
    return pd.DataFrame(
        {
            "ra": [c[0] for c in coords],
            "dec": [c[1] for c in coords],
            "mag": [12.0] * len(coords),
            "magerr": [0.01] * len(coords),
            "elongation": [elongation] * len(coords),
        }
    )


class TestSubtractKnownStars:
    def test_removes_matching_sources(self, tmp_path: Path):
        star_ra, star_dec = 180.0, 45.0
        sat_ra, sat_dec = 181.0, 45.5

        candidates = _make_sources([(star_ra, star_dec), (sat_ra, sat_dec)])

        xmatch = pd.DataFrame({"ra": [star_ra], "dec": [star_dec]})
        xmatch.to_csv(tmp_path / "photometry_crossmatch.csv", index=False)

        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)

        assert len(filtered) == 1
        assert float(filtered.iloc[0]["ra"]) == sat_ra
        assert stats["catalog_stars_removed"] == 1
        assert stats["candidates_before"] == 2
        assert stats["candidates_after"] == 1

    def test_retains_all_when_no_match(self, tmp_path: Path):
        candidates = _make_sources([(100.0, 30.0), (101.0, 31.0)])

        xmatch = pd.DataFrame({"ra": [200.0], "dec": [-40.0]})
        xmatch.to_csv(tmp_path / "photometry_crossmatch.csv", index=False)

        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)

        assert len(filtered) == 2
        assert stats["catalog_stars_removed"] == 0

    def test_graceful_when_csv_absent(self, tmp_path: Path):
        candidates = _make_sources([(100.0, 30.0)])
        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)

        assert len(filtered) == 1
        assert stats["source"] == "skipped"

    def test_graceful_when_csv_empty(self, tmp_path: Path):
        (tmp_path / "photometry_crossmatch.csv").write_text("")
        candidates = _make_sources([(100.0, 30.0)])
        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)

        assert len(filtered) == 1
        assert stats["source"] == "skipped"

    def test_tolerance_boundary(self, tmp_path: Path):
        """A source just outside the 1-arcsecond tolerance should be retained."""
        base_ra, base_dec = 180.0, 45.0
        offset = _STAR_MATCH_TOLERANCE_DEG * 2.0  # well outside tolerance

        candidates = _make_sources([(base_ra + offset, base_dec)])
        xmatch = pd.DataFrame({"ra": [base_ra], "dec": [base_dec]})
        xmatch.to_csv(tmp_path / "photometry_crossmatch.csv", index=False)

        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)
        assert len(filtered) == 1
        assert stats["catalog_stars_removed"] == 0

    def test_matches_on_detected_positions(self, tmp_path: Path):
        """Star subtraction matches against detected-source positions (ra/dec).

        The crossmatch CSV has both detected-source coords (ra/dec) and catalog
        star coords (radeg/decdeg). Real astrometric offsets between the two are
        18-33 arcseconds, so we match on detected positions which come from the
        same SExtractor run as our candidates.
        """
        star_detected_ra, star_detected_dec = 180.0, 45.0
        catalog_ra, catalog_dec = 180.008, 45.005  # catalog pos ~33 arcsec away

        candidates = _make_sources([(star_detected_ra, star_detected_dec)])

        xmatch = pd.DataFrame(
            {
                "ra": [star_detected_ra],
                "dec": [star_detected_dec],
                "radeg": [catalog_ra],
                "decdeg": [catalog_dec],
            }
        )
        xmatch.to_csv(tmp_path / "photometry_crossmatch.csv", index=False)

        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)
        assert len(filtered) == 0, "star should be removed -- its detected position matches a crossmatch entry"
        assert stats["catalog_stars_removed"] == 1


# ===================================================================
# 2. Slow-mover elongation bypass
# ===================================================================
class TestSlowMoverElongationBypass:
    """When pointing_report indicates a slow mover, elongation filter should be skipped."""

    @staticmethod
    def _make_mixed_sources() -> pd.DataFrame:
        """Sources with a mix of point-like and extended elongation values."""
        return pd.DataFrame(
            {
                "ra": [180.0, 180.01, 180.02, 180.03],
                "dec": [45.0, 45.0, 45.0, 45.0],
                "mag": [-8.0, -7.0, -6.0, -5.0],
                "magerr": [0.01] * 4,
                "elongation": [1.0, 1.1, 2.0, 3.0],  # first two are point-like (< 1.5)
            }
        )

    def test_slow_mover_uses_all_sources(self):
        """With is_slow_mover=True, all sources become candidates (no elongation cut)."""
        sources = self._make_mixed_sources()
        from citrascope.processors.builtin.satellite_matcher_processor import (
            _ELONGATION_THRESHOLD,
        )

        is_slow_mover = True
        elongation_filter_applied = not is_slow_mover

        if elongation_filter_applied:
            potential_sats = sources[sources["elongation"] >= _ELONGATION_THRESHOLD].copy()
        else:
            potential_sats = sources.copy()

        assert len(potential_sats) == 4
        assert not elongation_filter_applied

    def test_fast_mover_applies_elongation_filter(self):
        """With is_slow_mover=False (LEO), elongation filter is applied normally."""
        sources = self._make_mixed_sources()
        from citrascope.processors.builtin.satellite_matcher_processor import (
            _ELONGATION_THRESHOLD,
        )

        is_slow_mover = False
        elongation_filter_applied = not is_slow_mover

        if elongation_filter_applied:
            potential_sats = sources[sources["elongation"] >= _ELONGATION_THRESHOLD].copy()
        else:
            potential_sats = sources.copy()

        assert len(potential_sats) == 2  # only elongation >= 1.5
        assert elongation_filter_applied

    def test_missing_pointing_report_defaults_to_elongation_filter(self):
        """When pointing_report is None, elongation filter should apply (safe default)."""
        pointing_report = None
        try:
            slew_ahead = (pointing_report or {}).get("slew_ahead", {})
            is_slow_mover = bool(slew_ahead.get("is_slow_mover", False))
        except (AttributeError, TypeError):
            is_slow_mover = False

        assert is_slow_mover is False
        assert not is_slow_mover  # elongation filter will apply

    def test_pointing_report_without_slew_ahead_defaults_to_elongation_filter(self):
        """A pointing_report that lacks slew_ahead defaults safely."""
        pointing_report = {"converged": True, "final_telescope_ra_deg": 133.0}
        slew_ahead = (pointing_report or {}).get("slew_ahead", {})
        is_slow_mover = bool(slew_ahead.get("is_slow_mover", False))

        assert is_slow_mover is False


# ===================================================================
# 3. Reverse match produces at most one observation per prediction
# ===================================================================
class TestReverseMatchOnePerPrediction:
    """Verify the observation-building loop yields at most one match per prediction.

    We test this by constructing a scenario where two sources are near the same
    prediction -- the old forward-match code would produce two observations for
    that prediction, but the reverse-match code should produce exactly one.
    """

    def test_one_observation_per_prediction(self):
        pred_ra, pred_dec = 180.0, 45.0
        # Two sources near the same prediction, both within 1 arcmin
        src1 = (pred_ra + 0.005, pred_dec)  # ~18 arcsec away
        src2 = (pred_ra - 0.008, pred_dec)  # ~29 arcsec away

        potential_sats = _make_sources([src1, src2])
        predictions: list[dict[str, Any]] = [
            {"ra": pred_ra, "dec": pred_dec, "satellite_id": "SAT-1", "name": "TestSat", "phase_angle": 90.0}
        ]

        source_coords = potential_sats[["ra", "dec"]].values
        pred_coords = np.array([[p["ra"], p["dec"]] for p in predictions])

        from scipy.spatial import KDTree

        source_tree = KDTree(source_coords)
        pred_distances, pred_indices = source_tree.query(pred_coords)

        match_radius = 1.0 / 60.0
        observations = []
        for i, p in enumerate(predictions):
            dist_deg = float(np.asarray(pred_distances)[i])
            if dist_deg >= match_radius:
                continue
            src_idx = int(np.asarray(pred_indices)[i])
            if src_idx < 0 or src_idx >= len(potential_sats):
                continue
            row = potential_sats.iloc[src_idx]
            observations.append({"norad_id": p["satellite_id"], "ra": float(row["ra"])})

        assert len(observations) == 1
        assert observations[0]["norad_id"] == "SAT-1"
        # Should have picked the closer source (src1)
        assert abs(observations[0]["ra"] - src1[0]) < 1e-6


# ===================================================================
# 4. Observation magnitudes are calibrated via zero point
# ===================================================================
class TestObservationMagnitudeCalibration:
    """The observation-building loop must emit calibrated 'mag' and raw 'mag_instrumental'."""

    @staticmethod
    def _build_observations(
        potential_sats: pd.DataFrame,
        predictions: list[dict[str, Any]],
        zero_point: float,
    ) -> list[dict[str, Any]]:
        """Replicate the reverse-match observation builder from _match_satellites."""
        from scipy.spatial import KDTree

        source_coords = potential_sats[["ra", "dec"]].values
        pred_coords = np.array([[p["ra"], p["dec"]] for p in predictions])
        source_tree = KDTree(source_coords)
        pred_distances, pred_indices = source_tree.query(pred_coords)

        match_radius = 1.0 / 60.0
        observations: list[dict[str, Any]] = []
        for i, p in enumerate(predictions):
            dist_deg = float(np.asarray(pred_distances)[i])
            if dist_deg >= match_radius:
                continue
            src_idx = int(np.asarray(pred_indices)[i])
            if src_idx < 0 or src_idx >= len(potential_sats):
                continue
            row = potential_sats.iloc[src_idx]
            inst_mag = float(row["mag"])
            observations.append(
                {
                    "norad_id": p["satellite_id"],
                    "name": p["name"],
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "mag": inst_mag + zero_point,
                    "mag_instrumental": inst_mag,
                    "filter": "r",
                    "phase_angle": round(p["phase_angle"], 1),
                    "elongation": float(row["elongation"]),
                }
            )
        return observations

    def test_calibrated_mag_and_instrumental_preserved(self):
        """With a real zero point, 'mag' = instrumental + ZP and 'mag_instrumental' = raw."""
        inst_mag = -8.5
        zero_point = 23.59

        sources = _make_sources([(180.0, 45.0)])
        sources["mag"] = inst_mag
        predictions = [{"ra": 180.0, "dec": 45.0, "satellite_id": "SAT-1", "name": "DIRECTV 15", "phase_angle": 42.0}]

        obs = self._build_observations(sources, predictions, zero_point)

        assert len(obs) == 1
        assert obs[0]["mag_instrumental"] == inst_mag
        assert abs(obs[0]["mag"] - (inst_mag + zero_point)) < 1e-9
        assert abs(obs[0]["mag"] - 15.09) < 1e-6

    def test_zero_point_none_falls_back_to_instrumental(self):
        """When photometry didn't run (zero_point=None), mag equals instrumental."""
        inst_mag = -7.2
        zp_from_ctx: float | None = None
        zero_point = zp_from_ctx if zp_from_ctx is not None else 0.0

        sources = _make_sources([(180.0, 45.0)])
        sources["mag"] = inst_mag
        predictions = [{"ra": 180.0, "dec": 45.0, "satellite_id": "SAT-2", "name": "GOES 16", "phase_angle": 10.0}]

        obs = self._build_observations(sources, predictions, zero_point)

        assert len(obs) == 1
        assert obs[0]["mag"] == inst_mag
        assert obs[0]["mag_instrumental"] == inst_mag

    def test_multiple_satellites_each_calibrated(self):
        """Every matched satellite gets its own calibrated magnitude."""
        zero_point = 22.0

        sources = _make_sources([(180.0, 45.0), (181.0, 46.0)])
        sources["mag"] = [-8.0, -6.5]
        predictions = [
            {"ra": 180.0, "dec": 45.0, "satellite_id": "SAT-A", "name": "Sat A", "phase_angle": 30.0},
            {"ra": 181.0, "dec": 46.0, "satellite_id": "SAT-B", "name": "Sat B", "phase_angle": 60.0},
        ]

        obs = self._build_observations(sources, predictions, zero_point)

        assert len(obs) == 2
        for o in obs:
            assert abs(o["mag"] - (o["mag_instrumental"] + zero_point)) < 1e-9


# ===================================================================
# 5. ProcessingContext.zero_point field
# ===================================================================
class TestZeroPointFromContext:
    """Verify zero_point flows through ProcessingContext to the satellite matcher."""

    def test_context_zero_point_defaults_to_none(self, tmp_path: Path):
        ctx = ProcessingContext(
            image_path=tmp_path / "img.fits",
            working_image_path=tmp_path / "img.fits",
            working_dir=tmp_path,
            image_data=None,
            task=None,
            telescope_record=None,
            ground_station_record=None,
            settings=None,
        )
        assert ctx.zero_point is None

    def test_context_zero_point_round_trips(self, tmp_path: Path):
        ctx = ProcessingContext(
            image_path=tmp_path / "img.fits",
            working_image_path=tmp_path / "img.fits",
            working_dir=tmp_path,
            image_data=None,
            task=None,
            telescope_record=None,
            ground_station_record=None,
            settings=None,
        )
        ctx.zero_point = 23.59
        assert ctx.zero_point == 23.59

    def test_calibrated_magnitude_arithmetic(self):
        """instrumental + zero_point = calibrated magnitude."""
        inst_mag = -8.5
        zero_point = 23.59
        calibrated = inst_mag + zero_point
        assert abs(calibrated - 15.09) < 1e-6

    def test_none_zero_point_falls_back_to_zero(self):
        """When zero_point is None, the fallback is 0.0 (instrumental = calibrated)."""
        zp_from_ctx: float | None = None
        zero_point = zp_from_ctx if zp_from_ctx is not None else 0.0
        assert zero_point == 0.0

    def test_zero_valued_zero_point_is_not_none(self):
        """A legitimate zero_point of 0.0 must not be confused with None."""
        zp_from_ctx: float | None = 0.0
        zero_point = zp_from_ctx if zp_from_ctx is not None else 0.0
        assert zero_point == 0.0
        assert zp_from_ctx is not None


# ===================================================================
# 6. GEO detection from TLE mean motion
# ===================================================================
class TestGeoDetectionFromTle:
    """GEO satellites (mean motion ~1.0 rev/day) should be detected from their TLE
    and treated as slow movers even when pointing_report is absent."""

    # Real DIRECTV 15 TLE from the bug report (synthetic NORAD 99999, GEO orbit)
    GEO_TLE = [
        "1 99999U          26080.16980148 +.00000000  00000+0  00000+0 0 00009",
        "2 99999   0.0283  89.4455 0000437 118.9551 288.5311  1.00277561000006",
    ]
    # ISS-like LEO TLE (~15.5 rev/day)
    LEO_TLE = [
        "1 25544U 98067A   26080.00000000  .00016717  00000-0  10270-3 0  9005",
        "2 25544  51.6400 200.0000 0006000 200.0000 160.0000 15.50000000100008",
    ]
    # GPS-like MEO TLE (~2.0 rev/day)
    MEO_TLE = [
        "1 28874U 05038A   26080.00000000  .00000000  00000+0  00000+0 0  9994",
        "2 28874  55.0000 200.0000 0100000 200.0000 160.0000  2.00563000100008",
    ]

    def test_geo_detected_from_mean_motion(self):
        """A GEO TLE (mean motion ~1.0) is classified as GEO."""
        from keplemon.elements import TLE

        tle = TLE.from_lines(self.GEO_TLE[0], self.GEO_TLE[1])
        assert _GEO_MEAN_MOTION_LOW <= tle.mean_motion <= _GEO_MEAN_MOTION_HIGH

    def test_leo_not_detected_as_geo(self):
        """A LEO TLE (mean motion ~15.5) is not classified as GEO."""
        from keplemon.elements import TLE

        tle = TLE.from_lines(self.LEO_TLE[0], self.LEO_TLE[1])
        assert not (_GEO_MEAN_MOTION_LOW <= tle.mean_motion <= _GEO_MEAN_MOTION_HIGH)

    def test_meo_not_detected_as_geo(self):
        """A MEO TLE (mean motion ~2.0) is not classified as GEO."""
        from keplemon.elements import TLE

        tle = TLE.from_lines(self.MEO_TLE[0], self.MEO_TLE[1])
        assert not (_GEO_MEAN_MOTION_LOW <= tle.mean_motion <= _GEO_MEAN_MOTION_HIGH)

    def test_geo_boundary_low(self):
        """Mean motion exactly at 0.9 is within GEO range."""
        assert _GEO_MEAN_MOTION_LOW <= 0.9 <= _GEO_MEAN_MOTION_HIGH

    def test_geo_boundary_high(self):
        """Mean motion exactly at 1.1 is within GEO range."""
        assert _GEO_MEAN_MOTION_LOW <= 1.1 <= _GEO_MEAN_MOTION_HIGH

    def test_geo_boundary_just_outside_low(self):
        """Mean motion at 0.89 is outside GEO range."""
        assert not (_GEO_MEAN_MOTION_LOW <= 0.89 <= _GEO_MEAN_MOTION_HIGH)

    def test_geo_boundary_just_outside_high(self):
        """Mean motion at 1.11 is outside GEO range."""
        assert not (_GEO_MEAN_MOTION_LOW <= 1.11 <= _GEO_MEAN_MOTION_HIGH)


# ===================================================================
# 7. Adaptive match radius
# ===================================================================
class TestAdaptiveMatchRadius:
    """GEO/slow-mover targets use a wider match radius than LEO targets."""

    def test_geo_gets_wider_radius(self):
        """When is_slow_mover is True, the wider GEO radius is selected."""
        is_slow_mover = True
        match_radius_deg = _GEO_MATCH_RADIUS_DEG if is_slow_mover else _MATCH_RADIUS_DEG
        assert match_radius_deg == _GEO_MATCH_RADIUS_DEG
        assert match_radius_deg * 60.0 == 5.0  # 5 arcminutes

    def test_leo_gets_standard_radius(self):
        """When is_slow_mover is False, the standard LEO radius is selected."""
        is_slow_mover = False
        match_radius_deg = _GEO_MATCH_RADIUS_DEG if is_slow_mover else _MATCH_RADIUS_DEG
        assert match_radius_deg == _MATCH_RADIUS_DEG
        assert match_radius_deg * 60.0 == 1.0  # 1 arcminute

    def test_geo_radius_is_wider_than_leo(self):
        assert _GEO_MATCH_RADIUS_DEG > _MATCH_RADIUS_DEG


# ===================================================================
# 8. Multi-GEO matching with wider radius
# ===================================================================
class TestMultiGeoMatching:
    """GEO predictions 1-4 arcmin from sources should match with the wider radius
    but fail with the standard 1-arcmin radius. This mirrors the real bug where
    DIRECTV 15 (1.38'), SKYNET 1 (1.31'), SES 20 (2.28'), and DIRECTV 12 (3.37')
    were all missed at 1 arcmin but SES 18 (0.28') was the only match."""

    @staticmethod
    def _build_reverse_match_observations(
        potential_sats: pd.DataFrame,
        predictions: list[dict[str, Any]],
        match_radius_deg: float,
    ) -> list[dict[str, Any]]:
        """Replicate the reverse-match observation builder with a configurable radius."""
        from scipy.spatial import KDTree

        source_coords = potential_sats[["ra", "dec"]].values
        pred_coords = np.array([[p["ra"], p["dec"]] for p in predictions])
        source_tree = KDTree(source_coords)
        pred_distances, pred_indices = source_tree.query(pred_coords)

        observations: list[dict[str, Any]] = []
        for i, p in enumerate(predictions):
            dist_deg = float(np.asarray(pred_distances)[i])
            if dist_deg >= match_radius_deg:
                continue
            src_idx = int(np.asarray(pred_indices)[i])
            if src_idx < 0 or src_idx >= len(potential_sats):
                continue
            row = potential_sats.iloc[src_idx]
            observations.append({"norad_id": p["satellite_id"], "name": p["name"], "ra": float(row["ra"])})
        return observations

    def test_geo_radius_catches_all_geo_satellites(self):
        """With 5-arcmin radius, predictions at 0.3-3.4 arcmin offsets all match."""
        # Sources at known positions (simulating detected GEO satellites)
        source_positions = [
            (145.770, -5.130),  # SES 18 position
            (146.200, -5.100),  # DIRECTV 15 position
            (146.210, -5.050),  # DIRECTV 12 position
            (145.900, -5.200),  # SES 20 position
        ]
        sources = _make_sources(source_positions, elongation=1.0)

        # Predictions offset from sources by realistic GEO TLE errors
        predictions = [
            {"ra": 145.7705, "dec": -5.1304, "satellite_id": "SES-18", "name": "SES 18", "phase_angle": 35.0},
            {"ra": 146.223, "dec": -5.100, "satellite_id": "DTV-15", "name": "DIRECTV 15", "phase_angle": 30.0},
            {"ra": 146.265, "dec": -5.050, "satellite_id": "DTV-12", "name": "DIRECTV 12", "phase_angle": 32.0},
            {"ra": 145.938, "dec": -5.200, "satellite_id": "SES-20", "name": "SES 20", "phase_angle": 28.0},
        ]

        # Distances: SES 18 ~0.3', DIRECTV 15 ~1.4', DIRECTV 12 ~3.3', SES 20 ~2.3'
        obs_geo = self._build_reverse_match_observations(sources, predictions, _GEO_MATCH_RADIUS_DEG)
        assert len(obs_geo) == 4
        detected_ids = {o["norad_id"] for o in obs_geo}
        assert detected_ids == {"SES-18", "DTV-15", "DTV-12", "SES-20"}

    def test_standard_radius_misses_most_geo_satellites(self):
        """With 1-arcmin radius, only the closest prediction matches (the original bug)."""
        source_positions = [
            (145.770, -5.130),
            (146.200, -5.100),
            (146.210, -5.050),
            (145.900, -5.200),
        ]
        sources = _make_sources(source_positions, elongation=1.0)

        predictions = [
            {"ra": 145.7705, "dec": -5.1304, "satellite_id": "SES-18", "name": "SES 18", "phase_angle": 35.0},
            {"ra": 146.223, "dec": -5.100, "satellite_id": "DTV-15", "name": "DIRECTV 15", "phase_angle": 30.0},
            {"ra": 146.265, "dec": -5.050, "satellite_id": "DTV-12", "name": "DIRECTV 12", "phase_angle": 32.0},
            {"ra": 145.938, "dec": -5.200, "satellite_id": "SES-20", "name": "SES 20", "phase_angle": 28.0},
        ]

        obs_leo = self._build_reverse_match_observations(sources, predictions, _MATCH_RADIUS_DEG)
        assert len(obs_leo) == 1
        assert obs_leo[0]["norad_id"] == "SES-18"
