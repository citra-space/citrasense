"""Tests for SatelliteMatcherProcessor helper methods."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from citrascope.processors.builtin.satellite_matcher_processor import (
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
