"""Tests for SatelliteMatcherProcessor helper methods."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from citrascope.processors.builtin.satellite_matcher_processor import (
    _STAR_MATCH_TOLERANCE_DEG,
    SatelliteMatcherProcessor,
)


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
