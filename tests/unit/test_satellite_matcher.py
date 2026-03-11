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
def _make_sources(coords: list[tuple[float, float]], fwhm: float = 2.0) -> pd.DataFrame:
    """Create a minimal source DataFrame from (ra, dec) pairs."""
    return pd.DataFrame(
        {
            "ra": [c[0] for c in coords],
            "dec": [c[1] for c in coords],
            "mag": [12.0] * len(coords),
            "magerr": [0.01] * len(coords),
            "fwhm": [fwhm] * len(coords),
        }
    )


class TestSubtractKnownStars:
    def test_removes_matching_sources(self, tmp_path: Path):
        star_ra, star_dec = 180.0, 45.0
        sat_ra, sat_dec = 181.0, 45.5

        candidates = _make_sources([(star_ra, star_dec), (sat_ra, sat_dec)])

        xmatch = pd.DataFrame({"ra": [star_ra], "dec": [star_dec], "radeg": [star_ra], "decdeg": [star_dec]})
        xmatch.to_csv(tmp_path / "photometry_crossmatch.csv", index=False)

        filtered, stats = SatelliteMatcherProcessor._subtract_known_stars(candidates, tmp_path)

        assert len(filtered) == 1
        assert float(filtered.iloc[0]["ra"]) == sat_ra
        assert stats["catalog_stars_removed"] == 1
        assert stats["candidates_before"] == 2
        assert stats["candidates_after"] == 1

    def test_retains_all_when_no_match(self, tmp_path: Path):
        candidates = _make_sources([(100.0, 30.0), (101.0, 31.0)])

        xmatch = pd.DataFrame({"ra": [200.0], "dec": [-40.0], "radeg": [200.0], "decdeg": [-40.0]})
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


# ===================================================================
# 2. Reverse match produces at most one observation per prediction
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
