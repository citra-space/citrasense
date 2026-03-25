"""Unit tests for the HTML report generator."""

import json

from citrascope.processors.report_generator import (
    _deg_to_dms,
    _deg_to_hms,
    generate_html_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TASK = {
    "id": "abc-123",
    "type": "Track",
    "status": "Scheduled",
    "satelliteName": "DIRECTV 15",
    "satelliteId": "sat-001",
    "groundStationName": "AstronomyAcres",
    "telescopeName": "Planewave",
    "username": "testuser",
    "taskStart": "2026-03-24T03:06:00Z",
    "taskStop": "2026-03-24T03:07:00Z",
    "assigned_filter_name": "r",
}

_SAMPLE_FITS_HEADER = {
    "DATE-OBS": "2026-03-24T03:06:13.664",
    "NAXIS1": 7104,
    "NAXIS2": 5328,
    "XBINNING": 2,
    "YBINNING": 2,
    "EXPTIME": 1.0,
    "FILTER": "r",
    "INSTRUME": "Moravian USB Camera 1",
    "TELESCOP": "Planewave",
    "SITELAT": 31.907,
    "SITELONG": -109.021,
    "SITEALT": 1239.0,
    "GAIN": 0,
}

_SAMPLE_SUMMARY = {
    "should_upload": True,
    "skip_reason": None,
    "total_time": 12.5,
    "processors": [
        {
            "processor_name": "plate_solver",
            "confidence": 1.0,
            "reason": "Plate solved in 1.1s",
            "processing_time_seconds": 1.1,
            "should_upload": True,
        },
        {
            "processor_name": "satellite_matcher",
            "confidence": 1.0,
            "reason": "Matched 2 satellite(s)",
            "processing_time_seconds": 1.3,
            "should_upload": True,
        },
    ],
    "extracted_data": {},
}

_SAMPLE_PLATE = {
    "processor_name": "plate_solver",
    "confidence": 1.0,
    "reason": "Plate solved",
    "extracted_data": {
        "plate_solved": True,
        "ra_center": 125.849,
        "dec_center": -5.052,
        "pixel_scale": 1.012,
        "field_width_deg": 2.0,
        "field_height_deg": 1.5,
        "num_sources": 4170,
    },
}

_SAMPLE_PHOTOMETRY = {
    "processor_name": "photometry",
    "extracted_data": {"zero_point": 20.11, "num_calibration_stars": 3975, "filter": "r"},
}

_SAMPLE_SAT_DEBUG = {
    "tracking_mode": "rate",
    "elongation_filter_applied": True,
    "elongation_threshold": 1.5,
    "field_radius_deg": 2.0,
    "match_radius_arcmin": 1.0,
    "elset_count": 29992,
    "source_classification": {
        "total_sources": 4170,
        "satellite_candidate_count": 3678,
        "star_like_count": 492,
        "elongation_min": 1.0,
        "elongation_max": 10.86,
        "elongation_median": 2.46,
    },
    "target_satellite": {
        "satellite_id": "sat-001",
        "pointing_tle": ["1 99999U ...", "2 99999 ..."],
        "pointing_elset_epoch": "2026-03-23T13:25:40Z",
        "cache_tle": ["1 99999U ...", "2 99999 ..."],
        "tle_match": True,
    },
    "predictions_in_field": [
        {
            "satellite_id": "sat-001",
            "name": "DIRECTV 15",
            "predicted_ra_deg": 125.88,
            "predicted_dec_deg": -5.13,
            "distance_from_center_deg": 0.08,
            "in_field": True,
            "phase_angle": 57.0,
        },
    ],
    "predictions_in_field_count": 1,
    "reverse_match": [
        {
            "satellite_id": "sat-001",
            "name": "DIRECTV 15",
            "predicted_ra": 125.88,
            "predicted_dec": -5.13,
            "nearest_source_distance_arcmin": 0.34,
            "within_match_radius": True,
            "nearest_source_ra": 125.88,
            "nearest_source_dec": -5.13,
            "nearest_source_elongation": 3.99,
            "nearest_source_mag": -3.68,
        },
    ],
    "satellite_observations": [
        {
            "name": "DIRECTV 15",
            "ra": 125.88,
            "dec": -5.13,
            "mag": 16.43,
            "mag_instrumental": -3.68,
            "phase_angle": 57.0,
            "elongation": 3.99,
        },
    ],
}


def _write_png(path):
    """Create a minimal valid PNG (1x1 red pixel)."""
    import struct
    import zlib

    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\x00\x00")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _populate_report_dir(base, *, include_all=True):
    """Write artifacts for report generation testing."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "task.json").write_text(json.dumps(_SAMPLE_TASK))
    (base / "fits_header.json").write_text(json.dumps(_SAMPLE_FITS_HEADER))
    (base / "processing_summary.json").write_text(json.dumps(_SAMPLE_SUMMARY))

    if include_all:
        (base / "plate_solver_result.json").write_text(json.dumps(_SAMPLE_PLATE))
        (base / "photometry_result.json").write_text(json.dumps(_SAMPLE_PHOTOMETRY))
        (base / "satellite_matcher_debug.json").write_text(json.dumps(_SAMPLE_SAT_DEBUG))
        _write_png(base / "annotated.png")


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


class TestCoordinateConversion:
    def test_deg_to_hms(self):
        result = _deg_to_hms(125.849)
        assert result.startswith("08h 23m")

    def test_deg_to_dms_positive(self):
        result = _deg_to_dms(31.907)
        assert result.startswith("+31")

    def test_deg_to_dms_negative(self):
        result = _deg_to_dms(-5.052)
        assert result.startswith("-05")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    def test_full_report(self, tmp_path):
        _populate_report_dir(tmp_path)
        result = generate_html_report(tmp_path)
        assert result is not None
        assert result.name == "report.html"
        assert result.exists()

        html = result.read_text()
        assert "<!DOCTYPE html>" in html
        assert "DIRECTV 15" in html
        assert "Pipeline Summary" in html
        assert "Plate Solve" in html
        assert "Photometry" in html
        assert "Satellite Matching Results" in html
        assert "Source Classification" in html
        assert "Observation Details" in html
        assert "Task Metadata" in html
        assert "data:image/png;base64," in html

    def test_minimal_report(self, tmp_path):
        """Report generates even with only the basics (no optional artifacts)."""
        _populate_report_dir(tmp_path, include_all=False)
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "DIRECTV 15" in html
        assert "Pipeline Summary" in html

    def test_empty_directory(self, tmp_path):
        """Report generates even with no artifacts (all N/A)."""
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "<!DOCTYPE html>" in html

    def test_report_contains_satellite_matches(self, tmp_path):
        _populate_report_dir(tmp_path)
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "DIRECTV 15" in html
        assert "16.43" in html  # calibrated mag

    def test_report_contains_predictions(self, tmp_path):
        _populate_report_dir(tmp_path)
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "Predictions In Field" in html
        assert "29992" in html  # elset count

    def test_report_contains_tle_comparison(self, tmp_path):
        _populate_report_dir(tmp_path)
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "TLE Comparison" in html
        assert "Identical" in html  # tle_match is True

    def test_report_contains_source_classification(self, tmp_path):
        _populate_report_dir(tmp_path)
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "4170" in html  # total sources
        assert "3678" in html  # satellite candidates
        assert "rate" in html  # tracking mode

    def test_report_contains_observation_details(self, tmp_path):
        _populate_report_dir(tmp_path)
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "Moravian" in html
        assert "31.907" in html  # SITELAT

    def test_pointing_section_present_when_data_exists(self, tmp_path):
        _populate_report_dir(tmp_path)
        pointing = {
            "converged": True,
            "attempts": 3,
            "max_attempts": 10,
            "final_angular_distance_deg": 0.001,
            "convergence_threshold_deg": 0.01,
        }
        (tmp_path / "pointing_report.json").write_text(json.dumps(pointing))
        result = generate_html_report(tmp_path)
        assert result is not None
        html = result.read_text()
        assert "Pointing Analysis" in html
        assert "Converged" in html
