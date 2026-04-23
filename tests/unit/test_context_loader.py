"""Unit tests for the processing debug directory context loader."""

import json

import numpy as np
import pytest
from astropy.io import fits

from citrasense.pipelines.common.context_loader import (
    FixedLocationService,
    _discover_fits,
    _task_from_saved_dict,
    load_context_from_debug_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TASK = {
    "id": "abc-123",
    "type": "SATELLITE",
    "status": "ASSIGNED",
    "creationEpoch": "2025-11-11T00:00:00Z",
    "updateEpoch": "2025-11-11T00:00:00Z",
    "taskStart": "2025-11-11T18:00:00Z",
    "taskStop": "2025-11-11T19:00:00Z",
    "userId": "user-1",
    "username": "testuser",
    "satelliteId": "25544",
    "satelliteName": "ISS",
    "telescopeId": "tel-1",
    "telescopeName": "TestScope",
    "groundStationId": "gs-1",
    "groundStationName": "TestStation",
    "assigned_filter_name": "Clear",
}

_SAMPLE_LOCATION = {"latitude": 40.0, "longitude": -111.0, "altitude": 1400.0, "source": "gps"}

_SAMPLE_TELESCOPE = {
    "focalLength": 1200,
    "pixelSize": 3.76,
    "horizontalPixelCount": 4656,
    "verticalPixelCount": 3520,
}

_SAMPLE_ELSETS = [
    {
        "satellite_id": "25544",
        "name": "ISS",
        "tle": [
            "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
            "2 25544  51.6416 208.5340 0001234 123.4567 236.5433 15.50000000999999",
        ],
    }
]


def _write_dummy_fits(path):
    """Create a minimal valid FITS file."""
    data = np.zeros((64, 64), dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(path, overwrite=True)


def _populate_debug_dir(base, *, include_optional=True, fits_name="original_capture.fits"):
    """Write the standard set of debug artifacts into *base*."""
    base.mkdir(parents=True, exist_ok=True)

    (base / "task.json").write_text(json.dumps(_SAMPLE_TASK))
    (base / "observer_location.json").write_text(json.dumps(_SAMPLE_LOCATION))
    (base / "telescope_record.json").write_text(json.dumps(_SAMPLE_TELESCOPE))
    (base / "elset_cache_snapshot.json").write_text(json.dumps(_SAMPLE_ELSETS))
    (base / "fits_header.json").write_text(json.dumps({"DATE-OBS": "2025-11-11T18:38:11"}))

    _write_dummy_fits(base / fits_name)

    if include_optional:
        sat_data = {"most_recent_elset": {"tle": _SAMPLE_ELSETS[0]["tle"]}}
        (base / "target_satellite.json").write_text(json.dumps(sat_data))
        (base / "pointing_report.json").write_text(json.dumps({"converged": True, "attempts": 3}))
        (base / "satellite_matcher_debug.json").write_text(json.dumps({"tracking_mode": "rate"}))


# ---------------------------------------------------------------------------
# FixedLocationService
# ---------------------------------------------------------------------------


class TestFixedLocationService:
    def test_returns_stored_location(self):
        svc = FixedLocationService(_SAMPLE_LOCATION)
        assert svc.get_current_location() == _SAMPLE_LOCATION

    def test_returns_empty_dict(self):
        svc = FixedLocationService({})
        assert svc.get_current_location() == {}


# ---------------------------------------------------------------------------
# _task_from_saved_dict
# ---------------------------------------------------------------------------


class TestTaskFromSavedDict:
    def test_round_trip(self):
        task = _task_from_saved_dict(_SAMPLE_TASK)
        assert task.id == "abc-123"
        assert task.satelliteName == "ISS"
        assert task.assigned_filter_name == "Clear"

    def test_missing_fields_default(self):
        task = _task_from_saved_dict({"id": "x"})
        assert task.id == "x"
        assert task.satelliteName == ""
        assert task.assigned_filter_name is None


# ---------------------------------------------------------------------------
# _discover_fits
# ---------------------------------------------------------------------------


class TestDiscoverFits:
    def test_prefers_original(self, tmp_path):
        _write_dummy_fits(tmp_path / "original_raw.fits")
        _write_dummy_fits(tmp_path / "calibrated.fits")
        _write_dummy_fits(tmp_path / "something_wcs.fits")
        result = _discover_fits(tmp_path)
        assert result is not None
        assert result.name == "original_raw.fits"

    def test_skips_calibrated_and_wcs(self, tmp_path):
        _write_dummy_fits(tmp_path / "calibrated.fits")
        _write_dummy_fits(tmp_path / "image_wcs.fits")
        _write_dummy_fits(tmp_path / "capture.fits")
        result = _discover_fits(tmp_path)
        assert result is not None
        assert result.name == "capture.fits"

    def test_falls_back_to_calibrated(self, tmp_path):
        _write_dummy_fits(tmp_path / "calibrated.fits")
        result = _discover_fits(tmp_path)
        assert result is not None
        assert result.name == "calibrated.fits"

    def test_returns_none_when_empty(self, tmp_path):
        assert _discover_fits(tmp_path) is None


# ---------------------------------------------------------------------------
# load_context_from_debug_dir
# ---------------------------------------------------------------------------


class TestLoadContextFromDebugDir:
    def test_full_reconstruction(self, tmp_path):
        debug_dir = tmp_path / "processing" / "task-abc"
        output_dir = tmp_path / "output"
        _populate_debug_dir(debug_dir)

        from unittest.mock import Mock

        settings = Mock()
        settings.enabled_processors = {}

        ctx = load_context_from_debug_dir(debug_dir, output_dir, settings)

        assert ctx.task is not None
        assert ctx.task.id == "abc-123"
        assert ctx.task.satelliteName == "ISS"
        assert ctx.task.assigned_filter_name == "Clear"
        assert ctx.telescope_record == _SAMPLE_TELESCOPE
        assert ctx.tracking_mode == "rate"
        assert ctx.satellite_data is not None
        assert ctx.pointing_report is not None
        assert ctx.pointing_report["converged"] is True
        assert ctx.elset_cache is not None
        assert len(ctx.elset_cache.get_elsets()) == 1

        # FITS was copied into output dir
        assert ctx.image_path.parent == output_dir
        assert ctx.working_image_path == ctx.image_path
        assert ctx.image_path.exists()

        # Original is untouched
        assert (debug_dir / "original_capture.fits").exists()

    def test_optional_files_missing(self, tmp_path):
        debug_dir = tmp_path / "processing" / "task-def"
        output_dir = tmp_path / "output"
        _populate_debug_dir(debug_dir, include_optional=False)

        from unittest.mock import Mock

        settings = Mock()
        settings.enabled_processors = {}

        ctx = load_context_from_debug_dir(debug_dir, output_dir, settings)

        assert ctx.satellite_data is None
        assert ctx.pointing_report is None
        assert ctx.tracking_mode is None
        assert ctx.elset_cache is not None
        assert len(ctx.elset_cache.get_elsets()) == 1  # elsets still loaded

    def test_image_override(self, tmp_path):
        debug_dir = tmp_path / "processing" / "task-ghi"
        output_dir = tmp_path / "output"
        _populate_debug_dir(debug_dir)

        override_fits = tmp_path / "my_custom.fits"
        _write_dummy_fits(override_fits)

        from unittest.mock import Mock

        settings = Mock()
        settings.enabled_processors = {}

        ctx = load_context_from_debug_dir(debug_dir, output_dir, settings, image_override=override_fits)

        assert ctx.image_path.name == "my_custom.fits"
        assert ctx.image_path.parent == output_dir

    def test_missing_debug_dir_raises(self, tmp_path):
        from unittest.mock import Mock

        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_context_from_debug_dir(tmp_path / "nope", tmp_path / "out", Mock())

    def test_no_fits_raises(self, tmp_path):
        debug_dir = tmp_path / "processing" / "task-jkl"
        debug_dir.mkdir(parents=True)
        (debug_dir / "task.json").write_text(json.dumps(_SAMPLE_TASK))
        (debug_dir / "observer_location.json").write_text(json.dumps(_SAMPLE_LOCATION))
        (debug_dir / "telescope_record.json").write_text(json.dumps(_SAMPLE_TELESCOPE))

        from unittest.mock import Mock

        with pytest.raises(FileNotFoundError, match="No FITS"):
            load_context_from_debug_dir(debug_dir, tmp_path / "out", Mock())

    def test_missing_task_json_raises(self, tmp_path):
        debug_dir = tmp_path / "processing" / "task-mno"
        debug_dir.mkdir(parents=True)
        _write_dummy_fits(debug_dir / "original_capture.fits")
        (debug_dir / "observer_location.json").write_text(json.dumps(_SAMPLE_LOCATION))
        (debug_dir / "telescope_record.json").write_text(json.dumps(_SAMPLE_TELESCOPE))

        from unittest.mock import Mock

        with pytest.raises(ValueError, match=r"task\.json"):
            load_context_from_debug_dir(debug_dir, tmp_path / "out", Mock())
