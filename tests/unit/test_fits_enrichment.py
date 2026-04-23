"""Unit tests for FITS metadata enrichment."""

from unittest.mock import MagicMock

import pytest
from astropy.io import fits

from citrasense.tasks.fits_enrichment import (
    _add_location_metadata,
    _add_task_metadata,
    enrich_fits_metadata,
)


@pytest.fixture
def fits_file(tmp_path):
    """Create a minimal FITS file for testing."""
    filepath = tmp_path / "test.fits"
    hdu = fits.PrimaryHDU()
    hdu.writeto(str(filepath))
    return str(filepath)


@pytest.fixture
def mock_task():
    t = MagicMock()
    t.id = "task-uuid-123"
    t.sensor_type = "telescope"
    t.satelliteName = "ISS (ZARYA)"
    t.groundStationName = "Desert Station"
    t.telescopeName = "PlaneWave CDK14"
    t.assigned_filter_name = "Luminance"
    return t


@pytest.fixture
def mock_location_service():
    ls = MagicMock()
    ls.get_current_location.return_value = {
        "latitude": 34.05,
        "longitude": -118.25,
        "altitude": 510.0,
        "source": "gps",
    }
    return ls


@pytest.fixture
def telescope_record():
    return {"id": "tel-1", "name": "CDK14"}


@pytest.fixture
def ground_station():
    return {
        "id": "gs-1",
        "name": "Desert Station",
        "latitude": 34.0,
        "longitude": -118.0,
        "altitude": 500.0,
    }


# ---------------------------------------------------------------------------
# enrich_fits_metadata
# ---------------------------------------------------------------------------


def test_enrich_adds_origin(fits_file, mock_location_service):
    enrich_fits_metadata(fits_file, location_service=mock_location_service)
    with fits.open(fits_file) as hdul:
        assert hdul[0].header["ORIGIN"] == "Citra.space"


def test_enrich_adds_task_metadata(fits_file, mock_task, mock_location_service, telescope_record, ground_station):
    enrich_fits_metadata(
        fits_file,
        task=mock_task,
        location_service=mock_location_service,
        telescope_record=telescope_record,
        ground_station=ground_station,
    )
    with fits.open(fits_file) as hdul:
        h = hdul[0].header
        assert h["TASKID"] == "task-uuid-123"
        assert h["OBJECT"] == "ISS (ZARYA)"
        assert h["FILTER"] == "Luminance"
        assert h["TELESCOP"] == "PlaneWave CDK14"


def test_enrich_adds_location_from_gps(fits_file, mock_location_service):
    enrich_fits_metadata(fits_file, location_service=mock_location_service)
    with fits.open(fits_file) as hdul:
        h = hdul[0].header
        assert h["SITELAT"] == pytest.approx(34.05)
        assert h["SITELONG"] == pytest.approx(-118.25)


def test_enrich_idempotent(fits_file, mock_task, mock_location_service, telescope_record, ground_station):
    enrich_fits_metadata(
        fits_file,
        task=mock_task,
        location_service=mock_location_service,
        telescope_record=telescope_record,
        ground_station=ground_station,
    )
    enrich_fits_metadata(
        fits_file,
        task=mock_task,
        location_service=mock_location_service,
        telescope_record=telescope_record,
        ground_station=ground_station,
    )
    with fits.open(fits_file) as hdul:
        assert hdul[0].header["TASKID"] == "task-uuid-123"


def test_enrich_missing_file():
    enrich_fits_metadata("/nonexistent/path.fits")


def test_enrich_no_task_no_daemon(fits_file):
    enrich_fits_metadata(fits_file)
    with fits.open(fits_file) as hdul:
        assert hdul[0].header["ORIGIN"] == "Citra.space"
        assert "TASKID" not in hdul[0].header


def test_enrich_corrupted_file(tmp_path):
    bad_file = tmp_path / "bad.fits"
    bad_file.write_text("not a fits file")
    enrich_fits_metadata(str(bad_file))


# ---------------------------------------------------------------------------
# _add_location_metadata
# ---------------------------------------------------------------------------


def test_add_location_from_ground_station():
    header = fits.Header()
    gs = {"latitude": 35.0, "longitude": -120.0, "altitude": 300.0}
    _add_location_metadata(header, ground_station_record=gs)
    assert header["SITELAT"] == 35.0
    assert header["SITELONG"] == -120.0
    assert header["SITEALT"] == 300.0


def test_add_location_no_source():
    header = fits.Header()
    _add_location_metadata(header)
    assert "SITELAT" not in header


def test_add_location_gps_preferred_over_gs(mock_location_service):
    header = fits.Header()
    gs = {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0}
    _add_location_metadata(header, location_service=mock_location_service, ground_station_record=gs)
    assert header["SITELAT"] == pytest.approx(34.05)


# ---------------------------------------------------------------------------
# _add_task_metadata
# ---------------------------------------------------------------------------


def test_add_task_metadata_full(mock_task):
    header = fits.Header()
    _add_task_metadata(header, mock_task, telescope_record=None, ground_station_record=None)
    assert header["OBJECT"] == "ISS (ZARYA)"
    assert header["OBSERVER"] == "Desert Station"
    assert header["TELESCOP"] == "PlaneWave CDK14"
    assert header["FILTER"] == "Luminance"
    assert header["TASKID"] == "task-uuid-123"


def test_add_task_metadata_fallback_to_records():
    task = MagicMock()
    task.sensor_type = "telescope"
    task.id = ""
    task.satelliteId = ""
    task.satelliteName = ""
    task.telescopeId = ""
    task.telescopeName = ""
    task.groundStationId = ""
    task.groundStationName = ""
    task.assigned_filter_name = None
    header = fits.Header()
    _add_task_metadata(
        header,
        task,
        telescope_record={"name": "Scope A"},
        ground_station_record={"name": "Station B"},
    )
    assert header["TELESCOP"] == "Scope A"
    assert header["OBSERVER"] == "Station B"


def test_add_task_metadata_minimal():
    task = MagicMock()
    task.sensor_type = "telescope"
    task.satelliteName = None
    task.groundStationName = None
    task.telescopeName = None
    task.assigned_filter_name = None
    task.id = None
    header = fits.Header()
    _add_task_metadata(header, task, None, None)
    assert "OBJECT" not in header
    assert "TASKID" not in header
