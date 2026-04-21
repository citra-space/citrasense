"""Unit tests for UploadQueue."""

from unittest.mock import MagicMock

import pytest

from citrasense.tasks.upload_queue import UploadQueue


@pytest.fixture
def uq():
    settings = MagicMock()
    settings.max_task_retries = 3
    settings.initial_retry_delay_seconds = 1
    settings.max_retry_delay_seconds = 10
    return UploadQueue(num_workers=1, settings=settings, logger=MagicMock())


def test_get_task_from_item(uq):
    task = MagicMock()
    assert uq._get_task_from_item({"task": task}) is task


def test_cleanup_files(uq, tmp_path):
    fits = tmp_path / "img.fits"
    fits.write_text("data")
    new = tmp_path / "img.new"
    new.write_text("data")
    cat = tmp_path / "img.cat"
    cat.write_text("data")
    uq._cleanup_files(str(fits))
    assert not fits.exists()
    assert not new.exists()
    assert not cat.exists()


def test_cleanup_nonexistent(uq):
    uq._cleanup_files("/nonexistent/path.fits")


def test_execute_work_obs_path(uq):
    pr = MagicMock()
    pr.extracted_data = {
        "satellite_matcher.satellite_observations": [{"norad_id": "25544", "mag": 12.5}],
    }
    mock_api = MagicMock()
    mock_api.upload_optical_observations.return_value = True
    mock_api.mark_task_complete.return_value = True
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": "/tmp/img.fits",
        "processing_result": pr,
        "api_client": mock_api,
        "telescope_record": {"id": "tel-1"},
        "sensor_location": {"latitude": 34.0, "longitude": -118.0, "altitude": 500.0},
        "settings": MagicMock(keep_images=False),
        "on_complete": MagicMock(),
    }
    success, result = uq._execute_work(item)
    assert success is True
    assert result["obs_path"] is True
    mock_api.upload_optical_observations.assert_called_once()


def test_execute_work_fits_path(uq):
    mock_api = MagicMock()
    mock_api.upload_image.return_value = "/url"
    mock_api.mark_task_complete.return_value = True
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": "/tmp/img.fits",
        "processing_result": None,
        "api_client": mock_api,
        "telescope_record": {"id": "tel-1"},
        "sensor_location": None,
        "settings": MagicMock(keep_images=True),
        "on_complete": MagicMock(),
    }
    success, _result = uq._execute_work(item)
    assert success is True
    mock_api.upload_image.assert_called_once()


def test_execute_work_upload_fails(uq):
    mock_api = MagicMock()
    mock_api.upload_image.return_value = None
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": "/tmp/img.fits",
        "processing_result": None,
        "api_client": mock_api,
        "telescope_record": {"id": "tel-1"},
        "sensor_location": None,
        "settings": MagicMock(),
        "on_complete": MagicMock(),
    }
    success, _ = uq._execute_work(item)
    assert success is False


def test_on_success_cleanup(uq, tmp_path):
    fits = tmp_path / "img.fits"
    fits.write_text("data")
    cb = MagicMock()
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": str(fits),
        "settings": MagicMock(keep_images=False),
        "on_complete": cb,
    }
    uq._on_success(item, {"obs_path": False})
    cb.assert_called_once_with("t1", success=True)
    assert not fits.exists()


def test_on_success_keep_images(uq, tmp_path):
    fits = tmp_path / "img.fits"
    fits.write_text("data")
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": str(fits),
        "settings": MagicMock(keep_images=True),
        "on_complete": MagicMock(),
    }
    uq._on_success(item, {"obs_path": False})
    assert fits.exists()


def test_on_permanent_failure(uq):
    cb = MagicMock()
    item = {"task_id": "t1", "task": MagicMock(), "on_complete": cb}
    uq._on_permanent_failure(item)
    cb.assert_called_once_with("t1", success=False)


def test_execute_work_uncalibrated_obs_falls_back_to_fits(uq):
    """Observations with mag=None (photometry failed) must not take the optical path."""
    pr = MagicMock()
    pr.extracted_data = {
        "satellite_matcher.satellite_observations": [
            {"norad_id": "25544", "mag": None, "mag_instrumental": -8.5},
        ],
    }
    mock_api = MagicMock()
    mock_api.upload_image.return_value = "/url"
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": "/tmp/img.fits",
        "processing_result": pr,
        "api_client": mock_api,
        "telescope_record": {"id": "tel-1"},
        "sensor_location": {"latitude": 34.0, "longitude": -118.0, "altitude": 500.0},
        "settings": MagicMock(keep_images=True),
        "on_complete": MagicMock(),
    }
    success, result = uq._execute_work(item)
    assert success is True
    assert result["obs_path"] is False
    mock_api.upload_optical_observations.assert_not_called()
    mock_api.upload_image.assert_called_once()


def test_execute_work_calibrated_obs_uses_optical_path(uq):
    """Observations with real calibrated mag should take the optical path."""
    pr = MagicMock()
    pr.extracted_data = {
        "satellite_matcher.satellite_observations": [
            {"norad_id": "25544", "mag": 14.3, "mag_instrumental": -8.5},
        ],
    }
    mock_api = MagicMock()
    mock_api.upload_optical_observations.return_value = True
    item = {
        "task_id": "t1",
        "task": MagicMock(),
        "image_path": "/tmp/img.fits",
        "processing_result": pr,
        "api_client": mock_api,
        "telescope_record": {"id": "tel-1"},
        "sensor_location": {"latitude": 34.0, "longitude": -118.0, "altitude": 500.0},
        "settings": MagicMock(keep_images=True),
        "on_complete": MagicMock(),
    }
    success, result = uq._execute_work(item)
    assert success is True
    assert result["obs_path"] is True
    mock_api.upload_optical_observations.assert_called_once()
