"""Tests for TrackingTelescopeTask — custom tracking rate lifecycle."""

from unittest.mock import MagicMock, patch

import pytest

from citrasense.tasks.scope.tracking_telescope_task import TrackingTelescopeTask
from citrasense.tasks.task import Task


def _make_task_dict(**overrides):
    base = {
        "id": "task-1",
        "type": "Track",
        "status": "Pending",
        "taskStart": "2025-01-01T00:00:00Z",
        "taskStop": "2025-01-01T00:05:00Z",
        "satelliteId": "sat-iss",
        "satelliteName": "ISS",
    }
    base.update(overrides)
    return Task.from_dict(base)


def _make_tracking_task():
    api = MagicMock()
    api.get_satellite.return_value = {
        "name": "ISS",
        "elsets": [{"tle": ["line1", "line2"], "creationEpoch": "2025-01-01T00:00:00Z"}],
    }
    adapter = MagicMock()
    adapter.filter_map = {}
    adapter.set_custom_tracking_rate.return_value = True
    adapter.take_image.return_value = "/tmp/test.fits"

    settings = MagicMock()
    settings.processors_enabled = True
    task_manager = MagicMock()
    location_service = MagicMock()
    location_service.get_current_location.return_value = {
        "latitude": 37.0,
        "longitude": -122.0,
        "altitude": 100.0,
    }

    task_obj = _make_task_dict()
    logger = MagicMock()

    ct = TrackingTelescopeTask(
        api,
        adapter,
        logger,
        task_obj,
        settings=settings,
        task_manager=task_manager,
        location_service=location_service,
        telescope_record={"id": "tel-1"},
        ground_station={"id": "gs-1"},
        elset_cache=None,
        processor_registry=None,
    )
    return ct


class TestTrackingRateReset:
    def test_reset_called_after_successful_exposure(self):
        ct = _make_tracking_task()

        ra_rate = MagicMock()
        ra_rate.arcseconds.per_second = 50.0
        dec_rate = MagicMock()
        dec_rate.arcseconds.per_second = -10.0

        with (
            patch.object(ct, "point_to_lead_position"),
            patch.object(ct, "get_target_radec_and_rates", return_value=(None, None, ra_rate, dec_rate)),
            patch.object(ct, "upload_image_and_mark_complete", return_value=True),
        ):
            ct.execute()

        ct.hardware_adapter.reset_tracking_rates.assert_called_once()

    def test_reset_called_even_when_tracking_fails(self):
        ct = _make_tracking_task()
        ct.hardware_adapter.set_custom_tracking_rate.return_value = False

        ra_rate = MagicMock()
        ra_rate.arcseconds.per_second = 50.0
        dec_rate = MagicMock()
        dec_rate.arcseconds.per_second = -10.0

        with (
            patch.object(ct, "point_to_lead_position"),
            patch.object(ct, "get_target_radec_and_rates", return_value=(None, None, ra_rate, dec_rate)),
        ):
            ct.execute()

        ct.hardware_adapter.reset_tracking_rates.assert_called_once()

    def test_reset_called_even_when_exposure_raises(self):
        ct = _make_tracking_task()

        ra_rate = MagicMock()
        ra_rate.arcseconds.per_second = 50.0
        dec_rate = MagicMock()
        dec_rate.arcseconds.per_second = -10.0
        ct.hardware_adapter.take_image.side_effect = RuntimeError("camera error")

        with (
            patch.object(ct, "point_to_lead_position"),
            patch.object(ct, "get_target_radec_and_rates", return_value=(None, None, ra_rate, dec_rate)),
            pytest.raises(RuntimeError, match="camera error"),
        ):
            ct.execute()

        ct.hardware_adapter.reset_tracking_rates.assert_called_once()

    def test_reset_not_called_before_tracking_set(self):
        """If the task fails before reaching the tracking block, reset should not be called."""
        ct = _make_tracking_task()

        with patch.object(ct, "fetch_satellite", return_value=None):
            with pytest.raises(ValueError, match="satellite data"):
                ct.execute()

        ct.hardware_adapter.reset_tracking_rates.assert_not_called()


class TestDirectAdapterTrackingReturn:
    """Verify DirectHardwareAdapter.set_custom_tracking_rate returns the mount's boolean."""

    def test_returns_true_when_mount_succeeds(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        adapter = MagicMock()
        adapter._mount = MagicMock()
        adapter._mount.set_custom_tracking_rates.return_value = True

        result = DirectHardwareAdapter.set_custom_tracking_rate(adapter, 10.0, 5.0)
        assert result is True

    def test_returns_false_when_mount_fails(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        adapter = MagicMock()
        adapter._mount = MagicMock()
        adapter._mount.set_custom_tracking_rates.return_value = False

        result = DirectHardwareAdapter.set_custom_tracking_rate(adapter, 10.0, 5.0)
        assert result is False

    def test_returns_false_when_no_mount(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        adapter = MagicMock()
        adapter._mount = None

        result = DirectHardwareAdapter.set_custom_tracking_rate(adapter, 10.0, 5.0)
        assert result is False
