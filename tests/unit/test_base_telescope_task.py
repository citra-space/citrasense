"""Tests for AbstractBaseTelescopeTask and StaticTelescopeTask."""

from unittest.mock import MagicMock, patch

import pytest

from citrascope.tasks.task import Task


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


def _make_daemon():
    daemon = MagicMock()
    daemon.settings.processors_enabled = True
    daemon.telescope_record = {"id": "tel-1"}
    daemon.ground_station = {"id": "gs-1"}
    daemon.location_service.get_current_location.return_value = {
        "latitude": 37.0,
        "longitude": -122.0,
        "altitude": 100.0,
    }
    daemon.task_manager = MagicMock()
    daemon.task_manager.record_task_started = MagicMock()
    daemon.task_manager.record_task_succeeded = MagicMock()
    daemon.task_manager.record_task_failed = MagicMock()
    daemon.hardware_adapter = MagicMock()
    return daemon


class TestFetchSatellite:
    def test_returns_satellite_with_elset(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {
            "name": "ISS",
            "elsets": [
                {
                    "tle": ["line1", "line2"],
                    "creationEpoch": "2025-01-01T00:00:00Z",
                }
            ],
        }
        task_obj = _make_task_dict()
        daemon = _make_daemon()
        logger = MagicMock()

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(api, daemon.hardware_adapter, logger, task_obj, daemon)
        result = ct.fetch_satellite()
        assert result is not None
        assert "most_recent_elset" in result

    def test_returns_none_when_no_data(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = None
        task_obj = _make_task_dict()
        daemon = _make_daemon()

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), task_obj, daemon)
        assert ct.fetch_satellite() is None

    def test_returns_none_when_no_elsets(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {"name": "ISS", "elsets": []}
        task_obj = _make_task_dict()
        daemon = _make_daemon()

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), task_obj, daemon)
        assert ct.fetch_satellite() is None


class TestGetMostRecentElset:
    def test_returns_cached_elset(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        sat_data = {"most_recent_elset": {"tle": ["a", "b"]}}
        assert ct._get_most_recent_elset(sat_data) == {"tle": ["a", "b"]}

    def test_selects_most_recent_by_epoch(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        sat_data = {
            "elsets": [
                {"tle": ["old1", "old2"], "creationEpoch": "2024-01-01T00:00:00Z"},
                {"tle": ["new1", "new2"], "creationEpoch": "2025-06-01T00:00:00Z"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data)
        assert result["tle"] == ["new1", "new2"]

    def test_empty_elsets_returns_none(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        assert ct._get_most_recent_elset({"elsets": []}) is None

    def test_missing_creation_epoch_uses_fallback(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        sat_data = {
            "elsets": [
                {"tle": ["a", "b"]},
                {"tle": ["c", "d"], "creationEpoch": "2025-01-01T00:00:00Z"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data)
        assert result["tle"] == ["c", "d"]


class TestUploadImageAndMarkComplete:
    def test_single_filepath_str(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch("citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            result = ct.upload_image_and_mark_complete("/path/to/img.fits")

        assert result is True
        daemon.task_manager.record_task_started.assert_called_once()

    def test_multiple_filepaths(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch("citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            result = ct.upload_image_and_mark_complete(["/a.fits", "/b.fits"])

        assert result is True
        assert daemon.task_manager.processing_queue.submit.call_count == 2

    def test_processors_disabled_goes_to_upload(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.settings.processors_enabled = False
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch("citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            with patch.object(ct, "_queue_for_upload") as mock_upload:
                ct.upload_image_and_mark_complete("/img.fits")
                mock_upload.assert_called_once()

    def test_enrichment_failure_continues(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch(
            "citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata",
            side_effect=Exception("boom"),
        ):
            result = ct.upload_image_and_mark_complete("/img.fits")
            assert result is True


class TestOnProcessingComplete:
    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)

    def test_skip_upload_when_should_upload_false(self):
        ct = self._make_concrete()
        result = MagicMock()
        result.should_upload = False
        result.skip_reason = "Test skip"
        ct._on_processing_complete("/img.fits", "task-1", result)
        ct.api_client.mark_task_complete.assert_called_once_with("task-1")

    def test_feeds_plate_solve_to_adapter(self):
        ct = self._make_concrete()
        result = MagicMock()
        result.should_upload = True
        result.extracted_data = {
            "plate_solver.ra_center": 180.0,
            "plate_solver.dec_center": 45.0,
        }
        with patch.object(ct, "_queue_for_upload"):
            ct._on_processing_complete("/img.fits", "task-1", result)
        ct.daemon.hardware_adapter.update_from_plate_solve.assert_called_once()

    def test_no_result_queues_for_upload(self):
        ct = self._make_concrete()
        with patch.object(ct, "_queue_for_upload") as mock_upload:
            ct._on_processing_complete("/img.fits", "task-1", None)
            mock_upload.assert_called_once()


class TestQueueForUpload:
    def test_submits_to_upload_queue(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)
        ct._queue_for_upload("/img.fits", processing_result=None)
        daemon.task_manager.upload_queue.submit.assert_called_once()

    def test_location_service_failure_passes_none(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.location_service.get_current_location.side_effect = Exception("no GPS")
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)
        ct._queue_for_upload("/img.fits", processing_result=None)
        call_kwargs = daemon.task_manager.upload_queue.submit.call_args
        assert call_kwargs[1]["sensor_location"] is None


class TestOnUploadComplete:
    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)

    def test_success(self):
        ct = self._make_concrete()
        ct._on_upload_complete("task-1", True)
        ct.daemon.task_manager.record_task_succeeded.assert_called_once()

    def test_failure(self):
        ct = self._make_concrete()
        ct._on_upload_complete("task-1", False)
        ct.daemon.task_manager.record_task_failed.assert_called_once()


class TestCancellation:
    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)

    def test_cancel_sets_flag(self):
        ct = self._make_concrete()
        assert ct.is_cancelled is False
        ct.cancel()
        assert ct.is_cancelled is True

    def test_point_to_lead_exits_on_cancel_before_slew(self):
        ct = self._make_concrete()
        ct.cancel()
        with pytest.raises(RuntimeError, match=r"(?i)cancelled"):
            ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

    def test_point_to_lead_exits_on_cancel_during_slew(self):
        ct = self._make_concrete()

        call_count = 0

        def moving_then_cancel():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                ct.cancel()
            return True

        ct.hardware_adapter.telescope_is_moving.side_effect = moving_then_cancel
        ct.hardware_adapter.point_telescope = MagicMock()

        with patch.object(
            ct, "estimate_lead_position", return_value=(MagicMock(degrees=10.0), MagicMock(degrees=20.0), 1.0)
        ):
            with pytest.raises(RuntimeError, match=r"(?i)cancelled"):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})
