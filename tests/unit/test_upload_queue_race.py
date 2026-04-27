"""Tests for the upload queue race condition fix (Issue #178).

Validates two fixes:
1. _queue_for_upload bails out when the upload queue is stopped
2. Config reload does NOT restore processing_tasks / uploading_tasks metadata
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from citrasense.sensors.telescope.tasks.base_telescope_task import AbstractBaseTelescopeTask

# ---------------------------------------------------------------------------
# Minimal concrete subclass — AbstractBaseTelescopeTask is abstract
# ---------------------------------------------------------------------------


class _StubTelescopeTask(AbstractBaseTelescopeTask):
    """Minimal concrete subclass that only exists to test _queue_for_upload."""

    @property
    def tracking_mode(self) -> str:
        return "sidereal"

    def execute(self) -> bool:
        return True

    def cancel(self) -> None:
        pass


def _make_stub_task(upload_queue_running: bool) -> _StubTelescopeTask:
    """Build a _StubTelescopeTask with just enough mocks for _queue_for_upload."""
    upload_queue = MagicMock()
    upload_queue.running = upload_queue_running

    runtime = MagicMock()
    runtime.upload_queue = upload_queue

    task = MagicMock()
    task.id = "task-1"
    task.sensor_type = "telescope"

    return _StubTelescopeTask(
        api_client=MagicMock(),
        hardware_adapter=MagicMock(),
        logger=MagicMock(),
        task=task,
        settings=MagicMock(),
        runtime=runtime,
        location_service=MagicMock(),
        telescope_record={},
        ground_station=None,
        elset_cache=None,
        processor_registry=MagicMock(),
    )


# ---------------------------------------------------------------------------
# _queue_for_upload guard
# ---------------------------------------------------------------------------


class TestQueueForUploadGuard:
    """_queue_for_upload must not submit when the upload queue is stopped."""

    def test_dead_queue_does_not_submit(self):
        stub = _make_stub_task(upload_queue_running=False)
        stub._queue_for_upload("/fake/image.fits", MagicMock())

        stub.runtime.upload_queue.submit.assert_not_called()
        stub.runtime.update_task_stage.assert_not_called()

    def test_dead_queue_logs_warning(self):
        stub = _make_stub_task(upload_queue_running=False)
        stub._queue_for_upload("/fake/image.fits", MagicMock())

        stub.logger.warning.assert_called_once()
        msg = stub.logger.warning.call_args[0][0]
        assert "task-1" in msg
        assert "not queueing" in msg.lower()

    def test_running_queue_submits_normally(self):
        stub = _make_stub_task(upload_queue_running=True)
        stub._queue_for_upload("/fake/image.fits", MagicMock())

        stub.runtime.upload_queue.submit.assert_called_once()
        stub.runtime.update_task_stage.assert_called_once_with("task-1", "uploading")


# ---------------------------------------------------------------------------
# Config reload does NOT restore processing/uploading metadata
# ---------------------------------------------------------------------------


class TestConfigReloadDropsInFlightStages:
    """_initialize_telescopes must not restore processing_tasks or uploading_tasks."""

    def _make_daemon(self):
        from citrasense.citrasense_daemon import CitraSenseDaemon

        daemon = CitraSenseDaemon.__new__(CitraSenseDaemon)
        daemon.settings = MagicMock()
        daemon.api_client = MagicMock()
        daemon.api_client.does_api_server_accept_key.return_value = True
        daemon.api_client.get_telescope.return_value = {
            "id": "t1",
            "groundStationId": "gs1",
            "maxSlewRate": 5.0,
            "name": "Test",
        }
        daemon.api_client.get_ground_station.return_value = {"id": "gs1", "name": "TestGS"}

        telescope_sensor = MagicMock()
        telescope_sensor.sensor_id = "t1"
        telescope_sensor.sensor_type = "telescope"
        telescope_sensor.adapter = MagicMock()
        telescope_sensor.adapter.connect.return_value = True
        telescope_sensor.adapter.is_mount_homed.return_value = True
        telescope_sensor.adapter.supports_direct_camera_control.return_value = False
        telescope_sensor.adapter.get_filter_config.return_value = {}
        telescope_sensor.citra_record = None

        sensor_manager = MagicMock()
        sensor_manager.iter_by_type.return_value = iter([telescope_sensor])
        sensor_manager.get.return_value = telescope_sensor
        sensor_manager.get_sensor.return_value = telescope_sensor
        sensor_manager.__iter__ = lambda self: iter([telescope_sensor])
        # first_of_type was removed — delete the MagicMock auto-attribute so
        # callers that still reach for it fail loudly.
        del sensor_manager.first_of_type

        daemon.sensor_manager = sensor_manager
        daemon.location_service = MagicMock()
        daemon.time_monitor = MagicMock()
        daemon.web_server = MagicMock()
        daemon.elset_cache = MagicMock()
        daemon.apass_catalog = MagicMock()
        daemon.processor_registry = MagicMock()
        daemon.task_dispatcher = None
        daemon.safety_monitor = MagicMock()
        daemon.ground_station = None
        daemon.latest_annotated_image_path = None
        daemon.preview_bus = MagicMock()
        daemon.task_index = MagicMock()
        daemon.sensor_bus = MagicMock()
        daemon._retention_timer = None
        daemon._stop_requested = False
        return daemon

    def test_processing_and_uploading_not_restored(self):
        daemon = self._make_daemon()

        old_processing = {"task-p1": MagicMock()}
        old_uploading = {"task-u1": MagicMock(), "task-u2": MagicMock()}

        with (
            patch("citrasense.citrasense_daemon.SensorRuntime") as MockRT,
            patch("citrasense.citrasense_daemon.TaskDispatcher") as MockTD,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_td_instance = MagicMock()
            mock_td_instance.imaging_tasks = {}
            mock_td_instance.processing_tasks = {}
            mock_td_instance.uploading_tasks = {}
            MockTD.return_value = mock_td_instance
            MockRT.return_value = MagicMock()

            old_task_dict = {"task-1": MagicMock()}
            success, _error = daemon._initialize_telescopes(
                old_task_dict=old_task_dict,
                old_imaging_tasks={"task-i1": MagicMock()},
                old_processing_tasks=old_processing,
                old_uploading_tasks=old_uploading,
            )

        assert success is True
        assert mock_td_instance.processing_tasks == {}
        assert mock_td_instance.uploading_tasks == {}
        mock_td_instance.restore_task_dict.assert_called_once_with(old_task_dict)
        assert "task-i1" in mock_td_instance.imaging_tasks
