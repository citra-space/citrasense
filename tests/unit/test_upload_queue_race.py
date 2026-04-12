"""Tests for the upload queue race condition fix (Issue #178).

Validates two fixes:
1. _queue_for_upload bails out when the upload queue is stopped
2. Config reload does NOT restore processing_tasks / uploading_tasks metadata
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

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

    task_manager = MagicMock()
    task_manager.upload_queue = upload_queue

    task = MagicMock()
    task.id = "task-1"

    return _StubTelescopeTask(
        api_client=MagicMock(),
        hardware_adapter=MagicMock(),
        logger=MagicMock(),
        task=task,
        settings=MagicMock(),
        task_manager=task_manager,
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

        stub.task_manager.upload_queue.submit.assert_not_called()
        stub.task_manager.update_task_stage.assert_not_called()

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

        stub.task_manager.upload_queue.submit.assert_called_once()
        stub.task_manager.update_task_stage.assert_called_once_with("task-1", "uploading")


# ---------------------------------------------------------------------------
# Config reload does NOT restore processing/uploading metadata
# ---------------------------------------------------------------------------


class TestConfigReloadDropsInFlightStages:
    """_initialize_telescope must not restore processing_tasks or uploading_tasks."""

    def _make_daemon(self):
        from citrascope.citra_scope_daemon import CitraScopeDaemon

        daemon = CitraScopeDaemon.__new__(CitraScopeDaemon)
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

        daemon.hardware_adapter = MagicMock()
        daemon.hardware_adapter.connect.return_value = True
        daemon.hardware_adapter.is_mount_homed.return_value = True
        daemon.hardware_adapter.supports_direct_camera_control.return_value = False
        daemon.hardware_adapter.get_filter_config.return_value = {}

        daemon.location_service = MagicMock()
        daemon.time_monitor = MagicMock()
        daemon.web_server = MagicMock()
        daemon.elset_cache = MagicMock()
        daemon.apass_catalog = MagicMock()
        daemon.processor_registry = MagicMock()
        daemon.task_manager = None
        daemon.safety_monitor = None
        daemon.telescope_record = None
        daemon.ground_station = None
        daemon.latest_annotated_image_path = None
        daemon.preview_bus = MagicMock()
        return daemon

    def test_processing_and_uploading_not_restored(self):
        daemon = self._make_daemon()

        old_processing = {"task-p1": MagicMock()}
        old_uploading = {"task-u1": MagicMock(), "task-u2": MagicMock()}

        with (
            patch("citrascope.citra_scope_daemon.TaskManager") as MockTM,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_tm_instance = MagicMock()
            mock_tm_instance.task_dict = {}
            mock_tm_instance.imaging_tasks = {}
            mock_tm_instance.processing_tasks = {}
            mock_tm_instance.uploading_tasks = {}
            MockTM.return_value = mock_tm_instance

            success, _error = daemon._initialize_telescope(
                old_task_dict={"task-1": MagicMock()},
                old_imaging_tasks={"task-i1": MagicMock()},
                old_processing_tasks=old_processing,
                old_uploading_tasks=old_uploading,
            )

        assert success is True
        assert mock_tm_instance.processing_tasks == {}
        assert mock_tm_instance.uploading_tasks == {}
        assert "task-1" in mock_tm_instance.task_dict
        assert "task-i1" in mock_tm_instance.imaging_tasks
