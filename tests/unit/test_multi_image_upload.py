"""Tests for multi-image task completion tracking.

Validates that _on_image_done aggregates per-image results and only calls
mark_task_complete, record stats, and remove_task_from_all_stages once —
after ALL images for a task have finished.
"""

import threading
from unittest.mock import MagicMock, patch


def _make_task_instance():
    """Create a minimal AbstractBaseTelescopeTask for testing _on_image_done."""
    from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

    class ConcreteTask(AbstractBaseTelescopeTask):
        def execute(self):
            pass

    api_client = MagicMock()
    api_client.mark_task_complete.return_value = True
    hardware_adapter = MagicMock()
    logger = MagicMock()
    task = MagicMock()
    task.id = "task-abc"
    task.sensor_type = "telescope"
    task_manager = MagicMock()
    settings = MagicMock()
    location_service = MagicMock()

    return ConcreteTask(
        api_client,
        hardware_adapter,
        logger,
        task,
        settings=settings,
        runtime=task_manager,
        location_service=location_service,
        telescope_record=None,
        ground_station=None,
        elset_cache=None,
        processor_registry=None,
    )


class TestOnImageDone:
    def test_single_image_success(self):
        t = _make_task_instance()
        t._pending_images = 1

        t._on_image_done("task-abc", success=True)

        t.api_client.mark_task_complete.assert_called_once_with("task-abc")
        t.runtime.record_task_succeeded.assert_called_once()
        t.runtime.record_task_failed.assert_not_called()
        t.runtime.remove_task_from_all_stages.assert_called_once_with("task-abc")

    def test_single_image_failure(self):
        t = _make_task_instance()
        t._pending_images = 1

        t._on_image_done("task-abc", success=False)

        t.api_client.mark_task_complete.assert_not_called()
        t.runtime.record_task_failed.assert_called_once()
        t.runtime.record_task_succeeded.assert_not_called()
        t.runtime.remove_task_from_all_stages.assert_called_once_with("task-abc")

    def test_three_images_all_succeed(self):
        t = _make_task_instance()
        t._pending_images = 3

        t._on_image_done("task-abc", success=True)
        t._on_image_done("task-abc", success=True)

        # Not yet — 2 of 3
        t.api_client.mark_task_complete.assert_not_called()
        t.runtime.remove_task_from_all_stages.assert_not_called()

        t._on_image_done("task-abc", success=True)

        # Now all 3 done
        t.api_client.mark_task_complete.assert_called_once_with("task-abc")
        t.runtime.record_task_succeeded.assert_called_once()
        t.runtime.remove_task_from_all_stages.assert_called_once_with("task-abc")

    def test_three_images_partial_success(self):
        """2 succeed, 1 fails — mark_task_complete still called (data was uploaded)."""
        t = _make_task_instance()
        t._pending_images = 3

        t._on_image_done("task-abc", success=True)
        t._on_image_done("task-abc", success=False)
        t._on_image_done("task-abc", success=True)

        t.api_client.mark_task_complete.assert_called_once_with("task-abc")
        t.runtime.record_task_succeeded.assert_called_once()
        t.runtime.record_task_failed.assert_not_called()

    def test_three_images_all_fail(self):
        """All 3 fail — mark_task_complete NOT called, record_task_failed once."""
        t = _make_task_instance()
        t._pending_images = 3

        t._on_image_done("task-abc", success=False)
        t._on_image_done("task-abc", success=False)
        t._on_image_done("task-abc", success=False)

        t.api_client.mark_task_complete.assert_not_called()
        t.runtime.record_task_failed.assert_called_once()
        t.runtime.record_task_succeeded.assert_not_called()
        t.runtime.remove_task_from_all_stages.assert_called_once_with("task-abc")

    def test_no_premature_removal(self):
        """Task stays in stage tracking until all images finish."""
        t = _make_task_instance()
        t._pending_images = 3

        for _ in range(2):
            t._on_image_done("task-abc", success=True)
            t.runtime.remove_task_from_all_stages.assert_not_called()

        t._on_image_done("task-abc", success=True)
        t.runtime.remove_task_from_all_stages.assert_called_once()


class TestUploadImageAndMarkComplete:
    """Test that upload_image_and_mark_complete resets counters correctly."""

    def test_resets_counters_for_multi_image(self):
        t = _make_task_instance()
        t.settings.processors_enabled = False

        t.upload_image_and_mark_complete(["/img1.fits", "/img2.fits", "/img3.fits"])

        assert t._pending_images == 3
        assert t._completed_images == 0
        assert t._any_upload_succeeded is False
        assert t._finalized is False

    def test_single_image_sets_pending_to_one(self):
        t = _make_task_instance()
        t.settings.processors_enabled = False

        t.upload_image_and_mark_complete(["/img1.fits"])

        assert t._pending_images == 1


class TestSkipUploadPath:
    """Test that processor skip-upload routes through _on_image_done."""

    def test_skip_upload_counts_as_completed(self):
        t = _make_task_instance()
        t._pending_images = 2

        result = MagicMock()
        result.should_upload = False
        result.skip_reason = "No satellites detected"

        t._on_processing_complete("/img1.fits", "task-abc", result)

        # Image counted, but task not yet done (1 of 2)
        assert t._completed_images == 1
        t.runtime.remove_task_from_all_stages.assert_not_called()

    def test_skip_plus_upload_marks_complete_once(self):
        t = _make_task_instance()
        t._pending_images = 2

        # Image 1: processor says skip
        skip_result = MagicMock()
        skip_result.should_upload = False
        skip_result.skip_reason = "No satellites"
        t._on_processing_complete("/img1.fits", "task-abc", skip_result)

        # Image 2: upload succeeds
        t._on_image_done("task-abc", success=True)

        # Both done now
        t.api_client.mark_task_complete.assert_called_once_with("task-abc")
        t.runtime.remove_task_from_all_stages.assert_called_once()


class TestFinalizationGuard:
    """Verify that duplicate/spurious callbacks don't finalize twice."""

    def test_extra_callbacks_ignored(self):
        t = _make_task_instance()
        t._pending_images = 1

        t._on_image_done("task-abc", success=True)
        t._on_image_done("task-abc", success=True)  # spurious duplicate

        t.api_client.mark_task_complete.assert_called_once()
        t.runtime.record_task_succeeded.assert_called_once()
        t.runtime.remove_task_from_all_stages.assert_called_once()

    def test_zero_pending_never_finalizes(self):
        """_pending_images=0 (uninitialised) should not finalize."""
        t = _make_task_instance()
        # _pending_images defaults to 0

        t._on_image_done("task-abc", success=True)

        t.api_client.mark_task_complete.assert_not_called()
        t.runtime.remove_task_from_all_stages.assert_not_called()


class TestMarkCompleteRetry:
    """Verify mark_task_complete retries and failure handling."""

    @patch("citrasense.tasks.scope.base_telescope_task.time.sleep")
    def test_retries_on_transient_failure(self, mock_sleep):
        t = _make_task_instance()
        t._pending_images = 1
        t.api_client.mark_task_complete.side_effect = [False, False, True]

        t._on_image_done("task-abc", success=True)

        assert t.api_client.mark_task_complete.call_count == 3
        t.runtime.record_task_succeeded.assert_called_once()

    @patch("citrasense.tasks.scope.base_telescope_task.time.sleep")
    def test_records_failure_after_exhausted_retries(self, mock_sleep):
        t = _make_task_instance()
        t._pending_images = 1
        t.api_client.mark_task_complete.return_value = False

        t._on_image_done("task-abc", success=True)

        assert t.api_client.mark_task_complete.call_count == 3
        t.runtime.record_task_failed.assert_called_once()
        t.runtime.record_task_succeeded.assert_not_called()
        # Still removes from stages so the task doesn't get stuck
        t.runtime.remove_task_from_all_stages.assert_called_once()


class TestThreadSafety:
    """Verify counter operations are thread-safe under concurrent callbacks."""

    def test_concurrent_completions(self):
        t = _make_task_instance()
        t._pending_images = 50
        barrier = threading.Barrier(50)

        def complete_one():
            barrier.wait()
            t._on_image_done("task-abc", success=True)

        threads = [threading.Thread(target=complete_one) for _ in range(50)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5)

        assert t._completed_images == 50
        t.api_client.mark_task_complete.assert_called_once()
        t.runtime.remove_task_from_all_stages.assert_called_once()
