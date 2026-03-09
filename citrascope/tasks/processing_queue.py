"""Background processing queue for image processing."""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from citrascope.processors.processor_result import ProcessingContext
from citrascope.tasks.base_work_queue import BaseWorkQueue


class ProcessingQueue(BaseWorkQueue):
    """
    Background worker queue for image processing.
    Allows multiple processing tasks to run concurrently without blocking telescope.
    """

    def __init__(self, num_workers: int = 1, settings=None, logger=None):
        """
        Initialize processing queue.

        Args:
            num_workers: Number of concurrent processing threads (default: 1)
            settings: Settings instance with retry configuration
            logger: Logger instance
        """
        super().__init__(num_workers, settings, logger)

    def submit(self, task_id: str, image_path: Path, context: dict, on_complete: Callable):
        """
        Submit image for processing.

        Args:
            task_id: Task identifier
            image_path: Path to captured image
            context: Processing context (task, settings, daemon, etc.)
            on_complete: Callback(task_id, result) when processing finishes
        """
        self.logger.info(f"Queuing task {task_id} for processing")
        self.work_queue.put(
            {"task_id": task_id, "image_path": image_path, "context": context, "on_complete": on_complete}
        )

    def _get_working_dir(self, task_id: str, settings) -> Path:
        """Return the task-specific working directory path."""
        if settings:
            return settings.get_images_dir().parent / "processing" / task_id
        return Path(tempfile.gettempdir()) / "citrascope" / "processing" / task_id

    def _cleanup_working_dir(self, task_id: str, settings):
        """Remove the task-specific working directory, logging any failure."""
        try:
            working_dir = self._get_working_dir(task_id, settings)
            if working_dir.exists():
                shutil.rmtree(working_dir)
                self.logger.debug(f"[ProcessingWorker] Cleaned up working directory: {working_dir}")
        except Exception as e:
            self.logger.warning(f"[ProcessingWorker] Failed to clean up working directory for {task_id}: {e}")

    def _execute_work(self, item):
        """Execute image processing work."""
        task_id = item["task_id"]
        task_obj = item["context"].get("task")

        self.logger.info(f"[ProcessingWorker] Processing task {task_id}")

        try:
            # Create task-specific working directory
            settings = item["context"].get("settings")
            working_dir = self._get_working_dir(task_id, settings)
            working_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"[ProcessingWorker] Created working directory: {working_dir}")

            # Build processing context, injecting specific services rather than the whole daemon
            daemon = item["context"].get("daemon")
            context = ProcessingContext(
                image_path=item["image_path"],
                working_image_path=item["image_path"],  # Initialize to original image
                working_dir=working_dir,
                image_data=None,  # Loaded by processors
                task=task_obj,
                telescope_record=item["context"].get("telescope_record"),
                ground_station_record=item["context"].get("ground_station_record"),
                settings=item["context"].get("settings"),
                location_service=getattr(daemon, "location_service", None),
                elset_cache=getattr(daemon, "elset_cache", None),
                satellite_data=item["context"].get("satellite_data"),
                logger=self.logger,  # Pass logger to processors
            )
            result = daemon.processor_registry.process_all(context)

            # Success
            self.logger.info(f"[ProcessingWorker] Task {task_id} processed in {result.total_time:.2f}s")
            return (True, result)

        except Exception as e:
            self.logger.error(f"[ProcessingWorker] Processing failed for {task_id}: {e}", exc_info=True)
            return (False, None)

    def _on_success(self, item, result):
        """Handle successful processing completion."""
        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        on_complete = item["on_complete"]

        if task_obj:
            task_obj.set_status_msg("Processing complete")

        settings = item["context"].get("settings")
        if not settings or not settings.keep_processing_output:
            self._cleanup_working_dir(task_id, settings)

        on_complete(task_id, result)

    def _on_permanent_failure(self, item):
        """Handle permanent processing failure (fail-open: upload raw image)."""
        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        on_complete = item["on_complete"]

        self.logger.error(f"[ProcessingWorker] Task {task_id} processing permanently failed, uploading raw image")

        if task_obj:
            task_obj.set_status_msg("Processing permanently failed (uploading raw image)")

        settings = item["context"].get("settings")
        if not settings or not settings.keep_processing_output:
            self._cleanup_working_dir(task_id, settings)

        # Fail-open: notify with None result (will upload raw image)
        on_complete(task_id, None)

    def _get_task_from_item(self, item):
        """Get Task object from work item."""
        return item["context"].get("task")
