"""Calibration capture management: operator-requested calibration with imaging queue gate.

Follows the same request/check_and_execute pattern as AutofocusManager
and HomingManager.  Calibration defers tasks while active (like homing)
because capture sessions can take minutes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from citrasense.calibration import FilterSlot
from citrasense.calibration.calibration_library import CalibrationLibrary
from citrasense.calibration.master_builder import MasterBuilder

if TYPE_CHECKING:
    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.tasks.base_work_queue import BaseWorkQueue


class CalibrationManager:
    """Manages operator-requested calibration capture with safe task gating.

    The web UI posts a capture request via ``request(params)``.  On the
    next runner iteration, ``check_and_execute()`` picks it up, waits for
    the imaging queue to drain, then delegates to :class:`MasterBuilder`.
    """

    def __init__(
        self,
        logger: logging.Logger,
        hardware_adapter: AbstractAstroHardwareAdapter,
        library: CalibrationLibrary,
        imaging_queue: BaseWorkQueue | None = None,
    ) -> None:
        self.logger = logger.getChild(type(self).__name__)
        self.hardware_adapter = hardware_adapter
        self.library = library
        self.imaging_queue = imaging_queue

        self._requested = False
        self._running = False
        self._capture_params: dict[str, Any] = {}
        self._progress: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

        self._job_queue: list[dict[str, Any]] = []
        self._batch_total: int = 0
        self._batch_index: int = 0

    # ------------------------------------------------------------------
    # Public API (called from web endpoints)
    # ------------------------------------------------------------------

    def request(self, capture_params: dict[str, Any]) -> bool:
        """Queue a calibration capture job.

        Args:
            capture_params: Dict with keys like ``frame_type``, ``count``,
                ``exposure_time``, ``gain``, ``binning``, ``filter_name``.
        """
        with self._lock:
            if self._running:
                self.logger.info("Calibration already in progress")
                return False
            self._capture_params = capture_params
            self._requested = True
            self.logger.info("Calibration capture requested: %s", capture_params)
            return True

    def request_suite(self, jobs: list[dict[str, Any]]) -> bool:
        """Queue a batch of calibration jobs to execute sequentially.

        Returns False if calibration is already running.
        """
        if not jobs:
            return False
        with self._lock:
            if self._running:
                self.logger.info("Calibration already in progress")
                return False
            self._job_queue = list(jobs)
            self._batch_total = len(jobs)
            self._batch_index = 0
            self._requested = True
            self.logger.info("Calibration suite queued: %d jobs", len(jobs))
            return True

    def cancel(self) -> bool:
        """Cancel calibration whether queued or actively running."""
        with self._lock:
            was_requested = self._requested
            is_running = self._running
            self._requested = False
            self._job_queue.clear()

        if is_running:
            self._cancel_event.set()
            self.logger.info("Calibration cancellation requested (run in progress)")
            return True
        if was_requested:
            self.logger.info("Calibration request cancelled")
            return True
        return False

    def is_requested(self) -> bool:
        with self._lock:
            return self._requested

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._progress)

    # ------------------------------------------------------------------
    # Runner integration
    # ------------------------------------------------------------------

    def check_and_execute(self) -> bool:
        """Check if calibration should run and execute if so.

        Call this between tasks in the runner loop.  Returns True if
        calibration ran.  Handles both single jobs and batch suites.
        """
        with self._lock:
            should_run = self._requested
            has_batch = len(self._job_queue) > 0
            params = dict(self._capture_params) if should_run and not has_batch else {}
            if should_run:
                self._requested = False

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Calibration deferred — waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        if has_batch:
            self._execute_batch()
        else:
            self._execute(params)
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, frame_type: str, status: str) -> None:
        with self._lock:
            progress: dict[str, Any] = {
                "running": True,
                "frame_type": frame_type,
                "current_frame": current,
                "total_frames": total,
                "status": status,
            }
            if self._batch_total > 0:
                progress["batch_current"] = self._batch_index
                progress["batch_total"] = self._batch_total
            self._progress = progress

    @staticmethod
    def _make_batch_label(params: dict[str, Any]) -> str:
        """Short human-readable label for a batch job (e.g. "dark bin2 2.0s")."""
        ft = params.get("frame_type", "?")
        binning = params.get("binning", 1)
        label = f"{ft} bin{binning}"
        if ft == "dark":
            exp = params.get("exposure_time", "?")
            label += f" {exp}s"
        elif ft == "flat":
            fname = params.get("filter_name", "")
            if fname:
                label += f" {fname}"
        elif ft == "interleaved_flat":
            filters = params.get("filters", [])
            names = [f["name"] for f in filters]
            label = f"interleaved flats bin{binning} ({', '.join(names)})"
        return label

    # Temperature tolerance for considering the sensor "at target" (degrees C).
    TEMP_STABILITY_TOLERANCE = 1.0
    TEMP_POLL_INTERVAL = 5.0
    TEMP_STABILITY_TIMEOUT = 300.0

    def _wait_for_temperature(self, camera: Any, frame_type: str) -> bool:
        """Block until the sensor temperature is within tolerance of the target.

        Skipped for flats (shutter open, ambient light dominates) and cameras
        without cooling.  Returns True if stable (or not applicable), False
        if cancelled or timed out.
        """
        if frame_type in ("bias", "flat", "interleaved_flat"):
            return True

        profile = camera.get_calibration_profile()
        if not profile.has_cooling or profile.target_temperature is None:
            return True

        target = profile.target_temperature
        deadline = time.monotonic() + self.TEMP_STABILITY_TIMEOUT

        while not self._cancel_event.is_set():
            current = camera.get_temperature()
            if current is not None and abs(current - target) <= self.TEMP_STABILITY_TOLERANCE:
                self.logger.info("Sensor temperature stable at %.1f°C (target %.1f°C)", current, target)
                return True

            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                self.logger.warning(
                    "Temperature stabilization timed out (current %.1f°C, target %.1f°C). Proceeding anyway.",
                    current if current is not None else float("nan"),
                    target,
                )
                return True

            current_str = f"{current:.1f}" if current is not None else "?"
            with self._lock:
                self._progress = {
                    "running": True,
                    "status": f"Waiting for sensor: {current_str}°C → {target:.0f}°C ({remaining}s remaining)",
                }
            self._cancel_event.wait(self.TEMP_POLL_INTERVAL)

        return False

    def _execute_batch(self) -> None:
        """Execute all queued jobs sequentially."""
        with self._lock:
            jobs = list(self._job_queue)
            self._job_queue.clear()
            self._batch_total = len(jobs)
            self._batch_index = 0
            self._running = True
            self._progress = {"running": True, "status": f"Starting suite ({len(jobs)} jobs)..."}

        self._cancel_event.clear()
        try:
            for i, params in enumerate(jobs):
                if self._cancel_event.is_set():
                    self.logger.info("Calibration suite cancelled at job %d/%d", i + 1, len(jobs))
                    break

                with self._lock:
                    self._batch_index = i + 1
                label = self._make_batch_label(params)
                self.logger.info("Suite job %d/%d: %s", i + 1, len(jobs), label)
                with self._lock:
                    self._progress = {
                        "running": True,
                        "status": f"Job {i + 1}/{len(jobs)}: {label}",
                        "batch_current": i + 1,
                        "batch_total": len(jobs),
                    }

                self._execute_single(params)

            if not self._cancel_event.is_set():
                self.logger.info("Calibration suite completed: %d jobs", len(jobs))
        finally:
            with self._lock:
                self._running = False
                self._batch_total = 0
                self._batch_index = 0
                self._progress = {}

    def _execute(self, params: dict[str, Any]) -> None:
        """Execute a single standalone calibration job (non-batch)."""
        self._cancel_event.clear()
        with self._lock:
            self._running = True
            self._progress = {"running": True, "status": "Starting calibration..."}

        try:
            self._execute_single(params)

            if self._cancel_event.is_set():
                self.logger.info("Calibration capture was cancelled")
            else:
                ft = params.get("frame_type", "?")
                self.logger.info("Calibration capture completed: %s", ft)
        except Exception as e:
            self.logger.error("Calibration capture failed: %s", e, exc_info=True)
        finally:
            with self._lock:
                self._running = False
                self._progress = {}

    def _execute_single(self, params: dict[str, Any]) -> None:
        """Run one calibration job.  Used by both _execute and _execute_batch."""
        camera = self.hardware_adapter.camera
        if camera is None:
            self.logger.error("No direct camera available for calibration")
            return

        profile = camera.get_calibration_profile()
        if not profile.calibration_applicable:
            self.logger.error("Camera does not support CCD-style calibration")
            return

        frame_type = params.get("frame_type", "bias")

        if not self._wait_for_temperature(camera, frame_type):
            self.logger.info("Calibration cancelled while waiting for temperature")
            return

        profile = camera.get_calibration_profile()
        builder = MasterBuilder(camera, self.library, profile)

        count = int(params.get("count", 30))
        gain = params.get("gain")
        if gain is not None:
            gain = int(gain)
        binning = int(params.get("binning", profile.current_binning))

        if frame_type == "bias":
            result = builder.build_bias(
                count=count,
                gain=gain,
                binning=binning,
                cancel_event=self._cancel_event,
                on_progress=self._on_progress,
            )
            if result is None:
                self.logger.warning("Bias capture returned no frames (cancelled?)")
        elif frame_type == "dark":
            exposure_time = float(params.get("exposure_time", 1.0))
            result = builder.build_dark(
                count=count,
                exposure_time=exposure_time,
                gain=gain,
                binning=binning,
                cancel_event=self._cancel_event,
                on_progress=self._on_progress,
            )
            if result is None:
                self.logger.warning("Dark capture returned no frames (cancelled?)")
        elif frame_type == "flat":
            exposure_time = float(params.get("exposure_time", 1.0))
            filter_name = str(params.get("filter_name", ""))
            filter_position = params.get("filter_position")

            if filter_position is not None:
                filter_position = int(filter_position)
                if not filter_name:
                    fdata = self.hardware_adapter.filter_map.get(filter_position, {})
                    filter_name = fdata.get("name", f"Filter {filter_position}")

                with self._lock:
                    self._progress = {"running": True, "status": f"Moving filter wheel to {filter_name}..."}
                if not self.hardware_adapter.set_filter(filter_position):
                    self.logger.error("Failed to set filter %s (position %d)", filter_name, filter_position)
                    return

            result = builder.build_flat(
                count=count,
                exposure_time=exposure_time,
                filter_name=filter_name,
                gain=gain,
                binning=binning,
                cancel_event=self._cancel_event,
                on_progress=self._on_progress,
            )
            if result is None:
                self.logger.warning("Flat capture completed but master was rejected by quality validation")
        elif frame_type == "interleaved_flat":
            raw_filters = params.get("filters", [])
            filters = [FilterSlot(**f) if isinstance(f, dict) else f for f in raw_filters]
            if not filters:
                self.logger.error("Interleaved flat job has no filters")
                return
            initial_exposure = float(params.get("initial_exposure", 1.0))
            builder.build_interleaved_flats(
                filters=filters,
                set_filter=self.hardware_adapter.set_filter,
                count=count,
                initial_exposure=initial_exposure,
                gain=gain,
                binning=binning,
                cancel_event=self._cancel_event,
                on_progress=self._on_progress,
            )
        else:
            self.logger.error("Unknown calibration frame type: %s", frame_type)
            return
