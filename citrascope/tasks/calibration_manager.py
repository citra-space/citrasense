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

from citrascope.calibration.calibration_library import CalibrationLibrary
from citrascope.calibration.master_builder import MasterBuilder

if TYPE_CHECKING:
    from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrascope.tasks.base_work_queue import BaseWorkQueue


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
        self.logger = logger
        self.hardware_adapter = hardware_adapter
        self.library = library
        self.imaging_queue = imaging_queue

        self._requested = False
        self._running = False
        self._capture_params: dict[str, Any] = {}
        self._progress: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

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

    def cancel(self) -> bool:
        """Cancel calibration whether queued or actively running."""
        with self._lock:
            was_requested = self._requested
            is_running = self._running
            self._requested = False

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
        calibration ran.
        """
        with self._lock:
            should_run = self._requested
            params = dict(self._capture_params) if should_run else {}
            if should_run:
                self._requested = False

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Calibration deferred — waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        self._execute(params)
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, frame_type: str, status: str) -> None:
        with self._lock:
            self._progress = {
                "running": True,
                "frame_type": frame_type,
                "current_frame": current,
                "total_frames": total,
                "status": status,
            }

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
        if frame_type in ("bias", "flat"):
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

    def _execute(self, params: dict[str, Any]) -> None:
        self._cancel_event.clear()
        with self._lock:
            self._running = True
            self._progress = {"running": True, "status": "Starting calibration..."}

        try:
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
                builder.build_bias(
                    count=count,
                    gain=gain,
                    binning=binning,
                    cancel_event=self._cancel_event,
                    on_progress=self._on_progress,
                )
            elif frame_type == "dark":
                exposure_time = float(params.get("exposure_time", 1.0))
                builder.build_dark(
                    count=count,
                    exposure_time=exposure_time,
                    gain=gain,
                    binning=binning,
                    cancel_event=self._cancel_event,
                    on_progress=self._on_progress,
                )
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

                builder.build_flat(
                    count=count,
                    exposure_time=exposure_time,
                    filter_name=filter_name,
                    gain=gain,
                    binning=binning,
                    cancel_event=self._cancel_event,
                    on_progress=self._on_progress,
                )
            else:
                self.logger.error("Unknown calibration frame type: %s", frame_type)
                return

            if self._cancel_event.is_set():
                self.logger.info("Calibration capture was cancelled")
            else:
                self.logger.info("Calibration capture completed: %s", frame_type)

        except Exception as e:
            self.logger.error("Calibration capture failed: %s", e, exc_info=True)
        finally:
            with self._lock:
                self._running = False
                self._progress = {}
