"""Calibration capture management: operator-requested calibration with imaging queue gate.

Follows the same request/check_and_execute pattern as AutofocusManager
and HomingManager.  Calibration defers tasks while active (like homing)
because capture sessions can take minutes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from citrasense.calibration import FilterSlot
from citrasense.calibration.calibration_library import CalibrationLibrary
from citrasense.calibration.flat_capture_backend import FlatCaptureBackend
from citrasense.calibration.master_builder import MasterBuilder
from citrasense.hardware.devices.camera.abstract_camera import CalibrationProfile
from citrasense.location.twilight import compute_twilight
from citrasense.logging.sensor_logger import SensorLoggerAdapter

if TYPE_CHECKING:
    from citrasense.acquisition.base_work_queue import BaseWorkQueue
    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.settings.citrasense_settings import CitraSenseSettings

LocationProvider = Callable[[], dict[str, float] | None]


class CalibrationManager:
    """Manages operator-requested calibration capture with safe task gating.

    The web UI posts a capture request via ``request(params)``.  On the
    next runner iteration, ``check_and_execute()`` picks it up, waits for
    the imaging queue to drain, then delegates to :class:`MasterBuilder`.
    """

    def __init__(
        self,
        logger: logging.Logger | SensorLoggerAdapter,
        hardware_adapter: AbstractAstroHardwareAdapter,
        library: CalibrationLibrary,
        imaging_queue: BaseWorkQueue | None = None,
        flat_backend: FlatCaptureBackend | None = None,
        settings: CitraSenseSettings | None = None,
        sensor_id: str | None = None,
        location_provider: LocationProvider | None = None,
    ) -> None:
        self.logger = logger.getChild(type(self).__name__)
        self.hardware_adapter = hardware_adapter
        self.library = library
        self.imaging_queue = imaging_queue
        self._flat_backend = flat_backend
        self._settings = settings
        self._sensor_id = sensor_id
        self._location_provider = location_provider

        self._requested = False
        self._running = False
        self._capture_params: dict[str, Any] = {}
        self._progress: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

        self._job_queue: list[dict[str, Any]] = []
        self._batch_total: int = 0
        self._batch_index: int = 0

        # Scheduled-flats dedup: remember the flat_window.start we already
        # served to avoid re-firing within the same dusk/dawn window.
        self._last_served_window_start: str | None = None

    @property
    def flat_backend(self) -> FlatCaptureBackend | None:
        return self._flat_backend

    def supports_frame_type(self, frame_type: str) -> bool:
        """Whether this manager's wiring can satisfy a given frame type.

        Bias and dark always require a direct camera (there is no
        orchestrator shutter-closed path).  Flats work whenever any
        backend claims ``"flat"`` in its ``supported_frame_types``.
        """
        if frame_type == "flat":
            if self._flat_backend is not None:
                return "flat" in self._flat_backend.supported_frame_types
            return self.hardware_adapter.camera is not None
        if frame_type in ("bias", "dark", "interleaved_flat"):
            return self.hardware_adapter.camera is not None
        return False

    # ------------------------------------------------------------------
    # Public API (called from web endpoints)
    # ------------------------------------------------------------------

    def request(self, capture_params: dict[str, Any]) -> bool:
        """Queue a calibration capture job.

        Args:
            capture_params: Dict with keys like ``frame_type``, ``count``,
                ``exposure_time``, ``gain``, ``binning``, ``filter_name``.
        """
        frame_type = str(capture_params.get("frame_type", ""))
        if frame_type and not self.supports_frame_type(frame_type):
            self.logger.warning(
                "Calibration request rejected: frame_type %r not supported on this sensor",
                frame_type,
            )
            return False
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

        Returns False if calibration is already running or contains an
        unsupported frame type.
        """
        if not jobs:
            return False
        for j in jobs:
            ft = str(j.get("frame_type", ""))
            if ft and not self.supports_frame_type(ft):
                self.logger.warning("Calibration suite rejected: frame_type %r not supported", ft)
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
        calibration ran.  Handles both single jobs and batch suites,
        plus the scheduled flat-window auto-capture path when enabled.
        """
        # Scheduled auto-capture is a no-op unless the backend and settings
        # wiring is in place — kept before the manual-request branch so that
        # an opening flat window gets picked up promptly.
        with self._lock:
            already_running = self._running
            already_requested = self._requested
        if not already_running and not already_requested:
            self._maybe_auto_capture_flats()

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
            # Scheduled suites: persist completion timestamp so the next
            # tick does not re-fire for the same dusk/dawn window.
            if self._last_served_window_start is not None and not self._cancel_event.is_set():
                self.mark_flats_capture_complete()
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
        frame_type = params.get("frame_type", "bias")
        camera = self.hardware_adapter.camera

        if not self.supports_frame_type(frame_type):
            self.logger.error("Cannot run frame_type=%r: no backend/camera wired for this sensor", frame_type)
            return

        if camera is None and frame_type == "flat" and self._flat_backend is not None:
            self._execute_flat_via_backend(params)
            return

        if camera is None:
            self.logger.error("No direct camera available for calibration")
            return

        profile = camera.get_calibration_profile()
        if not profile.calibration_applicable:
            self.logger.error("Camera does not support CCD-style calibration")
            return

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

                # Direct-camera flats need the wheel moved before capture.
                # NINA trained flats pass filterId into /flats/trained-flat
                # and NINA drives the wheel itself — handled in the backend branch.
                if self._flat_backend is None:
                    with self._lock:
                        self._progress = {"running": True, "status": f"Moving filter wheel to {filter_name}..."}
                    if not self.hardware_adapter.set_filter(filter_position):
                        self.logger.error("Failed to set filter %s (position %d)", filter_name, filter_position)
                        return

            result = builder.build_flat(
                count=count,
                exposure_time=exposure_time,
                filter_name=filter_name,
                filter_position=filter_position,
                gain=gain,
                binning=binning,
                cancel_event=self._cancel_event,
                on_progress=self._on_progress,
                flat_backend=self._flat_backend,
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

    def _profile_from_adapter_summary(self) -> CalibrationProfile | None:
        """Assemble a :class:`CalibrationProfile` from the adapter's upload-only summary.

        Used on the NINA-only flat path where we have no ``AbstractCamera``
        to query.  Returns ``None`` when the adapter does not expose
        ``get_calibration_profile_summary`` or returns an unusable dict.
        """
        fn = getattr(self.hardware_adapter, "get_calibration_profile_summary", None)
        if not callable(fn):
            return None
        try:
            raw = fn()
        except Exception as e:
            self.logger.error("get_calibration_profile_summary failed: %s", e)
            return None
        if not isinstance(raw, dict) or not raw.get("camera_id"):
            return None
        summary: dict[str, Any] = raw
        return CalibrationProfile(
            calibration_applicable=True,
            camera_id=str(summary["camera_id"]),
            model=str(summary.get("model") or summary["camera_id"]),
            has_mechanical_shutter=bool(summary.get("has_mechanical_shutter", False)),
            has_cooling=bool(summary.get("has_cooling", False)),
            current_gain=summary.get("current_gain"),
            current_binning=int(summary.get("current_binning") or 1),
            current_temperature=summary.get("current_temperature"),
            target_temperature=summary.get("target_temperature"),
            read_mode=str(summary.get("read_mode") or ""),
        )

    def _execute_flat_via_backend(self, params: dict[str, Any]) -> None:
        """Run a flat capture job via ``self._flat_backend`` with no direct camera.

        Builds a :class:`MasterBuilder` with ``camera=None`` (safe for
        the flat path once a backend is injected) and a profile derived
        from the adapter's ``get_calibration_profile_summary``.
        """
        profile = self._profile_from_adapter_summary()
        if profile is None:
            self.logger.error("Cannot execute flat via backend: no adapter profile summary")
            return

        builder = MasterBuilder(None, self.library, profile)

        count = int(params.get("count", 15))
        gain_param = params.get("gain")
        gain = int(gain_param) if gain_param is not None else (profile.current_gain or 0)
        binning = int(params.get("binning", profile.current_binning))
        exposure_time = float(params.get("exposure_time", 1.0))

        filter_name = str(params.get("filter_name", ""))
        filter_position = params.get("filter_position")
        if filter_position is not None:
            filter_position = int(filter_position)
            if not filter_name:
                fdata = self.hardware_adapter.filter_map.get(filter_position, {})
                filter_name = fdata.get("name", f"Filter {filter_position}")

        try:
            result = builder.build_flat(
                count=count,
                exposure_time=exposure_time,
                filter_name=filter_name,
                filter_position=filter_position,
                gain=gain,
                binning=binning,
                cancel_event=self._cancel_event,
                on_progress=self._on_progress,
                flat_backend=self._flat_backend,
            )
            if result is None:
                self.logger.warning("Flat capture via backend produced no master")
        except Exception as e:
            self.logger.error("Flat capture via backend failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Scheduled flat-window auto-capture
    # ------------------------------------------------------------------

    def _maybe_auto_capture_flats(self) -> bool:
        """If a flat window is open and scheduling is enabled, queue a flat suite.

        Returns ``True`` when an auto-capture was requested (so the caller
        can skip this tick), ``False`` otherwise.  Dedupes on the current
        window start so we only fire once per dusk/dawn.
        """
        if self._flat_backend is None:
            return False
        if self._settings is None or self._sensor_id is None:
            return False
        sc = self._settings.get_sensor_config(self._sensor_id)
        if sc is None or not getattr(sc, "auto_capture_flats_enabled", False):
            return False

        # Skip when the operator has tasks pending — their schedule wins.
        if self.imaging_queue and not self.imaging_queue.is_idle():
            return False

        if self._location_provider is None:
            return False
        location = self._location_provider()
        if not location:
            return False
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat is None or lon is None:
            return False

        try:
            twilight = compute_twilight(float(lat), float(lon))
        except Exception as e:
            self.logger.warning("Flat-window auto-capture skipped: twilight compute failed: %s", e)
            return False

        if not twilight.in_flat_window or twilight.flat_window is None:
            return False

        window_start = str(twilight.flat_window.start)
        last_served = getattr(sc, "last_flats_capture_iso", None)
        if last_served and str(last_served) >= window_start:
            # We already served this (or a later) window — wait for the next one.
            return False
        if self._last_served_window_start == window_start:
            return False

        filters: list[FilterSlot] = []
        if self.hardware_adapter.supports_filter_management():
            for pos, f in self.hardware_adapter.get_filter_config().items():
                if f.get("enabled", True) and f.get("name"):
                    filters.append(FilterSlot(position=int(pos), name=f["name"]))
        if not filters:
            return False

        flat_count = int(getattr(sc, "flat_frame_count", 15) or 15)
        binning = _profile_current_binning(self) or 1
        jobs: list[dict[str, Any]] = [
            {
                "frame_type": "flat",
                "count": flat_count,
                "binning": binning,
                "filter_position": slot.position,
                "filter_name": slot.name,
            }
            for slot in filters
        ]

        if not self.request_suite(jobs):
            return False

        self._last_served_window_start = window_start
        self.logger.info(
            "Auto-capture flats queued for window starting %s (%d filters)",
            window_start,
            len(filters),
        )
        return True

    def mark_flats_capture_complete(self, iso_timestamp: str | None = None) -> None:
        """Persist ``last_flats_capture_iso`` to the sensor config.

        Called after a successful scheduled flats suite so the next
        ``check_and_execute`` pass does not re-fire for the same window.
        """
        if self._settings is None or self._sensor_id is None:
            return
        iso = iso_timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            self._settings.update_and_save({"sensors": [{"id": self._sensor_id, "last_flats_capture_iso": iso}]})
        except Exception as e:
            self.logger.warning("Failed to persist last_flats_capture_iso: %s", e)


def _profile_current_binning(mgr: CalibrationManager) -> int | None:
    """Return the adapter's current binning, prefering the direct camera profile.

    Keeps the scheduled-flats job construction resilient to adapters
    that do not expose a camera by falling back to the upload-only
    summary and finally to ``None`` (caller defaults to 1).
    """
    cam = mgr.hardware_adapter.camera
    if cam is not None:
        try:
            return int(cam.get_calibration_profile().current_binning or 1)
        except Exception:
            pass
    profile = mgr._profile_from_adapter_summary()
    if profile is not None:
        return int(profile.current_binning or 1)
    return None
