"""Autofocus management: scheduling, target resolution, and execution.

Also tracks ongoing focus health (HFR history from the imaging pipeline)
to support at-a-glance monitoring and future adaptive autofocus (#203).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from citrasense.constants import AUTOFOCUS_TARGET_PRESETS
from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrasense.logging.sensor_logger import SensorLoggerAdapter
from citrasense.preview_bus import PreviewBus

if TYPE_CHECKING:
    import numpy as np

    from citrasense.acquisition.base_work_queue import BaseWorkQueue
    from citrasense.location.location_service import LocationService
    from citrasense.settings.citrasense_settings import CitraSenseSettings, SensorConfig


class AutofocusManager:
    """Manages autofocus requests, scheduling, and execution.

    Owns the autofocus request flag and lock, determines when scheduled
    autofocus should run, resolves the target star from settings, and
    executes the routine via the hardware adapter.
    """

    def __init__(
        self,
        logger: logging.Logger | SensorLoggerAdapter,
        hardware_adapter: AbstractAstroHardwareAdapter,
        settings: CitraSenseSettings,
        imaging_queue: BaseWorkQueue | None = None,
        location_service: LocationService | None = None,
        preview_bus: PreviewBus | None = None,
        on_toast: Callable[[str, str, str | None], None] | None = None,
    ):
        self.logger = logger.getChild(type(self).__name__)
        self.hardware_adapter = hardware_adapter
        self.settings = settings
        self.imaging_queue = imaging_queue
        self.location_service = location_service
        self._preview_bus = preview_bus
        self._sensor_id: str = ""
        self._sensor_config: SensorConfig | None = None
        self.on_toast = on_toast
        self._requested = False
        self._running = False
        self._progress = ""
        self._last_result = ""
        self._current_af_filter = ""
        self._points: list[tuple[int, float, str]] = []
        self._filter_results: list[dict[str, str | int | float]] = []
        self._hfr_history: deque[tuple[float, int, str]] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    @property
    def _af_settings(self) -> SensorConfig | None:
        """Per-sensor autofocus settings (SensorConfig)."""
        return self._sensor_config

    def request(self) -> bool:
        """Request autofocus to run at next safe point between tasks."""
        with self._lock:
            self._requested = True
            self.logger.info("Autofocus requested - will run between tasks")
            return True

    def cancel(self) -> bool:
        """Cancel autofocus whether it is queued or actively running.

        Returns:
            True if something was cancelled, False if nothing to cancel.
        """
        with self._lock:
            was_requested = self._requested
            is_running = self._running
            self._requested = False

        if is_running:
            self._cancel_event.set()
            self.logger.info("Autofocus cancellation requested (run in progress)")
            return True
        if was_requested:
            self.logger.info("Autofocus request cancelled")
            return True
        return False

    def is_requested(self) -> bool:
        """Check if autofocus is currently requested/queued."""
        with self._lock:
            return self._requested

    def is_running(self) -> bool:
        """Check if autofocus is currently executing."""
        with self._lock:
            return self._running

    @property
    def progress(self) -> str:
        """Current autofocus progress description (empty if not running)."""
        with self._lock:
            return self._progress

    def _set_progress(self, msg: str) -> None:
        with self._lock:
            self._progress = msg

    def _add_point(self, position: int, hfr: float) -> None:
        """Record a V-curve sample tagged with the current filter (thread-safe)."""
        with self._lock:
            self._points.append((position, hfr, self._current_af_filter))

    def _on_filter_start(self, filter_name: str) -> None:
        """Called by adapters at the start of each per-filter AF run.

        Snapshots the previous filter's result and records the new filter name
        for point tagging.  Points are kept across filters so the chart shows
        all V-curves together.
        """
        with self._lock:
            self._snapshot_current_filter_result()
            self._current_af_filter = filter_name

    def _snapshot_current_filter_result(self) -> None:
        """Snapshot the best point for the current filter into _filter_results.

        Must be called while holding self._lock.
        """
        if not self._current_af_filter or not self._points:
            return
        filter_pts = [p for p in self._points if p[2] == self._current_af_filter]
        if not filter_pts:
            return
        best = min(filter_pts, key=lambda p: p[1])
        self._filter_results.append(
            {
                "filter": self._current_af_filter,
                "position": best[0],
                "hfr": round(best[1], 2),
            }
        )

    def _reconcile_results_with_adapter(self) -> None:
        """Update filter results with the actual positions from the adapter's filter map.

        run_autofocus() returns a curve-fitted position that may differ from any
        measured V-curve point.  The adapter stores the fitted value in its
        filter_map.  This method patches each snapshot entry to use that
        authoritative position while keeping the measured HFR.

        Must be called while holding self._lock, after do_autofocus() returns.
        """
        if not self._filter_results:
            return
        filter_map = self.hardware_adapter.filter_map
        if not filter_map:
            return
        name_to_pos: dict[str, int] = {}
        for fdata in filter_map.values():
            name = fdata.get("name", "")
            pos = fdata.get("focus_position")
            if name and pos is not None:
                name_to_pos[name] = int(pos)
        for result in self._filter_results:
            actual_pos = name_to_pos.get(str(result["filter"]))
            if actual_pos is not None:
                result["position"] = actual_pos

    def _update_hfr_baseline(self) -> None:
        """Persist the best HFR from this AF run as the per-adapter monitoring baseline.

        Stored in adapter_settings so each adapter gets its own baseline.
        Must be called while holding self._lock.
        """
        if not self._filter_results or not self.settings:
            return
        hfr_values = [float(r["hfr"]) for r in self._filter_results if isinstance(r.get("hfr"), (int, float))]
        if not hfr_values:
            return
        baseline = round(min(hfr_values), 2)
        if self._sensor_config:
            self._sensor_config.adapter_settings["hfr_baseline"] = baseline
        self.logger.info(f"HFR baseline updated to {baseline:.2f} px")

    def clear_points(self) -> None:
        """Clear V-curve points (thread-safe)."""
        with self._lock:
            self._points.clear()

    @property
    def points(self) -> list[tuple[int, float, str]]:
        """Copy of the current autofocus V-curve points as (pos, hfr, filter)."""
        with self._lock:
            return list(self._points)

    @property
    def filter_results(self) -> list[dict[str, str | int | float]]:
        """Per-filter autofocus results from the last run."""
        with self._lock:
            return list(self._filter_results)

    def record_hfr(self, value: float, filter_name: str = "") -> None:
        """Record an HFR measurement from the imaging pipeline (thread-safe)."""
        with self._lock:
            self._hfr_history.append((value, int(time.time()), filter_name))

    @property
    def hfr_history(self) -> list[tuple[float, int, str]]:
        """Copy of the rolling HFR history as (hfr, unix_ts, filter) tuples."""
        with self._lock:
            return list(self._hfr_history)

    @property
    def last_result(self) -> str:
        """Summary of the last autofocus run (empty if none yet)."""
        with self._lock:
            return self._last_result

    _HFR_REFOCUS_COOLDOWN_SECONDS = 600  # 10 min — don't re-trigger immediately after an AF run

    def _should_trigger_hfr_refocus(self) -> bool:
        """Check if recent imaging HFR has degraded enough to warrant a refocus.

        Must be called while holding self._lock (reads _hfr_history).
        """
        afs = self._af_settings
        if not afs or not afs.autofocus_on_hfr_increase_enabled:
            return False
        baseline = self._sensor_config.adapter_settings.get("hfr_baseline") if self._sensor_config else None
        if baseline is None or baseline <= 0:
            return False
        if not self.hardware_adapter.supports_autofocus():
            return False

        window = afs.autofocus_hfr_sample_window
        if len(self._hfr_history) < window:
            return False

        last_ts = afs.last_autofocus_timestamp
        if last_ts is not None and (time.time() - last_ts) < self._HFR_REFOCUS_COOLDOWN_SECONDS:
            return False

        history = list(self._hfr_history)
        current_filter = history[-1][2] if history else ""
        same_filter = [h[0] for h in history if h[2] == current_filter]
        recent = same_filter[-window:]
        if len(recent) < window:
            return False
        median_hfr = statistics.median(recent)
        threshold = baseline * (1 + afs.autofocus_hfr_increase_percent / 100)
        if median_hfr > threshold:
            self.logger.info(
                f"HFR degradation detected: median {median_hfr:.2f} > threshold {threshold:.2f} "
                f"(baseline {baseline:.2f} + {afs.autofocus_hfr_increase_percent}%)"
            )
            return True
        return False

    def check_and_execute(self) -> bool:
        """Check if autofocus should run (manual or scheduled) and execute if so.

        Call this between tasks in the runner loop. Returns True if autofocus ran.
        Waits for the imaging queue to drain before starting so we don't slew
        mid-exposure.
        """
        with self._lock:
            should_run = self._requested
            if should_run:
                self._requested = False
            elif self._should_run_scheduled():
                should_run = True
                self._requested = False
            elif self._should_trigger_hfr_refocus():
                should_run = True

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Autofocus deferred - waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        self._execute()
        return True

    def _should_run_scheduled(self) -> bool:
        """Check if scheduled autofocus should run based on settings."""
        afs = self._af_settings
        if not afs:
            return False

        if not afs.scheduled_autofocus_enabled:
            return False

        if not self.hardware_adapter.supports_autofocus():
            return False

        mode = afs.autofocus_schedule_mode
        if mode == "after_sunset":
            return self._should_run_after_sunset()
        return self._should_run_interval()

    def _should_run_interval(self) -> bool:
        """Interval mode: trigger when elapsed time exceeds the configured interval."""
        afs = self._af_settings
        assert afs is not None
        interval_minutes = afs.autofocus_interval_minutes
        last_timestamp = afs.last_autofocus_timestamp

        if last_timestamp is None:
            return True

        elapsed_minutes = (int(time.time()) - last_timestamp) / 60
        return elapsed_minutes >= interval_minutes

    def _should_run_after_sunset(self) -> bool:
        """After-sunset mode: trigger once per night at sunset + offset."""
        location = self._get_location()
        if location is None:
            return False

        lat, lon = location
        trigger_time = self._compute_sunset_trigger_time(lat, lon)
        if trigger_time is None:
            return False

        now_utc = datetime.now(timezone.utc)
        if now_utc < trigger_time:
            return False

        afs = self._af_settings
        assert afs is not None
        last_ts = afs.last_autofocus_timestamp
        if last_ts is None:
            return True

        last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
        return last_dt < trigger_time

    def _get_location(self) -> tuple[float, float] | None:
        """Get observatory lat/lon from location service, or None if unavailable."""
        if self.location_service is None:
            return None
        loc = self.location_service.get_current_location()
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            return loc["latitude"], loc["longitude"]
        return None

    def _compute_sunset_trigger_time(self, latitude: float, longitude: float) -> datetime | None:
        """Return sunset + offset as a UTC datetime, or None on failure."""
        try:
            from citrasense.location.twilight import compute_sunset_utc

            sunset = compute_sunset_utc(latitude, longitude)
            if sunset is None:
                self.logger.warning("Could not compute sunset for after-sunset autofocus (polar day?)")
                return None
            afs = self._af_settings
            assert afs is not None
            offset = afs.autofocus_after_sunset_offset_minutes
            return sunset + timedelta(minutes=offset)
        except Exception as e:
            self.logger.warning(f"Failed to compute sunset time: {e}")
            return None

    def get_next_autofocus_minutes(self) -> int | None:
        """Compute minutes until next scheduled autofocus, or None if not scheduled.

        Used by the web status endpoint to display countdown.
        """
        afs = self._af_settings
        if not afs or not afs.scheduled_autofocus_enabled:
            return None
        if not self.hardware_adapter.supports_autofocus():
            return None

        mode = afs.autofocus_schedule_mode
        if mode == "after_sunset":
            return self._next_minutes_after_sunset()
        return self._next_minutes_interval()

    def _next_minutes_interval(self) -> int:
        afs = self._af_settings
        assert afs is not None
        last_ts = afs.last_autofocus_timestamp
        interval = afs.autofocus_interval_minutes
        if last_ts is None:
            return 0
        elapsed = (int(time.time()) - last_ts) / 60
        return max(0, int(interval - elapsed))

    def _next_minutes_after_sunset(self) -> int | None:
        location = self._get_location()
        if location is None:
            return None
        lat, lon = location
        trigger_time = self._compute_sunset_trigger_time(lat, lon)
        if trigger_time is None:
            return None
        now_utc = datetime.now(timezone.utc)
        if now_utc >= trigger_time:
            afs = self._af_settings
            assert afs is not None
            last_ts = afs.last_autofocus_timestamp
            if last_ts is not None:
                last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
                if last_dt >= trigger_time:
                    return None
            return 0
        remaining = (trigger_time - now_utc).total_seconds() / 60
        return max(0, int(remaining))

    def _resolve_target(self) -> tuple[float | None, float | None]:
        """Resolve autofocus target RA/Dec from settings (preset or custom)."""
        afs = self._af_settings
        if not afs:
            return None, None

        preset_key = afs.autofocus_target_preset or "mirach"

        if preset_key == "current":
            self.logger.info("Autofocus target: current position (no slew)")
            return None, None

        if preset_key == "custom":
            ra = afs.autofocus_target_custom_ra
            dec = afs.autofocus_target_custom_dec
            if ra is not None and dec is not None:
                self.logger.info(f"Autofocus target: custom (RA={ra:.4f}, Dec={dec:.4f})")
                return ra, dec
            self.logger.warning("Custom autofocus target missing RA/Dec, falling back to Mirach")
            preset_key = "mirach"

        preset = AUTOFOCUS_TARGET_PRESETS.get(preset_key)
        if not preset:
            self.logger.warning(f"Unknown autofocus preset '{preset_key}', falling back to Mirach")
            preset = AUTOFOCUS_TARGET_PRESETS["mirach"]

        self.logger.info(f"Autofocus target: {preset['name']} ({preset['designation']})")
        return preset["ra"], preset["dec"]

    def _on_af_image(self, image: np.ndarray) -> None:
        """Convert an autofocus sweep frame to JPEG and push to the preview bus."""
        if not self._preview_bus:
            return
        try:
            from citrasense.preview_bus import array_to_jpeg_data_url

            data_url = array_to_jpeg_data_url(image)
            self._preview_bus.push(data_url, "autofocus", sensor_id=self._sensor_id)
        except Exception as e:
            self.logger.debug(f"Failed to push AF preview frame: {e}")

    def _execute(self) -> None:
        """Execute autofocus routine and update timestamp on both success and failure."""
        self._cancel_event.clear()
        with self._lock:
            self._running = True
            self._progress = "Starting..."
            self._last_result = ""
            self._points.clear()
            self._filter_results.clear()
            self._current_af_filter = ""
        try:
            target_ra, target_dec = self._resolve_target()
            self.logger.info("Starting autofocus routine...")
            self.hardware_adapter.do_autofocus(
                target_ra=target_ra,
                target_dec=target_dec,
                on_progress=self._set_progress,
                cancel_event=self._cancel_event,
                on_point=self._add_point,
                on_filter_start=self._on_filter_start,
                on_image=self._on_af_image,
            )

            with self._lock:
                self._snapshot_current_filter_result()
                self._reconcile_results_with_adapter()
                self._update_hfr_baseline()

            if self.hardware_adapter.supports_filter_management():
                try:
                    filter_config = self.hardware_adapter.get_filter_config()
                    if filter_config and self._sensor_config:
                        self._sensor_config.adapter_settings["filters"] = filter_config
                        self.logger.info(f"Saved filter configuration with {len(filter_config)} filters")
                except Exception as e:
                    self.logger.warning(f"Failed to save filter configuration after autofocus: {e}")

            self.logger.info("Autofocus routine completed successfully")
            with self._lock:
                if self._filter_results:
                    parts = [f"{r['filter']}: {r['position']} (HFR {r['hfr']})" for r in self._filter_results]
                    self._last_result = " | ".join(parts)
                else:
                    self._last_result = self._progress
            if self.on_toast:
                prefix = f"[{self._sensor_id}] " if self._sensor_id else ""
                toast_id = f"autofocus-result:{self._sensor_id}" if self._sensor_id else "autofocus-result"
                self.on_toast(f"{prefix}Autofocus completed successfully", "success", toast_id)
        except Exception as e:
            self.logger.error(f"Autofocus failed: {e!s}", exc_info=True)
            with self._lock:
                self._last_result = f"Failed: {e!s}"
            if self.on_toast:
                prefix = f"[{self._sensor_id}] " if self._sensor_id else ""
                toast_id = f"autofocus-result:{self._sensor_id}" if self._sensor_id else "autofocus-result"
                self.on_toast(f"{prefix}Autofocus failed: {e!s}", "danger", toast_id)
        finally:
            with self._lock:
                self._running = False
                self._progress = ""
            ts = int(time.time())
            if self._sensor_config:
                self._sensor_config.last_autofocus_timestamp = ts
            if self.settings:
                self.settings.save()
