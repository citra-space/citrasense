"""Mount alignment and pointing calibration management.

Provides two modes:
- **Quick align** (``request`` / ``_execute``): single plate solve + sync at
  the current position.  The existing "Align Now" button.
- **Full calibration** (``request_calibration`` / ``_execute_calibration``):
  bootstrap sync → sky grid walk → fit pointing model.  Triggered by the
  "Calibrate" button or the NIGHT_STARTUP sequence.

Follows the same request/check_and_execute pattern as AutofocusManager.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from citrascope.preview_bus import PreviewBus

if TYPE_CHECKING:
    from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrascope.hardware.devices.mount.altaz_pointing_model import AltAzPointingModel
    from citrascope.settings.citrascope_settings import CitraScopeSettings
    from citrascope.tasks.base_work_queue import BaseWorkQueue


class AlignmentManager:
    """Manages on-demand alignment and full pointing model calibration.

    Takes a short exposure at the mount's current position, plate-solves it,
    and syncs the mount.  Also drives the multi-point calibration procedure
    that builds an ``AltAzPointingModel``.

    Follows the same request/check_and_execute pattern as AutofocusManager.
    """

    def __init__(
        self,
        logger: logging.Logger,
        hardware_adapter: AbstractAstroHardwareAdapter,
        settings: CitraScopeSettings,
        imaging_queue: BaseWorkQueue | None = None,
        safety_monitor=None,
        location_service=None,
        preview_bus: PreviewBus | None = None,
    ):
        self.logger = logger.getChild(type(self).__name__)
        self.hardware_adapter = hardware_adapter
        self.settings = settings
        self.imaging_queue = imaging_queue
        self.safety_monitor = safety_monitor
        self.location_service = location_service
        self._preview_bus = preview_bus

        # Quick-align state
        self._requested = False
        self._running = False
        self._progress = ""
        self._lock = threading.Lock()

        # Calibration state
        self._calibration_requested = False
        self._calibrating = False
        self._calibration_cancel = threading.Event()
        self._calibration_step: int = 0
        self._calibration_total: int = 0
        self._pointing_model: AltAzPointingModel | None = None

    # ------------------------------------------------------------------
    # Pointing model reference
    # ------------------------------------------------------------------

    def set_pointing_model(self, model: AltAzPointingModel) -> None:
        """Wire the pointing model (owned by DirectHardwareAdapter)."""
        self._pointing_model = model

    # ------------------------------------------------------------------
    # Quick align (existing "Align Now")
    # ------------------------------------------------------------------

    def request(self) -> bool:
        """Request alignment to run at next safe point between tasks."""
        with self._lock:
            self._requested = True
            self.logger.info("Alignment requested — will run between tasks")
            return True

    def cancel(self) -> bool:
        """Cancel pending alignment request.

        Returns:
            True if alignment was cancelled, False if nothing to cancel.
        """
        with self._lock:
            was_requested = self._requested
            self._requested = False
            if was_requested:
                self.logger.info("Alignment request cancelled")
            return was_requested

    def is_requested(self) -> bool:
        with self._lock:
            return self._requested

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def progress(self) -> str:
        with self._lock:
            return self._progress

    def _set_progress(self, msg: str) -> None:
        with self._lock:
            self._progress = msg

    def check_and_execute(self) -> bool:
        """Check if alignment or calibration should run and execute if so.

        Call this between tasks in the runner loop.  Returns True if work ran.
        Waits for the imaging queue to drain before starting.
        """
        # Calibration takes priority over quick-align
        with self._lock:
            should_calibrate = self._calibration_requested
            if should_calibrate:
                self._calibration_requested = False

        if should_calibrate:
            if self.imaging_queue and not self.imaging_queue.is_idle():
                self.logger.info("Calibration deferred — waiting for imaging queue to drain")
                with self._lock:
                    self._calibration_requested = True
                return False
            self._execute_calibration()
            return True

        with self._lock:
            should_run = self._requested
            if should_run:
                self._requested = False

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Alignment deferred — waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        self._execute()
        return True

    def _get_exposure_seconds(self) -> float:
        if self.settings:
            return self.settings.alignment_exposure_seconds
        return 2.0

    def _execute(self) -> None:
        """Execute alignment: take image → plate solve → sync mount."""
        with self._lock:
            self._running = True
            self._progress = "Starting alignment..."
        try:
            telescope_record: dict[str, Any] | None = self.hardware_adapter.telescope_record
            if not telescope_record:
                self.logger.error("Cannot align — no telescope_record available (configure telescope in Citra)")
                return

            mount = self.hardware_adapter.mount
            if not self.hardware_adapter.is_camera_connected() or not mount:
                self.logger.error("Cannot align — camera and mount are both required")
                return

            from citrascope.processors.builtin.plate_solver_processor import PlateSolverProcessor

            if self.safety_monitor and not self.safety_monitor.is_action_safe("capture"):
                self.logger.warning("Alignment aborted — safety monitor blocked capture")
                return

            exposure_s = self._get_exposure_seconds()
            self._set_progress(f"Exposing ({exposure_s:.0f}s)...")
            self.logger.info(f"Alignment: taking {exposure_s:.0f}s exposure...")

            try:
                image_path = self.hardware_adapter.take_image("alignment", exposure_s)
            except Exception as exc:
                self.logger.error(f"Alignment exposure failed: {exc}")
                return

            if self._preview_bus:
                try:
                    self._preview_bus.push_file(image_path, "alignment")
                except Exception as e:
                    self.logger.debug(f"Failed to push alignment preview: {e}")
            self._set_progress("Plate solving...")
            self.logger.info(f"Alignment: plate solving {image_path}...")
            result = PlateSolverProcessor.solve(
                Path(image_path), telescope_record, location_service=self.location_service
            )

            if result is None:
                self.logger.error("Alignment failed — plate solve returned no solution")
                return

            solved_ra, solved_dec = result
            self.logger.info(f"Alignment: solved RA={solved_ra:.4f}°, Dec={solved_dec:.4f}°")

            # Health check against pointing model if trained
            if self._pointing_model and self._pointing_model.is_trained and self.location_service:
                location = self.location_service.get_current_location()
                if location:
                    current_ra, current_dec = mount.get_radec()
                    predicted = self._pointing_model.predict_error(
                        current_ra, current_dec, location["latitude"], location["longitude"]
                    )
                    observed_error = self.hardware_adapter.angular_distance(
                        current_ra, current_dec, solved_ra, solved_dec
                    )
                    if predicted > 0 and observed_error > predicted * 3.0:
                        self.logger.warning(
                            "Align Now: observed error %.4f° far exceeds model prediction %.4f° "
                            "— pointing model may be stale, consider recalibrating",
                            observed_error,
                            predicted,
                        )

            current_ra, current_dec = mount.get_radec()
            residual = self.hardware_adapter.angular_distance(solved_ra, solved_dec, current_ra, current_dec)

            if self._pointing_model and self._pointing_model.is_trained:
                self._pointing_model.record_verification_residual(residual)
                self._set_progress("Verified (model active)")
                self.logger.info(
                    "Alignment verified (model active, no sync): residual %.4f° "
                    "(RA offset %+.4f°, Dec offset %+.4f°)",
                    residual,
                    solved_ra - current_ra,
                    solved_dec - current_dec,
                )
            else:
                self._set_progress("Syncing mount...")
                mount.sync_to_radec(solved_ra, solved_dec)
                self.logger.info(
                    "Alignment successful: mount synced " "(offset RA=%+.4f°, Dec=%+.4f°)",
                    solved_ra - current_ra,
                    solved_dec - current_dec,
                )

        except Exception as e:
            self.logger.error(f"Alignment failed: {e!s}", exc_info=True)
        finally:
            with self._lock:
                self._running = False
                self._progress = ""
            if self.settings:
                self.settings.last_alignment_timestamp = int(time.time())
                self.settings.save()

    # ------------------------------------------------------------------
    # Full pointing calibration
    # ------------------------------------------------------------------

    def request_calibration(self) -> bool:
        """Request a full pointing model calibration run."""
        with self._lock:
            if self._calibrating:
                self.logger.warning("Calibration already running — ignoring request")
                return False
            self._calibration_requested = True
            self._calibration_cancel.clear()
            self.logger.info("Pointing calibration requested — will run between tasks")
            return True

    def cancel_calibration(self) -> None:
        """Cancel an in-progress or pending calibration."""
        with self._lock:
            self._calibration_requested = False
        self._calibration_cancel.set()
        self.logger.info("Pointing calibration cancel requested")

    def is_calibrating(self) -> bool:
        with self._lock:
            return self._calibrating

    @property
    def calibration_progress(self) -> dict[str, int] | None:
        """Return structured calibration progress, or None if not calibrating."""
        with self._lock:
            if not self._calibrating:
                return None
            return {"step": self._calibration_step, "total": self._calibration_total}

    def _execute_calibration(self) -> None:
        """Execute the full pointing calibration: bootstrap sync → grid walk → fit.

        Snapshots the current model before starting so a failed or cancelled
        calibration can roll back rather than leaving the operator with no model.
        """
        with self._lock:
            self._calibrating = True
            self._progress = "Starting pointing calibration..."

        previous_state: dict[str, Any] | None = None
        had_working_model = False
        if self._pointing_model:
            previous_state = self._pointing_model.to_dict()
            had_working_model = self._pointing_model.is_active

        try:
            self._do_calibration()
        except Exception as e:
            self.logger.error(f"Pointing calibration failed: {e!s}", exc_info=True)
        finally:
            if had_working_model and self._pointing_model and not self._pointing_model.is_active:
                assert previous_state is not None
                self.logger.warning("Calibration did not produce a working model — restoring previous model")
                self._pointing_model.restore_from_dict(previous_state)

            with self._lock:
                self._calibrating = False
                self._calibration_step = 0
                self._calibration_total = 0
                self._progress = ""
            self._calibration_cancel.clear()

    def _do_calibration(self) -> None:
        """Inner calibration logic — separated for clean exception handling."""
        from citrascope.hardware.devices.mount.altaz_pointing_model import generate_calibration_grid

        if not self._pointing_model:
            self.logger.error("Cannot calibrate — no pointing model available")
            return

        telescope_record: dict[str, Any] | None = self.hardware_adapter.telescope_record
        if not telescope_record:
            self.logger.error("Cannot calibrate — no telescope_record available")
            return

        mount = self.hardware_adapter.mount
        if not self.hardware_adapter.is_camera_connected() or not mount:
            self.logger.error("Cannot calibrate — camera and mount are both required")
            return

        if not self.location_service:
            self.logger.error("Cannot calibrate — no location service available")
            return

        location = self.location_service.get_current_location()
        if not location:
            self.logger.error("Cannot calibrate — no location fix available")
            return

        site_lat = location["latitude"]
        site_lon = location["longitude"]

        if self.safety_monitor:
            if not self.safety_monitor.is_action_safe("capture"):
                self.logger.warning("Calibration aborted — safety monitor blocked capture")
                return
            if not self.safety_monitor.is_action_safe("slew"):
                self.logger.warning("Calibration aborted — safety monitor blocked slew")
                return

        # Reset model for fresh calibration
        self._pointing_model.reset()

        # ---- Step 0: Unwind cable wrap ----
        self._unwind_before_calibration()

        if self._calibration_cancel.is_set():
            self.logger.info("Calibration cancelled during unwind")
            return

        # ---- Step 1: Bootstrap sync ----
        self._set_progress("Bootstrap alignment...")
        self.logger.info("Calibration: bootstrap sync — plate solving at current position")
        bootstrap_result = self._plate_solve_at_current_position(telescope_record)
        if bootstrap_result is None:
            self.logger.error("Calibration aborted — bootstrap plate solve failed. Check focus and sky conditions.")
            return

        solved_ra, solved_dec = bootstrap_result
        current_ra, current_dec = mount.get_radec()
        mount.sync_to_radec(solved_ra, solved_dec)
        self.logger.info(
            "Bootstrap sync complete: offset RA=%+.4f° Dec=%+.4f°",
            solved_ra - current_ra,
            solved_dec - current_dec,
        )

        if self._calibration_cancel.is_set():
            self.logger.info("Calibration cancelled after bootstrap")
            return

        # ---- Generate sky grid ----
        self._set_progress("Generating sky grid...")
        cable_cumulative = self._get_cable_wrap_cumulative()

        mount_az = mount.get_azimuth()
        current_az = mount_az if mount_az is not None else 180.0

        if hasattr(mount, "get_limits"):
            h_limit, o_limit = mount.get_limits()
        else:
            h_limit, o_limit = None, None
        horizon_limit = float(h_limit) if h_limit is not None else 15.0
        overhead_limit = float(o_limit) if o_limit is not None else 89.0

        targets = generate_calibration_grid(
            current_az_deg=current_az,
            cable_wrap_cumulative_deg=cable_cumulative,
            horizon_limit_deg=horizon_limit,
            overhead_limit_deg=overhead_limit,
            lat_deg=site_lat,
            lon_deg=site_lon,
            n_points=15,
        )

        if not targets:
            self.logger.error("Calibration aborted — grid generation produced no targets")
            return

        self.logger.info("Calibration grid: %d targets generated", len(targets))

        # ---- Grid walk ----
        successful = 0
        with self._lock:
            self._calibration_total = len(targets)
            self._calibration_step = 0
        for i, (target_ra, target_dec) in enumerate(targets):
            if self._calibration_cancel.is_set():
                self.logger.info("Calibration cancelled at point %d/%d", i + 1, len(targets))
                break

            with self._lock:
                self._calibration_step = i + 1
            self._set_progress(f"Calibrating {i + 1}/{len(targets)}...")
            self.logger.info(
                "Calibration point %d/%d: slewing to RA=%.4f° Dec=%.4f°",
                i + 1,
                len(targets),
                target_ra,
                target_dec,
            )

            # Raw slew — bypasses point_telescope() because we're building
            # the model (no correction to apply yet).  Safety is checked here.
            if self.safety_monitor and not self.safety_monitor.is_action_safe("slew"):
                self.logger.warning("Calibration: slew blocked by safety monitor at point %d — skipping", i + 1)
                continue

            try:
                if not mount.slew_to_radec(target_ra, target_dec):
                    self.logger.warning("Calibration: slew failed for point %d — skipping", i + 1)
                    continue

                timeout = 300
                start = time.time()
                while mount.is_slewing():
                    if time.time() - start > timeout:
                        mount.abort_slew()
                        self.logger.warning("Calibration: slew timeout for point %d — skipping", i + 1)
                        break
                    if self._calibration_cancel.is_set():
                        mount.abort_slew()
                        break
                    time.sleep(0.5)

                if self._calibration_cancel.is_set():
                    break
            except Exception as exc:
                self.logger.warning("Calibration: slew error for point %d: %s — skipping", i + 1, exc)
                continue

            # Read mount-reported position before plate solve
            mount_ra, mount_dec = mount.get_radec()

            # Plate solve
            solve_result = self._plate_solve_at_current_position(telescope_record)
            if solve_result is None:
                self.logger.warning("Calibration: plate solve failed for point %d — skipping", i + 1)
                continue

            solved_ra, solved_dec = solve_result

            # Record point (NO sync)
            self._pointing_model.add_point(mount_ra, mount_dec, solved_ra, solved_dec, site_lat, site_lon)
            successful += 1

        # ---- Fit model ----
        if successful >= 3:
            self._pointing_model.fit()
            status = self._pointing_model.status()
            self.logger.info(
                "Pointing calibration complete: %d/%d points, %s model, tilt=%.3f° toward %s, accuracy=%.4f°",
                successful,
                len(targets),
                status["state"],
                status["tilt_deg"],
                status["tilt_direction_label"],
                status["pointing_accuracy_deg"],
            )

            # ---- Verification: slew with correction, plate solve, compare ----
            if not self._calibration_cancel.is_set():
                self._unwind_if_needed("verification")
                self._verify_calibration(
                    mount, telescope_record, site_lat, site_lon, model_rms=status["pointing_accuracy_deg"]
                )
        else:
            self.logger.warning(
                "Calibration finished with only %d successful points — insufficient for model fit",
                successful,
            )

    # ------------------------------------------------------------------
    # Post-calibration verification
    # ------------------------------------------------------------------

    _VERIFY_COUNT = 2
    _VERIFY_FAIL_FACTOR = 2.0
    _VERIFY_FLOOR_DEG = 5.0 / 60.0

    def _verify_calibration(
        self,
        mount: Any,
        telescope_record: dict[str, Any],
        site_lat: float,
        site_lon: float,
        model_rms: float,
    ) -> None:
        """Slew to a few targets using the corrected path and plate-solve to
        verify the model actually reduces error.

        Generates synthetic targets at +/-90 degrees azimuth from the current
        mount position so one slew goes CW and one CCW, keeping cable wrap
        bounded.  Uses ``point_telescope()`` (the production correction path)
        so this exercises the full ``correct() → slew → solve`` chain.

        If median verification residual is worse than ``_VERIFY_FAIL_FACTOR``
        times the model RMS, the model is reset — a wrong correction is worse
        than no correction.
        """
        assert self._pointing_model is not None

        self._set_progress("Verifying calibration...")
        self.logger.info("Calibration verification: checking %d targets with correction applied", self._VERIFY_COUNT)

        verify_targets = self._pick_verification_targets(mount, site_lat, site_lon)
        residuals: list[float] = []

        for i, (target_ra, target_dec) in enumerate(verify_targets):
            if self._calibration_cancel.is_set():
                break

            try:
                self.hardware_adapter.point_telescope(target_ra, target_dec)

                timeout = 300
                start = time.time()
                while mount.is_slewing():
                    if time.time() - start > timeout:
                        mount.abort_slew()
                        self.logger.warning("Verification: slew timeout for target %d — skipping", i + 1)
                        break
                    time.sleep(0.5)

                solve_result = self._plate_solve_at_current_position(telescope_record)
                if solve_result is None:
                    self.logger.warning("Verification: plate solve failed for target %d — skipping", i + 1)
                    continue

                solved_ra, solved_dec = solve_result
                residual_deg = self.hardware_adapter.angular_distance(target_ra, target_dec, solved_ra, solved_dec)
                residuals.append(residual_deg)

                self._pointing_model.record_verification_residual(residual_deg)
                self.logger.info(
                    "Verification target %d/%d: residual %.4f° (target RA=%.4f° Dec=%.4f°)",
                    i + 1,
                    len(verify_targets),
                    residual_deg,
                    target_ra,
                    target_dec,
                )
            except Exception as exc:
                self.logger.warning("Verification target %d failed: %s", i + 1, exc)

        if not residuals:
            self.logger.warning(
                "Calibration verification: no verification solves succeeded — model kept but unverified"
            )
            return

        median_residual = statistics.median(residuals)
        threshold = max(model_rms * self._VERIFY_FAIL_FACTOR, self._VERIFY_FLOOR_DEG)

        if median_residual > threshold:
            self.logger.error(
                "CALIBRATION VERIFICATION FAILED: median residual %.4f° is > %.4f° threshold "
                "(%.0f× model RMS of %.4f°). Correction is making pointing WORSE — resetting model.",
                median_residual,
                threshold,
                median_residual / model_rms if model_rms > 0 else 0,
                model_rms,
            )
            self._pointing_model.reset()
        else:
            self.logger.info(
                "Calibration verification passed: median residual %.4f° (threshold %.4f°, model RMS %.4f°)",
                median_residual,
                threshold,
                model_rms,
            )

    def _pick_verification_targets(self, mount: Any, site_lat: float, site_lon: float) -> list[tuple[float, float]]:
        """Generate verification targets at +/-90 degrees az from the current mount position.

        One target is CW, the other CCW, both at 45 degrees altitude.
        The opposing slew directions keep cable wrap bounded (peak +/-90 degrees).
        """
        from citrascope.hardware.devices.mount.altaz_pointing_model import altaz_to_radec

        current_az = mount.get_azimuth()
        if current_az is None:
            current_az = 180.0

        targets: list[tuple[float, float]] = []
        for offset in [90.0, -90.0]:
            az = (current_az + offset) % 360.0
            ra, dec = altaz_to_radec(az, 45.0, site_lat, site_lon)
            targets.append((ra, dec))
        return targets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _UNWIND_THRESHOLD_DEG = 30.0

    def _unwind_before_calibration(self) -> None:
        """Unwind cable wrap before calibration to maximize sky coverage."""
        self._unwind_if_needed("calibration")

    def _unwind_if_needed(self, phase: str = "calibration") -> None:
        """Unwind cable wrap if cumulative rotation exceeds threshold.

        Reusable for any calibration phase (pre-walk, pre-verification, etc.).
        """
        if not self.safety_monitor:
            return

        from citrascope.safety.cable_wrap_check import CableWrapCheck

        check = self.safety_monitor.get_check("cable_wrap")
        if not isinstance(check, CableWrapCheck):
            return

        cumulative = check.cumulative_deg
        if abs(cumulative) <= self._UNWIND_THRESHOLD_DEG:
            self.logger.info("Cable wrap at %.0f° — no unwind needed before %s", cumulative, phase)
            return

        self.logger.info("Cable wrap at %.0f° — unwinding before %s...", cumulative, phase)
        self._set_progress("Unwinding cable wrap...")

        check.execute_action()
        while check.is_unwinding:
            if self._calibration_cancel.is_set():
                self.logger.info("Calibration cancelled during %s unwind", phase)
                return
            time.sleep(1.0)

        final = check.cumulative_deg
        if abs(final) <= self._UNWIND_THRESHOLD_DEG:
            self.logger.info("Cable unwind complete: %.0f° cumulative — ready for %s", final, phase)
        else:
            self.logger.warning(
                "Cable unwind finished at %.0f° (target <%.0f°) — proceeding with %s anyway",
                final,
                self._UNWIND_THRESHOLD_DEG,
                phase,
            )

    def _plate_solve_at_current_position(self, telescope_record: dict[str, Any]) -> tuple[float, float] | None:
        """Take an image and plate solve at the current mount position.

        Tries increasing exposure times (2s, 4s, 8s).  Returns (ra, dec)
        on success or None.
        """
        from citrascope.processors.builtin.plate_solver_processor import PlateSolverProcessor

        exposure_attempts = [2.0, 4.0, 8.0]
        for exposure_s in exposure_attempts:
            try:
                image_path = self.hardware_adapter.take_image("calibration", exposure_s)
            except Exception as exc:
                self.logger.warning("Calibration exposure failed (%.0fs): %s", exposure_s, exc)
                continue

            if self._preview_bus:
                try:
                    self._preview_bus.push_file(image_path, "calibration")
                except Exception as e:
                    self.logger.debug(f"Failed to push calibration preview: {e}")
            result = PlateSolverProcessor.solve(
                Path(image_path), telescope_record, location_service=self.location_service
            )
            if result is not None:
                return result

            self.logger.warning("Plate solve failed with %.0fs exposure, retrying...", exposure_s)

        return None

    def _get_cable_wrap_cumulative(self) -> float:
        """Read current cable wrap cumulative rotation, or 0.0 if unavailable."""
        if not self.safety_monitor:
            return 0.0
        try:
            from citrascope.safety.cable_wrap_check import CableWrapCheck

            check = self.safety_monitor.get_check("cable_wrap")
            if isinstance(check, CableWrapCheck):
                return check.cumulative_deg
        except Exception:
            pass
        return 0.0
