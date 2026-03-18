"""MasterBuilder — captures raw calibration frames and median-combines into masters.

Reads CalibrationProfile from the camera to determine capabilities.
No camera-specific code — all behaviour is driven by the profile.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore[attr-defined]

from citrascope.calibration import FilterSlot
from citrascope.calibration.calibration_library import CalibrationLibrary
from citrascope.hardware.devices.camera.abstract_camera import AbstractCamera, CalibrationProfile

logger = logging.getLogger("citrascope")

ProgressCallback = Callable[[int, int, str, str], None]


class MasterBuilder:
    """Captures N raw calibration frames and median-combines them into a master."""

    def __init__(
        self,
        camera: AbstractCamera,
        library: CalibrationLibrary,
        profile: CalibrationProfile,
    ) -> None:
        self._camera = camera
        self._library = library
        self._profile = profile

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def build_bias(
        self,
        count: int,
        gain: int | None = None,
        binning: int = 1,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Path | None:
        """Capture and build a master bias frame.

        Returns the saved master path, or ``None`` if no frames were
        captured (e.g. immediate cancellation).
        """
        gain_val = gain if gain is not None else (self._profile.current_gain or 0)
        label = f"bias g{gain_val} bin{binning}"

        raw_paths = self._capture_frames(
            count=count,
            duration=0.0,
            gain=gain_val,
            binning=binning,
            shutter_closed=True,
            frame_type="bias",
            label=label,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

        if not raw_paths:
            self._library.cleanup_tmp()
            return None

        self._report(on_progress, count, count, "bias", f"Stacking {len(raw_paths)} frames...")
        master = self._median_stack(raw_paths)
        self._library.cleanup_tmp()

        return self._library.save_master(
            frame_type="bias",
            camera_id=self._profile.camera_id,
            data=master,
            gain=gain_val,
            binning=binning,
            ncombine=len(raw_paths),
            camera_model=self._profile.model,
            read_mode=self._profile.read_mode,
        )

    def build_dark(
        self,
        count: int,
        exposure_time: float,
        gain: int | None = None,
        binning: int = 1,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Path | None:
        """Capture and build a master dark frame (bias-subtracted if bias available).

        Returns the saved master path, or ``None`` if no frames were
        captured (e.g. immediate cancellation).
        """
        gain_val = gain if gain is not None else (self._profile.current_gain or 0)
        label = f"dark g{gain_val} bin{binning} {exposure_time}s"

        raw_paths = self._capture_frames(
            count=count,
            duration=exposure_time,
            gain=gain_val,
            binning=binning,
            shutter_closed=True,
            frame_type="dark",
            label=label,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

        if not raw_paths:
            self._library.cleanup_tmp()
            return None

        self._report(on_progress, count, count, "dark", f"Stacking {len(raw_paths)} frames...")
        master = self._median_stack(raw_paths)

        # Subtract master bias if available
        rm = self._profile.read_mode
        bias_path = self._library.get_master_bias(self._profile.camera_id, gain_val, binning, rm)
        did_subtract_bias = False
        if bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
            master = master - bias_data
            did_subtract_bias = True
            logger.info("Subtracted master bias from dark")

        self._library.cleanup_tmp()

        temperature = self._resolve_dark_temperature()

        return self._library.save_master(
            frame_type="dark",
            camera_id=self._profile.camera_id,
            data=master,
            gain=gain_val,
            binning=binning,
            exposure_time=exposure_time,
            temperature=temperature,
            ncombine=len(raw_paths),
            camera_model=self._profile.model,
            read_mode=rm,
            bias_subtracted=did_subtract_bias,
        )

    def build_flat(
        self,
        count: int,
        exposure_time: float,
        filter_name: str = "",
        gain: int | None = None,
        binning: int = 1,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Path | None:
        """Capture and build a master flat (bias-subtracted, normalised to median=1.0).

        Runs auto-exposure before the main capture to find an exposure time
        that places the median ADU at ~50% of the sensor's dynamic range.

        Returns the saved master path, or ``None`` if quality validation
        rejected the flat.
        """
        gain_val = gain if gain is not None else (self._profile.current_gain or 0)

        exposure_time = self._auto_expose_flat(
            initial_exposure=exposure_time,
            gain=gain_val,
            binning=binning,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

        label = f"flat g{gain_val} bin{binning} {exposure_time:.3f}s"
        if filter_name:
            label += f" {filter_name}"

        raw_paths = self._capture_frames(
            count=count,
            duration=exposure_time,
            gain=gain_val,
            binning=binning,
            shutter_closed=False,
            frame_type="flat",
            label=label,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

        if not raw_paths:
            self._library.cleanup_tmp()
            return None

        self._report(on_progress, count, count, "flat", f"Stacking {len(raw_paths)} frames...")
        master = self._median_stack(raw_paths)

        # Subtract master bias if available
        rm = self._profile.read_mode
        bias_path = self._library.get_master_bias(self._profile.camera_id, gain_val, binning, rm)
        if bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
            master = master - bias_data
            logger.info("Subtracted master bias from flat")

        self._library.cleanup_tmp()

        max_adu = float(self._camera.get_max_pixel_value(binning))
        ok, reason = self._validate_flat_quality(master, max_adu, filter_name)
        if not ok:
            logger.warning("Flat quality check failed for %s: %s — master NOT saved", filter_name or "nofilter", reason)
            return None

        # Normalise to median = 1.0
        med = float(np.median(master))
        if med > 0:
            master = master / med
        else:
            logger.warning("Flat master median is zero — normalisation skipped")

        return self._library.save_master(
            frame_type="flat",
            camera_id=self._profile.camera_id,
            data=master,
            gain=gain_val,
            binning=binning,
            filter_name=filter_name,
            ncombine=len(raw_paths),
            camera_model=self._profile.model,
            read_mode=rm,
        )

    def build_interleaved_flats(
        self,
        filters: list[FilterSlot],
        set_filter: Callable[[int], bool],
        count: int,
        initial_exposure: float = 1.0,
        gain: int | None = None,
        binning: int = 1,
        reexpose_interval: int = 3,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[Path]:
        """Capture interleaved flat frames across all filters, then stack per filter.

        Cycles through filters in rounds so every filter samples across
        the sky-brightness gradient during twilight.  Raw frames are written
        to temp FITS on disk (not held in memory) so the capture phase stays
        bounded regardless of frame count or sensor resolution.  Stacking
        loads one filter at a time.

        Args:
            filters: Enabled filter slots to cycle through.
            set_filter: Callback to switch the filter wheel.
            count: Number of frames per filter.
            initial_exposure: Starting exposure for auto-expose (seconds).
            gain: Camera gain (None = use profile default).
            binning: Pixel binning factor.
            reexpose_interval: Re-check auto-expose every N rounds.
            cancel_event: Threading event for cancellation.
            on_progress: Progress callback ``(current, total, type, status)``.

        Returns:
            List of saved master flat paths (one per filter that passed
            quality validation).
        """
        gain_val = gain if gain is not None else (self._profile.current_gain or 0)
        max_adu = float(self._camera.get_max_pixel_value(binning))
        target_adu = max_adu * self.TARGET_ADU_FRACTION
        lo = max_adu * (self.TARGET_ADU_FRACTION - self.ADU_TOLERANCE)
        hi = max_adu * (self.TARGET_ADU_FRACTION + self.ADU_TOLERANCE)

        # Per-filter state: file paths on disk, not in-memory arrays
        frame_paths: dict[str, list[Path]] = {f.name: [] for f in filters}
        exposures: dict[str, float] = {}
        last_median: dict[str, float] = {}

        # --- Phase 1: auto-expose each filter (carry forward) ----------
        carry_exposure = initial_exposure
        for fi, filt in enumerate(filters):
            if cancel_event and cancel_event.is_set():
                return []
            fname = filt.name
            self._report(
                on_progress,
                0,
                0,
                "flat",
                f"Auto-expose filter {fi + 1}/{len(filters)}: {fname}",
            )
            if not set_filter(filt.position):
                logger.error("Failed to set filter %s (position %d)", fname, filt.position)
                continue
            converged = self._auto_expose_flat(
                initial_exposure=carry_exposure,
                gain=gain_val,
                binning=binning,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            exposures[fname] = converged
            carry_exposure = converged

        if not exposures:
            logger.error("No filters could auto-expose — aborting interleaved flats")
            return []

        # --- Phase 2: round-robin capture (disk-backed) ----------------
        round_start = time.monotonic()
        round_times: list[float] = []
        total_captures = count * len(filters)

        for rnd in range(1, count + 1):
            if cancel_event and cancel_event.is_set():
                logger.info("Interleaved flat capture cancelled at round %d/%d", rnd, count)
                break

            rnd_t0 = time.monotonic()
            for fi, filt in enumerate(filters):
                if cancel_event and cancel_event.is_set():
                    break
                fname = filt.name
                if fname not in exposures:
                    continue

                captured_so_far = (rnd - 1) * len(filters) + fi
                eta_str = ""
                if round_times:
                    avg_round = sum(round_times) / len(round_times)
                    remaining_rounds = count - rnd + 1
                    eta_s = avg_round * remaining_rounds
                    if eta_s >= 60:
                        eta_str = f" [ETA {eta_s / 60:.0f}m{eta_s % 60:.0f}s]"
                    else:
                        eta_str = f" [ETA {eta_s:.0f}s]"

                self._report(
                    on_progress,
                    captured_so_far + 1,
                    total_captures,
                    "flat",
                    f"Round {rnd}/{count}: {fname}{eta_str}",
                )

                if not set_filter(filt.position):
                    logger.warning("Filter switch failed for %s in round %d", fname, rnd)
                    continue

                data = self._camera.capture_array(
                    duration=exposures[fname],
                    gain=gain_val,
                    binning=binning,
                    shutter_closed=False,
                )
                last_median[fname] = float(np.median(data))

                safe_fname = fname.replace(" ", "_").replace("/", "-").replace("\\", "-")
                tmp_path = self._library.tmp_dir / f"flat_{safe_fname}_{rnd:04d}_{int(time.time())}.fits"
                hdu = fits.PrimaryHDU(data)
                hdu.writeto(tmp_path, overwrite=True)
                frame_paths[fname].append(tmp_path)

            rnd_elapsed = time.monotonic() - rnd_t0
            round_times.append(rnd_elapsed)

            if reexpose_interval > 0 and rnd % reexpose_interval == 0 and rnd < count:
                self._reexpose_check(
                    filters,
                    exposures,
                    last_median,
                    set_filter,
                    gain_val,
                    binning,
                    target_adu,
                    lo,
                    hi,
                    on_progress,
                    cancel_event,
                )

        # --- Phase 3: stack one filter at a time, validate, save -------
        saved_paths: list[Path] = []
        rm = self._profile.read_mode
        bias_path = self._library.get_master_bias(self._profile.camera_id, gain_val, binning, rm)
        bias_data: np.ndarray | None = None
        if bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]

        for filt in filters:
            fname = filt.name
            paths = frame_paths.get(fname, [])
            if len(paths) < 3:
                logger.warning("Skipping flat for %s — only %d frames captured (need >= 3)", fname, len(paths))
                continue

            self._report(on_progress, 0, 0, "flat", f"Stacking {len(paths)} frames for {fname}...")
            master = self._median_stack(paths)

            if bias_data is not None:
                master = master - bias_data
                logger.info("Subtracted master bias from flat %s", fname)

            ok, reason = self._validate_flat_quality(master, max_adu, fname)
            if not ok:
                logger.warning("Flat quality check FAILED for %s: %s — master NOT saved", fname, reason)
                self._report(on_progress, 0, 0, "flat", f"REJECTED {fname}: {reason}")
                continue

            med = float(np.median(master))
            if med > 0:
                master = master / med
            else:
                logger.warning("Flat master median is zero for %s — normalisation skipped", fname)
                continue

            path = self._library.save_master(
                frame_type="flat",
                camera_id=self._profile.camera_id,
                data=master,
                gain=gain_val,
                binning=binning,
                filter_name=fname,
                ncombine=len(paths),
                camera_model=self._profile.model,
                read_mode=rm,
            )
            saved_paths.append(path)
            self._report(on_progress, 0, 0, "flat", f"Saved master flat for {fname}")

        self._library.cleanup_tmp()

        total_elapsed = time.monotonic() - round_start
        logger.info(
            "Interleaved flats complete: %d/%d filters saved in %.0fs",
            len(saved_paths),
            len(filters),
            total_elapsed,
        )
        return saved_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    TARGET_ADU_FRACTION = 0.50
    ADU_TOLERANCE = 0.10
    AUTO_EXPOSE_MIN_S = 0.001
    AUTO_EXPOSE_MAX_S = 30.0
    AUTO_EXPOSE_MAX_ITERATIONS = 8
    AUTO_EXPOSE_MAX_STEP = 4.0

    def _auto_expose_flat(
        self,
        initial_exposure: float,
        gain: int,
        binning: int,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> float:
        """Find an exposure time that puts the flat at ~50% of the sensor's dynamic range.

        Takes test frames, measures median ADU, and scales exposure linearly
        (sensor response to uniform illumination is approximately linear).
        Each step is clamped to at most 4x change to avoid jarring jumps.
        Bails early if already at max exposure and still far below target.
        Returns the tuned exposure time.
        """
        max_adu = float(self._camera.get_max_pixel_value(binning))
        logger.info("Auto-expose using max_adu=%d for binning=%d", int(max_adu), binning)
        target_adu = max_adu * self.TARGET_ADU_FRACTION
        lo = max_adu * (self.TARGET_ADU_FRACTION - self.ADU_TOLERANCE)
        hi = max_adu * (self.TARGET_ADU_FRACTION + self.ADU_TOLERANCE)

        exposure = max(self.AUTO_EXPOSE_MIN_S, min(initial_exposure, self.AUTO_EXPOSE_MAX_S))

        for attempt in range(self.AUTO_EXPOSE_MAX_ITERATIONS):
            if cancel_event and cancel_event.is_set():
                break

            self._report(
                on_progress,
                0,
                0,
                "flat",
                f"Auto-expose: testing {exposure:.3f}s (attempt {attempt + 1})",
            )

            data = self._camera.capture_array(
                duration=exposure,
                gain=gain,
                binning=binning,
                shutter_closed=False,
            )
            median_adu = float(np.median(data))
            pct = (median_adu / max_adu) * 100

            logger.info(
                "Flat auto-expose attempt %d: %.3fs → median %.0f ADU (%.0f%% of max)",
                attempt + 1,
                exposure,
                median_adu,
                pct,
            )
            self._report(
                on_progress,
                0,
                0,
                "flat",
                f"Auto-expose: {median_adu:.0f} ADU ({pct:.0f}%) at {exposure:.3f}s",
            )

            if lo <= median_adu <= hi:
                logger.info("Flat auto-expose converged: %.3fs → %.0f ADU (%.0f%%)", exposure, median_adu, pct)
                return exposure

            if median_adu < 1.0:
                new_exposure = exposure * self.AUTO_EXPOSE_MAX_STEP
            else:
                ratio = target_adu / median_adu
                ratio = max(1.0 / self.AUTO_EXPOSE_MAX_STEP, min(ratio, self.AUTO_EXPOSE_MAX_STEP))
                new_exposure = exposure * ratio

            new_exposure = max(self.AUTO_EXPOSE_MIN_S, min(new_exposure, self.AUTO_EXPOSE_MAX_S))

            if new_exposure == exposure and median_adu < lo:
                logger.warning(
                    "Flat auto-expose: at max exposure (%.1fs) but only %.0f ADU (%.0f%%). "
                    "Light source may be too dim.",
                    exposure,
                    median_adu,
                    pct,
                )
                break
            if new_exposure == exposure and median_adu > hi:
                logger.warning(
                    "Flat auto-expose: at min exposure (%.4fs) but still %.0f ADU (%.0f%%). "
                    "Light source may be too bright.",
                    exposure,
                    median_adu,
                    pct,
                )
                break

            exposure = new_exposure

        logger.warning(
            "Flat auto-expose did not converge after %d attempts, using %.3fs",
            attempt + 1,
            exposure,
        )
        return exposure

    def _resolve_dark_temperature(self) -> float | None:
        """Determine the temperature to record for a dark master.

        Uses the cooling target (operator intent) when available, since
        the sensor reading drifts by fractions of a degree around it.
        Falls back to a fresh sensor reading so the FITS header reflects
        the actual post-capture temperature rather than a stale profile
        snapshot.
        """
        if self._profile.target_temperature is not None:
            return float(self._profile.target_temperature)
        live = self._camera.get_temperature()
        if live is not None:
            return live
        return self._profile.current_temperature

    def _capture_frames(
        self,
        count: int,
        duration: float,
        gain: int,
        binning: int,
        shutter_closed: bool,
        frame_type: str,
        label: str,
        cancel_event: threading.Event | None,
        on_progress: ProgressCallback | None,
    ) -> list[Path]:
        """Capture *count* raw frames, writing each to a temp FITS on disk."""
        paths: list[Path] = []
        for i in range(count):
            if cancel_event and cancel_event.is_set():
                logger.info("Calibration capture cancelled at frame %d/%d", i + 1, count)
                break

            self._report(on_progress, i + 1, count, frame_type, f"Capturing {label} ({i + 1}/{count})")
            data = self._camera.capture_array(
                duration=duration,
                gain=gain,
                binning=binning,
                shutter_closed=shutter_closed,
            )

            tmp_path = self._library.tmp_dir / f"{frame_type}_{i:04d}_{int(time.time())}.fits"
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(tmp_path, overwrite=True)
            paths.append(tmp_path)

        return paths

    ADU_DRIFT_THRESHOLD = 0.20

    FLAT_MAX_OVER_MEDIAN = 5.0
    """Reject flats where max/median exceeds this (indicates stars)."""

    FLAT_MIN_SIGNAL_FRACTION = 0.05
    """Reject flats where pre-normalisation median is below this fraction of max ADU."""

    FLAT_HIGH_VARIANCE_RATIO = 0.30
    """Warn (but allow) flats where std/median exceeds this."""

    def _reexpose_check(
        self,
        filters: list[FilterSlot],
        exposures: dict[str, float],
        last_median: dict[str, float],
        set_filter: Callable[[int], bool],
        gain: int,
        binning: int,
        target_adu: float,
        lo_adu: float,
        hi_adu: float,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> None:
        """Check ADU drift on the most recent frame per filter; re-expose if needed."""
        for filt in filters:
            if cancel_event and cancel_event.is_set():
                return
            fname = filt.name
            if fname not in last_median or fname not in exposures:
                continue

            latest_median = last_median[fname]
            drift = abs(latest_median - target_adu) / target_adu if target_adu > 0 else 0

            if drift > self.ADU_DRIFT_THRESHOLD and (latest_median < lo_adu or latest_median > hi_adu):
                logger.info(
                    "ADU drift %.0f%% for %s (median %.0f, target %.0f) — re-exposing",
                    drift * 100,
                    fname,
                    latest_median,
                    target_adu,
                )
                self._report(on_progress, 0, 0, "flat", f"Re-exposing {fname} (ADU drifted {drift * 100:.0f}%)")
                if not set_filter(filt.position):
                    continue
                new_exp = self._auto_expose_flat(
                    initial_exposure=exposures[fname],
                    gain=gain,
                    binning=binning,
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                )
                exposures[fname] = new_exp

    @staticmethod
    def _validate_flat_quality(
        master: np.ndarray,
        max_adu: float,
        filter_name: str,
    ) -> tuple[bool, str]:
        """Check a stacked (pre-normalization) flat master for obvious defects.

        Returns ``(True, "")`` if acceptable, ``(False, reason)`` if the
        master should be rejected.
        """
        med = float(np.median(master))
        peak = float(np.max(master))
        std = float(np.std(master))

        if med <= 0:
            return False, "median is zero or negative"

        ratio = peak / med
        if ratio > MasterBuilder.FLAT_MAX_OVER_MEDIAN:
            return False, f"star contamination (max/median={ratio:.1f}, threshold {MasterBuilder.FLAT_MAX_OVER_MEDIAN})"

        min_signal = MasterBuilder.FLAT_MIN_SIGNAL_FRACTION * max_adu
        if med < min_signal:
            return False, f"insufficient signal (median={med:.0f}, need >{min_signal:.0f})"

        if std / med > MasterBuilder.FLAT_HIGH_VARIANCE_RATIO:
            logger.warning(
                "Flat %s has high variance (std/median=%.2f) — saving but may indicate issues",
                filter_name,
                std / med,
            )

        return True, ""

    @staticmethod
    def _median_stack(paths: list[Path]) -> np.ndarray:
        """Load temporary FITS files and compute the pixel-wise median.

        Raises ``ValueError`` if *paths* is empty (caller should guard).
        """
        if not paths:
            raise ValueError("Cannot median-stack zero frames")
        arrays: list[np.ndarray] = []
        for p in paths:
            with fits.open(p) as hdul:
                arrays.append(hdul[0].data.astype(np.float32))  # type: ignore[index]
        return np.median(np.array(arrays), axis=0)  # type: ignore[arg-type]

    @staticmethod
    def _report(
        cb: ProgressCallback | None,
        current: int,
        total: int,
        frame_type: str,
        status: str,
    ) -> None:
        if cb:
            cb(current, total, frame_type, status)
