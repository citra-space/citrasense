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
    ) -> Path:
        """Capture and build a master bias frame."""
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
    ) -> Path:
        """Capture and build a master dark frame (bias-subtracted if bias available)."""
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
    ) -> Path:
        """Capture and build a master flat (bias-subtracted, normalised to median=1.0).

        Runs auto-exposure before the main capture to find an exposure time
        that places the median ADU at ~50% of the sensor's dynamic range.
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

        # Normalise to median = 1.0
        med = float(np.median(master))
        if med > 0:
            master = master / med
        else:
            logger.warning("Flat master median is zero — normalisation skipped")

        self._library.cleanup_tmp()

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

    @staticmethod
    def _median_stack(paths: list[Path]) -> np.ndarray:
        """Load temporary FITS files and compute the pixel-wise median."""
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
