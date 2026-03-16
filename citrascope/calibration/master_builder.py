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
        bias_path = self._library.get_master_bias(self._profile.camera_id, gain_val, binning)
        if bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
            master = master - bias_data
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
        """Capture and build a master flat (bias-subtracted, normalised to median=1.0)."""
        gain_val = gain if gain is not None else (self._profile.current_gain or 0)
        label = f"flat g{gain_val} bin{binning}" + (f" {filter_name}" if filter_name else "")

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
        bias_path = self._library.get_master_bias(self._profile.camera_id, gain_val, binning)
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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

            self._report(on_progress, i, count, frame_type, f"Capturing {label} ({i + 1}/{count})")
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
