"""Flat-capture backend protocol and direct-camera implementation.

``MasterBuilder.build_flat`` shapes raw flat frames into a stacked,
bias-subtracted, median-normalized master.  How those *raw* flat frames
are produced is the only thing that differs between direct hardware and
an orchestrator like NINA — so we isolate exactly that step behind
:class:`FlatCaptureBackend`.  Everything else (stacking, quality
validation, normalization, library save, progress reporting) stays in
MasterBuilder and runs unchanged on both paths.

Backends return a list of temp FITS paths written under
``library.tmp_dir``.  That matches MasterBuilder's existing memory
model (frames never held in RAM concurrently) so swapping backends is a
seam change with no surprises downstream.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from astropy.io import fits  # type: ignore[attr-defined]

from citrasense.calibration import FilterSlot
from citrasense.calibration.calibration_library import CalibrationLibrary

if TYPE_CHECKING:
    from citrasense.hardware.devices.camera.abstract_camera import AbstractCamera

logger = logging.getLogger("citrasense.FlatCaptureBackend")

ProgressCallback = Callable[[int, int, str, str], None]


class FlatCaptureBackend(Protocol):
    """Produce *count* raw flat frames for a filter.

    Implementations must write each frame to a temp FITS under
    ``library.tmp_dir`` (caller-supplied) and return the list of paths.
    Stacking, bias subtraction, and normalization happen downstream in
    :class:`~citrasense.calibration.master_builder.MasterBuilder`.
    """

    def capture_flat_frames(
        self,
        *,
        filter_slot: FilterSlot | None,
        count: int,
        gain: int,
        binning: int,
        initial_exposure: float,
        library: CalibrationLibrary,
        cancel_event: threading.Event | None,
        on_progress: ProgressCallback | None,
    ) -> list[Path]: ...

    def cancel(self) -> None: ...

    @property
    def supported_frame_types(self) -> set[str]: ...


TARGET_ADU_FRACTION = 0.50
ADU_TOLERANCE = 0.10
AUTO_EXPOSE_MIN_S = 0.001
AUTO_EXPOSE_MAX_S = 30.0
AUTO_EXPOSE_MAX_ITERATIONS = 8
AUTO_EXPOSE_MAX_STEP = 4.0


def auto_expose_flat(
    camera: AbstractCamera,
    initial_exposure: float,
    gain: int,
    binning: int,
    on_progress: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> float:
    """Tune exposure until a test frame medians at ~50% of max ADU.

    Extracted module-level so both :class:`MasterBuilder` and
    :class:`DirectCameraFlatBackend` share a single implementation.  The
    behaviour matches MasterBuilder's original private ``_auto_expose_flat``
    exactly — sensor response to uniform illumination is approximately
    linear, so each step scales exposure by ``target / median`` clamped
    to 4x.  Bails early when pinned at min/max exposure.
    """
    max_adu = float(camera.get_max_pixel_value(binning))
    logger.info("Auto-expose using max_adu=%d for binning=%d", int(max_adu), binning)
    target_adu = max_adu * TARGET_ADU_FRACTION
    lo = max_adu * (TARGET_ADU_FRACTION - ADU_TOLERANCE)
    hi = max_adu * (TARGET_ADU_FRACTION + ADU_TOLERANCE)

    exposure = max(AUTO_EXPOSE_MIN_S, min(initial_exposure, AUTO_EXPOSE_MAX_S))

    attempt = 0
    for attempt in range(AUTO_EXPOSE_MAX_ITERATIONS):
        if cancel_event and cancel_event.is_set():
            break

        if on_progress:
            on_progress(0, 0, "flat", f"Auto-expose: testing {exposure:.3f}s (attempt {attempt + 1})")

        data = camera.capture_array(
            duration=exposure,
            gain=gain,
            binning=binning,
            shutter_closed=False,
        )
        median_adu = float(np.median(data))
        pct = (median_adu / max_adu) * 100

        logger.info(
            "Flat auto-expose attempt %d: %.3fs -> median %.0f ADU (%.0f%% of max)",
            attempt + 1,
            exposure,
            median_adu,
            pct,
        )
        if on_progress:
            on_progress(0, 0, "flat", f"Auto-expose: {median_adu:.0f} ADU ({pct:.0f}%) at {exposure:.3f}s")

        if lo <= median_adu <= hi:
            logger.info("Flat auto-expose converged: %.3fs -> %.0f ADU (%.0f%%)", exposure, median_adu, pct)
            return exposure

        if median_adu < 1.0:
            new_exposure = exposure * AUTO_EXPOSE_MAX_STEP
        else:
            ratio = target_adu / median_adu
            ratio = max(1.0 / AUTO_EXPOSE_MAX_STEP, min(ratio, AUTO_EXPOSE_MAX_STEP))
            new_exposure = exposure * ratio

        new_exposure = max(AUTO_EXPOSE_MIN_S, min(new_exposure, AUTO_EXPOSE_MAX_S))

        if new_exposure == exposure and median_adu < lo:
            logger.warning(
                "Flat auto-expose: at max exposure (%.1fs) but only %.0f ADU (%.0f%%). " "Light source may be too dim.",
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


class DirectCameraFlatBackend:
    """Capture flats by driving an ``AbstractCamera`` directly.

    Auto-exposes once for the filter, then takes ``count`` frames with
    the shutter open.  Each frame is written to a temp FITS under
    ``library.tmp_dir`` so memory usage stays bounded regardless of
    sensor resolution or frame count.
    """

    def __init__(self, camera: AbstractCamera) -> None:
        self._camera = camera
        self._local_cancel = threading.Event()

    @property
    def supported_frame_types(self) -> set[str]:
        return {"flat"}

    def cancel(self) -> None:
        self._local_cancel.set()

    def capture_flat_frames(
        self,
        *,
        filter_slot: FilterSlot | None,
        count: int,
        gain: int,
        binning: int,
        initial_exposure: float,
        library: CalibrationLibrary,
        cancel_event: threading.Event | None,
        on_progress: ProgressCallback | None,
    ) -> list[Path]:
        self._local_cancel.clear()
        exposure = auto_expose_flat(
            self._camera,
            initial_exposure=initial_exposure,
            gain=gain,
            binning=binning,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

        label = f"flat g{gain} bin{binning} {exposure:.3f}s"
        if filter_slot is not None and filter_slot.name:
            label += f" {filter_slot.name}"

        paths: list[Path] = []
        for i in range(count):
            if cancel_event and cancel_event.is_set():
                break
            if self._local_cancel.is_set():
                break

            if on_progress:
                on_progress(i + 1, count, "flat", f"Capturing {label} ({i + 1}/{count})")

            data = self._camera.capture_array(
                duration=exposure,
                gain=gain,
                binning=binning,
                shutter_closed=False,
            )

            tmp_path = library.tmp_dir / f"flat_{i:04d}_{int(time.time())}.fits"
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(tmp_path, overwrite=True)
            paths.append(tmp_path)

        return paths
