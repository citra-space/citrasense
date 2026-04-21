"""CalibrationProcessor — applies master calibration frames to science images.

Registered as the first pipeline stage (index 0), before plate solving.
Performs CCD calibration math: ``(raw - master_dark) / master_flat``.
Gracefully skips with warnings when masters are missing.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits  # type: ignore[attr-defined]

from citrasense.calibration.calibration_library import CalibrationLibrary, resolve_camera_id
from citrasense.processors.abstract_processor import AbstractImageProcessor
from citrasense.processors.processor_result import ProcessingContext, ProcessorResult

if TYPE_CHECKING:
    pass


class CalibrationProcessor(AbstractImageProcessor):
    """Apply master bias/dark/flat calibration to science frames."""

    name = "calibration"
    friendly_name = "Calibration"
    description = "Apply bias/dark/flat master calibration frames to raw science images"

    def __init__(self, library: CalibrationLibrary | None = None) -> None:
        self._library = library

    @property
    def library(self) -> CalibrationLibrary | None:
        return self._library

    @library.setter
    def library(self, value: CalibrationLibrary) -> None:
        self._library = value

    def process(self, context: ProcessingContext) -> ProcessorResult:
        start = time.time()
        logger = context.logger

        if self._library is None:
            return self._skip(start, "CalibrationLibrary not initialized")

        image_path = context.working_image_path
        try:
            with fits.open(image_path) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                header = primary.header
                raw_data = primary.data
        except Exception as e:
            if logger:
                logger.warning("CalibrationProcessor: cannot read FITS — %s", e)
            return self._skip(start, f"Cannot read FITS: {e}")

        if raw_data is None:
            return self._skip(start, "FITS file has no image data")

        camera_id = resolve_camera_id(header)
        gain = int(header.get("GAIN", 0))  # type: ignore[arg-type]
        binning = int(header.get("XBINNING", 1))  # type: ignore[arg-type]
        exposure = float(header.get("EXPTIME", 0))  # type: ignore[arg-type]
        temperature = header.get("CCD-TEMP")
        filter_name = str(header.get("FILTER", "")).strip()
        read_mode = str(header.get("READMODE", "")).strip()

        # Check if this camera has any masters at all — if not, skip silently
        if not self._library.has_any_masters(camera_id):
            return self._skip(start, "No calibration masters for this camera (silent)")

        # Look up matching masters
        bias_path = self._library.get_master_bias(camera_id, gain, binning, read_mode)
        dark_path = self._library.get_master_dark(
            camera_id,
            gain,
            binning,
            float(temperature) if temperature is not None else None,  # type: ignore[arg-type]
            read_mode,
        )
        flat_path = (
            self._library.get_master_flat(camera_id, gain, binning, filter_name, read_mode) if filter_name else None
        )

        # Log warnings for missing masters
        missing: list[str] = []
        if not bias_path:
            missing.append(f"bias (gain {gain}, bin {binning})")
        if not dark_path:
            temp_str = f"{temperature:.1f}C" if temperature is not None else "unknown temp"
            missing.append(f"dark (at {temp_str}, gain {gain}, bin {binning})")
        if filter_name and not flat_path:
            missing.append(f"flat ({filter_name}, gain {gain}, bin {binning})")

        if missing and logger:
            logger.warning(
                "CalibrationProcessor: missing masters for %s — %s",
                camera_id,
                ", ".join(missing),
            )

        if not bias_path and not dark_path and not flat_path:
            return self._skip(start, f"No matching masters for current settings: {', '.join(missing)}")

        # Apply calibration
        calibrated = raw_data.astype(np.float32)

        if dark_path:
            with fits.open(dark_path) as hdul:
                dark_hdr = hdul[0].header  # type: ignore[index]
                dark_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
            dark_exposure = float(dark_hdr.get("EXPTIME", exposure))
            dark_bias_subtracted = bool(dark_hdr.get("BIASSUB", True))

            if bias_path and dark_bias_subtracted:
                # Dark is already bias-subtracted — contains only thermal D(T_ref).
                # Subtract bias and scaled thermal separately:
                #   calibrated = raw - bias - D(T_ref) * (T_science / T_ref)
                with fits.open(bias_path) as hdul:
                    bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
                if dark_exposure > 0 and exposure > 0:
                    scale = exposure / dark_exposure
                    calibrated = calibrated - bias_data - dark_data * scale
                    if logger:
                        logger.info(
                            "Applied dark scaling (bias-sub'd ref): %.1fs → %.1fs (×%.3f) from %s",
                            dark_exposure,
                            exposure,
                            scale,
                            dark_path.name,
                        )
                else:
                    calibrated = calibrated - bias_data - dark_data
                    if logger:
                        logger.info("Applied bias + dark (unscaled): %s", dark_path.name)
            elif not dark_bias_subtracted:
                # Dark still contains bias+thermal.  If a bias master is
                # available we can separate them: subtract bias from both
                # the raw frame and the dark, then scale only the thermal
                # component.  Without bias we scale the whole dark, which
                # leaves a bias*(1-scale) residual — best we can do.
                if bias_path and dark_exposure > 0 and exposure > 0:
                    scale = exposure / dark_exposure
                    with fits.open(bias_path) as hdul:
                        bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
                    calibrated = calibrated - bias_data - (dark_data - bias_data) * scale
                    if logger:
                        logger.info(
                            "Applied dark scaling (bias-incl'd ref, bias separated): " "%.1fs → %.1fs (×%.3f) from %s",
                            dark_exposure,
                            exposure,
                            scale,
                            dark_path.name,
                        )
                elif dark_exposure > 0 and exposure > 0:
                    scale = exposure / dark_exposure
                    calibrated = calibrated - dark_data * scale
                    if logger:
                        logger.info(
                            "Applied dark scaling (bias-incl'd ref, no bias available): "
                            "%.1fs → %.1fs (×%.3f) from %s",
                            dark_exposure,
                            exposure,
                            scale,
                            dark_path.name,
                        )
                else:
                    calibrated = calibrated - dark_data
                    if logger:
                        logger.info("Applied dark (no bias sub, unscaled): %s", dark_path.name)
            else:
                # dark_bias_subtracted=True but no bias master available now —
                # apply the bias-subtracted dark as-is (thermal-only).
                if dark_exposure > 0 and exposure > 0:
                    scale = exposure / dark_exposure
                    calibrated = calibrated - dark_data * scale
                    if logger:
                        logger.info(
                            "Applied dark scaling (no current bias): %.1fs → %.1fs (×%.3f) from %s",
                            dark_exposure,
                            exposure,
                            scale,
                            dark_path.name,
                        )
                else:
                    calibrated = calibrated - dark_data
                    if logger:
                        logger.info("Applied dark (thermal-only, unscaled): %s", dark_path.name)
        elif bias_path:
            with fits.open(bias_path) as hdul:
                bias_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
            calibrated = calibrated - bias_data
            if logger:
                logger.info("Applied master bias: %s", bias_path.name)

        if flat_path:
            with fits.open(flat_path) as hdul:
                flat_data = hdul[0].data.astype(np.float32)  # type: ignore[index]
            # Flat is normalised to median=1.0, so division is safe
            flat_data = np.where(flat_data > 0.01, flat_data, 1.0)
            calibrated = calibrated / flat_data
            if logger:
                logger.info("Applied master flat: %s", flat_path.name)

        # Write calibrated FITS
        out_path = context.working_dir / "calibrated.fits"
        hdu = fits.PrimaryHDU(calibrated.astype(np.float32), header=header)
        hdu.header["CALPROC"] = (True, "Calibration processor applied")
        hdu.writeto(out_path, overwrite=True)

        context.working_image_path = out_path
        context.image_data = calibrated

        elapsed = time.time() - start
        applied = []
        if bias_path:
            applied.append("bias")
        if dark_path:
            applied.append("dark")
        if flat_path:
            applied.append("flat")

        return ProcessorResult(
            should_upload=True,
            extracted_data={"calibration_applied": applied, "calibration_missing": missing},
            confidence=1.0,
            reason=f"Applied calibration: {', '.join(applied)}" if applied else "No calibration applied",
            processing_time_seconds=elapsed,
            processor_name=self.name,
        )

    def _skip(self, start: float, reason: str) -> ProcessorResult:
        return ProcessorResult(
            should_upload=True,
            extracted_data={},
            confidence=1.0,
            reason=reason,
            processing_time_seconds=time.time() - start,
            processor_name=self.name,
        )
