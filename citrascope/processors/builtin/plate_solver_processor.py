"""Plate solving processor using Pixelemon (Tetra3)."""

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from pixelemon import Telescope
from pixelemon.optics._base_optical_assembly import BaseOpticalAssembly
from pixelemon.sensors._base_sensor import BaseSensor

from citrascope.processors.abstract_processor import AbstractImageProcessor
from citrascope.processors.artifact_writer import dump_processor_result
from citrascope.processors.processor_result import ProcessingContext, ProcessorResult

from .processor_dependencies import check_pixelemon, normalize_fits_timestamp


def _build_telescope_for_image(image_path: Path, context: Optional["ProcessingContext"] = None):
    """Build a Pixelemon Telescope from telescope_record and FITS binning info.

    telescope_record (from the Citra API) must supply the physical sensor specifications:
        pixelSize            — physical pixel size in μm (unbinned)
        focalLength          — focal length in mm
        horizontalPixelCount — full-resolution pixel count (unbinned)
        verticalPixelCount   — full-resolution pixel count (unbinned)

    XBINNING / YBINNING are read from the FITS header (default 1) to derive the
    effective pixel size and image dimensions for the current acquisition.

    focalRatio and imageCircleDiameter are used from telescope_record when present;
    imageCircleDiameter falls back to the sensor diagonal (geometric minimum).

    Raises:
        ValueError: If telescope_record is absent or any required field is missing.
    """

    telescope_record = getattr(context, "telescope_record", None) if context else None
    if not telescope_record:
        raise ValueError("telescope_record is required for plate solving — configure the telescope in Citra first")

    required = ("pixelSize", "focalLength", "focalRatio", "horizontalPixelCount", "verticalPixelCount")
    missing = [f for f in required if not telescope_record.get(f)]
    if missing:
        raise ValueError(
            f"telescope_record is missing required field(s): {', '.join(missing)} — "
            "populate the telescope sensor specs in Citra to enable plate solving"
        )

    pixel_size_um = float(telescope_record["pixelSize"])
    focal_length_mm = float(telescope_record["focalLength"])
    focal_ratio = float(telescope_record["focalRatio"])
    h_px = int(telescope_record["horizontalPixelCount"])
    v_px = int(telescope_record["verticalPixelCount"])

    with fits.open(image_path) as hdul:
        primary = hdul[0]
        assert isinstance(primary, fits.PrimaryHDU)
        header = primary.header
        x_binning = int(header.get("XBINNING", 1))  # type: ignore[arg-type]
        y_binning = int(header.get("YBINNING", 1))  # type: ignore[arg-type]

    effective_pixel_w_um = pixel_size_um * x_binning
    effective_pixel_h_um = pixel_size_um * y_binning
    nx = h_px // x_binning
    ny = v_px // y_binning

    sensor_diag_mm = math.sqrt((nx * effective_pixel_w_um) ** 2 + (ny * effective_pixel_h_um) ** 2) / 1000.0
    image_circle_diameter = (
        float(telescope_record["imageCircleDiameter"])
        if telescope_record.get("imageCircleDiameter")
        else sensor_diag_mm
    )

    sensor = BaseSensor(
        x_pixel_count=nx,
        y_pixel_count=ny,
        pixel_width=effective_pixel_w_um,
        pixel_height=effective_pixel_h_um,
    )
    optics = BaseOpticalAssembly(
        focal_length=focal_length_mm,
        focal_ratio=focal_ratio,
        image_circle_diameter=image_circle_diameter,
    )
    return Telescope(sensor=sensor, optics=optics)


def _fits_has_observer_location(header: fits.Header) -> bool:
    """True if FITS has full observer location (Pixelemon expects SITELAT/SITELONG/SITEALT or equivalents)."""
    if "SITELAT" in header and "SITELONG" in header and "SITEALT" in header:
        return True
    if "OBSGEO-L" in header and "OBSGEO-B" in header and "OBSGEO-H" in header:
        return True
    if "LAT-OBS" in header and "LONG-OBS" in header:
        return True
    return False


def _ensure_fits_has_observer_location(image_path: Path, context: ProcessingContext, working_dir: Path) -> Path:
    """If FITS lacks observer location and context has it, write a copy with
    SITELAT/SITELONG/SITEALT. Return path to use."""
    with fits.open(image_path) as hdul:
        primary = hdul[0]
        assert isinstance(primary, fits.PrimaryHDU)
        header = primary.header
        if _fits_has_observer_location(header):
            return image_path
        location = None
        try:
            if context.location_service:
                location = context.location_service.get_current_location()
        except Exception:
            pass
        if not location or not isinstance(location, dict):
            return image_path
        lat = location.get("latitude")
        lon = location.get("longitude")
        alt = location.get("altitude", 0)
        if lat is None or lon is None:
            return image_path
        out_path = working_dir / image_path.name
        if out_path.resolve() == image_path.resolve():
            return image_path
        new_header = header.copy()
        new_header["SITELAT"] = float(lat)
        new_header["SITELONG"] = float(lon)
        new_header["SITEALT"] = float(alt)

        # Pixelemon parses DATE-OBS internally via fromisoformat(); normalize to
        # 6 fractional digits so it works on Python 3.10 as well as 3.11+.
        if "DATE-OBS" in new_header:
            new_header["DATE-OBS"] = normalize_fits_timestamp(str(new_header["DATE-OBS"]))

        fits.writeto(out_path, primary.data, new_header, overwrite=True)
        return out_path


class PlateSolverProcessor(AbstractImageProcessor):
    """
    Plate solving processor using Pixelemon (Tetra3).

    Determines exact telescope pointing and embeds WCS (World Coordinate System)
    into a .new FITS file. Updates context.working_image_path to point to that file.

    Typical processing time: a few seconds (Tetra3).
    """

    name = "plate_solver"
    friendly_name = "Plate Solver"
    description = "Astrometric calibration via Pixelemon/Tetra3 (determines exact pointing and WCS)"

    @classmethod
    def solve(
        cls,
        image_path: Path,
        telescope_record: dict,
        location_service: Any | None = None,
    ) -> tuple[float, float] | None:
        """Plate solve an image and return ``(ra_deg, dec_deg)`` or ``None``.

        Convenience method for callers outside the processing pipeline
        (e.g. alignment manager, manual solve requests).
        """
        logger = logging.getLogger("citrascope.plate_solver")
        with tempfile.TemporaryDirectory(prefix="alignment_") as tmp:
            working_dir = Path(tmp)
            context = ProcessingContext(
                image_path=image_path,
                working_image_path=image_path,
                working_dir=working_dir,
                image_data=None,
                task=None,
                telescope_record=telescope_record,
                ground_station_record=None,
                settings=None,
                location_service=location_service,
                logger=logger,
            )
            result = cls().process(context)
            if result.extracted_data.get("plate_solved"):
                ra = result.extracted_data.get("ra_center")
                dec = result.extracted_data.get("dec_center")
                if ra is not None and dec is not None:
                    return float(ra), float(dec)
        return None

    def _solve_with_pixelemon(self, image_path: Path, context: ProcessingContext | None = None) -> Path:
        """Run Pixelemon (Tetra3) plate solve and write WCS to a .new file.

        Pixelemon internally fits a full 5th-degree SIP WCS from matched star centroids
        (equivalent to astrometry.net quality). We write that fitted WCS directly to the
        .new file via image._wcs.to_header(relax=True), which includes the CD rotation
        matrix and SIP distortion coefficients.

        Args:
            image_path: Path to FITS image to solve
            context: Optional processing context (unused; for API consistency)

        Returns:
            Path to .new file with full SIP WCS in header

        Raises:
            RuntimeError: If plate solving fails or solution is None
        """
        from pixelemon import TelescopeImage, TetraSolver

        TetraSolver.high_memory()
        telescope = _build_telescope_for_image(image_path, context)
        image = TelescopeImage.from_fits_file(image_path, telescope)
        solve = image.plate_solve  # triggers internal fit_wcs_from_points(sip_degree=5)

        if solve is None:
            raise RuntimeError("Pixelemon plate solving returned no solution")

        with fits.open(image_path) as hdul:
            primary = hdul[0]
            assert isinstance(primary, fits.PrimaryHDU)
            naxis1 = int(primary.header.get("NAXIS1", 0))  # type: ignore[arg-type]
            naxis2 = int(primary.header.get("NAXIS2", 0))  # type: ignore[arg-type]
            if naxis1 <= 0 or naxis2 <= 0:
                raise RuntimeError("FITS image has invalid dimensions")

            new_header = primary.header.copy()
            # Clear any legacy WCS scale/rotation keys from the original FITS before
            # applying Pixelemon's new solution.  This prevents ambiguous headers where
            # both a CD matrix and CDELT+PC keywords coexist.  The update() call below
            # will then write exactly the keywords Pixelemon's fitted WCS requires.
            for stale_key in (
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",  # CD-matrix convention
                "CDELT1",
                "CDELT2",  # CDELT+PC convention
                "PC1_1",
                "PC1_2",
                "PC2_1",
                "PC2_2",
                "CROTA1",
                "CROTA2",  # deprecated rotation
            ):
                new_header.remove(stale_key, ignore_missing=True)
            # Use Pixelemon's fitted SIP WCS (includes rotation matrix + distortion coefficients).
            # _wcs is always initialised in from_fits_file and updated by fit_wcs_from_points
            # during plate_solve; the assertion guards against unexpected library changes.
            if image._wcs is None:
                raise RuntimeError("Pixelemon _wcs not available after successful plate solve")
            new_header.update(image._wcs.to_header(relax=True))
            new_file = image_path.with_suffix(".new")
            fits.writeto(new_file, primary.data, new_header, overwrite=True)

        return new_file

    def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process image with plate solving.

        Args:
            context: Processing context with image and settings

        Returns:
            ProcessorResult with plate solving outcome
        """
        start_time = time.time()

        if not check_pixelemon():
            return ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason="Pixelemon not available",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )

        try:
            path_to_solve = _ensure_fits_has_observer_location(context.working_image_path, context, context.working_dir)
            context.working_image_path = path_to_solve
            wcs_image_path = self._solve_with_pixelemon(path_to_solve, context)
            context.working_image_path = wcs_image_path

            with fits.open(wcs_image_path) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                header = primary.header
                ra_center = header.get("CRVAL1")
                dec_center = header.get("CRVAL2")
                naxis1 = int(header.get("NAXIS1", 0))  # type: ignore[arg-type]
                naxis2 = int(header.get("NAXIS2", 0))  # type: ignore[arg-type]
                # proj_plane_pixel_scales handles both CDELT and CD-matrix WCS conventions
                try:
                    pixel_scale = float(proj_plane_pixel_scales(WCS(header)).mean()) * 3600
                except Exception:
                    pixel_scale = 0.0

            field_width_deg = naxis1 * pixel_scale / 3600 if pixel_scale and naxis1 > 0 else None
            field_height_deg = naxis2 * pixel_scale / 3600 if pixel_scale and naxis2 > 0 else None

            elapsed = time.time() - start_time

            result = ProcessorResult(
                should_upload=True,
                extracted_data={
                    "plate_solved": True,
                    "ra_center": ra_center,
                    "dec_center": dec_center,
                    "pixel_scale": pixel_scale,
                    "field_width_deg": field_width_deg,
                    "field_height_deg": field_height_deg,
                    "wcs_image_path": str(wcs_image_path),
                },
                confidence=1.0,
                reason=f"Plate solved in {elapsed:.1f}s",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "plate_solver_result.json", result)
            return result

        except Exception as e:
            result = ProcessorResult(
                should_upload=True,
                extracted_data={"plate_solved": False},
                confidence=0.0,
                reason=f"Plate solving failed: {e!s}",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "plate_solver_result.json", result)
            return result
