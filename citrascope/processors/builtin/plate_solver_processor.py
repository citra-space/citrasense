"""Plate solving processor using Pixelemon (Tetra3)."""

from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from pixelemon import Telescope
from pixelemon.optics._base_optical_assembly import BaseOpticalAssembly
from pixelemon.processing import BackgroundSettings, DetectionSettings
from pixelemon.sensors._base_sensor import BaseSensor

from citrascope.processors.abstract_processor import AbstractImageProcessor
from citrascope.processors.artifact_writer import dump_processor_result
from citrascope.processors.processor_result import ProcessingContext, ProcessorResult

from .processor_dependencies import check_pixelemon, normalize_fits_timestamp

if TYPE_CHECKING:
    from pixelemon import TelescopeImage

_MIN_SOURCE_SNR = 3.0


def _detection_settings() -> DetectionSettings:
    """SEP detection settings derived from MSI's tuned SExtractor config.

    These replace Pixelemon's streak_source_defaults (detection_sigma=2,
    deblend_mesh_count=1, deblend_contrast=1.0) which effectively disable
    deblending and cause bright stars to be missed.
    """
    return DetectionSettings(
        detection_sigma=5.0,
        min_pixel_count=3,
        deblend_mesh_count=32,
        deblend_contrast=0.005,
        merge_small_detections=True,
        full_width_half_maximum=5,
        kernel_array_size=13,
    )


def _background_settings() -> BackgroundSettings:
    """SEP background settings derived from MSI's tuned SExtractor config."""
    return BackgroundSettings(
        mesh_count=64,
        filter_size=3,
    )


def _build_telescope_for_image(image_path: Path, context: ProcessingContext | None = None):
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


def _normalize_pixelemon_wcs(header: fits.Header, img_height: int) -> None:
    """Absorb Pixelemon's y-flip into the WCS header so it matches SEP/numpy coordinates.

    Pixelemon's plate_solve fits its WCS with y_wcs = height-1-y_sep before
    calling fit_wcs_from_points (see pixelemon#10).  This produces a WCS whose
    pixel y-axis is in a different coordinate system from SEP, numpy, and
    Astropy's 0-based indexing — all of which share y=0 at the first stored row.

    This function absorbs the y-flip into the WCS parameters so the resulting
    header maps raw SEP pixel coordinates directly to sky coordinates, consistent
    with the coordinate system used by the rest of the pipeline.

    The transform is: y_new = H + 1 - y_old  (FITS 1-based)

    which negates the y-column of the CD matrix, shifts CRPIX2, and flips the
    sign of SIP polynomial coefficients by their y-power parity.
    """
    # Linear WCS: shift CRPIX2 and negate the y-column of the rotation matrix.
    # Astropy's fit_wcs_from_points may produce either CD matrix or PC+CDELT;
    # handle whichever representation is present.
    # FITS header values are floats at runtime; casts satisfy pyright.
    header["CRPIX2"] = img_height + 1 - float(header["CRPIX2"])  # type: ignore[arg-type]
    if "CD1_2" in header:
        header["CD1_2"] = -float(header["CD1_2"])  # type: ignore[arg-type]
        header["CD2_2"] = -float(header["CD2_2"])  # type: ignore[arg-type]
    elif "PC1_2" in header:
        header["PC1_2"] = -float(header["PC1_2"])  # type: ignore[arg-type]
        header["PC2_2"] = -float(header["PC2_2"])  # type: ignore[arg-type]
    else:
        # CDELT-only (no rotation): negate CDELT2
        if "CDELT2" in header:
            header["CDELT2"] = -float(header["CDELT2"])  # type: ignore[arg-type]

    # SIP distortion: v_new = -v_old, so each coefficient picks up (-1)^q
    # from its y-power q.  The B (y-distortion) and BP (inverse y) polynomials
    # get an additional negation because the distortion direction flips.
    for prefix, extra_negate in (("A", False), ("B", True), ("AP", False), ("BP", True)):
        order_key = f"{prefix}_ORDER"
        if order_key not in header:
            continue
        order = int(header[order_key])  # type: ignore[arg-type]
        for p in range(order + 1):
            for q in range(order + 1 - p):
                key = f"{prefix}_{p}_{q}"
                if key not in header:
                    continue
                sign = (-1) ** q
                if extra_negate:
                    sign = -sign
                if sign == -1:
                    header[key] = -float(header[key])  # type: ignore[arg-type]


class PlateSolverProcessor(AbstractImageProcessor):
    """
    Plate solving processor using Pixelemon (Tetra3).

    Determines exact telescope pointing and embeds WCS (World Coordinate System)
    into a *_wcs.fits file. Updates context.working_image_path to point to that file.

    Typical processing time: a few seconds (Tetra3).
    """

    _tetra_loaded: bool = False

    @classmethod
    def warm_cache(cls, logger: logging.Logger | None = None) -> None:
        """Pre-download and load the Tetra3 star database so the first solve is fast."""
        if cls._tetra_loaded:
            return
        try:
            from pixelemon import TetraSolver

            if logger:
                logger.info("Pre-loading Tetra3 star database...")
            TetraSolver.high_memory()
            cls._tetra_loaded = True
            if logger:
                logger.info("Tetra3 star database loaded")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to pre-load Tetra3 database (will retry on first solve): {e}")

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

    def _solve_with_pixelemon(
        self, image_path: Path, context: ProcessingContext | None = None
    ) -> tuple[Path, TelescopeImage]:
        """Run Pixelemon (Tetra3) plate solve and write WCS to a _wcs.fits file.

        Pixelemon internally fits a full 5th-degree SIP WCS from matched star centroids
        (equivalent to astrometry.net quality).  However, Pixelemon fits the WCS with
        y-flipped pixel coordinates (y_wcs = height-1-y_sep; see pixelemon#10), so the
        raw WCS is non-standard.  We normalize it via _normalize_pixelemon_wcs before
        writing to disk, producing a standard FITS WCS that maps raw SEP pixel coords
        to sky coordinates.  All downstream consumers can use the WCS without any
        Pixelemon-specific workarounds.

        Returns:
            (wcs_fits_path, telescope_image) — the WCS FITS and the Pixelemon image
            whose detections and _wcs are still available for source extraction.

        Raises:
            RuntimeError: If plate solving fails or solution is None
        """
        from pixelemon import TelescopeImage as _TelescopeImage
        from pixelemon import TetraSolver

        if not PlateSolverProcessor._tetra_loaded:
            TetraSolver.high_memory()
            PlateSolverProcessor._tetra_loaded = True
        telescope = _build_telescope_for_image(image_path, context)
        image = _TelescopeImage.from_fits_file(image_path, telescope)
        image.detection_settings = _detection_settings()
        image.background_settings = _background_settings()
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
            for stale_key in (
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",
                "CDELT1",
                "CDELT2",
                "PC1_1",
                "PC1_2",
                "PC2_1",
                "PC2_2",
                "CROTA1",
                "CROTA2",
            ):
                new_header.remove(stale_key, ignore_missing=True)
            if image._wcs is None:
                raise RuntimeError("Pixelemon _wcs not available after successful plate solve")
            new_header.update(image._wcs.to_header(relax=True))
            _normalize_pixelemon_wcs(new_header, naxis2)
            new_file = image_path.with_stem(image_path.stem + "_wcs").with_suffix(".fits")
            fits.writeto(new_file, primary.data, new_header, overwrite=True)

        return new_file, image

    @staticmethod
    def _extract_sources(image: TelescopeImage, wcs: WCS) -> pd.DataFrame:
        """Convert Pixelemon SEP detections to a sky-coordinate source catalog.

        Reuses the detections already computed during plate solving so there is
        no redundant source-extraction pass.  Applies a per-source SNR floor
        (_MIN_SOURCE_SNR) so downstream processors only receive statistically
        reliable detections.
        """
        objects = image.detections._objects

        flux = objects["flux"]
        npix = objects["npix"]
        noise = image.background.globalrms * np.sqrt(npix.astype(np.float64))
        snr = flux / noise
        mask = snr >= _MIN_SOURCE_SNR

        x = objects["x"][mask]
        y = objects["y"][mask]
        ra, dec = wcs.all_pix2world(x, y, 0)

        return pd.DataFrame(
            {
                "ra": np.asarray(ra, dtype=np.float64),
                "dec": np.asarray(dec, dtype=np.float64),
                "mag": np.asarray(-2.5 * np.log10(flux[mask] / image.exposure_time), dtype=np.float64),
                "magerr": np.zeros(int(mask.sum()), dtype=np.float64),
                "elongation": np.asarray((objects["a"] / objects["b"])[mask], dtype=np.float64),
            }
        )

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
            wcs_image_path, pix_image = self._solve_with_pixelemon(path_to_solve, context)
            context.working_image_path = wcs_image_path

            with fits.open(wcs_image_path) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                header = primary.header
                ra_center = header.get("CRVAL1")
                dec_center = header.get("CRVAL2")
                naxis1 = int(header.get("NAXIS1", 0))  # type: ignore[arg-type]
                naxis2 = int(header.get("NAXIS2", 0))  # type: ignore[arg-type]
                try:
                    wcs_obj = WCS(header)
                    pixel_scale = float(proj_plane_pixel_scales(wcs_obj).mean()) * 3600
                except Exception:
                    pixel_scale = 0.0
                    wcs_obj = WCS(header)

            field_width_deg = naxis1 * pixel_scale / 3600 if pixel_scale and naxis1 > 0 else None
            field_height_deg = naxis2 * pixel_scale / 3600 if pixel_scale and naxis2 > 0 else None

            # Extract source catalog from Pixelemon's SEP detections (reuses the
            # detections already computed during plate solving — no second pass).
            sources_df = self._extract_sources(pix_image, wcs_obj)
            context.detected_sources = sources_df
            num_sources = len(sources_df)

            # Write an output.cat artifact for debugging / offline inspection
            self._write_source_catalog(sources_df, context.working_dir / "output.cat")

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
                    "num_sources": num_sources,
                },
                confidence=1.0,
                reason=f"Plate solved in {elapsed:.1f}s, {num_sources} sources extracted",
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

    @staticmethod
    def _write_source_catalog(sources_df: pd.DataFrame, path: Path) -> None:
        """Write a SExtractor-compatible ASCII catalog for artifact preservation."""
        with open(path, "w") as f:
            f.write("#   1 MAG          Instrumental magnitude\n")
            f.write("#   2 MAGERR       Magnitude error\n")
            f.write("#   3 RA           Right ascension (J2000) [deg]\n")
            f.write("#   4 DEC          Declination (J2000) [deg]\n")
            f.write("#   5 ELONGATION   a/b axis ratio\n")
            for _, row in sources_df.iterrows():
                f.write(
                    f"  {row['mag']:12.4f} {row['magerr']:12.4f}"
                    f" {row['ra']:14.7f} {row['dec']:+14.7f} {row['elongation']:8.3f}\n"
                )
