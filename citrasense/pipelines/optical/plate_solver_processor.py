"""Plate solving processor using astrometry.net (solve-field)."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from citrasense.pipelines.common.abstract_processor import AbstractImageProcessor
from citrasense.pipelines.common.artifact_writer import dump_processor_result
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import ProcessorResult

from .processor_dependencies import check_astrometry

if TYPE_CHECKING:
    pass

_logger = logging.getLogger("citrasense.PlateSolver")


def _compute_plate_scale(telescope_record: dict, x_binning: int = 1, y_binning: int = 1) -> float | None:
    """Compute plate scale in arcsec/pixel from telescope_record and binning.

    Returns None if the required fields are missing.
    """
    pixel_size = telescope_record.get("pixelSize")
    focal_length = telescope_record.get("focalLength")
    if not pixel_size or not focal_length:
        return None
    binning = max(x_binning, y_binning)
    return 206265.0 * (float(pixel_size) * binning / 1000.0) / float(focal_length)


def _parse_solve_quality(stdout: str) -> dict[str, Any]:
    """Extract solve quality metrics from solve-field stdout.

    Parses the log-odds ratio, match/conflict/distractor counts, pixel scale,
    solving index, number of sources extracted, and field geometry from
    astrometry.net's text output.
    """
    quality: dict[str, Any] = {}

    m = re.search(
        r"log-odds ratio ([\d.e+\-]+) \(([^)]+)\),\s*"
        r"(\d+) match,\s*(\d+) conflict,\s*(\d+) distractors?,\s*(\d+) index",
        stdout,
    )
    if m:
        quality["log_odds"] = float(m.group(1))
        quality["log_odds_scientific"] = m.group(2)
        quality["n_match"] = int(m.group(3))
        quality["n_conflict"] = int(m.group(4))
        quality["n_distractor"] = int(m.group(5))
        quality["n_index"] = int(m.group(6))

    m = re.search(r"pixel scale ([\d.]+) arcsec/pix", stdout)
    if m:
        quality["solved_pixel_scale"] = float(m.group(1))

    m = re.search(r"solved with index (\S+)", stdout)
    if m:
        quality["solved_index"] = m.group(1)

    m = re.search(r"simplexy: found (\d+) sources", stdout)
    if m:
        quality["n_sources_extracted"] = int(m.group(1))

    m = re.search(r"Field rotation angle: up is ([\d.+-]+) degrees", stdout)
    if m:
        quality["field_rotation_deg"] = float(m.group(1))

    m = re.search(r"Field parity: (\w+)", stdout)
    if m:
        quality["field_parity"] = m.group(1)

    return quality


class PlateSolverProcessor(AbstractImageProcessor):
    """Plate solving processor using astrometry.net (solve-field).

    Determines exact telescope pointing and embeds WCS (World Coordinate System)
    into a solved FITS file. Updates context.working_image_path to point to that file.
    """

    name = "plate_solver"
    friendly_name = "Plate Solver"
    description = "Astrometric calibration via astrometry.net solve-field (determines exact pointing and WCS)"

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
        import tempfile

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
                logger=_logger,
            )
            result = cls().process(context)
            if result.extracted_data.get("plate_solved"):
                ra = result.extracted_data.get("ra_center")
                dec = result.extracted_data.get("dec_center")
                if ra is not None and dec is not None:
                    return float(ra), float(dec)
        return None

    def _solve_field(
        self,
        image_path: Path,
        context: ProcessingContext,
    ) -> tuple[Path, dict[str, Any]]:
        """Run astrometry.net solve-field and return the solved FITS path + quality metrics.

        Passes RA/Dec and plate scale hints from context when available for
        faster, more reliable solves. Falls back to a blind solve if hints
        are unavailable.

        Returns:
            (solved_fits_path, solve_quality_dict).

        Raises:
            RuntimeError: If solve-field is not found, fails, or times out.
        """
        timeout = 60
        if context.settings and hasattr(context.settings, "plate_solve_timeout"):
            timeout = int(context.settings.plate_solve_timeout)

        solved_path = context.working_dir / (image_path.stem + "_wcs.fits")

        cmd: list[str] = [
            "solve-field",
            str(image_path),
            "--cpulimit",
            str(timeout),
            "--overwrite",
            "--no-plots",
            "--crpix-center",
            "--new-fits",
            str(solved_path),
        ]

        # Index directory hint
        index_path = ""
        if context.settings and hasattr(context.settings, "astrometry_index_path"):
            index_path = context.settings.astrometry_index_path or ""
        if index_path:
            cmd.extend(["--index-dir", index_path])

        # Pointing hints from task coordinates or FITS header
        ra_hint, dec_hint = None, None
        if context.task and hasattr(context.task, "ra") and hasattr(context.task, "dec"):
            ra_hint = getattr(context.task, "ra", None)
            dec_hint = getattr(context.task, "dec", None)

        if ra_hint is None or dec_hint is None:
            with fits.open(image_path) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                header = primary.header
                ra_hint = ra_hint or header.get("OBJRA") or header.get("RA")  # type: ignore[assignment]
                dec_hint = dec_hint or header.get("OBJDEC") or header.get("DEC")  # type: ignore[assignment]

        if ra_hint is not None and dec_hint is not None:
            try:
                cmd.extend(["--ra", str(float(ra_hint)), "--dec", str(float(dec_hint)), "--radius", "10"])  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass

        # Plate scale hints from telescope_record + FITS binning
        if context.telescope_record:
            with fits.open(image_path) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                header = primary.header
                x_bin = int(header.get("XBINNING", 1))  # type: ignore[arg-type]
                y_bin = int(header.get("YBINNING", 1))  # type: ignore[arg-type]

            scale = _compute_plate_scale(context.telescope_record, x_bin, y_bin)
            if scale is not None:
                scale_low = scale * 0.9
                scale_high = scale * 1.1
                cmd.extend(
                    [
                        "--scale-low",
                        f"{scale_low:.4f}",
                        "--scale-high",
                        f"{scale_high:.4f}",
                        "--scale-units",
                        "arcsecperpix",
                    ]
                )

        # Downsample large images for faster source extraction
        with fits.open(image_path) as hdul:
            primary = hdul[0]
            assert isinstance(primary, fits.PrimaryHDU)
            naxis1 = int(primary.header.get("NAXIS1", 0))  # type: ignore[arg-type]
            naxis2 = int(primary.header.get("NAXIS2", 0))  # type: ignore[arg-type]
        if naxis1 > 4000 or naxis2 > 4000:
            cmd.extend(["--downsample", "2"])

        logger = context.logger or _logger
        logger.info(f"Running solve-field: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 30,
            )
        except FileNotFoundError as e:
            raise RuntimeError("solve-field not found. Install with: brew install astrometry-net") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"solve-field timed out after {timeout + 30}s") from e

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            raise RuntimeError(f"solve-field failed (exit {result.returncode}): {stderr_tail}")

        if not solved_path.exists():
            raise RuntimeError("solve-field completed but solved FITS not found — image may be unsolvable")

        solve_quality = _parse_solve_quality(result.stdout or "")
        if solve_quality:
            parts = []
            if "log_odds" in solve_quality:
                parts.append(f"log-odds={solve_quality['log_odds']:.1f}")
            if "n_match" in solve_quality:
                parts.append(f"{solve_quality['n_match']} match / {solve_quality.get('n_conflict', '?')} conflict")
            if "solved_index" in solve_quality:
                parts.append(f"index={solve_quality['solved_index']}")
            logger.info("Solve quality: %s", ", ".join(parts))

        return solved_path, solve_quality

    @staticmethod
    def _compute_hfr_from_image(context: ProcessingContext) -> float | None:
        """Compute median HFR from the image using the autofocus SEP routine.

        Uses context.image_data (pre-loaded) or falls back to reading from the
        working image path.  Returns None if computation fails or too few stars.
        """
        try:
            from citrasense.hardware.direct.autofocus import compute_hfr

            img = context.image_data
            if img is None:
                with fits.open(context.working_image_path) as hdul:
                    primary = hdul[0]
                    assert isinstance(primary, fits.PrimaryHDU)
                    img = primary.data
            if img is None:
                _logger.debug("HFR: image_data is None, skipping")
                return None
            hfr = compute_hfr(img, crop_ratio=0.5)
            if hfr is None:
                hfr = compute_hfr(img, crop_ratio=1.0)
            if hfr is not None:
                _logger.debug(f"HFR computation: {hfr:.2f}")
            else:
                _logger.debug("HFR: too few stars even at full frame")
            return hfr
        except Exception as e:
            _logger.warning(f"HFR computation failed: {e}")
            return None

    def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process image with plate solving via astrometry.net solve-field."""
        start_time = time.time()

        if not check_astrometry():
            return ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason="astrometry.net (solve-field) not available",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )

        try:
            wcs_image_path, solve_quality = self._solve_field(context.working_image_path, context)
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

            field_width_deg = naxis1 * pixel_scale / 3600 if pixel_scale and naxis1 > 0 else None
            field_height_deg = naxis2 * pixel_scale / 3600 if pixel_scale and naxis2 > 0 else None

            elapsed = time.time() - start_time

            hfr_median = self._compute_hfr_from_image(context)

            extracted = {
                "plate_solved": True,
                "ra_center": ra_center,
                "dec_center": dec_center,
                "pixel_scale": pixel_scale,
                "field_width_deg": field_width_deg,
                "field_height_deg": field_height_deg,
                "wcs_image_path": str(wcs_image_path),
                "hfr_median": hfr_median,
            }
            if solve_quality:
                extracted["solve_quality"] = solve_quality

            result = ProcessorResult(
                should_upload=True,
                extracted_data=extracted,
                confidence=1.0,
                reason=f"Plate solved in {elapsed:.1f}s",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "plate_solver_result.json", result)
            return result

        except Exception as e:
            hfr_median = None
            try:
                hfr_median = self._compute_hfr_from_image(context)
            except Exception:
                pass
            result = ProcessorResult(
                should_upload=True,
                extracted_data={"plate_solved": False, "hfr_median": hfr_median},
                confidence=0.0,
                reason=f"Plate solving failed: {e!s}",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "plate_solver_result.json", result)
            return result
