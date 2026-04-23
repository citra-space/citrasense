"""Source extraction processor using SExtractor."""

import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from astropy.io import fits

from citrasense.pipelines.common.abstract_processor import AbstractImageProcessor
from citrasense.pipelines.common.artifact_writer import dump_processor_result
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import ProcessorResult
from citrasense.pipelines.optical.optical_processing_context import OpticalProcessingContext

from .processor_dependencies import check_sextractor


class SourceExtractorProcessor(AbstractImageProcessor):
    """
    Source extraction processor using SExtractor.

    Detects all sources (stars and satellites) in the image and extracts
    their positions, magnitudes, and FWHM. Requires plate-solved image with WCS.

    Typical processing time: 2-5 seconds.
    """

    name = "source_extractor"
    friendly_name = "Source Extractor"
    description = "Detect stars and satellites via SExtractor (requires plate-solved image)"

    def _parse_sex_catalog(self, catalog_path: Path) -> pd.DataFrame:
        """Parse SExtractor catalog into DataFrame.

        Column layout from default.param (0-indexed):
          0=NUMBER, 1=EXT_NUMBER, 2=FLUX_AUTO, 3=FLUXERR_AUTO,
          4=MAG_AUTO, 5=MAGERR_AUTO, 6=X_IMAGE, 7=Y_IMAGE,
          8=ALPHA_J2000, 9=DELTA_J2000, 10=FWHM_IMAGE,
          11=ELONGATION, 12=ELLIPTICITY

        Returns:
            DataFrame with columns: ra, dec, mag, magerr, fwhm, elongation
        """
        sources = []

        with open(catalog_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue

                cols = line.split()
                if len(cols) < 12:
                    continue

                try:
                    sources.append(
                        {
                            "ra": float(cols[8]),  # ALPHA_J2000
                            "dec": float(cols[9]),  # DELTA_J2000
                            "mag": float(cols[4]),  # MAG_AUTO
                            "magerr": float(cols[5]),  # MAGERR_AUTO
                            "fwhm": float(cols[10]),  # FWHM_IMAGE
                            "elongation": float(cols[11]),  # ELONGATION
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(sources)

    def _extract_sources(
        self,
        image_path: Path,
        config_dir: Path,
        working_dir: Path,
        logger=None,
        *,
        detect_thresh: float | None = None,
        detect_minarea: int | None = None,
        filter_name: str | None = None,
    ) -> pd.DataFrame:
        """Run SExtractor and parse catalog.

        Args:
            image_path: Path to FITS image (should be plate-solved with WCS)
            config_dir: Path to directory containing SExtractor config files
            working_dir: Directory for temporary files (catalog will be written here)
            logger: Logger instance (falls back to module-level citrasense logger if None)
            detect_thresh: Override DETECT_THRESH (sigmas). None keeps default.sex value.
            detect_minarea: Override DETECT_MINAREA (pixels). None keeps default.sex value.
            filter_name: Convolution kernel name (without .conv extension).
                ``"default"`` or ``None`` uses ``default.conv``; other values
                select the corresponding ``<name>.conv`` file from config_dir.

        Returns:
            DataFrame with columns: ra, dec, mag, magerr, fwhm

        Raises:
            RuntimeError: If SExtractor fails
        """
        if logger is None:
            from citrasense.logging import CITRASENSE_LOGGER

            logger = CITRASENSE_LOGGER
        # Ensure paths are absolute
        image_path = image_path.resolve()
        config_dir = config_dir.resolve()
        working_dir = working_dir.resolve()

        # Create a symlink in working_dir with a simple name (no spaces in path)
        image_symlink = working_dir / "input.fits"
        catalog_name = "output.cat"
        catalog_path = working_dir / catalog_name

        # Determine which config files need to be copied
        config_files_to_copy = ["default.sex", "default.param", "default.conv", "default.nnw"]
        if filter_name and filter_name != "default":
            conv_file = f"{filter_name}.conv"
            if (config_dir / conv_file).exists():
                config_files_to_copy.append(conv_file)
            else:
                logger.warning("Convolution kernel %s not found in %s, using default.conv", conv_file, config_dir)
                filter_name = None

        try:
            # Create symlink to avoid passing image path with spaces to SExtractor
            if image_symlink.exists():
                image_symlink.unlink()
            image_symlink.symlink_to(image_path)

            # Copy config files to working_dir so relative paths work
            for config_file in config_files_to_copy:
                src = config_dir / config_file
                dst = working_dir / config_file
                if not dst.exists():
                    shutil.copy(src, dst)

            # Build SExtractor command - all files in working_dir now
            cmd = [
                "sex",
                "input.fits",
                "-c",
                "default.sex",
                "-CATALOG_NAME",
                "output.cat",
            ]

            # Apply user-configurable overrides via CLI flags (override default.sex values)
            if detect_thresh is not None:
                cmd.extend(["-DETECT_THRESH", str(detect_thresh), "-ANALYSIS_THRESH", str(detect_thresh)])
            if detect_minarea is not None:
                cmd.extend(["-DETECT_MINAREA", str(detect_minarea)])
            if filter_name and filter_name != "default":
                cmd.extend(["-FILTER_NAME", f"{filter_name}.conv"])

            logger.info("Running SExtractor from cwd: %s", working_dir)
            logger.info("SExtractor command: %s", " ".join(cmd))

            # Try 'sex' command first (most common)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(working_dir))

                # If sex not found, try 'source-extractor' alias
                if result.returncode == 127 or "not found" in (result.stderr or "").lower():
                    cmd[0] = "source-extractor"
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(working_dir))
            except FileNotFoundError:
                # Command doesn't exist, try alternate name
                cmd[0] = "source-extractor"
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(working_dir))
                except FileNotFoundError as e:
                    raise RuntimeError("SExtractor not found. Install with: brew install sextractor") from e

            if result.returncode != 0:
                raise RuntimeError(f"SExtractor failed: {result.stderr}")

        finally:
            # Clean up symlink and any config files copied into working_dir
            if image_symlink.exists():
                image_symlink.unlink()
            for config_file in config_files_to_copy:
                cfg = working_dir / config_file
                if cfg.exists():
                    cfg.unlink()

        # Parse catalog
        sources = self._parse_sex_catalog(catalog_path)
        return sources

    def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process image with source extraction.

        Args:
            context: Processing context with image and settings

        Returns:
            ProcessorResult with source extraction outcome
        """
        assert isinstance(context, OpticalProcessingContext)
        start_time = time.time()

        # Check if image has WCS (requires plate solver to have run)
        try:
            with fits.open(context.working_image_path) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                if "CRVAL1" not in primary.header:
                    return ProcessorResult(
                        should_upload=True,
                        extracted_data={},
                        confidence=0.0,
                        reason="Image not plate-solved (WCS missing)",
                        processing_time_seconds=time.time() - start_time,
                        processor_name=self.name,
                    )
        except Exception as e:
            return ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason=f"Could not read image: {e}",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )

        # Check dependencies
        if not check_sextractor():
            return ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason="SExtractor not installed",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )

        try:
            config_dir = Path(__file__).parent / "sextractor_configs"

            detect_thresh: float | None = None
            detect_minarea: int | None = None
            filter_name: str | None = None
            if context.settings is not None:
                detect_thresh = getattr(context.settings, "sextractor_detect_thresh", None)
                detect_minarea = getattr(context.settings, "sextractor_detect_minarea", None)
                filter_name = getattr(context.settings, "sextractor_filter_name", None)

            sources_df = self._extract_sources(
                context.working_image_path,
                config_dir,
                context.working_dir,
                logger=context.logger,
                detect_thresh=detect_thresh,
                detect_minarea=detect_minarea,
                filter_name=filter_name,
            )
            context.detected_sources = sources_df

            elapsed = time.time() - start_time

            fwhm_stats = {}
            if len(sources_df) and "fwhm" in sources_df.columns:
                fwhm = sources_df["fwhm"]
                fwhm_stats = {
                    "fwhm_min": float(fwhm.min()),
                    "fwhm_max": float(fwhm.max()),
                    "fwhm_median": float(fwhm.median()),
                    "fwhm_mean": float(fwhm.mean()),
                    "count_fwhm_lt_1_5": int((fwhm < 1.5).sum()),
                    "count_fwhm_gte_1_5": int((fwhm >= 1.5).sum()),
                }

            elongation_stats = {}
            if len(sources_df) and "elongation" in sources_df.columns:
                elong = sources_df["elongation"]
                elongation_stats = {"elongation_median": float(elong.median())}

            result = ProcessorResult(
                should_upload=True,
                extracted_data={
                    "num_sources": len(sources_df),
                    "sources_catalog": str(context.working_dir / "output.cat"),
                    "detect_thresh": detect_thresh,
                    "detect_minarea": detect_minarea,
                    "filter_name": filter_name or "default",
                },
                confidence=1.0,
                reason=f"Extracted {len(sources_df)} sources in {elapsed:.1f}s",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )
            dump_processor_result(
                context.working_dir, "source_extractor_result.json", result, extra={**fwhm_stats, **elongation_stats}
            )
            return result

        except Exception as e:
            result = ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason=f"Source extraction failed: {e!s}",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "source_extractor_result.json", result)
            return result
