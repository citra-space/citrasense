"""Photometric calibration processor using APASS catalog."""

import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import KDTree

from citrascope.processors.abstract_processor import AbstractImageProcessor
from citrascope.processors.artifact_writer import dump_csv, dump_processor_result
from citrascope.processors.processor_result import ProcessingContext, ProcessorResult


class PhotometryProcessor(AbstractImageProcessor):
    """
    Photometric calibration processor using APASS catalog.

    Queries APASS all-sky catalog, cross-matches detected sources with catalog stars,
    and calculates magnitude zero point. Requires source extraction to have run.

    Typical processing time: 2-5 seconds.
    """

    name = "photometry"
    friendly_name = "Photometry Calibrator"
    description = "Photometric calibration via APASS catalog (requires source extraction)"

    def _query_apass(self, ra: float, dec: float, radius: float = 2.0) -> pd.DataFrame:
        """Query APASS catalog via AAVSO.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees (default: 2.0)

        Returns:
            DataFrame with APASS stars

        Raises:
            RuntimeError: If query fails
        """
        url = "https://www.aavso.org/cgi-bin/apass_dr10_download.pl"

        form_data = {
            "ra": str(ra),
            "dec": str(dec),
            "radius": str(radius),
            "outtype": "1",  # CSV format
        }

        try:
            response = requests.post(url, data=form_data, timeout=30)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"APASS query failed: {e}") from e

        # Parse CSV response
        try:
            # APASS returns CSV with header
            apass_df = pd.read_csv(StringIO(response.text))

            # Check if we got valid data
            if apass_df.empty:
                raise RuntimeError("APASS query returned no results")

            return apass_df

        except Exception as e:
            raise RuntimeError(f"Failed to parse APASS response: {e}") from e

    def _cross_match_catalogs(
        self, sources: pd.DataFrame, catalog: pd.DataFrame, max_separation: float
    ) -> pd.DataFrame:
        """Cross-match two catalogs using KDTree.

        Args:
            sources: DataFrame with detected sources (columns: ra, dec)
            catalog: DataFrame with catalog stars (columns: radeg, decdeg)
            max_separation: Maximum separation in degrees

        Returns:
            DataFrame with matched sources and catalog data concatenated
        """
        # Build KDTree from catalog coordinates
        coords_catalog = catalog[["radeg", "decdeg"]].values
        tree = KDTree(coords_catalog)

        # Query tree with source coordinates
        coords_sources = sources[["ra", "dec"]].values
        distances, indices = tree.query(coords_sources, distance_upper_bound=max_separation)

        # Filter to valid matches
        valid = distances < max_separation

        if not valid.any():
            return pd.DataFrame()

        # Concatenate matched rows
        matched_indices = np.asarray(indices)[valid]
        matched = pd.concat(
            [sources.iloc[valid].reset_index(drop=True), catalog.iloc[matched_indices].reset_index(drop=True)],
            axis=1,
        )

        return matched

    def _calibrate_photometry(
        self, sources: pd.DataFrame, image_path: Path, filter_name: str
    ) -> tuple[float, int, pd.DataFrame, pd.DataFrame]:
        """Query APASS catalog and calculate magnitude zero point.

        Args:
            sources: DataFrame with detected sources (columns: ra, dec, mag)
            image_path: Path to FITS image (for WCS info)
            filter_name: Filter name (Clear, g, r, i)

        Returns:
            Tuple of (zero_point, num_matched_stars, apass_catalog, crossmatch_df)

        Raises:
            RuntimeError: If calibration fails
        """
        # Get field center from WCS
        with fits.open(image_path) as hdul:
            primary = hdul[0]
            assert isinstance(primary, fits.PrimaryHDU)
            wcs = WCS(primary.header)
            header = primary.header
            nx = int(header["NAXIS1"])  # type: ignore[arg-type]
            ny = int(header["NAXIS2"])  # type: ignore[arg-type]
            center = wcs.pixel_to_world(nx / 2, ny / 2)
            ra_center = float(center.ra.deg)  # type: ignore[union-attr]
            dec_center = float(center.dec.deg)  # type: ignore[union-attr]

        # Query APASS catalog
        apass_stars = self._query_apass(ra_center, dec_center, radius=2.0)

        if apass_stars.empty:
            raise RuntimeError("No APASS stars found in field")

        # Cross-match detected sources with APASS
        matched = self._cross_match_catalogs(sources, apass_stars, max_separation=1.0 / 60.0)

        if matched.empty or len(matched) < 3:
            raise RuntimeError(f"Insufficient matched stars for calibration: {len(matched)}")

        # Calculate zero point for specified filter
        filter_col = {"Clear": "Johnson_V (V)", "g": "Sloan_g (SG)", "r": "Sloan_r (SR)", "i": "Sloan_i (SI)"}.get(
            filter_name, "Johnson_V (V)"
        )

        # Convert to numeric and drop NaN
        matched["mag"] = pd.to_numeric(matched["mag"], errors="coerce")
        matched[filter_col] = pd.to_numeric(matched[filter_col], errors="coerce")
        matched_clean = matched.dropna(subset=["mag", filter_col])

        if len(matched_clean) < 3:
            raise RuntimeError(f"Insufficient valid stars after cleaning: {len(matched_clean)}")

        # Calculate zero point (median difference between catalog and instrumental mags)
        zero_point = float(np.nanmedian(matched_clean[filter_col] - matched_clean["mag"]))

        return zero_point, len(matched_clean), apass_stars, matched_clean

    def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process image with photometric calibration.

        Args:
            context: Processing context with image and settings

        Returns:
            ProcessorResult with photometry outcome
        """
        start_time = time.time()

        # Check if sources were extracted
        catalog_path = context.working_dir / "output.cat"
        if not catalog_path.exists():
            return ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason="Source catalog not found (source extraction must run first)",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )

        try:
            # Load source catalog (SExtractor format: no header, cols 4=mag 5=magerr 8=ra 9=dec 10=fwhm)
            sources_df = pd.read_csv(
                catalog_path,
                sep=r"\s+",
                comment="#",
                header=None,
                usecols=[4, 5, 8, 9, 10],
                names=["mag", "magerr", "ra", "dec", "fwhm"],
            )

            # Get filter name
            filter_name = context.task.assigned_filter_name if context.task else None

            # Calibrate
            zero_point, num_matched, apass_catalog, crossmatch_df = self._calibrate_photometry(
                sources_df, context.working_image_path, filter_name or "Clear"
            )

            elapsed = time.time() - start_time

            result = ProcessorResult(
                should_upload=True,
                extracted_data={
                    "zero_point": zero_point,
                    "num_calibration_stars": num_matched,
                    "filter": filter_name,
                },
                confidence=1.0 if num_matched >= 10 else 0.5,
                reason=f"Calibrated with {num_matched} stars (ZP={zero_point:.2f}) in {elapsed:.1f}s",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )

            dump_processor_result(context.working_dir, "photometry_result.json", result)
            dump_csv(context.working_dir, "photometry_apass_catalog.csv", apass_catalog)
            dump_csv(context.working_dir, "photometry_crossmatch.csv", crossmatch_df)

            return result

        except Exception as e:
            result = ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason=f"Photometry failed: {e!s}",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "photometry_result.json", result)
            return result
