"""Unit tests for MSI science processors."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from astropy.io import fits

# CitraScope repo root (tests/unit -> tests -> root). Used so demo-FITS tests run with stable cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def run_from_repo_root():
    """Run test with cwd = citrascope repo root, then restore. Makes demo-FITS tests match terminal runs."""
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


from citrascope.elset_cache import ElsetCache
from citrascope.processors.builtin.photometry_processor import PhotometryProcessor
from citrascope.processors.builtin.plate_solver_processor import PlateSolverProcessor
from citrascope.processors.builtin.processor_dependencies import (
    check_pixelemon,
    check_sextractor,
)
from citrascope.processors.builtin.satellite_matcher_processor import SatelliteMatcherProcessor
from citrascope.processors.builtin.source_extractor_processor import SourceExtractorProcessor
from citrascope.processors.processor_registry import ProcessorRegistry
from citrascope.processors.processor_result import ProcessingContext, ProcessorResult

# Telescope record matching the PlaneWave scope used to capture the demo FITS.
# pixelSize/focalLength are null in the API (pre-sensor-specs migration), so
# _build_telescope_for_image derives focal_length from legacyFieldOfView + FITS XPIXSZ.
_PLANEWAVE_TELESCOPE_RECORD = {
    "id": "4861f510-e3bd-470f-ab31-98766ecba2ca",
    "angularNoise": 1.0,
    "legacyFieldOfView": 2.0,
    # Physical sensor specs (Moravian USB Camera, XBINNING=4 in demo FITS).
    # XPIXSZ=15.04 μm (effective) / XBINNING=4 → physical pixel = 3.76 μm.
    # Unbinned counts = NAXIS1*4 × NAXIS2*4 = 14208 × 10656.
    # focalLength derived from legacyFieldOfView=2.0° and effective sensor:
    #   f = (3552 × 15.04e-3) / (2 × tan(1°)) ≈ 1530 mm
    # focalRatio = f/D = 1530 / 356 ≈ 4.3 (CDK14 356mm aperture with ~0.6× reducer)
    "pixelSize": 3.76,
    "focalLength": 1530.0,
    "focalRatio": 4.3,
    "horizontalPixelCount": 14208,
    "verticalPixelCount": 10656,
    "imageCircleDiameter": None,
    "spectralMinWavelengthNm": None,
    "spectralMaxWavelengthNm": None,
}


def _first_tle_from_file(tle_path: Path) -> list:
    """Read first two-line TLE pair from a space-track-style file. Returns [line1, line2] or []."""
    if not tle_path.exists():
        return []
    with open(tle_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("1 "):
                line1 = line
                next_line = f.readline()
                if next_line and next_line.startswith("2 "):
                    return [line1, next_line.rstrip("\n")]
                break
    return []


def _first_three_elsets_from_file(tle_path: Path) -> list:
    """Read first three TLE pairs from file; return processor-ready list of {satellite_id, name, tle}."""
    if not tle_path.exists():
        return []
    result = []
    with open(tle_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("1 "):
                line1 = line
                next_line = f.readline()
                if next_line and next_line.startswith("2 "):
                    line2 = next_line.rstrip("\n")
                    norad = line2[2:7].strip() if len(line2) >= 7 else "00000"
                    result.append({"satellite_id": norad, "name": f"SAT-{norad}", "tle": [line1, line2]})
                    if len(result) >= 3:
                        break
    return result


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    return settings


@pytest.fixture
def mock_context(tmp_path, mock_settings):
    """Create a mock processing context."""
    image_path = tmp_path / "test_image.fits"
    working_dir = tmp_path / "working"
    working_dir.mkdir(exist_ok=True)

    return ProcessingContext(
        image_path=image_path,
        working_image_path=image_path,
        working_dir=working_dir,
        image_data=None,
        task=Mock(satelliteName="TEST SAT", satelliteId="12345", assigned_filter_name="Clear"),
        telescope_record=None,
        ground_station_record=None,
        settings=mock_settings,
        logger=Mock(),
    )


class TestPlateSolverProcessor:
    """Tests for PlateSolverProcessor."""

    def test_processor_metadata(self):
        """Test processor has correct metadata."""
        processor = PlateSolverProcessor()
        assert processor.name == "plate_solver"
        assert processor.friendly_name == "Plate Solver"
        assert "Pixelemon" in processor.description or "Tetra3" in processor.description

    @patch("citrascope.processors.builtin.plate_solver_processor.check_pixelemon")
    def test_pixelemon_not_available(self, mock_check, mock_context):
        """Test processor fails gracefully when Pixelemon not available."""
        mock_check.return_value = False

        processor = PlateSolverProcessor()
        result = processor.process(mock_context)

        assert isinstance(result, ProcessorResult)
        assert result.should_upload is True  # Fail-open
        assert result.confidence == 0.0
        assert "not available" in result.reason or "Pixelemon" in result.reason

    @patch("citrascope.processors.builtin.plate_solver_processor.check_pixelemon")
    @patch("citrascope.processors.builtin.plate_solver_processor.PlateSolverProcessor._solve_with_pixelemon")
    @patch("astropy.io.fits.open")
    def test_successful_plate_solve(self, mock_fits_open, mock_solve, mock_check, mock_context, tmp_path):
        """Test successful plate solving with Pixelemon."""
        mock_check.return_value = True

        new_file = tmp_path / "test_image.new"
        mock_solve.return_value = new_file

        mock_primary = MagicMock(spec=fits.PrimaryHDU)
        mock_header = {
            "CRVAL1": 120.5,
            "CRVAL2": 45.3,
            "CDELT1": 0.001,
        }
        mock_primary.header = mock_header
        mock_hdul = MagicMock()
        mock_hdul.__getitem__ = MagicMock(return_value=mock_primary)
        mock_fits_open.return_value.__enter__.return_value = mock_hdul

        processor = PlateSolverProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 1.0
        assert result.extracted_data["plate_solved"] is True
        assert result.extracted_data["ra_center"] == 120.5
        assert result.extracted_data["dec_center"] == 45.3
        assert mock_context.working_image_path == new_file

    @patch("citrascope.processors.builtin.plate_solver_processor.check_pixelemon")
    @patch("citrascope.processors.builtin.plate_solver_processor.PlateSolverProcessor._solve_with_pixelemon")
    def test_plate_solve_failure(self, mock_solve, mock_check, mock_context):
        """Test plate solving failure handling (fail-open)."""
        mock_check.return_value = True
        mock_solve.side_effect = RuntimeError("Solve failed")

        processor = PlateSolverProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True  # Fail-open
        assert result.confidence == 0.0
        assert "failed" in result.reason.lower()


class TestSourceExtractorProcessor:
    """Tests for SourceExtractorProcessor."""

    def test_processor_metadata(self):
        """Test processor has correct metadata."""
        processor = SourceExtractorProcessor()
        assert processor.name == "source_extractor"
        assert processor.friendly_name == "Source Extractor"
        assert "SExtractor" in processor.description

    @patch("astropy.io.fits.open")
    def test_missing_wcs(self, mock_fits_open, mock_context):
        """Test processor fails gracefully when WCS missing."""
        mock_primary = MagicMock(spec=fits.PrimaryHDU)
        mock_primary.header = {}  # No CRVAL1
        mock_hdul = MagicMock()
        mock_hdul.__getitem__ = MagicMock(return_value=mock_primary)
        mock_fits_open.return_value.__enter__.return_value = mock_hdul

        processor = SourceExtractorProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 0.0
        assert "WCS missing" in result.reason

    @patch("astropy.io.fits.open")
    @patch("citrascope.processors.builtin.source_extractor_processor.check_sextractor")
    def test_sextractor_not_installed(self, mock_check, mock_fits_open, mock_context):
        """Test processor fails gracefully when SExtractor not installed."""
        mock_primary = MagicMock(spec=fits.PrimaryHDU)
        mock_primary.header = {"CRVAL1": 120.0}
        mock_hdul = MagicMock()
        mock_hdul.__getitem__ = MagicMock(return_value=mock_primary)
        mock_fits_open.return_value.__enter__.return_value = mock_hdul

        mock_check.return_value = False

        processor = SourceExtractorProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 0.0
        assert "not installed" in result.reason

    @patch("astropy.io.fits.open")
    @patch("citrascope.processors.builtin.source_extractor_processor.check_sextractor")
    @patch("citrascope.processors.builtin.source_extractor_processor.SourceExtractorProcessor._extract_sources")
    def test_successful_extraction(self, mock_extract, mock_check, mock_fits_open, mock_context, tmp_path):
        """Test successful source extraction."""
        mock_primary = MagicMock(spec=fits.PrimaryHDU)
        mock_primary.header = {"CRVAL1": 120.0}
        mock_hdul = MagicMock()
        mock_hdul.__getitem__ = MagicMock(return_value=mock_primary)
        mock_fits_open.return_value.__enter__.return_value = mock_hdul

        mock_check.return_value = True

        # Mock source extraction
        sources = pd.DataFrame({"ra": [120.1, 120.2], "dec": [45.1, 45.2], "mag": [10.5, 11.2], "fwhm": [2.0, 2.1]})
        mock_extract.return_value = sources

        processor = SourceExtractorProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 1.0
        assert result.extracted_data["num_sources"] == 2


class TestPhotometryProcessor:
    """Tests for PhotometryProcessor."""

    def test_processor_metadata(self):
        """Test processor has correct metadata."""
        processor = PhotometryProcessor()
        assert processor.name == "photometry"
        assert processor.friendly_name == "Photometry Calibrator"
        assert "APASS" in processor.description

    def test_missing_catalog(self, mock_context):
        """Test processor fails gracefully when catalog missing."""
        processor = PhotometryProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 0.0
        assert "catalog not found" in result.reason

    @patch("citrascope.processors.builtin.photometry_processor.PhotometryProcessor._calibrate_photometry")
    @patch("pandas.read_csv")
    def test_successful_calibration(self, mock_read, mock_calibrate, mock_context, tmp_path):
        """Test successful photometric calibration."""
        # Create mock catalog file (processor expects output.cat in working_dir)
        (mock_context.working_dir / "output.cat").touch()
        mock_context.working_image_path = tmp_path / "test_image.fits"

        # Mock catalog reading
        sources = pd.DataFrame({"ra": [120.1, 120.2], "dec": [45.1, 45.2], "mag": [10.5, 11.2]})
        mock_read.return_value = sources

        # Mock calibration (zero_point, num_matched, apass_catalog, crossmatch_df)
        mock_calibrate.return_value = (25.3, 15, pd.DataFrame(), pd.DataFrame())

        processor = PhotometryProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 1.0
        assert result.extracted_data["zero_point"] == 25.3
        assert result.extracted_data["num_calibration_stars"] == 15


class TestSatelliteMatcherProcessorUnit:
    """Unit tests for SatelliteMatcherProcessor (mocked dependencies)."""

    def test_processor_metadata(self):
        """Test processor has correct metadata."""
        processor = SatelliteMatcherProcessor()
        assert processor.name == "satellite_matcher"
        assert processor.friendly_name == "Satellite Matcher"
        assert "TLE" in processor.description

    def test_missing_catalog(self, mock_context):
        """Test processor fails gracefully when catalog missing."""
        processor = SatelliteMatcherProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 0.0
        assert "catalog not found" in result.reason


class TestDependencyChecks:
    """Tests for dependency checking utilities."""

    def test_check_pixelemon(self):
        """Test Pixelemon detection (import-based)."""
        # When pixelemon is installed (as in dev), check_pixelemon() may be True
        # We only assert it returns a bool
        result = check_pixelemon()
        assert isinstance(result, bool)

    @patch("citrascope.processors.builtin.processor_dependencies.shutil.which")
    def test_check_sextractor(self, mock_which):
        """Test SExtractor detection."""
        # Test source-extractor command
        mock_which.side_effect = lambda cmd: "/usr/bin/source-extractor" if cmd == "source-extractor" else None
        assert check_sextractor() is True

        # Reset mock and test sex alias
        mock_which.reset_mock()
        mock_which.side_effect = lambda cmd: "/usr/bin/sex" if cmd == "sex" else None
        assert check_sextractor() is True

        # Reset mock and test neither found
        mock_which.reset_mock()
        mock_which.side_effect = lambda cmd: None
        assert check_sextractor() is False


@pytest.mark.slow
class TestPixelemonDemoFits:
    """Run the real PlateSolverProcessor on the demo FITS (no mocks). Skips only if file missing."""

    DEMO_FITS = Path(__file__).parent / "test_assets" / "2025-11-11_18-38-11_r_-0.05_1.00s_0131.fits"

    def test_plate_solver_processor_solves_demo_fits(self, tmp_path, mock_settings):
        """Run PlateSolverProcessor.process() on the demo FITS; assert success and valid .new file with WCS."""
        if not self.DEMO_FITS.exists():
            pytest.skip("Demo FITS not found")
        if not check_pixelemon():
            pytest.skip(
                "Pixelemon not available (import failed). Use the Python env where citrascope is installed; "
                "in Cursor, use Python: Select Interpreter."
            )

        from astropy.io import fits

        working_dir = tmp_path / "working"
        working_dir.mkdir(exist_ok=True)

        # Provide observer location via location_service; processor injects into working copy for Pixelemon
        class FakeLocationService:
            def get_current_location(self):
                return {"latitude": 40.0, "longitude": -111.0, "altitude": 1400.0}

        context = ProcessingContext(
            image_path=self.DEMO_FITS,
            working_image_path=self.DEMO_FITS,
            working_dir=working_dir,
            image_data=None,
            task=Mock(satelliteName="TEST", satelliteId="0", assigned_filter_name="Clear"),
            telescope_record=_PLANEWAVE_TELESCOPE_RECORD,
            ground_station_record=None,
            settings=mock_settings,
            location_service=FakeLocationService(),
            logger=Mock(),
        )

        processor = PlateSolverProcessor()
        result = processor.process(context)

        assert result.should_upload is True
        if not result.extracted_data.get("plate_solved"):
            pytest.fail(f"Plate solver did not succeed: {result.reason}")
        assert result.confidence == 1.0
        assert result.extracted_data.get("plate_solved") is True
        assert "ra_center" in result.extracted_data
        assert "dec_center" in result.extracted_data
        wcs_path = context.working_image_path
        assert wcs_path.suffix == ".new"
        assert wcs_path.exists()

        with fits.open(wcs_path) as hdul:
            header = hdul[0].header
            assert "CRVAL1" in header
            assert "CRVAL2" in header
            ra, dec = header["CRVAL1"], header["CRVAL2"]
        assert 0 <= ra <= 360
        assert -90 <= dec <= 90


@pytest.mark.slow
class TestFullPipelineDemoFits:
    """Run the full MSI pipeline (plate solve → source extract → photometry → satellite match) on the demo FITS."""

    DEMO_FITS = Path(__file__).parent / "test_assets" / "2025-11-11_18-38-11_r_-0.05_1.00s_0131.fits"
    TLE_FILE = Path(__file__).parent / "test_assets" / "space-track-2025-11-12--2025-11-13.tle"

    def test_full_pipeline_solves_demo_fits(self, tmp_path, run_from_repo_root):
        """Run ProcessorRegistry.process_all() on the demo FITS with a TLE from test assets; assert full chain."""
        if not self.DEMO_FITS.exists():
            pytest.skip("Demo FITS not found")
        if not self.TLE_FILE.exists():
            pytest.skip("TLE file not found")
        if not check_pixelemon():
            pytest.skip("Pixelemon not available")
        if not check_sextractor():
            pytest.skip("SExtractor not available")

        tle_lines = _first_tle_from_file(self.TLE_FILE)
        if len(tle_lines) != 2:
            pytest.skip("No valid TLE pair in TLE file")

        # Task with TLE from file (any satellite; we just need the pipeline to run)
        task = Mock(
            satelliteName="TEST-SAT",
            satelliteId="00005",
            assigned_filter_name="Clear",
            most_recent_elset={"tle": tle_lines},
        )

        settings = Mock()
        settings.enabled_processors = {}

        class FakeLocationService:
            def get_current_location(self):
                # Coordinates from demo FITS SITELAT/SITELONG/SITEELEV header keywords
                return {"latitude": 31.9070277777778, "longitude": -109.021111111111, "altitude": 1250.0}

        working_dir = tmp_path / "working"
        working_dir.mkdir(exist_ok=True)
        context = ProcessingContext(
            image_path=self.DEMO_FITS,
            working_image_path=self.DEMO_FITS,
            working_dir=working_dir,
            image_data=None,
            task=task,
            telescope_record=_PLANEWAVE_TELESCOPE_RECORD,
            ground_station_record=None,
            settings=settings,
            location_service=FakeLocationService(),
            elset_cache=None,  # no hot list: use single-TLE from task (backward compatibility)
            logger=Mock(),
        )

        registry = ProcessorRegistry(settings, Mock())
        result = registry.process_all(context)

        assert result.should_upload is True
        assert "plate_solver.plate_solved" in result.extracted_data
        if not result.extracted_data["plate_solver.plate_solved"]:
            ps = next((r for r in result.all_results if r.processor_name == "plate_solver"), None)
            reason = ps.reason if ps else "unknown"
            pytest.fail(f"Plate solver did not succeed: {reason}")
        assert result.extracted_data["plate_solver.plate_solved"] is True
        assert "plate_solver.ra_center" in result.extracted_data
        assert "plate_solver.dec_center" in result.extracted_data
        assert "source_extractor.num_sources" in result.extracted_data
        assert result.extracted_data["source_extractor.num_sources"] >= 0
        assert "photometry.zero_point" in result.extracted_data
        assert "satellite_matcher.num_satellites_detected" in result.extracted_data
        assert "satellite_matcher.satellite_observations" in result.extracted_data
        # Pipeline ran end-to-end against the FITS
        assert context.working_image_path.suffix == ".new"
        assert context.working_image_path.exists()

    def test_full_pipeline_with_elset_cache_uses_multi_tle(self, tmp_path, run_from_repo_root):
        """Run full pipeline with daemon.elset_cache populated; matcher uses
        multi-TLE path, observations match elsets."""
        if not self.DEMO_FITS.exists():
            pytest.skip("Demo FITS not found")
        if not self.TLE_FILE.exists():
            pytest.skip("TLE file not found")
        if not check_pixelemon():
            pytest.skip("Pixelemon not available")
        if not check_sextractor():
            pytest.skip("SExtractor not available")

        elsets = _first_three_elsets_from_file(self.TLE_FILE)
        if len(elsets) < 3:
            pytest.skip("Need at least 3 TLE pairs in TLE file")

        # File-backed elset cache with 3 elsets (one is in-frame for demo FITS)
        cache_file = tmp_path / "elset_cache.json"
        cache_file.write_text(json.dumps(elsets), encoding="utf-8")
        elset_cache = ElsetCache(cache_path=cache_file)
        elset_cache.load_from_file()
        assert len(elset_cache.get_elsets()) == 3
        elset_ids = {e["satellite_id"] for e in elsets}
        elset_names = {e["name"] for e in elsets}

        task = Mock(
            satelliteName="TEST-SAT",
            satelliteId="00005",
            assigned_filter_name="Clear",
            most_recent_elset={"tle": elsets[0]["tle"]},
        )
        settings = Mock()
        settings.enabled_processors = {}

        class FakeLocationService:
            def get_current_location(self):
                # Coordinates from demo FITS SITELAT/SITELONG/SITEELEV header keywords
                return {"latitude": 31.9070277777778, "longitude": -109.021111111111, "altitude": 1250.0}

        working_dir = tmp_path / "working"
        working_dir.mkdir(exist_ok=True)
        context = ProcessingContext(
            image_path=self.DEMO_FITS,
            working_image_path=self.DEMO_FITS,
            working_dir=working_dir,
            image_data=None,
            task=task,
            telescope_record=_PLANEWAVE_TELESCOPE_RECORD,
            ground_station_record=None,
            settings=settings,
            location_service=FakeLocationService(),
            elset_cache=elset_cache,
            logger=Mock(),
        )

        registry = ProcessorRegistry(settings, Mock())
        result = registry.process_all(context)

        assert result.should_upload is True
        assert result.extracted_data["plate_solver.plate_solved"] is True
        observations = result.extracted_data.get("satellite_matcher.satellite_observations", [])
        # Multi-TLE path: pipeline ran end-to-end and the satellite matcher executed.
        # We do not assert a minimum count here because Pixelemon's plate-solve WCS has
        # ~2-4 arcmin RMS accuracy across the full field, which may not meet the satellite
        # matcher's 1-arcmin KDTree threshold.  Algorithm correctness (6/6 detections) is
        # validated by TestSatelliteMatcherProcessor using the reference astrometry.net WCS.
        assert isinstance(observations, list)
        for obs in observations:
            assert obs.get("norad_id") in elset_ids
            assert obs.get("name") in elset_names


# ---------------------------------------------------------------------------
# Known-good WCS from MSIimageworker astrometry.net solve of the demo image.
# Used to inject a reference plate solution without running Pixelemon/SExtractor.
# ---------------------------------------------------------------------------
_REFERENCE_WCS = {
    "CTYPE1": "RA---TAN-SIP",
    "CTYPE2": "DEC--TAN-SIP",
    "CRVAL1": 333.745911734,
    "CRVAL2": -5.75518768792,
    "CRPIX1": 2037.60963949,
    "CRPIX2": 1996.33288574,
    "CD1_1": -0.000548395161164,
    "CD1_2": 0.000126264675178,
    "CD2_1": -0.00012712684364,
    "CD2_2": -0.000546811129231,
}


def _make_reference_wcs_fits(src: Path, dst: Path) -> None:
    """Copy src FITS to dst, injecting the MSIimageworker reference WCS header."""
    from astropy.io import fits as _fits

    with _fits.open(src) as hdul:
        new_header = hdul[0].header.copy()
        for k, v in _REFERENCE_WCS.items():
            new_header[k] = v
        _fits.writeto(dst, hdul[0].data, new_header, overwrite=True)


def _all_elsets_from_file(tle_path: Path) -> list:
    """Read all TLE pairs from file; return processor-ready list of {satellite_id, name, tle}."""
    if not tle_path.exists():
        return []
    result = []
    with open(tle_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("1 "):
                line1 = line
                next_line = f.readline()
                if next_line and next_line.startswith("2 "):
                    line2 = next_line.rstrip("\n")
                    norad = line2[2:7].strip() if len(line2) >= 7 else "00000"
                    result.append({"satellite_id": norad, "name": f"SAT-{norad}", "tle": [line1, line2]})
    return result


class TestSatelliteMatcherProcessor:
    """Isolation tests for SatelliteMatcherProcessor using reference inputs from MSIimageworker.

    These tests bypass Pixelemon and SExtractor entirely, injecting the known-good
    astrometry.net WCS and the reference source catalog directly.  They prove that
    the matcher algorithm is equivalent to MSIimageworker's generate_obs.py and
    provide a hard 6/6 regression anchor independent of the plate-solving toolchain.
    """

    DEMO_FITS = Path(__file__).parent / "test_assets" / "2025-11-11_18-38-11_r_-0.05_1.00s_0131.fits"
    REF_CAT = Path(__file__).parent / "test_assets" / "2025-11-11_18-38-11_r_-0.05_1.00s_0131.cat"
    TLE_FILE = Path(__file__).parent / "test_assets" / "space-track-2025-11-12--2025-11-13.tle"
    # All 6 GEO satellites confirmed present in the demo image (from MSIimageworker obs.csv)
    EXPECTED_NORADS = {"31862", "36131", "37748", "40663", "53960", "55970"}

    def test_detects_all_known_satellites_with_reference_wcs(self, tmp_path):
        """Satellite matcher finds all 6 known GEO satellites when given
        a correctly plate-solved FITS and the matching MSIimageworker source catalog.

        This validates that our matcher algorithm is algorithmically equivalent to
        MSIimageworker's generate_obs.py and will produce correct results at runtime
        when the plate solver provides accurate WCS.
        """
        if not self.DEMO_FITS.exists():
            pytest.skip("Demo FITS not found")
        if not self.REF_CAT.exists():
            pytest.skip("Reference catalog not found")
        if not self.TLE_FILE.exists():
            pytest.skip("TLE file not found")
        elsets = _all_elsets_from_file(self.TLE_FILE)
        if len(elsets) < 6:
            pytest.skip("Need all 6 TLE pairs in TLE file")

        # Build elset cache with all 6 known in-frame satellites
        cache_file = tmp_path / "elset_cache.json"
        cache_file.write_text(json.dumps(elsets), encoding="utf-8")
        elset_cache = ElsetCache(cache_path=cache_file)
        elset_cache.load_from_file()

        # Inject the reference astrometry.net WCS into a copy of the demo FITS
        wcs_fits = tmp_path / "demo_reference_wcs.new"
        _make_reference_wcs_fits(self.DEMO_FITS, wcs_fits)

        # Copy the MSIimageworker reference catalog to the expected working_dir location
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        import shutil

        shutil.copy(self.REF_CAT, working_dir / "output.cat")

        task = Mock(
            satelliteName="SAT-31862",
            satelliteId="31862",
            assigned_filter_name="r",
            most_recent_elset={"tle": elsets[0]["tle"]},
        )
        settings = Mock()
        settings.enabled_processors = {}

        class FakeLocationService:
            def get_current_location(self):
                # Coordinates from demo FITS SITELAT/SITELONG/SITEELEV header keywords
                return {"latitude": 31.9070277777778, "longitude": -109.021111111111, "altitude": 1250.0}

        context = ProcessingContext(
            image_path=self.DEMO_FITS,
            working_image_path=wcs_fits,  # pre-solved reference FITS
            working_dir=working_dir,
            image_data=None,
            task=task,
            telescope_record=None,
            ground_station_record=None,
            settings=settings,
            location_service=FakeLocationService(),
            elset_cache=elset_cache,
            logger=Mock(),
        )

        processor = SatelliteMatcherProcessor()
        result = processor.process(context)

        observations = result.extracted_data.get("satellite_observations", [])
        detected_norads = {obs["norad_id"] for obs in observations}

        assert result.extracted_data.get("num_satellites_detected", 0) == len(observations)
        assert detected_norads == self.EXPECTED_NORADS, (
            f"Expected all 6 known GEO satellites {self.EXPECTED_NORADS}, "
            f"got {detected_norads} ({len(observations)} detections)"
        )
