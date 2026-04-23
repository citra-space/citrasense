"""Unit tests for MSI science processors."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from astropy.io import fits

# CitraSense repo root (tests/unit -> tests -> root). Used so demo-FITS tests run with stable cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def run_from_repo_root():
    """Run test with cwd = citrasense repo root, then restore. Makes demo-FITS tests match terminal runs."""
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


from citrasense.elset_cache import ElsetCache
from citrasense.pipelines.common.pipeline_registry import PipelineRegistry
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import ProcessorResult
from citrasense.pipelines.optical.photometry_processor import PhotometryProcessor
from citrasense.pipelines.optical.plate_solver_processor import PlateSolverProcessor
from citrasense.pipelines.optical.processor_dependencies import (
    check_astrometry,
    check_sextractor,
)
from citrasense.pipelines.optical.satellite_matcher_processor import SatelliteMatcherProcessor

# Telescope record matching the PlaneWave scope used to capture the demo FITS.
_PLANEWAVE_TELESCOPE_RECORD = {
    "id": "4861f510-e3bd-470f-ab31-98766ecba2ca",
    "angularNoise": 1.0,
    "legacyFieldOfView": 2.0,
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


def _make_mock_settings(**overrides) -> Mock:
    """Create a Mock settings with plate solving defaults from CitraSenseSettings."""
    from citrasense.settings.citrasense_settings import CitraSenseSettings

    attrs = {
        name: field.default
        for name, field in CitraSenseSettings.model_fields.items()
        if name.startswith(("plate_solve_", "astrometry_"))
    }
    attrs.update(overrides)
    return Mock(**attrs)


@pytest.fixture
def mock_settings():
    """Mock settings for testing with plate solving defaults."""
    return _make_mock_settings()


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
        assert "astrometry" in processor.description.lower()

    @patch("citrasense.pipelines.optical.plate_solver_processor.check_astrometry")
    def test_astrometry_not_available(self, mock_check, mock_context):
        """Test processor fails gracefully when solve-field not available."""
        mock_check.return_value = False

        processor = PlateSolverProcessor()
        result = processor.process(mock_context)

        assert isinstance(result, ProcessorResult)
        assert result.should_upload is True  # Fail-open
        assert result.confidence == 0.0
        assert "not available" in result.reason

    @patch("citrasense.pipelines.optical.plate_solver_processor.check_astrometry")
    @patch("citrasense.pipelines.optical.plate_solver_processor.PlateSolverProcessor._solve_field")
    @patch("astropy.io.fits.open")
    def test_successful_plate_solve(self, mock_fits_open, mock_solve, mock_check, mock_context, tmp_path):
        """Test successful plate solving with astrometry.net."""
        mock_check.return_value = True

        new_file = tmp_path / "test_image_wcs.fits"
        mock_solve.return_value = (new_file, {"log_odds": 42.0, "n_match": 15, "n_conflict": 0})

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

    @patch("citrasense.pipelines.optical.plate_solver_processor.check_astrometry")
    @patch("citrasense.pipelines.optical.plate_solver_processor.PlateSolverProcessor._solve_field")
    def test_plate_solve_failure(self, mock_solve, mock_check, mock_context):
        """Test plate solving failure handling (fail-open)."""
        mock_check.return_value = True
        mock_solve.side_effect = RuntimeError("Solve failed")

        processor = PlateSolverProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True  # Fail-open
        assert result.confidence == 0.0
        assert "failed" in result.reason.lower()


class TestPhotometryProcessor:
    """Tests for PhotometryProcessor."""

    def test_processor_metadata(self):
        """Test processor has correct metadata."""
        processor = PhotometryProcessor()
        assert processor.name == "photometry"
        assert processor.friendly_name == "Photometry Calibrator"
        assert "APASS" in processor.description

    def test_missing_catalog(self, mock_context):
        """Test processor fails gracefully when source catalog is missing."""
        processor = PhotometryProcessor()
        result = processor.process(mock_context)

        assert result.should_upload is True
        assert result.confidence == 0.0
        assert "source catalog not found" in result.reason.lower()

    @patch("citrasense.pipelines.optical.photometry_processor.PhotometryProcessor._calibrate_photometry")
    @patch("pandas.read_csv")
    def test_successful_calibration(self, mock_read, mock_calibrate, mock_context, tmp_path):
        """Test successful photometric calibration."""
        # Create mock catalog file (processor expects output.cat in working_dir)
        (mock_context.working_dir / "output.cat").touch()
        mock_context.working_image_path = tmp_path / "test_image.fits"

        # Provide a mock APASS catalog so the processor doesn't bail out early
        mock_apass = Mock()
        mock_apass.is_available.return_value = True
        mock_context.apass_catalog = mock_apass

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

    def test_check_astrometry(self):
        """Test astrometry.net detection (which-based)."""
        result = check_astrometry()
        assert isinstance(result, bool)

    @patch("citrasense.pipelines.optical.processor_dependencies.shutil.which")
    def test_check_astrometry_mocked(self, mock_which):
        """Test astrometry.net detection with mocked which."""
        mock_which.side_effect = lambda cmd: "/usr/local/bin/solve-field" if cmd == "solve-field" else None
        assert check_astrometry() is True

        mock_which.reset_mock()
        mock_which.side_effect = lambda cmd: None
        assert check_astrometry() is False

    @patch("citrasense.pipelines.optical.processor_dependencies.shutil.which")
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


class TestPlateScaleComputation:
    """Tests for _compute_plate_scale helper."""

    def test_compute_plate_scale_planewave(self):
        """Verify plate scale computation for the PlaneWave CDK14."""
        from citrasense.pipelines.optical.plate_solver_processor import _compute_plate_scale

        scale = _compute_plate_scale(_PLANEWAVE_TELESCOPE_RECORD, x_binning=4, y_binning=4)
        assert scale is not None
        # PlaneWave CDK14: 3.76um * 4 / 1530mm * 206265 = ~2.03 arcsec/px
        assert 1.5 < scale < 2.5

    def test_compute_plate_scale_no_binning(self):
        from citrasense.pipelines.optical.plate_solver_processor import _compute_plate_scale

        scale = _compute_plate_scale(_PLANEWAVE_TELESCOPE_RECORD)
        assert scale is not None
        # Unbinned: 3.76um / 1530mm * 206265 = ~0.507 arcsec/px
        assert 0.3 < scale < 0.7

    def test_compute_plate_scale_missing_fields(self):
        from citrasense.pipelines.optical.plate_solver_processor import _compute_plate_scale

        assert _compute_plate_scale({}) is None
        assert _compute_plate_scale({"pixelSize": 3.76}) is None
        assert _compute_plate_scale({"focalLength": 1530}) is None


@pytest.mark.slow
class TestAstrometryDemoFits:
    """Run the real PlateSolverProcessor on the demo FITS (no mocks). Skips only if file or solve-field missing."""

    DEMO_FITS = Path(__file__).parent / "test_assets" / "2025-11-11_18-38-11_r_-0.05_1.00s_0131.fits"

    def test_plate_solver_processor_solves_demo_fits(self, tmp_path, mock_settings):
        """Run PlateSolverProcessor.process() on the demo FITS; assert success and valid _wcs.fits file with WCS."""
        if not self.DEMO_FITS.exists():
            pytest.skip("Demo FITS not found")
        if not check_astrometry():
            pytest.skip("astrometry.net (solve-field) not available")

        working_dir = tmp_path / "working"
        working_dir.mkdir(exist_ok=True)

        context = ProcessingContext(
            image_path=self.DEMO_FITS,
            working_image_path=self.DEMO_FITS,
            working_dir=working_dir,
            image_data=None,
            task=Mock(satelliteName="TEST", satelliteId="0", assigned_filter_name="Clear"),
            telescope_record=_PLANEWAVE_TELESCOPE_RECORD,
            ground_station_record=None,
            settings=mock_settings,
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
        assert wcs_path.suffix == ".fits"
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
    """Run the full MSI pipeline (plate solve + source extraction -> photometry -> satellite match) on the demo FITS."""

    DEMO_FITS = Path(__file__).parent / "test_assets" / "2025-11-11_18-38-11_r_-0.05_1.00s_0131.fits"
    TLE_FILE = Path(__file__).parent / "test_assets" / "space-track-2025-11-12--2025-11-13.tle"

    def test_full_pipeline_solves_demo_fits(self, tmp_path, run_from_repo_root):
        """Run PipelineRegistry.process_all() on the demo FITS with a TLE from test assets; assert full chain."""
        if not self.DEMO_FITS.exists():
            pytest.skip("Demo FITS not found")
        if not self.TLE_FILE.exists():
            pytest.skip("TLE file not found")
        if not check_astrometry():
            pytest.skip("astrometry.net (solve-field) not available")
        if not check_sextractor():
            pytest.skip("SExtractor not available")

        tle_lines = _first_tle_from_file(self.TLE_FILE)
        if len(tle_lines) != 2:
            pytest.skip("No valid TLE pair in TLE file")

        most_recent_elset = {"tle": tle_lines}
        task = Mock(
            satelliteName="TEST-SAT",
            satelliteId="00005",
            assigned_filter_name="Clear",
            most_recent_elset=most_recent_elset,
        )

        settings = _make_mock_settings(enabled_processors={})

        class FakeLocationService:
            def get_current_location(self):
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
            elset_cache=None,
            satellite_data={"most_recent_elset": most_recent_elset},
            tracking_mode="rate",
            logger=Mock(),
        )

        registry = PipelineRegistry(settings, Mock())
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
        assert "satellite_matcher.num_satellites_detected" in result.extracted_data
        assert "satellite_matcher.satellite_observations" in result.extracted_data
        assert context.working_image_path.suffix == ".fits"
        assert context.working_image_path.exists()


# ---------------------------------------------------------------------------
# Known-good WCS from MSIimageworker astrometry.net solve of the demo image.
# Used to inject a reference plate solution without running solve-field.
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

    These tests bypass the plate solver and SExtractor entirely, injecting the known-good
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
        settings = _make_mock_settings(enabled_processors={})

        class FakeLocationService:
            def get_current_location(self):
                return {"latitude": 31.9070277777778, "longitude": -109.021111111111, "altitude": 1250.0}

        context = ProcessingContext(
            image_path=self.DEMO_FITS,
            working_image_path=wcs_fits,
            working_dir=working_dir,
            image_data=None,
            task=task,
            telescope_record=None,
            ground_station_record=None,
            settings=settings,
            location_service=FakeLocationService(),
            elset_cache=elset_cache,
            tracking_mode="rate",
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


# Spirit/Pi telescope record (Moravian C2-12000 on 180mm f/4.5 scope, 2x2 binning).
_SPIRIT_TELESCOPE_RECORD = {
    "pixelSize": 3.45,
    "focalLength": 180.0,
    "focalRatio": 4.5,
    "horizontalPixelCount": 4112,
    "verticalPixelCount": 3008,
    "imageCircleDiameter": 28.0,
}
