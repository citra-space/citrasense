"""Tests for CalibrationProcessor: calibration math and graceful skip."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from citrascope.calibration.calibration_library import CalibrationLibrary
from citrascope.processors.builtin.calibration_processor import CalibrationProcessor
from citrascope.processors.processor_result import ProcessingContext


@pytest.fixture
def library(tmp_path):
    return CalibrationLibrary(root=tmp_path / "calibration")


@pytest.fixture
def working_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


def _make_fits(path: Path, data: np.ndarray, **header_kwargs) -> Path:
    """Write a FITS file with the given data and header keywords."""
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    for k, v in header_kwargs.items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=True)
    return path


def _make_context(image_path: Path, working_dir: Path) -> ProcessingContext:
    return ProcessingContext(
        image_path=image_path,
        working_image_path=image_path,
        working_dir=working_dir,
        image_data=None,
        task=None,
        telescope_record=None,
        ground_station_record=None,
        settings=None,
    )


class TestCalibrationMath:
    def test_dark_subtraction(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        # Save a master dark (already bias-subtracted)
        dark = np.full((10, 10), 50.0, dtype=np.float32)
        library.save_master("dark", "SN1234", dark, gain=0, binning=1, exposure_time=2.0, temperature=-10.0)

        # Science frame
        science = np.full((10, 10), 250.0, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            science,
            CAMSER="SN1234",
            GAIN=0,
            XBINNING=1,
            EXPTIME=2.0,
            **{"CCD-TEMP": -10.0},
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        result = proc.process(ctx)

        assert result.should_upload is True
        assert "dark" in result.extracted_data.get("calibration_applied", [])

        calibrated = fits.getdata(ctx.working_image_path)
        np.testing.assert_array_almost_equal(calibrated, 200.0)

    def test_dark_and_flat(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        dark = np.full((10, 10), 30.0, dtype=np.float32)
        library.save_master("dark", "SN1234", dark, gain=0, binning=1, exposure_time=2.0, temperature=-10.0)

        # Flat normalised to 1.0 everywhere except right half at 0.5
        flat = np.ones((10, 10), dtype=np.float32)
        flat[:, 5:] = 0.5
        library.save_master("flat", "SN1234", flat, gain=0, binning=1, filter_name="Lum")

        science = np.full((10, 10), 130.0, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            science,
            CAMSER="SN1234",
            GAIN=0,
            XBINNING=1,
            EXPTIME=2.0,
            FILTER="Lum",
            **{"CCD-TEMP": -10.0},
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        proc.process(ctx)

        calibrated = fits.getdata(ctx.working_image_path)
        # Left half: (130 - 30) / 1.0 = 100
        np.testing.assert_array_almost_equal(calibrated[:, :5], 100.0)
        # Right half: (130 - 30) / 0.5 = 200
        np.testing.assert_array_almost_equal(calibrated[:, 5:], 200.0)

    def test_dark_scaling_with_bias(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        """Dark scaling: 30s reference dark scaled to a 2s science frame."""
        bias = np.full((10, 10), 100.0, dtype=np.float32)
        library.save_master("bias", "SN1234", bias, gain=0, binning=1)

        # MasterBuilder subtracts bias before saving, so stored dark is
        # pure thermal: raw_dark(30s) - bias = 150 - 100 = 50
        thermal = np.full((10, 10), 50.0, dtype=np.float32)
        library.save_master("dark", "SN1234", thermal, gain=0, binning=1, exposure_time=30.0, temperature=-10.0)

        # Science frame: signal(500) + bias(100) + thermal_2s(50 * 2/30 ≈ 3.33)
        raw_science = np.full((10, 10), 603.33, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            raw_science,
            CAMSER="SN1234",
            GAIN=0,
            XBINNING=1,
            EXPTIME=2.0,
            **{"CCD-TEMP": -10.0},
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        result = proc.process(ctx)

        assert "bias" in result.extracted_data.get("calibration_applied", [])
        assert "dark" in result.extracted_data.get("calibration_applied", [])

        calibrated = fits.getdata(ctx.working_image_path)
        # Expected: 603.33 - 100 (bias) - 50 * (2/30) (scaled thermal) = 500
        np.testing.assert_array_almost_equal(calibrated, 500.0, decimal=1)

    def test_dark_scaling_no_bias_at_build_time(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        """Dark built without bias (BIASSUB=False), then bias available at processing time.

        The processor must NOT subtract bias separately — the dark already
        includes it, so scaling the dark is sufficient.
        """
        bias = np.full((10, 10), 100.0, dtype=np.float32)
        library.save_master("bias", "SN1234", bias, gain=0, binning=1)

        # Dark was built WITHOUT bias subtraction: raw_dark = bias(100) + thermal(50) = 150
        raw_dark = np.full((10, 10), 150.0, dtype=np.float32)
        library.save_master(
            "dark",
            "SN1234",
            raw_dark,
            gain=0,
            binning=1,
            exposure_time=30.0,
            temperature=-10.0,
            bias_subtracted=False,
        )

        # Science: signal(500) + bias(100) + thermal_2s(150 * 2/30 = 10)
        raw_science = np.full((10, 10), 610.0, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            raw_science,
            CAMSER="SN1234",
            GAIN=0,
            XBINNING=1,
            EXPTIME=2.0,
            **{"CCD-TEMP": -10.0},
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        result = proc.process(ctx)

        assert "dark" in result.extracted_data.get("calibration_applied", [])

        calibrated = fits.getdata(ctx.working_image_path)
        # Expected: 610 - 150 * (2/30) = 610 - 10 = 600
        # Note: bias is NOT subtracted separately because dark includes it
        np.testing.assert_array_almost_equal(calibrated, 600.0, decimal=1)


class TestGracefulSkip:
    def test_no_library(self, working_dir: Path, tmp_path: Path):
        science = np.full((10, 10), 100.0, dtype=np.float32)
        science_path = _make_fits(tmp_path / "science.fits", science)

        proc = CalibrationProcessor(library=None)
        ctx = _make_context(science_path, working_dir)
        result = proc.process(ctx)

        assert result.should_upload is True
        assert "not initialized" in result.reason.lower()

    def test_no_masters_for_camera(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        science = np.full((10, 10), 100.0, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            science,
            CAMSER="SN_NEW",
            GAIN=0,
            XBINNING=1,
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        result = proc.process(ctx)

        assert result.should_upload is True
        # Should skip silently (no masters at all for this camera)
        assert "silent" in result.reason.lower() or "no calibration" in result.reason.lower()

    def test_bias_only_applied(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        bias = np.full((10, 10), 20.0, dtype=np.float32)
        library.save_master("bias", "SN1234", bias, gain=0, binning=1)

        science = np.full((10, 10), 120.0, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            science,
            CAMSER="SN1234",
            GAIN=0,
            XBINNING=1,
            EXPTIME=2.0,
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        result = proc.process(ctx)

        assert "bias" in result.extracted_data.get("calibration_applied", [])
        calibrated = fits.getdata(ctx.working_image_path)
        np.testing.assert_array_almost_equal(calibrated, 100.0)


class TestHeaderPreservation:
    def test_original_headers_preserved(self, library: CalibrationLibrary, working_dir: Path, tmp_path: Path):
        dark = np.full((10, 10), 10.0, dtype=np.float32)
        library.save_master("dark", "SN1234", dark, gain=0, binning=1, exposure_time=2.0, temperature=-10.0)

        science = np.full((10, 10), 100.0, dtype=np.float32)
        science_path = _make_fits(
            tmp_path / "science.fits",
            science,
            CAMSER="SN1234",
            GAIN=0,
            XBINNING=1,
            EXPTIME=2.0,
            **{"CCD-TEMP": -10.0, "DATE-OBS": "2025-01-01T00:00:00"},
        )

        proc = CalibrationProcessor(library=library)
        ctx = _make_context(science_path, working_dir)
        proc.process(ctx)

        with fits.open(ctx.working_image_path) as hdul:
            hdr = hdul[0].header  # type: ignore[index]
            assert hdr["CAMSER"] == "SN1234"
            assert hdr["EXPTIME"] == 2.0
            assert hdr["DATE-OBS"] == "2025-01-01T00:00:00"
            assert hdr["CALPROC"] is True
