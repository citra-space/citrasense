"""Tests for CalibrationLibrary: save/load, dark matching, status queries."""

import numpy as np
import pytest
from astropy.io import fits

from citrasense.calibration.calibration_library import CalibrationLibrary, resolve_camera_id


@pytest.fixture
def library(tmp_path):
    return CalibrationLibrary(root=tmp_path / "calibration")


class TestResolveCamera:
    def test_prefers_serial(self):
        hdr = fits.Header()
        hdr["CAMSER"] = "SN1234"
        hdr["INSTRUME"] = "C2-12000"
        assert resolve_camera_id(hdr) == "SN1234"

    def test_falls_back_to_instrume(self):
        hdr = fits.Header()
        hdr["INSTRUME"] = "C2-12000"
        assert resolve_camera_id(hdr) == "C2-12000"

    def test_unknown_when_neither(self):
        hdr = fits.Header()
        assert resolve_camera_id(hdr) == "unknown"


class TestBiasMaster:
    def test_roundtrip(self, library: CalibrationLibrary):
        data = np.ones((100, 100), dtype=np.float32) * 42.0
        path = library.save_master("bias", "SN1234", data, gain=0, binning=1, ncombine=30)
        assert path.exists()

        found = library.get_master_bias("SN1234", gain=0, binning=1)
        assert found == path

        with fits.open(found) as hdul:
            assert hdul[0].header["CALTYPE"] == "BIAS"  # type: ignore[index]
            assert hdul[0].header["NCOMBINE"] == 30  # type: ignore[index]
            assert hdul[0].header["GAIN"] == 0  # type: ignore[index]
            np.testing.assert_array_almost_equal(hdul[0].data, data)  # type: ignore[index]

    def test_missing_returns_none(self, library: CalibrationLibrary):
        assert library.get_master_bias("SN1234", gain=0, binning=1) is None

    def test_different_gain_not_found(self, library: CalibrationLibrary):
        data = np.zeros((10, 10), dtype=np.float32)
        library.save_master("bias", "SN1234", data, gain=0, binning=1)
        assert library.get_master_bias("SN1234", gain=5, binning=1) is None


class TestDarkMaster:
    def test_exact_match(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32) * 10.0
        library.save_master(
            "dark",
            "SN1234",
            data,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.0,
            ncombine=20,
        )
        found = library.get_master_dark("SN1234", gain=0, binning=1, temperature=-10.0)
        assert found is not None

    def test_within_temp_tolerance(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32)
        library.save_master(
            "dark",
            "SN1234",
            data,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.0,
        )
        found = library.get_master_dark("SN1234", gain=0, binning=1, temperature=-10.5)
        assert found is not None

    def test_outside_temp_tolerance(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32)
        library.save_master(
            "dark",
            "SN1234",
            data,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.0,
        )
        found = library.get_master_dark("SN1234", gain=0, binning=1, temperature=-12.0)
        assert found is None

    def test_picks_closest_temp(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32)
        library.save_master(
            "dark",
            "SN1234",
            data * 1,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.0,
        )
        library.save_master(
            "dark",
            "SN1234",
            data * 2,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.3,
        )
        # Should pick -10.3 when science is at -10.2
        found = library.get_master_dark("SN1234", gain=0, binning=1, temperature=-10.2)
        assert found is not None
        with fits.open(found) as hdul:
            assert float(hdul[0].header["CCD-TEMP"]) == pytest.approx(-10.3)  # type: ignore[index]

    def test_prefers_longest_exposure(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32)
        library.save_master(
            "dark",
            "SN1234",
            data,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.0,
        )
        library.save_master(
            "dark",
            "SN1234",
            data,
            gain=0,
            binning=1,
            exposure_time=30.0,
            temperature=-10.0,
        )
        found = library.get_master_dark("SN1234", gain=0, binning=1, temperature=-10.0)
        assert found is not None
        with fits.open(found) as hdul:
            assert float(hdul[0].header["EXPTIME"]) == pytest.approx(30.0)  # type: ignore[index]


class TestFlatMaster:
    def test_roundtrip(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32)
        path = library.save_master("flat", "SN1234", data, gain=0, binning=1, filter_name="Luminance")
        assert path.exists()
        found = library.get_master_flat("SN1234", gain=0, binning=1, filter_name="Luminance")
        assert found == path

    def test_different_filter_not_found(self, library: CalibrationLibrary):
        data = np.ones((50, 50), dtype=np.float32)
        library.save_master("flat", "SN1234", data, gain=0, binning=1, filter_name="Luminance")
        assert library.get_master_flat("SN1234", gain=0, binning=1, filter_name="Red") is None


class TestDelete:
    def test_delete_bias(self, library: CalibrationLibrary):
        data = np.ones((10, 10), dtype=np.float32)
        library.save_master("bias", "SN1234", data, gain=0, binning=1)
        assert library.delete_master("bias", "SN1234", gain=0, binning=1)
        assert library.get_master_bias("SN1234", gain=0, binning=1) is None

    def test_delete_nonexistent(self, library: CalibrationLibrary):
        assert not library.delete_master("bias", "SN1234", gain=0, binning=1)


class TestLibraryStatus:
    def test_empty_library(self, library: CalibrationLibrary):
        status = library.get_library_status("SN1234")
        assert status == {"bias": [], "darks": [], "flats": []}

    def test_populated_library(self, library: CalibrationLibrary):
        data = np.ones((10, 10), dtype=np.float32)
        library.save_master("bias", "SN1234", data, gain=0, binning=1, ncombine=30)
        library.save_master(
            "dark",
            "SN1234",
            data,
            gain=0,
            binning=1,
            exposure_time=2.0,
            temperature=-10.0,
            ncombine=20,
        )
        status = library.get_library_status("SN1234")
        assert len(status["bias"]) == 1
        assert len(status["darks"]) == 1
        assert status["bias"][0]["ncombine"] == 30

    def test_has_any_masters(self, library: CalibrationLibrary):
        assert not library.has_any_masters("SN1234")
        data = np.ones((10, 10), dtype=np.float32)
        library.save_master("bias", "SN1234", data, gain=0, binning=1)
        assert library.has_any_masters("SN1234")
