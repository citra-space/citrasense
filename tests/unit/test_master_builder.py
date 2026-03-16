"""Tests for MasterBuilder: median combine math, bias subtraction, flat normalization."""

import numpy as np
import pytest
from astropy.io import fits

from citrascope.calibration.calibration_library import CalibrationLibrary
from citrascope.calibration.master_builder import MasterBuilder
from citrascope.hardware.devices.camera.abstract_camera import CalibrationProfile


class FakeCamera:
    """Minimal camera stub for testing MasterBuilder."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._idx = 0

    def capture_array(
        self,
        duration: float,
        gain: int | None = None,
        binning: int = 1,
        shutter_closed: bool = False,
        offset: int | None = None,
    ) -> np.ndarray:
        frame = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return frame

    def get_temperature(self) -> float | None:
        return -10.0

    def get_max_pixel_value(self, binning: int = 1) -> int:
        return 65535


@pytest.fixture
def library(tmp_path):
    return CalibrationLibrary(root=tmp_path / "calibration")


@pytest.fixture
def profile():
    return CalibrationProfile(
        calibration_applicable=True,
        camera_id="TESTCAM",
        model="TestModel",
        has_mechanical_shutter=True,
        has_cooling=True,
        current_gain=0,
        current_binning=1,
        current_temperature=-10.0,
        supported_binning=[1, 2],
    )


class TestBiasBuild:
    def test_median_of_three_frames(self, library: CalibrationLibrary, profile: CalibrationProfile):
        frames = [
            np.full((4, 4), 100.0, dtype=np.uint16),
            np.full((4, 4), 200.0, dtype=np.uint16),
            np.full((4, 4), 150.0, dtype=np.uint16),
        ]
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_bias(count=3, gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            # Median of [100, 200, 150] = 150
            np.testing.assert_array_almost_equal(data, 150.0)
            assert hdul[0].header["CALTYPE"] == "BIAS"  # type: ignore[index]
            assert hdul[0].header["NCOMBINE"] == 3  # type: ignore[index]


class TestDarkBuild:
    def test_bias_subtracted_from_dark(self, library: CalibrationLibrary, profile: CalibrationProfile):
        # Pre-save a bias master
        bias_data = np.full((4, 4), 50.0, dtype=np.float32)
        library.save_master("bias", "TESTCAM", bias_data, gain=0, binning=1, ncombine=10)

        # Dark frames at 200 ADU
        frames = [np.full((4, 4), 200.0, dtype=np.uint16)] * 3
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_dark(count=3, exposure_time=2.0, gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            # 200 (median) - 50 (bias) = 150
            np.testing.assert_array_almost_equal(data, 150.0)
            assert hdul[0].header["BIASSUB"] is True  # type: ignore[index]

    def test_dark_without_bias(self, library: CalibrationLibrary, profile: CalibrationProfile):
        frames = [np.full((4, 4), 200.0, dtype=np.uint16)] * 3
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_dark(count=3, exposure_time=2.0, gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            np.testing.assert_array_almost_equal(data, 200.0)
            assert hdul[0].header["BIASSUB"] is False  # type: ignore[index]


class TestFlatBuild:
    def test_normalized_to_median_one(self, library: CalibrationLibrary, profile: CalibrationProfile):
        # Flat frames with gradient (100 on left, 200 on right)
        frame = np.zeros((4, 4), dtype=np.uint16)
        frame[:, :2] = 100
        frame[:, 2:] = 200
        frames = [frame.copy()] * 3

        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_flat(count=3, exposure_time=1.0, filter_name="Luminance", gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            median_val = np.median(data)
            # After normalization, median should be ~1.0
            assert abs(float(median_val) - 1.0) < 0.01

    def test_flat_with_bias_subtracted(self, library: CalibrationLibrary, profile: CalibrationProfile):
        # Pre-save a bias master
        bias_data = np.full((4, 4), 20.0, dtype=np.float32)
        library.save_master("bias", "TESTCAM", bias_data, gain=0, binning=1, ncombine=10)

        # Flat frames at 200
        frames = [np.full((4, 4), 200.0, dtype=np.uint16)] * 3
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_flat(count=3, exposure_time=1.0, filter_name="Red", gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            # (200 - 20) / median(180) = 1.0
            np.testing.assert_array_almost_equal(data, 1.0)
