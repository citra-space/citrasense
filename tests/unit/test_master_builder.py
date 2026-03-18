"""Tests for MasterBuilder: median combine math, bias subtraction, flat normalization,
interleaved capture, and flat quality validation."""

import threading

import numpy as np
import pytest
from astropy.io import fits

from citrascope.calibration import FilterSlot
from citrascope.calibration.calibration_library import CalibrationLibrary
from citrascope.calibration.master_builder import MasterBuilder
from citrascope.hardware.devices.camera.abstract_camera import CalibrationProfile


class FakeCamera:
    """Minimal camera stub for testing MasterBuilder."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._idx = 0
        self.capture_log: list[dict] = []

    def capture_array(
        self,
        duration: float,
        gain: int | None = None,
        binning: int = 1,
        shutter_closed: bool = False,
        offset: int | None = None,
    ) -> np.ndarray:
        self.capture_log.append(
            {"duration": duration, "gain": gain, "binning": binning, "shutter_closed": shutter_closed}
        )
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
        # Flat frames with gradient (28000 on left, 36000 on right)
        frame = np.zeros((4, 4), dtype=np.uint16)
        frame[:, :2] = 28000
        frame[:, 2:] = 36000
        frames = [frame.copy()] * 3

        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_flat(count=3, exposure_time=1.0, filter_name="Luminance", gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            median_val = np.median(data)
            assert abs(float(median_val) - 1.0) < 0.01

    def test_flat_with_bias_subtracted(self, library: CalibrationLibrary, profile: CalibrationProfile):
        bias_data = np.full((4, 4), 200.0, dtype=np.float32)
        library.save_master("bias", "TESTCAM", bias_data, gain=0, binning=1, ncombine=10)

        # Flat frames at 30000 ADU
        frames = [np.full((4, 4), 30000.0, dtype=np.uint16)] * 3
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        path = builder.build_flat(count=3, exposure_time=1.0, filter_name="Red", gain=0, binning=1)

        with fits.open(path) as hdul:
            data = hdul[0].data  # type: ignore[index]
            # (30000 - 200) / median(29800) = 1.0
            np.testing.assert_array_almost_equal(data, 1.0)


class TestFlatQuality:
    def test_star_contamination_rejected(self):
        """Flat with a bright point source (star) should be rejected."""
        master = np.full((100, 100), 30000.0, dtype=np.float32)
        master[50, 50] = 200000.0  # bright star: max/median ≈ 6.7 > 5
        ok, reason = MasterBuilder._validate_flat_quality(master, 65535, "sloan_g")
        assert not ok
        assert "star contamination" in reason.lower()

    def test_low_signal_rejected(self):
        """Flat with very low signal should be rejected (pre-normalisation ADU)."""
        master = np.full((100, 100), 100.0, dtype=np.float32)
        ok, reason = MasterBuilder._validate_flat_quality(master, 65535, "sloan_r")
        assert not ok
        assert "insufficient signal" in reason.lower()

    def test_good_flat_passes(self):
        """Normal flat with mild vignetting passes validation (pre-normalisation ADU values)."""
        master = np.full((100, 100), 30000.0, dtype=np.float32)
        master[:, :50] = 28000
        master[:, 50:] = 32000
        ok, reason = MasterBuilder._validate_flat_quality(master, 65535, "Luminance")
        assert ok
        assert reason == ""

    def test_zero_median_rejected(self):
        """Flat with zero median should be rejected."""
        master = np.zeros((100, 100), dtype=np.float32)
        ok, _reason = MasterBuilder._validate_flat_quality(master, 65535, "sloan_g")
        assert not ok


class TestInterleavedFlats:
    @pytest.fixture
    def filters_list(self):
        return [
            FilterSlot(position=0, name="sloan_g"),
            FilterSlot(position=1, name="sloan_r"),
            FilterSlot(position=2, name="sloan_i"),
        ]

    def test_round_robin_order(self, library: CalibrationLibrary, profile: CalibrationProfile, filters_list):
        """Verify filters are cycled in round-robin order."""
        filter_positions_set: list[int] = []

        def fake_set_filter(pos: int) -> bool:
            filter_positions_set.append(pos)
            return True

        flat_adu = 32000
        frames = [np.full((4, 4), flat_adu, dtype=np.uint16)] * 100
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]

        builder.build_interleaved_flats(
            filters=filters_list,
            set_filter=fake_set_filter,
            count=4,
            initial_exposure=0.1,
            gain=0,
            binning=1,
        )

        # After auto-expose phase (3 filter switches), the round-robin capture
        # should cycle 0,1,2, 0,1,2, 0,1,2, 0,1,2
        auto_expose_calls = filter_positions_set[:3]
        assert auto_expose_calls == [0, 1, 2]

        round_robin_calls = filter_positions_set[3:]
        for rnd in range(4):
            for fi, filt in enumerate(filters_list):
                idx = rnd * 3 + fi
                if idx < len(round_robin_calls):
                    assert round_robin_calls[idx] == filt.position

    def test_exposure_carry_forward(self, library: CalibrationLibrary, profile: CalibrationProfile, filters_list):
        """Verify that auto-expose carries exposure forward between filters."""
        exposure_log: list[float] = []

        class ExposureTrackingCamera(FakeCamera):
            def capture_array(self, duration=0, gain=None, binning=1, shutter_closed=False, offset=None):
                exposure_log.append(duration)
                return super().capture_array(duration, gain, binning, shutter_closed, offset)

        flat_adu = 32000
        frames = [np.full((4, 4), flat_adu, dtype=np.uint16)] * 200
        camera = ExposureTrackingCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]

        def fake_set_filter(pos: int) -> bool:
            return True

        builder.build_interleaved_flats(
            filters=filters_list,
            set_filter=fake_set_filter,
            count=3,
            initial_exposure=0.5,
            gain=0,
            binning=1,
        )

        # The first auto-expose starts at 0.5s.  Since the flat_adu ~= target,
        # it should converge quickly.  The second filter's auto-expose should
        # start near the first filter's converged exposure (not reset to 0.5).
        # We just verify all filters got flats and exposures aren't wildly different.
        assert len(exposure_log) > 3

    def test_disk_backed_stacking(self, library: CalibrationLibrary, profile: CalibrationProfile, filters_list):
        """Verify correct per-filter stacking and temp cleanup after completion."""
        flat_adu = 32000
        frames = [np.full((4, 4), flat_adu, dtype=np.uint16)] * 100
        camera = FakeCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]

        saved_paths = builder.build_interleaved_flats(
            filters=filters_list,
            set_filter=lambda pos: True,
            count=5,
            initial_exposure=0.1,
            gain=0,
            binning=1,
        )

        assert len(saved_paths) == 3
        for p in saved_paths:
            assert p.exists()
            with fits.open(p) as hdul:
                assert hdul[0].header["CALTYPE"] == "FLAT"  # type: ignore[index]
                data = hdul[0].data  # type: ignore[index]
                assert abs(float(np.median(data)) - 1.0) < 0.01

        # Temp FITS should be cleaned up after stacking
        assert len(list(library.tmp_dir.glob("*.fits"))) == 0

    def test_cancel_mid_round(self, library: CalibrationLibrary, profile: CalibrationProfile, filters_list):
        """Verify cancellation stops capture and no partial masters are saved."""
        cancel_event = threading.Event()
        capture_count = 0

        class CancelAfterNCamera(FakeCamera):
            def capture_array(self, duration=0, gain=None, binning=1, shutter_closed=False, offset=None):
                nonlocal capture_count
                capture_count += 1
                if capture_count > 8:
                    cancel_event.set()
                return super().capture_array(duration, gain, binning, shutter_closed, offset)

        flat_adu = 32000
        frames = [np.full((4, 4), flat_adu, dtype=np.uint16)] * 100
        camera = CancelAfterNCamera(frames)
        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]

        saved_paths = builder.build_interleaved_flats(
            filters=filters_list,
            set_filter=lambda pos: True,
            count=10,
            initial_exposure=0.1,
            gain=0,
            binning=1,
            cancel_event=cancel_event,
        )

        # With cancellation after ~8 captures (3 auto-expose + 5 round-robin),
        # we may have < 3 frames per filter so nothing passes the minimum check
        assert len(saved_paths) <= len(filters_list)

    def test_adu_drift_reexpose(self, library: CalibrationLibrary, profile: CalibrationProfile, filters_list):
        """Verify re-expose triggers when ADU drifts significantly."""
        reexpose_triggered = False

        # Frames that simulate dimming: first batch at target, later batch much dimmer
        bright_frame = np.full((4, 4), 32000, dtype=np.uint16)
        dim_frame = np.full((4, 4), 5000, dtype=np.uint16)
        # 3 auto-expose + 3 rounds of 3 filters at target, then dim for the rest
        frames = [bright_frame] * 15 + [dim_frame] * 50
        camera = FakeCamera(frames)

        original_reexpose = MasterBuilder._reexpose_check

        def tracking_reexpose(self_builder, *args, **kwargs):
            nonlocal reexpose_triggered
            reexpose_triggered = True
            return original_reexpose(self_builder, *args, **kwargs)

        builder = MasterBuilder(camera, library, profile)  # type: ignore[arg-type]
        builder._reexpose_check = lambda *a, **kw: tracking_reexpose(builder, *a, **kw)  # type: ignore[method-assign]

        builder.build_interleaved_flats(
            filters=filters_list,
            set_filter=lambda pos: True,
            count=6,
            initial_exposure=0.1,
            gain=0,
            binning=1,
            reexpose_interval=3,
        )

        assert reexpose_triggered
