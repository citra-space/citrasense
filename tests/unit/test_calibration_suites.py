"""Tests for calibration suite generators."""

from citrascope.calibration import FilterSlot
from citrascope.calibration.calibration_suites import all_flats_suite, bias_and_dark_suite
from citrascope.hardware.devices.camera.abstract_camera import CalibrationProfile


def _profile(current_binning: int = 1) -> CalibrationProfile:
    return CalibrationProfile(
        calibration_applicable=True,
        camera_id="TESTCAM",
        model="TestModel",
        has_mechanical_shutter=True,
        has_cooling=True,
        current_gain=100,
        current_binning=current_binning,
        current_temperature=-10.0,
        supported_binning=[1, 2, 3],
    )


class TestBiasAndDarkSuite:
    def test_uses_only_current_binning(self):
        """Suite should only generate jobs for the current binning, not all supported."""
        profile = _profile(current_binning=2)
        jobs = bias_and_dark_suite(profile, frame_count=30)

        assert len(jobs) == 2  # one bias + one dark
        for job in jobs:
            assert job["binning"] == 2
            assert job["gain"] == 100

    def test_bias_before_dark(self):
        jobs = bias_and_dark_suite(_profile(), frame_count=20)
        assert jobs[0]["frame_type"] == "bias"
        assert jobs[1]["frame_type"] == "dark"
        assert jobs[0]["count"] == 20
        assert jobs[1]["count"] == 20

    def test_dark_uses_reference_exposure(self):
        jobs = bias_and_dark_suite(_profile(), frame_count=30)
        dark = next(j for j in jobs if j["frame_type"] == "dark")
        assert dark["exposure_time"] == 30.0


class TestAllFlatsSuite:
    def test_returns_single_interleaved_job(self):
        """Should return exactly one interleaved_flat job, not per-filter jobs."""
        filters = [
            FilterSlot(position=0, name="sloan_g"),
            FilterSlot(position=1, name="sloan_r"),
            FilterSlot(position=2, name="sloan_i"),
        ]
        jobs = all_flats_suite(_profile(current_binning=1), filters, frame_count=15)

        assert len(jobs) == 1
        job = jobs[0]
        assert job["frame_type"] == "interleaved_flat"
        assert job["count"] == 15
        assert job["gain"] == 100
        assert job["binning"] == 1
        assert len(job["filters"]) == 3
        assert job["filters"][0]["name"] == "sloan_g"
        assert job["filters"][2]["position"] == 2

    def test_carries_initial_exposure(self):
        filters = [FilterSlot(position=0, name="Lum")]
        jobs = all_flats_suite(_profile(), filters, frame_count=10, initial_exposure=2.5)
        assert jobs[0]["initial_exposure"] == 2.5

    def test_empty_filters(self):
        jobs = all_flats_suite(_profile(), [], frame_count=15)
        assert len(jobs) == 1
        assert jobs[0]["filters"] == []
