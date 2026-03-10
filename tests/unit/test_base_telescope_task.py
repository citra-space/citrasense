"""Tests for AbstractBaseTelescopeTask and StaticTelescopeTask."""

import math
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrascope.tasks.task import Task


def _make_task_dict(**overrides):
    base = {
        "id": "task-1",
        "type": "Track",
        "status": "Pending",
        "taskStart": "2025-01-01T00:00:00Z",
        "taskStop": "2025-01-01T00:05:00Z",
        "satelliteId": "sat-iss",
        "satelliteName": "ISS",
    }
    base.update(overrides)
    return Task.from_dict(base)


def _make_hardware_adapter(**overrides):
    """Build a spec'd mock hardware adapter with sensible defaults for slew tests."""
    adapter = create_autospec(AbstractAstroHardwareAdapter, instance=True)
    adapter.observed_fov_short_deg = None
    adapter.observed_slew_rate_deg_per_s = None
    adapter.telescope_record = None
    adapter.scope_slew_rate_degrees_per_second = 5.0
    for k, v in overrides.items():
        setattr(adapter, k, v)
    return adapter


def _make_daemon():
    daemon = MagicMock()
    daemon.settings.processors_enabled = True
    daemon.telescope_record = {"id": "tel-1"}
    daemon.ground_station = {"id": "gs-1"}
    daemon.location_service.get_current_location.return_value = {
        "latitude": 37.0,
        "longitude": -122.0,
        "altitude": 100.0,
    }
    daemon.task_manager = MagicMock()
    daemon.task_manager.record_task_started = MagicMock()
    daemon.task_manager.record_task_succeeded = MagicMock()
    daemon.task_manager.record_task_failed = MagicMock()
    daemon.hardware_adapter = _make_hardware_adapter()
    return daemon


class TestFetchSatellite:
    def test_returns_satellite_with_elset(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {
            "name": "ISS",
            "elsets": [
                {
                    "tle": ["line1", "line2"],
                    "creationEpoch": "2025-01-01T00:00:00Z",
                }
            ],
        }
        task_obj = _make_task_dict()
        daemon = _make_daemon()
        logger = MagicMock()

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(api, daemon.hardware_adapter, logger, task_obj, daemon)
        result = ct.fetch_satellite()
        assert result is not None
        assert "most_recent_elset" in result

    def test_returns_none_when_no_data(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = None
        task_obj = _make_task_dict()
        daemon = _make_daemon()

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), task_obj, daemon)
        assert ct.fetch_satellite() is None

    def test_returns_none_when_no_elsets(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {"name": "ISS", "elsets": []}
        task_obj = _make_task_dict()
        daemon = _make_daemon()

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), task_obj, daemon)
        assert ct.fetch_satellite() is None


class TestGetMostRecentElset:
    def test_returns_cached_elset(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        sat_data = {"most_recent_elset": {"tle": ["a", "b"]}}
        assert ct._get_most_recent_elset(sat_data) == {"tle": ["a", "b"]}

    def test_selects_most_recent_by_epoch(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        sat_data = {
            "elsets": [
                {"tle": ["old1", "old2"], "creationEpoch": "2024-01-01T00:00:00Z"},
                {"tle": ["new1", "new2"], "creationEpoch": "2025-06-01T00:00:00Z"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data)
        assert result["tle"] == ["new1", "new2"]

    def test_empty_elsets_returns_none(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        assert ct._get_most_recent_elset({"elsets": []}) is None

    def test_missing_creation_epoch_uses_fallback(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), _make_daemon())
        sat_data = {
            "elsets": [
                {"tle": ["a", "b"]},
                {"tle": ["c", "d"], "creationEpoch": "2025-01-01T00:00:00Z"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data)
        assert result["tle"] == ["c", "d"]


class TestUploadImageAndMarkComplete:
    def test_single_filepath_str(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch("citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            result = ct.upload_image_and_mark_complete("/path/to/img.fits")

        assert result is True
        daemon.task_manager.record_task_started.assert_called_once()

    def test_multiple_filepaths(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch("citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            result = ct.upload_image_and_mark_complete(["/a.fits", "/b.fits"])

        assert result is True
        assert daemon.task_manager.processing_queue.submit.call_count == 2

    def test_processors_disabled_goes_to_upload(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.settings.processors_enabled = False
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch("citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            with patch.object(ct, "_queue_for_upload") as mock_upload:
                ct.upload_image_and_mark_complete("/img.fits")
                mock_upload.assert_called_once()

    def test_enrichment_failure_continues(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, daemon)

        with patch(
            "citrascope.tasks.scope.base_telescope_task.enrich_fits_metadata",
            side_effect=Exception("boom"),
        ):
            result = ct.upload_image_and_mark_complete("/img.fits")
            assert result is True


class TestOnProcessingComplete:
    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), daemon.hardware_adapter, MagicMock(), _make_task_dict(), daemon)

    def test_skip_upload_when_should_upload_false(self):
        ct = self._make_concrete()
        ct._pending_images = 1
        result = MagicMock()
        result.should_upload = False
        result.skip_reason = "Test skip"
        ct._on_processing_complete("/img.fits", "task-1", result)
        ct.api_client.mark_task_complete.assert_called_once_with("task-1")

    def test_feeds_plate_solve_to_adapter(self):
        ct = self._make_concrete()
        result = MagicMock()
        result.should_upload = True
        result.extracted_data = {
            "plate_solver.ra_center": 180.0,
            "plate_solver.dec_center": 45.0,
        }
        with patch.object(ct, "_queue_for_upload"):
            ct._on_processing_complete("/img.fits", "task-1", result)
        ct.hardware_adapter.update_from_plate_solve.assert_called_once()

    def test_no_result_queues_for_upload(self):
        ct = self._make_concrete()
        with patch.object(ct, "_queue_for_upload") as mock_upload:
            ct._on_processing_complete("/img.fits", "task-1", None)
            mock_upload.assert_called_once()


class TestQueueForUpload:
    def test_submits_to_upload_queue(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)
        ct._queue_for_upload("/img.fits", processing_result=None)
        daemon.task_manager.upload_queue.submit.assert_called_once()

    def test_location_service_failure_passes_none(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.location_service.get_current_location.side_effect = Exception("no GPS")
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)
        ct._queue_for_upload("/img.fits", processing_result=None)
        call_kwargs = daemon.task_manager.upload_queue.submit.call_args
        assert call_kwargs[1]["sensor_location"] is None


class TestOnImageDone:
    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), daemon)

    def test_success(self):
        ct = self._make_concrete()
        ct._pending_images = 1
        ct._on_image_done("task-1", True)
        ct.daemon.task_manager.record_task_succeeded.assert_called_once()

    def test_failure(self):
        ct = self._make_concrete()
        ct._pending_images = 1
        ct._on_image_done("task-1", False)
        ct.daemon.task_manager.record_task_failed.assert_called_once()


class TestCancellation:
    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), daemon.hardware_adapter, MagicMock(), _make_task_dict(), daemon)

    def test_cancel_sets_flag(self):
        ct = self._make_concrete()
        assert ct.is_cancelled is False
        ct.cancel()
        assert ct.is_cancelled is True

    def test_point_to_lead_exits_on_cancel_before_slew(self):
        ct = self._make_concrete()
        ct.cancel()
        with pytest.raises(RuntimeError, match=r"(?i)cancelled"):
            ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

    def test_point_to_lead_exits_on_cancel_during_slew(self):
        ct = self._make_concrete()

        call_count = 0

        def moving_then_cancel():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                ct.cancel()
            return True

        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        ct.hardware_adapter.telescope_is_moving.side_effect = moving_then_cancel
        ct.hardware_adapter.point_telescope = MagicMock()
        ct.hardware_adapter.angular_distance.return_value = 0.5

        with patch.object(
            ct, "estimate_lead_position", return_value=(MagicMock(degrees=10.0), MagicMock(degrees=20.0), 1.0)
        ):
            with pytest.raises(RuntimeError, match=r"(?i)cancelled"):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})


class TestEstimateSlewTime:
    """Tests for the estimate_slew_time pure function (trapezoidal velocity model)."""

    def test_short_distance_triangle_profile(self):
        import math

        from citrascope.tasks.scope.base_telescope_task import estimate_slew_time

        # 0.5 deg at 5 deg/s with 2 deg/s² accel → triangle (threshold = 25/2 = 12.5 deg)
        result = estimate_slew_time(0.5, max_rate=5.0, acceleration=2.0, settle_time=0.0)
        expected = 2.0 * math.sqrt(0.5 / 2.0)
        assert abs(result - expected) < 0.001

    def test_long_distance_trapezoid_profile(self):
        from citrascope.tasks.scope.base_telescope_task import estimate_slew_time

        # 30 deg at 5 deg/s with 2 deg/s² accel → trapezoid (threshold = 12.5 deg)
        result = estimate_slew_time(30.0, max_rate=5.0, acceleration=2.0, settle_time=0.0)
        expected = 30.0 / 5.0 + 5.0 / 2.0  # 6.0 + 2.5 = 8.5s
        assert abs(result - expected) < 0.001

    def test_settle_time_always_added(self):
        from citrascope.tasks.scope.base_telescope_task import estimate_slew_time

        result_zero = estimate_slew_time(0.0, max_rate=5.0, settle_time=1.5)
        assert result_zero == 1.5

        result_nonzero = estimate_slew_time(10.0, max_rate=5.0, settle_time=2.0)
        result_no_settle = estimate_slew_time(10.0, max_rate=5.0, settle_time=0.0)
        assert abs(result_nonzero - result_no_settle - 2.0) < 0.001

    def test_minimum_floor_prevents_sub_second_predictions(self):
        from citrascope.tasks.scope.base_telescope_task import estimate_slew_time

        # Tiny distance: the trapezoidal model + settle_time should still give a realistic minimum
        result = estimate_slew_time(0.01, max_rate=5.0, acceleration=2.0, settle_time=1.5)
        assert result >= 1.5  # At least settle time

    def test_negative_or_zero_inputs_return_settle_time(self):
        from citrascope.tasks.scope.base_telescope_task import estimate_slew_time

        assert estimate_slew_time(-1.0, max_rate=5.0, settle_time=1.5) == 1.5
        assert estimate_slew_time(10.0, max_rate=0.0, settle_time=1.5) == 1.5
        assert estimate_slew_time(10.0, max_rate=5.0, acceleration=0.0, settle_time=1.5) == 1.5

    def test_transition_point_continuity(self):
        """At the triangle/trapezoid boundary, both formulas should give the same time."""
        from citrascope.tasks.scope.base_telescope_task import estimate_slew_time

        max_rate, accel = 5.0, 2.0
        d_transition = max_rate**2 / accel  # 12.5 deg

        # Slightly below transition → triangle
        t_triangle = estimate_slew_time(d_transition - 0.001, max_rate, accel, settle_time=0.0)
        # Slightly above → trapezoid
        t_trapezoid = estimate_slew_time(d_transition + 0.001, max_rate, accel, settle_time=0.0)
        # At transition both should equal 2 * max_rate / accel = 5.0s
        expected = 2.0 * max_rate / accel
        assert abs(t_triangle - expected) < 0.01
        assert abs(t_trapezoid - expected) < 0.01


class TestPredictSlewTime:
    """Tests for predict_slew_time_seconds (angular distance + trapezoidal model)."""

    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = daemon.hardware_adapter
        adapter.scope_slew_rate_degrees_per_second = 5.0
        # angular_distance should be called — use a real implementation
        import math

        def real_angular_distance(ra1, dec1, ra2, dec2):
            ra1_r, ra2_r = math.radians(ra1), math.radians(ra2)
            dec1_r, dec2_r = math.radians(dec1), math.radians(dec2)
            cos_a = math.sin(dec1_r) * math.sin(dec2_r) + math.cos(dec1_r) * math.cos(dec2_r) * math.cos(ra1_r - ra2_r)
            return math.degrees(math.acos(min(1.0, max(-1.0, cos_a))))

        adapter.angular_distance.side_effect = real_angular_distance
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def test_uses_angular_distance_not_axis_max(self):
        ct = self._make_concrete()
        # Scope at (0, 0), target at (1, 1) — angular distance ≈ 1.414°, not max(1,1) = 1°
        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        target_ra = MagicMock(degrees=1.0)
        target_dec = MagicMock(degrees=1.0)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(target_ra, target_dec, None, None)):
            ct.predict_slew_time_seconds({})
        ct.hardware_adapter.angular_distance.assert_called_once()

    def test_max_rate_override(self):
        ct = self._make_concrete()
        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        target_ra = MagicMock(degrees=30.0)
        target_dec = MagicMock(degrees=0.0)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(target_ra, target_dec, None, None)):
            t_default = ct.predict_slew_time_seconds({})
            t_fast = ct.predict_slew_time_seconds({}, max_rate=10.0)
        assert t_fast < t_default


class TestConvergenceThreshold:
    """Tests for _compute_convergence_threshold (FOV-aware pointing tolerance)."""

    def _make_concrete(self, **adapter_kwargs):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(**adapter_kwargs)
        daemon.hardware_adapter = adapter
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def test_from_telescope_record(self):
        from citrascope.tasks.scope.base_telescope_task import _FOV_CONVERGENCE_FRACTION

        tr = {
            "pixelSize": 5.86,
            "focalLength": 200.0,
            "horizontalPixelCount": 4112,
            "verticalPixelCount": 3008,
        }
        ct = self._make_concrete(telescope_record=tr)
        threshold = ct._compute_convergence_threshold()

        pixel_scale = 5.86 / 200.0 * 206.265  # ~6.04 arcsec/px
        half_fov = (3008 * pixel_scale / 3600) / 2  # ~2.52 deg
        expected = half_fov * _FOV_CONVERGENCE_FRACTION
        assert abs(threshold - expected) < 0.01

    def test_fallback_when_no_telescope_record(self):
        from citrascope.tasks.scope.base_telescope_task import _DEFAULT_CONVERGENCE_THRESHOLD_DEG

        ct = self._make_concrete(telescope_record=None)
        assert ct._compute_convergence_threshold() == _DEFAULT_CONVERGENCE_THRESHOLD_DEG

    def test_plate_solved_fov_overrides_nominal(self):
        from citrascope.tasks.scope.base_telescope_task import _FOV_CONVERGENCE_FRACTION

        tr = {
            "pixelSize": 5.86,
            "focalLength": 200.0,
            "horizontalPixelCount": 4112,
            "verticalPixelCount": 3008,
        }
        ct = self._make_concrete(telescope_record=tr, observed_fov_short_deg=6.64)
        threshold = ct._compute_convergence_threshold()

        expected = (6.64 / 2) * _FOV_CONVERGENCE_FRACTION
        assert abs(threshold - expected) < 0.01

        # Verify it's different from the nominal
        ct_nominal = self._make_concrete(telescope_record=tr)
        assert threshold != ct_nominal._compute_convergence_threshold()

    def test_fallback_with_incomplete_telescope_record(self):
        from citrascope.tasks.scope.base_telescope_task import _DEFAULT_CONVERGENCE_THRESHOLD_DEG

        ct = self._make_concrete(telescope_record={"pixelSize": 5.86})
        assert ct._compute_convergence_threshold() == _DEFAULT_CONVERGENCE_THRESHOLD_DEG


class TestFormatProcessingSummary:
    """Tests for FOV dimensions in processing summary."""

    def test_fov_included_in_summary(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        result = AbstractBaseTelescopeTask._format_processing_summary(
            "abcd1234",
            {
                "plate_solver.plate_solved": True,
                "plate_solver.ra_center": 180.0,
                "plate_solver.dec_center": 45.0,
                "plate_solver.pixel_scale": 7.95,
                "plate_solver.field_width_deg": 9.08,
                "plate_solver.field_height_deg": 6.64,
            },
        )
        assert "FOV=" in result
        assert "9.08" in result
        assert "6.64" in result

    def test_fov_omitted_when_not_present(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        result = AbstractBaseTelescopeTask._format_processing_summary(
            "abcd1234",
            {"plate_solver.plate_solved": True, "plate_solver.pixel_scale": 7.95},
        )
        assert "FOV=" not in result


class TestGetFovRadiusDeg:
    """Tests for _get_fov_radius_deg helper."""

    def _make_concrete(self, **adapter_kwargs):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(**adapter_kwargs)
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def test_from_observed_fov(self):
        ct = self._make_concrete(observed_fov_short_deg=2.0)
        assert ct._get_fov_radius_deg() == 1.0

    def test_from_telescope_record(self):
        tr = {
            "pixelSize": 5.86,
            "focalLength": 200.0,
            "horizontalPixelCount": 4112,
            "verticalPixelCount": 3008,
        }
        ct = self._make_concrete(telescope_record=tr)
        pixel_scale = 5.86 / 200.0 * 206.265
        expected = (3008 * pixel_scale / 3600) / 2
        assert abs(ct._get_fov_radius_deg() - expected) < 0.01

    def test_fallback_when_nothing_available(self):
        ct = self._make_concrete()
        assert ct._get_fov_radius_deg() == 0.5


class TestComputeAngularRate:
    """Tests for compute_angular_rate."""

    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter()
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    @staticmethod
    def _make_rate_mock(arcsec_per_s: float) -> MagicMock:
        """Build a mock Skyfield Rate with .arcseconds.per_second."""
        rate = MagicMock()
        rate.arcseconds.per_second = arcsec_per_s
        return rate

    def test_pure_dec_rate(self):
        """A satellite moving purely in Dec: angular rate = Dec rate."""
        ct = self._make_concrete()
        dec_mock = MagicMock(degrees=0.0)
        ra_rate = self._make_rate_mock(0.0)
        dec_rate = self._make_rate_mock(3600.0)  # 3600 arcsec/s = 1 deg/s
        with patch.object(ct, "get_target_radec_and_rates", return_value=(None, dec_mock, ra_rate, dec_rate)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - 1.0) < 0.001

    def test_pure_ra_rate_at_equator(self):
        """At dec=0, RA rate on sky equals the RA rate directly."""
        ct = self._make_concrete()
        dec_mock = MagicMock(degrees=0.0)
        ra_rate = self._make_rate_mock(3600.0)  # 1 deg/s in RA
        dec_rate = self._make_rate_mock(0.0)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(None, dec_mock, ra_rate, dec_rate)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - 1.0) < 0.001

    def test_ra_rate_contracts_at_high_dec(self):
        """At dec=60, RA rate on sky is halved (cos(60) = 0.5)."""
        ct = self._make_concrete()
        dec_mock = MagicMock(degrees=60.0)
        ra_rate = self._make_rate_mock(3600.0)  # 1 deg/s in RA coordinate
        dec_rate = self._make_rate_mock(0.0)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(None, dec_mock, ra_rate, dec_rate)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - 0.5) < 0.001

    def test_combined_rate(self):
        """Both RA and Dec rates combine as sqrt(ra^2+dec^2) on sky."""
        ct = self._make_concrete()
        dec_mock = MagicMock(degrees=0.0)
        ra_rate = self._make_rate_mock(3600.0)  # 1 deg/s in RA
        dec_rate = self._make_rate_mock(3600.0)  # 1 deg/s in Dec
        with patch.object(ct, "get_target_radec_and_rates", return_value=(None, dec_mock, ra_rate, dec_rate)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - math.sqrt(2)) < 0.001


class TestComputeSatelliteTiming:
    """Tests for compute_satellite_timing."""

    def _make_concrete(self, fov_short_deg=2.0):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(observed_fov_short_deg=fov_short_deg)
        adapter.get_telescope_direction.return_value = (180.0, 45.0)
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def test_approaching_satellite(self):
        """Satellite is 5° away and closing at 1°/s — should report ~4s to FOV entry."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [5.0, 4.0]  # now=5°, 1s later=4°

        sat_now = (MagicMock(degrees=185.0), MagicMock(degrees=45.0), None, None)
        sat_1s = (MagicMock(degrees=184.0), MagicMock(degrees=45.0), None, None)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert abs(timing["closure_rate_deg_per_s"] - 1.0) < 0.001
        assert abs(timing["time_to_center_s"] - 5.0) < 0.01
        assert abs(timing["fov_radius_deg"] - 1.0) < 0.01
        assert abs(timing["time_to_fov_entry_s"] - 4.0) < 0.01

    def test_receding_satellite(self):
        """Satellite is moving away — closure rate <= 0, time_to_fov_entry = 0."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [3.0, 4.0]  # getting further away

        sat_now = (MagicMock(degrees=183.0), MagicMock(degrees=45.0), None, None)
        sat_1s = (MagicMock(degrees=184.0), MagicMock(degrees=45.0), None, None)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert timing["closure_rate_deg_per_s"] <= 0
        assert timing["time_to_center_s"] == 0.0
        assert timing["time_to_fov_entry_s"] == 0.0

    def test_tangential_satellite(self):
        """Satellite at constant distance — closure rate ~0, no waiting."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [3.0, 3.0]

        sat_now = (MagicMock(degrees=183.0), MagicMock(degrees=45.0), None, None)
        sat_1s = (MagicMock(degrees=183.0), MagicMock(degrees=46.0), None, None)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert timing["closure_rate_deg_per_s"] == 0.0
        assert timing["time_to_fov_entry_s"] == 0.0

    def test_already_in_fov(self):
        """Satellite is already within the FOV — time_to_fov_entry = 0."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [0.5, 0.3]  # within 1° radius, approaching

        sat_now = (MagicMock(degrees=180.5), MagicMock(degrees=45.0), None, None)
        sat_1s = (MagicMock(degrees=180.3), MagicMock(degrees=45.0), None, None)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert timing["closure_rate_deg_per_s"] > 0
        assert timing["time_to_fov_entry_s"] == 0.0


class TestEstimateLeadPositionExtraLead:
    """Tests for extra_lead_seconds parameter in estimate_lead_position."""

    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter()
        adapter.scope_slew_rate_degrees_per_second = 5.0
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def test_zero_extra_lead_matches_original(self):
        ct = self._make_concrete()
        ra_mock = MagicMock(degrees=100.0)
        dec_mock = MagicMock(degrees=30.0)
        with patch.object(ct, "predict_slew_time_seconds", return_value=3.0):
            with patch.object(ct, "get_target_radec_and_rates", return_value=(ra_mock, dec_mock, None, None)):
                _, _, total_lead = ct.estimate_lead_position({}, extra_lead_seconds=0.0)
        assert abs(total_lead - 3.0) < 0.01

    def test_extra_lead_adds_to_slew_time(self):
        ct = self._make_concrete()
        ra_mock = MagicMock(degrees=100.0)
        dec_mock = MagicMock(degrees=30.0)
        with patch.object(ct, "predict_slew_time_seconds", return_value=3.0):
            with patch.object(ct, "get_target_radec_and_rates", return_value=(ra_mock, dec_mock, None, None)):
                _, _, total_lead = ct.estimate_lead_position({}, extra_lead_seconds=10.0)
        assert abs(total_lead - 13.0) < 0.01

    def test_extra_lead_recomputes_position(self):
        """With extra lead, get_target_radec_and_rates is called again with total_lead."""
        ct = self._make_concrete()
        ra_mock = MagicMock(degrees=100.0)
        dec_mock = MagicMock(degrees=30.0)
        call_args = []

        def track_calls(sat_data, seconds_from_now=0.0):
            call_args.append(seconds_from_now)
            return (ra_mock, dec_mock, None, None)

        with patch.object(ct, "predict_slew_time_seconds", return_value=3.0):
            with patch.object(ct, "get_target_radec_and_rates", side_effect=track_calls):
                ct.estimate_lead_position({}, extra_lead_seconds=10.0)

        # Last call should be with total_lead = 3.0 + 10.0 = 13.0
        assert abs(call_args[-1] - 13.0) < 0.01


class TestAdaptiveSlewRate:
    """Tests for EMA smoothing, adapter persistence, and session initialization of slew rate."""

    def _make_concrete(self, adapter_rate=None):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(observed_slew_rate_deg_per_s=adapter_rate)
        adapter.scope_slew_rate_degrees_per_second = 5.0
        daemon.hardware_adapter = adapter
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def _run_single_slew(self, ct, slew_duration, slewed_distance):
        """Simulate one iteration of point_to_lead_position's slew loop.

        Mocks all external dependencies so the loop runs once, converges, and exits.
        Patches time.time at the module level to control slew_duration.
        """
        lead_ra = MagicMock(degrees=10.0)
        lead_dec = MagicMock(degrees=20.0)
        sat_pos = (MagicMock(degrees=10.0), MagicMock(degrees=20.0), None, None)

        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        ct.hardware_adapter.telescope_is_moving.return_value = False

        call_count = [0]

        def angular_distance_side_effect(*_args):
            call_count[0] += 1
            if call_count[0] == 1:
                return slewed_distance
            return 0.01  # converged

        ct.hardware_adapter.angular_distance.side_effect = angular_distance_side_effect

        base_time = [1000.0]

        def fake_time():
            return base_time[0]

        def point_and_advance(*_args, **_kwargs):
            base_time[0] += slew_duration

        ct.hardware_adapter.point_telescope.side_effect = point_and_advance

        with patch("citrascope.tasks.scope.base_telescope_task.time.time", side_effect=fake_time):
            with patch("citrascope.tasks.scope.base_telescope_task.time.sleep"):
                with patch.object(ct, "estimate_lead_position", return_value=(lead_ra, lead_dec, 1.0)):
                    with patch.object(ct, "get_target_radec_and_rates", return_value=sat_pos):
                        ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

    def test_first_measurement_sets_rate_directly(self):
        """With no prior rate, first observation sets effective_rate = observed_rate."""
        ct = self._make_concrete(adapter_rate=None)
        self._run_single_slew(ct, slew_duration=1.0, slewed_distance=4.0)

        stored = ct.hardware_adapter.observed_slew_rate_deg_per_s
        assert stored is not None
        assert abs(stored - 4.0) < 0.01

    def test_ema_blends_with_previous_rate(self):
        """With a prior rate, new measurement is EMA-blended."""
        from citrascope.tasks.scope.base_telescope_task import _SLEW_RATE_EMA_ALPHA

        ct = self._make_concrete(adapter_rate=3.0)
        self._run_single_slew(ct, slew_duration=1.0, slewed_distance=5.0)

        stored = ct.hardware_adapter.observed_slew_rate_deg_per_s
        expected = _SLEW_RATE_EMA_ALPHA * 5.0 + (1 - _SLEW_RATE_EMA_ALPHA) * 3.0
        assert stored is not None
        assert abs(stored - expected) < 0.01

    def test_session_persistence_initializes_from_adapter(self):
        """effective_rate should start from adapter's persisted value, not None."""
        ct = self._make_concrete(adapter_rate=6.0)

        lead_ra = MagicMock(degrees=10.0)
        lead_dec = MagicMock(degrees=20.0)
        sat_pos = (MagicMock(degrees=10.0), MagicMock(degrees=20.0), None, None)

        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        ct.hardware_adapter.telescope_is_moving.return_value = False
        ct.hardware_adapter.angular_distance.side_effect = [0.5, 0.01, 0.01]

        with patch.object(ct, "estimate_lead_position") as mock_est:
            mock_est.return_value = (lead_ra, lead_dec, 1.0)
            with patch.object(ct, "get_target_radec_and_rates", return_value=sat_pos):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

            first_call = mock_est.call_args_list[0]
            max_rate_used = first_call[1].get("max_rate")
            assert max_rate_used is not None
            assert abs(max_rate_used - 6.0) < 0.5

    def test_rate_clamped_to_bounds(self):
        """Extreme observed rates should be clamped to [MIN, MAX]."""
        from citrascope.tasks.scope.base_telescope_task import _MAX_OBSERVED_RATE_DEG_PER_S

        ct = self._make_concrete(adapter_rate=None)
        self._run_single_slew(ct, slew_duration=0.3, slewed_distance=100.0)

        stored = ct.hardware_adapter.observed_slew_rate_deg_per_s
        assert stored is not None
        assert stored <= _MAX_OBSERVED_RATE_DEG_PER_S

    def test_small_slew_below_threshold_does_not_update_rate(self):
        """Slews below _MIN_SLEW_DISTANCE_DEG should not update the rate."""
        ct = self._make_concrete(adapter_rate=4.0)

        lead_ra = MagicMock(degrees=10.0)
        lead_dec = MagicMock(degrees=20.0)
        sat_pos = (MagicMock(degrees=10.0), MagicMock(degrees=20.0), None, None)

        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        ct.hardware_adapter.telescope_is_moving.return_value = False
        ct.hardware_adapter.angular_distance.side_effect = [0.01, 0.01, 0.01]

        with patch.object(ct, "estimate_lead_position", return_value=(lead_ra, lead_dec, 1.0)):
            with patch.object(ct, "get_target_radec_and_rates", return_value=sat_pos):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

        assert ct.hardware_adapter.observed_slew_rate_deg_per_s == 4.0


class TestVerifyPointing:
    """Tests for verify_pointing helper."""

    def _make_concrete(self):
        from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter()
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), daemon)

    def test_returns_true_on_success(self):
        ct = self._make_concrete()
        ct.hardware_adapter.perform_alignment.return_value = True
        assert ct.verify_pointing(180.0, 45.0) is True

    def test_returns_false_on_failure(self):
        ct = self._make_concrete()
        ct.hardware_adapter.perform_alignment.return_value = False
        assert ct.verify_pointing(180.0, 45.0) is False
