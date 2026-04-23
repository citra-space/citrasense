"""Tests for AbstractBaseTelescopeTask and SiderealTelescopeTask."""

import math
import platform
from datetime import datetime, timezone
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from citrasense.astro.sidereal import SIDEREAL_RATE_DEG_PER_S
from citrasense.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    SlewRateTracker,
)
from citrasense.tasks.task import Task


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


def _make_hardware_adapter(initial_slew_samples=None, **overrides):
    """Build a spec'd mock hardware adapter with sensible defaults for slew tests.

    ``initial_slew_samples`` pre-populates the SlewRateTracker; pass ``None``
    for a fresh tracker with no history (matches a just-started daemon).
    ``observed_slew_rate_deg_per_s`` is a property on the real class backed by
    the tracker — tests should prefer inspecting ``adapter.slew_rate_tracker.mean``
    directly, but the property is also wired below for backwards-compat.
    """
    adapter = create_autospec(AbstractAstroHardwareAdapter, instance=True)
    adapter.observed_fov_short_deg = None
    adapter.telescope_record = None
    adapter.scope_slew_rate_degrees_per_second = 5.0
    adapter.select_elset_types.return_value = None

    tracker = SlewRateTracker()
    for sample in initial_slew_samples or []:
        tracker.record(sample)
    adapter.slew_rate_tracker = tracker
    # Autospec treats @property as a plain attribute on the mock; keep the
    # attribute in sync with the tracker so existing assertions keep working.
    adapter.observed_slew_rate_deg_per_s = tracker.mean

    for k, v in overrides.items():
        setattr(adapter, k, v)
    return adapter


def _make_daemon():
    daemon = MagicMock()
    daemon.settings.processors_enabled = True
    daemon.settings.skip_upload = False
    daemon.telescope_record = {"id": "tel-1"}
    daemon.ground_station = {"id": "gs-1"}
    daemon.location_service.get_current_location.return_value = {
        "latitude": 37.0,
        "longitude": -122.0,
        "altitude": 100.0,
    }
    daemon.runtime = MagicMock()
    daemon.runtime.record_task_started = MagicMock()
    daemon.runtime.record_task_succeeded = MagicMock()
    daemon.runtime.record_task_failed = MagicMock()
    daemon.hardware_adapter = _make_hardware_adapter()
    return daemon


def _daemon_kwargs(daemon):
    """Extract keyword args for AbstractBaseTelescopeTask from a daemon mock."""
    return {
        "settings": daemon.settings,
        "runtime": daemon.runtime,
        "location_service": daemon.location_service,
        "telescope_record": daemon.telescope_record,
        "ground_station": daemon.ground_station,
        "elset_cache": daemon.elset_cache,
        "processor_registry": daemon.processor_registry,
        "on_annotated_image": lambda path: setattr(daemon, "latest_annotated_image_path", path),
    }


class TestFetchSatellite:
    def test_uses_best_elset_endpoint(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {"name": "ISS", "elsets": []}
        api.get_best_elset.return_value = {
            "tle": ["best1", "best2"],
            "epoch": "2025-06-01T00:00:00Z",
            "type": "MEAN_BROUWER_XP",
        }

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        result = ct.fetch_satellite()
        assert result is not None
        assert result["most_recent_elset"]["tle"] == ["best1", "best2"]
        assert result["most_recent_elset"]["type"] == "MEAN_BROUWER_XP"
        api.get_best_elset.assert_called_once_with("sat-iss", types=None)

    def test_falls_back_to_client_side_selection(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {
            "name": "ISS",
            "elsets": [
                {"tle": ["old1", "old2"], "epoch": "2024-01-01T00:00:00Z"},
                {"tle": ["new1", "new2"], "epoch": "2025-06-01T00:00:00Z"},
            ],
        }
        api.get_best_elset.return_value = None

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        result = ct.fetch_satellite()
        assert result is not None
        assert result["most_recent_elset"]["tle"] == ["new1", "new2"]

    def test_returns_none_when_no_data(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = None

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        assert ct.fetch_satellite() is None

    def test_returns_none_when_no_elsets_anywhere(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {"name": "ISS", "elsets": []}
        api.get_best_elset.return_value = None

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        assert ct.fetch_satellite() is None

    def test_threads_adapter_elset_types_into_best_elset(self):
        """NINA-style adapters narrow the server-side ranking to classic SGP4."""
        from citrasense.astro.elset_types import CLASSIC_SGP4_ELSET_TYPES
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {"name": "ISS", "elsets": []}
        api.get_best_elset.return_value = {
            "tle": ["a", "b"],
            "epoch": "2025-06-01T00:00:00Z",
            "type": "SGP4 with Brouwer mean motion",
        }

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.hardware_adapter.select_elset_types.return_value = CLASSIC_SGP4_ELSET_TYPES
        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        assert ct.fetch_satellite() is not None
        api.get_best_elset.assert_called_once_with("sat-iss", types=CLASSIC_SGP4_ELSET_TYPES)

    def test_fallback_respects_adapter_elset_types(self):
        """When get_best_elset returns nothing, the client-side fallback must
        honour the NINA adapter's type filter — silently picking an XP TLE here
        would mis-propagate downstream.
        """
        from citrasense.astro.elset_types import CLASSIC_SGP4_ELSET_TYPES
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        api = MagicMock()
        api.get_satellite.return_value = {
            "name": "ISS",
            "elsets": [
                {
                    "tle": ["xp1", "xp2"],
                    "epoch": "2025-06-02T00:00:00Z",
                    "type": "SGP4-XP with Brouwer mean motion",
                },
                {
                    "tle": ["cls1", "cls2"],
                    "epoch": "2025-06-01T00:00:00Z",
                    "type": "SGP4 with Brouwer mean motion",
                },
            ],
        }
        api.get_best_elset.return_value = None

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.hardware_adapter.select_elset_types.return_value = CLASSIC_SGP4_ELSET_TYPES
        ct = ConcreteTask(api, daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        result = ct.fetch_satellite()
        assert result is not None
        # The XP elset is newer but must be filtered out — classic SGP4 wins.
        assert result["most_recent_elset"]["tle"] == ["cls1", "cls2"]


class TestGetMostRecentElset:
    def test_returns_cached_elset(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        sat_data = {"most_recent_elset": {"tle": ["a", "b"]}}
        assert ct._get_most_recent_elset(sat_data) == {"tle": ["a", "b"]}

    def test_selects_most_recent_by_epoch(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        sat_data = {
            "elsets": [
                {"tle": ["old1", "old2"], "epoch": "2024-01-01T00:00:00Z"},
                {"tle": ["new1", "new2"], "epoch": "2025-06-01T00:00:00Z"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data)
        assert result["tle"] == ["new1", "new2"]

    def test_empty_elsets_returns_none(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        assert ct._get_most_recent_elset({"elsets": []}) is None

    def test_missing_epoch_uses_fallback(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        sat_data = {
            "elsets": [
                {"tle": ["a", "b"]},
                {"tle": ["c", "d"], "epoch": "2025-01-01T00:00:00Z"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data)
        assert result["tle"] == ["c", "d"]

    def test_types_filter_excludes_non_matching(self):
        from citrasense.astro.elset_types import CLASSIC_SGP4_ELSET_TYPES
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        sat_data = {
            "elsets": [
                {"tle": ["xp1", "xp2"], "epoch": "2025-06-02T00:00:00Z", "type": "SGP4-XP with Brouwer mean motion"},
                {"tle": ["c1", "c2"], "epoch": "2025-06-01T00:00:00Z", "type": "SGP4 with Kozai mean motion"},
                {"tle": ["osc1", "osc2"], "epoch": "2025-06-03T00:00:00Z", "type": "Osculating"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data, types=CLASSIC_SGP4_ELSET_TYPES)
        assert result is not None
        assert result["tle"] == ["c1", "c2"]

    def test_types_filter_returns_none_when_nothing_matches(self):
        from citrasense.astro.elset_types import CLASSIC_SGP4_ELSET_TYPES
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        sat_data = {
            "elsets": [
                {"tle": ["xp1", "xp2"], "epoch": "2025-06-02T00:00:00Z", "type": "SGP4-XP with Brouwer mean motion"},
                {"tle": ["osc1", "osc2"], "epoch": "2025-06-03T00:00:00Z", "type": "Osculating"},
            ]
        }
        assert ct._get_most_recent_elset(sat_data, types=CLASSIC_SGP4_ELSET_TYPES) is None

    def test_types_none_matches_all(self):
        """Passing ``types=None`` bypasses the filter entirely (direct-adapter path)."""
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(_make_daemon()))
        sat_data = {
            "elsets": [
                {"tle": ["xp1", "xp2"], "epoch": "2025-06-02T00:00:00Z", "type": "SGP4-XP with Brouwer mean motion"},
                {"tle": ["c1", "c2"], "epoch": "2025-06-01T00:00:00Z", "type": "SGP4 with Kozai mean motion"},
            ]
        }
        result = ct._get_most_recent_elset(sat_data, types=None)
        assert result is not None
        assert result["tle"] == ["xp1", "xp2"]


class TestUploadImageAndMarkComplete:
    def test_single_filepath_str(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, **_daemon_kwargs(daemon))

        with patch("citrasense.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            result = ct.upload_image_and_mark_complete(["/path/to/img.fits"])

        assert result is True
        daemon.runtime.record_task_started.assert_called_once()

    def test_multiple_filepaths(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, **_daemon_kwargs(daemon))

        with patch("citrasense.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            result = ct.upload_image_and_mark_complete(["/a.fits", "/b.fits"])

        assert result is True
        assert daemon.runtime.processing_queue.submit.call_count == 2

    def test_processors_disabled_goes_to_upload(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.settings.processors_enabled = False
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, **_daemon_kwargs(daemon))

        with patch("citrasense.tasks.scope.base_telescope_task.enrich_fits_metadata"):
            with patch.object(ct, "_queue_for_upload") as mock_upload:
                ct.upload_image_and_mark_complete(["/img.fits"])
                mock_upload.assert_called_once()

    def test_enrichment_failure_continues(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        task_obj = _make_task_dict()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), task_obj, **_daemon_kwargs(daemon))

        with patch(
            "citrasense.tasks.scope.base_telescope_task.enrich_fits_metadata",
            side_effect=Exception("boom"),
        ):
            result = ct.upload_image_and_mark_complete(["/img.fits"])
            assert result is True


class TestOnProcessingComplete:
    def _make_concrete(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(
            MagicMock(), daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon)
        )

    def test_skip_upload_when_should_upload_false(self):
        ct = self._make_concrete()
        ct._pending_images = 1
        result = MagicMock()
        result.should_upload = False
        result.skip_reason = "Test skip"
        ct._on_processing_complete("/img.fits", "task-1", result)
        ct.api_client.mark_task_complete.assert_called_once_with("task-1")

    def test_settings_skip_upload_does_not_mark_complete(self):
        ct = self._make_concrete()
        ct.settings.skip_upload = True
        ct._pending_images = 1
        result = MagicMock()
        result.should_upload = True
        result.extracted_data = {}
        ct._on_processing_complete("/img.fits", "task-1", result)
        ct.api_client.mark_task_complete.assert_not_called()
        ct.runtime.remove_task_from_all_stages.assert_called_once_with("task-1")

    def test_feeds_plate_solve_to_adapter(self):
        ct = self._make_concrete()
        ct.pointing_report = {
            "pointing_model_correction": {
                "target_ra_deg": 179.0,
                "target_dec_deg": 44.0,
                "command_ra_deg": 179.5,
                "command_dec_deg": 44.5,
                "correction_ra_deg": 0.5,
                "correction_dec_deg": 0.5,
                "correction_total_deg": 0.707,
                "model_n_terms": 3,
                "model_n_points": 10,
            }
        }
        result = MagicMock()
        result.should_upload = True
        result.extracted_data = {
            "plate_solver.ra_center": 180.0,
            "plate_solver.dec_center": 45.0,
        }
        with patch.object(ct, "_queue_for_upload"):
            ct._on_processing_complete("/img.fits", "task-1", result)
        ct.hardware_adapter.update_from_plate_solve.assert_called_once_with(
            180.0,
            45.0,
            expected_ra_deg=179.5,
            expected_dec_deg=44.5,
            target_ra_deg=179.0,
            target_dec_deg=44.0,
        )

    def test_no_result_queues_for_upload(self):
        ct = self._make_concrete()
        with patch.object(ct, "_queue_for_upload") as mock_upload:
            ct._on_processing_complete("/img.fits", "task-1", None)
            mock_upload.assert_called_once()


class TestQueueForUpload:
    def test_submits_to_upload_queue(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        ct._queue_for_upload("/img.fits", processing_result=None)
        daemon.runtime.upload_queue.submit.assert_called_once()

    def test_location_service_failure_passes_none(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        daemon.location_service.get_current_location.side_effect = Exception("no GPS")
        ct = ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))
        ct._queue_for_upload("/img.fits", processing_result=None)
        call_kwargs = daemon.runtime.upload_queue.submit.call_args
        assert call_kwargs[1]["sensor_location"] is None


class TestOnImageDone:
    def _make_concrete(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(MagicMock(), MagicMock(), MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def test_success(self):
        ct = self._make_concrete()
        ct._pending_images = 1
        ct._on_image_done("task-1", True)
        ct.runtime.record_task_succeeded.assert_called_once()

    def test_failure(self):
        ct = self._make_concrete()
        ct._pending_images = 1
        ct._on_image_done("task-1", False)
        ct.runtime.record_task_failed.assert_called_once()


class TestCancellation:
    def _make_concrete(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        return ConcreteTask(
            MagicMock(), daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon)
        )

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

        with patch.object(ct, "estimate_lead_position", return_value=(10.0, 20.0, 1.0)):
            with pytest.raises(RuntimeError, match=r"(?i)cancelled"):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})


class TestEstimateSlewTime:
    """Tests for the estimate_slew_time pure function (trapezoidal velocity model)."""

    def test_short_distance_triangle_profile(self):
        import math

        from citrasense.tasks.scope.base_telescope_task import estimate_slew_time

        # 0.5 deg at 5 deg/s with 2 deg/s² accel → triangle (threshold = 25/2 = 12.5 deg)
        result = estimate_slew_time(0.5, max_rate=5.0, acceleration=2.0, settle_time=0.0)
        expected = 2.0 * math.sqrt(0.5 / 2.0)
        assert abs(result - expected) < 0.001

    def test_long_distance_trapezoid_profile(self):
        from citrasense.tasks.scope.base_telescope_task import estimate_slew_time

        # 30 deg at 5 deg/s with 2 deg/s² accel → trapezoid (threshold = 12.5 deg)
        result = estimate_slew_time(30.0, max_rate=5.0, acceleration=2.0, settle_time=0.0)
        expected = 30.0 / 5.0 + 5.0 / 2.0  # 6.0 + 2.5 = 8.5s
        assert abs(result - expected) < 0.001

    def test_settle_time_always_added(self):
        from citrasense.tasks.scope.base_telescope_task import estimate_slew_time

        result_zero = estimate_slew_time(0.0, max_rate=5.0, settle_time=1.5)
        assert result_zero == 1.5

        result_nonzero = estimate_slew_time(10.0, max_rate=5.0, settle_time=2.0)
        result_no_settle = estimate_slew_time(10.0, max_rate=5.0, settle_time=0.0)
        assert abs(result_nonzero - result_no_settle - 2.0) < 0.001

    def test_minimum_floor_prevents_sub_second_predictions(self):
        from citrasense.tasks.scope.base_telescope_task import estimate_slew_time

        # Tiny distance: the trapezoidal model + settle_time should still give a realistic minimum
        result = estimate_slew_time(0.01, max_rate=5.0, acceleration=2.0, settle_time=1.5)
        assert result >= 1.5  # At least settle time

    def test_negative_or_zero_inputs_return_settle_time(self):
        from citrasense.tasks.scope.base_telescope_task import estimate_slew_time

        assert estimate_slew_time(-1.0, max_rate=5.0, settle_time=1.5) == 1.5
        assert estimate_slew_time(10.0, max_rate=0.0, settle_time=1.5) == 1.5
        assert estimate_slew_time(10.0, max_rate=5.0, acceleration=0.0, settle_time=1.5) == 1.5

    def test_transition_point_continuity(self):
        """At the triangle/trapezoid boundary, both formulas should give the same time."""
        from citrasense.tasks.scope.base_telescope_task import estimate_slew_time

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
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

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
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def test_uses_angular_distance_not_axis_max(self):
        ct = self._make_concrete()
        # Scope at (0, 0), target at (1, 1) — angular distance ≈ 1.414°, not max(1,1) = 1°
        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(1.0, 1.0, 0.0, 0.0)):
            ct.predict_slew_time_seconds({})
        ct.hardware_adapter.angular_distance.assert_called_once()

    def test_max_rate_override(self):
        ct = self._make_concrete()
        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(30.0, 0.0, 0.0, 0.0)):
            t_default = ct.predict_slew_time_seconds({})
            t_fast = ct.predict_slew_time_seconds({}, max_rate=10.0)
        assert t_fast < t_default


class TestConvergenceThreshold:
    """Tests for _compute_convergence_threshold (FOV-aware pointing tolerance)."""

    def _make_concrete(self, **adapter_kwargs):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(**adapter_kwargs)
        daemon.hardware_adapter = adapter
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def test_from_telescope_record(self):
        from citrasense.tasks.scope.base_telescope_task import _FOV_CONVERGENCE_FRACTION

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
        from citrasense.tasks.scope.base_telescope_task import _DEFAULT_CONVERGENCE_THRESHOLD_DEG

        ct = self._make_concrete(telescope_record=None)
        assert ct._compute_convergence_threshold() == _DEFAULT_CONVERGENCE_THRESHOLD_DEG

    def test_plate_solved_fov_overrides_nominal(self):
        from citrasense.tasks.scope.base_telescope_task import _FOV_CONVERGENCE_FRACTION

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
        from citrasense.tasks.scope.base_telescope_task import _DEFAULT_CONVERGENCE_THRESHOLD_DEG

        ct = self._make_concrete(telescope_record={"pixelSize": 5.86})
        assert ct._compute_convergence_threshold() == _DEFAULT_CONVERGENCE_THRESHOLD_DEG


class TestFormatProcessingSummary:
    """Tests for FOV dimensions in processing summary."""

    def test_fov_included_in_summary(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

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
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        result = AbstractBaseTelescopeTask._format_processing_summary(
            "abcd1234",
            {"plate_solver.plate_solved": True, "plate_solver.pixel_scale": 7.95},
        )
        assert "FOV=" not in result


class TestGetFovRadiusDeg:
    """Tests for _get_fov_radius_deg helper."""

    def _make_concrete(self, **adapter_kwargs):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(**adapter_kwargs)
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

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
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter()
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def test_pure_dec_rate(self):
        """A satellite moving purely in Dec: angular rate = Dec rate."""
        ct = self._make_concrete()
        # get_target_radec_and_rates returns (ra_deg, dec_deg, ra_rate_deg_s, dec_rate_deg_s)
        with patch.object(ct, "get_target_radec_and_rates", return_value=(0.0, 0.0, 0.0, 1.0)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - 1.0) < 0.001

    def test_pure_ra_rate_at_equator(self):
        """At dec=0, RA rate on sky equals the RA rate directly."""
        ct = self._make_concrete()
        with patch.object(ct, "get_target_radec_and_rates", return_value=(0.0, 0.0, 1.0, 0.0)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - 1.0) < 0.001

    def test_ra_rate_contracts_at_high_dec(self):
        """At dec=60, RA rate on sky is halved (cos(60) = 0.5)."""
        ct = self._make_concrete()
        with patch.object(ct, "get_target_radec_and_rates", return_value=(0.0, 60.0, 1.0, 0.0)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - 0.5) < 0.001

    def test_combined_rate(self):
        """Both RA and Dec rates combine as sqrt(ra^2+dec^2) on sky."""
        ct = self._make_concrete()
        with patch.object(ct, "get_target_radec_and_rates", return_value=(0.0, 0.0, 1.0, 1.0)):
            rate = ct.compute_angular_rate({})
        assert abs(rate - math.sqrt(2)) < 0.001

    def test_inertial_parameter_forwarded(self):
        """compute_angular_rate passes inertial kwarg to get_target_radec_and_rates."""
        ct = self._make_concrete()
        with patch.object(ct, "get_target_radec_and_rates", return_value=(0.0, 0.0, 1.0, 0.0)) as mock:
            ct.compute_angular_rate({}, inertial=True)
        mock.assert_called_once_with({}, inertial=True)

    def test_inertial_false_by_default(self):
        """compute_angular_rate defaults to inertial=False (Earth-fixed)."""
        ct = self._make_concrete()
        with patch.object(ct, "get_target_radec_and_rates", return_value=(0.0, 0.0, 1.0, 0.0)) as mock:
            ct.compute_angular_rate({})
        mock.assert_called_once_with({}, inertial=False)


class TestComputeSatelliteTiming:
    """Tests for compute_satellite_timing."""

    def _make_concrete(self, fov_short_deg=2.0):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(observed_fov_short_deg=fov_short_deg)
        adapter.get_telescope_direction.return_value = (180.0, 45.0)
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def test_approaching_satellite(self):
        """Satellite is 5° away and closing at 1°/s — should report ~4s to FOV entry."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [5.0, 4.0]  # now=5°, 1s later=4°

        sat_now = (185.0, 45.0, 0.0, 0.0)
        sat_1s = (184.0, 45.0, 0.0, 0.0)
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

        sat_now = (183.0, 45.0, 0.0, 0.0)
        sat_1s = (184.0, 45.0, 0.0, 0.0)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert timing["closure_rate_deg_per_s"] <= 0
        assert timing["time_to_center_s"] == 0.0
        assert timing["time_to_fov_entry_s"] == 0.0

    def test_tangential_satellite(self):
        """Satellite at constant distance — closure rate ~0, no waiting."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [3.0, 3.0]

        sat_now = (183.0, 45.0, 0.0, 0.0)
        sat_1s = (183.0, 46.0, 0.0, 0.0)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert timing["closure_rate_deg_per_s"] == 0.0
        assert timing["time_to_fov_entry_s"] == 0.0

    def test_already_in_fov(self):
        """Satellite is already within the FOV — time_to_fov_entry = 0."""
        ct = self._make_concrete(fov_short_deg=2.0)
        ct.hardware_adapter.angular_distance.side_effect = [0.5, 0.3]  # within 1° radius, approaching

        sat_now = (180.5, 45.0, 0.0, 0.0)
        sat_1s = (180.3, 45.0, 0.0, 0.0)
        with patch.object(ct, "get_target_radec_and_rates", side_effect=[sat_now, sat_1s]):
            timing = ct.compute_satellite_timing({})

        assert timing["closure_rate_deg_per_s"] > 0
        assert timing["time_to_fov_entry_s"] == 0.0


class TestEstimateLeadPositionExtraLead:
    """Tests for extra_lead_seconds parameter in estimate_lead_position."""

    def _make_concrete(self):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter()
        adapter.scope_slew_rate_degrees_per_second = 5.0
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def test_zero_extra_lead_matches_original(self):
        ct = self._make_concrete()
        with patch.object(ct, "predict_slew_time_seconds", return_value=3.0):
            with patch.object(ct, "get_target_radec_and_rates", return_value=(100.0, 30.0, 0.0, 0.0)):
                _, _, total_lead = ct.estimate_lead_position({}, extra_lead_seconds=0.0)
        assert abs(total_lead - 3.0) < 0.01

    def test_extra_lead_adds_to_slew_time(self):
        ct = self._make_concrete()
        with patch.object(ct, "predict_slew_time_seconds", return_value=3.0):
            with patch.object(ct, "get_target_radec_and_rates", return_value=(100.0, 30.0, 0.0, 0.0)):
                _, _, total_lead = ct.estimate_lead_position({}, extra_lead_seconds=10.0)
        assert abs(total_lead - 13.0) < 0.01

    def test_extra_lead_recomputes_position(self):
        """With extra lead, get_target_radec_and_rates is called again with total_lead."""
        ct = self._make_concrete()
        call_args = []

        def track_calls(sat_data, seconds_from_now=0.0):
            call_args.append(seconds_from_now)
            return (100.0, 30.0, 0.0, 0.0)

        with patch.object(ct, "predict_slew_time_seconds", return_value=3.0):
            with patch.object(ct, "get_target_radec_and_rates", side_effect=track_calls):
                ct.estimate_lead_position({}, extra_lead_seconds=10.0)

        # Last call should be with total_lead = 3.0 + 10.0 = 13.0
        assert abs(call_args[-1] - 13.0) < 0.01


class TestAdaptiveSlewRate:
    """Tests for rolling-mean slew rate tracking via ``SlewRateTracker``."""

    def _make_concrete(self, initial_samples=None):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        daemon = _make_daemon()
        adapter = _make_hardware_adapter(initial_slew_samples=initial_samples)
        adapter.scope_slew_rate_degrees_per_second = 5.0
        daemon.hardware_adapter = adapter
        return ConcreteTask(MagicMock(), adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def _run_single_slew(self, ct, slew_duration, slewed_distance):
        """Simulate one iteration of point_to_lead_position's slew loop.

        Mocks all external dependencies so the loop runs once, converges, and exits.
        Patches time.time at the module level to control slew_duration.
        """
        sat_pos = (10.0, 20.0, 0.0, 0.0)

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

        def point_and_advance(ra, dec):
            base_time[0] += slew_duration
            return {
                "target_ra_deg": ra,
                "target_dec_deg": dec,
                "correction_ra_deg": 0.0,
                "correction_dec_deg": 0.0,
                "correction_total_deg": 0.0,
                "command_ra_deg": ra,
                "command_dec_deg": dec,
                "model_n_terms": 0,
                "model_n_points": 0,
            }

        ct.hardware_adapter.point_telescope.side_effect = point_and_advance

        with patch("citrasense.tasks.scope.base_telescope_task.time.time", side_effect=fake_time):
            with patch("citrasense.tasks.scope.base_telescope_task.time.sleep"):
                with patch.object(ct, "estimate_lead_position", return_value=(10.0, 20.0, 1.0)):
                    with patch.object(ct, "get_target_radec_and_rates", return_value=sat_pos):
                        ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

    def test_first_measurement_sets_rate_directly(self):
        """With an empty tracker, first observation becomes the mean."""
        ct = self._make_concrete(initial_samples=None)
        # Duration/distance chosen to exceed _MIN_MOTION_TIME_S (2.0s) and
        # _MIN_SLEW_DISTANCE_DEG (1.0°) so the sample actually gets recorded.
        self._run_single_slew(ct, slew_duration=2.5, slewed_distance=10.0)

        tracker = ct.hardware_adapter.slew_rate_tracker
        assert tracker.count == 1
        assert tracker.mean is not None
        assert abs(tracker.mean - 4.0) < 0.01

    def test_rolling_mean_averages_recent_samples(self):
        """With prior samples, a new measurement contributes a simple (1/N) share."""
        ct = self._make_concrete(initial_samples=[3.0, 3.0, 3.0])
        self._run_single_slew(ct, slew_duration=2.5, slewed_distance=12.5)

        tracker = ct.hardware_adapter.slew_rate_tracker
        assert tracker.count == 4
        expected = (3.0 + 3.0 + 3.0 + 5.0) / 4.0
        assert tracker.mean is not None
        assert abs(tracker.mean - expected) < 0.01

    def test_session_persistence_seeds_lead_estimate(self):
        """With prior samples, the first lead-time estimate uses the rolling mean."""
        ct = self._make_concrete(initial_samples=[6.0, 6.0, 6.0])

        sat_pos = (10.0, 20.0, 0.0, 0.0)

        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        ct.hardware_adapter.telescope_is_moving.return_value = False
        ct.hardware_adapter.angular_distance.side_effect = [0.5, 0.01, 0.01]

        with patch.object(ct, "estimate_lead_position") as mock_est:
            mock_est.return_value = (10.0, 20.0, 1.0)
            with patch.object(ct, "get_target_radec_and_rates", return_value=sat_pos):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

            first_call = mock_est.call_args_list[0]
            max_rate_used = first_call[1].get("max_rate")
            assert max_rate_used is not None
            assert abs(max_rate_used - 6.0) < 0.5

    def test_rate_clamped_to_bounds(self):
        """Extreme observed rates should be clamped by the tracker."""
        ct = self._make_concrete(initial_samples=None)
        # 250° / 2.5s = 100 deg/s — well above the tracker's hi bound,
        # while still passing the steady-state duration/distance guards.
        self._run_single_slew(ct, slew_duration=2.5, slewed_distance=250.0)

        tracker = ct.hardware_adapter.slew_rate_tracker
        assert tracker.mean is not None
        # Default hi=50.0 on SlewRateTracker.
        assert tracker.mean <= 50.0

    def test_small_slew_below_threshold_does_not_record(self):
        """Slews below _MIN_SLEW_DISTANCE_DEG should not append a sample."""
        ct = self._make_concrete(initial_samples=[4.0])

        sat_pos = (10.0, 20.0, 0.0, 0.0)

        ct.hardware_adapter.get_telescope_direction.return_value = (0.0, 0.0)
        ct.hardware_adapter.telescope_is_moving.return_value = False
        ct.hardware_adapter.angular_distance.side_effect = [0.01, 0.01, 0.01]

        with patch.object(ct, "estimate_lead_position", return_value=(10.0, 20.0, 1.0)):
            with patch.object(ct, "get_target_radec_and_rates", return_value=sat_pos):
                ct.point_to_lead_position({"most_recent_elset": {"tle": ["a", "b"]}})

        tracker = ct.hardware_adapter.slew_rate_tracker
        assert tracker.count == 1
        assert tracker.mean == 4.0


class TestAdaptiveExposure:
    """Tests for compute_adaptive_exposure."""

    def _make_concrete(self, telescope_record=None, **settings_overrides):
        from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

        class ConcreteTask(AbstractBaseTelescopeTask):
            def execute(self):
                pass

        api = MagicMock()
        adapter = _make_hardware_adapter(telescope_record=telescope_record)
        daemon = _make_daemon()

        daemon.settings.adaptive_exposure = True
        daemon.settings.adaptive_exposure_max_trail_pixels = 3.0
        daemon.settings.adaptive_exposure_min_seconds = 0.1
        daemon.settings.adaptive_exposure_max_seconds = 30.0
        for k, v in settings_overrides.items():
            setattr(daemon.settings, k, v)

        return ConcreteTask(api, adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))

    def _telescope_record(self):
        return {
            "id": "tel-1",
            "pixelSize": 3.76,
            "focalLength": 500.0,
            "horizontalPixelCount": 4096,
            "verticalPixelCount": 3072,
        }

    def test_leo_fast_mover_short_exposure(self):
        """A LEO satellite at ~0.5 deg/s should produce a sub-second exposure."""
        ct = self._make_concrete(telescope_record=self._telescope_record())
        # plate_scale = 3.76 / 500 * 206.265 = ~1.551 arcsec/px
        # max_trail = 3.0 * 1.551 = 4.653 arcsec
        # angular_rate = 0.5 deg/s = 1800 arcsec/s
        # exposure = 4.653 / 1800 = ~0.00258s → clamped to min 0.1s
        result = ct.compute_adaptive_exposure(0.5)
        assert result is not None
        assert result == pytest.approx(0.1, abs=0.001)

    def test_geo_slow_mover_long_exposure(self):
        """A GEO satellite at ~0.004 deg/s should produce a long exposure (clamped to max)."""
        ct = self._make_concrete(telescope_record=self._telescope_record())
        # angular_rate = 0.004 deg/s = 14.4 arcsec/s
        # exposure = 4.653 / 14.4 = ~0.323s
        result = ct.compute_adaptive_exposure(0.004)
        assert result is not None
        assert 0.3 < result < 0.4

    def test_very_slow_mover_clamped_to_max(self):
        """Near-stationary object should be clamped to max exposure."""
        ct = self._make_concrete(telescope_record=self._telescope_record())
        # angular_rate = 0.0001 deg/s = 0.36 arcsec/s
        # exposure = 4.653 / 0.36 = ~12.9s → within max
        # angular_rate = 0.00001 deg/s = 0.036 arcsec/s
        # exposure = 4.653 / 0.036 = ~129.25s → clamped to 30s
        result = ct.compute_adaptive_exposure(0.00001)
        assert result is not None
        assert result == pytest.approx(30.0, abs=0.001)

    def test_zero_angular_rate_returns_none(self):
        ct = self._make_concrete(telescope_record=self._telescope_record())
        assert ct.compute_adaptive_exposure(0.0) is None

    def test_negative_angular_rate_returns_none(self):
        ct = self._make_concrete(telescope_record=self._telescope_record())
        assert ct.compute_adaptive_exposure(-1.0) is None

    def test_no_telescope_record_returns_none(self):
        ct = self._make_concrete(telescope_record=None)
        assert ct.compute_adaptive_exposure(0.5) is None

    def test_missing_focal_length_returns_none(self):
        tr = self._telescope_record()
        del tr["focalLength"]
        ct = self._make_concrete(telescope_record=tr)
        assert ct.compute_adaptive_exposure(0.5) is None

    def test_custom_trail_limit(self):
        """Larger trail limit should allow proportionally longer exposures."""
        ct = self._make_concrete(
            telescope_record=self._telescope_record(),
            adaptive_exposure_max_trail_pixels=10.0,
        )
        result = ct.compute_adaptive_exposure(0.5)
        assert result is not None
        # 10 px trail → 3.33x longer than 3 px trail, but both get clamped to min
        # plate_scale = 1.551 arcsec/px, max_trail = 10 * 1.551 = 15.51 arcsec
        # exposure = 15.51 / 1800 = ~0.00862s → clamped to 0.1s
        assert result == pytest.approx(0.1, abs=0.001)

    def test_moderate_rate_produces_unclamped_exposure(self):
        """A medium angular rate should produce an exposure between min and max."""
        ct = self._make_concrete(telescope_record=self._telescope_record())
        # angular_rate = 0.01 deg/s = 36 arcsec/s
        # exposure = 4.653 / 36 = ~0.129s
        result = ct.compute_adaptive_exposure(0.01)
        assert result is not None
        assert 0.1 < result < 30.0


# ---------------------------------------------------------------------------
# XP propagation gate + parity + rate-units tests. These exercise the real
# keplemon path end-to-end through get_target_radec_and_rates — no mock of
# propagation itself. They're the "would the issue-300 bug come back" safety
# net: if someone re-introduces a skyfield/sgp4 dependency on the hot path,
# or swaps rate units, or stops catching NaN, one of these fires.
# ---------------------------------------------------------------------------

# Real SGP4-XP TLE that triggered issue #300 on citrasense-galileo
# (2026-04-19 incident, DIRECTV 15, task dd782a8d). Pulled live from the
# backend ``GET /satellites/{id}/elsets`` endpoint and verified to:
#   - NaN in skyfield/sgp4 2.25 on galileo's Raspberry Pi (the actual
#     observed failure: RA/Dec/range all NaN),
#   - Propagate to finite values in keplemon.
# Column 63 on line 1 is ``4``, the XP ephemeris-type marker; the ``.03000``
# ballistic-coefficient and ``10000-1`` Bstar fields are XP-specific
# encodings that classic sgp4 silently mis-parses into NaN-producing state.
_DIRECTV15_XP_TLE = (
    "1     0U          26110.51972782  .00000000  0.03000  10000-1 4    02",
    "2     0   0.0088 131.7837 0000602  81.2681  79.7779  1.00268173    06",
)

# Pinned UTC time for DIRECTV15-XP propagation. DOY 110.51972782 of 2026
# = 2026-04-20 12:28:24 UTC, i.e. the TLE's own epoch. Evaluating near-
# epoch keeps SGP4 in its sweet spot and (more importantly here) makes
# the tests deterministic — without this, a 6-month-old wall clock could
# push the XP propagator into a regime where the NaN-vs-finite signal
# gets confounded with ordinary SGP4 drift.
_DIRECTV15_TLE_EPOCH = datetime(2026, 4, 20, 12, 28, 24, tzinfo=timezone.utc)

# Back-compat alias — older tests below reference this name.
_GOES18_XP_TLE = _DIRECTV15_XP_TLE
_GOES18_XP_TLE_EPOCH = _DIRECTV15_TLE_EPOCH

# Plain SGP4 (GP) TLE for a GEO sat; sanity baseline.
_GOES18_GP_TLE = (
    "1 51850U 22021A   26103.19597709 +.00000000  00000+0  00000+0 0  9995",
    "2 51850   0.0118 102.2391 0001121   4.4195  28.2815  1.00267201000001",
)
# DOY 103.19597709 of 2026 = 2026-04-13 04:42:12 UTC (the TLE's own epoch).
# Pinned so the propagation-regression tests below don't drift with wall clock.
_GOES18_GP_TLE_EPOCH = datetime(2026, 4, 13, 4, 42, 12, tzinfo=timezone.utc)

# ISS (ZARYA) — real SpaceTrack TLE from 2024. 51.6° inclination means
# topocentric declination sweeps ± ~60° over an orbit, so picking any
# moment in-pass gives a solid "high dec" test sample — exactly what the
# inertial-vs-earth-fixed subtraction needs to prove it's cos(dec)-
# independent. Epoch is historical and deterministic; combine with
# ``target_dt`` when propagating so results don't drift with wall-clock.
_ISS_GP_TLE = (
    "1 25544U 98067A   24100.52307878  .00012183  00000+0  22025-3 0  9997",
    "2 25544  51.6395  79.5476 0003815 224.6891 288.9172 15.49848886447068",
)

# Deterministic reference time for ISS_GP_TLE. Close to the TLE epoch
# (2024-04-09 ≈ DOY 100.5) so SGP4 is evaluating near-epoch where its
# accuracy is best. Chosen by inspection so the ISS is at abs(dec) > 30°
# from the fixture observer location, which is what exercises the
# cos(dec)-independence of the sidereal subtraction.
_ISS_HIGH_DEC_DT = datetime(2024, 4, 9, 13, 0, 0, tzinfo=timezone.utc)


def _make_sat_data(tle: tuple[str, str], name: str = "TEST-SAT") -> dict:
    return {
        "name": name,
        "most_recent_elset": {
            "tle": [tle[0], tle[1]],
            "epoch": "2026-04-13T04:42:12Z",
            "type": "MEAN_BROUWER_XP",
        },
    }


def _make_concrete_for_propagation():
    """Build a ConcreteTask wired with a real location service (not propagation-mocked)."""
    from citrasense.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask

    class ConcreteTask(AbstractBaseTelescopeTask):
        def execute(self):
            pass

    daemon = _make_daemon()
    return ConcreteTask(MagicMock(), daemon.hardware_adapter, MagicMock(), _make_task_dict(), **_daemon_kwargs(daemon))


class TestXPTLEPropagationGate:
    """End-to-end: a real XP TLE through the real keplemon path yields finite floats.

    On main (skyfield path) this returns Angle/Rate objects, not floats, and for
    certain XP TLEs produces NaN coordinates. On this branch (keplemon path)
    it must always return four finite plain floats.
    """

    def test_xp_tle_returns_four_finite_floats(self):
        ct = _make_concrete_for_propagation()
        ra, dec, ra_rate, dec_rate = ct.get_target_radec_and_rates(
            _make_sat_data(_GOES18_XP_TLE, "GOES 18"),
            target_dt=_GOES18_XP_TLE_EPOCH,
        )

        # Plain floats, not Angle/Rate wrappers. Fails loudly on main.
        assert isinstance(ra, float)
        assert isinstance(dec, float)
        assert isinstance(ra_rate, float)
        assert isinstance(dec_rate, float)

        # All finite — no silent NaN.
        for label, val in [("ra", ra), ("dec", dec), ("ra_rate", ra_rate), ("dec_rate", dec_rate)]:
            assert math.isfinite(val), f"{label} is not finite: {val}"

        # RA in [0, 360), Dec in [-90, 90]. Values are small but non-degenerate.
        assert 0.0 <= ra < 360.0
        assert -90.0 <= dec <= 90.0

    def test_gp_tle_also_works(self):
        """Regression guard: we didn't break plain SGP4 TLEs in the process."""
        ct = _make_concrete_for_propagation()
        ra, dec, ra_rate, dec_rate = ct.get_target_radec_and_rates(
            _make_sat_data(_GOES18_GP_TLE, "GOES 18"),
            target_dt=_GOES18_GP_TLE_EPOCH,
        )
        for val in (ra, dec, ra_rate, dec_rate):
            assert isinstance(val, float)
            assert math.isfinite(val)

    def test_inertial_kwarg_changes_ra_rate(self):
        """``inertial=True`` returns J2000 inertial rate; ``inertial=False`` subtracts the sidereal term.

        Uses ISS pinned at 2024-04-09 13:00 UTC where the sat sits at
        |dec| ≈ 62.7° from the fixture observer — well above the 30° threshold
        the plan called out. That's the point of this test: the subtraction is
        a pure *coordinate* rate (sidereal drive moves the mount's RA axis at
        a constant ~15.041"/s regardless of declination), so the invariant
        ``ra_rate_inertial - ra_rate_earth_fixed == SIDEREAL_RATE_DEG_PER_S``
        must hold independently of cos(dec). Any cos(dec) projection onto the
        sky happens downstream in ``compute_angular_rate`` (see
        ``TestAngularRateProjection`` below).
        """
        ct = _make_concrete_for_propagation()
        sat_data = _make_sat_data(_ISS_GP_TLE, "ISS")
        _, dec_i, ra_rate_i, _ = ct.get_target_radec_and_rates(sat_data, inertial=True, target_dt=_ISS_HIGH_DEC_DT)
        _, dec_m, ra_rate_m, _ = ct.get_target_radec_and_rates(sat_data, inertial=False, target_dt=_ISS_HIGH_DEC_DT)

        # Fixture sanity: we're actually at high dec. If this assertion ever
        # starts failing it means the pinned time or TLE got edited out from
        # under us; fix the pin, not the tolerance below.
        assert abs(dec_i) > 30.0, f"ISS pinned sample is not high-dec: {dec_i}"
        assert dec_i == pytest.approx(dec_m, abs=1e-9)

        # Earth-fixed (mount) rate = inertial − sidereal. Pure coordinate rate,
        # no cos(dec). Tolerance: ~1e-9 deg/s (~3.6e-6 arcsec/s) since this
        # is literally subtracting a constant from a float.
        assert (ra_rate_i - ra_rate_m) == pytest.approx(SIDEREAL_RATE_DEG_PER_S, abs=1e-9)


class TestSkyfieldFailsOnXPTLE:
    """Canary tests that document *why* this issue exists.

    The 2026-04-19 galileo incident looked like "every GEO task fails with
    ``cannot convert float NaN to integer``". Root cause: the backend's
    ``GET /satellites/{id}/elsets`` endpoint returns SGP4-XP elsets as the
    preferred "best" elset for many GEOs, and skyfield's bundled sgp4
    implementation doesn't support SGP4-XP. What skyfield *does* with XP
    input turns out to be platform-dependent:

        linux/arm64 (the Pi fleet): NaN — exactly reproduces the log.
        darwin/arm64 (dev macs):    silently mis-propagates with ~arcmin
                                    error (finite, plausible-looking, wrong).

    Both are broken. Both are fixed by the keplemon migration. The NaN
    variant is platform-specific so the hard-NaN assertion below only runs
    on linux/arm64, where the prod failure actually lives. The
    "keplemon-produces-finite" assertion runs everywhere — that's the fix.

    The exact XP TLE below was pulled live from
    ``https://dev.api.citra.space/satellites/849d…/elsets?limit=1`` on
    2026-04-20, which is the same endpoint ``fetch_satellite`` hits at
    task time.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        not (platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64")),
        reason=(
            "skyfield's XP-TLE behavior is platform-specific: it NaN's on "
            "linux/arm64 (the Pi fleet, where the incident was reproduced) "
            "but silently mis-propagates on darwin. This NaN assertion only "
            "runs where it reliably reproduces the prod failure."
        ),
    )
    def test_skyfield_produces_nan_on_xp_tle_on_linux_arm64(self):
        from datetime import datetime, timezone

        from skyfield.api import EarthSatellite, load, wgs84

        ts = load.timescale()
        obs = wgs84.latlon(38.821757, -104.855860, elevation_m=1865.1)  # galileo GPS
        sat = EarthSatellite(_DIRECTV15_XP_TLE[0], _DIRECTV15_XP_TLE[1], "DIRECTV 15", ts)
        topo = (sat - obs).at(ts.from_datetime(datetime.now(timezone.utc)))
        ra, dec, dist = topo.radec()

        # These three assertions are the exact failure we saw on galileo.
        assert math.isnan(ra.degrees), f"skyfield should NaN XP TLE on linux/arm64 but got RA={ra.degrees}"  # type: ignore[attr-defined]
        assert math.isnan(dec.degrees), f"skyfield should NaN XP TLE on linux/arm64 but got Dec={dec.degrees}"  # type: ignore[attr-defined]
        assert math.isnan(dist.km), f"skyfield should NaN XP TLE on linux/arm64 but got km={dist.km}"  # type: ignore[attr-defined]

    def test_keplemon_produces_finite_on_same_xp_tle(self):
        """Same TLE, branch path: must be finite on every platform (this is the fix).

        Pinned at the TLE's own epoch (2026-04-13 04:42 UTC) via ``target_dt``
        so that as wall-clock drifts further from epoch, the test keeps
        exercising the same propagator state rather than quietly shifting
        into a regime where SGP4 accuracy degrades on its own merits.
        """
        ct = _make_concrete_for_propagation()
        ra, dec, ra_rate, dec_rate = ct.get_target_radec_and_rates(
            _make_sat_data(_DIRECTV15_XP_TLE, "DIRECTV 15"),
            target_dt=_DIRECTV15_TLE_EPOCH,
        )
        assert math.isfinite(ra)
        assert math.isfinite(dec)
        assert math.isfinite(ra_rate)
        assert math.isfinite(dec_rate)
        # Sanity: DIRECTV 15 is a near-equatorial GEO — dec should be small.
        assert -15.0 < dec < 15.0, f"DIRECTV 15 dec out of sane band: {dec}"


class TestPropagationParityWithSkyfield:
    """Cross-validate the new keplemon path against the old skyfield path
    on a plain SGP4 TLE (where both libraries agree).

    Marked slow so it doesn't run by default — it imports skyfield just for
    the comparison. An XP TLE is intentionally NOT compared here because
    skyfield silently mis-interprets XP extensions and the whole point of
    issue 300 is that those answers are wrong.
    """

    @pytest.mark.slow
    def test_keplemon_matches_skyfield_on_gp_tle(self):
        """Pin both paths to the GOES-18 TLE epoch (DOY 103.19597709 of 2026)
        so the parity assertion doesn't silently degrade as wall-clock drifts
        further from epoch and SGP4 solutions diverge on their own.
        """
        from skyfield.api import EarthSatellite, load, wgs84

        # DOY 103.19597709 of 2026 ≈ 2026-04-13 04:42:12 UTC.
        goes18_epoch = datetime(2026, 4, 13, 4, 42, 12, tzinfo=timezone.utc)

        ct = _make_concrete_for_propagation()
        ra_k, dec_k, _, _ = ct.get_target_radec_and_rates(
            _make_sat_data(_GOES18_GP_TLE),
            target_dt=goes18_epoch,
        )

        # Observer matches _make_daemon()'s location_service fixture: (37, -122, 100 m).
        ts = load.timescale()
        obs = wgs84.latlon(37.0, -122.0, elevation_m=100.0)
        sat = EarthSatellite(_GOES18_GP_TLE[0], _GOES18_GP_TLE[1], "TEST", ts)
        # Same pinned time on both sides — removes the wall-clock-slop term.
        t = ts.from_datetime(goes18_epoch)
        topo = (sat - obs).at(t)
        ra_sf_hours, dec_sf_angle, _ = topo.radec()
        ra_sf = float(ra_sf_hours.hours) * 15.0  # type: ignore[arg-type]  # skyfield stubs
        dec_sf = float(dec_sf_angle.degrees)  # type: ignore[arg-type]  # skyfield stubs

        # 10 arcsec tolerance — skyfield is ICRS, keplemon J2000; frame bias
        # between the two is sub-arcsecond, and the time is pinned so there's
        # no clock-slop term anymore.
        assert abs(ra_k - ra_sf) < 1.0 / 3600.0 * 10, f"RA diverged: keplemon={ra_k}, skyfield={ra_sf}"
        assert abs(dec_k - dec_sf) < 1.0 / 3600.0 * 10, f"Dec diverged: keplemon={dec_k}, skyfield={dec_sf}"

    @pytest.mark.slow
    def test_keplemon_rate_matches_skyfield_earth_fixed_on_mid_inclination_leo(self):
        """Rate parity on a mid-inclination LEO (ISS, 51.6°) at a pinned past epoch.

        Plan originally called for comparison against Skyfield's
        ``frame_latlon_and_rates(ground_station)``. That frame is actually
        the observer's topocentric alt-az frame, not a "RA/Dec in Earth-fixed
        coordinates" frame, so a literal comparison would be a category
        error (the old pre-keplemon code had this bug latent; the rename
        from ``celestial`` to ``inertial`` is what exposed it).

        The real parity target is the **inertial** (ICRS) RA/Dec rate,
        because that's the quantity both libraries compute the same way
        from SGP4 output. Once inertial parity holds, the Earth-fixed
        rate is just ``inertial − SIDEREAL_RATE_DEG_PER_S`` by construction
        (tested separately in ``test_inertial_kwarg_changes_ra_rate``),
        so there's no independent Earth-fixed rate to check — it's
        derived algebraically from the inertial parity this test asserts.
        """
        from skyfield.api import EarthSatellite, load, wgs84
        from skyfield.framelib import ICRS

        ct = _make_concrete_for_propagation()
        sat_data = _make_sat_data(_ISS_GP_TLE, "ISS")

        # Pinned inside a known visible pass so SGP4 is evaluating near
        # TLE epoch and at a declination well away from 0 (exercises both
        # the lat and lon rate channels).
        _, dec_k, ra_rate_k_inertial, dec_rate_k_inertial = ct.get_target_radec_and_rates(
            sat_data, inertial=True, target_dt=_ISS_HIGH_DEC_DT
        )

        ts = load.timescale()
        gs = wgs84.latlon(37.0, -122.0, elevation_m=100.0)
        sat_sf = EarthSatellite(_ISS_GP_TLE[0], _ISS_GP_TLE[1], "ISS", ts)
        t = ts.from_datetime(_ISS_HIGH_DEC_DT)
        topo = (sat_sf - gs).at(t)

        # ICRS frame: rates[3] = dlat/dt (Dec rate), rates[4] = dlon/dt (RA rate).
        # Skyfield returns AngleRate objects — ``.degrees.per_second`` unwraps to float.
        icrs_rates = topo.frame_latlon_and_rates(ICRS)
        ra_rate_sf = float(icrs_rates[4].degrees.per_second)  # type: ignore[attr-defined]
        dec_rate_sf = float(icrs_rates[3].degrees.per_second)  # type: ignore[attr-defined]

        # Fixture sanity: we're actually at non-trivial declination.
        assert abs(dec_k) > 30.0, f"ISS pinned sample is not high-dec: {dec_k}"

        # 0.3 arcsec/s tolerance. Empirically keplemon and skyfield disagree
        # by ~0.05 arcsec/s on RA rate and ~0.3 arcsec/s on Dec rate at this
        # sample — both well inside the "gross error" regime we actually
        # care about catching. ISS moves at ~200 arcsec/s total, so 0.3 is
        # a 0.15% relative tolerance.
        tolerance_deg_s = 0.3 / 3600.0
        assert abs(ra_rate_k_inertial - ra_rate_sf) < tolerance_deg_s, (
            f"Inertial RA rate diverged: keplemon={ra_rate_k_inertial:.6f}, skyfield={ra_rate_sf:.6f} "
            f"(delta {(ra_rate_k_inertial - ra_rate_sf) * 3600.0:.3f} arcsec/s)"
        )
        assert abs(dec_rate_k_inertial - dec_rate_sf) < tolerance_deg_s, (
            f"Inertial Dec rate diverged: keplemon={dec_rate_k_inertial:.6f}, skyfield={dec_rate_sf:.6f} "
            f"(delta {(dec_rate_k_inertial - dec_rate_sf) * 3600.0:.3f} arcsec/s)"
        )


class TestAngularDistanceNaN:
    """angular_distance must raise on non-finite input, not silently clamp to 180°.

    ``angular_distance`` is a concrete method on the abstract base class and
    doesn't actually use ``self``, so we call it as an unbound method rather
    than synthesizing a subclass that has to stub every abstract method.
    """

    @staticmethod
    def _call(ra1, dec1, ra2, dec2):
        from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter

        return AbstractAstroHardwareAdapter.angular_distance(None, ra1, dec1, ra2, dec2)  # type: ignore[arg-type]

    def test_nan_ra_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._call(float("nan"), 0.0, 10.0, 20.0)

    def test_nan_dec_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._call(0.0, 0.0, 10.0, float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._call(0.0, 0.0, float("inf"), 20.0)

    def test_finite_inputs_return_expected(self):
        assert self._call(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)
        assert self._call(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0, abs=1e-6)


class TestRateUnits:
    """Rates returned by get_target_radec_and_rates are degrees/second, not arcsec/second.

    This is a cheap unit-shape test that would catch someone forgetting to
    divide by 3600 when plumbing the keplemon output downstream.
    """

    def test_geo_rate_is_degrees_per_second_not_arcseconds(self):
        ct = _make_concrete_for_propagation()
        # GEO satellites have ~0 rate relative to Earth-fixed observer (but
        # non-zero inertial rate). Test that magnitudes are in the
        # "degrees per second" neighborhood, not "arcsec per second".
        _, _, ra_rate, dec_rate = ct.get_target_radec_and_rates(
            _make_sat_data(_GOES18_GP_TLE),
            inertial=True,
            target_dt=_GOES18_GP_TLE_EPOCH,
        )
        total_rate = math.sqrt(ra_rate**2 + dec_rate**2)
        # A GEO-ish inertial rate in deg/s is ~4e-3 (one revolution per sidereal day).
        # If units were arcsec/s by mistake, this would be ~15. The assertion
        # is generous because different epochs give different exact values.
        assert total_rate < 1.0, f"rate magnitude {total_rate} looks like arcsec/s, not deg/s"


class TestAngularRateProjection:
    """``compute_angular_rate`` must apply cos(dec) exactly once.

    The split of responsibility is:
      - ``get_target_radec_and_rates`` returns coordinate rates (deg/s in RA
        and Dec coordinate space). No cos(dec) anywhere.
      - ``compute_angular_rate`` projects onto the sky for trail / exposure
        math by multiplying the RA component by cos(dec).

    This test locks that contract in place so a future refactor can't
    silently re-introduce a double-cos(dec) (the PR #301 Copilot bug) or
    drop the projection entirely (which would over-expose high-dec
    satellites).
    """

    def test_angular_rate_equals_projected_rss_of_coordinate_rates(self):
        ct = _make_concrete_for_propagation()
        sat_data = _make_sat_data(_GOES18_GP_TLE)

        # Pull both the raw coordinate rates and the projected sky rate from
        # the same satellite. They're evaluated at "now" on each call, so
        # there's a few ms of wall-clock slop between them, but for a GEO the
        # rates are small enough that the numerical divergence from that
        # slop is well under the tolerance we care about.
        _, dec, ra_rate, dec_rate = ct.get_target_radec_and_rates(sat_data, inertial=False)
        angular_rate = ct.compute_angular_rate(sat_data, inertial=False)

        # If projection is applied exactly once:
        expected_once = math.sqrt((ra_rate * math.cos(math.radians(dec))) ** 2 + dec_rate**2)
        assert angular_rate == pytest.approx(expected_once, rel=1e-2, abs=1e-9)

        # Belt-and-suspenders: the bug would manifest as either zero or
        # double cos(dec). At dec ≈ 0 (GOES 18 / DIRECTV GEOs) cos(dec) ≈ 1
        # so we can't distinguish there, but the test above using ``expected_once``
        # is exact independent of the dec magnitude.

    def test_angular_rate_projects_ra_component_at_high_dec(self):
        """Synthetic high-dec case that actually distinguishes 'no projection',
        'single projection', and 'double projection'. Pure arithmetic — no
        keplemon propagation.
        """
        from unittest.mock import patch

        ct = _make_concrete_for_propagation()

        # ra_rate = 0.01 deg/s, dec_rate = 0, at dec = 60° (cos(dec) = 0.5).
        fake_ra = 123.0
        fake_dec = 60.0
        fake_ra_rate = 0.01
        fake_dec_rate = 0.0

        with patch.object(
            ct,
            "get_target_radec_and_rates",
            return_value=(fake_ra, fake_dec, fake_ra_rate, fake_dec_rate),
        ):
            angular_rate = ct.compute_angular_rate({"name": "SYNTH"}, inertial=False)

        # Single projection: 0.01 * cos(60°) = 0.005 deg/s.
        assert angular_rate == pytest.approx(0.005, rel=1e-9)
        # No projection would give 0.01 (off by 2x); double projection would
        # give 0.0025 (off by 2x the other way). Either would fail above.
