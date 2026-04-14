"""Tests for SiderealTelescopeTask.tracking_mode and sequence_provides_tracking."""

from unittest.mock import MagicMock

from citrascope.hardware.abstract_astro_hardware_adapter import ObservationStrategy
from citrascope.tasks.scope.sidereal_telescope_task import SiderealTelescopeTask
from citrascope.tasks.scope.tracking_telescope_task import TrackingTelescopeTask
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


def _make_sidereal_task(
    observation_strategy: ObservationStrategy = ObservationStrategy.MANUAL,
    sequence_provides_tracking: bool = False,
) -> SiderealTelescopeTask:
    adapter = MagicMock()
    adapter.get_observation_strategy.return_value = observation_strategy
    adapter.sequence_provides_tracking = sequence_provides_tracking
    adapter.filter_map = {}

    task_obj = _make_task_dict()
    return SiderealTelescopeTask(
        MagicMock(),
        adapter,
        MagicMock(),
        task_obj,
        settings=MagicMock(),
        task_manager=MagicMock(),
        location_service=MagicMock(),
        telescope_record={"id": "tel-1"},
        ground_station={"id": "gs-1"},
        elset_cache=None,
        processor_registry=None,
    )


class TestSiderealTelescopeTaskTrackingMode:
    def test_manual_strategy_returns_sidereal(self):
        task = _make_sidereal_task(ObservationStrategy.MANUAL, sequence_provides_tracking=False)
        assert task.tracking_mode == "sidereal"

    def test_manual_strategy_returns_sidereal_even_if_sequence_provides_tracking(self):
        task = _make_sidereal_task(ObservationStrategy.MANUAL, sequence_provides_tracking=True)
        assert task.tracking_mode == "sidereal"

    def test_sequence_to_controller_without_tracking_returns_sidereal(self):
        task = _make_sidereal_task(ObservationStrategy.SEQUENCE_TO_CONTROLLER, sequence_provides_tracking=False)
        assert task.tracking_mode == "sidereal"

    def test_sequence_to_controller_with_tracking_returns_rate(self):
        task = _make_sidereal_task(ObservationStrategy.SEQUENCE_TO_CONTROLLER, sequence_provides_tracking=True)
        assert task.tracking_mode == "rate"


class TestTrackingTelescopeTaskTrackingMode:
    def test_always_returns_rate(self):
        adapter = MagicMock()
        adapter.filter_map = {}
        task_obj = _make_task_dict()
        task = TrackingTelescopeTask(
            MagicMock(),
            adapter,
            MagicMock(),
            task_obj,
            settings=MagicMock(),
            task_manager=MagicMock(),
            location_service=MagicMock(),
            telescope_record={"id": "tel-1"},
            ground_station={"id": "gs-1"},
            elset_cache=None,
            processor_registry=None,
        )
        assert task.tracking_mode == "rate"


class TestNinaSequenceProvidesTracking:
    def test_nina_adapter_reports_sequence_provides_tracking(self):
        from citrascope.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter

        assert NinaAdvancedHttpAdapter.sequence_provides_tracking.fget(None) is True
