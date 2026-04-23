"""Unit tests for TelescopeTaskView."""

import pytest

from citrasense.tasks.task import Task
from citrasense.tasks.views.telescope_task_view import TelescopeTaskView


def _telescope_task(**overrides) -> Task:
    defaults = {
        "id": "task-abc-123",
        "type": "Track",
        "status": "Scheduled",
        "creationEpoch": "2025-03-01T00:00:00Z",
        "updateEpoch": "2025-03-01T00:00:00Z",
        "taskStart": "2025-03-01T01:00:00Z",
        "taskStop": "2025-03-01T01:05:00Z",
        "userId": "user-1",
        "username": "observer",
        "satelliteId": "sat-42",
        "satelliteName": "STARLINK-1234",
        "telescopeId": "tel-1",
        "telescopeName": "CDK14",
        "groundStationId": "gs-1",
        "groundStationName": "Desert",
        "sensor_type": "telescope",
        "sensor_id": "tel-1",
        "assigned_filter_name": "Clear",
    }
    defaults.update(overrides)
    return Task(**defaults)


class TestTelescopeTaskView:
    def test_properties(self):
        task = _telescope_task()
        tv = TelescopeTaskView(task)
        assert tv.satellite_id == "sat-42"
        assert tv.satellite_name == "STARLINK-1234"
        assert tv.telescope_id == "tel-1"
        assert tv.telescope_name == "CDK14"
        assert tv.ground_station_id == "gs-1"
        assert tv.ground_station_name == "Desert"
        assert tv.assigned_filter_name == "Clear"

    def test_generic_proxies(self):
        task = _telescope_task()
        tv = TelescopeTaskView(task)
        assert tv.id == "task-abc-123"
        assert tv.type == "Track"
        assert tv.status == "Scheduled"
        assert tv.task is task

    def test_status_msg_proxy(self):
        task = _telescope_task()
        tv = TelescopeTaskView(task)
        tv.set_status_msg("Running plate solver...")
        assert tv.get_status_msg() == "Running plate solver..."
        assert task.get_status_msg() == "Running plate solver..."

    def test_rejects_non_telescope(self):
        task = _telescope_task(sensor_type="rf")
        with pytest.raises(ValueError, match="rf"):
            TelescopeTaskView(task)

    def test_repr(self):
        task = _telescope_task()
        tv = TelescopeTaskView(task)
        r = repr(tv)
        assert "task-abc" in r
        assert "STARLINK-1234" in r
