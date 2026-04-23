"""Unit tests for the Task dataclass."""

from citrasense.tasks.task import Task


def _sample_dict():
    return {
        "id": "abc-123",
        "type": "tracking",
        "status": "scheduled",
        "creationEpoch": "1700000000",
        "updateEpoch": "1700000001",
        "taskStart": "1700000100",
        "taskStop": "1700000200",
        "userId": "user-1",
        "username": "operator",
        "satelliteId": "sat-1",
        "satelliteName": "ISS",
        "telescopeId": "tel-1",
        "telescopeName": "CDK14",
        "groundStationId": "gs-1",
        "groundStationName": "Desert",
        "assignedFilterName": "Red",
    }


def test_from_dict():
    t = Task.from_dict(_sample_dict())
    assert t.id == "abc-123"
    assert t.type == "tracking"
    assert t.satelliteName == "ISS"
    assert t.assigned_filter_name == "Red"


def test_from_dict_missing_fields():
    t = Task.from_dict({})
    assert t.id == ""
    assert t.status == ""
    assert t.assigned_filter_name is None


def test_from_dict_telescope_inference():
    """telescopeId present, no sensorType -> telescope."""
    t = Task.from_dict(_sample_dict())
    assert t.sensor_type == "telescope"
    assert t.sensor_id == "tel-1"


def test_from_dict_explicit_sensor_type():
    """Explicit sensorType/sensorId override inference."""
    data = {**_sample_dict(), "sensorType": "rf", "sensorId": "ant-5"}
    t = Task.from_dict(data)
    assert t.sensor_type == "rf"
    assert t.sensor_id == "ant-5"


def test_from_dict_antenna_inference():
    """antennaId present, no sensorType -> rf."""
    data = {
        "id": "task-rf-1",
        "type": "TDOA",
        "status": "Pending",
        "antennaId": "ant-3",
    }
    t = Task.from_dict(data)
    assert t.sensor_type == "rf"
    assert t.sensor_id == "ant-3"


def test_from_dict_empty_defaults_to_telescope():
    """No telescopeId or antennaId -> defaults to telescope with empty sensor_id."""
    t = Task.from_dict({})
    assert t.sensor_type == "telescope"
    assert t.sensor_id == ""


def test_set_get_status_msg():
    t = Task.from_dict(_sample_dict())
    t.set_status_msg("Processing...")
    assert t.get_status_msg() == "Processing..."
    t.set_status_msg(None)
    assert t.get_status_msg() is None


def test_set_get_retry_time():
    t = Task.from_dict(_sample_dict())
    t.set_retry_time(1700000500.0)
    assert t.get_retry_time() == 1700000500.0
    t.set_retry_time(None)
    assert t.get_retry_time() is None


def test_set_get_executing():
    t = Task.from_dict(_sample_dict())
    assert t.get_executing() is False
    t.set_executing(True)
    assert t.get_executing() is True


def test_get_status_info():
    t = Task.from_dict(_sample_dict())
    t.set_status_msg("hello")
    t.set_retry_time(123.0)
    t.set_executing(True)
    msg, retry, executing = t.get_status_info()
    assert msg == "hello"
    assert retry == 123.0
    assert executing is True


def test_repr():
    t = Task.from_dict(_sample_dict())
    r = repr(t)
    assert "abc-123" in r
    assert "tracking" in r
    assert "[telescope]" in r
