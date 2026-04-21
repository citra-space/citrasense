"""Unit tests for AbstractAstroHardwareAdapter concrete methods."""

from unittest.mock import MagicMock

import pytest

from citrasense.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
)


class StubAdapter(AbstractAstroHardwareAdapter):
    """Minimal concrete adapter for testing inherited behaviour."""

    logger = MagicMock()

    @classmethod
    def get_settings_schema(cls, **kwargs):
        return []

    def _do_point_telescope(self, ra, dec):
        pass

    def get_observation_strategy(self):
        return ObservationStrategy.MANUAL

    def perform_observation_sequence(self, task, satellite_data):
        return []

    def connect(self):
        return True

    def disconnect(self):
        pass

    def is_telescope_connected(self):
        return True

    def is_camera_connected(self):
        return True

    def list_devices(self):
        return []

    def select_telescope(self, device_name):
        return True

    def get_telescope_direction(self):
        return (0.0, 0.0)

    def telescope_is_moving(self):
        return False

    def select_camera(self, device_name):
        return True

    def take_image(self, task_id, exposure_duration_seconds=1.0):
        return "/tmp/img.fits"

    def set_custom_tracking_rate(self, ra_rate, dec_rate):
        pass

    def get_tracking_rate(self):
        return (15.0, 0.0)


@pytest.fixture
def adapter(tmp_path):
    return StubAdapter(images_dir=tmp_path)


# ---------------------------------------------------------------------------
# Init and filter map loading
# ---------------------------------------------------------------------------


def test_init_empty_filters(tmp_path):
    a = StubAdapter(images_dir=tmp_path)
    assert a.filter_map == {}


def test_init_with_filters(tmp_path):
    a = StubAdapter(
        images_dir=tmp_path,
        filters={
            "0": {"name": "Lum", "focus_position": 9000, "enabled": True},
            "1": {"name": "Red", "focus_position": 9100},
        },
    )
    assert 0 in a.filter_map
    assert a.filter_map[0]["name"] == "Lum"
    assert a.filter_map[1]["enabled"] is True


def test_init_bad_filter_id(tmp_path):
    a = StubAdapter(images_dir=tmp_path, filters={"not_int": {"name": "X", "focus_position": 0}})
    assert len(a.filter_map) == 0


# ---------------------------------------------------------------------------
# angular_distance
# ---------------------------------------------------------------------------


def test_angular_distance_same_point(adapter):
    assert adapter.angular_distance(180.0, 45.0, 180.0, 45.0) == pytest.approx(0.0, abs=1e-9)


def test_angular_distance_poles(adapter):
    assert adapter.angular_distance(0.0, 90.0, 0.0, -90.0) == pytest.approx(180.0, abs=1e-6)


def test_angular_distance_known(adapter):
    d = adapter.angular_distance(0.0, 0.0, 90.0, 0.0)
    assert d == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Default method implementations
# ---------------------------------------------------------------------------


def test_supports_autofocus_default(adapter):
    assert adapter.supports_autofocus() is False


def test_supports_filter_management_default(adapter):
    assert adapter.supports_filter_management() is False


def test_supports_direct_camera_control_default(adapter):
    assert adapter.supports_direct_camera_control() is False


def test_is_hyperspectral_default(adapter):
    assert adapter.is_hyperspectral() is False


def test_do_autofocus_raises(adapter):
    with pytest.raises(NotImplementedError):
        adapter.do_autofocus()


def test_expose_camera_raises(adapter):
    with pytest.raises(NotImplementedError):
        adapter.expose_camera(1.0)


def test_get_missing_dependencies_empty(adapter):
    assert adapter.get_missing_dependencies() == []


def test_update_from_plate_solve_noop(adapter):
    adapter.update_from_plate_solve(180.0, 45.0)


def test_point_telescope_delegates(adapter):
    adapter.point_telescope(100.0, 50.0)


# ---------------------------------------------------------------------------
# Filter management
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter_with_filters(tmp_path):
    return StubAdapter(
        images_dir=tmp_path,
        filters={
            "0": {"name": "Luminance", "focus_position": 9000, "enabled": True},
            "1": {"name": "Red", "focus_position": 9050, "enabled": True},
            "2": {"name": "Ha", "focus_position": 9100, "enabled": False},
        },
    )


def test_get_filter_config(adapter_with_filters):
    cfg = adapter_with_filters.get_filter_config()
    assert "0" in cfg
    assert cfg["0"]["name"] == "Luminance"
    assert cfg["2"]["enabled"] is False


def test_update_filter_focus(adapter_with_filters):
    assert adapter_with_filters.update_filter_focus("1", 9999) is True
    assert adapter_with_filters.filter_map[1]["focus_position"] == 9999


def test_update_filter_focus_bad_id(adapter_with_filters):
    assert adapter_with_filters.update_filter_focus("999", 1000) is False
    assert adapter_with_filters.update_filter_focus("bad", 1000) is False


def test_update_filter_enabled(adapter_with_filters):
    assert adapter_with_filters.update_filter_enabled("2", True) is True
    assert adapter_with_filters.filter_map[2]["enabled"] is True


def test_update_filter_enabled_bad_id(adapter_with_filters):
    assert adapter_with_filters.update_filter_enabled("nope", False) is False


# ---------------------------------------------------------------------------
# select_filters_for_task
# ---------------------------------------------------------------------------


def test_select_filters_assigned(adapter_with_filters):
    task = MagicMock()
    task.assigned_filter_name = "Red"
    task.id = "t1"
    result = adapter_with_filters.select_filters_for_task(task)
    assert 1 in result
    assert result[1]["name"] == "Red"


def test_select_filters_assigned_disabled(adapter_with_filters):
    task = MagicMock()
    task.assigned_filter_name = "Ha"
    task.id = "t1"
    with pytest.raises(RuntimeError, match="disabled"):
        adapter_with_filters.select_filters_for_task(task)


def test_select_filters_assigned_not_found(adapter_with_filters):
    task = MagicMock()
    task.assigned_filter_name = "NonExistent"
    task.id = "t1"
    with pytest.raises(RuntimeError, match="not found"):
        adapter_with_filters.select_filters_for_task(task)


def test_select_filters_default_luminance(adapter_with_filters):
    task = MagicMock()
    task.assigned_filter_name = None
    task.id = "t1"
    result = adapter_with_filters.select_filters_for_task(task)
    assert 0 in result
    assert result[0]["name"] == "Luminance"


def test_select_filters_no_clear_uses_first(tmp_path):
    a = StubAdapter(
        images_dir=tmp_path,
        filters={"5": {"name": "Narrowband", "focus_position": 5000, "enabled": True}},
    )
    task = MagicMock()
    task.assigned_filter_name = None
    task.id = "t1"
    result = a.select_filters_for_task(task)
    assert 5 in result


def test_select_filters_allow_no_filter(tmp_path):
    a = StubAdapter(images_dir=tmp_path)
    task = MagicMock()
    task.assigned_filter_name = None
    task.id = "t1"
    assert a.select_filters_for_task(task, allow_no_filter=True) is None


def test_select_filters_no_filter_raises(tmp_path):
    a = StubAdapter(images_dir=tmp_path)
    task = MagicMock()
    task.assigned_filter_name = None
    task.id = "t1"
    with pytest.raises(RuntimeError, match="No enabled filters"):
        a.select_filters_for_task(task, allow_no_filter=False)
