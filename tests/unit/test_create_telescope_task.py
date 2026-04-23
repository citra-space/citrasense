"""Tests for SensorRuntime._create_telescope_task observation mode selection."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from citrasense.sensors.sensor_runtime import SensorRuntime
from citrasense.tasks.scope.sidereal_telescope_task import SiderealTelescopeTask
from citrasense.tasks.scope.tracking_telescope_task import TrackingTelescopeTask


def _make_runtime(observation_mode: str, supports_custom_tracking: bool) -> SensorRuntime:
    """Build a SensorRuntime with just enough mocks to call _create_telescope_task."""
    settings = MagicMock()
    settings.observation_mode = observation_mode

    adapter = MagicMock()
    type(adapter).supports_custom_tracking = PropertyMock(return_value=supports_custom_tracking)

    rt = SensorRuntime.__new__(SensorRuntime)
    rt.api_client = MagicMock()
    rt.logger = MagicMock()
    rt.hardware_adapter = adapter
    rt.settings = settings
    rt.sensor_type = "telescope"
    rt.processor_registry = MagicMock()
    rt.location_service = MagicMock()
    rt.telescope_record = {"id": "tel-1"}
    rt.ground_station = {"id": "gs-1"}
    rt.elset_cache = None
    rt.apass_catalog = None
    rt._on_annotated_image = None
    rt.task_index = None
    return rt


def _telescope_mock_task():
    task = MagicMock()
    task.sensor_type = "telescope"
    return task


class TestCreateTelescopeTask:
    def test_sidereal_mode_always_returns_sidereal(self):
        rt = _make_runtime("sidereal", supports_custom_tracking=True)
        result = rt._create_telescope_task(_telescope_mock_task())
        assert isinstance(result, SiderealTelescopeTask)

    def test_tracking_mode_always_returns_tracking(self):
        rt = _make_runtime("tracking", supports_custom_tracking=False)
        result = rt._create_telescope_task(_telescope_mock_task())
        assert isinstance(result, TrackingTelescopeTask)

    def test_auto_mode_returns_tracking_when_supported(self):
        rt = _make_runtime("auto", supports_custom_tracking=True)
        result = rt._create_telescope_task(_telescope_mock_task())
        assert isinstance(result, TrackingTelescopeTask)

    def test_auto_mode_returns_sidereal_when_unsupported(self):
        rt = _make_runtime("auto", supports_custom_tracking=False)
        result = rt._create_telescope_task(_telescope_mock_task())
        assert isinstance(result, SiderealTelescopeTask)

    @pytest.mark.parametrize("mode", ["auto", "sidereal", "tracking"])
    def test_all_modes_log_selection(self, mode):
        rt = _make_runtime(mode, supports_custom_tracking=True)
        rt._create_telescope_task(_telescope_mock_task())
        rt.logger.info.assert_called()
        logged = rt.logger.info.call_args[0][0]
        assert "TelescopeTask" in logged
