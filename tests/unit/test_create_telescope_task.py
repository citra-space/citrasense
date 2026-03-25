"""Tests for TaskManager._create_telescope_task observation mode selection."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from citrascope.tasks.runner import TaskManager
from citrascope.tasks.scope.static_telescope_task import StaticTelescopeTask
from citrascope.tasks.scope.tracking_telescope_task import TrackingTelescopeTask


def _make_manager(observation_mode: str, supports_custom_tracking: bool) -> TaskManager:
    """Build a TaskManager with just enough mocks to call _create_telescope_task."""
    settings = MagicMock()
    settings.observation_mode = observation_mode

    adapter = MagicMock()
    type(adapter).supports_custom_tracking = PropertyMock(return_value=supports_custom_tracking)

    mgr = TaskManager.__new__(TaskManager)
    mgr.api_client = MagicMock()
    mgr.logger = MagicMock()
    mgr.hardware_adapter = adapter
    mgr.settings = settings
    mgr.processor_registry = MagicMock()
    mgr.location_service = MagicMock()
    mgr.telescope_record = {"id": "tel-1"}
    mgr.ground_station = {"id": "gs-1"}
    mgr.elset_cache = None
    mgr.apass_catalog = None
    mgr._on_annotated_image = None
    return mgr


class TestCreateTelescopeTask:
    def test_static_mode_always_returns_static(self):
        mgr = _make_manager("static", supports_custom_tracking=True)
        task = MagicMock()
        result = mgr._create_telescope_task(task)
        assert isinstance(result, StaticTelescopeTask)

    def test_tracking_mode_always_returns_tracking(self):
        mgr = _make_manager("tracking", supports_custom_tracking=False)
        task = MagicMock()
        result = mgr._create_telescope_task(task)
        assert isinstance(result, TrackingTelescopeTask)

    def test_auto_mode_returns_tracking_when_supported(self):
        mgr = _make_manager("auto", supports_custom_tracking=True)
        task = MagicMock()
        result = mgr._create_telescope_task(task)
        assert isinstance(result, TrackingTelescopeTask)

    def test_auto_mode_returns_static_when_unsupported(self):
        mgr = _make_manager("auto", supports_custom_tracking=False)
        task = MagicMock()
        result = mgr._create_telescope_task(task)
        assert isinstance(result, StaticTelescopeTask)

    @pytest.mark.parametrize("mode", ["auto", "static", "tracking"])
    def test_all_modes_log_selection(self, mode):
        mgr = _make_manager(mode, supports_custom_tracking=True)
        task = MagicMock()
        mgr._create_telescope_task(task)
        mgr.logger.info.assert_called()
        logged = mgr.logger.info.call_args[0][0]
        assert "TelescopeTask" in logged
