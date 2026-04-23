"""Tests for TelescopeSensor — the phase-1 wrapper around AbstractAstroHardwareAdapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import pytest

from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrasense.safety.safety_monitor import SafetyMonitor
from citrasense.sensors.abstract_sensor import (
    AcquisitionContext,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor
from citrasense.settings.citrasense_settings import SensorConfig


def _make_mock_adapter():
    adapter = create_autospec(AbstractAstroHardwareAdapter, instance=True)
    adapter.connect.return_value = True
    adapter.is_telescope_connected.return_value = True
    adapter.is_camera_connected.return_value = True
    adapter.get_settings_schema.return_value = [{"name": "host", "type": "string", "friendly_name": "Host"}]
    return adapter


class TestTelescopeSensorLifecycle:
    def test_connect_forwards(self):
        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        assert sensor.connect() is True
        adapter.connect.assert_called_once()

    def test_disconnect_forwards(self):
        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        sensor.disconnect()
        adapter.disconnect.assert_called_once()

    def test_is_connected_forwards(self):
        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        assert sensor.is_connected() is True
        adapter.is_telescope_connected.assert_called()
        adapter.is_camera_connected.assert_called()

    def test_is_connected_false_when_camera_down(self):
        adapter = _make_mock_adapter()
        adapter.is_camera_connected.return_value = False
        sensor = TelescopeSensor("ts-1", adapter)
        assert sensor.is_connected() is False


class TestTelescopeSensorCapabilities:
    def test_sensor_type(self):
        assert TelescopeSensor.sensor_type == "telescope"

    def test_capabilities(self):
        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        caps = sensor.get_capabilities()
        assert isinstance(caps, SensorCapabilities)
        assert caps.acquisition_mode == SensorAcquisitionMode.ON_DEMAND
        assert "optical" in caps.modalities


class TestTelescopeSensorSchema:
    def test_get_settings_schema_forwards(self, tmp_path: Path):
        from logging import getLogger

        cfg = SensorConfig(
            id="ts-schema",
            type="telescope",
            adapter="dummy",
            adapter_settings={},
            citra_sensor_id="",
        )
        sensor = TelescopeSensor.from_config(cfg, logger=getLogger("test"), images_dir=tmp_path)
        schema = sensor.get_settings_schema()
        assert isinstance(schema, list)
        assert len(schema) > 0
        assert "name" in schema[0]


class TestTelescopeSensorAcquisitionVerbs:
    def test_acquire_raises_not_implemented(self):
        from unittest.mock import MagicMock

        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        with pytest.raises(NotImplementedError, match="phase 4"):
            sensor.acquire(MagicMock(), AcquisitionContext())

    def test_start_stream_raises_not_implemented(self):
        from unittest.mock import MagicMock

        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        with pytest.raises(NotImplementedError):
            sensor.start_stream(MagicMock(), AcquisitionContext())

    def test_stop_stream_raises_not_implemented(self):
        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("ts-1", adapter)
        with pytest.raises(NotImplementedError):
            sensor.stop_stream()


class TestTelescopeSensorFromConfig:
    def test_from_config_builds_adapter(self, tmp_path: Path):
        from logging import getLogger

        cfg = SensorConfig(
            id="ts-cfg",
            type="telescope",
            adapter="dummy",
            adapter_settings={},
            citra_sensor_id="tel-99",
        )
        sensor = TelescopeSensor.from_config(cfg, logger=getLogger("test"), images_dir=tmp_path)
        assert sensor.sensor_id == "ts-cfg"
        assert sensor.adapter_key == "dummy"
        assert sensor.adapter is not None

    def test_from_config_passes_adapter_settings(self, tmp_path: Path):
        from logging import getLogger

        cfg = SensorConfig(
            id="ts-cfg2",
            type="telescope",
            adapter="dummy",
            adapter_settings={"simulate_slow_operations": True},
            citra_sensor_id="tel-99",
        )
        sensor = TelescopeSensor.from_config(cfg, logger=getLogger("test"), images_dir=tmp_path)
        assert sensor.adapter is not None


class TestTelescopeSensorSafetyChecks:
    """Tests for register_safety_checks / unregister_safety_checks."""

    def _make_sensor_with_mount(self):
        adapter = _make_mock_adapter()
        mount = MagicMock()
        type(adapter).mount = PropertyMock(return_value=mount)
        return TelescopeSensor("telescope-0", adapter), mount

    @patch(
        "citrasense.sensors.telescope.safety.cable_wrap_check.CableWrapCheck",
    )
    def test_register_creates_and_registers(self, MockCableWrap, tmp_path):
        sensor, mount = self._make_sensor_with_mount()
        mock_check = MockCableWrap.return_value
        mock_check.needs_startup_unwind.return_value = False
        mock_check.name = "cable_wrap"

        monitor = SafetyMonitor(MagicMock(), [])
        sensor.register_safety_checks(monitor, logger=MagicMock(), state_file=tmp_path / "state.json")

        MockCableWrap.assert_called_once()
        mock_check.start.assert_called_once()
        mount.register_sync_listener.assert_called_once_with(mock_check.notify_sync)
        assert monitor.get_sensor_checks("telescope-0") == [mock_check]
        assert mock_check.safety_gate is not None

    @patch(
        "citrasense.sensors.telescope.safety.cable_wrap_check.CableWrapCheck",
    )
    def test_register_handles_startup_unwind(self, MockCableWrap, tmp_path):
        sensor, _mount = self._make_sensor_with_mount()
        mock_check = MockCableWrap.return_value
        mock_check.needs_startup_unwind.return_value = True
        mock_check.did_last_unwind_fail.return_value = False
        mock_check.cumulative_deg = 400.0
        mock_check.name = "cable_wrap"

        monitor = SafetyMonitor(MagicMock(), [])
        sensor.register_safety_checks(monitor, logger=MagicMock(), state_file=tmp_path / "state.json")

        mock_check.execute_action.assert_called_once()

    @patch(
        "citrasense.sensors.telescope.safety.cable_wrap_check.CableWrapCheck",
    )
    def test_register_marks_intervention_on_failed_unwind(self, MockCableWrap, tmp_path):
        sensor, _mount = self._make_sensor_with_mount()
        mock_check = MockCableWrap.return_value
        mock_check.needs_startup_unwind.return_value = True
        mock_check.did_last_unwind_fail.return_value = True
        mock_check.cumulative_deg = 400.0
        mock_check.name = "cable_wrap"

        monitor = SafetyMonitor(MagicMock(), [])
        sensor.register_safety_checks(monitor, logger=MagicMock(), state_file=tmp_path / "state.json")

        mock_check.mark_intervention_required.assert_called_once()

    @patch(
        "citrasense.sensors.telescope.safety.cable_wrap_check.CableWrapCheck",
    )
    def test_unregister_stops_and_removes(self, MockCableWrap, tmp_path):
        sensor, _mount = self._make_sensor_with_mount()
        mock_check = MockCableWrap.return_value
        mock_check.needs_startup_unwind.return_value = False
        mock_check.name = "cable_wrap"

        monitor = SafetyMonitor(MagicMock(), [])
        sensor.register_safety_checks(monitor, logger=MagicMock(), state_file=tmp_path / "state.json")

        sensor.unregister_safety_checks(monitor)
        mock_check.join_unwind.assert_called_once_with(timeout=10.0)
        mock_check.stop.assert_called_once()
        assert monitor.get_sensor_checks("telescope-0") == []
        assert sensor._cable_wrap_check is None

    def test_register_skips_when_no_mount(self, tmp_path):
        adapter = _make_mock_adapter()
        type(adapter).mount = PropertyMock(return_value=None)
        sensor = TelescopeSensor("telescope-0", adapter)

        monitor = SafetyMonitor(MagicMock(), [])
        sensor.register_safety_checks(monitor, logger=MagicMock(), state_file=tmp_path / "state.json")

        assert monitor.get_sensor_checks("telescope-0") == []
        assert sensor._cable_wrap_check is None

    def test_unregister_noop_when_no_check(self):
        adapter = _make_mock_adapter()
        sensor = TelescopeSensor("telescope-0", adapter)
        monitor = SafetyMonitor(MagicMock(), [])
        sensor.unregister_safety_checks(monitor)
