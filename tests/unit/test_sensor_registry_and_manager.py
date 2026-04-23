"""Tests for the sensor registry and sensor manager."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.sensor_manager import SensorManager
from citrasense.sensors.sensor_registry import (
    REGISTERED_SENSORS,
    get_sensor_class,
    list_sensors,
)
from citrasense.settings.citrasense_settings import SensorConfig

_logger = getLogger("test")


class TestSensorRegistry:
    def test_telescope_registered(self):
        assert "telescope" in REGISTERED_SENSORS

    def test_get_sensor_class_telescope(self):
        cls = get_sensor_class("telescope")
        from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor

        assert cls is TelescopeSensor

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown sensor type"):
            get_sensor_class("nonexistent")

    def test_list_sensors_returns_all(self):
        result = list_sensors()
        assert "telescope" in result
        assert "description" in result["telescope"]


class _DummySensor(AbstractSensor):
    sensor_type: ClassVar[str] = "dummy_test"

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_capabilities(self) -> SensorCapabilities:
        return SensorCapabilities(
            acquisition_mode=SensorAcquisitionMode.ON_DEMAND,
            modalities=("test",),
        )

    def get_settings_schema(self) -> list[dict[str, Any]]:
        return []


class TestSensorManager:
    def test_register_and_get(self):
        mgr = SensorManager(logger=_logger)
        sensor = _DummySensor("s1")
        mgr.register(sensor)
        assert mgr.get("s1") is sensor

    def test_duplicate_id_raises(self):
        mgr = SensorManager(logger=_logger)
        mgr.register(_DummySensor("dup"))
        with pytest.raises(ValueError, match="Duplicate sensor id"):
            mgr.register(_DummySensor("dup"))

    def test_get_or_none(self):
        mgr = SensorManager(logger=_logger)
        assert mgr.get_or_none("missing") is None
        s = _DummySensor("ok")
        mgr.register(s)
        assert mgr.get_or_none("ok") is s

    def test_all_returns_all(self):
        mgr = SensorManager(logger=_logger)
        mgr.register(_DummySensor("a"))
        mgr.register(_DummySensor("b"))
        assert len(mgr.all()) == 2

    def test_iter_by_type(self):
        mgr = SensorManager(logger=_logger)
        mgr.register(_DummySensor("x"))
        assert len(list(mgr.iter_by_type("dummy_test"))) == 1
        assert len(list(mgr.iter_by_type("nonexistent"))) == 0

    def test_first_of_type(self):
        mgr = SensorManager(logger=_logger)
        s = _DummySensor("first")
        mgr.register(s)
        assert mgr.first_of_type("dummy_test") is s
        assert mgr.first_of_type("nope") is None

    def test_connect_all(self):
        mgr = SensorManager(logger=_logger)
        mgr.register(_DummySensor("a"))
        mgr.register(_DummySensor("b"))
        results = mgr.connect_all()
        assert results == {"a": True, "b": True}

    def test_disconnect_all_swallows_errors(self):
        mgr = SensorManager(logger=_logger)
        bad = _DummySensor("bad")
        bad.disconnect = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        mgr.register(bad)
        mgr.disconnect_all()  # should not raise

    def test_len_and_contains(self):
        mgr = SensorManager(logger=_logger)
        mgr.register(_DummySensor("q"))
        assert len(mgr) == 1
        assert "q" in mgr
        assert "z" not in mgr

    def test_from_configs_builds_telescope(self, tmp_path: Path):
        cfg = SensorConfig(
            id="telescope-0",
            type="telescope",
            adapter="dummy",
            adapter_settings={},
            citra_sensor_id="tel-abc",
        )
        mgr = SensorManager.from_configs(
            [cfg],
            logger=_logger,
            images_dir=tmp_path,
        )
        assert "telescope-0" in mgr
        from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor

        sensor = mgr.get("telescope-0")
        assert isinstance(sensor, TelescopeSensor)
        assert sensor.adapter is not None
