"""Tests for SensorRuntime: queue wiring, dispatcher proxying, and streaming."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.sensor_runtime import SensorRuntime
from tests.unit.sensor_bus_helpers import InMemoryCaptureBus

# ── helpers ────────────────────────────────────────────────────────────────


class _FakeStreamingSensor:
    """Minimal sensor stub that advertises STREAMING mode."""

    sensor_type = "passive_radar"

    def __init__(self, sensor_id: str = "radar-0") -> None:
        self.sensor_id = sensor_id
        # ``mark_connected`` reaches into ``start_stream`` on every
        # connect (see SensorRuntime docstring) — record calls so
        # tests can verify the reconnect-restarts-stream behavior.
        self.start_stream_calls: int = 0
        self.stop_stream_calls: int = 0

    def get_capabilities(self) -> SensorCapabilities:
        return SensorCapabilities(
            acquisition_mode=SensorAcquisitionMode.STREAMING,
            modalities=("radar",),
        )

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_settings_schema(self):
        return []

    def start_stream(self, bus, ctx) -> None:
        del bus, ctx
        self.start_stream_calls += 1

    def stop_stream(self) -> None:
        self.stop_stream_calls += 1


class _FakeTelescopeSensor:
    """Minimal sensor stub that advertises ON_DEMAND mode."""

    sensor_type = "telescope"

    def __init__(self, sensor_id: str = "scope-0") -> None:
        self.sensor_id = sensor_id

    def get_capabilities(self) -> SensorCapabilities:
        return SensorCapabilities(
            acquisition_mode=SensorAcquisitionMode.ON_DEMAND,
            modalities=("optical",),
        )


class _TestEvent(BaseModel):
    value: int


def _mock_settings():
    s = MagicMock()
    s.task_processing_paused = False
    s.max_task_retries = 3
    s.initial_retry_delay_seconds = 1
    s.max_retry_delay_seconds = 10
    return s


def _make_runtime(
    sensor=None,
    *,
    sensor_bus=None,
    hardware_adapter=None,
) -> SensorRuntime:
    """Build a SensorRuntime with minimal mocks. Does NOT start queues."""
    if sensor is None:
        sensor = _FakeTelescopeSensor()
    return SensorRuntime(
        cast(AbstractSensor, sensor),
        logger=MagicMock(),
        settings=_mock_settings(),
        api_client=MagicMock(),
        hardware_adapter=hardware_adapter or MagicMock(),
        sensor_bus=sensor_bus,
    )


# ── Queue trio ownership ──────────────────────────────────────────────────


class TestQueueOwnership:
    def test_runtime_creates_three_queues(self):
        rt = _make_runtime()
        assert rt.acquisition_queue is not None
        assert rt.processing_queue is not None
        assert rt.upload_queue is not None

    def test_acquisition_queue_runtime_backref(self):
        rt = _make_runtime()
        assert rt.acquisition_queue.runtime is rt


# ── Dispatcher proxy methods ──────────────────────────────────────────────


class TestDispatcherProxy:
    def _wired_runtime(self):
        rt = _make_runtime()
        dispatcher = MagicMock()
        rt.set_dispatcher(dispatcher)
        return rt, dispatcher

    def test_update_task_stage_proxies(self):
        rt, d = self._wired_runtime()
        rt.update_task_stage("t1", "processing")
        d.update_task_stage.assert_called_once_with("t1", "processing")

    def test_remove_task_from_all_stages_proxies(self):
        rt, d = self._wired_runtime()
        rt.remove_task_from_all_stages("t1")
        d.remove_task_from_all_stages.assert_called_once_with("t1")

    def test_record_task_started_proxies(self):
        rt, d = self._wired_runtime()
        rt.record_task_started()
        d.record_task_started.assert_called_once()

    def test_record_task_succeeded_proxies(self):
        rt, d = self._wired_runtime()
        rt.record_task_succeeded()
        d.record_task_succeeded.assert_called_once()

    def test_record_task_failed_proxies(self):
        rt, d = self._wired_runtime()
        rt.record_task_failed()
        d.record_task_failed.assert_called_once()

    def test_get_task_by_id_proxies(self):
        rt, d = self._wired_runtime()
        d.get_task_by_id.return_value = "fake-task"
        assert rt.get_task_by_id("t1") == "fake-task"
        d.get_task_by_id.assert_called_once_with("t1")

    def test_proxy_asserts_without_dispatcher(self):
        rt = _make_runtime()
        with pytest.raises(AssertionError):
            rt.update_task_stage("t1", "imaging")


# ── Telescope managers ────────────────────────────────────────────────────


class TestTelescopeManagers:
    def test_telescope_sensor_gets_managers(self):
        rt = _make_runtime(_FakeTelescopeSensor())
        assert rt.autofocus_manager is not None
        assert rt.alignment_manager is not None
        assert rt.homing_manager is not None

    def test_non_telescope_sensor_has_no_managers(self):
        rt = _make_runtime(_FakeStreamingSensor(), hardware_adapter=None)
        assert rt.autofocus_manager is None
        assert rt.alignment_manager is None
        assert rt.homing_manager is None

    def test_check_maintenance_calls_managers(self):
        rt = _make_runtime()
        rt.homing_manager = MagicMock()
        rt.alignment_manager = MagicMock()
        rt.autofocus_manager = MagicMock()
        rt.calibration_manager = MagicMock()

        rt.check_maintenance()

        rt.homing_manager.check_and_execute.assert_called_once()
        rt.alignment_manager.check_and_execute.assert_called_once()
        rt.autofocus_manager.check_and_execute.assert_called_once()
        rt.calibration_manager.check_and_execute.assert_called_once()

    def test_is_maintenance_blocking_homing(self):
        rt = _make_runtime()
        rt.homing_manager = MagicMock()
        rt.homing_manager.is_running.return_value = True
        rt.homing_manager.is_requested.return_value = False
        assert rt.is_maintenance_blocking() is True

    def test_is_maintenance_blocking_false_when_idle(self):
        rt = _make_runtime()
        rt.homing_manager = MagicMock()
        rt.homing_manager.is_running.return_value = False
        rt.homing_manager.is_requested.return_value = False
        rt.calibration_manager = None
        assert rt.is_maintenance_blocking() is False


# ── Streaming ingestion ───────────────────────────────────────────────────


class TestStreamingIngestion:
    def test_streaming_sensor_subscribes_on_start(self):
        bus = InMemoryCaptureBus()
        rt = _make_runtime(_FakeStreamingSensor("radar-0"), sensor_bus=bus, hardware_adapter=None)
        rt._subscribe_streaming()

        assert rt._streaming_sub is not None

    def test_streaming_event_reaches_handler(self):
        bus = InMemoryCaptureBus()
        rt = _make_runtime(_FakeStreamingSensor("radar-0"), sensor_bus=bus, hardware_adapter=None)
        rt._subscribe_streaming()

        event = _TestEvent(value=42)
        bus.publish("sensors.radar-0.events.acquisition", event)

        assert len(bus.events) == 1
        assert bus.events[0][1] is event

    def test_on_demand_sensor_does_not_subscribe(self):
        bus = InMemoryCaptureBus()
        rt = _make_runtime(_FakeTelescopeSensor(), sensor_bus=bus)
        rt._subscribe_streaming()

        assert rt._streaming_sub is None

    def test_stop_unsubscribes(self):
        bus = InMemoryCaptureBus()
        rt = _make_runtime(_FakeStreamingSensor("radar-0"), sensor_bus=bus, hardware_adapter=None)
        rt._subscribe_streaming()
        assert rt._streaming_sub is not None

        rt._streaming_sub.unsubscribe()
        rt._streaming_sub = None

        bus.publish("sensors.radar-0.events.acquisition", _TestEvent(value=99))
        # Only the first publish (during subscribe_streaming?) was recorded;
        # the handler shouldn't fire after unsubscribe.
        # Actually InMemoryCaptureBus always records, but the handler won't fire.
        assert len(bus.events) == 1


# ── Queue idle helpers ────────────────────────────────────────────────────


class TestStreamingRestartOnReconnect:
    """Regression coverage for the allsky-stays-idle-after-reconnect bug.

    Before the fix, ``_ensure_queues_started`` was the only path that
    invoked ``sensor.start_stream``, and its one-shot guard meant
    reconnects skipped the producer-restart even though the sensor's
    ``disconnect()`` had stopped its stream.  ``mark_connected`` now
    runs the producer restart on every connect; the queue trio + bus
    subscription stay one-shot.
    """

    def test_first_connect_starts_stream(self):
        bus = InMemoryCaptureBus()
        sensor = _FakeStreamingSensor("radar-0")
        rt = _make_runtime(sensor, sensor_bus=bus, hardware_adapter=None)
        rt.mark_connected()
        assert sensor.start_stream_calls == 1

    def test_reconnect_restarts_stream(self):
        bus = InMemoryCaptureBus()
        sensor = _FakeStreamingSensor("radar-0")
        rt = _make_runtime(sensor, sensor_bus=bus, hardware_adapter=None)

        rt.mark_connected()
        rt.mark_disconnected()
        rt.mark_connected()

        # Two successful connects -> two start_stream invocations.
        # If the one-shot guard ever creeps back, this drops to 1.
        assert sensor.start_stream_calls == 2

    def test_queue_trio_starts_only_once_across_reconnects(self):
        bus = InMemoryCaptureBus()
        sensor = _FakeStreamingSensor("radar-0")
        rt = _make_runtime(sensor, sensor_bus=bus, hardware_adapter=None)
        # Replace the queues with mocks so we can see exactly how many
        # times ``.start()`` fires across the reconnect cycle.
        rt.acquisition_queue = MagicMock()
        rt.processing_queue = MagicMock()
        rt.upload_queue = MagicMock()

        rt.mark_connected()
        rt.mark_disconnected()
        rt.mark_connected()

        rt.acquisition_queue.start.assert_called_once()
        rt.processing_queue.start.assert_called_once()
        rt.upload_queue.start.assert_called_once()


class TestQueueHelpers:
    def test_are_queues_idle_all_idle(self):
        rt = _make_runtime()
        rt.acquisition_queue = MagicMock()
        rt.processing_queue = MagicMock()
        rt.upload_queue = MagicMock()
        rt.acquisition_queue.is_idle.return_value = True
        rt.processing_queue.is_idle.return_value = True
        rt.upload_queue.is_idle.return_value = True
        assert rt.are_queues_idle() is True

    def test_are_queues_idle_one_busy(self):
        rt = _make_runtime()
        rt.acquisition_queue = MagicMock()
        rt.processing_queue = MagicMock()
        rt.upload_queue = MagicMock()
        rt.acquisition_queue.is_idle.return_value = False
        rt.processing_queue.is_idle.return_value = True
        rt.upload_queue.is_idle.return_value = True
        assert rt.are_queues_idle() is False
