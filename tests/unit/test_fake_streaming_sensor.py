"""End-to-end streaming-sensor contract test.

A throwaway ``_FakeStreamingSensor`` implements the STREAMING side of the
``AbstractSensor`` contract and publishes events through an
``InMemoryCaptureBus``. This proves the protocol holds before any real
streaming sensor (radar, RF) lands.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.bus import SensorBus

from .sensor_bus_helpers import InMemoryCaptureBus


class _StreamPayload(BaseModel):
    detection_id: str
    snr: float


class _FakeStreamingSensor(AbstractSensor):
    """In-memory streaming sensor for contract testing."""

    sensor_type: ClassVar[str] = "fake_streaming"

    def __init__(self, sensor_id: str, events_to_emit: int = 3) -> None:
        super().__init__(sensor_id)
        self._events_to_emit = events_to_emit
        self._connected = False
        self._streaming = False
        self._bus: SensorBus | None = None

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.stop_stream()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_capabilities(self) -> SensorCapabilities:
        return SensorCapabilities(
            acquisition_mode=SensorAcquisitionMode.STREAMING,
            modalities=("radar",),
        )

    def get_settings_schema(self) -> list[dict[str, Any]]:
        return []

    # ── Streaming verbs ───────────────────────────────────────────────

    def start_stream(self, bus: SensorBus, ctx: AcquisitionContext) -> None:
        self._bus = bus
        self._streaming = True
        subject = f"sensors.{self.sensor_id}.events.acquisition"
        for i in range(self._events_to_emit):
            bus.publish(
                subject,
                _StreamPayload(detection_id=f"det-{i}", snr=10.0 + i),
            )

    def stop_stream(self) -> None:
        self._streaming = False
        self._bus = None


class TestFakeStreamingSensorContract:
    def test_full_lifecycle(self):
        sensor = _FakeStreamingSensor("radar-0", events_to_emit=5)
        bus = InMemoryCaptureBus()

        assert not sensor.is_connected()

        assert sensor.connect() is True
        assert sensor.is_connected()

        caps = sensor.get_capabilities()
        assert caps.acquisition_mode == SensorAcquisitionMode.STREAMING
        assert "radar" in caps.modalities

        delivered: list[tuple[str, BaseModel]] = []
        bus.subscribe("sensors.*.events.acquisition", lambda s, e: delivered.append((s, e)))

        sensor.start_stream(bus, AcquisitionContext())

        assert len(bus.events) == 5
        assert len(delivered) == 5
        for i, (subject, event) in enumerate(delivered):
            assert subject == "sensors.radar-0.events.acquisition"
            assert isinstance(event, _StreamPayload)
            assert event.detection_id == f"det-{i}"

        sensor.stop_stream()
        sensor.disconnect()
        assert not sensor.is_connected()

    def test_stop_stream_is_idempotent(self):
        sensor = _FakeStreamingSensor("radar-1")
        sensor.stop_stream()  # no error even if never started

    def test_acquire_raises(self):
        from unittest.mock import MagicMock

        sensor = _FakeStreamingSensor("radar-2")
        with pytest.raises(NotImplementedError):
            sensor.acquire(MagicMock(), AcquisitionContext())
