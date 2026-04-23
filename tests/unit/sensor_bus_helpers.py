"""Shared test helpers for the sensor bus and streaming-sensor tests."""

from __future__ import annotations

from pydantic import BaseModel

from citrasense.sensors.bus import InProcessBus


class InMemoryCaptureBus(InProcessBus):
    """An :class:`InProcessBus` that records every published event for assertions.

    Usage::

        bus = InMemoryCaptureBus()
        bus.subscribe("sensors.*.events.acquisition", some_handler)
        bus.publish("sensors.radar-0.events.acquisition", SomeEvent(...))
        assert bus.events  # [(subject, event), ...]
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, BaseModel]] = []

    def publish(self, subject: str, event: BaseModel) -> None:
        self.events.append((subject, event))
        super().publish(subject, event)

    def clear(self) -> None:
        self.events.clear()
