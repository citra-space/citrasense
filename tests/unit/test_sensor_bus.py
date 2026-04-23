"""Tests for the InProcessBus implementation of SensorBus."""

from __future__ import annotations

import threading

from pydantic import BaseModel

from citrasense.sensors.bus import InProcessBus, SensorBus, Subscription


class _Ping(BaseModel):
    value: int


class TestInProcessBus:
    def test_publish_to_exact_subject(self):
        bus = InProcessBus()
        received: list[tuple[str, BaseModel]] = []
        bus.subscribe("sensors.telescope-0.events.status", lambda s, e: received.append((s, e)))

        event = _Ping(value=1)
        bus.publish("sensors.telescope-0.events.status", event)

        assert len(received) == 1
        assert received[0] == ("sensors.telescope-0.events.status", event)

    def test_glob_pattern_matches(self):
        bus = InProcessBus()
        received: list[tuple[str, BaseModel]] = []
        bus.subscribe("sensors.*.events.acquisition", lambda s, e: received.append((s, e)))

        bus.publish("sensors.radar-0.events.acquisition", _Ping(value=10))
        bus.publish("sensors.telescope-0.events.acquisition", _Ping(value=20))

        assert len(received) == 2
        assert received[0][1].value == 10  # type: ignore[attr-defined]
        assert received[1][1].value == 20  # type: ignore[attr-defined]

    def test_no_match_delivers_nothing(self):
        bus = InProcessBus()
        received: list[tuple[str, BaseModel]] = []
        bus.subscribe("sensors.telescope-0.events.status", lambda s, e: received.append((s, e)))

        bus.publish("sensors.radar-0.events.acquisition", _Ping(value=99))

        assert received == []

    def test_unsubscribe_stops_delivery(self):
        bus = InProcessBus()
        received: list[tuple[str, BaseModel]] = []
        sub = bus.subscribe("topic.a", lambda s, e: received.append((s, e)))

        bus.publish("topic.a", _Ping(value=1))
        assert len(received) == 1

        sub.unsubscribe()
        bus.publish("topic.a", _Ping(value=2))
        assert len(received) == 1  # still 1, second publish not delivered

    def test_unsubscribe_is_idempotent(self):
        bus = InProcessBus()
        sub = bus.subscribe("x", lambda s, e: None)
        sub.unsubscribe()
        sub.unsubscribe()  # no error

    def test_multiple_handlers_same_pattern(self):
        bus = InProcessBus()
        a: list[int] = []
        b: list[int] = []
        bus.subscribe("topic", lambda s, e: a.append(e.value))  # type: ignore[attr-defined]
        bus.subscribe("topic", lambda s, e: b.append(e.value))  # type: ignore[attr-defined]

        bus.publish("topic", _Ping(value=42))
        assert a == [42]
        assert b == [42]

    def test_publish_order_preserved(self):
        bus = InProcessBus()
        received: list[int] = []
        bus.subscribe("seq", lambda s, e: received.append(e.value))  # type: ignore[attr-defined]

        for i in range(5):
            bus.publish("seq", _Ping(value=i))
        assert received == [0, 1, 2, 3, 4]

    def test_satisfies_protocol(self):
        assert isinstance(InProcessBus(), SensorBus)

    def test_subscription_satisfies_protocol(self):
        bus = InProcessBus()
        sub = bus.subscribe("x", lambda s, e: None)
        assert isinstance(sub, Subscription)

    def test_thread_safety_basic(self):
        bus = InProcessBus()
        count = {"n": 0}
        lock = threading.Lock()

        def handler(s, e):
            with lock:
                count["n"] += 1

        bus.subscribe("stress.*", handler)
        threads = []
        for i in range(10):
            t = threading.Thread(target=bus.publish, args=(f"stress.{i}", _Ping(value=i)))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert count["n"] == 10
