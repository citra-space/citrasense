"""Unit tests for :class:`NatsDetectionSource` using a mocked
``nats-py`` client.

We don't spin up a real NATS server; instead we monkey-patch the
``nats.aio.client.Client`` class so we can:

- drive messages onto registered subscription callbacks,
- verify the correct subjects were subscribed,
- exercise JSON-decode error paths,
- assert ``send_command`` request/reply shaping,
- and assert the staleness tracker advances on ``status`` messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import pytest


class _FakeMsg:
    def __init__(self, subject: str, data: bytes) -> None:
        self.subject = subject
        self.data = data


class _FakeSubscription:
    def __init__(self, cb: Callable[[Any], Any]) -> None:
        self.cb = cb
        self._unsubscribed = False

    async def unsubscribe(self) -> None:
        self._unsubscribed = True


class _FakeNatsClient:
    """Drop-in replacement for ``nats.aio.client.Client``."""

    _instances: list[_FakeNatsClient] = []

    def __init__(self) -> None:
        self.is_connected = False
        self.is_closed = False
        self._subs: dict[str, _FakeSubscription] = {}
        self._request_replies: dict[str, bytes] = {}
        _FakeNatsClient._instances.append(self)

    async def connect(self, **kwargs) -> None:
        self.is_connected = True

    async def subscribe(self, subject: str, cb: Callable[[Any], Any]) -> _FakeSubscription:
        sub = _FakeSubscription(cb)
        self._subs[subject] = sub
        return sub

    async def drain(self) -> None:
        self.is_closed = True
        self.is_connected = False

    async def close(self) -> None:
        self.is_closed = True
        self.is_connected = False

    async def request(self, subject: str, data: bytes, timeout: float = 5.0):
        reply = self._request_replies.get(subject, b"{}")
        return _FakeMsg(subject, reply)

    async def _deliver(self, subject: str, payload: dict | bytes) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        await self._subs[subject].cb(_FakeMsg(subject, body))


@pytest.fixture
def patched_nats(monkeypatch):
    _FakeNatsClient._instances.clear()
    monkeypatch.setattr(
        "citrasense.sensors.radar.nats_detection_source.NatsClient",
        _FakeNatsClient,
    )
    return _FakeNatsClient


@pytest.fixture
def source(patched_nats):
    from citrasense.sensors.radar.nats_detection_source import NatsDetectionSource

    src = NatsDetectionSource(
        nats_url="nats://localhost:4222",
        sensor_id="pr-0",
        logger=logging.getLogger("test.nats_detection_source"),
        connect_timeout=2.0,
    )
    yield src
    src.stop()


def _deliver_sync(client: _FakeNatsClient, subject: str, payload: dict | bytes) -> None:
    """Dispatch to the ``_FakeNatsClient``'s registered subscription
    from the asyncio loop that owns it."""
    loop = asyncio.get_event_loop_policy().get_event_loop()
    # The source owns its own loop; we reach it through the subscribed
    # coroutine callback directly.
    sub = client._subs.get(subject)
    if sub is None:
        raise AssertionError(f"no subscription for {subject!r}")
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    fut = asyncio.run_coroutine_threadsafe(sub.cb(_FakeMsg(subject, body)), loop)
    fut.result(timeout=2.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("predicate never became true")


def _deliver_via_loop(src, subject: str, payload: dict | bytes) -> None:
    client = src._nc  # type: ignore[attr-defined]
    loop = src._loop  # type: ignore[attr-defined]
    assert client is not None
    assert loop is not None
    fut = asyncio.run_coroutine_threadsafe(client._deliver(subject, payload), loop)
    fut.result(timeout=2.0)


class TestSubscriptions:
    def test_start_subscribes_all_expected_subjects(self, source):
        source.start()
        assert source.is_running()
        client = source._nc  # type: ignore[attr-defined]
        subjects = set(client._subs.keys())
        assert "radar.sensor.pr-0.observations" in subjects
        assert "radar.sensor.pr-0.status" in subjects
        assert "radar.sensor.pr-0.health" in subjects
        assert "radar.sensor.pr-0.stations" in subjects
        assert "radar.sensor.pr-0.error" in subjects
        assert "radar.sensor.pr-0.detections" in subjects
        assert "radar.registry.announce" in subjects
        assert "radar.registry.depart" in subjects

    def test_observation_handler_invoked(self, source):
        received: list[dict] = []
        source.start(on_observation=received.append)
        _deliver_via_loop(source, "radar.sensor.pr-0.observations", {"detection_id": "a"})
        _wait_for(lambda: received == [{"detection_id": "a"}])

    def test_status_advances_staleness(self, source):
        source.start()
        assert source.is_stream_stale(0.1) is True
        _deliver_via_loop(source, "radar.sensor.pr-0.status", {"state": "running"})
        _wait_for(lambda: not source.is_stream_stale(5.0))


class TestMalformedMessages:
    def test_malformed_json_is_swallowed(self, source, caplog):
        received: list[dict] = []
        source.start(on_observation=received.append)
        with caplog.at_level("WARNING", logger="test.nats_detection_source"):
            _deliver_via_loop(source, "radar.sensor.pr-0.observations", b"not-json")
        # No handler call, no exception — just a warning log.
        assert received == []

    def test_non_dict_payload_is_swallowed(self, source):
        received: list[dict] = []
        source.start(on_status=received.append)
        _deliver_via_loop(source, "radar.sensor.pr-0.status", b"[1,2,3]")
        assert received == []


class TestRegistryFiltering:
    def test_announce_for_other_sensor_ignored(self, source):
        received: list[dict] = []
        source.start(on_announce=received.append)
        _deliver_via_loop(source, "radar.registry.announce", {"sensor_id": "other"})
        _deliver_via_loop(source, "radar.registry.announce", {"sensor_id": "pr-0"})
        _wait_for(lambda: len(received) == 1)
        assert received[0]["sensor_id"] == "pr-0"


class TestSendCommand:
    def test_request_reply_decodes_json(self, source):
        source.start()
        client = source._nc  # type: ignore[attr-defined]
        client._request_replies["radar.control.pr-0.ping"] = json.dumps({"ok": True, "pong": 1}).encode()
        reply = source.send_command("ping", {}, timeout=1.0)
        assert reply == {"ok": True, "pong": 1}

    def test_non_dict_reply_raises(self, source):
        source.start()
        client = source._nc  # type: ignore[attr-defined]
        client._request_replies["radar.control.pr-0.ping"] = json.dumps([1, 2, 3]).encode()
        with pytest.raises(ValueError, match="Unexpected reply shape"):
            source.send_command("ping", {}, timeout=1.0)

    def test_empty_reply_returns_empty_dict(self, source):
        source.start()
        client = source._nc  # type: ignore[attr-defined]
        client._request_replies["radar.control.pr-0.ping"] = b""
        reply = source.send_command("ping", {}, timeout=1.0)
        assert reply == {}

    def test_send_command_without_start_raises(self, source):
        with pytest.raises(ConnectionError):
            source.send_command("ping", {}, timeout=1.0)


class TestHandlerMutation:
    def test_set_handlers_updates_without_restart(self, source):
        first: list[dict] = []
        second: list[dict] = []
        source.start(on_observation=first.append)
        _deliver_via_loop(source, "radar.sensor.pr-0.observations", {"n": 1})
        _wait_for(lambda: first == [{"n": 1}])
        source.set_handlers(on_observation=second.append)
        _deliver_via_loop(source, "radar.sensor.pr-0.observations", {"n": 2})
        _wait_for(lambda: second == [{"n": 2}])
        assert first == [{"n": 1}]

    def test_set_handlers_sentinel_preserves_existing(self, source):
        received: list[dict] = []
        source.start(on_observation=received.append)
        source.set_handlers(on_status=lambda _p: None)  # do not touch on_observation
        _deliver_via_loop(source, "radar.sensor.pr-0.observations", {"n": 7})
        _wait_for(lambda: received == [{"n": 7}])
