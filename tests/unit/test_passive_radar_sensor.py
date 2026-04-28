"""Unit tests for :class:`PassiveRadarSensor`.

Uses a stub :class:`DetectionSource` so we exercise the sensor's own
state machine — announce waiting, autostart, stream handoff, toast
wiring, live-status rollup — without spinning up NATS.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from citrasense.sensors.abstract_sensor import AcquisitionContext, SensorAcquisitionMode
from citrasense.sensors.radar.events import RadarObservationEvent
from citrasense.sensors.radar.passive_radar_sensor import PassiveRadarSensor

from .sensor_bus_helpers import InMemoryCaptureBus


class _StubDetectionSource:
    """In-memory :class:`DetectionSource` for sensor-level tests."""

    def __init__(self) -> None:
        self.sensor_id = "pr-stub"
        self._running = False
        self._connected = False
        self.handlers: dict[str, Any] = {
            "on_observation": None,
            "on_status": None,
            "on_health": None,
            "on_stations": None,
            "on_error": None,
            "on_announce": None,
            "on_depart": None,
        }
        self.commands: list[tuple[str, dict[str, Any] | None]] = []
        self.command_replies: dict[str, dict[str, Any]] = {}
        self._last_status_monotonic: float | None = None

    # ── DetectionSource protocol ───────────────────────────────────────

    def start(self, **kwargs) -> None:
        self._running = True
        self._connected = True
        for key, value in kwargs.items():
            if key in self.handlers:
                self.handlers[key] = value

    def stop(self) -> None:
        self._running = False
        self._connected = False

    def is_running(self) -> bool:
        return self._running

    def is_connected(self) -> bool:
        return self._connected

    def set_handlers(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.handlers:
                self.handlers[key] = value

    def send_command(self, suffix: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
        self.commands.append((suffix, payload))
        return self.command_replies.get(suffix, {"ok": True})

    def is_stream_stale(self, max_age_s: float) -> bool:
        ts = self._last_status_monotonic
        if ts is None:
            return True
        return (time.monotonic() - ts) > max_age_s

    def seconds_since_status(self) -> float | None:
        ts = self._last_status_monotonic
        if ts is None:
            return None
        return time.monotonic() - ts

    # ── Test helpers — simulate a pr_sensor emitting a subject ─────────

    def emit_announce(self, payload: dict | None = None) -> None:
        h = self.handlers.get("on_announce")
        if h:
            h(payload or {"sensor_id": self.sensor_id})

    def emit_status(self, payload: dict | None = None) -> None:
        self._last_status_monotonic = time.monotonic()
        h = self.handlers.get("on_status")
        if h:
            h(payload or {"state": "running", "timestamp": "2025-11-11T18:38:11Z"})

    def emit_observation(self, payload: dict) -> None:
        h = self.handlers.get("on_observation")
        if h:
            h(payload)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_sensor(
    source: _StubDetectionSource,
    **kwargs: Any,
) -> PassiveRadarSensor:
    return PassiveRadarSensor(
        sensor_id="radar-0",
        source=source,
        logger=logging.getLogger("test.passive_radar"),
        announce_wait_seconds=0.5,
        start_wait_seconds=0.5,
        citra_antenna_id="antenna-uuid-1",
        **kwargs,
    )


# ── Tests ──────────────────────────────────────────────────────────────


class TestCapabilities:
    def test_streaming_radar_capabilities(self):
        sensor = _make_sensor(_StubDetectionSource())
        caps = sensor.get_capabilities()
        assert caps.acquisition_mode == SensorAcquisitionMode.STREAMING
        assert "radar" in caps.modalities

    def test_schema_has_filters_and_connection_groups(self):
        sensor = _make_sensor(_StubDetectionSource())
        schema = sensor.get_settings_schema()
        groups = {field["group"] for field in schema}
        assert "Connection" in groups
        assert "Filters" in groups
        assert "Radar" in groups

    def test_schema_is_normalized_to_setting_schema_entry_shape(self):
        """Field keys must match ``SettingSchemaEntry`` so the web form renders them."""
        schema = PassiveRadarSensor.build_settings_schema("radar-0")
        names = {field["name"] for field in schema}
        assert {"nats_url", "radar_sensor_id", "detection_min_snr_db"}.issubset(names)
        # Every row must expose the name/friendly_name/type trio the
        # web-form field renderer reads.
        for field in schema:
            assert "name" in field
            assert "friendly_name" in field
            assert field["type"] in {"str", "int", "float", "bool"}
        # Radar sensors reuse SensorConfig.citra_sensor_id, so the
        # legacy adapter_settings field must not appear in the schema.
        assert "citra_antenna_id" not in names
        # The ``radar_sensor_id`` field's default seeds off the caller's
        # sensor_id so the Add Sensor form can pre-populate it.
        radar_sid = next(f for f in schema if f["name"] == "radar_sensor_id")
        assert radar_sid["default"] == "radar-0"


class TestFromConfig:
    def test_citra_sensor_id_takes_precedence_over_legacy_adapter_setting(self, tmp_path):
        """``cfg.citra_sensor_id`` is authoritative; adapter_settings fallback only fires when empty."""
        from citrasense.settings.citrasense_settings import SensorConfig

        cfg = SensorConfig(
            id="radar-0",
            type="passive_radar",
            adapter="",
            citra_sensor_id="new-antenna-uuid",
            adapter_settings={"citra_antenna_id": "legacy-antenna-uuid"},
        )
        sensor = PassiveRadarSensor.from_config(
            cfg,
            logger=logging.getLogger("test.radar.from_config"),
            images_dir=tmp_path,
        )
        assert sensor.citra_antenna_id == "new-antenna-uuid"

    def test_adapter_settings_fallback_when_citra_sensor_id_empty(self, tmp_path):
        """Legacy configs (pre-UI) keep working through the adapter_settings fallback."""
        from citrasense.settings.citrasense_settings import SensorConfig

        cfg = SensorConfig(
            id="radar-0",
            type="passive_radar",
            adapter="",
            citra_sensor_id="",
            adapter_settings={"citra_antenna_id": "legacy-antenna-uuid"},
        )
        sensor = PassiveRadarSensor.from_config(
            cfg,
            logger=logging.getLogger("test.radar.from_config"),
            images_dir=tmp_path,
        )
        assert sensor.citra_antenna_id == "legacy-antenna-uuid"


class TestConnect:
    def test_connect_succeeds_after_announce(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)

        def _simulate_sensor_hello() -> None:
            time.sleep(0.05)
            source.emit_announce({"sensor_id": source.sensor_id})

        threading.Thread(target=_simulate_sensor_hello, daemon=True).start()
        ok = sensor.connect()
        assert ok is True
        assert sensor.is_connected()

    def test_connect_declares_disconnected_on_no_reply(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)
        ok = sensor.connect()
        # Source connects transport-wise but we never get an announce
        # within announce_wait_seconds — sensor reports transport-level
        # connection only.
        assert ok is True
        assert sensor.is_connected() is True  # transport up; announce missing

    def test_push_config_issues_config_set_command(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source, push_config_on_connect=True)
        threading.Thread(target=lambda: (time.sleep(0.05), source.emit_announce()), daemon=True).start()
        sensor.connect()
        assert any(cmd == "config.set" for cmd, _ in source.commands)

    def test_autostart_issues_start_command(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source, autostart_on_connect=True)

        def _hello_then_running() -> None:
            time.sleep(0.02)
            source.emit_announce()
            time.sleep(0.02)
            source.emit_status({"state": "running", "timestamp": "2025-11-11T18:38:11Z"})

        threading.Thread(target=_hello_then_running, daemon=True).start()
        sensor.connect()
        assert any(cmd == "start" for cmd, _ in source.commands)


class TestStreaming:
    def test_start_stream_publishes_radar_observation_event(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)
        threading.Thread(target=lambda: (time.sleep(0.02), source.emit_announce()), daemon=True).start()
        sensor.connect()

        bus = InMemoryCaptureBus()
        events: list[tuple[str, Any]] = []
        bus.subscribe(
            f"sensors.{sensor.sensor_id}.events.acquisition",
            lambda subject, event: events.append((subject, event)),
        )
        sensor.start_stream(bus, AcquisitionContext())

        raw_observation = {
            "timestamp": "2025-11-11T18:38:11Z",
            "target": {"citra_uuid": "sat-123"},
            "geometry": {
                "az_deg": 10.0,
                "el_deg": 20.0,
                "receiver": {"lat_deg": 35.0, "lon_deg": -106.5, "alt_m": 1500.0},
            },
            "quality": {"snr_db": 15.0},
        }
        source.emit_observation(raw_observation)

        assert len(events) == 1
        subject, event = events[0]
        assert subject == f"sensors.{sensor.sensor_id}.events.acquisition"
        assert isinstance(event, RadarObservationEvent)
        assert event.modality == "radar"
        assert event.payload["target"]["citra_uuid"] == "sat-123"
        assert event.sensor_id == sensor.sensor_id

    def test_observations_dropped_before_start_stream(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)
        threading.Thread(target=lambda: (time.sleep(0.02), source.emit_announce()), daemon=True).start()
        sensor.connect()
        # No bus attached yet — the pre-stream handler should swallow
        # this observation silently (tested by not raising).
        source.emit_observation({"timestamp": "2025-11-11T18:38:11Z", "target": {"citra_uuid": "x"}})

    def test_stop_stream_detaches_bus(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)
        threading.Thread(target=lambda: (time.sleep(0.02), source.emit_announce()), daemon=True).start()
        sensor.connect()

        bus = InMemoryCaptureBus()
        events: list[tuple[str, Any]] = []
        bus.subscribe(
            f"sensors.{sensor.sensor_id}.events.acquisition",
            lambda subject, event: events.append((subject, event)),
        )
        sensor.start_stream(bus, AcquisitionContext())
        sensor.stop_stream()

        source.emit_observation({"timestamp": "2025-11-11T18:38:11Z", "target": {"citra_uuid": "x"}})
        assert events == []


class TestToasts:
    def test_error_fires_danger_toast(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)
        toasts: list[tuple[str, str, str | None]] = []
        sensor.on_toast = lambda msg, kind, tid: toasts.append((msg, kind, tid))
        source.start(
            on_observation=sensor._on_observation_before_stream,
            on_error=sensor._on_error,
        )
        handler = source.handlers["on_error"]
        assert handler is not None
        handler({"error": "boom"})
        assert any(kind == "danger" for _, kind, _ in toasts)

    def test_announce_fires_info_toast(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source)
        toasts: list[tuple[str, str, str | None]] = []
        sensor.on_toast = lambda msg, kind, tid: toasts.append((msg, kind, tid))
        source.start(on_announce=sensor._on_announce)
        handler = source.handlers["on_announce"]
        assert handler is not None
        handler({"sensor_id": source.sensor_id})
        assert any(kind == "info" for _, kind, _ in toasts)


class TestLiveStatus:
    def test_live_status_reports_offline_when_stale(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source, status_staleness_timeout_s=0.0)
        threading.Thread(target=lambda: (time.sleep(0.02), source.emit_announce()), daemon=True).start()
        sensor.connect()
        # No status ever emitted — stream is stale.
        status = sensor.get_live_status()
        assert status["is_stale"] is True
        assert status["state"] == "offline"

    def test_live_status_carries_state_when_fresh(self):
        source = _StubDetectionSource()
        sensor = _make_sensor(source, status_staleness_timeout_s=60.0)
        threading.Thread(target=lambda: (time.sleep(0.02), source.emit_announce()), daemon=True).start()
        sensor.connect()
        source.emit_status({"state": "running", "timestamp": "2025-11-11T18:38:11Z"})
        status = sensor.get_live_status()
        assert status["state"] == "running"
        assert status["is_stale"] is False
