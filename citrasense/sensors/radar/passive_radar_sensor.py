"""Streaming radar sensor that bridges ``pr_sensor`` NATS traffic onto
citrasense's :class:`~citrasense.sensors.bus.SensorBus`.

All NATS-specific code lives in
:class:`~citrasense.sensors.radar.nats_detection_source.NatsDetectionSource`;
``PassiveRadarSensor`` itself only depends on the
:class:`~citrasense.sensors.radar.detection_source.DetectionSource`
protocol.  This satisfies acceptance criterion #3 from issue #307 — a
grep of this module for ``"nats"`` surfaces only the factory import in
:meth:`from_config`.

On ``connect()`` the sensor:

1. starts the detection source,
2. waits for either a ``radar.registry.announce`` matching its
   configured ``radar_sensor_id`` or a status heartbeat (whichever
   arrives first) before declaring the sensor reachable,
3. optionally pushes its configured ``RadarConfig`` via
   ``config.set`` (best-effort),
4. optionally auto-starts the pipeline via ``start`` and waits for
   ``state=="running"``.

On ``start_stream()`` the sensor installs an observation handler that
publishes a :class:`RadarObservationEvent` onto
``sensors.{sensor_id}.events.acquisition``; downstream subscribers
(``SensorRuntime``) drive the radar pipeline from there.

Status / health / stations / error subjects feed into an internal
cache consumed by the web UI through :meth:`get_live_status`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from logging import Logger, LoggerAdapter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.radar.detection_source import DetectionSource
from citrasense.sensors.radar.events import RadarObservationEvent

#: Speed of light in m/s — used for bistatic Doppler → range-rate so the
#: slim dict carries both quantities (the plot uses range-rate; some
#: readouts prefer Doppler).  Matches the constant in
#: :mod:`citrasense.pipelines.radar.radar_detection_formatter`.
_SPEED_OF_LIGHT_M_S = 299_792_458.0

#: Default FM-carrier sanity value when neither the payload nor the
#: sensor's configured radar_config provides one.
_DEFAULT_CARRIER_HZ = 100e6

#: Maximum number of detections retained in each sensor's ring buffer.
#: At ~10 Hz this is ~50 s of history — enough to hydrate a reloaded
#: browser tab without bloating the WebSocket payload size.
_RING_BUFFER_MAX = 500

#: Anything that quacks like a logger.  ``SensorLoggerAdapter`` (which
#: ``get_sensor_logger`` returns) extends :class:`logging.LoggerAdapter`
#: rather than :class:`logging.Logger`, so pyright rightly rejects
#: passing the former where the latter is annotated.  Every entry point
#: on this module accepts either.
LoggerLike = Logger | LoggerAdapter

if TYPE_CHECKING:
    from citrasense.sensors.bus import SensorBus
    from citrasense.settings.citrasense_settings import SensorConfig


#: Default values for every ``pr_sensor`` :struct:`RadarConfig` field —
#: matches the C++ defaults in ``passive_radar/server/pr_common.h``.
#: Used both for ``config.set`` pushes when no override is supplied and
#: for the web settings schema defaults.
DEFAULT_RADAR_CONFIG: dict[str, Any] = {
    "center_freq_hz": 98e6,
    "sample_rate_hz": 10e6,
    "cpi_samples": 524288,
    "decimation": 16,
    "gain_db": 14.0,
    "threshold_db": 8.0,
    "eca_delay_taps": 200,
    "eca_doppler_bins": 1,
    "eca_doppler_max_hz": 0.0,
    "eca_reg_factor": 1e-3,
    "nci_K": 4,
    "nci_min_count": 3,
    "max_stations": 6,
    "min_stations": 1,
    "max_resid_doppler": 75.0,
    "spread_doppler": 50.0,
    "min_range_sharp_db": 4.0,
    "min_doppler_sharp_db": 4.0,
}

#: Per-field schema rows for the ``pr_sensor`` ``RadarConfig`` subset.
#:
#: Each entry is ``(name, friendly_name, type, description)`` where
#: ``type`` uses the same vocabulary as the hardware-adapter schema
#: (:class:`~citrasense.hardware.abstract_astro_hardware_adapter.SettingSchemaEntry`):
#: ``"int"``, ``"float"``, ``"str"``, ``"bool"``.  Using that vocabulary
#: lets the existing web-form renderer (``components.adapterField``)
#: drive radar fields without modification.
_RADAR_CONFIG_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("center_freq_hz", "Center frequency (Hz)", "float", "FM illuminator carrier."),
    ("sample_rate_hz", "Sample rate (Hz)", "float", "USRP ADC rate."),
    ("cpi_samples", "CPI samples", "int", "Samples per coherent integration window."),
    ("decimation", "Decimation", "int", "Post-channelizer decimation factor."),
    ("gain_db", "USRP gain (dB)", "float", "Front-end analog gain."),
    ("threshold_db", "Detection threshold (dB)", "float", "Matched-filter SNR floor."),
    ("eca_delay_taps", "ECA delay taps", "int", "Cancellation filter tap count."),
    ("eca_doppler_bins", "ECA Doppler bins", "int", "ECA Doppler extent (bins)."),
    ("eca_doppler_max_hz", "ECA Doppler max (Hz)", "float", "ECA Doppler extent (Hz)."),
    ("eca_reg_factor", "ECA regularization", "float", "Tikhonov factor for ECA solve."),
    ("nci_K", "NCI integrations (K)", "int", "Non-coherent integration count."),
    ("nci_min_count", "NCI min count", "int", "Minimum hits across NCI frames."),
    ("max_stations", "Max stations", "int", "Maximum FM illuminators to track."),
    ("min_stations", "Min stations", "int", "Minimum illuminators for a detection."),
    ("max_resid_doppler", "Max residual Doppler (Hz)", "float", "Gate on |Doppler residual|."),
    ("spread_doppler", "Spread Doppler (Hz)", "float", "Doppler search half-width."),
    ("min_range_sharp_db", "Min range sharpness (dB)", "float", "Peak sharpness gate (range)."),
    ("min_doppler_sharp_db", "Min Doppler sharpness (dB)", "float", "Peak sharpness gate (Doppler)."),
)


class DetectionRingBuffer:
    """Thread-safe bounded queue of slim-dict radar detections.

    Backed by a :class:`collections.deque` with ``maxlen`` — appends
    drop the oldest entry in O(1), which matches what we want for a
    live scatter plot.  A single :class:`threading.Lock` serialises
    ``append`` and ``snapshot_since`` because the former runs on the
    NATS asyncio thread and the latter runs on the FastAPI worker
    thread (or any other reader).

    ``snapshot_since`` returns a *copy* — callers must never mutate
    the returned list because doing so would race other readers.
    """

    __slots__ = ("_buf", "_lock")

    def __init__(self, maxlen: int = _RING_BUFFER_MAX) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, slim: dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(slim)

    def snapshot_since(self, since_ts: float | None) -> list[dict[str, Any]]:
        """Return a copy of all detections with ``ts_unix > since_ts``.

        ``since_ts`` is interpreted as follows:

        - ``None`` → return the full buffer (bounded by ``maxlen``).
        - Non-negative value → treated as an absolute Unix epoch cursor.
        - Negative value → treated as "last ``|since_ts|`` seconds"
          relative to ``time.time()``.

        The list is returned in insertion order (oldest first).
        """
        with self._lock:
            snapshot = list(self._buf)
        if since_ts is None:
            return snapshot
        cutoff = since_ts if since_ts >= 0 else time.time() + since_ts
        return [d for d in snapshot if float(d.get("ts_unix") or 0.0) > cutoff]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class PassiveRadarSensor(AbstractSensor):
    """``AbstractSensor`` implementation for the ``pr_sensor`` NATS daemon."""

    sensor_type: ClassVar[str] = "passive_radar"

    _CAPABILITIES: ClassVar[SensorCapabilities] = SensorCapabilities(
        acquisition_mode=SensorAcquisitionMode.STREAMING,
        modalities=("radar",),
    )

    # ── Lifecycle toggles (configurable per-sensor) ────────────────────

    def __init__(
        self,
        sensor_id: str,
        *,
        source: DetectionSource,
        logger: LoggerLike | None = None,
        radar_config: dict[str, Any] | None = None,
        autostart_on_connect: bool = False,
        push_config_on_connect: bool = False,
        announce_wait_seconds: float = 15.0,
        start_wait_seconds: float = 20.0,
        status_staleness_timeout_s: float = 15.0,
        citra_antenna_id: str = "",
        detection_min_snr_db: float = 0.0,
        forward_only_tasked_satellites: bool = False,
        pr_control_url: str = "",
    ) -> None:
        super().__init__(sensor_id=sensor_id)
        self._source = source
        base_logger: LoggerLike = logger if logger is not None else logging.getLogger("citrasense")
        # LoggerAdapter exposes ``getChild`` at runtime but pyright only
        # sees it on ``Logger`` — branch so the type checker is happy.
        if isinstance(base_logger, logging.Logger):
            self._logger: LoggerLike = base_logger.getChild(f"PassiveRadarSensor[{sensor_id}]")
        else:
            self._logger = base_logger
        self._radar_config = {**DEFAULT_RADAR_CONFIG, **(radar_config or {})}
        self._autostart = autostart_on_connect
        self._push_config = push_config_on_connect
        self._announce_wait = announce_wait_seconds
        self._start_wait = start_wait_seconds
        self._status_staleness = status_staleness_timeout_s
        self._citra_antenna_id = citra_antenna_id
        self._detection_min_snr_db = detection_min_snr_db
        self._forward_only_tasked = forward_only_tasked_satellites
        self._pr_control_url = pr_control_url

        # Surfaced for web UI / pipeline context.
        self.name: str = sensor_id
        self.citra_antenna_id: str = citra_antenna_id

        # ── Reactive caches (guarded by _state_lock) ───────────────────
        self._state_lock = threading.RLock()
        self._last_status: dict[str, Any] | None = None
        self._last_health: dict[str, Any] | None = None
        self._last_stations: dict[str, Any] | None = None
        self._last_receiver: dict[str, Any] | None = None
        self._last_error: dict[str, Any] | None = None
        self._last_announce: dict[str, Any] | None = None
        self._announce_event = threading.Event()
        self._status_event = threading.Event()

        # Connection bookkeeping
        self._connected = False
        self._streaming = False
        self._bus: SensorBus | None = None

        # Optional toast callback set by the daemon (web_server.send_toast).
        self.on_toast: Callable[[str, str, str | None], None] | None = None
        self._last_staleness_warning: float = 0.0
        # Edge-trigger guards.  pr_sensor re-publishes its ``announce``
        # beacon every 60s as a liveness signal, so firing an "online"
        # toast on every announce would spam the operator once a minute.
        # These flags track the *transition* into online / offline so
        # toasts only fire on the edge.
        self._announced_online: bool = False
        self._announced_offline: bool = False

        # Detection ring buffer + live broadcast callback.  The buffer
        # always accumulates; the callback is only invoked when the web
        # layer has wired it — see :meth:`_on_observation`.  The callback
        # is free-threading on purpose: consumers are expected to marshal
        # onto their own event loop (the web app uses
        # :func:`asyncio.run_coroutine_threadsafe`).
        self._detections = DetectionRingBuffer()
        self.on_detection_broadcast: Callable[[str, dict[str, Any]], None] | None = None

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: SensorConfig,
        *,
        logger: Logger,
        images_dir: Path,  # unused; radar has no filesystem capture
    ) -> PassiveRadarSensor:
        """Build from a :class:`SensorConfig` entry.

        Reads NATS URL, radar sensor id, config push / autostart
        toggles, and the pr_sensor ``RadarConfig`` subset from
        ``cfg.adapter_settings``.  Per-sensor fields (detection SNR
        floor, antenna UUID, etc.) also live there — there is no
        top-level settings block for radar.
        """
        del images_dir
        from citrasense.logging.sensor_logger import get_sensor_logger
        from citrasense.sensors.radar.nats_detection_source import NatsDetectionSource

        adapter_settings = cfg.adapter_settings or {}
        nats_url = str(adapter_settings.get("nats_url") or "nats://127.0.0.1:4222")
        radar_sensor_id = str(adapter_settings.get("radar_sensor_id") or cfg.id)
        # Prefer the top-level ``citra_sensor_id`` (shared with the telescope
        # config form). Fall back to the legacy ``adapter_settings.citra_antenna_id``
        # slot so existing config.json files keep working.
        citra_antenna_id = str(cfg.citra_sensor_id or adapter_settings.get("citra_antenna_id") or "")

        source_logger = get_sensor_logger(
            logger.getChild(f"NatsDetectionSource[{cfg.id}]"),
            cfg.id,
        )
        source = NatsDetectionSource(
            nats_url=nats_url,
            sensor_id=radar_sensor_id,
            logger=source_logger,
        )
        sensor_logger = get_sensor_logger(
            logger.getChild(f"PassiveRadarSensor[{cfg.id}]"),
            cfg.id,
        )

        radar_config = {k: adapter_settings[k] for k in DEFAULT_RADAR_CONFIG if k in adapter_settings}

        return cls(
            sensor_id=cfg.id,
            source=source,
            logger=sensor_logger,
            radar_config=radar_config,
            autostart_on_connect=bool(adapter_settings.get("autostart_on_connect", False)),
            push_config_on_connect=bool(adapter_settings.get("push_config_on_connect", False)),
            announce_wait_seconds=float(adapter_settings.get("announce_wait_seconds", 15.0)),
            start_wait_seconds=float(adapter_settings.get("start_wait_seconds", 20.0)),
            status_staleness_timeout_s=float(adapter_settings.get("status_staleness_timeout_s", 15.0)),
            citra_antenna_id=citra_antenna_id,
            detection_min_snr_db=float(adapter_settings.get("detection_min_snr_db", 0.0)),
            forward_only_tasked_satellites=bool(adapter_settings.get("forward_only_tasked_satellites", False)),
            pr_control_url=str(adapter_settings.get("pr_control_url") or ""),
        )

    # ── AbstractSensor surface ────────────────────────────────────────

    def get_capabilities(self) -> SensorCapabilities:
        return self._CAPABILITIES

    @classmethod
    def build_settings_schema(cls, sensor_id: str = "") -> list[dict[str, Any]]:
        """Return the radar adapter-settings schema.

        Exposed as a classmethod so the web layer can fetch the schema
        before a ``PassiveRadarSensor`` instance exists (for example
        while the operator is still filling in the Add Sensor form).
        Field shape matches
        :class:`~citrasense.hardware.abstract_astro_hardware_adapter.SettingSchemaEntry`
        (``name`` / ``friendly_name`` / ``type`` / ``default`` / ``group`` /
        ``description``) so the existing Hardware-tab field renderer can
        drive it unchanged.

        ``sensor_id`` is used solely as the default for the
        ``radar_sensor_id`` field — when unknown, leave empty.  The
        ``citra_antenna_id`` field is intentionally absent: the Citra
        antenna UUID lives on ``SensorConfig.citra_sensor_id`` (reused
        for both telescopes and radars).
        """
        # ``required`` is set explicitly on every row — including optional
        # ones — so the web form's ``:required="field.required"`` binding
        # always resolves to a concrete boolean. Omitting the key leaves
        # ``field.required`` as ``undefined`` in JS, which Alpine normally
        # strips from the DOM but which has caused "this field is optional
        # but I can't save it blank" reports in practice.
        schema: list[dict[str, Any]] = [
            {
                "name": "nats_url",
                "friendly_name": "NATS URL",
                "type": "str",
                "default": "nats://127.0.0.1:4222",
                "group": "Connection",
                "description": "NATS server pr_sensor publishes to.",
                "required": True,
            },
            {
                "name": "radar_sensor_id",
                "friendly_name": "pr_sensor --sensor-id",
                "type": "str",
                "default": sensor_id,
                "group": "Connection",
                "description": "Target sensor id (as passed to pr_sensor on start).",
                "required": False,
            },
            {
                "name": "pr_control_url",
                "friendly_name": "pr_control URL",
                "type": "str",
                "default": "",
                "group": "Connection",
                "description": "Optional — link shown on the radar detail page.",
                "required": False,
            },
            {
                "name": "autostart_on_connect",
                "friendly_name": "Auto-start pipeline on connect",
                "type": "bool",
                "default": False,
                "group": "Connection",
                "description": "Send 'start' to pr_sensor right after a successful connect.",
                "required": False,
            },
            {
                "name": "push_config_on_connect",
                "friendly_name": "Push RadarConfig on connect",
                "type": "bool",
                "default": False,
                "group": "Connection",
                "description": "Send the configured RadarConfig to pr_sensor on connect.",
                "required": False,
            },
            {
                "name": "status_staleness_timeout_s",
                "friendly_name": "Status staleness timeout (s)",
                "type": "float",
                "default": 15.0,
                "group": "Connection",
                "description": "If no status heartbeat arrives for this long the sensor is marked offline.",
                "required": False,
            },
            {
                "name": "detection_min_snr_db",
                "friendly_name": "Min detection SNR (dB)",
                "type": "float",
                "default": 0.0,
                "group": "Filters",
                "description": "Observations below this SNR are not uploaded.",
                "required": False,
            },
            {
                "name": "forward_only_tasked_satellites",
                "friendly_name": "Only upload tasked satellites",
                "type": "bool",
                "default": False,
                "group": "Filters",
                "description": "Drop detections that don't match a satellite tasked at this site.",
                "required": False,
            },
        ]
        for name, friendly, typ, description in _RADAR_CONFIG_FIELDS:
            schema.append(
                {
                    "name": name,
                    "friendly_name": friendly,
                    "type": typ,
                    "default": DEFAULT_RADAR_CONFIG[name],
                    "group": "Radar",
                    "description": description,
                    "required": False,
                }
            )
        return schema

    def get_settings_schema(self) -> list[dict[str, Any]]:
        return self.build_settings_schema(self.sensor_id)

    def is_connected(self) -> bool:
        return self._connected and self._source.is_running() and self._source.is_connected()

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        del ctx
        if self._connected:
            return True

        # One unified observation handler handles both the pre-stream
        # and streaming cases.  Pre-stream it still populates the ring
        # buffer and pushes live WebSocket updates; it just can't
        # publish to the sensor bus until ``start_stream`` wires one up.
        self._source.start(
            on_observation=self._on_observation,
            on_status=self._on_status,
            on_health=self._on_health,
            on_stations=self._on_stations,
            on_error=self._on_error,
            on_announce=self._on_announce,
            on_depart=self._on_depart,
        )

        reachable = self._wait_for_reachability(self._announce_wait)
        if not reachable:
            self._logger.warning(
                "pr_sensor %s not reachable after %.1fs; continuing in disconnected state",
                self.sensor_id,
                self._announce_wait,
            )
            self._connected = self._source.is_connected()
            return self._connected

        if self._push_config:
            self._push_radar_config()

        if self._autostart:
            self._send_start()

        self._connected = True
        self._logger.info("PassiveRadarSensor %s connected", self.sensor_id)
        return True

    def disconnect(self) -> None:
        if not self._connected and not self._source.is_running():
            return
        self._connected = False
        try:
            self.stop_stream()
        except Exception:
            pass
        if self._source.is_running():
            try:
                self._source.send_command("stop", {}, timeout=2.0)
            except Exception as exc:
                self._logger.debug("Best-effort stop on disconnect failed: %s", exc)
        try:
            self._source.stop()
        except Exception as exc:
            self._logger.warning("Error stopping detection source: %s", exc)

    # ── Streaming ─────────────────────────────────────────────────────

    def start_stream(self, bus: SensorBus, ctx: AcquisitionContext) -> None:
        del ctx
        self._bus = bus
        self._streaming = True
        # ``_on_observation`` is already installed on connect(); it just
        # starts publishing to the bus now that ``self._bus`` is set
        # and ``self._streaming`` flipped on.  Ring-buffer appends and
        # WebSocket broadcasts keep running independently either way.
        self._logger.info("Radar streaming started for %s", self.sensor_id)

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        self._streaming = False
        self._bus = None
        # Leave the observation handler wired so the ring buffer and
        # live-broadcast path keep feeding the UI even with the bus
        # detached.
        self._logger.info("Radar streaming stopped for %s", self.sensor_id)

    # ── Control commands (used by web routes) ─────────────────────────

    def send_start_command(self, mock: bool = False, timeout: float = 5.0) -> dict[str, Any]:
        return self._source.send_command("start", {"mock": mock}, timeout=timeout)

    def send_stop_command(self, timeout: float = 5.0) -> dict[str, Any]:
        return self._source.send_command("stop", {}, timeout=timeout)

    def send_ping(self, timeout: float = 3.0) -> dict[str, Any]:
        return self._source.send_command("ping", {}, timeout=timeout)

    def push_radar_config(self, persist: bool = False, timeout: float = 5.0) -> dict[str, Any]:
        return self._source.send_command(
            "config.set",
            {"config": self._radar_config, "persist": persist},
            timeout=timeout,
        )

    # ── Live status accessor (web UI consumes this) ──────────────────

    def get_live_status(self) -> dict[str, Any]:
        with self._state_lock:
            status = dict(self._last_status) if self._last_status else None
            health = dict(self._last_health) if self._last_health else None
            stations = dict(self._last_stations) if self._last_stations else None
            receiver = dict(self._last_receiver) if self._last_receiver else None
            err = dict(self._last_error) if self._last_error else None
            announce = dict(self._last_announce) if self._last_announce else None

        # ``seconds_since_status`` is an optional NatsDetectionSource
        # convenience method — reach for it via getattr to stay faithful
        # to the ``DetectionSource`` protocol surface.
        seconds_since_status: float | None = None
        seconds_fn = getattr(self._source, "seconds_since_status", None)
        if callable(seconds_fn):
            result = seconds_fn()
            if isinstance(result, (int, float)):
                seconds_since_status = float(result)

        is_stale = self._source.is_stream_stale(self._status_staleness)
        state: str
        if status and isinstance(status.get("state"), str):
            state = "offline" if is_stale else status["state"]
        else:
            state = "offline" if is_stale else "unknown"

        return {
            "sensor_id": self.sensor_id,
            "radar_sensor_id": getattr(self._source, "sensor_id", self.sensor_id),
            "connected": self.is_connected(),
            "state": state,
            "is_stale": is_stale,
            "seconds_since_status": seconds_since_status,
            "status": status,
            "health": health,
            "stations": stations,
            "receiver": receiver,
            "error": err,
            "announce": announce,
            "streaming": self._streaming,
            "pr_control_url": self._pr_control_url or None,
            "citra_antenna_id": self._citra_antenna_id or None,
            "last_status_received_at": self._status_received_iso(status),
        }

    @staticmethod
    def _status_received_iso(status: dict[str, Any] | None) -> str | None:
        if not status:
            return None
        ts = status.get("timestamp")
        return ts if isinstance(ts, str) else None

    # ── Internal handlers ────────────────────────────────────────────

    def _on_status(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            self._last_status = payload
        self._status_event.set()
        # Clear any stale staleness-warning so the next lapse fires a toast.
        self._last_staleness_warning = 0.0

    def _on_health(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            self._last_health = payload

    def _on_stations(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            self._last_stations = payload

    def _on_error(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            self._last_error = payload
        msg = payload.get("error") or "Unknown pr_sensor error"
        self._logger.error("pr_sensor error: %s", msg)
        if self.on_toast:
            try:
                self.on_toast(
                    f"Radar sensor {self.sensor_id} error: {msg}",
                    "danger",
                    f"radar-error-{self.sensor_id}",
                )
            except Exception:
                pass

    def _on_announce(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            self._last_announce = payload
            # Only toast on the offline→online edge — pr_sensor
            # re-publishes its announce every 60s as a liveness
            # heartbeat, so toasting on every message would spam the
            # operator once a minute.
            should_toast = not self._announced_online
            self._announced_online = True
            self._announced_offline = False
        self._announce_event.set()
        if should_toast and self.on_toast:
            try:
                self.on_toast(
                    f"Radar sensor {self.sensor_id} online",
                    "info",
                    f"radar-announce-{self.sensor_id}",
                )
            except Exception:
                pass

    def _on_depart(self, payload: dict[str, Any]) -> None:
        self._logger.warning("pr_sensor %s departed: %s", self.sensor_id, payload.get("reason"))
        with self._state_lock:
            should_toast = not self._announced_offline
            self._announced_offline = True
            self._announced_online = False
        if should_toast and self.on_toast:
            try:
                self.on_toast(
                    f"Radar sensor {self.sensor_id} went offline",
                    "warning",
                    f"radar-depart-{self.sensor_id}",
                )
            except Exception:
                pass

    def _on_observation(self, payload: dict[str, Any]) -> None:
        """Unified observation handler.

        Runs on the NATS asyncio thread.  Always:

        1. Projects a slim dict for the UI and appends it to the
           detection ring buffer (hydrates deep-linked reloads).
        2. Invokes :attr:`on_detection_broadcast` when set — the web
           layer hands off to its own event loop via
           :func:`asyncio.run_coroutine_threadsafe` so we stay
           thread-agnostic here.
        3. Publishes the *full* payload to the sensor bus when a bus is
           attached *and* streaming is active.  Matches the old
           pre-stream / streaming split without needing two handlers.
        """
        slim = self._project_slim_dict(payload)
        if slim is not None:
            self._detections.append(slim)
            cb = self.on_detection_broadcast
            if cb is not None:
                try:
                    cb(self.sensor_id, slim)
                except Exception as exc:
                    # A broken callback must never take down the NATS
                    # subscriber; log and continue.
                    self._logger.debug("on_detection_broadcast raised: %s", exc)

        bus = self._bus
        if bus is None or not self._streaming:
            return
        timestamp = _parse_iso8601(payload.get("timestamp"))
        event = RadarObservationEvent(
            sensor_id=self.sensor_id,
            modality="radar",
            timestamp=timestamp,
            payload=payload,
        )
        try:
            bus.publish(f"sensors.{self.sensor_id}.events.acquisition", event)
        except Exception as exc:
            self._logger.error("Failed to publish observation to bus: %s", exc, exc_info=True)

    # ── Detection buffer / slim projection ───────────────────────────

    def get_recent_detections(self, since_ts: float | None = None) -> list[dict[str, Any]]:
        """Return a snapshot of the detection ring buffer.

        See :meth:`DetectionRingBuffer.snapshot_since` for the cursor
        semantics (``None`` = all, non-negative = absolute epoch,
        negative = "last N seconds").
        """
        return self._detections.snapshot_since(since_ts)

    def _project_slim_dict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Reduce a full radar observation payload to a UI-sized dict.

        Returns ``None`` for payloads without usable bistatic fields so
        garbage / keepalive messages don't pollute the ring buffer.
        Field names stay camelCase-free to match the rest of the
        citrasense API (snake_case); the frontend reads them directly.
        """
        bistatic = payload.get("bistatic") or {}
        quality = payload.get("quality") or {}
        target = payload.get("target") or {}
        geometry = payload.get("geometry") or {}
        transmitter = geometry.get("transmitter") or {}

        bistatic_range = _as_float(bistatic.get("bistatic_range_km"))
        if bistatic_range is None:
            bistatic_range = _as_float(payload.get("range_km"))
        doppler_hz = _as_float(bistatic.get("doppler_hz"))
        if doppler_hz is None:
            doppler_hz = _as_float(bistatic.get("doppler_meas"))

        # A "real" detection needs at least range *or* Doppler; without
        # either we have nothing to plot.  Drop silently.
        if bistatic_range is None and doppler_hz is None:
            return None

        carrier_hz = self._resolve_carrier_hz(payload)
        range_rate_km_s: float | None = None
        if doppler_hz is not None and carrier_hz and carrier_hz > 0:
            # Bistatic Doppler: f_D = -(dR/dt) * f_c / c
            range_rate_km_s = -doppler_hz * _SPEED_OF_LIGHT_M_S / carrier_hz / 1000.0

        timestamp_iso = payload.get("timestamp")
        ts_parsed = _parse_iso8601(timestamp_iso)
        ts_unix = ts_parsed.timestamp()

        # Receiver location comes in on every observation's geometry
        # block — cache the latest one so the web UI can anchor a plan
        # view at the receiver without needing a dedicated subject.
        receiver = geometry.get("receiver") or {}
        rx_lat = _as_float(receiver.get("lat_deg"))
        rx_lon = _as_float(receiver.get("lon_deg"))
        if rx_lat is not None and rx_lon is not None:
            rx_alt = _as_float(receiver.get("alt_m"))
            with self._state_lock:
                self._last_receiver = {
                    "lat_deg": rx_lat,
                    "lon_deg": rx_lon,
                    "alt_m": rx_alt,
                }

        station_name = transmitter.get("callsign") or payload.get("station_name") or None

        slim: dict[str, Any] = {
            "ts": timestamp_iso if isinstance(timestamp_iso, str) else ts_parsed.isoformat(),
            "ts_unix": ts_unix,
            "sensor_id": self.sensor_id,
            "range_km": bistatic_range,
            "range_rate_km_s": range_rate_km_s,
            "doppler_hz": doppler_hz,
            "snr_db": _as_float(quality.get("snr_db")) or _as_float(payload.get("snr_db")),
            "sat_uuid": target.get("citra_uuid") or None,
            "sat_name": target.get("name") or None,
            # Plan-view fields — pointing from the receiver plus target
            # altitude; lets the UI project the sub-satellite point.
            "az_deg": _as_float(geometry.get("az_deg")),
            "el_deg": _as_float(geometry.get("el_deg")),
            "alt_km": _as_float(geometry.get("alt_km")),
            "station": station_name,
        }
        return slim

    def _resolve_carrier_hz(self, payload: dict[str, Any]) -> float | None:
        """Best-effort FM-carrier resolution.

        Mirrors :meth:`RadarDetectionFormatter._resolve_carrier_hz`:
        configured override → transmitter field on the observation →
        first per-station entry → the default 100 MHz sanity value.
        """
        configured = _as_float(self._radar_config.get("center_freq_hz"))
        if configured and configured > 0:
            return configured
        geometry = payload.get("geometry") or {}
        transmitter = geometry.get("transmitter") or {}
        freq = _as_float(transmitter.get("freq_hz"))
        if freq:
            return freq
        per_station = payload.get("per_station") or []
        for ps in per_station:
            freq = _as_float(ps.get("freq_hz"))
            if freq:
                return freq
        return _DEFAULT_CARRIER_HZ

    # ── Staleness polling (called from a background timer) ───────────

    def poll_staleness(self) -> None:
        """Hook called by the daemon's status timer.

        Emits a warning toast the first time the status stream goes
        stale (quiet until reconnect); a fresh status message resets
        the suppression.
        """
        if not self._connected:
            return
        if not self._source.is_stream_stale(self._status_staleness):
            return
        now = time.monotonic()
        # Throttle to at most one toast per staleness window.
        if now - self._last_staleness_warning < self._status_staleness:
            return
        self._last_staleness_warning = now
        # Stream has lapsed; drop the "online" edge so that when
        # pr_sensor recovers and re-announces we fire a fresh
        # "online" toast instead of staying silent forever.
        with self._state_lock:
            self._announced_online = False
        self._logger.warning("No status heartbeat from pr_sensor %s in %.1fs", self.sensor_id, self._status_staleness)
        if self.on_toast:
            try:
                self.on_toast(
                    f"Radar sensor {self.sensor_id} status stale",
                    "warning",
                    f"radar-stale-{self.sensor_id}",
                )
            except Exception:
                pass

    # ── Connect-time helpers ─────────────────────────────────────────

    def _wait_for_reachability(self, timeout: float) -> bool:
        """Block until either an announce or status message arrives."""
        deadline = time.monotonic() + timeout
        remaining = timeout
        while remaining > 0:
            # ``Event.wait`` returns True if set; we race two events.
            if self._announce_event.wait(timeout=min(remaining, 0.25)):
                return True
            if self._status_event.is_set():
                return True
            remaining = deadline - time.monotonic()
        return self._status_event.is_set() or self._announce_event.is_set()

    def _push_radar_config(self) -> None:
        try:
            reply = self._source.send_command(
                "config.set",
                {"config": self._radar_config, "persist": False},
                timeout=5.0,
            )
        except Exception as exc:
            self._logger.warning("config.set to pr_sensor failed: %s", exc)
            return
        if not reply.get("ok", False):
            self._logger.warning("pr_sensor rejected config.set: %s", reply.get("error"))

    def _send_start(self) -> None:
        try:
            reply = self._source.send_command("start", {"mock": False}, timeout=5.0)
        except Exception as exc:
            self._logger.warning("start command to pr_sensor failed: %s", exc)
            return
        if not reply.get("ok", False):
            self._logger.warning("pr_sensor rejected start: %s", reply)
            return
        # Poll status for up to ``start_wait_seconds`` for state=running.
        deadline = time.monotonic() + self._start_wait
        while time.monotonic() < deadline:
            with self._state_lock:
                status = dict(self._last_status) if self._last_status else None
            if status and status.get("state") == "running":
                self._logger.info("pr_sensor %s is running", self.sensor_id)
                return
            time.sleep(0.5)
        self._logger.warning(
            "pr_sensor %s did not report state=running within %.1fs",
            self.sensor_id,
            self._start_wait,
        )


def _as_float(value: Any) -> float | None:
    """Coerce int/float/numeric-string to float; otherwise ``None``.

    Mirrors the helper in
    :mod:`citrasense.pipelines.radar.radar_detection_formatter` so the
    slim-dict projection and the upload formatter treat payload fields
    the same way (pr_sensor has been known to emit both numeric and
    string-formatted numbers for the same field).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_iso8601(value: Any) -> datetime:
    """Best-effort ISO 8601 parse with a sane fallback.

    ``pr_sensor`` emits ``iso_now()`` which is UTC ISO-8601; malformed
    strings fall back to ``datetime.now(UTC)`` so the event still lands
    on the bus — downstream processors can still see the payload even
    if the timestamp is unusable.
    """
    if isinstance(value, str):
        try:
            cleaned = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)
