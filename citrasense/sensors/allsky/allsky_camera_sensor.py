"""Streaming allsky camera sensor.

Wraps an :class:`~citrasense.hardware.devices.camera.AbstractCamera`
implementation (USB or Raspberry Pi HQ today) in a periodic-capture loop.
Each frame is encoded as a JPEG, stored as the latest snapshot for HTTP
download, and pushed to :class:`~citrasense.sensors.preview_bus.PreviewBus`
so the web UI can render it live.

The sensor advertises ``STREAMING`` so
:meth:`SensorRuntime._start_streaming_sensor` calls :meth:`start_stream`
automatically once the runtime starts.  We do not publish events to the
:class:`SensorBus` — there is no consumer pipeline yet.  When phase 2 lands
(satellite-detection pipeline), :meth:`_capture_once` will additionally push
events; the public surface stays unchanged.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from datetime import datetime, timezone
from logging import Logger, LoggerAdapter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from citrasense.hardware.devices.device_registry import (
    CAMERA_DEVICES,
    check_dependencies,
    get_camera_class,
    get_device_schema,
    list_devices,
)
from citrasense.sensors.abstract_sensor import (
    AbstractSensor,
    AcquisitionContext,
    SensorAcquisitionMode,
    SensorCapabilities,
)
from citrasense.sensors.preview_bus import array_to_jpeg_data_url

if TYPE_CHECKING:
    from citrasense.hardware.devices.camera import AbstractCamera
    from citrasense.sensors.bus import SensorBus
    from citrasense.sensors.preview_bus import PreviewBus
    from citrasense.settings.citrasense_settings import SensorConfig


LoggerLike = Logger | LoggerAdapter


#: Camera types this sensor can drive.  Restricted to wide-FOV-friendly
#: cameras already in the device registry.  IP / HTTP-snapshot cameras are
#: deliberately omitted (see issue #337 — defer to a follow-up).
_SUPPORTED_CAMERA_TYPES: tuple[str, ...] = ("usb_camera", "rpi_hq")

#: Defaults that strike a balance between situational-awareness latency
#: (operators want to see clouds quickly) and not pegging a Raspberry Pi
#: 4 with a tight loop.
_DEFAULT_CAPTURE_INTERVAL_S: float = 30.0
_DEFAULT_EXPOSURE_S: float = 1.0
_DEFAULT_GAIN: float = 1.0
_DEFAULT_JPEG_QUALITY: int = 85

#: How long capture-loop start/stop will wait for the worker thread.  The
#: loop checks the stop event before every sleep, so a graceful shutdown
#: from inside ``stop_stream`` should always complete in well under this.
_THREAD_JOIN_TIMEOUT_S: float = 5.0

#: Prefix attached to camera-owned settings inside ``adapter_settings`` so
#: they don't collide with allsky-loop settings.  Mirrors the
#: :class:`DirectHardwareAdapter` ``camera_*`` scheme so the existing
#: ``build_settings_schema`` / ``adapterField`` / ``visible_when`` form
#: machinery drives the same UX.
_CAMERA_SETTING_PREFIX: str = "camera_"


class AllskyCameraSensor(AbstractSensor):
    """Periodic full-sky capture pushed to :class:`PreviewBus`."""

    sensor_type: ClassVar[str] = "allsky"

    _CAPABILITIES: ClassVar[SensorCapabilities] = SensorCapabilities(
        acquisition_mode=SensorAcquisitionMode.STREAMING,
        modalities=("allsky",),
    )

    def __init__(
        self,
        sensor_id: str,
        *,
        camera: AbstractCamera,
        camera_type: str,
        logger: LoggerLike | None = None,
        capture_interval_s: float = _DEFAULT_CAPTURE_INTERVAL_S,
        exposure_s: float = _DEFAULT_EXPOSURE_S,
        gain: float = _DEFAULT_GAIN,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        flip_horizontal: bool = False,
        preview_bus: PreviewBus | None = None,
        dependency_issues: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(sensor_id=sensor_id)
        self._camera: AbstractCamera = camera
        self._camera_type = camera_type
        # Banner-shape dicts for the Missing Dependencies UI banner.  Computed
        # in ``from_config`` once the ``camera_type`` is known so a missing
        # ``opencv-python`` (or ``picamera2``) shows up at startup instead of
        # being discovered the hard way at ``connect()`` time.
        self._dependency_issues: list[dict[str, str]] = list(dependency_issues or [])
        base_logger: LoggerLike = logger if logger is not None else logging.getLogger("citrasense")
        if isinstance(base_logger, logging.Logger):
            self._logger: LoggerLike = base_logger.getChild(f"AllskyCameraSensor[{sensor_id}]")
        else:
            self._logger = base_logger
        # Lower bounds are enforced by the web schema (``min`` in
        # ``build_settings_schema``); keep the constructor permissive so
        # tests can drive the loop at sub-second intervals without fighting
        # production-grade floors.
        self._capture_interval_s = float(capture_interval_s)
        self._exposure_s = float(exposure_s)
        self._gain = float(gain)
        self._jpeg_quality = max(1, min(int(jpeg_quality), 100))
        self._flip_horizontal = bool(flip_horizontal)

        # Preview wiring.  Set in the constructor for unit tests; the daemon
        # overwrites it after construction for production use (mirrors the
        # ``on_toast`` pattern PassiveRadarSensor uses for the web toast
        # callback).  The capture loop tolerates ``None`` so a misconfigured
        # site doesn't crash — frames just won't reach the browser.
        self.preview_bus: PreviewBus | None = preview_bus

        # Public-facing bookkeeping (consumed by ``get_live_status`` and the
        # web routes).  Guarded by ``_state_lock`` so the capture thread and
        # the FastAPI worker thread don't tear it.
        self.name: str = sensor_id
        self._state_lock = threading.RLock()
        self._connected = False
        self._last_capture_at: datetime | None = None
        self._last_capture_duration_s: float | None = None
        self._last_frame_size: tuple[int, int] | None = None
        self._last_error: str | None = None
        self._capture_count: int = 0
        self._latest_jpeg: bytes | None = None

        # Capture-loop control.
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._streaming = False
        # Serialize capture invocations so the periodic loop and a manual
        # ``capture_now()`` can't drive the camera concurrently.
        self._capture_lock = threading.Lock()

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: SensorConfig,
        *,
        logger: Logger,
        images_dir: Path,
        **_kwargs: object,
    ) -> AllskyCameraSensor:
        """Build from a :class:`SensorConfig`.

        ``adapter_settings`` keys consumed:

        - ``camera_type``: one of ``usb_camera``, ``rpi_hq``.  Required.
        - ``capture_interval_s`` / ``exposure_s`` / ``gain`` / ``jpeg_quality``
          / ``flip_horizontal``: allsky-loop tuning, all optional.
        - any key prefixed with ``camera_`` is forwarded to the camera
          constructor with the prefix stripped (e.g.
          ``camera_camera_index`` → ``camera_index`` for ``UsbCamera``).
          Mirrors the prefix scheme :class:`DirectHardwareAdapter` uses
          so the existing form renderer drives both consistently.
        """
        del images_dir  # allsky writes nothing to disk in phase 1
        from citrasense.logging.sensor_logger import get_sensor_logger

        adapter_settings = dict(cfg.adapter_settings or {})
        camera_type = str(adapter_settings.get("camera_type") or "").strip()
        if not camera_type:
            raise ValueError(
                f"Allsky sensor {cfg.id!r} is missing 'camera_type' in adapter_settings; "
                f"set one of: {', '.join(_SUPPORTED_CAMERA_TYPES)}"
            )
        if camera_type not in _SUPPORTED_CAMERA_TYPES:
            raise ValueError(
                f"Allsky sensor {cfg.id!r} has unsupported camera_type={camera_type!r}; "
                f"supported types: {', '.join(_SUPPORTED_CAMERA_TYPES)}"
            )

        camera_class = get_camera_class(camera_type)
        # Pull out ``camera_*`` keys and strip the prefix.  Note the
        # ``camera_type`` discriminator itself is *not* stripped — it's the
        # selector for which camera to instantiate, not a kwarg the camera
        # constructor wants.
        prefix = _CAMERA_SETTING_PREFIX
        camera_kwargs = {
            k[len(prefix) :]: v for k, v in adapter_settings.items() if k.startswith(prefix) and k != "camera_type"
        }

        # Surface missing optional packages in the Missing Dependencies
        # banner via :meth:`get_missing_dependencies`.  Mirrors what
        # :class:`DirectHardwareAdapter` does for telescope cameras —
        # check now, warn loudly, but still construct the camera object
        # (its native imports are lazy in ``connect()``) so the form and
        # status pages render normally.
        deps = check_dependencies(camera_class)
        dependency_issues: list[dict[str, str]] = []
        if not deps["available"]:
            missing = ", ".join(deps["missing"])
            logger.warning(
                "Allsky %s: camera %s is missing dependencies (%s); install with: %s",
                cfg.id,
                camera_type,
                missing,
                deps["install_cmd"],
            )
            dependency_issues.append(
                {
                    "device_type": "Camera",
                    "device_name": camera_class.get_friendly_name(),
                    "missing_packages": missing,
                    "install_cmd": deps["install_cmd"],
                }
            )

        camera_logger = get_sensor_logger(
            logger.getChild(f"{camera_class.__name__}[{cfg.id}]"),
            cfg.id,
        )
        camera = camera_class(logger=camera_logger, **camera_kwargs)

        sensor_logger = get_sensor_logger(
            logger.getChild(f"AllskyCameraSensor[{cfg.id}]"),
            cfg.id,
        )
        return cls(
            sensor_id=cfg.id,
            camera=camera,
            camera_type=camera_type,
            logger=sensor_logger,
            dependency_issues=dependency_issues,
            capture_interval_s=float(adapter_settings.get("capture_interval_s", _DEFAULT_CAPTURE_INTERVAL_S)),
            exposure_s=float(adapter_settings.get("exposure_s", _DEFAULT_EXPOSURE_S)),
            gain=float(adapter_settings.get("gain", _DEFAULT_GAIN)),
            jpeg_quality=int(adapter_settings.get("jpeg_quality", _DEFAULT_JPEG_QUALITY)),
            flip_horizontal=bool(adapter_settings.get("flip_horizontal", False)),
        )

    # ── AbstractSensor surface ───────────────────────────────────────────

    def get_capabilities(self) -> SensorCapabilities:
        return self._CAPABILITIES

    @classmethod
    def build_settings_schema(cls, sensor_id: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        """Return the allsky adapter-settings schema.

        Class-method so the web layer can fetch the form before any sensor
        instance exists (Add Sensor wizard).  Returns the same
        :class:`~citrasense.hardware.abstract_astro_hardware_adapter.SettingSchemaEntry`
        shape the existing form renderer (``components.adapterField``)
        consumes.

        ``kwargs`` may contain a ``camera_type`` selector (forwarded by the
        web layer when the user picks a camera from the dropdown) so the
        chosen camera's own schema gets merged in with a ``camera_``
        prefix, matching :class:`DirectHardwareAdapter`'s conditional
        schema pattern.  Any other kwargs are ignored — they're typically
        the rest of the form's current values, which we don't need.
        """
        del sensor_id  # accepted for parity with PassiveRadarSensor.build_settings_schema

        camera_devices = list_devices("camera")
        camera_options = [
            {"value": key, "label": camera_devices.get(key, CAMERA_DEVICES[key]).get("friendly_name", key)}
            for key in _SUPPORTED_CAMERA_TYPES
            if key in CAMERA_DEVICES
        ]

        schema: list[dict[str, Any]] = [
            {
                "name": "camera_type",
                "friendly_name": "Camera type",
                "type": "str",
                "default": _SUPPORTED_CAMERA_TYPES[0],
                "group": "Camera",
                "description": (
                    "Which camera the allsky loop drives. Selecting a camera "
                    "reloads this form with that camera's own settings."
                ),
                "required": True,
                "options": camera_options,
            },
            {
                "name": "capture_interval_s",
                "friendly_name": "Capture interval (s)",
                "type": "float",
                "default": _DEFAULT_CAPTURE_INTERVAL_S,
                "group": "Capture",
                "description": "Seconds between captures. Lower = faster sky updates.",
                "required": False,
                "min": 1.0,
                "max": 600.0,
            },
            {
                "name": "exposure_s",
                "friendly_name": "Exposure (s)",
                "type": "float",
                "default": _DEFAULT_EXPOSURE_S,
                "group": "Capture",
                "description": "Per-frame exposure time. Wide-aperture allsky lenses tolerate 1-5 s on dark sky.",
                "required": False,
                "min": 0.001,
                "max": 60.0,
            },
            {
                "name": "gain",
                "friendly_name": "Gain",
                "type": "float",
                "default": _DEFAULT_GAIN,
                "group": "Capture",
                "description": "Sensor gain in device-specific units (forwarded to capture_array).",
                "required": False,
            },
            {
                "name": "jpeg_quality",
                "friendly_name": "JPEG quality",
                "type": "int",
                "default": _DEFAULT_JPEG_QUALITY,
                "group": "Capture",
                "description": "Output JPEG quality (1-100). 85 is a good default for browser preview.",
                "required": False,
                "min": 1,
                "max": 100,
            },
            {
                "name": "flip_horizontal",
                "friendly_name": "Flip horizontal",
                "type": "bool",
                "default": False,
                "group": "Capture",
                "description": "Mirror the image left/right (corrects for diagonal mirrors / dome flips).",
                "required": False,
            },
        ]

        # Pull in the chosen camera's own schema, prefixed so its keys
        # don't collide with ours (e.g. ``output_format`` is a camera
        # concept, not an allsky one — even though no allsky key uses
        # that name today, the prefix keeps us honest as new cameras /
        # new allsky settings are added).
        #
        # When the caller didn't supply ``camera_type`` (Add Sensor wizard
        # first render — ``adapter_settings`` is empty) fall back to the
        # same default the ``camera_type`` dropdown shows above, so the
        # form's visible state and the rendered fields stay consistent.
        # Without this fallback operators see the camera dropdown locked
        # on "USB Camera" and assume they're done — but the camera-owned
        # fields (device picker, output format) never appear because the
        # ``handleChange`` reload only fires on an actual user change.
        camera_type = kwargs.get("camera_type") or _SUPPORTED_CAMERA_TYPES[0]
        if isinstance(camera_type, str) and camera_type in CAMERA_DEVICES:
            try:
                camera_schema = get_device_schema("camera", camera_type)
            except Exception:
                # Best-effort: a missing dependency on the underlying camera
                # class shouldn't break the form — operators just won't see
                # camera-specific knobs until the dep is installed.
                camera_schema = []
            for entry in camera_schema:
                prefixed = dict(entry)
                prefixed["name"] = f"{_CAMERA_SETTING_PREFIX}{entry['name']}"
                if "visible_when" in entry:
                    vw = dict(entry["visible_when"])
                    vw["field"] = f"{_CAMERA_SETTING_PREFIX}{vw['field']}"
                    prefixed["visible_when"] = vw
                # Group label so operators can tell allsky-loop knobs apart
                # from camera-owned ones in the rendered form.
                prefixed.setdefault("group", "Camera Settings")
                schema.append(prefixed)

        return schema

    def get_settings_schema(self) -> list[dict[str, Any]]:
        return self.build_settings_schema(self.sensor_id, camera_type=self._camera_type)

    def get_missing_dependencies(self) -> list[dict[str, str]]:
        """Return banner-shape dicts for any missing camera packages.

        Computed once in :meth:`from_config` from the configured
        ``camera_type``; surfaced through the sensor-level interface so
        the Missing Dependencies banner picks up allsky issues without
        the status collector having to know about ``self._camera_type``.
        """
        # Defensive copy so a mutable banner consumer can't poison cached
        # state on the sensor instance.
        return [dict(entry) for entry in self._dependency_issues]

    def is_connected(self) -> bool:
        with self._state_lock:
            connected = self._connected
        if not connected:
            return False
        try:
            return bool(self._camera.is_connected())
        except Exception:
            return False

    def connect(self, ctx: AcquisitionContext | None = None) -> bool:
        """Connect the underlying camera.

        ``ctx.preview_bus`` is consumed if provided; otherwise ``self.preview_bus``
        (set by the daemon after construction) is used.  The capture loop
        starts later in :meth:`start_stream`, not here, to match the
        ``STREAMING`` lifecycle every other streaming sensor follows.
        """
        if ctx is not None and getattr(ctx, "preview_bus", None) is not None and self.preview_bus is None:
            self.preview_bus = ctx.preview_bus

        with self._state_lock:
            if self._connected:
                return True
        try:
            ok = bool(self._camera.connect())
        except Exception as exc:
            self._logger.error("Camera connect raised: %s", exc, exc_info=True)
            with self._state_lock:
                self._last_error = f"connect failed: {exc}"
            return False
        if not ok:
            self._logger.warning("Camera connect returned False")
            with self._state_lock:
                self._last_error = "camera connect returned False"
            return False
        with self._state_lock:
            self._connected = True
            self._last_error = None
        self._logger.info("AllskyCameraSensor %s connected (camera_type=%s)", self.sensor_id, self._camera_type)
        return True

    def disconnect(self) -> None:
        """Stop the capture loop (if running) and disconnect the camera."""
        try:
            self.stop_stream()
        except Exception as exc:
            self._logger.debug("stop_stream during disconnect raised: %s", exc)
        try:
            self._camera.disconnect()
        except Exception as exc:
            self._logger.warning("Camera disconnect raised: %s", exc)
        with self._state_lock:
            self._connected = False
        self._logger.info("AllskyCameraSensor %s disconnected", self.sensor_id)

    # ── Streaming ────────────────────────────────────────────────────────

    def start_stream(self, bus: SensorBus, ctx: AcquisitionContext) -> None:
        """Spawn the periodic-capture worker thread.

        ``bus`` and ``ctx.preview_bus`` are kept for protocol parity but
        unused — allsky publishes only to :class:`PreviewBus`, which the
        daemon assigns to ``self.preview_bus`` before runtime start.
        """
        del bus
        if ctx is not None and getattr(ctx, "preview_bus", None) is not None and self.preview_bus is None:
            self.preview_bus = ctx.preview_bus

        with self._state_lock:
            if self._streaming:
                return
            self._streaming = True
            self._stop_event.clear()

        thread = threading.Thread(
            target=self._capture_loop,
            name=f"allsky-capture[{self.sensor_id}]",
            daemon=True,
        )
        self._capture_thread = thread
        thread.start()
        self._logger.info(
            "Allsky capture loop started for %s (interval=%.1fs, exposure=%.3fs)",
            self.sensor_id,
            self._capture_interval_s,
            self._exposure_s,
        )

    def stop_stream(self) -> None:
        """Signal the capture loop to exit and join with a timeout."""
        with self._state_lock:
            if not self._streaming:
                return
            self._streaming = False
        self._stop_event.set()
        thread = self._capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            if thread.is_alive():
                self._logger.warning(
                    "Allsky capture thread for %s did not exit within %.1fs", self.sensor_id, _THREAD_JOIN_TIMEOUT_S
                )
        self._capture_thread = None
        self._logger.info("Allsky capture loop stopped for %s", self.sensor_id)

    # ── Manual control (web routes) ──────────────────────────────────────

    def capture_now(self) -> dict[str, Any]:
        """Trigger one off-cycle capture.

        Used by the web ``POST /allsky/capture`` route.  Returns a small
        result dict so the UI can confirm the trigger.  Raises if the
        camera isn't connected — callers convert that into an HTTP 409.
        """
        if not self.is_connected():
            raise RuntimeError(f"Allsky sensor {self.sensor_id!r} is not connected")
        ok = self._capture_once()
        with self._state_lock:
            ts = self._last_capture_at
            err = self._last_error
        return {
            "ok": ok,
            "captured_at": ts.isoformat() if ts else None,
            "error": err if not ok else None,
        }

    @property
    def latest_jpeg_bytes(self) -> bytes | None:
        """Return the most recent JPEG-encoded frame, or ``None`` if none yet.

        Consumed by ``GET /allsky/latest.jpg``.  Callers must not mutate the
        returned bytes — they are shared with the capture thread.
        """
        with self._state_lock:
            return self._latest_jpeg

    # ── Live status (consumed by StatusCollector + templates) ────────────

    def get_live_status(self) -> dict[str, Any]:
        """Return a snapshot consumed by the web UI as ``sensor.allsky.*``."""
        with self._state_lock:
            last_ts = self._last_capture_at
            duration = self._last_capture_duration_s
            frame_size = self._last_frame_size
            err = self._last_error
            count = self._capture_count
            connected = self._connected
            streaming = self._streaming
        age_s: float | None = None
        if last_ts is not None:
            age_s = (datetime.now(timezone.utc) - last_ts).total_seconds()
        return {
            "sensor_id": self.sensor_id,
            "camera_type": self._camera_type,
            "connected": connected and self._safe_camera_is_connected(),
            "streaming": streaming,
            "capture_interval_s": self._capture_interval_s,
            "exposure_s": self._exposure_s,
            "gain": self._gain,
            "jpeg_quality": self._jpeg_quality,
            "flip_horizontal": self._flip_horizontal,
            "last_capture_at": last_ts.isoformat() if last_ts else None,
            "last_capture_age_s": age_s,
            "last_capture_duration_s": duration,
            "frame_size": list(frame_size) if frame_size else None,
            "capture_count": count,
            "last_error": err,
        }

    def _safe_camera_is_connected(self) -> bool:
        try:
            return bool(self._camera.is_connected())
        except Exception:
            return False

    # ── Capture loop internals ───────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Run captures forever until :attr:`_stop_event` fires."""
        while not self._stop_event.is_set():
            try:
                self._capture_once()
            except Exception as exc:
                self._logger.error("Allsky capture iteration crashed: %s", exc, exc_info=True)
                with self._state_lock:
                    self._last_error = f"capture loop: {exc}"
            self._stop_event.wait(self._capture_interval_s)

    def _capture_once(self) -> bool:
        """Capture one frame; encode it; push to the preview bus.

        Returns ``True`` on success, ``False`` if the capture failed (the
        error is also recorded on ``self._last_error`` so ``get_live_status``
        and ``capture_now`` can surface it).
        """
        if not self._capture_lock.acquire(blocking=False):
            self._logger.debug("Allsky capture skipped — previous capture still in progress")
            return False
        try:
            t_start = time.perf_counter()
            try:
                frame = self._camera.capture_array(
                    duration=self._exposure_s,
                    gain=self._gain,
                    binning=self._camera.get_default_binning(),
                )
            except Exception as exc:
                with self._state_lock:
                    self._last_error = f"capture_array: {exc}"
                self._logger.error("capture_array raised: %s", exc, exc_info=True)
                return False
            duration = time.perf_counter() - t_start

            try:
                data_url = array_to_jpeg_data_url(
                    frame,
                    quality=self._jpeg_quality,
                    flip_horizontal=self._flip_horizontal,
                )
            except Exception as exc:
                with self._state_lock:
                    self._last_error = f"encode: {exc}"
                self._logger.error("JPEG encode failed: %s", exc, exc_info=True)
                return False

            jpeg_bytes = self._extract_jpeg_bytes(data_url)
            frame_size = self._frame_dimensions(frame)

            with self._state_lock:
                self._latest_jpeg = jpeg_bytes
                self._last_capture_at = datetime.now(timezone.utc)
                self._last_capture_duration_s = duration
                self._last_frame_size = frame_size
                self._capture_count += 1
                self._last_error = None

            if self.preview_bus is not None:
                try:
                    self.preview_bus.push(data_url, source="allsky", sensor_id=self.sensor_id)
                except Exception as exc:
                    self._logger.warning("PreviewBus push failed: %s", exc)
            else:
                self._logger.debug("PreviewBus not wired; capture stored but not broadcast")
            return True
        finally:
            self._capture_lock.release()

    @staticmethod
    def _extract_jpeg_bytes(data_url: str) -> bytes:
        """Strip the ``data:image/jpeg;base64,`` prefix and decode."""
        marker = ";base64,"
        idx = data_url.find(marker)
        if idx < 0:
            return b""
        return base64.b64decode(data_url[idx + len(marker) :])

    @staticmethod
    def _frame_dimensions(frame: Any) -> tuple[int, int] | None:
        """Return ``(width, height)`` from a 2-D or 3-D ndarray, else ``None``."""
        shape = getattr(frame, "shape", None)
        if not shape:
            return None
        if len(shape) >= 2:
            height = int(shape[0])
            width = int(shape[1])
            return (width, height)
        return None
