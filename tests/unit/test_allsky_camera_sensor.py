"""Unit tests for :class:`AllskyCameraSensor`.

Uses an in-memory fake camera so we exercise the sensor's lifecycle —
``connect`` / ``start_stream`` / capture loop / ``stop_stream`` / live
status / manual ``capture_now`` — without touching real USB or RPi
hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from citrasense.sensors.abstract_sensor import AcquisitionContext, SensorAcquisitionMode
from citrasense.sensors.allsky.allsky_camera_sensor import AllskyCameraSensor
from citrasense.sensors.preview_bus import PreviewBus


class _FakeCamera:
    """Minimal :class:`AbstractCamera` stand-in for sensor-level tests.

    Records every ``capture_array`` call so tests can assert how many
    captures the loop fired, and returns a deterministic gradient frame
    so JPEG encoding produces non-trivial bytes.
    """

    def __init__(self, *, height: int = 32, width: int = 64, dtype=np.uint8) -> None:
        self._height = height
        self._width = width
        self._dtype = dtype
        self._connected = False
        self.capture_calls: list[dict[str, Any]] = []
        self.disconnect_count = 0
        self.connect_should_succeed = True
        self.capture_should_raise: Exception | None = None

    # ── AbstractHardwareDevice surface ───────────────────────────────

    def connect(self) -> bool:
        if not self.connect_should_succeed:
            return False
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.disconnect_count += 1
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ── AbstractCamera surface (only what the sensor uses) ───────────

    def capture_array(self, duration: float, gain: int | None = None, binning: int = 1, **_kwargs) -> np.ndarray:
        self.capture_calls.append({"duration": duration, "gain": gain, "binning": binning})
        if self.capture_should_raise is not None:
            raise self.capture_should_raise
        # Gradient so percentile stretch in array_to_jpeg_data_url has signal
        x = np.arange(self._width, dtype=np.float32)
        y = np.arange(self._height, dtype=np.float32)
        grid = (x[None, :] + y[:, None]).astype(self._dtype)
        return grid

    def get_default_binning(self) -> int:
        return 1


# ── Helpers ────────────────────────────────────────────────────────────


def _make_sensor(
    *,
    capture_interval_s: float = 0.05,
    **kwargs: Any,
) -> tuple[AllskyCameraSensor, _FakeCamera, PreviewBus]:
    camera = _FakeCamera()
    bus = PreviewBus()
    sensor = AllskyCameraSensor(
        sensor_id="allsky-0",
        camera=camera,  # type: ignore[arg-type]
        camera_type="usb_camera",
        logger=logging.getLogger("test.allsky"),
        capture_interval_s=capture_interval_s,
        exposure_s=0.001,
        gain=1.0,
        jpeg_quality=70,
        preview_bus=bus,
        **kwargs,
    )
    return sensor, camera, bus


def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ── Tests ──────────────────────────────────────────────────────────────


def test_capabilities_reports_streaming_allsky() -> None:
    sensor, _camera, _bus = _make_sensor()
    caps = sensor.get_capabilities()
    assert caps.acquisition_mode is SensorAcquisitionMode.STREAMING
    assert caps.modalities == ("allsky",)


def test_connect_succeeds_when_camera_connects() -> None:
    sensor, camera, _bus = _make_sensor()
    assert sensor.connect() is True
    assert sensor.is_connected() is True
    assert camera.is_connected() is True


def test_connect_fails_gracefully_when_camera_refuses() -> None:
    sensor, camera, _bus = _make_sensor()
    camera.connect_should_succeed = False
    assert sensor.connect() is False
    assert sensor.is_connected() is False
    status = sensor.get_live_status()
    assert status["last_error"] is not None
    assert "connect" in status["last_error"]


def test_start_stream_runs_capture_loop_and_pushes_to_preview_bus() -> None:
    sensor, camera, bus = _make_sensor(capture_interval_s=0.02)
    assert sensor.connect()
    sensor.start_stream(bus=None, ctx=AcquisitionContext())  # type: ignore[arg-type]
    try:
        # Wait for the loop to fire at least twice — that confirms the
        # capture_interval_s sleep path works (not just the first frame).
        assert _wait_for(
            lambda: len(camera.capture_calls) >= 2, timeout=2.0
        ), f"capture loop produced only {len(camera.capture_calls)} captures"
    finally:
        sensor.stop_stream()

    frames = bus.pop_all()
    assert frames, "PreviewBus received no frames"
    payload, source, kind, sensor_id = frames[0]
    assert kind == "data"
    assert source == "allsky"
    assert sensor_id == "allsky-0"
    assert payload.startswith("data:image/jpeg;base64,")


def test_get_live_status_after_capture_has_shape() -> None:
    sensor, _camera, _bus = _make_sensor(capture_interval_s=0.02)
    assert sensor.connect()
    sensor.start_stream(bus=None, ctx=AcquisitionContext())  # type: ignore[arg-type]
    try:
        assert _wait_for(lambda: sensor.get_live_status()["capture_count"] >= 1, timeout=2.0)
    finally:
        sensor.stop_stream()

    status = sensor.get_live_status()
    assert status["sensor_id"] == "allsky-0"
    assert status["camera_type"] == "usb_camera"
    assert status["connected"] is True
    assert status["capture_interval_s"] == pytest.approx(0.02)
    assert status["frame_size"] == [64, 32]  # gradient is 32H x 64W
    assert status["last_capture_at"] is not None


def test_capture_now_succeeds_and_updates_latest_jpeg() -> None:
    sensor, _camera, _bus = _make_sensor()
    assert sensor.connect()
    result = sensor.capture_now()
    assert result["ok"] is True
    assert result["captured_at"] is not None
    jpeg = sensor.latest_jpeg_bytes
    assert jpeg is not None
    # JPEG SOI marker: every JPEG starts with 0xFF 0xD8.
    assert jpeg[:2] == b"\xff\xd8"


def test_capture_now_rejects_when_disconnected() -> None:
    sensor, _camera, _bus = _make_sensor()
    with pytest.raises(RuntimeError, match="not connected"):
        sensor.capture_now()


def test_capture_now_records_error_on_camera_failure() -> None:
    sensor, camera, _bus = _make_sensor()
    assert sensor.connect()
    camera.capture_should_raise = RuntimeError("simulated USB unplug")
    result = sensor.capture_now()
    assert result["ok"] is False
    assert "simulated USB unplug" in (result["error"] or "")
    assert "simulated USB unplug" in (sensor.get_live_status()["last_error"] or "")


def test_disconnect_stops_loop_and_clears_camera() -> None:
    sensor, camera, _bus = _make_sensor(capture_interval_s=0.02)
    assert sensor.connect()
    sensor.start_stream(bus=None, ctx=AcquisitionContext())  # type: ignore[arg-type]
    assert _wait_for(lambda: len(camera.capture_calls) >= 1, timeout=2.0)
    sensor.disconnect()
    # No new captures after disconnect — give the loop a chance to wake.
    count_after_stop = len(camera.capture_calls)
    time.sleep(0.1)
    assert len(camera.capture_calls) == count_after_stop
    assert camera.disconnect_count == 1
    # Streaming flag is back to false.
    assert sensor.get_live_status()["streaming"] is False


def test_stop_stream_is_idempotent() -> None:
    sensor, _camera, _bus = _make_sensor()
    sensor.connect()
    sensor.start_stream(bus=None, ctx=AcquisitionContext())  # type: ignore[arg-type]
    sensor.stop_stream()
    sensor.stop_stream()  # second call must be a no-op


def test_build_settings_schema_lists_supported_camera_types() -> None:
    schema = AllskyCameraSensor.build_settings_schema()
    by_name = {entry["name"]: entry for entry in schema}
    assert "camera_type" in by_name
    assert by_name["camera_type"]["required"] is True
    options = by_name["camera_type"].get("options")
    assert options is not None
    # Options are ``{value, label}`` dicts so the form picker shows
    # human-readable names.  Sanity-check both supported camera types
    # are present.
    option_values = {opt["value"] for opt in options}
    assert "usb_camera" in option_values
    assert "rpi_hq" in option_values
    # Capture-loop knobs are unprefixed and live alongside ``camera_type``.
    for key in ("capture_interval_s", "exposure_s", "gain", "jpeg_quality", "flip_horizontal"):
        assert key in by_name, f"schema missing {key}"
    # First-render path: no ``camera_type`` kwarg falls back to the same
    # default the dropdown shows, so the operator sees the camera-owned
    # fields without having to manually re-pick the camera type.
    assert "camera_camera_index" in by_name, "first-render schema missing default camera fields"


def test_build_settings_schema_merges_camera_fields_when_camera_type_set() -> None:
    """Picking a camera reloads the schema with that camera's own settings.

    Mirrors :class:`DirectHardwareAdapter`'s behaviour — the form re-fetch
    posts ``current_settings`` containing the chosen ``camera_type`` and
    the backend appends the camera's schema with a ``camera_`` prefix so
    sibling fields don't collide.
    """
    schema = AllskyCameraSensor.build_settings_schema(camera_type="usb_camera")
    names = {entry["name"] for entry in schema}
    # USB camera schema declares ``camera_index`` and ``output_format`` —
    # both should arrive prefixed.
    assert "camera_camera_index" in names
    assert "camera_output_format" in names
    # The discriminator itself is *not* prefixed — it's the selector.
    assert "camera_type" in names

    # Same call with ``rpi_hq`` should pull in the RPi-specific fields
    # (default_gain / default_exposure_ms / output_format) instead.
    rpi_schema = AllskyCameraSensor.build_settings_schema(camera_type="rpi_hq")
    rpi_names = {entry["name"] for entry in rpi_schema}
    assert "camera_default_gain" in rpi_names
    assert "camera_default_exposure_ms" in rpi_names


def test_build_settings_schema_ignores_unknown_camera_type() -> None:
    """Unknown ``camera_type`` falls back to the static schema rather than 500ing."""
    schema = AllskyCameraSensor.build_settings_schema(camera_type="not_a_real_camera")
    names = {entry["name"] for entry in schema}
    assert "camera_type" in names
    assert not any(n.startswith("camera_") and n != "camera_type" for n in names)


def test_streaming_thread_is_a_daemon() -> None:
    sensor, _camera, _bus = _make_sensor(capture_interval_s=0.02)
    sensor.connect()
    sensor.start_stream(bus=None, ctx=AcquisitionContext())  # type: ignore[arg-type]
    try:
        thread = sensor._capture_thread
        assert thread is not None
        assert thread.daemon is True
    finally:
        sensor.stop_stream()


def test_from_config_rejects_missing_camera_type() -> None:
    """Missing ``camera_type`` raises with a helpful message."""

    class _FakeSensorConfig:
        id = "allsky-0"
        adapter_settings: dict[str, Any] = {}

    with pytest.raises(ValueError, match="camera_type"):
        AllskyCameraSensor.from_config(
            _FakeSensorConfig(),  # type: ignore[arg-type]
            logger=logging.getLogger("test.allsky"),
            images_dir=Path("/tmp"),
        )


def test_from_config_rejects_unsupported_camera_type() -> None:
    class _FakeSensorConfig:
        id = "allsky-0"
        adapter_settings = {"camera_type": "moravian"}

    with pytest.raises(ValueError, match="moravian"):
        AllskyCameraSensor.from_config(
            _FakeSensorConfig(),  # type: ignore[arg-type]
            logger=logging.getLogger("test.allsky"),
            images_dir=Path("/tmp"),
        )


def test_from_config_strips_camera_prefix_before_passing_to_camera() -> None:
    """``camera_*`` keys land on the camera class with the prefix stripped.

    Mirrors :class:`DirectHardwareAdapter`'s scheme — the form sends
    ``camera_camera_index=2`` and the camera constructor receives
    ``camera_index=2``.
    """
    captured_kwargs: dict[str, Any] = {}

    class _RecordingCamera:
        @classmethod
        def get_friendly_name(cls) -> str:
            return "fake"

        @classmethod
        def get_dependencies(cls) -> dict[str, str | list[str]]:
            # No optional packages → ``check_dependencies`` reports
            # ``available=True`` and ``from_config`` skips the banner branch.
            return {"packages": [], "install_extra": ""}

        def __init__(self, logger, **kwargs):
            captured_kwargs.update(kwargs)

    class _FakeSensorConfig:
        id = "allsky-0"
        adapter_settings = {
            "camera_type": "usb_camera",
            "camera_camera_index": 2,
            "camera_output_format": "jpg",
            "capture_interval_s": 5.0,
            "gain": 4.0,
        }

    import citrasense.sensors.allsky.allsky_camera_sensor as mod

    original = mod.get_camera_class
    mod.get_camera_class = lambda _t: _RecordingCamera  # type: ignore[assignment]
    try:
        sensor = AllskyCameraSensor.from_config(
            _FakeSensorConfig(),  # type: ignore[arg-type]
            logger=logging.getLogger("test.allsky"),
            images_dir=Path("/tmp"),
        )
    finally:
        mod.get_camera_class = original  # type: ignore[assignment]

    # The camera saw its own settings, NOT the allsky-loop settings or
    # the discriminator key.
    assert captured_kwargs == {"camera_index": 2, "output_format": "jpg"}
    # The allsky-loop settings landed on the sensor itself.
    status = sensor.get_live_status()
    assert status["capture_interval_s"] == pytest.approx(5.0)
    assert status["gain"] == pytest.approx(4.0)
    assert status["camera_type"] == "usb_camera"


def test_from_config_records_missing_camera_dependencies_for_banner() -> None:
    """A camera class whose deps aren't importable surfaces banner entries.

    Mirrors :class:`DirectHardwareAdapter`'s contract — the sensor still
    constructs (the SDK imports are lazy), but :meth:`get_missing_dependencies`
    returns banner-shape dicts so the Missing Dependencies UI banner can
    surface the gap at startup instead of waiting for ``connect()`` to
    blow up.
    """

    class _DepsMissingCamera:
        @classmethod
        def get_friendly_name(cls) -> str:
            return "Pretend Camera"

        @classmethod
        def get_dependencies(cls) -> dict[str, str | list[str]]:
            # ``no_such_module_xyz`` is guaranteed not to be importable, so
            # ``check_dependencies`` will mark this camera as unavailable.
            return {"packages": ["no_such_module_xyz"], "install_extra": "usb-camera"}

        def __init__(self, logger, **kwargs):
            del logger, kwargs

    class _FakeCfg:
        id = "allsky-1"
        adapter_settings = {"camera_type": "usb_camera"}

    import citrasense.sensors.allsky.allsky_camera_sensor as mod

    original = mod.get_camera_class
    mod.get_camera_class = lambda _t: _DepsMissingCamera  # type: ignore[assignment]
    try:
        sensor = AllskyCameraSensor.from_config(
            _FakeCfg(),  # type: ignore[arg-type]
            logger=logging.getLogger("test.allsky.deps"),
            images_dir=Path("/tmp"),
        )
    finally:
        mod.get_camera_class = original  # type: ignore[assignment]

    issues = sensor.get_missing_dependencies()
    assert len(issues) == 1
    issue = issues[0]
    assert issue["device_type"] == "Camera"
    assert issue["device_name"] == "Pretend Camera"
    assert "no_such_module_xyz" in issue["missing_packages"]
    # Defensive copy: caller can't poison the cached banner state.
    issues[0]["device_type"] = "MUTATED"
    assert sensor.get_missing_dependencies()[0]["device_type"] == "Camera"


def test_concurrent_capture_now_does_not_double_drive_camera() -> None:
    """``capture_now`` while the loop is mid-capture should skip cleanly.

    The internal capture lock is non-reentrant; if a manual capture
    arrives during a periodic capture the second invocation must drop
    instead of trampling the camera.
    """
    sensor, camera, _bus = _make_sensor()
    sensor.connect()

    # Fire one capture in a thread holding the camera in capture_array
    # for a noticeable interval; while it's in flight, fire capture_now.
    blocking_event = threading.Event()
    release_event = threading.Event()
    original_capture = camera.capture_array

    def _slow_capture(*args, **kwargs):
        blocking_event.set()
        release_event.wait(timeout=2.0)
        return original_capture(*args, **kwargs)

    camera.capture_array = _slow_capture  # type: ignore[method-assign]

    t = threading.Thread(target=sensor._capture_once, daemon=True)
    t.start()
    assert blocking_event.wait(timeout=1.0)

    # Now request an off-cycle capture; it must return quickly with ok=False.
    result = sensor.capture_now()
    assert result["ok"] is False

    release_event.set()
    t.join(timeout=2.0)
