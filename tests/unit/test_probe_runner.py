"""Tests for subprocess hardware probe isolation (#103)."""

import time
from unittest.mock import patch

from citrascope.hardware.probe_runner import run_hardware_probe

# --- Standalone module-level functions for subprocess pickling ---


def _succeeds() -> list[str]:
    return ["cam1", "cam2"]


def _returns_tuple() -> tuple[list[int], list[int]]:
    return ([1, 2], [3, 4])


def _raises() -> list[str]:
    raise RuntimeError("USB on fire")


def _hangs_forever() -> list[str]:
    import time as _t

    _t.sleep(60)
    return ["never"]


def _slow_but_finishes() -> str:
    import time as _t

    _t.sleep(0.3)
    return "done"


# --- Tests ---


class TestRunHardwareProbe:
    def test_success(self):
        result = run_hardware_probe(
            _succeeds,
            timeout=5.0,
            fallback=[],
            description="test probe",
        )
        assert result == ["cam1", "cam2"]

    def test_success_tuple(self):
        result = run_hardware_probe(
            _returns_tuple,
            timeout=5.0,
            fallback=([], []),
            description="test tuple probe",
        )
        assert result == ([1, 2], [3, 4])

    def test_exception_returns_fallback(self):
        fallback = ["fallback"]
        result = run_hardware_probe(
            _raises,
            timeout=5.0,
            fallback=fallback,
            description="test error probe",
        )
        assert result == fallback

    def test_timeout_returns_fallback(self):
        fallback = ["timed_out"]
        start = time.monotonic()
        result = run_hardware_probe(
            _hangs_forever,
            timeout=1.0,
            fallback=fallback,
            description="test timeout probe",
        )
        elapsed = time.monotonic() - start
        assert result == fallback
        assert elapsed < 3.0, f"Should have timed out quickly, took {elapsed:.1f}s"

    def test_slow_probe_within_timeout_succeeds(self):
        result = run_hardware_probe(
            _slow_but_finishes,
            timeout=5.0,
            fallback="nope",
            description="slow but ok",
        )
        assert result == "done"


class TestMoravianProbeIntegration:
    """Verify the Moravian probe uses subprocess isolation and returns
    fallback when the underlying probe function fails."""

    def test_fallback_on_probe_failure(self):
        """When run_hardware_probe returns the fallback (e.g. native lib
        unavailable), _detect_available_cameras should surface it."""
        from citrascope.hardware.devices.camera.moravian_camera import (
            _CAMERA_FALLBACK,
            _READ_MODE_FALLBACK,
            MoravianCamera,
        )

        MoravianCamera._camera_cache = None
        MoravianCamera._cache_timestamp = 0

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=(_CAMERA_FALLBACK, _READ_MODE_FALLBACK),
        ):
            result = MoravianCamera._detect_available_cameras()

        assert len(result) >= 1
        assert result[0]["value"] == -1
        assert "Auto" in str(result[0]["label"])

        MoravianCamera._camera_cache = None
        MoravianCamera._cache_timestamp = 0

    def test_cache_hit_skips_probe(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera

        cached = [{"value": 42, "label": "Cached Camera"}]
        MoravianCamera._camera_cache = cached
        MoravianCamera._cache_timestamp = time.time()

        result = MoravianCamera._detect_available_cameras()
        assert result == cached

        MoravianCamera._camera_cache = None
        MoravianCamera._cache_timestamp = 0


class TestEafProbeIntegration:
    """Verify the EAF probe uses subprocess isolation and returns
    fallback when the underlying probe function fails."""

    def test_fallback_on_probe_failure(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import (
            _FOCUSER_FALLBACK,
            ZwoEafFocuser,
        )

        ZwoEafFocuser._focuser_cache = None
        ZwoEafFocuser._cache_timestamp = 0

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=_FOCUSER_FALLBACK,
        ):
            result = ZwoEafFocuser._detect_available_focusers()

        assert len(result) >= 1
        assert result[0]["value"] == -1
        assert "Auto" in str(result[0]["label"])

        ZwoEafFocuser._focuser_cache = None
        ZwoEafFocuser._cache_timestamp = 0

    def test_cache_hit_skips_probe(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        cached = [{"value": 7, "label": "Cached Focuser"}]
        ZwoEafFocuser._focuser_cache = cached
        ZwoEafFocuser._cache_timestamp = time.time()

        result = ZwoEafFocuser._detect_available_focusers()
        assert result == cached

        ZwoEafFocuser._focuser_cache = None
        ZwoEafFocuser._cache_timestamp = 0
