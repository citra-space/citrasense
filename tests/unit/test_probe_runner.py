"""Tests for subprocess hardware probe isolation (#103)."""

import time
from unittest.mock import patch

from citrascope.hardware.devices.abstract_hardware_device import AbstractHardwareDevice
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


# --- Low-level run_hardware_probe tests ---


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


# --- AbstractHardwareDevice._cached_hardware_probe tests ---


class TestCachedHardwareProbe:
    """Tests for the caching + subprocess infrastructure on the base class."""

    def setup_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def teardown_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def test_delegates_to_subprocess_and_caches(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=[{"value": 1, "label": "Focuser 1"}],
        ) as mock_probe:
            result = ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=30.0,
                timeout=5.0,
            )
            assert result == [{"value": 1, "label": "Focuser 1"}]
            assert mock_probe.call_count == 1

            # Second call should hit cache, not subprocess
            result2 = ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=30.0,
                timeout=5.0,
            )
            assert result2 == result
            assert mock_probe.call_count == 1  # still 1

    def test_cache_expires(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["first"],
        ) as mock_probe:
            ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=0.0,  # expire immediately
                timeout=5.0,
            )
            assert mock_probe.call_count == 1

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["second"],
        ) as mock_probe2:
            result = ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=0.0,
                timeout=5.0,
            )
            assert result == ["second"]
            assert mock_probe2.call_count == 1

    def test_clear_probe_cache(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["cached"],
        ):
            ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=300.0,
                timeout=5.0,
            )

        ZwoEafFocuser._clear_probe_cache()

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["fresh"],
        ) as mock_probe:
            result = ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=300.0,
                timeout=5.0,
            )
            assert result == ["fresh"]
            assert mock_probe.call_count == 1

    def test_different_classes_have_independent_caches(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera
        from citrascope.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["moravian_result"],
        ):
            MoravianCamera._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=300.0,
                timeout=5.0,
            )

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["eaf_result"],
        ):
            ZwoEafFocuser._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_ttl=300.0,
                timeout=5.0,
            )

        assert AbstractHardwareDevice._hardware_probe_cache["MoravianCamera:default"][0] == ["moravian_result"]
        assert AbstractHardwareDevice._hardware_probe_cache["ZwoEafFocuser:default"][0] == ["eaf_result"]

    def test_cache_key_separates_probes(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["cameras"],
        ):
            MoravianCamera._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_key="cameras",
                cache_ttl=300.0,
                timeout=5.0,
            )

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=["modes"],
        ):
            MoravianCamera._cached_hardware_probe(
                _succeeds,
                fallback=[],
                cache_key="read_modes",
                cache_ttl=300.0,
                timeout=5.0,
            )

        assert AbstractHardwareDevice._hardware_probe_cache["MoravianCamera:cameras"][0] == ["cameras"]
        assert AbstractHardwareDevice._hardware_probe_cache["MoravianCamera:read_modes"][0] == ["modes"]


# --- Device integration tests ---


class TestMoravianProbeIntegration:
    """Verify the Moravian probe uses subprocess isolation and returns
    fallback when the underlying probe function fails."""

    def setup_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def teardown_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def test_fallback_on_probe_failure(self):
        from citrascope.hardware.devices.camera.moravian_camera import (
            _CAMERA_FALLBACK,
            _READ_MODE_FALLBACK,
            MoravianCamera,
        )

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=(_CAMERA_FALLBACK, _READ_MODE_FALLBACK),
        ):
            result = MoravianCamera._detect_available_cameras()

        assert len(result) >= 1
        assert result[0]["value"] == -1
        assert "Auto" in str(result[0]["label"])

    def test_cache_hit_skips_probe(self):
        from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera

        cached_cameras = [{"value": 42, "label": "Cached Camera"}]
        cached_modes = [{"value": 0, "label": "Mode 0"}]
        AbstractHardwareDevice._hardware_probe_cache["MoravianCamera:default"] = (
            (cached_cameras, cached_modes),
            time.time(),
        )

        result = MoravianCamera._detect_available_cameras()
        assert result == cached_cameras


class TestEafProbeIntegration:
    """Verify the EAF probe uses subprocess isolation and returns
    fallback when the underlying probe function fails."""

    def setup_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def teardown_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def test_fallback_on_probe_failure(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import (
            _FOCUSER_FALLBACK,
            ZwoEafFocuser,
        )

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=_FOCUSER_FALLBACK,
        ):
            result = ZwoEafFocuser._detect_available_focusers()

        assert len(result) >= 1
        assert result[0]["value"] == -1
        assert "Auto" in str(result[0]["label"])

    def test_cache_hit_skips_probe(self):
        from citrascope.hardware.devices.focuser.zwo_eaf import ZwoEafFocuser

        cached = [{"value": 7, "label": "Cached Focuser"}]
        AbstractHardwareDevice._hardware_probe_cache["ZwoEafFocuser:default"] = (
            cached,
            time.time(),
        )

        result = ZwoEafFocuser._detect_available_focusers()
        assert result == cached


class TestUsbCameraProbeIntegration:
    """Verify the USB camera probe uses subprocess isolation and returns
    fallback when the underlying probe function fails."""

    def setup_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def teardown_method(self):
        AbstractHardwareDevice._hardware_probe_cache.clear()

    def test_fallback_on_probe_failure(self):
        from citrascope.hardware.devices.camera.usb_camera import (
            _USB_CAMERA_FALLBACK,
            UsbCamera,
        )

        with patch(
            "citrascope.hardware.probe_runner.run_hardware_probe",
            return_value=_USB_CAMERA_FALLBACK,
        ):
            result = UsbCamera._detect_available_cameras()

        assert len(result) >= 1
        assert result[0]["value"] == 0
        assert "default" in str(result[0]["label"]).lower()

    def test_cache_hit_skips_probe(self):
        from citrascope.hardware.devices.camera.usb_camera import UsbCamera

        cached = [{"value": 2, "label": "Cached USB Cam"}]
        AbstractHardwareDevice._hardware_probe_cache["UsbCamera:default"] = (
            cached,
            time.time(),
        )

        result = UsbCamera._detect_available_cameras()
        assert result == cached

    def test_clear_camera_cache_delegates(self):
        from citrascope.hardware.devices.camera.usb_camera import UsbCamera

        AbstractHardwareDevice._hardware_probe_cache["UsbCamera:default"] = (
            [{"value": 0, "label": "stale"}],
            time.time(),
        )

        UsbCamera.clear_camera_cache()
        assert "UsbCamera:default" not in AbstractHardwareDevice._hardware_probe_cache
