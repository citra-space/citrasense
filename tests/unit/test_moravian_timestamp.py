"""Tests for Moravian camera DATE-OBS timestamp handling.

Validates that _save_fits() uses the provided exposure_start and date_src,
and that _resolve_exposure_timestamp() correctly prioritizes GPS over host clock.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def moravian_camera():
    """Create a MoravianCamera instance without connecting to real hardware."""
    from citrascope.hardware.devices.camera.moravian_camera import MoravianCamera

    logger = MagicMock()
    cam = MoravianCamera(logger=logger)
    cam._camera_info = {"model": "TestCam", "serial_number": "SN123"}
    return cam


class TestSaveFitsTimestamp:
    def test_uses_provided_exposure_start(self, moravian_camera, tmp_path):
        ts = datetime(2026, 3, 10, 2, 0, 0, tzinfo=timezone.utc)
        buf = np.zeros((10, 10), dtype=np.uint16).tobytes()
        out = tmp_path / "test.fits"

        moravian_camera._save_fits(buf, 10, 10, 5.0, 0, 1, out, exposure_start=ts)

        with fits.open(out) as hdul:
            assert hdul[0].header["DATE-OBS"] == ts.isoformat()
            assert hdul[0].header["DATE-SRC"] == "host"

    def test_uses_gps_date_src(self, moravian_camera, tmp_path):
        ts = datetime(2026, 3, 10, 2, 0, 0, 123456, tzinfo=timezone.utc)
        buf = np.zeros((10, 10), dtype=np.uint16).tobytes()
        out = tmp_path / "test.fits"

        moravian_camera._save_fits(buf, 10, 10, 5.0, 0, 1, out, exposure_start=ts, date_src="gps")

        with fits.open(out) as hdul:
            assert hdul[0].header["DATE-SRC"] == "gps"
            assert "2026-03-10T02:00:00.123456" in hdul[0].header["DATE-OBS"]

    def test_falls_back_to_now_when_no_exposure_start(self, moravian_camera, tmp_path):
        buf = np.zeros((10, 10), dtype=np.uint16).tobytes()
        out = tmp_path / "test.fits"

        before = datetime.now(timezone.utc)
        moravian_camera._save_fits(buf, 10, 10, 1.0, 0, 1, out)
        after = datetime.now(timezone.utc)

        with fits.open(out) as hdul:
            date_obs = datetime.fromisoformat(hdul[0].header["DATE-OBS"])
            assert before <= date_obs <= after


class TestResolveExposureTimestamp:
    def test_returns_gps_when_available(self, moravian_camera):
        moravian_camera._has_gps = True
        mock_cam = MagicMock()
        mock_cam.get_image_time_stamp.return_value = (2026, 3, 10, 2, 11, 32.5)
        moravian_camera._gxccd = mock_cam

        ts, src = moravian_camera._resolve_exposure_timestamp()

        assert src == "gps"
        assert ts == datetime(2026, 3, 10, 2, 11, 32, 500000, tzinfo=timezone.utc)

    def test_falls_back_to_host_when_gps_returns_none(self, moravian_camera):
        moravian_camera._has_gps = True
        mock_cam = MagicMock()
        mock_cam.get_image_time_stamp.return_value = None
        moravian_camera._gxccd = mock_cam
        host_ts = datetime(2026, 3, 10, 2, 0, 0, tzinfo=timezone.utc)
        moravian_camera._last_exposure_start = host_ts

        ts, src = moravian_camera._resolve_exposure_timestamp()

        assert src == "host"
        assert ts == host_ts

    def test_falls_back_to_host_when_no_gps(self, moravian_camera):
        moravian_camera._has_gps = False
        host_ts = datetime(2026, 3, 10, 2, 0, 0, tzinfo=timezone.utc)
        moravian_camera._last_exposure_start = host_ts

        ts, src = moravian_camera._resolve_exposure_timestamp()

        assert src == "host"
        assert ts == host_ts

    def test_falls_back_to_host_when_gps_raises(self, moravian_camera):
        moravian_camera._has_gps = True
        mock_cam = MagicMock()
        mock_cam.get_image_time_stamp.side_effect = RuntimeError("GPS error")
        moravian_camera._gxccd = mock_cam
        host_ts = datetime(2026, 3, 10, 2, 0, 0, tzinfo=timezone.utc)
        moravian_camera._last_exposure_start = host_ts

        ts, src = moravian_camera._resolve_exposure_timestamp()

        assert src == "host"
        assert ts == host_ts

    def test_gps_sub_second_precision(self, moravian_camera):
        moravian_camera._has_gps = True
        mock_cam = MagicMock()
        mock_cam.get_image_time_stamp.return_value = (2026, 6, 15, 12, 30, 45.5)
        moravian_camera._gxccd = mock_cam

        ts, src = moravian_camera._resolve_exposure_timestamp()

        assert src == "gps"
        assert ts.microsecond == 500000
        assert ts.second == 45

    def test_gps_float_rounding_not_truncation(self, moravian_camera):
        """32.2 is 32.19999... in IEEE 754 — truncation gives 199999, round gives 200000."""
        moravian_camera._has_gps = True
        mock_cam = MagicMock()
        mock_cam.get_image_time_stamp.return_value = (2026, 1, 1, 0, 0, 32.2)
        moravian_camera._gxccd = mock_cam

        ts, _ = moravian_camera._resolve_exposure_timestamp()

        assert ts.second == 32
        assert ts.microsecond == 200000

    def test_gps_float_rounding_carry(self, moravian_camera):
        """59.9999999 should round to 60.000000 and carry into the next second."""
        moravian_camera._has_gps = True
        mock_cam = MagicMock()
        mock_cam.get_image_time_stamp.return_value = (2026, 1, 1, 0, 0, 59.9999999)
        moravian_camera._gxccd = mock_cam

        ts, _ = moravian_camera._resolve_exposure_timestamp()

        assert ts.second == 0
        assert ts.minute == 1
