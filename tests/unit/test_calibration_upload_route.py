"""Tests for ``POST /api/sensors/{id}/calibration/upload``.

The route ingests pre-built master FITS files into the
:class:`CalibrationLibrary` for sensors that do not expose a direct
:class:`AbstractCamera` (e.g. NINA).  These tests use a real
``CalibrationLibrary`` backed by ``tmp_path`` so that we assert the file
actually lands on disk with the expected canonical filename.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from citrasense.calibration.calibration_library import CalibrationLibrary
from citrasense.web.app import CitraSenseWebApp

# ---------------------------------------------------------------------------
# Synthetic FITS helpers
# ---------------------------------------------------------------------------


def _fits_bytes(
    data: np.ndarray,
    *,
    camser: str = "SN-TEST",
    instrume: str = "Test Camera",
    gain: int = 100,
    xbinning: int = 1,
    exptime: float | None = None,
    ccd_temp: float | None = None,
    filter_name: str | None = None,
    read_mode: str | None = None,
    readoutm: str | None = None,
) -> bytes:
    """Serialize a 2D array into a FITS byte string with common headers."""
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdr = hdu.header
    hdr["CAMSER"] = camser
    hdr["INSTRUME"] = instrume
    hdr["GAIN"] = gain
    hdr["XBINNING"] = xbinning
    if exptime is not None:
        hdr["EXPTIME"] = exptime
    if ccd_temp is not None:
        hdr["CCD-TEMP"] = ccd_temp
    if filter_name is not None:
        hdr["FILTER"] = filter_name
    if read_mode is not None:
        hdr["READMODE"] = read_mode
    if readoutm is not None:
        hdr["READOUTM"] = readoutm

    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def library(tmp_path):
    return CalibrationLibrary(root=tmp_path / "calibration")


@pytest.fixture
def client(library):
    """FastAPI TestClient wired to a minimal daemon with a real library."""
    daemon = MagicMock()
    daemon.settings = MagicMock()
    daemon.settings.get_sensor_config = lambda _sid: None

    sensor = MagicMock()
    sensor.sensor_id = "scope-0"
    sensor.adapter = MagicMock()
    # The route only reads ``adapter.camera`` (for the status endpoint),
    # not anything on ``adapter`` for the upload itself.  Keep it minimal.
    sensor.adapter.camera = None

    sensor_manager = MagicMock()
    sensor_manager.get_sensor.side_effect = lambda sid: sensor if sid == "scope-0" else None
    daemon.sensor_manager = sensor_manager

    runtime = MagicMock()
    runtime.calibration_library = library
    runtime.calibration_manager = None

    dispatcher = MagicMock()
    dispatcher.get_runtime.side_effect = lambda sid: runtime if sid == "scope-0" else None
    daemon.task_dispatcher = dispatcher

    with patch("citrasense.web.app.StaticFiles"):
        web = CitraSenseWebApp(daemon=daemon)
    return TestClient(web.app)


def _upload(
    client: TestClient,
    payload: bytes,
    *,
    frame_type: str,
    normalize_flat: bool = True,
    override_filter: str = "",
    sensor_id: str = "scope-0",
):
    params = {
        "frame_type": frame_type,
        "normalize_flat": str(normalize_flat).lower(),
        "override_filter": override_filter,
    }
    return client.post(
        f"/api/sensors/{sensor_id}/calibration/upload",
        params=params,
        content=payload,
        headers={"Content-Type": "application/fits"},
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestUploadHappyPath:
    def test_bias_ingest_writes_canonical_filename(self, client, library):
        data = np.full((16, 16), 500.0, dtype=np.float32)
        payload = _fits_bytes(data, gain=100, xbinning=1)
        resp = _upload(client, payload, frame_type="bias")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["camera_id"] == "SN-TEST"
        assert body["saved_as"].startswith("master_bias_SN-TEST_g100_bin1_")

        saved = library.get_master_bias("SN-TEST", gain=100, binning=1)
        assert saved is not None
        assert saved.name == body["saved_as"]

    def test_dark_ingest_records_exposure_and_temperature(self, client, library):
        data = np.full((16, 16), 50.0, dtype=np.float32)
        payload = _fits_bytes(data, exptime=30.0, ccd_temp=-10.5)
        resp = _upload(client, payload, frame_type="dark")
        assert resp.status_code == 200, resp.text
        saved = library.get_master_dark("SN-TEST", gain=100, binning=1, temperature=-10.5)
        assert saved is not None
        with fits.open(saved) as hdul:
            hdr = hdul[0].header  # type: ignore[index]
            assert float(hdr["EXPTIME"]) == pytest.approx(30.0)
            assert float(hdr["CCD-TEMP"]) == pytest.approx(-10.5)

    def test_flat_ingest_keyed_by_filter_from_header(self, client, library):
        data = np.full((32, 32), 1.0, dtype=np.float32)
        payload = _fits_bytes(data, filter_name="Luminance")
        resp = _upload(client, payload, frame_type="flat")
        assert resp.status_code == 200, resp.text
        assert library.get_master_flat("SN-TEST", gain=100, binning=1, filter_name="Luminance") is not None

    def test_flat_override_filter_takes_precedence(self, client, library):
        data = np.full((32, 32), 1.0, dtype=np.float32)
        payload = _fits_bytes(data, filter_name="Luminance")
        resp = _upload(client, payload, frame_type="flat", override_filter="Ha")
        assert resp.status_code == 200, resp.text
        assert library.get_master_flat("SN-TEST", gain=100, binning=1, filter_name="Ha") is not None
        assert library.get_master_flat("SN-TEST", gain=100, binning=1, filter_name="Luminance") is None

    def test_flat_normalization_divides_by_median(self, client, library):
        data = np.full((32, 32), 20_000.0, dtype=np.float32)
        payload = _fits_bytes(data, filter_name="Red")
        resp = _upload(client, payload, frame_type="flat", normalize_flat=True)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert any("normalized" in w.lower() for w in body.get("warnings", []))

        saved = library.get_master_flat("SN-TEST", gain=100, binning=1, filter_name="Red")
        assert saved is not None
        with fits.open(saved) as hdul:
            arr = np.asarray(hdul[0].data)  # type: ignore[index]
            assert float(np.median(arr)) == pytest.approx(1.0, rel=1e-4)

    def test_flat_already_normalized_left_alone(self, client, library):
        data = np.full((32, 32), 1.0, dtype=np.float32)
        payload = _fits_bytes(data, filter_name="Blue")
        resp = _upload(client, payload, frame_type="flat", normalize_flat=True)
        assert resp.status_code == 200, resp.text
        # No normalization warning when the flat was already ~1.
        assert not any("normalized" in w.lower() for w in resp.json().get("warnings", []))

    def test_readoutm_header_aliases_to_read_mode(self, client, library):
        data = np.full((16, 16), 10.0, dtype=np.float32)
        # NINA/ASCOM often emits READOUTM instead of READMODE.
        payload = _fits_bytes(data, readoutm="12-bit Slow")
        resp = _upload(client, payload, frame_type="bias")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # read_mode slug is embedded in the canonical filename (spaces → underscores)
        assert "12-bit_Slow" in body["saved_as"]

    def test_camera_id_falls_back_to_instrume(self, client, library):
        data = np.full((16, 16), 500.0, dtype=np.float32)
        payload = _fits_bytes(data, camser="", instrume="ZWO ASI294MM")
        resp = _upload(client, payload, frame_type="bias")
        assert resp.status_code == 200, resp.text
        assert resp.json()["camera_id"] == "ZWO ASI294MM"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestUploadValidation:
    def test_empty_body_rejected(self, client):
        resp = _upload(client, b"", frame_type="bias")
        assert resp.status_code == 400
        assert "empty" in resp.json()["error"].lower()

    def test_unknown_frame_type_rejected(self, client):
        data = np.zeros((8, 8), dtype=np.float32)
        resp = _upload(client, _fits_bytes(data), frame_type="lights")
        assert resp.status_code == 400

    def test_non_fits_body_rejected(self, client):
        resp = _upload(client, b"this is not FITS", frame_type="bias")
        assert resp.status_code == 400

    def test_dark_without_exposure_rejected(self, client):
        data = np.zeros((8, 8), dtype=np.float32)
        resp = _upload(client, _fits_bytes(data, exptime=None), frame_type="dark")
        assert resp.status_code == 400
        assert "EXPTIME" in resp.json()["error"]

    def test_flat_without_filter_rejected(self, client):
        data = np.ones((8, 8), dtype=np.float32)
        resp = _upload(client, _fits_bytes(data, filter_name=None), frame_type="flat")
        assert resp.status_code == 400
        assert "FILTER" in resp.json()["error"]

    def test_flat_override_filter_without_header_accepted(self, client, library):
        data = np.ones((8, 8), dtype=np.float32)
        resp = _upload(
            client,
            _fits_bytes(data, filter_name=None),
            frame_type="flat",
            override_filter="Luminance",
        )
        assert resp.status_code == 200, resp.text
        assert library.get_master_flat("SN-TEST", gain=100, binning=1, filter_name="Luminance") is not None

    def test_non_2d_image_rejected(self, client):
        data = np.zeros((4, 8, 8), dtype=np.float32)
        resp = _upload(client, _fits_bytes(data), frame_type="bias")
        assert resp.status_code == 400
        assert "2D" in resp.json()["error"]

    def test_flat_with_zero_median_rejected(self, client):
        data = np.zeros((8, 8), dtype=np.float32)
        resp = _upload(client, _fits_bytes(data, filter_name="Clear"), frame_type="flat", normalize_flat=True)
        assert resp.status_code == 400
        assert "median" in resp.json()["error"].lower()
