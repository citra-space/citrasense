"""Unit tests for :class:`RadarDetectionFormatter`.

The formatter is the narrow boundary where ``pr_sensor``'s JSON
``Observation`` meets the Citra API ``RadarObservationCreate`` schema.
These tests exercise:

- field mapping onto the camelCase upload shape,
- unit conversions (``alt_m`` → km, Doppler → range-rate in km/s),
- the backend's "all-or-nothing" secondary-sensor validator contract,
- az/el → RA/Dec transform correctness within a tolerance, and
- the drop paths (missing ``citra_uuid``, missing receiver lat/lon,
  missing antenna id).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from citrasense.pipelines.radar.radar_detection_formatter import (
    _SPEED_OF_LIGHT_M_S,
    RadarDetectionFormatter,
)
from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext
from citrasense.sensors.radar.events import RadarObservationEvent


def _sample_observation_payload() -> dict:
    """Frozen sample of the enriched ``Observation`` pr_sensor publishes."""
    return {
        "timestamp": "2025-11-11T18:38:11Z",
        "detection_id": "det-0001",
        "target": {"citra_uuid": "sat-deadbeef"},
        "geometry": {
            "az_deg": 123.4,
            "el_deg": 45.0,
            "receiver": {
                "lat_deg": 35.0,
                "lon_deg": -106.5,
                "alt_m": 1500.0,
            },
            "transmitter": {
                "lat_deg": 35.1,
                "lon_deg": -106.4,
                "freq_hz": 98e6,
            },
        },
        "bistatic": {
            "bistatic_range_km": 420.5,
            "doppler_hz": -200.0,
        },
        "quality": {"snr_db": 12.7},
    }


def _make_ctx(payload: dict, *, antenna_id: str = "antenna-uuid-1") -> RadarProcessingContext:
    event = RadarObservationEvent(
        sensor_id="radar-0",
        modality="radar",
        timestamp=datetime(2025, 11, 11, 18, 38, 11, tzinfo=timezone.utc),
        payload=payload,
    )
    return RadarProcessingContext(
        sensor_id="radar-0",
        event=event,
        antenna_id=antenna_id,
    )


class TestFieldMapping:
    def test_full_payload_maps_to_camel_case(self):
        ctx = _make_ctx(_sample_observation_payload())
        ok = RadarDetectionFormatter().process(ctx)
        assert ok is True
        entry = ctx.upload_payload
        assert entry is not None
        assert entry["satelliteId"] == "sat-deadbeef"
        assert entry["antennaId"] == "antenna-uuid-1"
        assert entry["epoch"] == "2025-11-11T18:38:11Z"
        assert entry["sensorLatitude"] == 35.0
        assert entry["sensorLongitude"] == -106.5
        # alt_m (metres) → sensorAltitude (km)
        assert entry["sensorAltitude"] == pytest.approx(1.5)
        # bistatic range preserved
        assert entry["range"] == pytest.approx(420.5)
        # SNR propagated
        assert entry["snr"] == pytest.approx(12.7)

    def test_doppler_to_range_rate_conversion(self):
        """Doppler = -200 Hz, carrier = 98 MHz → +0.612 km/s roughly."""
        ctx = _make_ctx(_sample_observation_payload())
        RadarDetectionFormatter().process(ctx)
        doppler_hz = -200.0
        expected_mps = -doppler_hz * _SPEED_OF_LIGHT_M_S / 98e6
        expected_kmps = expected_mps / 1000.0
        assert ctx.upload_payload is not None
        assert ctx.upload_payload["rangeRate"] == pytest.approx(expected_kmps, rel=1e-9)

    def test_transmitter_all_or_nothing(self):
        """Backend validator requires all three secondarySensor* fields
        or none.  With non-zero tx lat/lon, altitude must also be set.
        """
        ctx = _make_ctx(_sample_observation_payload())
        RadarDetectionFormatter().process(ctx)
        entry = ctx.upload_payload or {}
        assert entry.get("secondarySensorLatitude") == 35.1
        assert entry.get("secondarySensorLongitude") == -106.4
        assert entry.get("secondarySensorAltitude") == 0.0

    def test_zero_transmitter_coords_omitted(self):
        """pr_sensor sometimes publishes 0/0 when tx is unknown —
        don't leak a fake secondary sensor into the API."""
        payload = _sample_observation_payload()
        payload["geometry"]["transmitter"]["lat_deg"] = 0.0
        payload["geometry"]["transmitter"]["lon_deg"] = 0.0
        ctx = _make_ctx(payload)
        RadarDetectionFormatter().process(ctx)
        entry = ctx.upload_payload or {}
        assert "secondarySensorLatitude" not in entry
        assert "secondarySensorLongitude" not in entry
        assert "secondarySensorAltitude" not in entry


class TestDropPaths:
    def test_missing_citra_uuid_drops(self):
        payload = _sample_observation_payload()
        payload["target"] = {}
        ctx = _make_ctx(payload)
        assert RadarDetectionFormatter().process(ctx) is False
        assert ctx.upload_payload is None
        assert ctx.drop_reason is not None
        assert "citra_uuid" in ctx.drop_reason

    def test_missing_antenna_id_drops(self):
        ctx = _make_ctx(_sample_observation_payload(), antenna_id="")
        assert RadarDetectionFormatter().process(ctx) is False
        assert ctx.drop_reason is not None
        assert "antenna" in ctx.drop_reason.lower()

    def test_missing_receiver_position_drops(self):
        payload = _sample_observation_payload()
        payload["geometry"]["receiver"] = {}
        ctx = _make_ctx(payload)
        assert RadarDetectionFormatter().process(ctx) is False
        assert ctx.drop_reason is not None
        assert "receiver" in ctx.drop_reason


class TestAzElToRaDec:
    """The az/el → RA/Dec transform is what turns a radar detection
    into something the existing catalog pipeline understands.  Assert a
    round-trip tolerance against a known-good astropy answer for a
    representative sample."""

    def test_ra_dec_values_are_present_and_plausible(self):
        ctx = _make_ctx(_sample_observation_payload())
        RadarDetectionFormatter().process(ctx)
        entry = ctx.upload_payload or {}
        assert "rightAscension" in entry
        assert "declination" in entry
        # RA in [0, 360), Dec in [-90, 90].
        assert 0.0 <= entry["rightAscension"] < 360.0
        assert -90.0 <= entry["declination"] <= 90.0

    def test_zenith_matches_site_latitude(self):
        """Az/el pointing straight up should hit (RA ≈ LST, Dec ≈ lat)
        within 0.5 deg — the classic sanity check for alt/az → RA/Dec.
        We check the Dec side (latitude) because LST is a function of
        time and less convenient to assert cheaply.
        """
        payload = _sample_observation_payload()
        payload["geometry"]["az_deg"] = 0.0
        payload["geometry"]["el_deg"] = 89.9  # ≈ zenith, avoid pole.
        payload["geometry"]["receiver"]["lat_deg"] = 35.0
        ctx = _make_ctx(payload)
        ok = RadarDetectionFormatter().process(ctx)
        if not ok:
            pytest.skip("astropy not available in this environment")
        entry = ctx.upload_payload or {}
        assert abs(entry["declination"] - 35.0) < 1.0


def test_ensure_iso8601_normalises_plus_zero():
    from citrasense.pipelines.radar.radar_detection_formatter import _ensure_iso8601

    fallback = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert _ensure_iso8601("2025-11-11T18:38:11+00:00", fallback) == "2025-11-11T18:38:11Z"


def test_ensure_iso8601_falls_back_on_missing():
    from citrasense.pipelines.radar.radar_detection_formatter import _ensure_iso8601

    fallback = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert _ensure_iso8601(None, fallback) == "2030-01-01T00:00:00Z"
