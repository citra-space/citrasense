"""Format an enriched ``pr_sensor`` observation into ``RadarObservationCreate``.

Maps the JSON published on ``radar.sensor.{id}.observations`` onto the
camelCase upload schema the Citra API accepts at
``POST /observations/radar``.  Field mapping follows the table in
issue #307's "Detection → Upload field mapping" section.

Key transforms:

- ``target.citra_uuid`` → ``satelliteId`` (required — observations
  without a UUID are dropped).
- ``geometry.receiver.alt_m`` → ``sensorAltitude`` in **km** (÷ 1000).
- ``geometry.transmitter`` (bistatic) → ``secondarySensor*`` fields.
  The backend's model validator requires all three or none, so we
  only include the transmitter block when lat/lon are both available.
  Altitude is omitted in v1 (``pr_sensor`` doesn't emit it).
- ``geometry.az_deg`` / ``el_deg`` → ``rightAscension`` / ``declination``
  via an astropy ``AltAz → ICRS`` transform anchored at the receiver
  position and the observation epoch.  If the receiver position is
  missing we skip the angular fields and upload range-only data.
- ``bistatic.doppler_hz`` → ``rangeRate`` via the standard
  bistatic Doppler-to-range-rate conversion, using the configured
  ``RadarConfig.center_freq_hz`` (defaults to the observation's
  ``bistatic.doppler_meas`` / ``pr_sensor`` default).

v1 explicitly omits: angular rates, noise estimates, and RCS — all
optional fields.  Future follow-ups can add them without churning
the field mapping itself.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext


#: Speed of light in m/s — used for bistatic Doppler → range-rate.
_SPEED_OF_LIGHT_M_S = 299_792_458.0

#: Default FM carrier frequency assumption (100 MHz) when neither the
#: observation nor the configured radar_config provides one.  Matches
#: the sanity value used in ``pr_common.h``.
_DEFAULT_CARRIER_HZ = 100e6

_module_logger = logging.getLogger("citrasense.RadarDetectionFormatter")


class RadarDetectionFormatter:
    """Build a ``RadarObservationCreate`` payload from an observation event."""

    name = "radar_detection_formatter"

    def __init__(self, *, carrier_freq_hz: float | None = None) -> None:
        """``carrier_freq_hz`` overrides the per-observation Doppler
        → range-rate conversion — useful when every detection in a
        session uses the same illuminator.  Leave ``None`` to derive
        it from the payload on every observation.
        """
        self._carrier_hz = carrier_freq_hz

    def process(self, ctx: RadarProcessingContext) -> bool:
        """Populate ``ctx.upload_payload`` with a camelCase dict.

        Returns ``True`` when a payload was built, ``False`` when the
        observation is unsuitable for upload (missing identifiers /
        receiver position).
        """
        payload = ctx.event.payload
        log = ctx.logger or _module_logger

        target = payload.get("target") or {}
        citra_uuid = target.get("citra_uuid")
        if not citra_uuid:
            ctx.drop_reason = "observation missing target.citra_uuid"
            log.info("Skipping observation: %s", ctx.drop_reason)
            return False

        if not ctx.antenna_id:
            ctx.drop_reason = "citra_antenna_id not configured on sensor"
            log.warning("Skipping observation: %s", ctx.drop_reason)
            return False

        geometry = payload.get("geometry") or {}
        receiver = geometry.get("receiver") or {}
        transmitter = geometry.get("transmitter") or {}
        bistatic = payload.get("bistatic") or {}
        quality = payload.get("quality") or {}

        receiver_lat = _as_float(receiver.get("lat_deg"))
        receiver_lon = _as_float(receiver.get("lon_deg"))
        receiver_alt_m = _as_float(receiver.get("alt_m"))
        if receiver_lat is None or receiver_lon is None:
            ctx.drop_reason = "observation missing receiver lat/lon (required by backend)"
            log.warning("Skipping observation: %s", ctx.drop_reason)
            return False

        entry: dict[str, Any] = {
            "satelliteId": citra_uuid,
            "antennaId": ctx.antenna_id,
            "epoch": _ensure_iso8601(payload.get("timestamp"), ctx.event.timestamp),
            "sensorLatitude": receiver_lat,
            "sensorLongitude": receiver_lon,
        }

        if receiver_alt_m is not None:
            # Backend expects **km** (per issue #307).  pr_sensor emits metres.
            entry["sensorAltitude"] = receiver_alt_m / 1000.0
        else:
            entry["sensorAltitude"] = 0.0

        # ── Bistatic range ────────────────────────────────────────────
        bistatic_range = _as_float(bistatic.get("bistatic_range_km"))
        if bistatic_range is None:
            bistatic_range = _as_float(payload.get("range_km"))
        if bistatic_range is not None:
            entry["range"] = bistatic_range

        # ── Range rate (derived from Doppler) ────────────────────────
        doppler_hz = _as_float(bistatic.get("doppler_hz"))
        if doppler_hz is None:
            doppler_hz = _as_float(bistatic.get("doppler_meas"))
        if doppler_hz is not None:
            carrier = self._resolve_carrier_hz(payload, transmitter)
            if carrier and carrier > 0:
                # Bistatic Doppler: f_D = -(dR/dt) * f_c / c  →  dR/dt = -f_D * c / f_c
                # Result in m/s; divide by 1000 for km/s (backend unit).
                range_rate_mps = -doppler_hz * _SPEED_OF_LIGHT_M_S / carrier
                entry["rangeRate"] = range_rate_mps / 1000.0

        # ── SNR ──────────────────────────────────────────────────────
        snr_db = _as_float(quality.get("snr_db"))
        if snr_db is None:
            snr_db = _as_float(payload.get("snr_db"))
        if snr_db is not None:
            entry["snr"] = snr_db

        # ── Angular coordinates (RA/Dec) ─────────────────────────────
        az_deg = _as_float(geometry.get("az_deg"))
        el_deg = _as_float(geometry.get("el_deg"))
        if az_deg is not None and el_deg is not None:
            ra_dec = _azel_to_radec(
                az_deg=az_deg,
                el_deg=el_deg,
                lat_deg=receiver_lat,
                lon_deg=receiver_lon,
                alt_m=receiver_alt_m or 0.0,
                epoch=entry["epoch"],
                log=log,
            )
            if ra_dec is not None:
                entry["rightAscension"], entry["declination"] = ra_dec

        # ── Bistatic transmitter (all-or-nothing per backend) ────────
        tx_lat = _as_float(transmitter.get("lat_deg"))
        tx_lon = _as_float(transmitter.get("lon_deg"))
        if tx_lat is not None and tx_lon is not None and not (tx_lat == 0.0 and tx_lon == 0.0):
            # Altitude not emitted by pr_sensor today — we treat the
            # transmitter as sea-level for v1 rather than drop the
            # bistatic block (the backend's validator requires all
            # three fields when any is present).
            entry["secondarySensorLatitude"] = tx_lat
            entry["secondarySensorLongitude"] = tx_lon
            entry["secondarySensorAltitude"] = 0.0

        ctx.upload_payload = entry
        ctx.drop_reason = None
        return True

    def _resolve_carrier_hz(self, payload: dict, transmitter: dict) -> float | None:
        if self._carrier_hz:
            return self._carrier_hz
        # Observations enrich with transmitter info including freq_hz
        freq = _as_float(transmitter.get("freq_hz"))
        if freq:
            return freq
        per_station = payload.get("per_station") or []
        for ps in per_station:
            freq = _as_float(ps.get("freq_hz"))
            if freq:
                return freq
        return _DEFAULT_CARRIER_HZ


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _ensure_iso8601(value: Any, fallback: datetime) -> str:
    """Return a Z-terminated UTC ISO-8601 string.

    ``pr_sensor`` emits ``iso_now()`` which is always Z-suffixed; we
    normalise for defensive consistency and fall back to the event's
    pre-parsed ``timestamp`` attribute if the raw payload field is
    missing or malformed.
    """
    if isinstance(value, str) and value:
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            return cleaned
        # Replace "+00:00" with "Z" for compactness; leave other offsets alone.
        if cleaned.endswith("+00:00"):
            return cleaned[: -len("+00:00")] + "Z"
        return cleaned
    fb = fallback.astimezone(timezone.utc) if fallback.tzinfo else fallback.replace(tzinfo=timezone.utc)
    return fb.isoformat().replace("+00:00", "Z")


def _azel_to_radec(
    *,
    az_deg: float,
    el_deg: float,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    epoch: str,
    log: Any,
) -> tuple[float, float] | None:
    """Transform local alt/az to ICRS RA/Dec via astropy.

    Imported lazily because astropy is already in the optical critical
    path; deferring the import keeps the radar cold-start fast on
    systems that don't run radar.
    """
    try:
        from astropy import units as u  # type: ignore[import-not-found]
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
    except Exception as exc:
        log.warning("astropy not importable; skipping RA/Dec conversion: %s", exc)
        return None

    try:
        location = EarthLocation(
            lat=lat_deg * u.deg,  # type: ignore[attr-defined]
            lon=lon_deg * u.deg,  # type: ignore[attr-defined]
            height=alt_m * u.m,  # type: ignore[attr-defined]
        )
        obstime = Time(epoch)
        altaz_frame = AltAz(obstime=obstime, location=location)
        coord = SkyCoord(
            alt=el_deg * u.deg,  # type: ignore[attr-defined]
            az=az_deg * u.deg,  # type: ignore[attr-defined]
            frame=altaz_frame,
        )
        icrs = coord.icrs
        return float(icrs.ra.deg), float(icrs.dec.deg)  # type: ignore[attr-defined,union-attr]
    except Exception as exc:
        log.warning("AltAz→ICRS conversion failed: %s", exc)
        return None
