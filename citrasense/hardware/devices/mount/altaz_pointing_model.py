"""Alt-az pointing model from plate-solve feedback.

Fits the standard 5-term alt-az mount error model to calibration data
collected via plate solving at multiple sky positions.  Applies corrections
to goto commands so the mount lands closer to the intended target.

Error model (alt-az terms)::

    dAz  = CA·sec(alt) + NPAE·tan(alt) + AN·sin(az)·tan(alt) − AW·cos(az)·tan(alt)
    dAlt = IE − AN·cos(az) − AW·sin(az)

Terms:
    AN   — Azimuth axis tilt in N-S direction (leveling error)
    AW   — Azimuth axis tilt in E-W direction (leveling error)
    IE   — Index error in elevation (altitude zero-point offset)
    CA   — Collimation error (optical axis vs altitude axis)
    NPAE — Non-perpendicularity of altitude and azimuth axes

Sign convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The error terms describe the **optical displacement from the encoder
position**.  Concretely:

    optics_az  = encoder_az  + dAz
    optics_alt = encoder_alt + dAlt

``add_point()`` records ``solved − mount`` (plate-solved optics position
minus mount encoder position), which equals ``(dAz, dAlt)`` — a positive
value when the optics overshoot the encoder.

``correct()`` precompensates by *subtracting* the predicted error::

    command_az  = target_az  − dAz
    command_alt = target_alt − dAlt

The mount drives to ``(target − dAz)``, the optics end up at
``(target − dAz) + dAz = target``.  If the signs in ``add_point`` are
reversed, the model terms flip sign and ``correct()`` doubles the error
instead of cancelling it.

Graceful degradation:
    0-2 points  → passthrough (no correction)
    3-7 points  → 3-term fit (AN, AW, IE)
    8+  points  → full 5-term fit
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from citrasense.astro.sidereal import gast_degrees

_logger = logging.getLogger("citrasense.AltAzPointingModel")


_MIN_POINTS_3TERM = 3
_MIN_POINTS_5TERM = 8
_MIN_AZ_SPREAD_5TERM = 90.0  # degrees — need this much azimuth diversity for 5-term
_SIGMA_CLIP_THRESHOLD = 2.5
_SIGMA_CLIP_MAX_ITER = 3
_SIGMA_CLIP_FLOOR_DEG = 1.0 / 60.0  # never clip points with < 1 arcmin residual
_HEALTH_WINDOW = 5
_HEALTH_DEGRADED_FACTOR = 3.0
_LIVE_ACCURACY_WINDOW = 100
_NEARBY_POINT_MIN_SEP = 1.0  # degrees — operational feeding guard


# ---------------------------------------------------------------------------
# Coordinate conversion utilities
# ---------------------------------------------------------------------------


def lst_deg(longitude_deg: float, *, _gast_override: float | None = None) -> float:
    """Local Sidereal Time in degrees for the given longitude.

    Args:
        longitude_deg: Observer longitude in degrees.
        _gast_override: If provided, use this GAST (degrees) instead of
            computing a fresh one.  Callers that need two conversions at
            the same instant should capture ``gast_degrees()`` once and
            pass it to both calls.
    """
    gast = _gast_override if _gast_override is not None else gast_degrees()
    return (gast + longitude_deg) % 360.0


def radec_to_altaz(
    ra_deg: float,
    dec_deg: float,
    lat_deg: float,
    lon_deg: float,
    *,
    _gast_override: float | None = None,
) -> tuple[float, float]:
    """Convert RA/Dec to (azimuth, altitude) in degrees.

    Uses standard spherical trigonometry with LST derived from
    :func:`~citrasense.astro.sidereal.gast_degrees` (true apparent sidereal
    time via astropy/ERFA, IAU 2006/2000A).

    Args:
        ra_deg: Right Ascension in degrees.
        dec_deg: Declination in degrees.
        lat_deg: Observer latitude in degrees.
        lon_deg: Observer longitude in degrees.
        _gast_override: Frozen GAST in degrees for paired conversions.

    Returns:
        (azimuth_deg, altitude_deg) — azimuth measured from north through east.
    """
    lat = math.radians(lat_deg)
    local_lst = lst_deg(lon_deg, _gast_override=_gast_override)
    ha = math.radians((local_lst - ra_deg) % 360.0)
    dec = math.radians(dec_deg)

    sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(ha)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))

    cos_alt = math.cos(alt)
    if cos_alt < 1e-10:
        return 0.0, math.degrees(alt)

    cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (cos_alt * math.cos(lat) + 1e-10)
    az_raw = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    az = (360.0 - az_raw) if math.sin(ha) > 0 else az_raw

    return az, math.degrees(alt)


def altaz_to_radec(
    az_deg: float,
    alt_deg: float,
    lat_deg: float,
    lon_deg: float,
    *,
    _gast_override: float | None = None,
) -> tuple[float, float]:
    """Convert (azimuth, altitude) to RA/Dec in degrees.

    Inverse of :func:`radec_to_altaz`.

    Args:
        az_deg: Azimuth in degrees (north=0, east=90).
        alt_deg: Altitude in degrees.
        lat_deg: Observer latitude in degrees.
        lon_deg: Observer longitude in degrees.
        _gast_override: Frozen GAST in degrees for paired conversions.

    Returns:
        (ra_deg, dec_deg).
    """
    lat = math.radians(lat_deg)
    az = math.radians(az_deg)
    alt = math.radians(alt_deg)

    sin_dec = math.sin(alt) * math.sin(lat) + math.cos(alt) * math.cos(lat) * math.cos(az)
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))

    cos_dec = math.cos(dec)
    if cos_dec < 1e-10:
        return lst_deg(lon_deg, _gast_override=_gast_override), math.degrees(dec)

    cos_ha = (math.sin(alt) - math.sin(dec) * math.sin(lat)) / (cos_dec * math.cos(lat) + 1e-10)
    ha_abs = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    ha = -ha_abs if math.sin(az) > 0 else ha_abs

    ra = (lst_deg(lon_deg, _gast_override=_gast_override) - ha) % 360.0
    return ra, math.degrees(dec)


# ---------------------------------------------------------------------------
# Calibration grid generation
# ---------------------------------------------------------------------------


def generate_calibration_grid(
    current_az_deg: float,
    cable_wrap_cumulative_deg: float,
    horizon_limit_deg: float = 15.0,
    overhead_limit_deg: float = 89.0,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    n_points: int = 15,
    cable_wrap_soft_limit_deg: float = 240.0,
) -> list[tuple[float, float]]:
    """Generate well-distributed sky positions for a seed calibration.

    Returns an ordered list of ``(ra_deg, dec_deg)`` targets that:
    - Stay within mount altitude limits (drops 75° band — tan(75)=3.73
      amplifies azimuth noise by nearly 4x)
    - Respect cable-wrap budget (asymmetric CW/CCW allocation)
    - Use serpentine (boustrophedon) ordering that sweeps azimuth in one
      direction per altitude band, reversing for the next.  This keeps
      net cable-wrap accumulation near zero.

    Cable wrap budget is **asymmetric**: unwinding the cable gives far
    more range than further winding.  The grid extends more in the
    unwinding direction to maximize sky coverage.

    Args:
        current_az_deg: Mount's current azimuth in degrees.
        cable_wrap_cumulative_deg: Current cable-wrap cumulative rotation
            (positive = CW winding from neutral).
        horizon_limit_deg: Minimum altitude above horizon.
        overhead_limit_deg: Maximum altitude (avoid zenith singularity).
        lat_deg: Observer latitude in degrees.
        lon_deg: Observer longitude in degrees.
        n_points: Desired number of calibration points (default 15).
        cable_wrap_soft_limit_deg: Cable-wrap soft limit for budget calculation.

    Returns:
        Ordered list of (ra_deg, dec_deg) targets.
    """
    # Asymmetric cable wrap budget: CW = further winding, CCW = unwinding
    cw_budget = cable_wrap_soft_limit_deg - cable_wrap_cumulative_deg
    ccw_budget = cable_wrap_soft_limit_deg + cable_wrap_cumulative_deg
    total_budget = max(cw_budget, 0.0) + max(ccw_budget, 0.0)

    if total_budget < 60.0:
        _logger.warning(
            "Cable wrap budget very limited (CW: %.0f°, CCW: %.0f°) — calibration grid will be narrow.",
            max(cw_budget, 0.0),
            max(ccw_budget, 0.0),
        )

    usable_range = min(total_budget * 0.8, 360.0)

    if total_budget > 0:
        frac_ccw = max(ccw_budget, 0.0) / total_budget
    else:
        frac_ccw = 0.5
    range_ccw = usable_range * frac_ccw
    range_cw = usable_range * (1.0 - frac_ccw)

    _logger.info(
        "Calibration grid: %.0f° usable range (CW %.0f°, CCW %.0f° from az %.0f°)",
        usable_range,
        range_cw,
        range_ccw,
        current_az_deg,
    )

    alt_bands = [alt for alt in [30.0, 45.0, 60.0] if horizon_limit_deg <= alt <= overhead_limit_deg]
    if not alt_bands:
        alt_bands = [(horizon_limit_deg + min(overhead_limit_deg, 65.0)) / 2.0]

    n_az = max(3, n_points // len(alt_bands))
    if usable_range >= 360.0:
        az_step = 360.0 / n_az
    else:
        az_step = usable_range / max(n_az - 1, 1)

    # Start near current_az and step toward the CCW end.  The first
    # band sweeps CW→CCW, the second reverses CCW→CW, etc.  This
    # serpentine pattern keeps cumulative cable wrap near zero.
    cw_end = current_az_deg + range_cw
    base_positions = [(cw_end - j * az_step) % 360.0 for j in range(n_az)]

    grid_altaz: list[tuple[float, float]] = []
    for i, alt in enumerate(alt_bands):
        az_positions = list(base_positions)
        if i % 2 == 1:
            az_positions.reverse()
        for az in az_positions:
            grid_altaz.append((az, alt))

    if len(grid_altaz) > n_points:
        step = len(grid_altaz) / n_points
        grid_altaz = [grid_altaz[int(i * step)] for i in range(n_points)]

    targets: list[tuple[float, float]] = []
    for az, alt in grid_altaz:
        ra, dec = altaz_to_radec(az, alt, lat_deg, lon_deg)
        targets.append((ra, dec))

    return targets


# ---------------------------------------------------------------------------
# Pointing model
# ---------------------------------------------------------------------------


class AltAzPointingModel:
    """5-term alt-az pointing error model with least-squares fitting.

    Owns calibration data, fitted terms, and health monitoring state.
    Persists to a JSON state file (like ``CableWrapCheck``).

    Data flow:
        ``add_point(encoder, plate_solved)``  →  ``fit()``  →  ``correct(target)``

    All stored residuals use the convention **solved − mount** (optics
    minus encoder).  See the module docstring for the full sign convention.
    """

    def __init__(
        self,
        state_file: Path | None = None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._state_file = state_file
        self._lock = threading.Lock()
        self._log: logging.Logger | logging.LoggerAdapter = logger if logger is not None else _logger

        # Calibration data: list of (az, alt, d_az, d_alt) in degrees
        self._points: list[tuple[float, float, float, float]] = []

        # Fitted terms (degrees)
        self._AN: float = 0.0
        self._AW: float = 0.0
        self._IE: float = 0.0
        self._CA: float = 0.0
        self._NPAE: float = 0.0

        self._rms_deg: float = 0.0
        self._fit_timestamp: float | None = None
        self._n_terms: int = 0

        # Health monitoring: rolling window of recent verification residuals
        self._recent_residuals: deque[float] = deque(maxlen=_HEALTH_WINDOW)
        self._health: str = "unknown"  # "good", "degraded", "unknown"

        # Live accuracy tracking: wider window for the UI
        self._live_residuals: deque[tuple[float, float]] = deque(maxlen=_LIVE_ACCURACY_WINDOW)  # (timestamp, deg)

        self._load_state()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True when the full 5-term model has been fitted (8+ points)."""
        return self._n_terms == 5

    @property
    def is_active(self) -> bool:
        """True when any correction is available (3+ points fitted)."""
        return self._n_terms >= 3

    @property
    def point_count(self) -> int:
        return len(self._points)

    @property
    def n_terms(self) -> int:
        return self._n_terms

    @property
    def rms_deg(self) -> float:
        return self._rms_deg

    @property
    def health(self) -> str:
        return self._health

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_point(
        self,
        mount_ra: float,
        mount_dec: float,
        solved_ra: float,
        solved_dec: float,
        site_lat: float,
        site_lon: float,
    ) -> None:
        """Record a (mount-reported, plate-solved) calibration pair.

        Converts both positions to alt/az and stores the mount error:
        ``solved - mount``, i.e. the displacement of the optics from the
        encoder position.  This matches the sign convention used by
        ``correct()`` which *subtracts* the predicted error from the
        target coordinates.

        Automatically triggers a refit when enough points are available.

        Args:
            mount_ra: Mount-reported (encoder) RA in degrees.
            mount_dec: Mount-reported (encoder) Dec in degrees.
            solved_ra: Plate-solved (true optics) RA in degrees.
            solved_dec: Plate-solved (true optics) Dec in degrees.
            site_lat: Observer latitude in degrees.
            site_lon: Observer longitude in degrees.
        """
        gast = gast_degrees()
        mount_az, mount_alt = radec_to_altaz(mount_ra, mount_dec, site_lat, site_lon, _gast_override=gast)
        solved_az, solved_alt = radec_to_altaz(solved_ra, solved_dec, site_lat, site_lon, _gast_override=gast)

        d_az = solved_az - mount_az
        if d_az > 180.0:
            d_az -= 360.0
        elif d_az < -180.0:
            d_az += 360.0

        d_alt = solved_alt - mount_alt

        with self._lock:
            self._points.append((mount_az, mount_alt, d_az, d_alt))
            n_points = len(self._points)

        self._log.info(
            "Pointing model: added point #%d — az=%.1f° alt=%.1f° dAz=%.4f° dAlt=%.4f°",
            n_points,
            mount_az,
            mount_alt,
            d_az,
            d_alt,
        )

        if n_points >= _MIN_POINTS_3TERM:
            self.fit()

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Fit error terms to accumulated calibration points via least-squares.

        Selects 3-term (AN, AW, IE) or 5-term (+ CA, NPAE) based on point
        count AND azimuth diversity.  The 5-term fit requires >= 90° of
        azimuth spread; with less, the terms become degenerate and the fit
        distributes error into CA/NPAE incorrectly.

        Uses iterative sigma-clipping (2.5-sigma, up to 3 rounds) to reject
        outliers.  Clipped points are excluded from the fit only — raw data
        in ``self._points`` is preserved for diagnostics and persistence.
        """
        with self._lock:
            n = len(self._points)
            if n < _MIN_POINTS_3TERM:
                self._log.info("Pointing model: only %d points, need %d for fit", n, _MIN_POINTS_3TERM)
                return

            points_snapshot = list(self._points)

        az_values = [p[0] for p in points_snapshot]
        az_spread = self._azimuth_spread(az_values)
        use_5term = n >= _MIN_POINTS_5TERM and az_spread >= _MIN_AZ_SPREAD_5TERM
        if n >= _MIN_POINTS_5TERM and not use_5term:
            self._log.info(
                "Azimuth spread %.0f° < %.0f° minimum — staying with 3-term fit despite %d points",
                az_spread,
                _MIN_AZ_SPREAD_5TERM,
                n,
            )

        active = list(range(len(points_snapshot)))
        total_clipped = 0
        # Need spare degrees of freedom for sigma-clipping to be meaningful
        min_for_clip = (_MIN_POINTS_5TERM if use_5term else _MIN_POINTS_3TERM) + 3
        A = np.empty((0, 0))
        b = np.empty(0)
        result = np.empty(0)

        for clip_iter in range(_SIGMA_CLIP_MAX_ITER + 1):
            if len(active) < _MIN_POINTS_3TERM:
                self._log.warning("Sigma clip: too few points remaining (%d) — aborting clip", len(active))
                break

            if use_5term and len(active) < _MIN_POINTS_5TERM:
                use_5term = False
                self._log.info("Sigma clip reduced points below %d — downgrading to 3-term", _MIN_POINTS_5TERM)

            rows_az: list[list[float]] = []
            rows_alt: list[list[float]] = []
            obs_az: list[float] = []
            obs_alt: list[float] = []

            for idx in active:
                az_deg, alt_deg, d_az, d_alt = points_snapshot[idx]
                az = math.radians(az_deg)
                alt = math.radians(alt_deg)

                sin_az = math.sin(az)
                cos_az = math.cos(az)
                tan_alt = math.tan(alt) if abs(math.cos(alt)) > 1e-10 else 0.0
                sec_alt = 1.0 / math.cos(alt) if abs(math.cos(alt)) > 1e-10 else 0.0

                if use_5term:
                    rows_az.append([sin_az * tan_alt, -cos_az * tan_alt, 0.0, sec_alt, tan_alt])
                    rows_alt.append([-cos_az, -sin_az, 1.0, 0.0, 0.0])
                else:
                    rows_az.append([sin_az * tan_alt, -cos_az * tan_alt, 0.0])
                    rows_alt.append([-cos_az, -sin_az, 1.0])
                obs_az.append(d_az)
                obs_alt.append(d_alt)

            A = np.array(rows_az + rows_alt)
            b = np.array(obs_az + obs_alt)

            result, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)

            if clip_iter == _SIGMA_CLIP_MAX_ITER or len(active) < min_for_clip:
                break

            predicted = A @ result
            fit_residuals = b - predicted
            n_active = len(active)
            resid_az = fit_residuals[:n_active]
            resid_alt = fit_residuals[n_active:]

            sky_resid = np.empty(n_active)
            for k in range(n_active):
                alt_deg = points_snapshot[active[k]][1]
                cos_alt = math.cos(math.radians(alt_deg))
                sky_resid[k] = math.sqrt((resid_az[k] * cos_alt) ** 2 + resid_alt[k] ** 2)

            sigma = float(np.std(sky_resid)) if len(sky_resid) > 1 else 0.0
            if sigma < 1e-12:
                break

            threshold = max(_SIGMA_CLIP_THRESHOLD * sigma, _SIGMA_CLIP_FLOOR_DEG)
            new_active: list[int] = []
            for k, idx in enumerate(active):
                if sky_resid[k] <= threshold:
                    new_active.append(idx)
                else:
                    az_deg, alt_deg = points_snapshot[idx][0], points_snapshot[idx][1]
                    self._log.info(
                        "Sigma clip: rejected point #%d (az=%.0f° alt=%.0f°, residual=%.4f° > %.4f° threshold)",
                        idx + 1,
                        az_deg,
                        alt_deg,
                        sky_resid[k],
                        threshold,
                    )

            clipped_this_round = len(active) - len(new_active)
            total_clipped += clipped_this_round
            if clipped_this_round == 0:
                break
            active = new_active

        predicted = A @ result
        fit_residuals = b - predicted
        n_active = len(active)
        resid_az_final = fit_residuals[:n_active]
        resid_alt_final = fit_residuals[n_active:]
        sky_resid_final = np.empty(n_active)
        for k in range(n_active):
            alt_deg_k = points_snapshot[active[k]][1]
            cos_alt_k = math.cos(math.radians(alt_deg_k))
            sky_resid_final[k] = math.sqrt((resid_az_final[k] * cos_alt_k) ** 2 + resid_alt_final[k] ** 2)
        rms_deg = float(np.sqrt(np.mean(sky_resid_final**2))) if n_active > 0 else 0.0

        with self._lock:
            self._AN = float(result[0])
            self._AW = float(result[1])
            self._IE = float(result[2])
            if use_5term:
                self._CA = float(result[3])
                self._NPAE = float(result[4])
                self._n_terms = 5
            else:
                self._CA = 0.0
                self._NPAE = 0.0
                self._n_terms = 3

            self._rms_deg = rms_deg
            self._fit_timestamp = time.time()
            # Health state is owned by record_verification_residual — don't
            # reset it here.  fit() now runs on every solve (add *or* replace),
            # so clearing _recent_residuals would defang the health monitor
            # by preventing the rolling window from ever filling.  See PR #295.

            tilt_mag = math.sqrt(self._AN**2 + self._AW**2)
            tilt_dir = math.degrees(math.atan2(self._AW, self._AN)) % 360.0

        self._log.info(
            "Pointing model fit (%d-term, %d used, %d clipped): "
            "AN=%.4f° AW=%.4f° IE=%.4f° CA=%.4f° NPAE=%.4f° "
            "| tilt=%.3f° toward %.0f° | RMS=%.4f°",
            self._n_terms,
            len(active),
            total_clipped,
            self._AN,
            self._AW,
            self._IE,
            self._CA,
            self._NPAE,
            tilt_mag,
            tilt_dir,
            self._rms_deg,
        )

        self._save_state()

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct(
        self,
        ra_deg: float,
        dec_deg: float,
        site_lat: float,
        site_lon: float,
    ) -> tuple[float, float]:
        """Apply pointing model correction to a goto target.

        Returns corrected (ra, dec) that should be sent to the mount.
        Returns the original coordinates unchanged if the model is not active.

        Args:
            ra_deg: Target RA in degrees.
            dec_deg: Target Dec in degrees.
            site_lat: Observer latitude in degrees.
            site_lon: Observer longitude in degrees.

        Returns:
            (corrected_ra_deg, corrected_dec_deg).
        """
        if not self.is_active:
            return ra_deg, dec_deg

        gast = gast_degrees()
        az, alt = radec_to_altaz(ra_deg, dec_deg, site_lat, site_lon, _gast_override=gast)

        with self._lock:
            d_az, d_alt = self._predict_error_altaz(az, alt)

        corrected_az = az - d_az
        corrected_alt = alt - d_alt
        corrected_ra, corrected_dec = altaz_to_radec(
            corrected_az, corrected_alt, site_lat, site_lon, _gast_override=gast
        )

        return corrected_ra, corrected_dec

    def is_replacement_flyer(
        self,
        observed_residual_deg: float,
        ra_deg: float,
        dec_deg: float,
        site_lat: float,
        site_lon: float,
    ) -> bool:
        """Return True when ``observed_residual_deg`` is implausibly large vs.
        the model's own prediction at this sky position — i.e. the new solve
        looks like a flyer (trail contamination, edge-of-field extraction,
        wrong plate solution) and should not overwrite an established
        calibration point.

        Threshold is based on the local predicted error at this sky
        position: ``max(predict_error(...) * _HEALTH_DEGRADED_FACTOR,
        10 arcmin)``.  Reuses the same 3x factor and 10-arcmin floor as the
        health-degraded gate in ``record_verification_residual`` — the two
        checks share a language but measure different things (this is
        point-local via ``predict_error``; the health gate is global via
        ``rms_deg``).  Returns ``False`` when the model is not yet active
        (no prediction available) so seed calibration data is never
        rejected.

        This is advisory — callers (operational feeders) decide whether to
        skip replacement when it returns ``True``.  ``add_point`` is
        untouched; a flyer in a *new* cell still gets recorded and sigma-clip
        in ``fit()`` handles it.
        """
        if not self.is_active:
            return False
        predicted = self.predict_error(ra_deg, dec_deg, site_lat, site_lon)
        threshold = max(predicted * _HEALTH_DEGRADED_FACTOR, 10.0 / 60.0)
        return observed_residual_deg > threshold

    def predict_error(
        self,
        ra_deg: float,
        dec_deg: float,
        site_lat: float,
        site_lon: float,
    ) -> float:
        """Predicted pointing error magnitude in degrees at the given position.

        Returns ``sqrt((dAz * cos(alt))^2 + dAlt^2)`` so the azimuth
        component is projected onto the sky.  Used by callers to compare
        against observed residuals for health checks.
        Returns 0.0 if the model is not active.
        """
        if not self.is_active:
            return 0.0
        az, alt = radec_to_altaz(ra_deg, dec_deg, site_lat, site_lon)
        with self._lock:
            d_az, d_alt = self._predict_error_altaz(az, alt)
        cos_alt = math.cos(math.radians(alt))
        return math.sqrt((d_az * cos_alt) ** 2 + d_alt**2)

    def _predict_error_altaz(self, az_deg: float, alt_deg: float) -> tuple[float, float]:
        """Predicted (dAz, dAlt) error in degrees at the given alt/az."""
        az = math.radians(az_deg)
        alt = math.radians(alt_deg)

        sin_az = math.sin(az)
        cos_az = math.cos(az)
        tan_alt = math.tan(alt) if abs(math.cos(alt)) > 1e-10 else 0.0
        sec_alt = 1.0 / math.cos(alt) if abs(math.cos(alt)) > 1e-10 else 0.0

        d_az = self._CA * sec_alt + self._NPAE * tan_alt + self._AN * sin_az * tan_alt - self._AW * cos_az * tan_alt
        d_alt = self._IE - self._AN * cos_az - self._AW * sin_az

        return d_az, d_alt

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def record_verification_residual(self, residual_deg: float) -> None:
        """Record a post-slew plate-solve residual for health and live accuracy.

        Called by the adapter after a plate-solve-after-slew verification
        (not during calibration).  Feeds both the health monitor (degraded
        after sustained bad residuals) and the live accuracy tracker shown
        in the web UI.  All values in degrees.
        """
        now = time.time()
        with self._lock:
            self._recent_residuals.append(residual_deg)
            self._live_residuals.append((now, residual_deg))

            if len(self._recent_residuals) < _HEALTH_WINDOW:
                return

            threshold = max(self._rms_deg * _HEALTH_DEGRADED_FACTOR, 10.0 / 60.0)
            above = sum(1 for r in self._recent_residuals if r > threshold)
            if above >= _HEALTH_WINDOW:
                if self._health != "degraded":
                    self._log.warning(
                        "Pointing model health DEGRADED: last %d residuals exceeded %.4f° threshold (3x RMS)",
                        _HEALTH_WINDOW,
                        threshold,
                    )
                self._health = "degraded"
            else:
                self._health = "good"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all calibration data and fitted terms."""
        with self._lock:
            self._points.clear()
            self._AN = self._AW = self._IE = self._CA = self._NPAE = 0.0
            self._rms_deg = 0.0
            self._fit_timestamp = None
            self._n_terms = 0
            self._recent_residuals.clear()
            self._live_residuals.clear()
            self._health = "unknown"
        self._save_state()
        self._log.info("Pointing model reset")

    # ------------------------------------------------------------------
    # Status / display
    # ------------------------------------------------------------------

    @staticmethod
    def _azimuth_spread(az_values: list[float]) -> float:
        """Compute the angular spread of azimuth values on the circle.

        Returns the smallest arc (in degrees) that contains all the values.
        A spread of 180° means good diversity; < 90° means under-constrained.
        """
        if len(az_values) < 2:
            return 0.0
        sorted_az = sorted(a % 360.0 for a in az_values)
        max_gap = 0.0
        for i in range(len(sorted_az)):
            gap = (sorted_az[(i + 1) % len(sorted_az)] - sorted_az[i]) % 360.0
            if gap > max_gap:
                max_gap = gap
        return 360.0 - max_gap

    def find_nearby_point_index(
        self,
        az_deg: float,
        alt_deg: float,
        min_sep_deg: float = _NEARBY_POINT_MIN_SEP,
    ) -> int | None:
        """Return the index of the nearest stored point within ``min_sep_deg``.

        Scans all stored points once and returns the index of the closest
        one whose flat-euclidean separation ``sqrt(d_az**2 + d_alt**2)`` is
        strictly less than ``min_sep_deg``, handling azimuth wraparound.
        Returns ``None`` when no point is within range.
        """
        nearest_idx: int | None = None
        nearest_sep: float = min_sep_deg
        with self._lock:
            for idx, (p_az, p_alt, _daz, _dalt) in enumerate(self._points):
                d_alt = abs(alt_deg - p_alt)
                if d_alt > min_sep_deg:
                    continue
                d_az = abs(az_deg - p_az)
                if d_az > 180.0:
                    d_az = 360.0 - d_az
                sep = math.sqrt(d_az**2 + d_alt**2)
                if sep < nearest_sep:
                    nearest_sep = sep
                    nearest_idx = idx
        return nearest_idx

    def has_nearby_point(self, az_deg: float, alt_deg: float, min_sep_deg: float = _NEARBY_POINT_MIN_SEP) -> bool:
        """Check if any existing point is within ``min_sep_deg`` of (az, alt).

        Used by operational feeders to avoid adding near-duplicate points
        from repeated observations of the same target.
        """
        return self.find_nearby_point_index(az_deg, alt_deg, min_sep_deg) is not None

    def replace_point(
        self,
        index: int,
        mount_ra: float,
        mount_dec: float,
        solved_ra: float,
        solved_dec: float,
        site_lat: float,
        site_lon: float,
    ) -> None:
        """Overwrite ``self._points[index]`` with a fresh plate-solve measurement.

        Used by operational feeders when a new measurement falls inside the
        nearby-point guard radius: instead of discarding it (losing the
        refresh) or duplicating the cell (biasing the fit), we replace the
        stale point in-place.  Mirrors ``add_point``'s alt/az conversion and
        ``d_az`` wraparound, then triggers a refit so the new terms (and the
        persisted state written inside ``fit()``) apply immediately.
        """
        gast = gast_degrees()
        mount_az, mount_alt = radec_to_altaz(mount_ra, mount_dec, site_lat, site_lon, _gast_override=gast)
        solved_az, solved_alt = radec_to_altaz(solved_ra, solved_dec, site_lat, site_lon, _gast_override=gast)

        d_az = solved_az - mount_az
        if d_az > 180.0:
            d_az -= 360.0
        elif d_az < -180.0:
            d_az += 360.0

        d_alt = solved_alt - mount_alt

        with self._lock:
            if index < 0 or index >= len(self._points):
                self._log.warning(
                    "replace_point: index %d out of range (have %d points) — ignoring",
                    index,
                    len(self._points),
                )
                return
            self._points[index] = (mount_az, mount_alt, d_az, d_alt)
            n_points = len(self._points)

        self._log.info(
            "Pointing model: replaced point #%d — az=%.1f° alt=%.1f° dAz=%.4f° dAlt=%.4f°",
            index + 1,
            mount_az,
            mount_alt,
            d_az,
            d_alt,
        )

        if n_points >= _MIN_POINTS_3TERM:
            self.fit()

    def _compass_label(self, bearing_deg: float) -> str:
        """Convert a bearing in degrees to an 8-point compass label."""
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((bearing_deg + 22.5) % 360.0 / 45.0) % 8
        return dirs[idx]

    def status(self) -> dict[str, Any]:
        """Build a status dict for the web UI."""
        with self._lock:
            if self._n_terms == 0:
                state = "untrained"
            elif self._n_terms == 3:
                state = "partial"
            else:
                state = "trained"

            tilt_mag_deg = math.sqrt(self._AN**2 + self._AW**2)
            tilt_bearing_deg = math.degrees(math.atan2(self._AW, self._AN)) % 360.0

            live: dict[str, Any] = {"count": len(self._live_residuals)}
            if self._live_residuals:
                values = [r for _, r in self._live_residuals]
                last_ts, last_val = self._live_residuals[-1]
                live["last_deg"] = round(last_val, 5)
                live["last_timestamp"] = last_ts
                live["median_deg"] = round(sorted(values)[len(values) // 2], 5)
                live["history"] = [{"t": t, "v": round(v, 5)} for t, v in self._live_residuals]

            return {
                "state": state,
                "health": self._health,
                "point_count": len(self._points),
                "n_terms": self._n_terms,
                "tilt_deg": round(tilt_mag_deg, 3),
                "tilt_direction_deg": round(tilt_bearing_deg, 1),
                "tilt_direction_label": self._compass_label(tilt_bearing_deg) if tilt_mag_deg > 0.001 else "",
                "pointing_accuracy_deg": round(self._rms_deg, 5),
                "fit_timestamp": self._fit_timestamp,
                "terms": {
                    "AN": round(self._AN, 5),
                    "AW": round(self._AW, 5),
                    "IE": round(self._IE, 5),
                    "CA": round(self._CA, 5),
                    "NPAE": round(self._NPAE, 5),
                },
                "live_accuracy": live,
            }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize model state for persistence."""
        with self._lock:
            return {
                "points": list(self._points),
                "terms": {
                    "AN": self._AN,
                    "AW": self._AW,
                    "IE": self._IE,
                    "CA": self._CA,
                    "NPAE": self._NPAE,
                },
                "n_terms": self._n_terms,
                "rms_deg": self._rms_deg,
                "fit_timestamp": self._fit_timestamp,
            }

    def _apply_dict(self, data: dict[str, Any]) -> None:
        """Apply serialized state to this instance (no lock — caller must hold it if needed)."""
        self._points = [tuple(p) for p in data.get("points", [])]
        terms = data.get("terms", {})
        self._AN = terms.get("AN", 0.0)
        self._AW = terms.get("AW", 0.0)
        self._IE = terms.get("IE", 0.0)
        self._CA = terms.get("CA", 0.0)
        self._NPAE = terms.get("NPAE", 0.0)
        self._n_terms = data.get("n_terms", 0)
        self._rms_deg = data.get("rms_deg", data.get("rms_arcmin", 0.0) / 60.0)
        self._fit_timestamp = data.get("fit_timestamp")
        if self._n_terms > 0:
            self._health = "good"

    @classmethod
    def from_dict(cls, data: dict[str, Any], state_file: Path | None = None) -> AltAzPointingModel:
        """Restore a model from a serialized dict."""
        model = cls.__new__(cls)
        model._state_file = state_file
        model._lock = threading.Lock()
        model._points = []
        model._AN = model._AW = model._IE = model._CA = model._NPAE = 0.0
        model._rms_deg = 0.0
        model._fit_timestamp = None
        model._n_terms = 0
        model._recent_residuals = deque(maxlen=_HEALTH_WINDOW)
        model._live_residuals = deque(maxlen=_LIVE_ACCURACY_WINDOW)
        model._health = "unknown"
        model._apply_dict(data)
        return model

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore model state from a previously captured ``to_dict()`` snapshot.

        Used by the calibration workflow to roll back to a working model
        when a fresh calibration fails or is cancelled.
        """
        with self._lock:
            self._apply_dict(data)
            self._recent_residuals.clear()
            self._live_residuals.clear()
        self._save_state()
        self._log.info(
            "Pointing model restored: %d-term, %d points, RMS=%.4f°",
            self._n_terms,
            len(self._points),
            self._rms_deg,
        )

    # ------------------------------------------------------------------
    # File persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        if self._state_file is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        except Exception:
            self._log.debug("Failed to persist pointing model state", exc_info=True)

    def _load_state(self) -> None:
        if self._state_file is None:
            return
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            self._apply_dict(data)
            self._log.info(
                "Loaded pointing model: %d-term, %d points, RMS=%.4f°",
                self._n_terms,
                len(self._points),
                self._rms_deg,
            )
        except Exception:
            self._log.warning("Failed to load pointing model state", exc_info=True)
