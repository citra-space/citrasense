"""Dummy hardware adapter for testing without real hardware."""

from __future__ import annotations

import datetime
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from citrascope.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
    SettingSchemaEntry,
)
from citrascope.hardware.devices.camera.abstract_camera import AbstractCamera
from citrascope.hardware.devices.focuser.abstract_focuser import AbstractFocuser
from citrascope.hardware.devices.mount.abstract_mount import AbstractMount
from citrascope.hardware.devices.mount.mount_state_cache import MountStateCache

# Default observer location — Pikes Peak, matches DummyApiClient.
_OBSERVER_LAT_DEG = 38.8409
_OBSERVER_LON_DEG = -105.0423


def _current_lst_deg() -> float:
    """Approximate Local Sidereal Time in degrees for the default observer."""
    now = datetime.datetime.now(datetime.timezone.utc)
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    d = (now - j2000).total_seconds() / 86400.0
    ut_h = now.hour + now.minute / 60.0 + now.second / 3600.0
    return (100.46 + 0.985647 * d + _OBSERVER_LON_DEG + 15.0 * ut_h) % 360.0


def _radec_to_altaz(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Approximate RA/Dec → (Az, Alt) for the default observer at the current time."""
    lat = math.radians(_OBSERVER_LAT_DEG)
    lst_deg = _current_lst_deg()

    ha = math.radians((lst_deg - ra_deg) % 360.0)
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


def _radec_to_az(ra_deg: float, dec_deg: float) -> float:
    """Approximate RA/Dec → Azimuth for the default observer at the current time."""
    return _radec_to_altaz(ra_deg, dec_deg)[0]


def _altaz_to_radec(az_deg: float, alt_deg: float) -> tuple[float, float]:
    """Approximate Az/Alt → (RA, Dec) for the default observer at the current time."""
    lat = math.radians(_OBSERVER_LAT_DEG)
    az = math.radians(az_deg)
    alt = math.radians(alt_deg)

    sin_dec = math.sin(alt) * math.sin(lat) + math.cos(alt) * math.cos(lat) * math.cos(az)
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))

    cos_dec = math.cos(dec)
    if cos_dec < 1e-10:
        return _current_lst_deg(), math.degrees(dec)

    cos_ha = (math.sin(alt) - math.sin(dec) * math.sin(lat)) / (cos_dec * math.cos(lat) + 1e-10)
    ha_abs = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    ha = ha_abs if az_deg > 180.0 else -ha_abs

    ra = (_current_lst_deg() - ha) % 360.0
    return ra, math.degrees(dec)


class _DummyMount(AbstractMount):
    """Simulated alt-az mount with realistic azimuth behaviour.

    - **Slews** compute a real target azimuth from RA/Dec + observer location
      + current sidereal time, so cable-wrap accumulates bidirectionally.
    - **Tracking** drifts the azimuth at a configurable rate (sidereal-ish)
      so cumulative wrap grows slowly even between slews.
    - **Directional motion** (``start_move`` / ``stop_move``) shifts azimuth
      at a fixed rate per ``get_azimuth()`` call, supporting the cable-unwind loop.
    """

    _SLEW_RATE_DEG_PER_S = 6.0
    _MOVE_RATE_DEG_PER_S = 16.0
    _TRACKING_DRIFT_DEG_PER_S = 0.005

    def __init__(
        self,
        logger: logging.Logger,
        sim_AN: float = 0.4,
        sim_AW: float = -0.25,
        sim_IE: float = 0.08,
    ) -> None:
        super().__init__(logger)
        self._sim_AN = sim_AN
        self._sim_AW = sim_AW
        self._sim_IE = sim_IE

        self._ra: float = _current_lst_deg()
        self._dec: float = _OBSERVER_LAT_DEG
        az, alt = _radec_to_altaz(self._ra, self._dec)
        self._base_az: float = az
        self._alt: float = alt
        self._ref_time: float = time.monotonic()
        self._slewing = False
        self._tracking = True
        self._parked = False
        self._homed = False
        self._moving_dir: str | None = None
        self._motion_rate: float = self._SLEW_RATE_DEG_PER_S
        self._abort_event = threading.Event()

    # -- AbstractHardwareDevice ------------------------------------------------

    @classmethod
    def get_friendly_name(cls) -> str:
        return "Dummy Mount"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {"packages": [], "install_extra": ""}

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        return []

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    # -- Internal motion simulation --------------------------------------------

    def _simulate_motion(self, target_az: float, target_alt: float, rate: float) -> bool:
        """Sweep azimuth toward *target_az* at *rate* deg/s, then set altitude.

        All simulated motion funnels through here so ``get_azimuth()`` always
        reports gradual interpolation and the cable-wrap observer never sees
        an instantaneous jump.  The wait is interruptible via ``_abort_event``.

        Returns True if completed, False if aborted.
        """
        self._abort_event.clear()
        self._snap()
        current_az = self._base_az % 360.0
        delta = ((target_az - current_az + 180.0) % 360.0) - 180.0

        aborted = False
        if abs(delta) > 1.0:
            self._slewing = True
            self._motion_rate = rate
            self._moving_dir = "east" if delta > 0 else "west"
            wait_s = abs(delta) / rate
            aborted = self._abort_event.wait(wait_s)
            self._snap()
            self._moving_dir = None
            self._slewing = False

        if aborted:
            self._ra, self._dec = _altaz_to_radec(self._base_az % 360.0, self._alt)
            return False

        self._base_az = target_az
        self._alt = target_alt
        self._ra, self._dec = _altaz_to_radec(target_az, target_alt)
        self._ref_time = time.monotonic()
        return True

    # -- AbstractMount core ----------------------------------------------------

    def slew_to_radec(self, ra: float, dec: float) -> bool:
        self._homed = False
        target_az, target_alt = _radec_to_altaz(ra, dec)
        if not self._simulate_motion(target_az, target_alt, self._SLEW_RATE_DEG_PER_S):
            self.logger.info("Slew aborted mid-transit")
            return False
        # Recompute az/alt at current LST — the sky rotated during the slew.
        self._base_az, self._alt = _radec_to_altaz(ra, dec)
        self._ra = ra
        self._dec = dec
        self._ref_time = time.monotonic()
        return True

    def is_slewing(self) -> bool:
        return self._slewing

    def abort_slew(self) -> None:
        self._abort_event.set()
        self._slewing = False

    def get_radec(self) -> tuple[float, float]:
        if self._moving_dir is not None:
            az = self.get_azimuth()
            alt = self.get_altitude()
            if az is not None and alt is not None:
                return _altaz_to_radec(az, alt)
        return self._ra, self._dec

    def start_tracking(self, rate: str | None = "sidereal") -> bool:
        self._snap()
        self._tracking = True
        return True

    def stop_tracking(self) -> bool:
        self._snap()
        self._tracking = False
        return True

    def is_tracking(self) -> bool:
        return self._tracking

    def park(self) -> bool:
        park_alt = 90.0 - abs(_OBSERVER_LAT_DEG)
        if not self._simulate_motion(0.0, park_alt, self._SLEW_RATE_DEG_PER_S):
            return False
        self._parked = True
        return True

    def unpark(self) -> bool:
        self._parked = False
        return True

    def is_parked(self) -> bool:
        return self._parked

    def get_mount_info(self) -> dict:
        return {"name": "Dummy Mount", "supports_sync": True}

    def sync_to_radec(self, ra: float, dec: float) -> bool:
        self._snap()
        self._ra = ra
        self._dec = dec
        az, alt = _radec_to_altaz(ra, dec)
        self._base_az = az
        self._alt = alt
        self._ref_time = time.monotonic()
        self.logger.info(f"Synced to RA={ra:.4f}°, Dec={dec:.4f}°")
        self._fire_sync_listeners(az)
        return True

    _HOME_AZ = 0.0

    # -- Homing ----------------------------------------------------------------

    def find_home(self) -> bool:
        home_alt = 90.0 - abs(_OBSERVER_LAT_DEG)
        if not self._simulate_motion(self._HOME_AZ, home_alt, self._MOVE_RATE_DEG_PER_S):
            self.logger.info("Homing aborted mid-transit")
            return False
        self._tracking = False
        self._homed = True
        self.logger.info("DummyMount homed to az=%.1f°", self._HOME_AZ)
        self._fire_sync_listeners(self._HOME_AZ)
        return True

    def is_home(self) -> bool:
        return self._homed

    # -- Safety-related overrides ---------------------------------------------

    def get_mount_mode(self) -> str:
        return "altaz"

    def get_azimuth(self) -> float | None:
        dt = time.monotonic() - self._ref_time
        if self._moving_dir in ("east", "west"):
            sign = 1.0 if self._moving_dir == "east" else -1.0
            return (self._base_az + sign * self._motion_rate * dt) % 360.0
        if self._tracking:
            return (self._base_az + self._TRACKING_DRIFT_DEG_PER_S * dt) % 360.0
        return self._base_az % 360.0

    def get_altitude(self) -> float | None:
        dt = time.monotonic() - self._ref_time
        if self._moving_dir in ("north", "south"):
            sign = 1.0 if self._moving_dir == "north" else -1.0
            return max(-90.0, min(90.0, self._alt + sign * self._motion_rate * dt))
        return self._alt

    # -- Simulated pointing error (where the optics actually land) ----------

    def true_altaz(self) -> tuple[float, float]:
        """Apply the simulated tripod-tilt error model to the mount's position.

        Returns where the optics actually point, as opposed to where the
        mount's encoders believe they point.  Uses the same 3-term model
        (AN, AW, IE) the ``AltAzPointingModel`` fits.

        Returns:
            ``(true_az, true_alt)`` in degrees.
        """
        az_deg = self.get_azimuth() or 0.0
        alt_deg = self.get_altitude() or 0.0

        az_r = math.radians(az_deg)
        alt_r = math.radians(alt_deg)
        tan_alt = math.tan(alt_r) if abs(math.cos(alt_r)) > 1e-10 else 0.0

        d_az = self._sim_AN * math.sin(az_r) * tan_alt - self._sim_AW * math.cos(az_r) * tan_alt
        d_alt = self._sim_IE - self._sim_AN * math.cos(az_r) - self._sim_AW * math.sin(az_r)

        true_az = (az_deg + d_az) % 360.0
        true_alt = max(-90.0, min(90.0, alt_deg + d_alt))
        return true_az, true_alt

    def true_radec(self) -> tuple[float, float]:
        """Return the RA/Dec where the optics actually point (with error)."""
        true_az, true_alt = self.true_altaz()
        return _altaz_to_radec(true_az, true_alt)

    def start_move(self, direction: str, rate: int | None = None) -> bool:
        if self._slewing:
            return False
        self._snap()
        self._motion_rate = self._MOVE_RATE_DEG_PER_S
        self._moving_dir = direction
        return True

    def stop_move(self, direction: str) -> bool:
        if self._slewing:
            return False
        if self._moving_dir != direction:
            return True
        self._snap()
        self._moving_dir = None
        self._ra, self._dec = _altaz_to_radec(self._base_az % 360.0, self._alt)
        return True

    def _snap(self) -> None:
        """Commit the current computed az/alt as the new base, resetting the reference clock."""
        dt = time.monotonic() - self._ref_time
        if self._moving_dir is not None:
            if self._moving_dir in ("north", "south"):
                sign = 1.0 if self._moving_dir == "north" else -1.0
                self._alt = max(-90.0, min(90.0, self._alt + sign * self._motion_rate * dt))
            else:
                sign = 1.0 if self._moving_dir == "east" else -1.0
                self._base_az = (self._base_az + sign * self._motion_rate * dt) % 360.0
        elif self._tracking:
            self._base_az = (self._base_az + self._TRACKING_DRIFT_DEG_PER_S * dt) % 360.0
        self._ref_time = time.monotonic()


class _DummyFocuser(AbstractFocuser):
    """Simulated focuser for DummyAdapter."""

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger)
        self._position: int = 25000
        self._max_position: int = 100000
        self._connected = True
        self._moving = False

    @classmethod
    def get_friendly_name(cls) -> str:
        return "Dummy Focuser"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        return {"packages": [], "install_extra": ""}

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        return []

    def connect(self) -> bool:
        self._connected = True
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_position(self) -> int | None:
        return self._position

    def get_max_position(self) -> int | None:
        return self._max_position

    def get_temperature(self) -> float | None:
        return 18.5

    def is_moving(self) -> bool:
        return self._moving

    def move_absolute(self, position: int) -> bool:
        if position < 0 or position > self._max_position:
            self.logger.error(f"DummyFocuser: position {position} out of range")
            return False
        self._position = position
        return True

    def move_relative(self, steps: int) -> bool:
        target = self._position + steps
        if target < 0 or target > self._max_position:
            self.logger.error(f"DummyFocuser: relative move to {target} out of range (0–{self._max_position})")
            return False
        self._position = target
        return True

    def abort_move(self) -> None:
        self._moving = False

    def disconnect(self) -> None:
        self._connected = False


# Synthetic camera constants — consistent across take_image() and the WCS header
# so the image geometry is self-describing.
_DUMMY_IMG_SIZE = 1024  # pixels per side
_DUMMY_PIXEL_SCALE = 6.0  # arcsec/pixel  →  ~1.7° FOV
_DUMMY_BIAS = 100.0  # ADU pedestal — keeps read noise above zero like a real CCD
_DUMMY_SKY_RATE = 5.0  # ADU/s sky background rate (dark site)
_DUMMY_READ_NOISE = 1.0  # electrons RMS
_DUMMY_GAIN = 1.5  # electrons/ADU
_DUMMY_PSF_SIGMA_PX = 3.0 / 2.3548  # sigma from 3.0 px FWHM seeing
_DUMMY_MAG_LIMIT = 14.0  # faintest catalog star to render (Vmag)
_DUMMY_MAG_ZERO = 20.0  # instrument zero-point: V=10 → SNR~58, V=12 → SNR~9
_DUMMY_OPTIMAL_FOCUS = 25000  # focuser position for sharpest PSF
_DUMMY_DEFOCUS_K = 0.002  # PSF broadening per focuser step from optimal
_DUMMY_FILTER_FOCUS_OFFSETS: dict[str, int] = {
    "luminance": 0,
    "clear": 0,
    "red": 500,
    "green": 350,
    "blue": 700,
    "ha": 1200,
}


_MAG_PREFERENCE = [
    ("Johnson_V (V)", 0.0),
    ("Sloan_r (SR)", 0.16),
    ("Sloan_g (SG)", -0.31),
    ("Sloan_i (SI)", 0.37),
    ("Johnson_B (B)", -0.36),
]


def _best_mag(df: pd.DataFrame) -> np.ndarray:
    """Pick the best available magnitude per star, converting to approx V.

    Cascades through APASS bands in preference order, filling NaN gaps
    with the next available band offset to approximate Johnson V.
    """
    result = np.full(len(df), np.nan)
    unfilled = np.isnan(result)
    for col, offset in _MAG_PREFERENCE:
        if col not in df.columns or not unfilled.any():
            continue
        vals = np.asarray(df[col].values, dtype=np.float64)
        usable = unfilled & np.isfinite(vals)
        result[usable] = vals[usable] + offset
        unfilled = np.isnan(result)
    return result


class DummyAdapter(AbstractAstroHardwareAdapter):
    """
    Dummy hardware adapter that simulates hardware without requiring real devices.

    Perfect for testing, development, and demonstrations. All operations are logged
    and return realistic fake data. Images are synthetic starfields with a proper
    WCS header keyed to the current simulated telescope pointing.
    """

    def __init__(self, logger: logging.Logger, images_dir: Path, **kwargs):
        """Initialize dummy adapter.

        Args:
            logger: Logger instance
            images_dir: Path to images directory
            **kwargs: Additional settings including 'simulate_slow_operations'
        """
        super().__init__(images_dir, **kwargs)
        self.logger = logger
        self.simulate_slow = kwargs.get("simulate_slow_operations", False)
        self.slow_delay = kwargs.get("slow_delay_seconds", 2.0)

        # Fake hardware state
        self._connected = False
        self._telescope_connected = False
        self._camera_connected = False
        self._is_moving = False
        self._tracking_rate = (15.041, 0.0)  # arcsec/sec (sidereal rate)
        self._mount = _DummyMount(
            logger,
            sim_AN=float(kwargs.get("sim_AN", 0.4)),
            sim_AW=float(kwargs.get("sim_AW", -0.25)),
            sim_IE=float(kwargs.get("sim_IE", 0.08)),
        )
        self._mount_cache: MountStateCache | None = None
        self._focuser = _DummyFocuser(logger)
        self._current_filter_position: int = 0
        self._current_filter_offset: int = 0

        # Set by the daemon after connecting, mirrors the real telescope_record from the API.
        # When present, take_image() derives sensor dimensions and pixel scale from it.
        self.telescope_record: dict | None = None

        self.logger.info("DummyAdapter initialized")

    @property
    def mount(self) -> _DummyMount:
        return self._mount

    @property
    def focuser(self) -> AbstractFocuser:
        return self._focuser

    @classmethod
    def get_settings_schema(cls, **kwargs) -> list[SettingSchemaEntry]:
        """Return configuration schema for dummy adapter."""
        return [
            {
                "name": "simulate_slow_operations",
                "friendly_name": "Simulate Slow Operations",
                "type": "bool",
                "default": False,
                "description": "Add artificial delays to simulate slow hardware responses",
                "required": False,
                "group": "Testing",
            },
            {
                "name": "slow_delay_seconds",
                "friendly_name": "Delay Duration (seconds)",
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 10.0,
                "description": "Duration of artificial delays when slow simulation is enabled",
                "required": False,
                "group": "Testing",
            },
            {
                "name": "sim_AN",
                "friendly_name": "Tripod Tilt N-S (AN)",
                "type": "float",
                "default": 0.4,
                "min": -2.0,
                "max": 2.0,
                "description": "Simulated azimuth-axis tilt north-south in degrees",
                "required": False,
                "group": "Pointing Error Simulation",
            },
            {
                "name": "sim_AW",
                "friendly_name": "Tripod Tilt E-W (AW)",
                "type": "float",
                "default": -0.25,
                "min": -2.0,
                "max": 2.0,
                "description": "Simulated azimuth-axis tilt east-west in degrees",
                "required": False,
                "group": "Pointing Error Simulation",
            },
            {
                "name": "sim_IE",
                "friendly_name": "Altitude Offset (IE)",
                "type": "float",
                "default": 0.08,
                "min": -2.0,
                "max": 2.0,
                "description": "Simulated constant altitude index error in degrees",
                "required": False,
                "group": "Pointing Error Simulation",
            },
        ]

    def get_observation_strategy(self) -> ObservationStrategy:
        """Dummy adapter uses manual strategy."""
        return ObservationStrategy.MANUAL

    def perform_observation_sequence(self, task, satellite_data) -> list[str]:
        """Not used for manual strategy."""
        raise NotImplementedError("DummyAdapter uses MANUAL strategy")

    def connect(self) -> bool:
        """Simulate connection."""
        self.logger.info("DummyAdapter: Connecting...")
        self._simulate_delay()
        self._connected = True
        self._telescope_connected = True
        self._camera_connected = True

        if not self.filter_map:
            self.filter_map = {
                fid: {
                    "name": name,
                    "focus_position": _DUMMY_OPTIMAL_FOCUS + _DUMMY_FILTER_FOCUS_OFFSETS.get(name.lower(), 0),
                    "enabled": enabled,
                }
                for fid, (name, enabled) in enumerate(
                    [("Luminance", True), ("Red", True), ("Green", True), ("Blue", True), ("Ha", False)]
                )
            }
            self.logger.info(f"DummyAdapter: Populated {len(self.filter_map)} simulated filters")

        cache = MountStateCache(self._mount)
        cache.refresh_static()
        cache.start()
        self._mount._state_cache = cache  # type: ignore[attr-defined]
        self._mount_cache = cache

        self._init_pointing_model("DummyAdapter")

        self.logger.info("DummyAdapter: Connected successfully")
        return True

    def disconnect(self):
        """Simulate disconnection."""
        self.logger.info("DummyAdapter: Disconnecting...")
        if self._mount_cache:
            self._mount_cache.stop()
            self._mount_cache = None
        self._connected = False
        self._telescope_connected = False
        self._camera_connected = False
        self.logger.info("DummyAdapter: Disconnected")

    def is_telescope_connected(self) -> bool:
        """Check fake telescope connection."""
        return self._telescope_connected

    def is_camera_connected(self) -> bool:
        """Check fake camera connection."""
        return self._camera_connected

    def list_devices(self) -> list[str]:
        """Return list of fake devices."""
        return ["Dummy Telescope", "Dummy Camera", "Dummy Filter Wheel", "Dummy Focuser"]

    def select_telescope(self, device_name: str) -> bool:
        """Simulate telescope selection."""
        self.logger.info(f"DummyAdapter: Selected telescope '{device_name}'")
        self._telescope_connected = True
        return True

    def _do_point_telescope(self, ra: float, dec: float):
        """Simulate telescope slew."""
        self.logger.info(f"DummyAdapter: Slewing to RA={ra:.4f}°, Dec={dec:.4f}°")
        self._is_moving = True
        success = self._mount.slew_to_radec(ra, dec)
        self._is_moving = False
        if not success:
            raise RuntimeError("Slew aborted by safety monitor")
        self.logger.info("DummyAdapter: Slew complete")

    def get_telescope_direction(self) -> tuple[float, float]:
        """Return current telescope position from the simulated mount."""
        return self._mount.get_radec()

    def abort_slew(self) -> None:
        self._mount.abort_slew()

    def telescope_is_moving(self) -> bool:
        """Check if fake telescope is moving."""
        return self._is_moving

    def home_if_needed(self) -> bool:
        if self._mount.is_home():
            self.logger.info("DummyAdapter: Mount already homed")
            return True
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        self.logger.info("DummyAdapter: Homing mount (safety-monitored)...")
        self._is_moving = True
        self._mount.find_home()
        self._is_moving = False
        self.logger.info("DummyAdapter: Mount homed")
        return True

    def home_mount(self) -> bool:
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        self.logger.info("DummyAdapter: Homing mount...")
        self._is_moving = True
        self._mount.find_home()
        self._is_moving = False
        self.logger.info("DummyAdapter: Mount homed")
        return True

    def is_mount_homed(self) -> bool:
        return self._mount.is_home()

    def select_camera(self, device_name: str) -> bool:
        """Simulate camera selection."""
        self.logger.info(f"DummyAdapter: Selected camera '{device_name}'")
        self._camera_connected = True
        return True

    def take_image(self, task_id: str, exposure_duration_seconds=1.0) -> str:
        """Simulate image capture, producing a synthetic starfield FITS."""
        self.logger.info(f"DummyAdapter: Starting {exposure_duration_seconds}s exposure for task {task_id}")
        self._simulate_delay(exposure_duration_seconds)

        timestamp = time.time_ns()
        filename = f"dummy_{task_id}_{timestamp}.fits"
        filepath = self.images_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        true_ra, true_dec = self._mount.true_radec()
        sigma = self._focus_dependent_psf_sigma()
        image_data, wcs = self._generate_starfield(
            true_ra,
            true_dec,
            exposure_duration_seconds,
            seed=timestamp % (2**31),
            psf_sigma=sigma,
        )

        hdu = fits.PrimaryHDU(image_data, header=wcs.to_header())
        hdu.header["INSTRUME"] = ("DummyCamera", "Simulated camera")
        hdu.header["EXPTIME"] = (exposure_duration_seconds, "Exposure time (seconds)")
        hdu.header["DATE-OBS"] = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "UTC start of exposure",
        )
        hdu.header["TASKID"] = task_id
        hdu.writeto(filepath, overwrite=True)

        self.logger.info(f"DummyAdapter: Image saved to {filepath}")
        return str(filepath)

    def _generate_starfield(
        self,
        ra_center_deg: float,
        dec_center: float,
        exptime: float,
        seed: int,
        psf_sigma: float | None = None,
    ) -> tuple[np.ndarray, WCS]:
        """Generate a synthetic starfield with a TAN WCS.

        Queries the local APASS catalog (or falls back to synthetic stars)
        for the current FOV, projects them onto the pixel grid, and renders
        each as a Gaussian PSF.

        Args:
            ra_center_deg: RA of the field centre in degrees.
            dec_center:    Dec of the field centre in degrees.
            exptime:       Exposure duration in seconds (scales star brightness).
            seed:          RNG seed for the noise model (use Unix timestamp).
            psf_sigma:     Override PSF sigma in pixels (None uses default seeing).

        Returns:
            Tuple of (image array uint16, WCS object).
        """
        rng = np.random.default_rng(seed)
        ra_center = ra_center_deg

        # Derive sensor geometry from telescope_record when available,
        # so the simulated image matches the real instrument's FOV and resolution.
        tr = self.telescope_record
        if (
            tr
            and tr.get("pixelSize")
            and tr.get("focalLength")
            and tr.get("horizontalPixelCount")
            and tr.get("verticalPixelCount")
        ):
            pixel_scale = float(tr["pixelSize"]) / float(tr["focalLength"]) * 206.265
            size_x = int(tr["horizontalPixelCount"])
            size_y = int(tr["verticalPixelCount"])
            self.logger.debug(
                f"DummyAdapter: using telescope sensor {size_x}×{size_y}px " f"@ {pixel_scale:.2f} arcsec/px"
            )
        else:
            pixel_scale = _DUMMY_PIXEL_SCALE
            size_x = size_y = _DUMMY_IMG_SIZE

        fov_x_deg = size_x * pixel_scale / 3600.0
        fov_y_deg = size_y * pixel_scale / 3600.0
        search_radius = math.sqrt(fov_x_deg**2 + fov_y_deg**2) / 2

        # --- WCS: TAN projection centred on current pointing -----------------
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [(size_x + 1) / 2.0, (size_y + 1) / 2.0]
        wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
        wcs.wcs.crval = [ra_center, dec_center]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # --- Star catalog query ----------------------------------------------
        star_ras, star_decs, star_mags = self._fetch_catalog_stars(ra_center, dec_center, search_radius)

        # --- Sky background + noise ------------------------------------------
        sky_adu = _DUMMY_SKY_RATE * exptime + 0.01 * exptime / _DUMMY_GAIN
        sky_electrons = sky_adu * _DUMMY_GAIN
        image = rng.poisson(np.full((size_y, size_x), max(sky_electrons, 0.1))).astype(np.float64) / _DUMMY_GAIN
        image += rng.normal(0.0, _DUMMY_READ_NOISE / _DUMMY_GAIN, image.shape)
        image += _DUMMY_BIAS

        # --- Render each catalog star as a Gaussian PSF ----------------------
        if len(star_ras) == 0:
            self.logger.debug("DummyAdapter: No catalog stars in FOV — returning noise-only image")
            return np.clip(image, 0, 65535).astype(np.uint16), wcs

        effective_psf_sigma = psf_sigma if psf_sigma is not None else _DUMMY_PSF_SIGMA_PX
        stamp_r = math.ceil(5 * effective_psf_sigma)

        # Pre-compute a unit-flux PSF stamp once — just scale by each star's ADU
        psf_stamp = np.zeros((2 * stamp_r + 1, 2 * stamp_r + 1))
        psf_stamp[stamp_r, stamp_r] = 1.0
        psf_stamp = gaussian_filter(psf_stamp, sigma=effective_psf_sigma)
        psf_stamp /= psf_stamp.sum()

        pixel_coords = wcs.all_world2pix(np.column_stack([star_ras, star_decs]), 0)

        valid = np.all(np.isfinite(pixel_coords), axis=1)
        pixel_coords = pixel_coords[valid]
        star_mags = star_mags[valid]

        n_rendered = 0
        for (xp, yp), mag in zip(pixel_coords, star_mags, strict=False):
            flux_e = 10.0 ** ((_DUMMY_MAG_ZERO - mag) / 2.5) * exptime
            total_adu = flux_e / _DUMMY_GAIN

            xi, yi = round(xp), round(yp)

            x0, y0 = xi - stamp_r, yi - stamp_r
            ix0 = max(0, x0)
            iy0 = max(0, y0)
            ix1 = min(size_x, x0 + psf_stamp.shape[1])
            iy1 = min(size_y, y0 + psf_stamp.shape[0])
            sx0 = ix0 - x0
            sy0 = iy0 - y0
            sx1 = sx0 + (ix1 - ix0)
            sy1 = sy0 + (iy1 - iy0)

            if ix1 > ix0 and iy1 > iy0:
                image[iy0:iy1, ix0:ix1] += psf_stamp[sy0:sy1, sx0:sx1] * total_adu
                n_rendered += 1

        self.logger.debug(f"DummyAdapter: Rendered {n_rendered}/{len(star_mags)} catalog stars")

        image = np.clip(image, 0, 65535).astype(np.uint16)
        return image, wcs

    def _fetch_catalog_stars(
        self,
        ra: float,
        dec: float,
        fov_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query the local APASS catalog for real stars in the current FOV.

        Falls back to a deterministic synthetic field if the APASS catalog
        is not available (e.g. first run before download completes).

        Returns:
            Tuple of (ras_deg, decs_deg, mags) as float arrays.
        """
        try:
            from citrascope.catalogs.apass_catalog import ApassCatalog

            catalog = ApassCatalog()
            if catalog.is_available():
                df = catalog.cone_search(ra, dec, radius=fov_deg)
                if len(df) > 0:
                    ras = df["radeg"].values
                    decs = df["decdeg"].values
                    mags = _best_mag(df)
                    valid = np.isfinite(mags) & (mags < _DUMMY_MAG_LIMIT)
                    ras, decs, mags = ras[valid], decs[valid], mags[valid]
                    self.logger.debug(
                        f"DummyAdapter: APASS returned {len(df)} raw, "
                        f"rendering {len(mags)} stars (V < {_DUMMY_MAG_LIMIT})"
                    )
                    return np.asarray(ras), np.asarray(decs), np.asarray(mags)
        except Exception as e:
            self.logger.debug(f"DummyAdapter: APASS catalog unavailable: {e}")

        self.logger.debug(f"DummyAdapter: No catalog stars around RA={ra:.3f}, Dec={dec:.3f}")
        return np.array([]), np.array([]), np.array([])

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        """Simulate setting tracking rate."""
        self.logger.info(f"DummyAdapter: Setting tracking rate RA={ra_rate} arcsec/s, Dec={dec_rate} arcsec/s")
        self._tracking_rate = (ra_rate, dec_rate)

    def get_tracking_rate(self) -> tuple[float, float]:
        """Return current fake tracking rate."""
        return self._tracking_rate

    def perform_alignment(self, target_ra: float, target_dec: float) -> bool:
        """Plate-solve at the current position and conditionally sync the mount."""
        if not self.telescope_record:
            self.logger.warning("No telescope_record available — falling back to simulated alignment")
            self._simulate_delay()
            self._mount.sync_to_radec(target_ra, target_dec)
            return True

        return self._perform_alignment_with_model(target_ra, target_dec)

    def update_from_plate_solve(
        self,
        solved_ra_deg: float,
        solved_dec_deg: float,
        expected_ra_deg: float | None = None,
        expected_dec_deg: float | None = None,
        target_ra_deg: float | None = None,
        target_dec_deg: float | None = None,
    ) -> None:
        """Feed pipeline plate-solve results into the pointing model."""
        if expected_ra_deg is None or expected_dec_deg is None:
            return

        residual_ra = target_ra_deg if target_ra_deg is not None else expected_ra_deg
        residual_dec = target_dec_deg if target_dec_deg is not None else expected_dec_deg
        residual_deg = self.angular_distance(solved_ra_deg, solved_dec_deg, residual_ra, residual_dec)

        if self._pointing_model and self.location_service:
            location = self.location_service.get_current_location()
            if location:
                from citrascope.hardware.devices.mount.altaz_pointing_model import radec_to_altaz

                lat, lon = location["latitude"], location["longitude"]
                cmd_az, cmd_alt = radec_to_altaz(expected_ra_deg, expected_dec_deg, lat, lon)
                if not self._pointing_model.has_nearby_point(cmd_az, cmd_alt):
                    self._pointing_model.add_point(
                        expected_ra_deg, expected_dec_deg, solved_ra_deg, solved_dec_deg, lat, lon
                    )
                    self.logger.info("Pipeline fed pointing model (residual %.4f°)", residual_deg)
                else:
                    self._pointing_model.record_verification_residual(residual_deg)
                    self.logger.debug("Pipeline skip: nearby point exists (residual %.4f°)", residual_deg)

    def supports_autofocus(self) -> bool:
        """Dummy adapter supports autofocus."""
        return True

    @property
    def supports_hardware_safety_monitor(self) -> bool:
        return True

    def query_hardware_safety(self) -> bool | None:
        return True

    def _focus_dependent_psf_sigma(self) -> float:
        """Compute PSF sigma based on current focuser distance from optimal.

        Uses `_current_filter_offset` to shift the optimal position per filter,
        simulating wavelength-dependent focus shifts.
        """
        optimal = _DUMMY_OPTIMAL_FOCUS + self._current_filter_offset
        current_pos = self._focuser.get_position()
        pos = optimal if current_pos is None else current_pos
        return max(0.5, _DUMMY_PSF_SIGMA_PX + _DUMMY_DEFOCUS_K * abs(pos - optimal))

    def do_autofocus(
        self,
        target_ra: float | None = None,
        target_dec: float | None = None,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        on_point: Callable[[int, float], None] | None = None,
        on_filter_start: Callable[[str], None] | None = None,
    ) -> None:
        """Run V-curve autofocus for each enabled filter.

        Mirrors the DirectHardwareAdapter pattern: iterates enabled filters,
        switches to each, runs autofocus, labels progress with the filter
        name, and stores the resulting focus_position per filter.
        """
        from citrascope.hardware.direct.autofocus import run_autofocus

        if (target_ra is None) != (target_dec is None):
            raise ValueError(
                f"target_ra and target_dec must both be set or both be None, got ra={target_ra}, dec={target_dec}"
            )

        report = on_progress or (lambda _msg: None)

        if target_ra is not None and target_dec is not None:
            report("Slewing to autofocus target...")
            self.logger.info(f"DummyAdapter: Slewing to AF target RA={target_ra:.4f}, Dec={target_dec:.4f}")
            self._mount.slew_to_radec(target_ra, target_dec)

        adapter = self

        class _DummyAfCamera(AbstractCamera):
            """Minimal camera that renders focus-dependent synthetic starfields."""

            @classmethod
            def get_friendly_name(cls) -> str:
                return "Dummy AF Camera"

            @classmethod
            def get_dependencies(cls) -> dict[str, str | list[str]]:
                return {"packages": [], "install_extra": ""}

            @classmethod
            def get_settings_schema(cls) -> list[SettingSchemaEntry]:
                return []

            def connect(self) -> bool:
                return True

            def disconnect(self) -> None:
                pass

            def is_connected(self) -> bool:
                return True

            def capture_array(self, duration: float, **kwargs) -> np.ndarray:
                ra, dec = adapter._mount.get_radec()
                sigma = adapter._focus_dependent_psf_sigma()
                seed = int(time.time_ns() % (2**31))
                img, _ = adapter._generate_starfield(ra, dec, duration, seed=seed, psf_sigma=sigma)
                return img

            def take_exposure(self, duration: float, **kwargs):
                raise NotImplementedError

            def abort_exposure(self):
                pass

            def get_temperature(self) -> float | None:
                return 20.0

            def set_temperature(self, temperature: float) -> bool:
                return True

            def start_cooling(self) -> bool:
                return True

            def stop_cooling(self) -> bool:
                return True

            def get_camera_info(self) -> dict:
                return {"model": "DummyAfCamera", "width": _DUMMY_IMG_SIZE, "height": _DUMMY_IMG_SIZE}

        cam = _DummyAfCamera(adapter.logger)

        enabled_filters = {fid: fdata for fid, fdata in self.filter_map.items() if fdata.get("enabled", True)}

        if not enabled_filters:
            report("Running autofocus (no filters)...")
            run_autofocus(
                camera=cam,
                focuser=self._focuser,
                step_size=500,
                num_steps=5,
                exposure_time=1.0,
                crop_ratio=1.0,
                on_progress=on_progress,
                logger=self.logger,
                cancel_event=cancel_event,
                on_point=on_point,
            )
            self.logger.info("DummyAdapter: Autofocus complete")
            return

        total = len(enabled_filters)
        self.logger.info(f"DummyAdapter: Autofocusing {total} enabled filter(s)")

        for idx, (fid, fdata) in enumerate(enabled_filters.items(), 1):
            if cancel_event and cancel_event.is_set():
                report("Autofocus cancelled")
                self.logger.info("Autofocus cancelled between filters")
                raise RuntimeError("Autofocus cancelled")

            fname = fdata.get("name", f"Filter {fid}")

            if on_filter_start:
                on_filter_start(fname)

            def filter_progress(msg: str, _prefix=f"Filter {idx}/{total}: {fname}"):
                report(f"{_prefix} — {msg}")

            filter_progress("focusing...")
            self.logger.info(f"Autofocusing filter '{fname}' (position {fid})")

            self.set_filter(fid)

            starting_pos = fdata.get("focus_position")
            if starting_pos is not None:
                self.logger.info(f"Moving focuser to last known position {starting_pos} for '{fname}'")
                self._focuser.move_absolute(starting_pos)

            self._current_filter_offset = _DUMMY_FILTER_FOCUS_OFFSETS.get(fname.lower(), 0)

            try:
                best = run_autofocus(
                    camera=cam,
                    focuser=self._focuser,
                    step_size=500,
                    num_steps=5,
                    exposure_time=1.0,
                    crop_ratio=1.0,
                    on_progress=filter_progress,
                    logger=self.logger,
                    cancel_event=cancel_event,
                    on_point=on_point,
                )
                self.filter_map[fid]["focus_position"] = best
                self.logger.info(f"Filter '{fname}' focus position: {best}")
                filter_progress(f"done (position {best})")
            finally:
                self._current_filter_offset = 0

        report("Autofocus complete for all filters")

    def supports_filter_management(self) -> bool:
        """Dummy adapter supports filter management."""
        return True

    def supports_filter_rename(self) -> bool:
        return True

    def update_filter_name(self, filter_id: str, name: str) -> bool:
        """Rename a simulated filter position."""
        try:
            fid: int | str = int(filter_id)
            if fid not in self.filter_map:
                fid = filter_id
            if fid in self.filter_map:
                self.filter_map[fid]["name"] = name
                return True
            return False
        except (ValueError, KeyError):
            return False

    def get_filter_position(self) -> int | None:
        """Get the current simulated filter wheel position."""
        return self._current_filter_position

    def set_filter(self, filter_position: int) -> bool:
        """Switch to a simulated filter position."""
        if filter_position in self.filter_map:
            fdata = self.filter_map[filter_position]
        elif str(filter_position) in self.filter_map:
            fdata = self.filter_map[str(filter_position)]  # type: ignore[arg-type]
        else:
            self.logger.warning(f"DummyAdapter: Unknown filter position {filter_position}")
            return False
        name = fdata.get("name", f"Filter {filter_position}")
        self._current_filter_position = filter_position
        self._current_filter_offset = _DUMMY_FILTER_FOCUS_OFFSETS.get(name.lower(), 0)

        focus_position = fdata.get("focus_position")
        if focus_position is not None and self._focuser:
            self.logger.info(f"DummyAdapter: Adjusting focus to {focus_position} for filter {name}")
            self._focuser.move_absolute(focus_position)

        self.logger.info(f"DummyAdapter: Filter changed to {name} (position {filter_position})")
        return True

    def set_focus(self, position: int) -> bool:
        """Move simulated focuser to absolute position."""
        if not self._focuser:
            return False
        self.logger.info(f"DummyAdapter: Moving focuser to {position}")
        return self._focuser.move_absolute(position)

    def get_focus_position(self) -> int | None:
        """Get simulated focuser position."""
        if not self._focuser:
            return None
        return self._focuser.get_position()

    def supports_direct_camera_control(self) -> bool:
        """Dummy adapter supports direct camera control."""
        return True

    def capture_preview(self, exposure_time: float, flip_horizontal: bool = False) -> str:
        """Return a synthetic preview image as a JPEG data URL."""
        import time

        from citrascope.web.preview import array_to_jpeg_data_url

        ra, dec = self._mount.get_radec()
        sigma = self._focus_dependent_psf_sigma()
        image_data, _ = self._generate_starfield(ra, dec, exposure_time, seed=int(time.time() * 1000), psf_sigma=sigma)
        return array_to_jpeg_data_url(image_data, flip_horizontal=flip_horizontal)

    def expose_camera(self, exposure_seconds: float = 1.0) -> str:
        """Simulate manual camera exposure."""
        return self.take_image("manual_test", exposure_seconds)

    def _simulate_delay(self, override_delay: float | None = None):
        """Add artificial delay if slow simulation is enabled."""
        if self.simulate_slow:
            delay = override_delay if override_delay is not None else self.slow_delay
            time.sleep(delay)
