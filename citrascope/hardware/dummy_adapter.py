"""Dummy hardware adapter for testing without real hardware."""

import datetime
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from citrascope.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
    SettingSchemaEntry,
)
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

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger)
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

    # -- AbstractMount core ----------------------------------------------------

    def slew_to_radec(self, ra: float, dec: float) -> bool:
        """Simulate a slew by sweeping azimuth at a realistic rate.

        Uses shortest-arc to determine direction, then advances azimuth
        via start_move/stop_move so the cable wrap check sees gradual
        accumulation during the transit.  The sleep is interruptible via
        abort_slew() so the safety monitor can cut a slew short.
        """
        self._homed = False
        self._abort_event.clear()
        target_az, target_alt = _radec_to_altaz(ra, dec)

        self._snap_az()
        current_az = self._base_az % 360.0
        delta = ((target_az - current_az + 180.0) % 360.0) - 180.0

        aborted = False
        if abs(delta) > 1.0:
            self._slewing = True
            direction = "east" if delta > 0 else "west"
            self.start_move(direction)
            wait_s = abs(delta) / self._SLEW_RATE_DEG_PER_S
            aborted = self._abort_event.wait(wait_s)
            self.stop_move(direction)
            self._slewing = False

        if aborted:
            self.logger.info("Slew aborted mid-transit")
            return False

        self._ra = ra
        self._dec = dec
        self._base_az = target_az
        self._alt = target_alt
        self._ref_time = time.monotonic()
        return True

    def is_slewing(self) -> bool:
        return self._slewing

    def abort_slew(self) -> None:
        self._abort_event.set()
        self._slewing = False
        self._snap_az()
        self._moving_dir = None
        self._ra, self._dec = _altaz_to_radec(self._base_az % 360.0, self._alt)

    def get_radec(self) -> tuple[float, float]:
        if self._moving_dir is not None:
            az = self.get_azimuth()
            if az is not None:
                return _altaz_to_radec(az, self._alt)
        return self._ra, self._dec

    def start_tracking(self, rate: str | None = "sidereal") -> bool:
        self._snap_az()
        self._tracking = True
        return True

    def stop_tracking(self) -> bool:
        self._snap_az()
        self._tracking = False
        return True

    def is_tracking(self) -> bool:
        return self._tracking

    def park(self) -> bool:
        self._parked = True
        self._base_az = 0.0
        self._ref_time = time.monotonic()
        return True

    def unpark(self) -> bool:
        self._parked = False
        return True

    def is_parked(self) -> bool:
        return self._parked

    def get_mount_info(self) -> dict:
        return {"name": "Dummy Mount", "supports_sync": True}

    _HOME_AZ = 0.0

    # -- Homing ----------------------------------------------------------------

    def find_home(self) -> bool:
        """Simulate homing by slewing back toward az=0 via continuous motion.

        Uses start_move/stop_move so the cable wrap check sees the azimuth
        sweeping through intermediate positions, just like a real mount.
        Returns True once the mount reaches the home position.
        """
        self._snap_az()
        current_az = self._base_az % 360.0
        delta = ((self._HOME_AZ - current_az + 180.0) % 360.0) - 180.0

        if abs(delta) < 1.0:
            self._finish_home()
            return True

        self._abort_event.clear()
        direction = "east" if delta > 0 else "west"
        self.start_move(direction)

        travel_needed = abs(delta)
        wait_s = travel_needed / self._MOVE_RATE_DEG_PER_S
        aborted = self._abort_event.wait(wait_s + 0.05)

        self.stop_move(direction)
        if aborted:
            self.logger.info("Homing aborted mid-transit")
            return False
        self._finish_home()
        return True

    def _finish_home(self) -> None:
        self._base_az = self._HOME_AZ
        self._alt = 90.0 - abs(_OBSERVER_LAT_DEG)
        self._ra, self._dec = _altaz_to_radec(self._HOME_AZ, self._alt)
        self._ref_time = time.monotonic()
        self._tracking = False
        self._homed = True

    def is_home(self) -> bool:
        return self._homed

    # -- Safety-related overrides ---------------------------------------------

    def get_mount_mode(self) -> str:
        return "altaz"

    def get_azimuth(self) -> float | None:
        dt = time.monotonic() - self._ref_time
        if self._moving_dir is not None:
            sign = -1.0 if self._moving_dir in ("west", "south") else 1.0
            rate = self._SLEW_RATE_DEG_PER_S if self._slewing else self._MOVE_RATE_DEG_PER_S
            return (self._base_az + sign * rate * dt) % 360.0
        if self._tracking:
            return (self._base_az + self._TRACKING_DRIFT_DEG_PER_S * dt) % 360.0
        return self._base_az % 360.0

    def get_altitude(self) -> float | None:
        return self._alt

    def start_move(self, direction: str, rate: int = 7) -> bool:
        self._snap_az()
        self._moving_dir = direction
        return True

    def stop_move(self, direction: str) -> bool:
        self._snap_az()
        self._moving_dir = None
        self._ra, self._dec = _altaz_to_radec(self._base_az % 360.0, self._alt)
        return True

    def _snap_az(self) -> None:
        """Commit the current computed azimuth as the new base, resetting the reference clock."""
        self._base_az = self.get_azimuth() or self._base_az
        self._ref_time = time.monotonic()


# Synthetic camera constants — consistent across take_image() and the WCS header
# so the image geometry is self-describing.
_DUMMY_IMG_SIZE = 1024  # pixels per side
_DUMMY_PIXEL_SCALE = 6.0  # arcsec/pixel  →  ~1.7° FOV  (wide enough for Tetra3)
_DUMMY_SKY_BG = 500.0  # ADU sky background
_DUMMY_READ_NOISE = 8.0  # electrons RMS
_DUMMY_GAIN = 1.5  # electrons/ADU
_DUMMY_PSF_SIGMA_PX = 3.0 / 2.3548  # sigma from 3.0 px FWHM seeing
_DUMMY_MAG_LIMIT = 14.0  # faintest catalog star to render (Vmag)
_DUMMY_MAG_ZERO = 20.0  # instrument zero-point: V=10 → SNR~58, V=12 → SNR~9


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
        self.mount = _DummyMount(logger)
        self._mount_cache: MountStateCache | None = None

        # Set by the daemon after connecting, mirrors the real telescope_record from the API.
        # When present, take_image() derives sensor dimensions and pixel scale from it.
        self.telescope_record: dict | None = None

        self.logger.info("DummyAdapter initialized")

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
        ]

    def get_observation_strategy(self) -> ObservationStrategy:
        """Dummy adapter uses manual strategy."""
        return ObservationStrategy.MANUAL

    def perform_observation_sequence(self, task, satellite_data) -> str:
        """Not used for manual strategy."""
        raise NotImplementedError("DummyAdapter uses MANUAL strategy")

    def connect(self) -> bool:
        """Simulate connection."""
        self.logger.info("DummyAdapter: Connecting...")
        self._simulate_delay()
        self._connected = True
        self._telescope_connected = True
        self._camera_connected = True
        cache = MountStateCache(self.mount)
        cache.refresh_static()
        cache.start()
        self.mount._state_cache = cache  # type: ignore[attr-defined]
        self._mount_cache = cache
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
        success = self.mount.slew_to_radec(ra, dec)
        self._is_moving = False
        if not success:
            raise RuntimeError("Slew aborted by safety monitor")
        self.logger.info("DummyAdapter: Slew complete")

    def get_telescope_direction(self) -> tuple[float, float]:
        """Return current telescope position from the simulated mount."""
        return self.mount.get_radec()

    def abort_slew(self) -> None:
        self.mount.abort_slew()

    def telescope_is_moving(self) -> bool:
        """Check if fake telescope is moving."""
        return self._is_moving

    def home_if_needed(self) -> bool:
        if self.mount.is_home():
            self.logger.info("DummyAdapter: Mount already homed")
            return True
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        self.logger.info("DummyAdapter: Homing mount (safety-monitored)...")
        self._is_moving = True
        self.mount.find_home()
        self._is_moving = False
        self.logger.info("DummyAdapter: Mount homed")
        return True

    def home_mount(self) -> bool:
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        self.logger.info("DummyAdapter: Homing mount...")
        self._is_moving = True
        self.mount.find_home()
        self._is_moving = False
        self.logger.info("DummyAdapter: Mount homed")
        return True

    def is_mount_homed(self) -> bool:
        return self.mount.is_home()

    def select_camera(self, device_name: str) -> bool:
        """Simulate camera selection."""
        self.logger.info(f"DummyAdapter: Selected camera '{device_name}'")
        self._camera_connected = True
        return True

    def take_image(self, task_id: str, exposure_duration_seconds=1.0) -> str:
        """Simulate image capture, producing a synthetic starfield FITS."""
        self.logger.info(f"DummyAdapter: Starting {exposure_duration_seconds}s exposure for task {task_id}")
        self._simulate_delay(exposure_duration_seconds)

        timestamp = int(time.time())
        filename = f"dummy_{task_id}_{timestamp}.fits"
        filepath = self.images_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        ra, dec = self.mount.get_radec()
        image_data, wcs = self._generate_starfield(
            ra,
            dec,
            exposure_duration_seconds,
            seed=timestamp,
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
    ) -> tuple[np.ndarray, WCS]:
        """Generate a synthetic starfield from the Tycho-2 catalog with a TAN WCS.

        Queries the Tycho-2 catalog via Vizier for real stars in the current FOV,
        projects them onto the pixel grid, and renders each as a Gaussian PSF.

        Args:
            ra_center_deg: RA of the field centre in degrees.
            dec_center:    Dec of the field centre in degrees.
            exptime:       Exposure duration in seconds (scales star brightness).
            seed:          RNG seed for the noise model (use Unix timestamp).

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

        fov_deg = max(size_x, size_y) * pixel_scale / 3600.0

        # --- WCS: TAN projection centred on current pointing -----------------
        # Standard TAN (gnomonic) projection.  CDELT1 is in RA-coordinate
        # degrees per pixel; the TAN projection equations already fold in
        # cos(dec) when mapping (RA, Dec) → intermediate world coords, so
        # the on-sky pixel scale is isotropic at the field centre without any
        # additional correction here.
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [(size_x + 1) / 2.0, (size_y + 1) / 2.0]
        wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
        wcs.wcs.crval = [ra_center, dec_center]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # --- Star catalog query ----------------------------------------------
        star_ras, star_decs, star_mags = self._fetch_catalog_stars(ra_center, dec_center, fov_deg)

        # --- Sky background + noise ------------------------------------------
        dark_adu = 0.01 * exptime / _DUMMY_GAIN
        image = np.full((size_y, size_x), _DUMMY_SKY_BG + dark_adu, dtype=np.float64)
        sky_electrons = image * _DUMMY_GAIN
        image = rng.poisson(sky_electrons).astype(np.float64) / _DUMMY_GAIN
        image += rng.normal(0.0, _DUMMY_READ_NOISE / _DUMMY_GAIN, image.shape)

        # --- Render each catalog star as a Gaussian PSF ----------------------
        if len(star_ras) == 0:
            self.logger.debug("DummyAdapter: No catalog stars in FOV — returning noise-only image")
            return np.clip(image, 0, 65535).astype(np.uint16), wcs

        psf_sigma = _DUMMY_PSF_SIGMA_PX
        stamp_r = max(int(5 * psf_sigma), 15)

        pixel_coords = wcs.all_world2pix(np.column_stack([star_ras, star_decs]), 0)

        # Drop any stars whose projection produced NaN (behind tangent plane, etc.)
        valid = np.all(np.isfinite(pixel_coords), axis=1)
        pixel_coords = pixel_coords[valid]
        star_mags = star_mags[valid]

        n_rendered = 0
        for (xp, yp), mag in zip(pixel_coords, star_mags, strict=False):
            flux_e = 10.0 ** ((_DUMMY_MAG_ZERO - mag) / 2.5) * exptime
            total_adu = flux_e / _DUMMY_GAIN

            xi, yi = round(xp), round(yp)

            # Build a small stamp and blur it to the PSF shape
            stamp = np.zeros((2 * stamp_r + 1, 2 * stamp_r + 1))
            stamp[stamp_r, stamp_r] = total_adu
            stamp = gaussian_filter(stamp, sigma=psf_sigma)
            if stamp.sum() > 0:
                stamp *= total_adu / stamp.sum()

            # Blit stamp onto the full image, clipping to chip boundaries.
            # Compute the overlap between the stamp and the image array.
            x0, y0 = xi - stamp_r, yi - stamp_r
            ix0 = max(0, x0)
            iy0 = max(0, y0)
            ix1 = min(size_x, x0 + stamp.shape[1])
            iy1 = min(size_y, y0 + stamp.shape[0])
            sx0 = ix0 - x0
            sy0 = iy0 - y0
            sx1 = sx0 + (ix1 - ix0)
            sy1 = sy0 + (iy1 - iy0)

            if ix1 > ix0 and iy1 > iy0:
                image[iy0:iy1, ix0:ix1] += stamp[sy0:sy1, sx0:sx1]
                n_rendered += 1

        self.logger.debug(f"DummyAdapter: Rendered {n_rendered}/{len(star_mags)} catalog stars")

        image = np.clip(image, 0, 65535).astype(np.uint16)
        return image, wcs

    _pixelemon_star_table: np.ndarray | None = None

    @classmethod
    def _load_star_table(cls) -> np.ndarray:
        """Load the Tycho-2 star table from Pixelemon's bundled catalog (cached)."""
        if cls._pixelemon_star_table is not None:
            return cls._pixelemon_star_table
        import pixelemon  # type: ignore[import-untyped]

        cat_path = Path(pixelemon.__file__).parent / "tyc_db_to_40_deg.npz"
        if not cat_path.exists():
            raise FileNotFoundError(f"Pixelemon star catalog not found at {cat_path}")
        table: np.ndarray = np.load(str(cat_path))["star_table"]
        cls._pixelemon_star_table = table
        return table

    def _fetch_catalog_stars(
        self,
        ra: float,
        dec: float,
        fov_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query Pixelemon's local Tycho-2 catalog for stars in the current FOV.

        Uses the same star database that Tetra3 uses for plate solving, so
        synthetic images are solvable without any network access.

        Returns:
            Tuple of (ras_deg, decs_deg, mags) as float arrays.
        """
        star_table = self._load_star_table()

        # star_table columns: [ra_rad, dec_rad, ux, uy, uz, mag]
        all_ra_deg = np.degrees(star_table[:, 0])
        all_dec_deg = np.degrees(star_table[:, 1])
        all_mag = star_table[:, 5]

        # Coarse box filter (cheap) then angular distance refinement.
        # The RA tolerance must be widened by 1/cos(dec) near the poles.
        cos_dec = max(math.cos(math.radians(dec)), 0.05)
        ra_tol = fov_deg / cos_dec
        dec_tol = fov_deg

        d_ra = np.abs((all_ra_deg - ra + 180.0) % 360.0 - 180.0)
        d_dec = np.abs(all_dec_deg - dec)
        box = (d_ra < ra_tol) & (d_dec < dec_tol) & (all_mag < _DUMMY_MAG_LIMIT)

        ras = all_ra_deg[box]
        decs = all_dec_deg[box]
        mags = all_mag[box]

        if len(ras) == 0:
            self.logger.warning(
                f"DummyAdapter: No catalog stars brighter than V={_DUMMY_MAG_LIMIT} "
                f"around RA={ra:.3f}, Dec={dec:.3f} — generating empty field"
            )
            return np.array([]), np.array([]), np.array([])

        self.logger.debug(
            f"DummyAdapter: Local Tycho-2 returned {len(ras)} stars "
            f"(Vmag < {_DUMMY_MAG_LIMIT}) around RA={ra:.3f}, Dec={dec:.3f}"
        )
        return ras, decs, mags

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        """Simulate setting tracking rate."""
        self.logger.info(f"DummyAdapter: Setting tracking rate RA={ra_rate} arcsec/s, Dec={dec_rate} arcsec/s")
        self._tracking_rate = (ra_rate, dec_rate)

    def get_tracking_rate(self) -> tuple[float, float]:
        """Return current fake tracking rate."""
        return self._tracking_rate

    def perform_alignment(self, target_ra: float, target_dec: float) -> bool:
        """Simulate plate solving alignment."""
        self.logger.info(f"DummyAdapter: Performing alignment to RA={target_ra}°, Dec={target_dec}°")
        self._simulate_delay()
        self.mount.slew_to_radec(target_ra + 0.001, target_dec + 0.001)
        self.logger.info("DummyAdapter: Alignment successful")
        return True

    def supports_autofocus(self) -> bool:
        """Dummy adapter supports autofocus."""
        return True

    def do_autofocus(
        self,
        target_ra: float | None = None,
        target_dec: float | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Simulate autofocus routine."""
        if target_ra is not None and target_dec is not None:
            self.logger.info(f"DummyAdapter: Starting autofocus on target RA={target_ra:.4f}, Dec={target_dec:.4f}")
        else:
            self.logger.info("DummyAdapter: Starting autofocus (default target)")

        filters = [f for f in self.filter_map.values() if f.get("enabled", True)] if self.filter_map else []
        total = len(filters) or 1
        for idx, f in enumerate(filters or [{"name": "Default"}], 1):
            if on_progress:
                on_progress(f"Filter {idx}/{total}: {f['name']} — focusing...")
            self._simulate_delay(1.0)
            if on_progress:
                on_progress(f"Filter {idx}/{total}: {f['name']} — done")

        self.logger.info("DummyAdapter: Autofocus complete")

    def supports_filter_management(self) -> bool:
        """Dummy adapter supports filter management."""
        return True

    def supports_direct_camera_control(self) -> bool:
        """Dummy adapter supports direct camera control."""
        return True

    def expose_camera(self, exposure_seconds: float = 1.0) -> str:
        """Simulate manual camera exposure."""
        return self.take_image("manual_test", exposure_seconds)

    def _simulate_delay(self, override_delay: float | None = None):
        """Add artificial delay if slow simulation is enabled."""
        if self.simulate_slow:
            delay = override_delay if override_delay is not None else self.slow_delay
            time.sleep(delay)
