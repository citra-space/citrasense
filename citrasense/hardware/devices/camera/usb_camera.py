"""USB camera adapter using OpenCV."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np

from citrasense.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrasense.hardware.devices.camera import AbstractCamera


def _probe_usb_cameras() -> list[dict[str, str | int]]:
    """Enumerate USB cameras via OpenCV / cv2_enumerate_cameras.

    Standalone module-level function so it can be pickled and sent to a
    subprocess by :func:`run_hardware_probe`.
    """
    import logging as _logging

    cameras: list[dict[str, str | int]] = []
    try:
        import cv2  # type: ignore[reportMissingImports]

        try:
            from cv2_enumerate_cameras import enumerate_cameras  # type: ignore[reportMissingImports]

            camera_infos = list(enumerate_cameras())
            _logging.getLogger("citrasense.UsbCamera").debug(f"cv2_enumerate_cameras found {len(camera_infos)} cameras")

            for camera_info in camera_infos:
                name = camera_info.name or f"Camera {camera_info.index}"
                cameras.append({"value": camera_info.index, "label": name})

        except ImportError:
            for index in range(10):
                cap = cv2.VideoCapture(index)
                try:
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
                        backend_str = f" ({backend})" if backend else ""
                        cameras.append({"value": index, "label": f"Camera {index} - {width}x{height}{backend_str}"})
                    else:
                        break
                finally:
                    cap.release()

        if not cameras:
            cameras.append({"value": 0, "label": "Camera 0 (default)"})

    except ImportError:
        cameras.append({"value": 0, "label": "Camera 0 (opencv-python not installed)"})
    except Exception:
        cameras.append({"value": 0, "label": "Camera 0 (default)"})

    return cameras


_USB_CAMERA_FALLBACK: list[dict[str, str | int]] = [{"value": 0, "label": "Camera 0 (default)"}]


class UsbCamera(AbstractCamera):
    """Adapter for USB cameras accessible via OpenCV.

    Supports USB cameras including guide cameras, planetary cameras, and standard webcams.
    Note: Most USB cameras have limited exposure control compared to dedicated
    astronomy cameras, but many are suitable for planetary imaging, guiding,
    and testing.

    Configuration:
        camera_index (int): Camera device index (0 for first camera)
        output_format (str): Output format - 'fits', 'png', 'jpg'
    """

    @classmethod
    def get_friendly_name(cls) -> str:
        """Return human-readable name for this camera device.

        Returns:
            Friendly display name
        """
        return "USB Camera (OpenCV)"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        """Return required Python packages.

        Returns:
            Dict with packages and install extra
        """
        return {
            "packages": ["cv2", "cv2_enumerate_cameras"],
            "install_extra": "usb-camera",
        }

    @classmethod
    def clear_camera_cache(cls) -> None:
        """Clear cached camera list to force re-detection.

        Call this from a "Scan Hardware" button or when hardware changes are expected.
        """
        cls._clear_probe_cache()

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        """Return schema for USB camera settings.

        Returns:
            List of setting schema entries
        """
        available_cameras = cls._detect_available_cameras()

        schema = [
            {
                "name": "camera_index",
                "friendly_name": "Camera",
                "type": "int",
                "default": 0,
                "description": "Select which camera to use",
                "required": False,
                "options": available_cameras,
                "group": "Camera",
            },
            {
                "name": "output_format",
                "friendly_name": "Output Format",
                "type": "str",
                "default": "fits",
                "description": "Image output format",
                "required": False,
                "options": ["fits", "png", "jpg"],
                "group": "Camera",
            },
        ]
        return cast(list[SettingSchemaEntry], schema)

    @classmethod
    def _detect_available_cameras(cls) -> list[dict[str, str | int]]:
        """Detect available USB cameras on the system.

        Uses :meth:`_cached_hardware_probe` for subprocess isolation and
        caching so a hung native call cannot freeze the web server.

        Returns:
            List of camera options as dicts with 'value' (index) and 'label' (name)
        """
        return cls._cached_hardware_probe(
            _probe_usb_cameras,
            fallback=_USB_CAMERA_FALLBACK,
            cache_ttl=float("inf"),
            timeout=10.0,
        )

    def __init__(self, logger: logging.Logger, **kwargs):
        """Initialize the USB camera.

        Args:
            logger: Logger instance
            **kwargs: Configuration parameters matching the schema
        """
        super().__init__(logger, **kwargs)

        # Camera settings
        self.camera_index = kwargs.get("camera_index", 0)
        self.output_format = kwargs.get("output_format", "fits")

        # Camera instance (lazy loaded)
        self._camera = None
        self._connected = False
        self._last_exposure_start: datetime | None = None

        # Lazy import opencv to avoid hard dependency
        self._cv2_module = None

    def connect(self) -> bool:
        """Connect to the USB camera.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Lazy import
            if self._cv2_module is None:
                import cv2  # type: ignore[reportMissingImports]

                self._cv2_module = cv2

            self.logger.info(f"Connecting to USB camera at index {self.camera_index}...")

            # Open camera
            self._camera = self._cv2_module.VideoCapture(self.camera_index)

            if not self._camera.isOpened():
                self.logger.error(f"Failed to open camera at index {self.camera_index}")
                return False

            # Get camera properties
            width = int(self._camera.get(self._cv2_module.CAP_PROP_FRAME_WIDTH))
            height = int(self._camera.get(self._cv2_module.CAP_PROP_FRAME_HEIGHT))

            self._connected = True
            self.logger.info(f"USB camera connected: {width}x{height}")
            return True

        except ImportError:
            self.logger.error("opencv-python library not found. Install with: pip install opencv-python")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to USB camera: {e}")
            return False

    def disconnect(self):
        """Disconnect from the USB camera."""
        if self._camera:
            try:
                self._camera.release()
                self.logger.info("USB camera disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting camera: {e}")

        self._camera = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check if camera is connected and responsive.

        Returns:
            True if connected, False otherwise
        """
        return self._connected and self._camera is not None and self._camera.isOpened()

    def capture_array(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        shutter_closed: bool = False,
    ) -> np.ndarray:
        """Capture one frame and return it as an ``H x W x 3`` RGB array.

        OpenCV's :meth:`VideoCapture.read` returns frames in **BGR** order;
        we convert to **RGB** before returning so the array follows the
        same convention every other camera (Moravian, ZWO, Ximea, RPi HQ)
        and downstream tool (Pillow, matplotlib, the allsky preview bus,
        FITS pipelines) already use.  Without this swap, the JPEG encoder
        in :func:`citrasense.sensors.preview_bus.array_to_jpeg_data_url`
        renders skies and skin tones with red and blue exchanged.

        Internal cv2 consumers (``cv2.imwrite``, ``cvtColor(..., BGR2*)``)
        must therefore swap back to BGR or use the ``RGB2*`` variants —
        see :meth:`take_exposure` and :meth:`_save_as_fits`.
        """
        if not self.is_connected():
            raise RuntimeError("Camera not connected")
        assert self._camera is not None
        assert self._cv2_module is not None

        try:
            self.logger.info("Capturing USB camera frame...")
            self._last_exposure_start = datetime.now(timezone.utc)
            ret, frame = self._camera.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame from USB camera")

            if binning > 1:
                height, width = frame.shape[:2]
                new_size = (width // binning, height // binning)
                frame = self._cv2_module.resize(frame, new_size, interpolation=self._cv2_module.INTER_AREA)

            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = self._cv2_module.cvtColor(frame, self._cv2_module.COLOR_BGR2RGB)

            return np.asarray(frame)

        except RuntimeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            raise RuntimeError(f"Image capture failed: {e}") from e

    def take_exposure(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        save_path: Path | None = None,
        shutter_closed: bool = False,
    ) -> Path:
        assert self._cv2_module is not None
        frame = self.capture_array(duration, gain, offset, binning, shutter_closed=shutter_closed)

        if save_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"/tmp/usb_camera_{timestamp}.{self.output_format}")

        file_extension = save_path.suffix.lower().lstrip(".")
        format_to_use = file_extension if file_extension in ["fits", "png", "jpg", "jpeg"] else self.output_format

        if format_to_use == "fits":
            self._save_as_fits(frame, save_path)
        elif format_to_use in ("png", "jpg", "jpeg"):
            # ``capture_array`` now returns RGB, but ``cv2.imwrite`` expects
            # BGR — swap once before writing so PNG/JPG files on disk
            # stay correctly colored.
            bgr = (
                self._cv2_module.cvtColor(frame, self._cv2_module.COLOR_RGB2BGR)
                if frame.ndim == 3 and frame.shape[2] == 3
                else frame
            )
            if format_to_use == "png":
                self._cv2_module.imwrite(str(save_path), bgr)
            else:
                self._cv2_module.imwrite(str(save_path), bgr, [self._cv2_module.IMWRITE_JPEG_QUALITY, 95])

        self.logger.info(f"Image saved to {save_path}")
        return save_path

    def abort_exposure(self):
        """Abort current exposure.

        Note: USB camera captures via OpenCV are instantaneous, nothing to abort.
        """
        self.logger.debug("USB camera captures are instantaneous - nothing to abort")

    def get_temperature(self) -> float | None:
        """Get camera sensor temperature.

        Returns:
            None - USB cameras accessed via OpenCV don't expose temperature readings
        """
        return None

    def set_temperature(self, temperature: float) -> bool:
        """Set target camera sensor temperature.

        Args:
            temperature: Target temperature in degrees Celsius

        Returns:
            False - USB cameras accessed via OpenCV don't support temperature control
        """
        self.logger.debug("USB cameras do not support temperature control via OpenCV")
        return False

    def start_cooling(self) -> bool:
        """Enable camera cooling system.

        Returns:
            False - USB cameras accessed via OpenCV don't have cooling systems
        """
        self.logger.debug("USB cameras do not have cooling systems via OpenCV")
        return False

    def stop_cooling(self) -> bool:
        """Disable camera cooling system.

        Returns:
            False - USB cameras accessed via OpenCV don't have cooling systems
        """
        self.logger.debug("USB cameras do not have cooling systems via OpenCV")
        return False

    def get_camera_info(self) -> dict:
        """Get camera capabilities and information.

        Returns:
            Dictionary containing camera specs
        """
        info = {
            "name": f"USB Camera {self.camera_index}",
            "type": "USB Camera",
            "has_cooling": False,
            "has_temperature_sensor": False,
            "output_format": self.output_format,
        }

        if self.is_connected() and self._camera:
            assert self._cv2_module is not None
            info["width"] = int(self._camera.get(self._cv2_module.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self._camera.get(self._cv2_module.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = int(self._camera.get(self._cv2_module.CAP_PROP_FPS))

        return info

    def _save_as_fits(self, frame, save_path: Path):
        """Save frame as FITS format.

        Args:
            frame: RGB frame (the convention :meth:`capture_array` returns).
            save_path: Path to save FITS file
        """
        assert self._cv2_module is not None

        try:
            import numpy as np
            from astropy.io import fits

            # ``capture_array`` returns RGB; use the RGB→GRAY luminance
            # conversion so the per-channel weights match the actual
            # channel ordering (cv2 applies 0.299/0.587/0.114 to channels
            # 0/1/2 in order — the wrong ordering produces a luminance
            # image with miscolored intensity).
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = self._cv2_module.cvtColor(frame, self._cv2_module.COLOR_RGB2GRAY)
            else:
                gray = frame

            hdu = fits.PrimaryHDU(gray.astype(np.uint16))
            hdu.header["INSTRUME"] = "USB Camera"
            hdu.header["CAMERA"] = f"Index {self.camera_index}"
            date_obs = self._last_exposure_start or datetime.now(timezone.utc)
            hdu.header["DATE-OBS"] = (date_obs.isoformat(), "UTC exposure start")
            hdu.writeto(save_path, overwrite=True)

        except ImportError:
            self.logger.error("astropy not installed. Install with: pip install astropy")
            raise
