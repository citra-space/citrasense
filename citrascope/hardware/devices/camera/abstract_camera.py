"""Abstract camera device interface."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from citrascope.hardware.devices.abstract_hardware_device import AbstractHardwareDevice

if TYPE_CHECKING:
    from citrascope.hardware.devices.filter_wheel import AbstractFilterWheel


class AbstractCamera(AbstractHardwareDevice):
    """Abstract base class for camera devices.

    Provides a common interface for controlling imaging cameras including
    CCDs, CMOS sensors, and hyperspectral cameras.
    """

    @abstractmethod
    def take_exposure(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        save_path: Path | None = None,
    ) -> Path:
        """Capture an image exposure.

        Args:
            duration: Exposure duration in seconds
            gain: Camera gain setting (device-specific units)
            offset: Camera offset/black level setting
            binning: Pixel binning factor (1=no binning, 2=2x2, etc.)
            save_path: Optional path to save the image (if None, use default)

        Returns:
            Path to the saved image file
        """
        pass

    @abstractmethod
    def abort_exposure(self):
        """Abort the current exposure if one is in progress."""
        pass

    @abstractmethod
    def get_temperature(self) -> float | None:
        """Get the current camera sensor temperature.

        Returns:
            Temperature in degrees Celsius, or None if not available
        """
        pass

    @abstractmethod
    def set_temperature(self, temperature: float) -> bool:
        """Set the target camera sensor temperature.

        Args:
            temperature: Target temperature in degrees Celsius

        Returns:
            True if temperature setpoint accepted, False otherwise
        """
        pass

    @abstractmethod
    def start_cooling(self) -> bool:
        """Enable camera cooling system.

        Returns:
            True if cooling started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_cooling(self) -> bool:
        """Disable camera cooling system.

        Returns:
            True if cooling stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_camera_info(self) -> dict:
        """Get camera capabilities and information.

        Returns:
            Dictionary containing camera specs (resolution, pixel size, bit depth, etc.)
        """
        pass

    def is_hyperspectral(self) -> bool:
        """Indicates whether this camera captures hyperspectral data.

        Hyperspectral cameras capture multiple spectral bands simultaneously
        (e.g., snapshot mosaic sensors like Ximea MQ series).

        Returns:
            bool: True if hyperspectral camera, False otherwise (default)
        """
        return False

    def get_integrated_filter_wheel(self) -> AbstractFilterWheel | None:
        """Return an integrated filter wheel if the camera has one built in.

        Override in subclasses where the filter wheel is physically part of
        the camera and shares its control handle (e.g. Moravian Cx/Gx).
        DirectHardwareAdapter calls this after camera connect to auto-detect
        integrated filter wheels without requiring separate user configuration.
        """
        return None

    def get_preferred_file_extension(self) -> str:
        """Get the preferred file extension for saved images.

        This method allows each camera to define what file format it wants
        to use, without the hardware adapter needing to know camera internals.

        Returns:
            File extension string without the dot (e.g., 'fits', 'png', 'jpg')
        """
        # Default implementation: use output_format if available, otherwise FITS
        return getattr(self, "output_format", "fits")
