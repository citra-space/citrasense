"""Camera device adapters."""

from citrascope.hardware.devices.camera.abstract_camera import AbstractCamera, CalibrationProfile
from citrascope.hardware.devices.camera.rpi_hq_camera import RaspberryPiHQCamera
from citrascope.hardware.devices.camera.usb_camera import UsbCamera
from citrascope.hardware.devices.camera.ximea_camera import XimeaHyperspectralCamera

__all__ = [
    "AbstractCamera",
    "CalibrationProfile",
    "RaspberryPiHQCamera",
    "UsbCamera",
    "XimeaHyperspectralCamera",
]
