"""Calibration suite generators — build job lists for one-click calibration.

Pure functions that produce ordered lists of capture-parameter dicts
from a CalibrationProfile and settings.  No side effects, no hardware calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citrascope.hardware.devices.camera.abstract_camera import CalibrationProfile

DARK_REFERENCE_EXPOSURE_S = 30.0
"""Long reference dark used for dark scaling.

One 30 s dark at the current binning gives high SNR on the thermal
signal.  At runtime the calibration processor linearly scales it to
match the science frame's actual exposure.
"""


def bias_and_dark_suite(
    profile: CalibrationProfile,
    frame_count: int,
) -> list[dict[str, Any]]:
    """Generate ordered job list: bias then dark at the current binning.

    Biases are instant (0 s, shutter closed) so they run first without
    waiting for temperature.  Darks follow, sharing a single temp-wait
    gate in CalibrationManager.

    Darks are captured at :data:`DARK_REFERENCE_EXPOSURE_S` (30 s) so
    a single master can be linearly scaled to any science exposure at
    runtime — no need to re-calibrate when changing exposure settings.

    Only the current binning is calibrated; re-run the suite after
    switching binning modes.
    """
    gain = profile.current_gain or 0
    binning = profile.current_binning
    return [
        {"frame_type": "bias", "count": frame_count, "gain": gain, "binning": binning},
        {
            "frame_type": "dark",
            "count": frame_count,
            "gain": gain,
            "binning": binning,
            "exposure_time": DARK_REFERENCE_EXPOSURE_S,
        },
    ]


def all_flats_suite(
    profile: CalibrationProfile,
    filters: list[dict[str, Any]],
    frame_count: int,
    initial_exposure: float = 1.0,
) -> list[dict[str, Any]]:
    """Generate a single interleaved flat job covering all enabled filters.

    Instead of one monolithic job per filter (which exhausts the twilight
    window on early filters), returns a single ``interleaved_flat`` job.
    The :class:`~citrascope.calibration.master_builder.MasterBuilder`
    cycles through filters in rounds, so each filter samples across the
    brightness gradient and the median stack averages out drift.
    """
    gain = profile.current_gain or 0
    binning = profile.current_binning
    return [
        {
            "frame_type": "interleaved_flat",
            "count": frame_count,
            "gain": gain,
            "binning": binning,
            "initial_exposure": initial_exposure,
            "filters": [{"position": f["position"], "name": f["name"]} for f in filters],
        }
    ]
