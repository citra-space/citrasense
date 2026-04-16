"""FastAPI router factories for the CitraScope web layer.

Each `build_*_router(ctx)` function returns an `APIRouter` whose handlers close
over the `CitraScopeWebApp` instance (`ctx`). This preserves the exact closure
semantics of the original inline route definitions in `app.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import APIRouter

from citrascope.web.routes.alignment import build_alignment_router
from citrascope.web.routes.analysis import build_analysis_router
from citrascope.web.routes.autofocus import build_autofocus_router
from citrascope.web.routes.calibration import build_calibration_router
from citrascope.web.routes.camera import build_camera_router
from citrascope.web.routes.core import build_core_router
from citrascope.web.routes.filters import build_filters_router
from citrascope.web.routes.focuser import build_focuser_router
from citrascope.web.routes.hardware import build_hardware_router
from citrascope.web.routes.jobs import build_jobs_router
from citrascope.web.routes.mount import build_mount_router
from citrascope.web.routes.safety import build_safety_router
from citrascope.web.routes.tasks import build_tasks_router
from citrascope.web.routes.websocket import build_websocket_router

if TYPE_CHECKING:
    from citrascope.web.app import CitraScopeWebApp


def build_all_routers() -> list[Callable[[CitraScopeWebApp], APIRouter]]:
    """Return all router factories in the order they should be included."""
    return [
        build_websocket_router,
        build_core_router,
        build_hardware_router,
        build_tasks_router,
        build_filters_router,
        build_focuser_router,
        build_autofocus_router,
        build_alignment_router,
        build_calibration_router,
        build_mount_router,
        build_safety_router,
        build_camera_router,
        build_analysis_router,
        build_jobs_router,
    ]


__all__ = [
    "build_alignment_router",
    "build_all_routers",
    "build_analysis_router",
    "build_autofocus_router",
    "build_calibration_router",
    "build_camera_router",
    "build_core_router",
    "build_filters_router",
    "build_focuser_router",
    "build_hardware_router",
    "build_jobs_router",
    "build_mount_router",
    "build_safety_router",
    "build_tasks_router",
    "build_websocket_router",
]
