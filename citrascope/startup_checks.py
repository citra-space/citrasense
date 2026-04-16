"""Startup self-checks for processor runtime dependencies.

These surface missing Python packages (e.g. ``astropy_healpix`` when the
user opts into the local APASS catalog) and missing system binaries
(``solve-field``, ``sex``) in the Missing Dependencies banner at the top
of the monitoring dashboard, rather than failing silently once per task
with only a log line.

The check runs once at daemon startup (settings don't hot-change without
a restart) and is gated on the settings that actually matter, so a user
who hasn't opted into local APASS or who has plate solving disabled
doesn't see warnings for things they never asked for.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrascope.settings.citrascope_settings import CitraScopeSettings


def _binary_hint(binary: str) -> str:
    """Return a platform-appropriate install hint for a missing binary.

    The banner renders the hint in a single ``<code>`` block, so the user
    should be able to copy-paste exactly one command for their platform.
    """
    hints = {
        "solve-field": {
            "darwin": "brew install astrometry-net",
            "linux": "sudo apt install astrometry.net",
        },
        "source-extractor": {
            "darwin": "brew install sextractor",
            "linux": "sudo apt install source-extractor",
        },
    }
    entry = hints.get(binary, {})
    if sys.platform.startswith("darwin"):
        return entry.get("darwin", f"Install {binary}")
    if sys.platform.startswith("linux"):
        return entry.get("linux", f"Install {binary}")
    return f"Install {binary} and ensure it's on PATH"


def check_processor_runtime_deps(settings: CitraScopeSettings) -> list[dict[str, str]]:
    """Return banner-shape dicts for every missing processor dependency.

    Each entry matches the shape consumed by the Missing Dependencies
    banner (see ``web/templates/_monitoring.html``) and the hardware
    adapter's ``get_missing_dependencies()`` output — all string values,
    with ``missing_packages`` comma-joined for multi-package cases:

        {"device_type": "processor",
         "device_name": str,
         "missing_packages": str,   # e.g. "astropy_healpix" or "a, b"
         "install_cmd": str}

    Returns ``[]`` immediately if the whole processing pipeline is
    disabled. Per-processor flags are consulted for the other gates so
    operators don't see warnings for processors they've turned off.
    """
    if not settings.processors_enabled:
        return []

    issues: list[dict[str, str]] = []
    enabled = settings.enabled_processors

    # astropy_healpix is only imported when the local APASS catalog is
    # enabled; no point warning users who haven't opted in.
    if settings.use_local_apass_catalog and enabled.get("photometry", True):
        if importlib.util.find_spec("astropy_healpix") is None:
            issues.append(
                {
                    "device_type": "processor",
                    "device_name": "Photometry Calibrator",
                    "missing_packages": "astropy_healpix",
                    "install_cmd": "uv tool install --force citrascope",
                }
            )

    if enabled.get("plate_solver", True) and shutil.which("solve-field") is None:
        issues.append(
            {
                "device_type": "processor",
                "device_name": "Plate Solver",
                "missing_packages": "solve-field",
                "install_cmd": _binary_hint("solve-field"),
            }
        )

    # Either binary satisfies the dep — package-manager naming drift.
    if enabled.get("source_extractor", True) and not (shutil.which("source-extractor") or shutil.which("sex")):
        issues.append(
            {
                "device_type": "processor",
                "device_name": "Source Extractor",
                "missing_packages": "source-extractor",
                "install_cmd": _binary_hint("source-extractor"),
            }
        )

    return issues
