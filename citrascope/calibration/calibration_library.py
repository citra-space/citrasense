"""CalibrationLibrary — master frame storage and retrieval.

Manages FITS master frames (bias, dark, flat) on disk, keyed by camera
identity and imaging parameters.  Stateless lookups — every query reads
from the filesystem so there's no in-memory cache to invalidate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import platformdirs
from astropy.io import fits  # type: ignore[attr-defined]

_APP_NAME = "citrascope"
_APP_AUTHOR = "citra-space"

logger = logging.getLogger("citrascope")


def _default_calibration_root() -> Path:
    return Path(platformdirs.user_data_dir(_APP_NAME, appauthor=_APP_AUTHOR)) / "calibration"


def resolve_camera_id(header: fits.Header) -> str:
    """Extract camera identity from a FITS header.

    Prefers CAMSER (serial number) over INSTRUME (model name).
    """
    serial = str(header.get("CAMSER", "")).strip()
    if serial:
        return serial
    return str(header.get("INSTRUME", "unknown")).strip()


class CalibrationLibrary:
    """Manages storage and retrieval of master calibration frames."""

    DARK_TEMP_TOLERANCE_C = 1.0

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or _default_calibration_root()
        self._masters_dir = self._root / "masters"
        self._tmp_dir = self._root / "tmp"
        self._masters_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def masters_dir(self) -> Path:
        return self._masters_dir

    @property
    def tmp_dir(self) -> Path:
        return self._tmp_dir

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(value: str) -> str:
        """Sanitise a string for use in a filename."""
        return value.replace(" ", "_").replace("/", "-").replace("\\", "-")

    @classmethod
    def _read_mode_slug(cls, read_mode: str) -> str:
        """Produce a compact filename-safe slug from a read mode name.

        '12-bit Slow' → '12bit-Slow', '' → 'default'.
        """
        if not read_mode:
            return "default"
        return cls._safe_name(read_mode).replace(" ", "-")

    @classmethod
    def _bias_filename(cls, camera_id: str, gain: int, binning: int, read_mode: str = "") -> str:
        cam = cls._safe_name(camera_id)
        rm = cls._read_mode_slug(read_mode)
        return f"master_bias_{cam}_g{gain}_bin{binning}_{rm}.fits"

    @classmethod
    def _dark_filename(
        cls,
        camera_id: str,
        gain: int,
        binning: int,
        exposure_time: float,
        temperature: float | None,
        read_mode: str = "",
    ) -> str:
        cam = cls._safe_name(camera_id)
        rm = cls._read_mode_slug(read_mode)
        exp_ms = round(exposure_time * 1000)
        if temperature is not None:
            temp_str = f"{temperature:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
            return f"master_dark_{cam}_g{gain}_bin{binning}_{rm}_{exp_ms}ms_{temp_str}C.fits"
        return f"master_dark_{cam}_g{gain}_bin{binning}_{rm}_{exp_ms}ms_noTemp.fits"

    @classmethod
    def _flat_filename(cls, camera_id: str, gain: int, binning: int, filter_name: str, read_mode: str = "") -> str:
        cam = cls._safe_name(camera_id)
        rm = cls._read_mode_slug(read_mode)
        filt = cls._safe_name(filter_name) if filter_name else "nofilter"
        return f"master_flat_{cam}_g{gain}_bin{binning}_{rm}_{filt}.fits"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_master(
        self,
        frame_type: str,
        camera_id: str,
        data: np.ndarray,
        gain: int,
        binning: int,
        exposure_time: float = 0.0,
        temperature: float | None = None,
        filter_name: str = "",
        ncombine: int = 0,
        camera_model: str = "",
        read_mode: str = "",
        bias_subtracted: bool | None = None,
    ) -> Path:
        """Write a master frame to the library and return its path."""
        if frame_type == "bias":
            name = self._bias_filename(camera_id, gain, binning, read_mode)
        elif frame_type == "dark":
            name = self._dark_filename(camera_id, gain, binning, exposure_time, temperature, read_mode)
        elif frame_type == "flat":
            name = self._flat_filename(camera_id, gain, binning, filter_name, read_mode)
        else:
            raise ValueError(f"Unknown frame_type: {frame_type}")

        path = self._masters_dir / name

        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdr = hdu.header
        hdr["CALTYPE"] = (frame_type.upper(), "Calibration frame type")
        hdr["NCOMBINE"] = (ncombine, "Number of frames combined")
        hdr["INSTRUME"] = (camera_model or camera_id, "Camera model")
        hdr["CAMSER"] = (camera_id, "Camera serial or model id")
        hdr["GAIN"] = (gain, "Camera gain")
        hdr["XBINNING"] = (binning, "Horizontal binning")
        hdr["READMODE"] = (read_mode or "default", "Camera read mode")
        hdr["DATE-OBS"] = (datetime.now(timezone.utc).isoformat(), "Master creation time UTC")

        if frame_type in ("dark",) and exposure_time > 0:
            hdr["EXPTIME"] = (exposure_time, "Exposure time in seconds")
        if temperature is not None:
            hdr["CCD-TEMP"] = (temperature, "Sensor temperature in C")
        if bias_subtracted is not None:
            hdr["BIASSUB"] = (bias_subtracted, "Bias subtracted during master build")
        if filter_name:
            hdr["FILTER"] = (filter_name, "Filter name")

        hdu.writeto(path, overwrite=True)
        logger.info("Saved master %s → %s", frame_type, path.name)
        return path

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_master_bias(self, camera_id: str, gain: int, binning: int, read_mode: str = "") -> Path | None:
        name = self._bias_filename(camera_id, gain, binning, read_mode)
        path = self._masters_dir / name
        return path if path.exists() else None

    def get_master_dark(
        self,
        camera_id: str,
        gain: int,
        binning: int,
        temperature: float | None = None,
        read_mode: str = "",
    ) -> Path | None:
        """Find the best dark master for dark-scaling.

        Exact match on camera_id, gain, binning, read_mode.  When
        *temperature* is provided, candidates must be within
        ``DARK_TEMP_TOLERANCE_C``.  Among candidates that pass the
        temperature gate, the **longest** exposure is preferred because
        it gives the best signal-to-noise on the thermal component used
        for dark scaling.

        Falls back to temperature-agnostic darks (those built by cameras
        without a temperature sensor) when no temperature-matched dark is
        found or when *temperature* is ``None``.

        The caller reads the ``EXPTIME`` header of the returned file to
        know the reference exposure for the scaling ratio.
        """
        rm = self._read_mode_slug(read_mode)
        prefix = f"master_dark_{self._safe_name(camera_id)}_g{gain}_bin{binning}_{rm}_"
        temp_matched: list[tuple[float, float, Path]] = []
        no_temp: list[tuple[float, Path]] = []

        for p in self._masters_dir.glob(f"{prefix}*.fits"):
            try:
                with fits.open(p) as hdul:
                    hdr = hdul[0].header  # type: ignore[index]
                    d_exp = float(hdr.get("EXPTIME", 0.0))
                    d_temp = hdr.get("CCD-TEMP")
                    if d_temp is None:
                        no_temp.append((d_exp, p))
                        continue
                    if temperature is not None:
                        if abs(float(d_temp) - temperature) > self.DARK_TEMP_TOLERANCE_C:
                            continue
                        temp_penalty = abs(float(d_temp) - temperature)
                        temp_matched.append((temp_penalty, d_exp, p))
            except Exception:
                logger.debug("Skipping unreadable dark master: %s", p, exc_info=True)

        if temp_matched:
            temp_matched.sort(key=lambda t: (t[0], -t[1]))
            return temp_matched[0][2]
        if no_temp:
            no_temp.sort(key=lambda t: -t[0])
            return no_temp[0][1]
        return None

    def get_master_flat(
        self,
        camera_id: str,
        gain: int,
        binning: int,
        filter_name: str,
        read_mode: str = "",
    ) -> Path | None:
        name = self._flat_filename(camera_id, gain, binning, filter_name, read_mode)
        path = self._masters_dir / name
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_master(
        self,
        frame_type: str,
        camera_id: str,
        gain: int,
        binning: int,
        exposure_time: float = 0.0,
        temperature: float | None = None,
        filter_name: str = "",
        read_mode: str = "",
    ) -> bool:
        """Delete a specific master frame. Returns True if deleted."""
        if frame_type == "bias":
            name = self._bias_filename(camera_id, gain, binning, read_mode)
        elif frame_type == "dark":
            name = self._dark_filename(camera_id, gain, binning, exposure_time, temperature, read_mode)
        elif frame_type == "flat":
            name = self._flat_filename(camera_id, gain, binning, filter_name, read_mode)
        else:
            return False

        path = self._masters_dir / name
        if path.exists():
            path.unlink()
            logger.info("Deleted master %s: %s", frame_type, path.name)
            return True
        return False

    # ------------------------------------------------------------------
    # Status / inventory
    # ------------------------------------------------------------------

    def get_library_status(self, camera_id: str) -> dict[str, Any]:
        """Return an inventory of masters for a given camera.

        Returns a dict with keys: bias, darks, flats — each containing
        lists of metadata dicts for the UI.
        """
        safe_cam = self._safe_name(camera_id)
        result: dict[str, Any] = {"bias": [], "darks": [], "flats": []}

        for p in sorted(self._masters_dir.glob(f"master_*_{safe_cam}_*.fits")):
            try:
                with fits.open(p) as hdul:
                    hdr = hdul[0].header  # type: ignore[index]
                    cal_type = str(hdr.get("CALTYPE", "")).lower()
                    entry: dict[str, Any] = {
                        "filename": p.name,
                        "date": str(hdr.get("DATE-OBS", "")),
                        "ncombine": int(hdr.get("NCOMBINE", 0)),
                        "gain": int(hdr.get("GAIN", 0)),
                        "binning": int(hdr.get("XBINNING", 1)),
                        "read_mode": str(hdr.get("READMODE", "")),
                    }
                    if cal_type == "bias":
                        result["bias"].append(entry)
                    elif cal_type == "dark":
                        entry["exposure_time"] = float(hdr.get("EXPTIME", 0))
                        raw_temp = hdr.get("CCD-TEMP")
                        entry["temperature"] = float(raw_temp) if raw_temp is not None else None
                        result["darks"].append(entry)
                    elif cal_type == "flat":
                        entry["filter"] = str(hdr.get("FILTER", ""))
                        result["flats"].append(entry)
            except Exception:
                logger.debug("Skipping unreadable master: %s", p, exc_info=True)

        return result

    def has_any_masters(self, camera_id: str) -> bool:
        """Quick check whether *any* masters exist for this camera."""
        safe_cam = self._safe_name(camera_id)
        return any(self._masters_dir.glob(f"master_*_{safe_cam}_*.fits"))

    # ------------------------------------------------------------------
    # Temp file management
    # ------------------------------------------------------------------

    def cleanup_tmp(self) -> None:
        """Remove all temporary raw frames (call after master built)."""
        for f in self._tmp_dir.iterdir():
            if f.is_file():
                f.unlink()
