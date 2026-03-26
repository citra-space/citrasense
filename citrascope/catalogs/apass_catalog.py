"""Local APASS DR10 star catalog backed by SQLite."""

from __future__ import annotations

import gzip
import hashlib
import logging
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import platformdirs

from citrascope.api.abstract_api_client import AbstractCitraApiClient
from citrascope.settings.citrascope_settings import APP_AUTHOR, APP_NAME

_logger = logging.getLogger(__name__)

# HEALPix parameters matching build_apass_catalog.py
_HEALPIX_NSIDE = 64
_HEALPIX_ORDER = "nested"

# TODO: refactor away this translation layer
# Remap DB column names to the names the photometry processor expects.
# The DB uses short names (vmag, gmag, ...) from build_apass_catalog.py
# the processor expects the APASS HTTP API column names.
_COLUMN_REMAP = {
    "ra": "radeg",
    "dec": "decdeg",
    "vmag": "Johnson_V (V)",
    "gmag": "Sloan_g (SG)",
    "rmag": "Sloan_r (SR)",
    "imag": "Sloan_i (SI)",
    "bmag": "Johnson_B (B)",
}


def _default_db_path() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=APP_AUTHOR)) / "catalogs" / "apass_dr10.db"


class ApassCatalog:
    """Local APASS DR10 star catalog for offline photometric calibration.

    The catalog is a SQLite database (~16 GB on disk, ~6.4 GB compressed
    download) fetched once from a signed CloudFront URL obtained via the
    Citra API.  After the initial download the catalog works entirely offline.
    """

    def __init__(self, db_path: Path | str | None = None, logger: logging.Logger | None = None):
        self._db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._download_thread: threading.Thread | None = None
        self.logger = logger or _logger

    @property
    def db_path(self) -> Path:
        return self._db_path

    def is_available(self) -> bool:
        """Return True if the local database file exists and is a valid SQLite file."""
        if not self._db_path.is_file():
            return False
        try:
            with open(self._db_path, "rb") as f:
                return f.read(16) == b"SQLite format 3\000"
        except OSError:
            return False

    def ensure_available(self, api_client: AbstractCitraApiClient | None = None) -> bool:
        """Download the catalog if it is not already present.

        Args:
            api_client: Authenticated Citra API client (must implement get_catalog_download_url).
                        If None, download is skipped.

        Returns:
            True if the catalog is available after this call, False otherwise.
        """
        if self.is_available():
            return True

        if api_client is None:
            self.logger.info("ApassCatalog: no API client — skipping download")
            return False

        self.logger.info("ApassCatalog: catalog not found locally, requesting download URL...")

        try:
            meta = api_client.get_catalog_download_url("apass_dr10.db.gz")
        except Exception as e:
            self.logger.warning(f"ApassCatalog: failed to get download URL: {e}")
            return False

        if meta is None:
            self.logger.warning("ApassCatalog: API returned no download URL — photometry will be unavailable")
            return False

        url = meta.get("url") if isinstance(meta, dict) else meta
        expected_sha256 = meta.get("sha256") if isinstance(meta, dict) else None

        if not url:
            self.logger.warning("ApassCatalog: empty download URL")
            return False

        return self._download(url, expected_sha256=expected_sha256)

    @property
    def is_downloading(self) -> bool:
        """Return True if a background download is currently in progress."""
        return self._download_thread is not None and self._download_thread.is_alive()

    def start_background_download(self, api_client: AbstractCitraApiClient | None = None) -> None:
        """Kick off ensure_available() in a daemon thread.

        If the catalog is already present or a download is already running,
        this is a no-op.  The photometry processor checks is_available() on
        each invocation, so it will start using the catalog automatically
        once the download finishes.
        """
        if self.is_available():
            self.logger.info("ApassCatalog: local catalog already available at %s", self._db_path)
            return
        if self.is_downloading:
            return

        def _run() -> None:
            ok = self.ensure_available(api_client)
            if ok:
                self.logger.info("ApassCatalog: background download complete — photometry now available")
            else:
                self.logger.warning("ApassCatalog: background download failed — photometry will be unavailable")

        self._download_thread = threading.Thread(target=_run, name="apass-catalog-download", daemon=True)
        self._download_thread.start()
        self.logger.info("ApassCatalog: download started in background thread")

    def _download(self, url: str, expected_sha256: str | None = None) -> bool:
        """Download and decompress the catalog from a URL.

        Supports both .gz compressed and uncompressed files. Downloads to a temp
        file first, verifies checksum if provided, then atomically renames into place.
        Checksum (if provided) is verified against the decompressed file.
        """
        import requests

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd = tempfile.NamedTemporaryFile(dir=self._db_path.parent, suffix=".tmp", delete=False)
        tmp_path = Path(tmp_fd.name)
        tmp_fd.close()
        decompressed_path: Path | None = None

        try:
            self.logger.info(f"ApassCatalog: downloading from {url[:120]} ...")

            with requests.get(url, stream=True, timeout=600, allow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (50 * 1024 * 1024) < len(chunk):
                            pct = downloaded * 100 // total
                            self.logger.info(f"ApassCatalog: downloaded {pct}% ({downloaded // (1024 * 1024)} MB)")

            self.logger.info(f"ApassCatalog: download complete ({downloaded // (1024 * 1024)} MB)")

            # Check if the downloaded file is gzip-compressed (magic bytes \x1f\x8b)
            with open(tmp_path, "rb") as f:
                is_gzip = f.read(2) == b"\x1f\x8b"

            if is_gzip:
                self.logger.info("ApassCatalog: decompressing...")
                decompressed_path = tmp_path.with_suffix(".db")
                with gzip.open(tmp_path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                tmp_path.unlink()
                tmp_path = decompressed_path
                decompressed_path = None  # now tracked by tmp_path

            if expected_sha256:
                self.logger.info("ApassCatalog: verifying checksum...")
                sha = hashlib.sha256()
                with open(tmp_path, "rb") as f:
                    for block in iter(lambda: f.read(8192), b""):
                        sha.update(block)
                actual = sha.hexdigest()
                if actual != expected_sha256:
                    self.logger.error(f"ApassCatalog: checksum mismatch (expected={expected_sha256}, got={actual})")
                    tmp_path.unlink(missing_ok=True)
                    return False

            # Note: the DB ships with indexes (idx_stars_dec_ra, idx_stars_healpix)
            # pre-built by build_apass_catalog.py — no post-download indexing needed.
            tmp_path.rename(self._db_path)
            self.logger.info(f"ApassCatalog: ready at {self._db_path}")
            return True

        except BaseException as e:
            self.logger.warning(f"ApassCatalog: download failed: {e}")
            tmp_path.unlink(missing_ok=True)
            if decompressed_path:
                decompressed_path.unlink(missing_ok=True)
            if isinstance(e, KeyboardInterrupt):
                raise
            return False

    def cone_search(self, ra: float, dec: float, radius: float = 2.0) -> pd.DataFrame:
        """Query the local catalog for stars within a cone.

        Uses HEALPix pixel pre-filter (leveraging idx_stars_healpix) to narrow
        candidates, then a Haversine post-filter for exact circular selection.
        Columns are remapped to match the names the photometry processor expects.

        The DB is built with NSIDE=64, nested ordering by build_apass_catalog.py.

        Args:
            ra: Right ascension of field center in degrees
            dec: Declination of field center in degrees
            radius: Search radius in degrees

        Returns:
            DataFrame with remapped columns: radeg, decdeg, Johnson_V (V),
            Sloan_g (SG), Sloan_r (SR), Sloan_i (SI), etc.

        Raises:
            RuntimeError: If the catalog database is not available.
        """
        if not self.is_available():
            raise RuntimeError(f"APASS catalog not available at {self._db_path}")

        import math

        import astropy.units as u
        from astropy_healpix import HEALPix  # type: ignore[reportMissingImports]

        # Find all HEALPix pixels overlapping the search cone
        hp = HEALPix(nside=_HEALPIX_NSIDE, order=_HEALPIX_ORDER)
        pixels = hp.cone_search_lonlat(ra * u.deg, dec * u.deg, radius * u.deg)  # type: ignore[reportAttributeAccessIssue]
        pixel_list = [int(p) for p in pixels]

        if not pixel_list:
            return pd.DataFrame()

        # Query using HEALPix index (much faster than bounding-box for large catalogs)
        placeholders = ",".join(["?"] * len(pixel_list))
        query = f"SELECT * FROM stars WHERE healpix IN ({placeholders})"

        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query(query, conn, params=pixel_list)
        finally:
            conn.close()

        if df.empty:
            return df

        # Haversine post-filter for exact circular selection
        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)
        cat_ra = np.radians(df["ra"].values)
        cat_dec = np.radians(df["dec"].values)

        dlat = cat_dec - dec_rad
        dlon = cat_ra - ra_rad
        a = np.sin(dlat / 2) ** 2 + math.cos(dec_rad) * np.cos(cat_dec) * np.sin(dlon / 2) ** 2
        angular_dist = 2 * np.arcsin(np.sqrt(a))

        mask = angular_dist <= math.radians(radius)
        result = df[mask].reset_index(drop=True)

        result = result.rename(columns=_COLUMN_REMAP)  # type: ignore[arg-type]
        self.logger.info(
            f"ApassCatalog: cone search at RA={ra:.3f}, Dec={dec:.3f}, radius={radius:.3f} deg -> {len(result)} stars"
        )

        return result
