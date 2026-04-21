"""Tests for the local APASS catalog module."""

import gzip
import hashlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.units as u
import pandas as pd
import pytest
from astropy_healpix import HEALPix

from citrasense.catalogs.apass_catalog import ApassCatalog

_HEALPIX_NSIDE = 64
_HEALPIX_ORDER = "nested"


def _healpix_for(ra: float, dec: float) -> int:
    """Return the HEALPix pixel (NSIDE=64, nested) containing (ra, dec)."""
    hp = HEALPix(nside=_HEALPIX_NSIDE, order=_HEALPIX_ORDER)
    return int(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg))


@pytest.fixture
def tiny_db(tmp_path):
    """Create a minimal APASS SQLite database matching the real build_apass_catalog.py schema."""
    db_path = tmp_path / "apass_dr10.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE stars (
            ra       REAL NOT NULL,
            dec      REAL NOT NULL,
            healpix  INTEGER NOT NULL,
            bmag     REAL,
            vmag     REAL,
            umag     REAL,
            gmag     REAL,
            rmag     REAL,
            imag     REAL,
            zmag     REAL,
            bmag_err REAL,
            vmag_err REAL,
            umag_err REAL,
            gmag_err REAL,
            rmag_err REAL,
            imag_err REAL,
            zmag_err REAL
        )""")
    conn.execute("CREATE INDEX idx_stars_dec_ra ON stars(dec, ra)")
    conn.execute("CREATE INDEX idx_stars_healpix ON stars(healpix)")
    # Star positions with HEALPix computed at runtime to match build_apass_catalog.py (NSIDE=64, nested)
    star_positions = [
        (180.001, 45.001, 11.0, 10.5, 11.5, 10.8, 10.2, 10.0, 10.3, 0.02, 0.01, 0.03, 0.02, 0.01, 0.01, 0.02),
        (180.011, 45.011, 11.5, 11.0, 12.0, 11.3, 10.7, 10.5, 10.8, 0.02, 0.01, 0.03, 0.02, 0.01, 0.01, 0.02),
        (180.021, 44.991, 12.5, 12.0, 13.0, 12.3, 11.7, 11.5, 11.8, 0.03, 0.02, 0.04, 0.03, 0.02, 0.02, 0.03),
        (179.981, 45.021, 10.0, 9.5, 10.5, 9.8, 9.2, 9.0, 9.3, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01),
        (190.0, 50.0, 8.5, 8.0, 9.0, 8.3, 7.7, 7.5, 7.8, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01),
    ]
    stars = [(ra, dec, _healpix_for(ra, dec), *mags) for ra, dec, *mags in star_positions]
    conn.executemany(
        "INSERT INTO stars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        stars,
    )
    conn.commit()
    conn.close()
    return db_path


def test_is_available_true(tiny_db):
    cat = ApassCatalog(db_path=tiny_db)
    assert cat.is_available() is True


def test_is_available_false(tmp_path):
    cat = ApassCatalog(db_path=tmp_path / "nonexistent.db")
    assert cat.is_available() is False


def test_is_available_rejects_non_sqlite(tmp_path):
    """A gzip-compressed or otherwise invalid file should not pass is_available()."""
    bad_file = tmp_path / "apass_dr10.db"
    bad_file.write_bytes(b"\x1f\x8b" + b"\x00" * 100)  # gzip header
    cat = ApassCatalog(db_path=bad_file)
    assert cat.is_available() is False


def test_cone_search_returns_nearby_stars(tiny_db):
    cat = ApassCatalog(db_path=tiny_db)
    df = cat.cone_search(ra=180.0, dec=45.0, radius=1.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "radeg" in df.columns
    assert "decdeg" in df.columns
    assert "Johnson_V (V)" in df.columns
    assert "Sloan_g (SG)" in df.columns
    assert "Sloan_r (SR)" in df.columns
    assert "Sloan_i (SI)" in df.columns


def test_cone_search_excludes_distant_stars(tiny_db):
    cat = ApassCatalog(db_path=tiny_db)
    # Use a very small radius that only captures the star at (180.001, 45.001)
    df = cat.cone_search(ra=180.001, dec=45.001, radius=0.005)
    assert len(df) == 1


def test_cone_search_empty_field(tiny_db):
    cat = ApassCatalog(db_path=tiny_db)
    df = cat.cone_search(ra=0.0, dec=-80.0, radius=1.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_cone_search_not_available(tmp_path):
    cat = ApassCatalog(db_path=tmp_path / "nonexistent.db")
    with pytest.raises(RuntimeError, match="not available"):
        cat.cone_search(ra=180.0, dec=45.0, radius=1.0)


def test_cone_search_magnitude_values_correct(tiny_db):
    """Verify remapped magnitude columns contain the right data."""
    cat = ApassCatalog(db_path=tiny_db)
    df = cat.cone_search(ra=180.001, dec=45.001, radius=0.01)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Johnson_V (V)"] == pytest.approx(10.5)
    assert row["Sloan_g (SG)"] == pytest.approx(10.8)
    assert row["Sloan_r (SR)"] == pytest.approx(10.2)
    assert row["Sloan_i (SI)"] == pytest.approx(10.0)


# --- Download / ensure_available tests ---


def _make_tiny_sqlite(path: Path) -> bytes:
    """Create a minimal SQLite DB at path and return its raw bytes."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE stars (ra REAL, dec REAL, healpix INTEGER)")
    conn.commit()
    conn.close()
    return path.read_bytes()


class TestEnsureAvailable:
    """Tests for ensure_available and _download."""

    def test_already_available_skips_download(self, tiny_db):
        cat = ApassCatalog(db_path=tiny_db)
        assert cat.ensure_available(api_client=None) is True

    def test_no_api_client_returns_false(self, tmp_path):
        cat = ApassCatalog(db_path=tmp_path / "missing.db")
        assert cat.ensure_available(api_client=None) is False

    def test_api_returns_none_url(self, tmp_path):
        cat = ApassCatalog(db_path=tmp_path / "missing.db")
        api = MagicMock()
        api.get_catalog_download_url.return_value = None
        assert cat.ensure_available(api_client=api) is False

    def test_download_uncompressed(self, tmp_path):
        """_download writes an uncompressed file directly."""
        db_bytes = _make_tiny_sqlite(tmp_path / "source.db")

        cat = ApassCatalog(db_path=tmp_path / "catalog" / "apass_dr10.db")

        with patch("requests.get") as mock_stream:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": str(len(db_bytes))}
            mock_response.iter_content.return_value = [db_bytes]
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_response

            result = cat._download("https://example.com/apass_dr10.db")

        assert result is True
        assert cat.is_available() is True

    def test_download_gzip_decompresses(self, tmp_path):
        """_download detects gzip magic bytes and decompresses."""
        db_bytes = _make_tiny_sqlite(tmp_path / "source.db")
        gz_bytes = gzip.compress(db_bytes)

        cat = ApassCatalog(db_path=tmp_path / "catalog" / "apass_dr10.db")

        with patch("requests.get") as mock_stream:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": str(len(gz_bytes))}
            mock_response.iter_content.return_value = [gz_bytes]
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_response

            result = cat._download("https://example.com/apass_dr10.db.gz?Signature=abc")

        assert result is True
        assert cat.is_available() is True

    def test_download_checksum_mismatch_rejects(self, tmp_path):
        """_download rejects file when checksum doesn't match."""
        db_bytes = _make_tiny_sqlite(tmp_path / "source.db")

        cat = ApassCatalog(db_path=tmp_path / "catalog" / "apass_dr10.db")

        with patch("requests.get") as mock_stream:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": str(len(db_bytes))}
            mock_response.iter_content.return_value = [db_bytes]
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_response

            result = cat._download("https://example.com/db", expected_sha256="bad_hash")

        assert result is False
        assert cat.is_available() is False

    def test_download_checksum_match_accepts(self, tmp_path):
        """_download accepts file when checksum matches."""
        db_bytes = _make_tiny_sqlite(tmp_path / "source.db")
        expected = hashlib.sha256(db_bytes).hexdigest()

        cat = ApassCatalog(db_path=tmp_path / "catalog" / "apass_dr10.db")

        with patch("requests.get") as mock_stream:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": str(len(db_bytes))}
            mock_response.iter_content.return_value = [db_bytes]
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_response

            result = cat._download("https://example.com/db", expected_sha256=expected)

        assert result is True
        assert cat.is_available() is True

    def test_download_cleans_up_on_failure(self, tmp_path):
        """Temp files are cleaned up when download fails."""
        cat = ApassCatalog(db_path=tmp_path / "catalog" / "apass_dr10.db")

        with patch("requests.get") as mock_stream:
            mock_stream.side_effect = ConnectionError("network down")

            result = cat._download("https://example.com/db")

        assert result is False
        # No temp files left behind
        catalog_dir = tmp_path / "catalog"
        if catalog_dir.exists():
            assert not list(catalog_dir.glob("*.tmp"))


class TestBackgroundDownload:
    """Tests for start_background_download (threaded)."""

    def test_skips_when_already_available(self, tiny_db):
        cat = ApassCatalog(db_path=tiny_db)
        cat.start_background_download(api_client=None)
        assert not cat.is_downloading

    def test_skips_when_already_downloading(self, tmp_path):
        cat = ApassCatalog(db_path=tmp_path / "missing.db")
        # Simulate an in-flight download thread
        import threading

        blocker = threading.Event()
        cat._download_thread = threading.Thread(target=blocker.wait, daemon=True)
        cat._download_thread.start()
        try:
            assert cat.is_downloading
            cat.start_background_download(api_client=MagicMock())
            # Should not have replaced the existing thread
            assert cat._download_thread.is_alive()
        finally:
            blocker.set()
            cat._download_thread.join(timeout=2)

    def test_download_runs_in_background(self, tmp_path):
        """start_background_download returns immediately; catalog becomes available async."""
        db_bytes = _make_tiny_sqlite(tmp_path / "source.db")
        cat = ApassCatalog(db_path=tmp_path / "catalog" / "apass_dr10.db")

        with patch("requests.get") as mock_stream:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": str(len(db_bytes))}
            mock_response.iter_content.return_value = [db_bytes]
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_response

            api = MagicMock()
            api.get_catalog_download_url.return_value = {"url": "https://example.com/db", "sha256": None}

            cat.start_background_download(api_client=api)
            assert cat._download_thread is not None
            cat._download_thread.join(timeout=5)

        assert cat.is_available()
        assert not cat.is_downloading
