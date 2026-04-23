"""Tests for photometry processor with local APASS catalog."""

import sqlite3
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy_healpix import HEALPix

from citrasense.catalogs.apass_catalog import ApassCatalog
from citrasense.pipelines.optical.optical_processing_context import OpticalProcessingContext
from citrasense.pipelines.optical.photometry_processor import PhotometryProcessor

_HEALPIX_NSIDE = 64
_HEALPIX_ORDER = "nested"


def _healpix_for(ra: float, dec: float) -> int:
    """Return the HEALPix pixel (NSIDE=64, nested) containing (ra, dec)."""
    hp = HEALPix(nside=_HEALPIX_NSIDE, order=_HEALPIX_ORDER)
    return int(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg))


@pytest.fixture
def catalog_with_stars(tmp_path):
    """Create a catalog DB with known stars at positions matching the mock SExtractor sources."""
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

    # Stars placed within 1 arcsecond of the mock SExtractor sources so cross-matching succeeds.
    # HEALPix values computed at runtime so they match what cone_search expects.
    star_positions = [
        (180.001, 45.001, 11.0, 10.5, 11.5, 10.8, 10.2, 10.0, 10.3),
        (180.011, 45.011, 11.5, 11.0, 12.0, 11.3, 10.7, 10.5, 10.8),
        (180.021, 44.991, 12.5, 12.0, 13.0, 12.3, 11.7, 11.5, 11.8),
        (179.981, 45.021, 10.0, 9.5, 10.5, 9.8, 9.2, 9.0, 9.3),
        # Extra stars not near any source
        (180.5, 45.5, 13.5, 13.0, 14.0, 13.3, 12.7, 12.5, 12.8),
        (179.5, 44.5, 14.5, 14.0, 15.0, 14.3, 13.7, 13.5, 13.8),
    ]
    rows = [
        (
            ra,
            dec,
            _healpix_for(ra, dec),
            bmag,
            vmag,
            None,
            gmag,
            rmag,
            imag,
            None,
            0.02,
            0.01,
            None,
            0.02,
            0.01,
            0.01,
            None,
        )
        for ra, dec, bmag, vmag, _, gmag, rmag, imag, _ in star_positions
    ]
    conn.executemany(
        "INSERT INTO stars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return ApassCatalog(db_path=db_path)


@pytest.fixture
def fits_image_with_wcs(tmp_path):
    """Create a minimal FITS image with WCS centered on (180, 45)."""
    hdu = fits.PrimaryHDU(data=np.zeros((100, 100), dtype=np.float32))
    hdu.header["NAXIS1"] = 100
    hdu.header["NAXIS2"] = 100
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CRVAL2"] = 45.0
    hdu.header["CRPIX1"] = 50.0
    hdu.header["CRPIX2"] = 50.0
    hdu.header["CDELT1"] = -0.01
    hdu.header["CDELT2"] = 0.01
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    path = tmp_path / "test.fits"
    hdu.writeto(path, overwrite=True)
    return path


def _make_context(tmp_path, fits_path, catalog, task=None):
    """Build a minimal ProcessingContext."""
    working_dir = tmp_path / "working"
    working_dir.mkdir(exist_ok=True)
    catalog_lines = [
        "# SExtractor catalog",
        "1 50.0 50.0 0.0 -5.0 0.01 0 1000.0 180.001 45.001 2.5",
        "2 51.0 51.0 0.0 -4.7 0.02 0 800.0 180.011 45.011 2.3",
        "3 52.0 52.0 0.0 -4.4 0.03 0 600.0 180.021 44.991 2.1",
        "4 53.0 53.0 0.0 -4.1 0.04 0 500.0 179.981 45.021 2.4",
    ]
    (working_dir / "output.cat").write_text("\n".join(catalog_lines))
    if task is None:
        task = MagicMock()
        task.assigned_filter_name = "Clear"
    return OpticalProcessingContext(
        image_path=fits_path,
        working_image_path=fits_path,
        working_dir=working_dir,
        image_data=None,
        task=task,
        settings=None,
        apass_catalog=catalog,
    )


def test_photometry_uses_local_catalog(catalog_with_stars, fits_image_with_wcs, tmp_path):
    """Photometry processor uses context.apass_catalog instead of HTTP."""
    context = _make_context(tmp_path, fits_image_with_wcs, catalog_with_stars)

    processor = PhotometryProcessor()
    result = processor.process(context)

    assert result.should_upload is True
    assert result.extracted_data.get("num_calibration_stars", 0) >= 3
    assert "zero_point" in result.extracted_data


def test_photometry_falls_back_to_http_when_catalog_unavailable(fits_image_with_wcs, tmp_path):
    """Photometry falls back to AAVSO HTTP when no local catalog is available."""
    context = _make_context(tmp_path, fits_image_with_wcs, catalog=None)

    processor = PhotometryProcessor()
    with patch("requests.post") as mock_post:
        mock_post.side_effect = ConnectionError("mocked: no network in test")
        result = processor.process(context)

    assert result.should_upload is True
    assert result.confidence == 0.0
    assert "failed" in result.reason.lower()


def test_photometry_falls_back_to_http_when_catalog_file_missing(fits_image_with_wcs, tmp_path):
    """Photometry falls back to AAVSO HTTP when ApassCatalog points to a nonexistent file."""
    missing_catalog = ApassCatalog(db_path=tmp_path / "nonexistent.db")
    context = _make_context(tmp_path, fits_image_with_wcs, catalog=missing_catalog)

    processor = PhotometryProcessor()
    with patch("requests.post") as mock_post:
        mock_post.side_effect = ConnectionError("mocked: no network in test")
        result = processor.process(context)

    assert result.should_upload is True
    assert result.confidence == 0.0
    assert "failed" in result.reason.lower()
