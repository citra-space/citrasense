"""Tests for the AnnotatedImageProcessor."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageDraw

from citrascope.processors.builtin.annotated_image_processor import AnnotatedImageProcessor
from citrascope.processors.processor_result import ProcessingContext


def _make_context(
    tmp_path: Path,
    image_data: np.ndarray | None = None,
    has_wcs: bool = True,
    debug_json: dict | None = None,
    task_name: str | None = "ISS",
) -> ProcessingContext:
    """Build a ProcessingContext with test fixtures in tmp_path."""
    working_dir = tmp_path / "working"
    working_dir.mkdir(exist_ok=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)

    fits_path = images_dir / "test_image.fits"
    fits_path.touch()

    if debug_json is not None:
        (working_dir / "satellite_matcher_debug.json").write_text(json.dumps(debug_json))

    task = None
    if task_name:
        task = MagicMock()
        task.satelliteName = task_name

    return ProcessingContext(
        image_path=fits_path,
        working_image_path=fits_path,
        working_dir=working_dir,
        image_data=image_data if image_data is not None else np.random.randint(0, 65535, (100, 100), dtype=np.uint16),
        task=task,
        telescope_record=None,
        ground_station_record=None,
        settings=None,
        logger=MagicMock(),
    )


def _make_debug_json(
    matched: list[dict] | None = None,
    predictions_in_field: list[dict] | None = None,
    epoch: str = "2026-03-10T03:45:00Z",
) -> dict:
    """Build a minimal satellite_matcher_debug.json structure."""
    return {
        "epoch": epoch,
        "satellite_observations": matched or [],
        "predictions_in_field": predictions_in_field or [],
    }


class TestStretchToRgb:
    def test_produces_rgb_from_grayscale(self):
        data = np.random.randint(500, 3000, (64, 64), dtype=np.uint16)
        img = AnnotatedImageProcessor._stretch_to_rgb(data)
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_produces_rgb_from_3channel(self):
        data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = AnnotatedImageProcessor._stretch_to_rgb(data)
        assert img.mode == "RGB"

    def test_handles_constant_image(self):
        data = np.full((32, 32), 1000, dtype=np.uint16)
        img = AnnotatedImageProcessor._stretch_to_rgb(data)
        assert img.mode == "RGB"
        arr = np.array(img)
        assert arr.max() <= 255

    def test_output_uses_full_range(self):
        data = np.linspace(0, 65535, 10000).reshape(100, 100).astype(np.uint16)
        img = AnnotatedImageProcessor._stretch_to_rgb(data)
        arr = np.array(img)
        assert arr.max() == 255
        assert arr.min() == 0


class TestParseAnnotations:
    def test_separates_matched_and_unmatched(self):
        matched = [{"norad_id": "25544", "name": "ISS", "ra": 180.0, "dec": 45.0}]
        preds = [
            {
                "satellite_id": "25544",
                "name": "ISS",
                "predicted_ra_deg": 180.0,
                "predicted_dec_deg": 45.0,
                "in_field": True,
            },
            {
                "satellite_id": "99999",
                "name": "GOES-16",
                "predicted_ra_deg": 181.0,
                "predicted_dec_deg": 44.0,
                "in_field": True,
            },
        ]
        debug = _make_debug_json(matched=matched, predictions_in_field=preds)

        sats, unmatched, epoch, count = AnnotatedImageProcessor._parse_annotations(debug)
        assert count == 1
        assert len(sats) == 1
        assert sats[0]["name"] == "ISS"
        assert len(unmatched) == 1
        assert unmatched[0]["name"] == "GOES-16"
        assert epoch == "2026-03-10T03:45:00Z"

    def test_empty_debug_returns_zeros(self):
        sats, unmatched, epoch, count = AnnotatedImageProcessor._parse_annotations({})
        assert count == 0
        assert sats == []
        assert unmatched == []
        assert epoch is None

    def test_all_matched_leaves_no_unmatched(self):
        matched = [
            {"norad_id": "25544", "name": "ISS", "ra": 180.0, "dec": 45.0},
            {"norad_id": "99999", "name": "GOES-16", "ra": 181.0, "dec": 44.0},
        ]
        preds = [
            {
                "satellite_id": "25544",
                "name": "ISS",
                "predicted_ra_deg": 180.0,
                "predicted_dec_deg": 45.0,
                "in_field": True,
            },
            {
                "satellite_id": "99999",
                "name": "GOES-16",
                "predicted_ra_deg": 181.0,
                "predicted_dec_deg": 44.0,
                "in_field": True,
            },
        ]
        debug = _make_debug_json(matched=matched, predictions_in_field=preds)

        _, unmatched, _, count = AnnotatedImageProcessor._parse_annotations(debug)
        assert count == 2
        assert unmatched == []


class TestFailOpen:
    def test_no_image_data(self, tmp_path):
        ctx = _make_context(tmp_path, image_data=None)
        ctx.image_data = None
        proc = AnnotatedImageProcessor()
        result = proc.process(ctx)
        assert result.should_upload is True
        assert result.confidence == 0.0
        assert "No image data" in result.reason

    def test_missing_debug_json_still_saves_image(self, tmp_path):
        """Without satellite_matcher_debug.json, processor saves a plain stretched image."""
        proc = AnnotatedImageProcessor()
        ctx = _make_context(tmp_path, debug_json=None)

        with patch.object(AnnotatedImageProcessor, "_load_wcs", return_value=None):
            result = proc.process(ctx)

        assert result.should_upload is True
        assert result.extracted_data.get("image_path") is not None
        assert Path(result.extracted_data["image_path"]).exists()

    def test_bad_wcs_still_saves_image(self, tmp_path):
        """With invalid WCS, annotations are skipped but the stretched image is saved."""
        proc = AnnotatedImageProcessor()
        debug = _make_debug_json(
            matched=[{"norad_id": "25544", "name": "ISS", "ra": 180.0, "dec": 45.0}],
        )
        ctx = _make_context(tmp_path, debug_json=debug)

        with patch.object(AnnotatedImageProcessor, "_load_wcs", return_value=None):
            result = proc.process(ctx)

        assert result.should_upload is True
        assert Path(result.extracted_data["image_path"]).exists()


class TestAnnotatedImageSaving:
    def test_saves_preview_per_task_and_working_copies(self, tmp_path):
        proc = AnnotatedImageProcessor()
        debug = _make_debug_json()
        ctx = _make_context(tmp_path, debug_json=debug)

        with patch.object(AnnotatedImageProcessor, "_load_wcs", return_value=None):
            result = proc.process(ctx)

        assert result.should_upload is True
        permanent = Path(result.extracted_data["image_path"])
        working = Path(result.extracted_data["working_dir_path"])
        assert permanent.exists()
        assert permanent.suffix == ".png"
        assert permanent.name == "latest_preview.png"
        assert working.exists()
        assert working.name == "annotated.png"

        per_task = ctx.image_path.parent / f"{ctx.image_path.stem}_annotated.png"
        assert per_task.exists()
        assert per_task.suffix == ".png"

    def test_output_is_valid_png(self, tmp_path):
        proc = AnnotatedImageProcessor()
        debug = _make_debug_json()
        ctx = _make_context(tmp_path, debug_json=debug)

        with patch.object(AnnotatedImageProcessor, "_load_wcs", return_value=None):
            result = proc.process(ctx)

        img = Image.open(result.extracted_data["image_path"])
        assert img.format == "PNG"
        assert img.mode == "RGB"


class TestDrawAnnotations:
    def test_draws_match_circle_at_pixel_coords(self):
        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = AnnotatedImageProcessor._get_font()

        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        matches = [{"ra": 180.0, "dec": 45.0, "name": "ISS"}]
        proc = AnnotatedImageProcessor()
        proc._draw_matches(draw, wcs, matches, font, img.height)

        arr = np.array(img)
        assert arr.sum() > 0  # something was drawn

    def test_draws_dashed_prediction_circle(self):
        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = AnnotatedImageProcessor._get_font()

        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (80.0, 80.0)

        predictions = [{"predicted_ra_deg": 181.0, "predicted_dec_deg": 44.0, "name": "GOES-16"}]
        proc = AnnotatedImageProcessor()
        proc._draw_predictions(draw, wcs, predictions, font, img.height)

        arr = np.array(img)
        assert arr.sum() > 0  # something was drawn

    def test_overlay_strip_includes_task_name_and_legend(self):
        img = Image.new("RGB", (8000, 400), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = AnnotatedImageProcessor._get_font()

        AnnotatedImageProcessor._draw_overlay(draw, 8000, "ISS", "2026-03-10T03:45:00Z", 3, font)

        arr = np.array(img)
        top_half = arr[: arr.shape[0] // 2, :, :]
        assert top_half.sum() > 0


class TestRadecToPixelFlip:
    """Verify _radec_to_pixel inverts Y for display-flipped images."""

    def test_y_is_flipped_relative_to_image_height(self):
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (50.0, 30.0)
        px, py = AnnotatedImageProcessor._radec_to_pixel(wcs, 180.0, 45.0, img_height=200)
        assert px == 50
        assert py == 200 - 1 - 30  # 169

    def test_bottom_row_maps_to_top_of_display(self):
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (10.0, 0.0)
        _px, py = AnnotatedImageProcessor._radec_to_pixel(wcs, 180.0, 45.0, img_height=100)
        assert py == 99

    def test_top_row_maps_to_bottom_of_display(self):
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (10.0, 99.0)
        _px, py = AnnotatedImageProcessor._radec_to_pixel(wcs, 180.0, 45.0, img_height=100)
        assert py == 0

    def test_wcs_error_returns_none(self):
        wcs = MagicMock()
        wcs.world_to_pixel_values.side_effect = RuntimeError("bad coord")
        px, py = AnnotatedImageProcessor._radec_to_pixel(wcs, 180.0, 45.0, img_height=100)
        assert px is None
        assert py is None


class TestDrawStars:
    """Verify star circle markers are drawn from detected_sources."""

    def test_draws_circles_on_point_sources(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        sources = pd.DataFrame({"ra": [180.0], "dec": [45.0], "mag": [-5.0], "elongation": [1.1]})
        proc = AnnotatedImageProcessor()
        proc._draw_stars(draw, wcs, sources, img.height)

        arr = np.array(img)
        assert arr.sum() > 0
        assert arr[:, :, 1].max() == 151  # _STAR_COLOR green channel (#40978A)

    def test_skips_elongated_sources(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        sources = pd.DataFrame({"ra": [180.0], "dec": [45.0], "mag": [-5.0], "elongation": [3.0]})
        proc = AnnotatedImageProcessor()
        proc._draw_stars(draw, wcs, sources, img.height)

        arr = np.array(img)
        assert arr.sum() == 0

    def test_limits_to_max_markers(self):
        import pandas as pd

        wcs = MagicMock()
        call_count = 0

        def counting_radec(wcs, ra, dec, h):
            nonlocal call_count
            call_count += 1
            return 100, 100

        sources = pd.DataFrame(
            {
                "ra": np.linspace(170, 190, 100),
                "dec": np.linspace(40, 50, 100),
                "mag": np.linspace(-10, 0, 100),
                "elongation": np.ones(100) * 1.1,
            }
        )

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        proc = AnnotatedImageProcessor()

        with patch.object(AnnotatedImageProcessor, "_radec_to_pixel", side_effect=counting_radec):
            proc._draw_stars(draw, wcs, sources, img.height)

        assert call_count == 50

    def test_empty_sources_is_noop(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()

        sources = pd.DataFrame({"ra": [], "dec": [], "mag": [], "elongation": []})
        proc = AnnotatedImageProcessor()
        proc._draw_stars(draw, wcs, sources, img.height)

        arr = np.array(img)
        assert arr.sum() == 0


class TestDrawDetections:
    """Verify satellite detection (elongated source) markers."""

    def test_draws_blue_circles_on_elongated_sources_sidereal(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        sources = pd.DataFrame({"ra": [180.0], "dec": [45.0], "mag": [-5.0], "elongation": [3.0]})
        proc = AnnotatedImageProcessor()
        proc._draw_detections(draw, wcs, sources, img.height, tracking_mode="sidereal")

        arr = np.array(img)
        assert arr.sum() > 0
        assert arr[:, :, 1].max() == 186  # _DETECTION_COLOR green channel (#9ABA38)

    def test_skips_point_sources_in_sidereal(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        sources = pd.DataFrame({"ra": [180.0], "dec": [45.0], "mag": [-5.0], "elongation": [1.1]})
        proc = AnnotatedImageProcessor()
        proc._draw_detections(draw, wcs, sources, img.height, tracking_mode="sidereal")

        arr = np.array(img)
        assert arr.sum() == 0

    def test_rate_tracking_flips_elongation_filter(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        sources = pd.DataFrame({"ra": [180.0], "dec": [45.0], "mag": [-5.0], "elongation": [1.1]})
        proc = AnnotatedImageProcessor()
        proc._draw_detections(draw, wcs, sources, img.height, tracking_mode="rate")

        arr = np.array(img)
        assert arr.sum() > 0  # point-like sources are satellite candidates in rate mode

    def test_none_tracking_mode_defaults_to_sidereal(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()
        wcs.world_to_pixel_values.return_value = (100.0, 100.0)

        sources = pd.DataFrame({"ra": [180.0], "dec": [45.0], "mag": [-5.0], "elongation": [3.0]})
        proc = AnnotatedImageProcessor()
        proc._draw_detections(draw, wcs, sources, img.height, tracking_mode=None)

        arr = np.array(img)
        assert arr.sum() > 0  # elongated sources drawn when tracking_mode is None

    def test_empty_sources_is_noop(self):
        import pandas as pd

        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        wcs = MagicMock()

        sources = pd.DataFrame({"ra": [], "dec": [], "mag": [], "elongation": []})
        proc = AnnotatedImageProcessor()
        proc._draw_detections(draw, wcs, sources, img.height, tracking_mode="sidereal")

        arr = np.array(img)
        assert arr.sum() == 0
