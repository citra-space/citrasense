"""Annotated image processor — renders satellite annotations on captured images."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageFont

from citrascope.processors.abstract_processor import AbstractImageProcessor
from citrascope.processors.processor_result import ProcessingContext, ProcessorResult

if TYPE_CHECKING:
    import pandas as pd

_STRETCH_LO_PERCENTILE = 2.0
_STRETCH_HI_PERCENTILE = 98.0

_MATCH_COLOR = (0, 255, 255)  # Cyan for confirmed matches
_PREDICTION_COLOR = (255, 255, 0)  # Yellow for unmatched predictions
_TEXT_BG_COLOR = (0, 0, 0)  # Black overlay strip
_TEXT_COLOR = (255, 255, 255)  # White text

_FONT_SIZE = 18
_CIRCLE_RADIUS = 20
_LABEL_OFFSET = 6
_OVERLAY_HEIGHT = 36
_STROKE_WIDTH = 2
_IMAGE_FORMAT = "PNG"

_STAR_COLOR = (0, 255, 0)
_STAR_MARKER_SIZE = 6
_MAX_STAR_MARKERS = 50
_STAR_ELONGATION_LIMIT = 1.5


class AnnotatedImageProcessor(AbstractImageProcessor):
    """Renders a stretched PNG with satellite match annotations overlaid.

    Reads the satellite_matcher_debug.json from the working directory to draw:
    - Cyan circles + labels for confirmed satellite matches
    - Yellow dashed circles + labels for predicted-but-unmatched satellites
    - Text strip at the top with satellite name, timestamp, and match count

    Fail-open: if WCS or debug data is missing, saves a plain stretched image
    (or skips entirely) and never blocks upload.
    """

    name = "annotated_image"
    friendly_name = "Annotated Image"
    description = "Renders satellite annotations on captured images for visual review"

    def process(self, context: ProcessingContext) -> ProcessorResult:
        start_time = time.time()

        if context.image_data is None:
            return self._skip("No image data available", start_time)

        try:
            img = self._stretch_to_rgb(context.image_data)
            wcs = self._load_wcs(context.working_image_path)
            debug = self._load_debug_json(context.working_dir)

            matched_sats, unmatched_preds, epoch_str, match_count = self._parse_annotations(debug)

            draw = ImageDraw.Draw(img)
            font = self._get_font()

            if wcs is not None:
                self._draw_matches(draw, wcs, matched_sats, font, img.height)
                self._draw_predictions(draw, wcs, unmatched_preds, font, img.height)
                if context.detected_sources is not None:
                    self._draw_stars(draw, wcs, context.detected_sources, img.height)

            task_name = context.task.satelliteName if context.task else None
            self._draw_overlay(draw, img.width, task_name, epoch_str, match_count, font)

            preview_path = self._save_preview(img, context.image_path.parent)
            self._save_per_task(img, context.image_path.parent, context.image_path.stem)
            working_path = self._save_to_working_dir(img, context.working_dir)

            elapsed = time.time() - start_time
            return ProcessorResult(
                should_upload=True,
                extracted_data={
                    "image_path": str(preview_path),
                    "working_dir_path": str(working_path) if working_path else None,
                },
                confidence=1.0,
                reason=f"Annotated image saved in {elapsed:.1f}s ({match_count} matches drawn)",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            if context.logger:
                context.logger.warning(f"Annotated image processor failed: {e}")
            return ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason=f"Annotation failed: {e!s}",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )

    def _skip(self, reason: str, start_time: float) -> ProcessorResult:
        return ProcessorResult(
            should_upload=True,
            extracted_data={},
            confidence=0.0,
            reason=reason,
            processing_time_seconds=time.time() - start_time,
            processor_name=self.name,
        )

    @staticmethod
    def _stretch_to_rgb(data: np.ndarray) -> Image.Image:
        """Auto-stretch a raw pixel array to an 8-bit RGB PIL Image."""
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr

        lo = np.percentile(gray, _STRETCH_LO_PERCENTILE)
        hi = np.percentile(gray, _STRETCH_HI_PERCENTILE)
        if hi <= lo:
            hi = lo + 1.0

        stretched = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        stretched = np.flipud(stretched)

        if stretched.ndim == 2:
            return Image.fromarray(stretched, mode="L").convert("RGB")
        return Image.fromarray(stretched, mode="RGB")

    @staticmethod
    def _load_wcs(fits_path: Path) -> WCS | None:
        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header  # type: ignore[union-attr]
                if "CRVAL1" not in header:
                    return None
                return WCS(header)
        except Exception:
            return None

    @staticmethod
    def _load_debug_json(working_dir: Path) -> dict:
        debug_path = working_dir / "satellite_matcher_debug.json"
        if not debug_path.exists():
            return {}
        try:
            return json.loads(debug_path.read_text())
        except Exception:
            return {}

    @staticmethod
    def _parse_annotations(debug: dict) -> tuple[list[dict], list[dict], str | None, int]:
        """Extract matched satellites and unmatched predictions from debug data.

        Returns (matched_sats, unmatched_predictions, epoch_str, match_count).
        """
        observations = debug.get("satellite_observations", [])
        matched_ids = {obs.get("norad_id") for obs in observations}
        match_count = len(observations)

        predictions_in_field = debug.get("predictions_in_field", [])
        unmatched = [
            p for p in predictions_in_field if p.get("satellite_id") not in matched_ids and "predicted_ra_deg" in p
        ]

        epoch_str = debug.get("epoch")
        return observations, unmatched, epoch_str, match_count

    def _draw_matches(
        self, draw: ImageDraw.ImageDraw, wcs: WCS, matches: list[dict], font, img_height: int
    ) -> None:
        for sat in matches:
            ra, dec = sat.get("ra"), sat.get("dec")
            name = sat.get("name", "?")
            if ra is None or dec is None:
                continue
            px, py = self._radec_to_pixel(wcs, ra, dec, img_height)
            if px is None or py is None:
                continue
            self._draw_circle(draw, px, py, _MATCH_COLOR)
            self._safe_text(draw, (px + _CIRCLE_RADIUS + _LABEL_OFFSET, py - _FONT_SIZE // 4), name, _MATCH_COLOR, font)

    def _draw_predictions(
        self, draw: ImageDraw.ImageDraw, wcs: WCS, predictions: list[dict], font, img_height: int
    ) -> None:
        for pred in predictions:
            ra = pred.get("predicted_ra_deg")
            dec = pred.get("predicted_dec_deg")
            name = pred.get("name", "?")
            if ra is None or dec is None:
                continue
            px, py = self._radec_to_pixel(wcs, ra, dec, img_height)
            if px is None or py is None:
                continue
            self._draw_dashed_circle(draw, px, py, _PREDICTION_COLOR)
            self._safe_text(
                draw, (px + _CIRCLE_RADIUS + _LABEL_OFFSET, py - _FONT_SIZE // 4), name, _PREDICTION_COLOR, font
            )

    def _draw_stars(
        self, draw: ImageDraw.ImageDraw, wcs: WCS, sources: pd.DataFrame, img_height: int
    ) -> None:
        """Draw green crosshair markers on the brightest point-like detected sources."""
        if sources.empty:
            return

        point_like = sources[sources["elongation"] < _STAR_ELONGATION_LIMIT]
        brightest = point_like.nsmallest(_MAX_STAR_MARKERS, "mag")

        s = _STAR_MARKER_SIZE
        for _, row in brightest.iterrows():
            px, py = self._radec_to_pixel(wcs, row["ra"], row["dec"], img_height)
            if px is None or py is None:
                continue
            draw.line((px - s, py, px + s, py), fill=_STAR_COLOR, width=_STROKE_WIDTH)
            draw.line((px, py - s, px, py + s), fill=_STAR_COLOR, width=_STROKE_WIDTH)

    @staticmethod
    def _radec_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float, img_height: int) -> tuple[int | None, int | None]:
        try:
            result = wcs.world_to_pixel_values(ra_deg, dec_deg)
            px = round(float(result[0]))
            py = img_height - 1 - round(float(result[1]))
            return px, py
        except Exception:
            return None, None

    @staticmethod
    def _draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, color: tuple):
        draw.ellipse(
            (cx - _CIRCLE_RADIUS, cy - _CIRCLE_RADIUS, cx + _CIRCLE_RADIUS, cy + _CIRCLE_RADIUS),
            outline=color,
            width=_STROKE_WIDTH,
        )

    @staticmethod
    def _draw_dashed_circle(
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        color: tuple,
        dash_count: int = 16,
    ):
        """Draw a dashed circle as alternating arcs."""
        arc_span = 360.0 / dash_count
        for i in range(0, dash_count, 2):
            start_angle = i * arc_span
            end_angle = start_angle + arc_span
            draw.arc(
                (cx - _CIRCLE_RADIUS, cy - _CIRCLE_RADIUS, cx + _CIRCLE_RADIUS, cy + _CIRCLE_RADIUS),
                start=start_angle,
                end=end_angle,
                fill=color,
                width=_STROKE_WIDTH,
            )

    @staticmethod
    def _safe_text(draw: ImageDraw.ImageDraw, xy: tuple, text: str, color: tuple, font) -> None:
        """Draw text, silently skipping if the font backend is unavailable."""
        try:
            draw.text(xy, text, fill=color, font=font)
        except Exception:
            pass

    @staticmethod
    def _draw_overlay(
        draw: ImageDraw.ImageDraw,
        img_width: int,
        task_name: str | None,
        epoch_str: str | None,
        match_count: int,
        font,
    ):
        """Draw an info strip at the top of the image."""
        draw.rectangle((0, 0, img_width, _OVERLAY_HEIGHT), fill=_TEXT_BG_COLOR)

        parts: list[str] = []
        if task_name:
            parts.append(task_name)
        if epoch_str:
            parts.append(epoch_str)
        parts.append(f"{match_count} match{'es' if match_count != 1 else ''}")

        text = "  |  ".join(parts)
        text_y = max(2, (_OVERLAY_HEIGHT - _FONT_SIZE) // 2)
        AnnotatedImageProcessor._safe_text(draw, (10, text_y), text, _TEXT_COLOR, font)

    @staticmethod
    def _get_font():
        """Get a font for annotations. Uses Pillow's bundled font (no system deps)."""
        try:
            return ImageFont.load_default(size=_FONT_SIZE)  # type: ignore[call-arg]
        except TypeError:
            return ImageFont.load_default()
        except Exception:
            return None

    @staticmethod
    def _save_preview(img: Image.Image, images_dir: Path) -> Path:
        """Save annotated PNG as a single rotating preview file (overwritten each time).

        Uses atomic write (temp file + rename) to prevent serving a partially
        written image when the web endpoint reads during an overwrite.
        """
        out_path = images_dir / "latest_preview.png"
        tmp_path = images_dir / "latest_preview.tmp.png"
        img.save(str(tmp_path), _IMAGE_FORMAT)
        tmp_path.replace(out_path)
        return out_path

    @staticmethod
    def _save_per_task(img: Image.Image, images_dir: Path, stem: str) -> None:
        """Save a per-task annotated PNG alongside the original FITS in the images directory."""
        try:
            img.save(str(images_dir / f"{stem}_annotated.png"), _IMAGE_FORMAT)
        except Exception:
            pass

    @staticmethod
    def _save_to_working_dir(img: Image.Image, working_dir: Path) -> Path | None:
        """Save annotated PNG in the processing working directory."""
        try:
            out_path = working_dir / "annotated.png"
            img.save(str(out_path), _IMAGE_FORMAT)
            return out_path
        except Exception:
            return None
