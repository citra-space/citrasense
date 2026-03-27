"""Annotated image processor — renders satellite annotations on captured images."""

from __future__ import annotations

import io
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
_POWER_EXPONENT = 3.0

_MATCH_COLOR = (237, 79, 0)  # #ED4F00 — confirmed satellite matches
_PREDICTION_COLOR = (238, 162, 1)  # #EEA201 — TLE-predicted positions
_DETECTION_COLOR = (154, 186, 56)  # #9ABA38 — satellite detections (elongated sources)
_STAR_COLOR = (64, 151, 138)  # #40978A — star detections (point-like sources)
_TEXT_BG_COLOR = (0, 0, 0)  # Black overlay strip
_TEXT_COLOR = (255, 255, 255)  # White text

_ANNOTATED_WIDTH = 2500

_FONT_SIZE = 22
_HEADER_FONT_SIZE = 42
_STROKE_WIDTH = 3
_IMAGE_FORMAT = "JPEG"
_IMAGE_QUALITY = 80

_STAR_RADIUS = 36
_DETECTION_RADIUS = 30
_PREDICTION_RADIUS = 27
_MATCH_RADIUS = 24
_LABEL_GAP = 3

_MAX_STAR_MARKERS = 50
_MAX_DETECTION_MARKERS = 200
_ELONGATION_THRESHOLD = 1.5


class AnnotatedImageProcessor(AbstractImageProcessor):
    """Renders a stretched PNG with four concentric annotation layers overlaid.

    Reads satellite_matcher_debug.json and detected_sources to draw (outer to inner):
    - Teal circles: Plate Solver Stars (point-like sources, elongation < 1.5)
    - Yellow-green circles: Plate Solver Detections (elongated sources / satellite streaks)
    - Yellow dashed circles + labels: TLE Predicted positions
    - Orange circles + labels: Satellite Matched (confirmed matches)

    Includes a header strip with satellite name, timestamp, match count,
    and an inline color legend.

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
            original_height = img.height
            pixel_scale = 1.0
            if img.width > _ANNOTATED_WIDTH:
                pixel_scale = _ANNOTATED_WIDTH / img.width
                new_h = max(1, int(img.height * pixel_scale))
                _lanczos = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]
                img = img.resize((_ANNOTATED_WIDTH, new_h), _lanczos)

            wcs = self._load_wcs(context.working_image_path)
            debug = self._load_debug_json(context.working_dir)

            matched_sats, unmatched_preds, epoch_str, match_count = self._parse_annotations(debug)

            draw = ImageDraw.Draw(img)
            font = self._get_font()

            if wcs is not None:
                if context.detected_sources is not None:
                    self._draw_stars(draw, wcs, context.detected_sources, original_height, pixel_scale)
                    self._draw_detections(
                        draw, wcs, context.detected_sources, original_height, context.tracking_mode, pixel_scale
                    )
                self._draw_predictions(draw, wcs, unmatched_preds, font, original_height, pixel_scale)
                self._draw_matches(draw, wcs, matched_sats, font, original_height, pixel_scale)

            task_name = context.task.satelliteName if context.task else None
            self._draw_overlay(draw, img.width, task_name, epoch_str, match_count, font)

            png_bytes = self._encode_image(img)
            preview_path = self._save_preview(png_bytes, context.image_path.parent)
            working_path = self._save_to_working_dir(png_bytes, context.working_dir)

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
    def _linear_stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Linear stretch: maps [lo, hi] linearly to [0, 255]."""
        return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _power_stretch(arr: np.ndarray, lo: float | np.floating, hi: float | np.floating) -> np.ndarray:
        """Power stretch: normalizes to [0, 1] then applies x^exponent.

        Darkens the background while keeping bright stars visible, similar
        to DS9's "power" scale setting.
        """
        normalized = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return (np.power(normalized, _POWER_EXPONENT) * 255.0).astype(np.uint8)

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

        # stretched = AnnotatedImageProcessor._linear_stretch(arr, lo, hi)
        stretched = AnnotatedImageProcessor._power_stretch(arr, lo, hi)
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
        self, draw: ImageDraw.ImageDraw, wcs: WCS, matches: list[dict], font, img_height: int, pixel_scale: float = 1.0
    ) -> None:
        for sat in matches:
            ra, dec = sat.get("ra"), sat.get("dec")
            name = sat.get("name", "?")
            if ra is None or dec is None:
                continue
            px, py = self._radec_to_pixel(wcs, ra, dec, img_height, pixel_scale)
            if px is None or py is None:
                continue
            self._draw_circle(draw, px, py, _MATCH_COLOR, _MATCH_RADIUS)
            self._draw_label_below(draw, px, py, name, _MATCH_COLOR, font)

    def _draw_predictions(
        self,
        draw: ImageDraw.ImageDraw,
        wcs: WCS,
        predictions: list[dict],
        font,
        img_height: int,
        pixel_scale: float = 1.0,
    ) -> None:
        for pred in predictions:
            ra = pred.get("predicted_ra_deg")
            dec = pred.get("predicted_dec_deg")
            name = pred.get("name", "?")
            if ra is None or dec is None:
                continue
            px, py = self._radec_to_pixel(wcs, ra, dec, img_height, pixel_scale)
            if px is None or py is None:
                continue
            self._draw_dashed_circle(draw, px, py, _PREDICTION_COLOR, _PREDICTION_RADIUS)
            self._draw_label_below(draw, px, py, name, _PREDICTION_COLOR, font)

    def _draw_stars(
        self, draw: ImageDraw.ImageDraw, wcs: WCS, sources: pd.DataFrame, img_height: int, pixel_scale: float = 1.0
    ) -> None:
        """Draw teal circles on the brightest point-like detected sources."""
        if sources.empty:
            return

        point_like = sources[sources["elongation"] < _ELONGATION_THRESHOLD]
        brightest = point_like.sort_values(by="mag").head(_MAX_STAR_MARKERS)  # type: ignore[call-overload]

        for _, row in brightest.iterrows():
            px, py = self._radec_to_pixel(wcs, row["ra"], row["dec"], img_height, pixel_scale)
            if px is None or py is None:
                continue
            self._draw_circle(draw, px, py, _STAR_COLOR, _STAR_RADIUS)

    def _draw_detections(
        self,
        draw: ImageDraw.ImageDraw,
        wcs: WCS,
        sources: pd.DataFrame,
        img_height: int,
        tracking_mode: str | None,
        pixel_scale: float = 1.0,
    ) -> None:
        """Draw yellow-green circles on elongated sources likely to be satellite streaks.

        In sidereal tracking, satellites appear as elongated streaks (elongation >= threshold).
        In rate tracking, the mount follows the satellite so it appears round while stars
        streak — satellite candidates have elongation < threshold.
        """
        if sources.empty:
            return

        if tracking_mode == "rate":
            candidates = sources[sources["elongation"] < _ELONGATION_THRESHOLD]
        else:
            candidates = sources[sources["elongation"] >= _ELONGATION_THRESHOLD]

        capped = candidates.head(_MAX_DETECTION_MARKERS)

        for _, row in capped.iterrows():
            px, py = self._radec_to_pixel(wcs, row["ra"], row["dec"], img_height, pixel_scale)
            if px is None or py is None:
                continue
            self._draw_circle(draw, px, py, _DETECTION_COLOR, _DETECTION_RADIUS)

    @staticmethod
    def _radec_to_pixel(
        wcs: WCS, ra_deg: float, dec_deg: float, img_height: int, pixel_scale: float = 1.0
    ) -> tuple[int | None, int | None]:
        try:
            result = wcs.world_to_pixel_values(ra_deg, dec_deg)
            # WCS pixel y is 0 at bottom (FITS convention); display image is
            # np.flipud'd so row 0 is at top.  Invert for display coordinates.
            # img_height must be the ORIGINAL (pre-resize) height for the flip,
            # then pixel_scale maps to the downscaled canvas.
            px = round(float(result[0]) * pixel_scale)
            py = round((img_height - 1 - float(result[1])) * pixel_scale)
            return px, py
        except Exception:
            return None, None

    @staticmethod
    def _draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, color: tuple, radius: int):
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, outline=color, width=_STROKE_WIDTH)

    @staticmethod
    def _draw_dashed_circle(
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        color: tuple,
        radius: int,
        dash_count: int = 16,
    ):
        """Draw a dashed circle as alternating arcs."""
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        arc_span = 360.0 / dash_count
        for i in range(0, dash_count, 2):
            start_angle = i * arc_span
            end_angle = start_angle + arc_span
            draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=_STROKE_WIDTH)

    @staticmethod
    def _safe_text(draw: ImageDraw.ImageDraw, xy: tuple, text: str, color: tuple, font) -> None:
        """Draw text, silently skipping if the font backend is unavailable."""
        try:
            draw.text(xy, text, fill=color, font=font)
        except Exception:
            pass

    @staticmethod
    def _draw_label_below(
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        text: str,
        color: tuple,
        font,
    ) -> None:
        """Draw a label centered horizontally below the outermost annotation circle."""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            x = cx - text_w // 2
            y = cy + _STAR_RADIUS + _LABEL_GAP
            draw.text(
                (x, y),
                text,
                fill=color,
                font=font,
            )
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
        """Draw a single-line info strip with inline color legend at the top.

        The header font starts at _HEADER_FONT_SIZE and scales down so the full
        text (info + legend) fits within the image width.  Legend entries use
        drawn circles (font-independent) instead of Unicode bullet characters.
        """
        margin = 20
        legend_items = [
            (_STAR_COLOR, "Plate Solver Stars"),
            (_DETECTION_COLOR, "Plate Solver Detections"),
            (_PREDICTION_COLOR, "TLE Predicted"),
            (_MATCH_COLOR, "Satellite Matched"),
        ]

        parts: list[str] = []
        if task_name:
            parts.append(task_name)
        if epoch_str:
            parts.append(epoch_str)
        parts.append(f"{match_count} match{'es' if match_count != 1 else ''}")
        info_text = "  |  ".join(parts) + "  |  "

        legend_text = "  ".join(f"O {label}" for _, label in legend_items)
        full_text = info_text + legend_text

        header_font, font_size = AnnotatedImageProcessor._fit_header_font(draw, full_text, img_width, margin)

        text_height = font_size
        try:
            bbox = draw.textbbox((0, 0), "Xg", font=header_font)
            text_height = bbox[3] - bbox[1]
        except Exception:
            pass
        overlay_h = text_height + 2 * margin
        draw.rectangle((0, 0, img_width, overlay_h), fill=_TEXT_BG_COLOR)
        text_y = margin

        AnnotatedImageProcessor._safe_text(draw, (margin, text_y), info_text, _TEXT_COLOR, header_font)

        try:
            bbox = draw.textbbox((0, 0), info_text, font=header_font)
            x = margin + bbox[2] - bbox[0]
        except Exception:
            x = margin + len(info_text) * (font_size * 3 // 5)

        circle_d = max(6, int(text_height * 0.55)) + 7
        circle_r = circle_d // 2
        circle_gap = max(4, circle_d // 3)
        circle_stroke = max(2, circle_d // 5)

        for color, label in legend_items:
            cy = text_y + text_height // 2 + 2
            circle_bbox = (x, cy - circle_r, x + circle_d, cy + circle_r)
            if color == _PREDICTION_COLOR:
                arc_span = 360.0 / 8
                for i in range(0, 8, 2):
                    draw.arc(
                        circle_bbox, start=i * arc_span, end=i * arc_span + arc_span, fill=color, width=circle_stroke
                    )
            else:
                draw.ellipse(circle_bbox, outline=color, width=circle_stroke)
            x += circle_d + circle_gap

            item = f"{label}  "
            AnnotatedImageProcessor._safe_text(draw, (x, text_y), item, color, header_font)
            try:
                bbox = draw.textbbox((0, 0), item, font=header_font)
                x += bbox[2] - bbox[0]
            except Exception:
                x += len(item) * (font_size * 3 // 5)

    @staticmethod
    def _fit_header_font(draw: ImageDraw.ImageDraw, full_text: str, img_width: int, margin: int) -> tuple:
        """Return (font, actual_size) scaled so full_text fits within img_width."""
        available = img_width - 2 * margin
        header_font = AnnotatedImageProcessor._load_font(_HEADER_FONT_SIZE)
        if header_font is None:
            return ImageFont.load_default(), 11

        try:
            bbox = draw.textbbox((0, 0), full_text, font=header_font)
            text_w = bbox[2] - bbox[0]
        except Exception:
            return header_font, _HEADER_FONT_SIZE

        if text_w <= available:
            return header_font, _HEADER_FONT_SIZE

        scaled = max(20, int(_HEADER_FONT_SIZE * available / text_w))
        scaled_font = AnnotatedImageProcessor._load_font(scaled)
        if scaled_font is None:
            return header_font, _HEADER_FONT_SIZE
        return scaled_font, scaled

    @staticmethod
    def _get_font():
        """Get the annotation label font."""
        return AnnotatedImageProcessor._load_font(_FONT_SIZE)

    @staticmethod
    def _load_font(size: int):
        """Load a TrueType font at the requested pixel size.

        Tries multiple strategies for cross-platform compatibility:
        1. Pillow 10.1+ bundled font with size parameter
        2. System TrueType font by name (works on macOS/Linux/Windows)
        3. System font by absolute path (common Linux locations)
        4. Bitmap fallback (fixed ~11px, ignores size — last resort)
        """
        try:
            return ImageFont.load_default(size=size)  # type: ignore[call-arg]
        except TypeError:
            pass

        for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica", "FreeSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue

        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

        return ImageFont.load_default()

    @staticmethod
    def _encode_image(img: Image.Image) -> bytes:
        """Encode the image once — all save methods write these bytes."""
        buf = io.BytesIO()
        img.save(buf, _IMAGE_FORMAT, quality=_IMAGE_QUALITY, optimize=True)
        return buf.getvalue()

    @staticmethod
    def _save_preview(png_bytes: bytes, images_dir: Path) -> Path:
        """Save annotated PNG as a single rotating preview file (overwritten each time).

        Uses atomic write (temp file + rename) to prevent serving a partially
        written image when the web endpoint reads during an overwrite.
        """
        out_path = images_dir / "latest_preview.jpg"
        tmp_path = images_dir / "latest_preview.tmp.jpg"
        tmp_path.write_bytes(png_bytes)
        tmp_path.replace(out_path)
        return out_path

    @staticmethod
    def _save_to_working_dir(png_bytes: bytes, working_dir: Path) -> Path | None:
        """Save annotated PNG in the processing working directory."""
        try:
            out_path = working_dir / "annotated.jpg"
            out_path.write_bytes(png_bytes)
            return out_path
        except Exception:
            return None
