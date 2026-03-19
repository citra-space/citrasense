"""Satellite association processor using TLE propagation."""

import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import get_body_barycentric_posvel
from astropy.io import fits
from astropy.time import Time as AstropyTime
from keplemon import time as ktime
from keplemon.bodies import Observatory, Satellite
from keplemon.elements import TLE
from keplemon.enums import ReferenceFrame
from scipy.spatial import KDTree

from citrascope.processors.abstract_processor import AbstractImageProcessor
from citrascope.processors.artifact_writer import dump_json, dump_processor_result
from citrascope.processors.processor_result import ProcessingContext, ProcessorResult

from .processor_dependencies import normalize_fits_timestamp, read_source_catalog

_ELONGATION_THRESHOLD = 1.5
_FIELD_RADIUS_DEG = 2.0
_MATCH_RADIUS_DEG = 1.0 / 60.0  # 1 arcminute
_STAR_MATCH_TOLERANCE_DEG = 1.0 / 3600.0  # 1 arcsecond — tight match for star subtraction


class SatelliteMatcherProcessor(AbstractImageProcessor):
    """
    Satellite association processor using TLE propagation.

    Propagates TLEs for target satellite, predicts position at image timestamp,
    and matches detected sources with predicted positions. Requires all previous
    processors to have run successfully.

    Typical processing time: 1-2 seconds.
    """

    name = "satellite_matcher"
    friendly_name = "Satellite Matcher"
    description = "Match detected sources with TLE predictions (requires full pipeline)"

    @staticmethod
    def _subtract_known_stars(candidates: pd.DataFrame, working_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Remove sources that matched APASS catalog stars in the photometry step.

        Reads photometry_crossmatch.csv (written by PhotometryProcessor) and removes
        any candidate whose detected position matches a crossmatched source within
        the star-match tolerance.

        We match against detected-source positions (ra/dec) rather than APASS catalog
        positions (radeg/decdeg) because real astrometric offsets between WCS-solved
        detections and catalog entries are 18-33 arcseconds -- far too large for a
        tight proximity match.  The detected positions come from the same SExtractor
        run as our candidates, so the match is sub-pixel identical.
        """
        crossmatch_path = working_dir / "photometry_crossmatch.csv"
        if not crossmatch_path.exists():
            return candidates, {"source": "skipped", "reason": "photometry_crossmatch.csv not found"}

        try:
            xmatch = pd.read_csv(crossmatch_path)
        except Exception:
            return candidates, {"source": "skipped", "reason": "failed to parse photometry_crossmatch.csv"}

        if "ra" not in xmatch.columns or "dec" not in xmatch.columns:
            return candidates, {"source": "skipped", "reason": "crossmatch CSV missing ra/dec columns"}

        if xmatch.empty:
            return candidates, {"source": "skipped", "reason": "crossmatch CSV empty"}

        before = len(candidates)
        star_coords = xmatch[["ra", "dec"]].values
        star_tree = KDTree(star_coords)

        cand_coords = candidates[["ra", "dec"]].values
        dists, _ = star_tree.query(cand_coords)
        keep_mask = np.asarray(dists) >= _STAR_MATCH_TOLERANCE_DEG

        filtered = candidates[keep_mask].copy()
        removed = before - len(filtered)

        stats: dict[str, Any] = {
            "source": "photometry_crossmatch.csv",
            "catalog_stars_in_crossmatch": len(xmatch),
            "candidates_before": before,
            "candidates_after": len(filtered),
            "catalog_stars_removed": removed,
        }
        return filtered, stats

    def _parse_fits_timestamp(self, timestamp_str: str) -> ktime.Epoch:
        """Parse FITS DATE-OBS timestamp into a Keplemon Epoch.

        Args:
            timestamp_str: Timestamp string from FITS header (ISO format)

        Returns:
            ktime.Epoch in UTC
        """
        dt = datetime.fromisoformat(normalize_fits_timestamp(timestamp_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return ktime.Epoch.from_datetime(dt)

    def _match_satellites(
        self, sources: pd.DataFrame, context: ProcessingContext, tracking_mode: str = "rate"
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Propagate TLEs and match detected sources with predicted satellite positions.

        Returns (observations, debug_info) where debug_info contains the full diagnostic
        bundle for satellite_matcher_debug.json.
        """
        # GEO/slow-mover detection: skip elongation filter when the target barely moves,
        # because point-like GEO satellites are indistinguishable from stars by shape.
        is_slow_mover = False
        try:
            slew_ahead = (context.pointing_report or {}).get("slew_ahead", {})
            is_slow_mover = bool(slew_ahead.get("is_slow_mover", False))
        except (AttributeError, TypeError):
            pass

        elongation_filter_applied = not is_slow_mover

        debug: dict[str, Any] = {
            "tracking_mode": tracking_mode,
            "is_slow_mover": is_slow_mover,
            "elongation_filter_applied": elongation_filter_applied,
            "elongation_threshold": _ELONGATION_THRESHOLD,
            "field_radius_deg": _FIELD_RADIUS_DEG,
            "match_radius_arcmin": _MATCH_RADIUS_DEG * 60.0,
        }

        if elongation_filter_applied:
            if tracking_mode == "rate":
                potential_sats = sources[sources["elongation"] < _ELONGATION_THRESHOLD].copy()
                star_like_count = int((sources["elongation"] >= _ELONGATION_THRESHOLD).sum())
            else:
                potential_sats = sources[sources["elongation"] >= _ELONGATION_THRESHOLD].copy()
                star_like_count = int((sources["elongation"] < _ELONGATION_THRESHOLD).sum())
        else:
            potential_sats = sources.copy()
            star_like_count = 0

        elong_vals = sources["elongation"]
        debug["source_classification"] = {
            "total_sources": len(sources),
            "satellite_candidate_count": len(potential_sats),
            "star_like_count": star_like_count,
            "elongation_min": float(elong_vals.min()) if len(elong_vals) else None,
            "elongation_max": float(elong_vals.max()) if len(elong_vals) else None,
            "elongation_median": float(elong_vals.median()) if len(elong_vals) else None,
            "elongation_mean": float(elong_vals.mean()) if len(elong_vals) else None,
        }

        # Subtract known catalog stars from candidates
        potential_sats, star_sub_stats = self._subtract_known_stars(potential_sats, context.working_dir)
        debug["star_subtraction"] = star_sub_stats

        if potential_sats.empty:
            debug["early_exit"] = "no satellite candidates after elongation filtering and star subtraction"
            return [], debug

        # Observer location
        try:
            if not context.location_service:
                raise RuntimeError("No location service available")
            location = context.location_service.get_current_location()
            obs = Observatory(location["latitude"], location["longitude"], location.get("altitude", 0) / 1000.0)
        except Exception as e:
            raise RuntimeError(f"Failed to get observer location: {e}") from e

        # Image metadata from FITS header
        with fits.open(context.working_image_path) as hdul:
            primary = hdul[0]
            assert isinstance(primary, fits.PrimaryHDU)
            header = primary.header
            timestamp_str = header.get("DATE-OBS")
            if not timestamp_str:
                raise RuntimeError("No DATE-OBS in FITS header")
            exptime = float(header.get("EXPTIME", 0.0))  # type: ignore[arg-type]
            ra_center = float(header.get("CRVAL1", 0.0))  # type: ignore[arg-type]
            dec_center = float(header.get("CRVAL2", 0.0))  # type: ignore[arg-type]

        # Offset to mid-exposure for better satellite position prediction
        epoch = self._parse_fits_timestamp(str(timestamp_str))
        if exptime > 0:
            # to_datetime() returns naive; tag as UTC so from_datetime() doesn't assume local TZ
            mid_dt = epoch.to_datetime().replace(tzinfo=timezone.utc) + timedelta(seconds=exptime / 2.0)
            epoch = ktime.Epoch.from_datetime(mid_dt)
            timestamp_str = mid_dt.isoformat()

        debug["field_center"] = {"ra_deg": ra_center, "dec_deg": dec_center}
        debug["epoch"] = str(timestamp_str)
        debug["exptime"] = exptime
        debug["mid_exposure_offset_s"] = exptime / 2.0 if exptime > 0 else 0.0

        # Sun position (km, J2000/ECI) via astropy ERFA — no external ephemeris file required
        _t = AstropyTime(epoch.to_datetime())
        _sun_bary, _ = get_body_barycentric_posvel("sun", _t)
        _earth_bary, _ = get_body_barycentric_posvel("earth", _t)
        sun_pos_km = (_sun_bary.xyz - _earth_bary.xyz).to(u.km).value  # type: ignore[union-attr]

        # Observer position (km, J2000)
        obs_state = obs.get_state_at_epoch(epoch).to_frame(ReferenceFrame.J2000)
        obs_pos = obs_state.position

        # Build elset list: prefer cache, fall back to the task's single TLE
        elsets = (context.elset_cache.get_elsets() if context.elset_cache else []) or []
        elset_source = "cache" if elsets else "task_fallback"
        if not elsets:
            if not context.task:
                raise RuntimeError("No task context available for satellite matching")
            most_recent_elset = context.satellite_data.get("most_recent_elset") if context.satellite_data else None
            if not most_recent_elset:
                raise RuntimeError("No TLE data available in satellite_data")
            tle_data = most_recent_elset.get("tle", [])
            if len(tle_data) < 2:
                raise RuntimeError("Invalid TLE format")
            elsets = [{"satellite_id": context.task.satelliteId, "name": context.task.satelliteName, "tle": tle_data}]

        debug["elset_count"] = len(elsets)
        debug["elset_source"] = elset_source

        # Target satellite comparison: pointing TLE vs cache TLE
        target_sat_id = context.task.satelliteId if context.task else None
        target_section: dict[str, Any] = {"satellite_id": target_sat_id}
        if context.satellite_data:
            mre = context.satellite_data.get("most_recent_elset")
            target_section["pointing_tle"] = mre.get("tle") if mre else None
            target_section["pointing_elset_epoch"] = mre.get("creationEpoch") if mre else None
        else:
            target_section["pointing_tle"] = None
            target_section["pointing_tle_note"] = "satellite_data not available in processing context"

        cache_match = next((e for e in elsets if e.get("satellite_id") == target_sat_id), None)
        if cache_match:
            target_section["cache_tle"] = cache_match.get("tle")
            pointing_tle = target_section.get("pointing_tle")
            cache_tle = cache_match.get("tle")
            target_section["tle_match"] = pointing_tle == cache_tle if (pointing_tle and cache_tle) else None
        else:
            target_section["cache_tle"] = None
            target_section["cache_tle_note"] = "target satellite not found in elset cache"
        debug["target_satellite"] = target_section

        # Propagate all TLEs, keep only those within the field, collect predictions
        predictions: list[dict[str, Any]] = []
        all_propagations: list[dict[str, Any]] = []

        for elset in elsets:
            tle = elset.get("tle") or []
            if len(tle) < 2:
                continue
            sat_id = elset.get("satellite_id") or "unknown"
            name = elset.get("name") or sat_id
            prop_record: dict[str, Any] = {"satellite_id": sat_id, "name": name}
            try:
                kep_tle = TLE.from_lines(tle[0], tle[1])
                satellite = Satellite.from_tle(kep_tle)
                topo = obs.get_topocentric_to_satellite(epoch, satellite, ReferenceFrame.J2000)
                ra_deg = topo.right_ascension
                dec_deg = topo.declination

                delta_ra = abs(ra_center - ra_deg)
                if delta_ra > 180.0:
                    delta_ra = 360.0 - delta_ra
                delta_dec = abs(dec_center - dec_deg)
                distance_from_center = math.sqrt(delta_ra**2 + delta_dec**2)
                in_field = delta_ra < _FIELD_RADIUS_DEG and delta_dec < _FIELD_RADIUS_DEG

                prop_record.update(
                    {
                        "predicted_ra_deg": ra_deg,
                        "predicted_dec_deg": dec_deg,
                        "distance_from_center_deg": round(distance_from_center, 4),
                        "in_field": in_field,
                    }
                )

                if not in_field:
                    all_propagations.append(prop_record)
                    continue

                # Phase angle: Sun-Satellite-Observer
                sat_state = satellite.get_state_at_epoch(epoch).to_frame(ReferenceFrame.J2000)
                sat_pos = sat_state.position
                sat_to_sun = [sun_pos_km[i] - [sat_pos.x, sat_pos.y, sat_pos.z][i] for i in range(3)]
                sat_to_obs = [
                    [obs_pos.x, obs_pos.y, obs_pos.z][i] - [sat_pos.x, sat_pos.y, sat_pos.z][i] for i in range(3)
                ]
                dot = sum(sat_to_sun[i] * sat_to_obs[i] for i in range(3))
                mag_a = math.sqrt(sum(x * x for x in sat_to_sun))
                mag_b = math.sqrt(sum(x * x for x in sat_to_obs))
                cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
                phase_angle = math.degrees(math.acos(cos_angle))

                prop_record["phase_angle"] = round(phase_angle, 2)
                all_propagations.append(prop_record)

                predictions.append(
                    {"ra": ra_deg, "dec": dec_deg, "satellite_id": sat_id, "name": name, "phase_angle": phase_angle}
                )
            except Exception as exc:
                prop_record["propagation_error"] = str(exc)
                all_propagations.append(prop_record)
                continue

        debug["predictions_all"] = all_propagations
        debug["predictions_in_field"] = [p for p in all_propagations if p.get("in_field")]
        debug["predictions_in_field_count"] = len(predictions)

        if not predictions:
            debug["early_exit"] = "no TLE predictions in field"
            return [], debug

        # KDTree spatial match
        pred_coords = [[p["ra"], p["dec"]] for p in predictions]
        tree = KDTree(pred_coords)

        # Query without upper bound so we always get the nearest distance for diagnostics
        source_coords = potential_sats[["ra", "dec"]].values
        distances, indices = tree.query(source_coords)
        valid_mask = np.asarray(distances) < _MATCH_RADIUS_DEG

        # Record match details for every satellite candidate source
        match_details: list[dict[str, Any]] = []
        for i in range(len(potential_sats)):
            row = potential_sats.iloc[i]
            dist_deg = float(np.asarray(distances)[i])
            idx = int(np.asarray(indices)[i])
            detail: dict[str, Any] = {
                "source_ra": float(row["ra"]),
                "source_dec": float(row["dec"]),
                "source_elongation": float(row["elongation"]),
                "source_mag": float(row["mag"]),
                "nearest_prediction_distance_arcmin": round(dist_deg * 60.0, 4),
                "within_match_radius": bool(valid_mask[i]),
            }
            if 0 <= idx < len(predictions):
                detail["nearest_satellite_id"] = predictions[idx]["satellite_id"]
                detail["nearest_satellite_name"] = predictions[idx]["name"]
            if valid_mask[i]:
                detail["matched"] = True
            match_details.append(detail)
        debug["match_details"] = match_details

        # Reverse match: for each in-field prediction, find the nearest detected source
        source_tree = KDTree(source_coords)
        pred_distances, pred_indices = source_tree.query(pred_coords)
        reverse_match: list[dict[str, Any]] = []
        for i, p in enumerate(predictions):
            dist_deg = float(np.asarray(pred_distances)[i])
            src_idx = int(np.asarray(pred_indices)[i])
            entry: dict[str, Any] = {
                "satellite_id": p["satellite_id"],
                "name": p["name"],
                "predicted_ra": p["ra"],
                "predicted_dec": p["dec"],
                "nearest_source_distance_arcmin": round(dist_deg * 60.0, 4),
                "within_match_radius": dist_deg < _MATCH_RADIUS_DEG,
            }
            if 0 <= src_idx < len(potential_sats):
                src_row = potential_sats.iloc[src_idx]
                entry["nearest_source_ra"] = float(src_row["ra"])
                entry["nearest_source_dec"] = float(src_row["dec"])
                entry["nearest_source_elongation"] = float(src_row["elongation"])
                entry["nearest_source_mag"] = float(src_row["mag"])
            reverse_match.append(entry)
        debug["reverse_match"] = reverse_match

        # Build observations from reverse match: one best source per prediction.
        # The forward match (source->prediction) allows many sources to match the same
        # satellite, flooding results with false positives from nearby stars.
        filter_name = (context.task.assigned_filter_name if context.task else None) or "Clear"
        observations: list[dict[str, Any]] = []
        for i, p in enumerate(predictions):
            dist_deg = float(np.asarray(pred_distances)[i])
            if dist_deg >= _MATCH_RADIUS_DEG:
                continue
            src_idx = int(np.asarray(pred_indices)[i])
            if src_idx < 0 or src_idx >= len(potential_sats):
                continue
            row = potential_sats.iloc[src_idx]
            observations.append(
                {
                    "norad_id": p["satellite_id"],
                    "name": p["name"],
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "mag": float(row["mag"]),
                    "filter": filter_name,
                    "timestamp": timestamp_str,
                    "phase_angle": round(p["phase_angle"], 1),
                    "elongation": float(row["elongation"]),
                }
            )

        debug["satellite_observations"] = observations
        return observations, debug

    def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process image with satellite matching.

        Args:
            context: Processing context with image and settings

        Returns:
            ProcessorResult with satellite matching outcome
        """
        start_time = time.time()

        try:
            # Prefer in-memory sources from plate solver; fall back to output.cat on disk
            if context.detected_sources is not None:
                sources_df = context.detected_sources
            else:
                catalog_path = context.working_dir / "output.cat"
                if not catalog_path.exists():
                    return ProcessorResult(
                        should_upload=True,
                        extracted_data={},
                        confidence=0.0,
                        reason="Source catalog not found (plate solving must succeed first)",
                        processing_time_seconds=time.time() - start_time,
                        processor_name=self.name,
                    )
                sources_df = read_source_catalog(catalog_path)

            tracking_mode = context.tracking_mode or "sidereal"
            satellite_observations, debug_info = self._match_satellites(
                sources_df, context, tracking_mode=tracking_mode
            )

            elapsed = time.time() - start_time

            result = ProcessorResult(
                should_upload=True,
                extracted_data={
                    "num_satellites_detected": len(satellite_observations),
                    "satellite_observations": satellite_observations,
                },
                confidence=1.0 if satellite_observations else 0.5,
                reason=f"Matched {len(satellite_observations)} satellite(s) in {elapsed:.1f}s",
                processing_time_seconds=elapsed,
                processor_name=self.name,
            )

            dump_processor_result(context.working_dir, "satellite_matcher_result.json", result)
            dump_json(context.working_dir, "satellite_matcher_debug.json", debug_info)

            return result

        except Exception as e:
            result = ProcessorResult(
                should_upload=True,
                extracted_data={},
                confidence=0.0,
                reason=f"Satellite matching failed: {e!s}",
                processing_time_seconds=time.time() - start_time,
                processor_name=self.name,
            )
            dump_processor_result(context.working_dir, "satellite_matcher_result.json", result)
            return result
