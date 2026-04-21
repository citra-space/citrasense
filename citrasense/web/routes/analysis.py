"""Analysis endpoints: task queries, artifacts, reprocessing, and autotune."""

from __future__ import annotations

import asyncio
import gzip
import json as _json
import shutil
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.connection_manager import _TarStreamBuffer

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_analysis_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Analysis, artifact download, reprocess, and autotune endpoints."""
    router = APIRouter(prefix="/api", tags=["analysis"])

    @router.get("/analysis/tasks")
    async def analysis_tasks(
        limit: int = 50,
        offset: int = 0,
        sort: str = "completed_at",
        order: str = "desc",
        target_name: str | None = None,
        plate_solved: bool | None = None,
        target_matched: bool | None = None,
        missed_window: bool | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        filter_name: str | None = None,
        match_detail: str | None = None,
        upload_status: str | None = None,
    ):
        """Paginated, filterable list of completed tasks."""
        if not ctx.daemon or not ctx.daemon.task_index:
            return JSONResponse({"tasks": [], "total": 0})
        return ctx.daemon.task_index.query_tasks(
            limit=max(1, min(limit, 200)),
            offset=max(0, offset),
            sort=sort,
            order=order,
            target_name=target_name,
            plate_solved=plate_solved,
            target_matched=target_matched,
            missed_window=missed_window,
            date_from=date_from,
            date_to=date_to,
            filter_name=filter_name,
            match_detail=match_detail,
            upload_status=upload_status,
        )

    @router.get("/analysis/tasks/{task_id}")
    async def analysis_task_detail(task_id: str):
        """Single task detail with all fields."""
        if not ctx.daemon or not ctx.daemon.task_index:
            return JSONResponse({"error": "Analysis not available"}, status_code=503)
        safe_id = Path(task_id).name
        task = ctx.daemon.task_index.get_task(safe_id)
        if task is None:
            return JSONResponse({"error": "Task not found"}, status_code=404)
        bundle_dir = ctx.daemon.settings.directories.processing_dir / safe_id
        task["artifacts_available"] = bundle_dir.is_dir()

        reprocessed_summary = bundle_dir / "reprocessed" / "processing_summary.json"
        if reprocessed_summary.is_file():
            try:
                summary = _json.loads(reprocessed_summary.read_text())
                task["reprocessed_result"] = {
                    "should_upload": summary.get("should_upload"),
                    "skip_reason": summary.get("skip_reason"),
                    "total_time": summary.get("total_time"),
                    "processors": [
                        {
                            "name": p.get("processor_name", p.get("name", "")),
                            "confidence": p.get("confidence"),
                            "reason": p.get("reason"),
                            "time_s": round(p.get("processing_time_seconds", p.get("time_s", 0)), 3),
                            "ok": p.get("should_upload", False) and (p.get("confidence", 0) or 0) > 0,
                        }
                        for p in summary.get("processors", [])
                    ],
                    "extracted_data": summary.get("extracted_data", {}),
                }
            except Exception:
                pass

        return task

    @router.get("/analysis/tasks/{task_id}/image")
    async def analysis_task_image(task_id: str, thumb: int = 0):
        """Serve annotated preview image for a task."""
        if not ctx.daemon:
            return JSONResponse({"error": "Not available"}, status_code=503)
        safe_id = Path(task_id).name
        previews_dir = ctx.daemon.settings.directories.analysis_previews_dir

        if thumb:
            thumb_path = previews_dir / f"{safe_id}.thumb.jpg"
            if thumb_path.is_file():
                return FileResponse(
                    str(thumb_path),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=604800, immutable"},
                )

        preview = previews_dir / f"{safe_id}.jpg"
        if not preview.is_file():
            return JSONResponse({"error": "Image not available"}, status_code=404)
        return FileResponse(
            str(preview),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )

    @router.get("/analysis/tasks/{task_id}/artifacts/{filepath:path}")
    async def analysis_task_artifact(task_id: str, filepath: str):
        """Serve an artifact file from the processing directory.

        Supports nested paths (e.g. ``reprocessed/report.html``) so
        reprocessed output is accessible without a separate route.
        """
        if not ctx.daemon:
            return JSONResponse({"error": "Not available"}, status_code=503)
        safe_id = Path(task_id).name
        task_dir = ctx.daemon.settings.directories.processing_dir / safe_id
        resolved_task_dir = task_dir.resolve()
        artifact = (task_dir / filepath).resolve()
        if not artifact.is_relative_to(resolved_task_dir):
            return JSONResponse({"error": "Invalid path"}, status_code=400)
        if not artifact.is_file():
            return JSONResponse({"error": "Artifact not found or expired"}, status_code=404)
        return FileResponse(str(artifact))

    @router.get("/analysis/tasks/{task_id}/bundle")
    async def analysis_task_bundle(task_id: str):
        """Stream a tar.gz bundle of a task's processing directory."""
        if not ctx.daemon:
            return JSONResponse({"error": "Not available"}, status_code=503)
        safe_id = Path(task_id).name
        task_dir = ctx.daemon.settings.directories.processing_dir / safe_id
        if not task_dir.is_dir():
            return JSONResponse({"error": "Task artifacts not found or expired"}, status_code=404)

        def _generate():
            """Yield gzipped tar bytes incrementally without buffering the whole archive."""
            buf = _TarStreamBuffer()
            gz = gzip.GzipFile(fileobj=buf, mode="wb")
            tar = tarfile.open(fileobj=gz, mode="w|")
            try:
                for file_path in task_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                    arcname = f"{safe_id}/{file_path.relative_to(task_dir)}"
                    tar.add(str(file_path), arcname=arcname)
                    chunk = buf.drain()
                    if chunk:
                        yield chunk
            finally:
                tar.close()
                gz.close()
                final = buf.drain()
                if final:
                    yield final

        filename = f"{safe_id}.tar.gz"
        return StreamingResponse(
            _generate(),
            media_type="application/gzip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/analysis/stats")
    async def analysis_stats(hours: int = 24):
        """Aggregate statistics over the given time window."""
        if not ctx.daemon or not ctx.daemon.task_index:
            from citrasense.analysis.task_index import empty_stats

            return {**empty_stats(), "filter_names": []}
        stats = ctx.daemon.task_index.get_stats(hours=max(1, min(hours, 8760)))
        stats["filter_names"] = ctx.daemon.task_index.get_distinct_filter_names()
        return stats

    @router.post("/analysis/tasks/{task_id}/reprocess")
    async def reprocess_task(task_id: str, request: Request):
        """Reprocess a single task's debug bundle with optional settings overrides."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        safe_id = Path(task_id).name
        debug_dir = ctx.daemon.settings.directories.processing_dir / safe_id
        if not debug_dir.is_dir():
            return JSONResponse({"error": "Debug bundle not found (artifacts expired?)"}, status_code=404)

        body = await request.json() if await request.body() else {}
        settings_overrides = body.get("settings_overrides")

        from citrasense.reprocess import reprocess_bundle

        try:
            output_dir = debug_dir / "reprocessed"
            if output_dir.exists():
                shutil.rmtree(output_dir)

            result, _out_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: reprocess_bundle(
                    debug_dir=debug_dir,
                    output_dir=output_dir,
                    settings_overrides=settings_overrides,
                ),
            )
            summary = {
                "should_upload": result.should_upload,
                "skip_reason": result.skip_reason,
                "total_time": round(result.total_time, 3),
                "processors": [
                    {
                        "name": r.processor_name,
                        "confidence": r.confidence,
                        "reason": r.reason,
                        "time_s": round(r.processing_time_seconds, 3),
                        "ok": r.should_upload and r.confidence > 0,
                    }
                    for r in result.all_results
                ],
                "extracted_data": result.extracted_data,
            }
            return summary
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            CITRASENSE_LOGGER.exception("Reprocessing failed for task %s", safe_id)
            return JSONResponse({"error": f"Reprocessing failed: {exc}"}, status_code=500)

    @router.post("/analysis/reprocess-batch")
    async def reprocess_batch(request: Request):
        """Submit a batch reprocess job for multiple tasks."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        body = await request.json()
        task_ids: list[str] = body.get("task_ids", [])
        settings_overrides = body.get("settings_overrides")

        if not task_ids:
            return JSONResponse({"error": "task_ids required"}, status_code=400)

        processing_dir = ctx.daemon.settings.directories.processing_dir

        def _batch_worker(status):
            from citrasense.reprocess import reprocess_bundle

            for i, tid in enumerate(task_ids):
                safe_id = Path(tid).name
                debug_dir = processing_dir / safe_id
                item_result = {"task_id": safe_id, "ok": False, "error": None}
                try:
                    if not debug_dir.is_dir():
                        item_result["error"] = "Bundle not found"
                    else:
                        output_dir = debug_dir / "reprocessed"
                        if output_dir.exists():
                            shutil.rmtree(output_dir)
                        result, _ = reprocess_bundle(
                            debug_dir=debug_dir,
                            output_dir=output_dir,
                            settings_overrides=settings_overrides,
                        )
                        item_result["ok"] = True
                        item_result["should_upload"] = result.should_upload
                        item_result["total_time"] = round(result.total_time, 3)
                except Exception as exc:
                    item_result["error"] = str(exc)
                status.append_item_result(item_result)
                status.progress = i + 1

            ok_count = sum(1 for r in status.per_item_results if r["ok"])
            status.result = {"succeeded": ok_count, "failed": len(task_ids) - ok_count}

        job = ctx.job_runner.submit(_batch_worker, total=len(task_ids))
        return {"job_id": job.job_id}

    @router.post("/analysis/tasks/{task_id}/reprocess/upload")
    async def upload_reprocessed(task_id: str):
        """Upload reprocessed satellite observations via the Citra API."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)
        if not ctx.daemon.api_client:
            return JSONResponse({"error": "API client not available"}, status_code=503)

        safe_id = Path(task_id).name
        debug_dir = ctx.daemon.settings.directories.processing_dir / safe_id
        reprocessed_dir = debug_dir / "reprocessed"
        if not reprocessed_dir.is_dir():
            return JSONResponse({"error": "No reprocessed output found"}, status_code=404)

        summary_path = reprocessed_dir / "processing_summary.json"
        if not summary_path.exists():
            return JSONResponse({"error": "Reprocessed summary not found"}, status_code=404)

        summary = _json.loads(summary_path.read_text())
        if not summary.get("should_upload"):
            return JSONResponse(
                {"error": "Reprocessed result not eligible for upload", "skip_reason": summary.get("skip_reason")},
                status_code=400,
            )

        extracted = summary.get("extracted_data") or {}
        sat_obs = extracted.get("satellite_matcher.satellite_observations") or []
        if not sat_obs:
            return JSONResponse(
                {"error": "No satellite observations in reprocessed result"},
                status_code=400,
            )

        has_calibrated_mag = any(obs.get("mag") is not None for obs in sat_obs)
        if not has_calibrated_mag:
            return JSONResponse(
                {"error": "Photometry failed — no calibrated magnitudes available"},
                status_code=400,
            )

        telescope_path = debug_dir / "telescope_record.json"
        location_path = debug_dir / "observer_location.json"
        if not telescope_path.exists() or not location_path.exists():
            return JSONResponse(
                {"error": "Original debug bundle missing telescope_record or observer_location"},
                status_code=400,
            )

        telescope_record = _json.loads(telescope_path.read_text())
        observer_location = _json.loads(location_path.read_text())

        try:
            success = await asyncio.to_thread(
                ctx.daemon.api_client.upload_optical_observations,
                sat_obs,
                telescope_record,
                observer_location,
                task_id=safe_id,
            )
        except Exception as exc:
            CITRASENSE_LOGGER.exception("Reprocessed upload failed for task %s", safe_id)
            return JSONResponse({"error": f"Upload failed: {exc}"}, status_code=500)

        if not success:
            return JSONResponse({"error": "Upload rejected by API"}, status_code=502)

        return {
            "status": "uploaded",
            "task_id": safe_id,
            "observations_count": len(sat_obs),
        }

    @router.post("/analysis/autotune")
    async def autotune(request: Request):
        """Run SExtractor auto-tune as a background job."""
        if not ctx.daemon:
            return JSONResponse({"error": "Daemon not available"}, status_code=503)

        body = await request.json() if await request.body() else {}
        task_ids: list[str] | None = body.get("task_ids")
        num_bundles = body.get("num_bundles", 10)

        processing_dir = ctx.daemon.settings.directories.processing_dir

        if task_ids:
            debug_dirs = [
                processing_dir / Path(tid).name for tid in task_ids if (processing_dir / Path(tid).name).is_dir()
            ]
        else:
            from citrasense.autotune import _discover_bundles

            debug_dirs = _discover_bundles(processing_dir, max_bundles=num_bundles)

        if not debug_dirs:
            return JSONResponse({"error": "No debug bundles found"}, status_code=404)

        from citrasense.autotune import PARAM_GRID, autotune_extraction

        n_thresh = len(PARAM_GRID["detect_thresh"])
        n_area = len(PARAM_GRID["detect_minarea"])
        n_filt = len(PARAM_GRID["filter_name"])
        combos = n_thresh * n_area * n_filt
        total_evals = combos * len(debug_dirs)

        requested_count = len(debug_dirs)

        def _autotune_worker(status):
            def _progress(done: int, total: int) -> None:
                status.progress = done
                status.total = total

            results = autotune_extraction(
                debug_dirs,
                on_progress=_progress,
                is_cancelled=lambda: status.cancelled,
            )
            if status.cancelled:
                status.state = "cancelled"
            actual_used = results[0]["bundles_evaluated"] if results else 0
            status.result = {
                "configs": results[:20],
                "total_evaluated": total_evals,
                "bundles_used": actual_used,
                "bundles_requested": requested_count,
            }

        job = ctx.job_runner.submit(_autotune_worker, total=total_evals)
        return {"job_id": job.job_id, "total": total_evals, "bundles": len(debug_dirs)}

    return router
