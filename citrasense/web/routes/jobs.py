"""Background job polling and cancellation endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


def build_jobs_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Routes for polling and cancelling background jobs."""
    router = APIRouter(prefix="/api", tags=["jobs"])

    @router.get("/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Poll a background job's progress."""
        status = ctx.job_runner.get_status(job_id)
        if status is None:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        return status.to_dict()

    @router.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        """Request cooperative cancellation of a background job."""
        ok = ctx.job_runner.cancel(job_id)
        if not ok:
            return JSONResponse({"error": "Job not found or already finished"}, status_code=404)
        return {"status": "cancelling", "job_id": job_id}

    return router
