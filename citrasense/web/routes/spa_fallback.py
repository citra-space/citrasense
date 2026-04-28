"""SPA fallback catchall.

Must be included LAST in :func:`citrasense.web.routes.build_all_routers`.
FastAPI route resolution is first-match-wins, so everything that's a real
backend route (``/api/*``, ``/ws``, the ``/`` landing) must already be
registered by the time this module's router is attached.  Anything that
slips through — ``/monitoring``, ``/analysis``, ``/config``,
``/sensors/<id>``, or a typo — returns the rendered ``dashboard.html``
shell.  The client-side path router in ``app.js`` then reads
``location.pathname`` and paints the matching section.

Keeping this in its own router makes the "MUST be last" invariant
obvious at the call site instead of buried in the order of handlers
inside ``core.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from citrasense.web.app import CitraSenseWebApp


# Prefixes that are backend-owned.  A GET here that didn't match a
# specific route is a real 404 (a stale client hitting a removed
# endpoint, a typo, a misconfigured proxy) — not a client-side path we
# should silently paint the dashboard shell for.
#
# Every entry ends in ``/`` so ``startswith`` only fires on a true path
# segment boundary — ``/wstools`` and ``/apispecific`` don't get yanked
# into a 404 just because their names happen to start with one of our
# reserved segments.  Exact single-segment paths we also own (``/ws``
# for the websocket upgrade) live in ``_BACKEND_EXACT_PATHS`` so the
# boundary check and the bare-path check stay independently tunable.
_BACKEND_PATH_PREFIXES: tuple[str, ...] = ("api/", "static/", "ws/", "images/")
_BACKEND_EXACT_PATHS: frozenset[str] = frozenset({"ws"})


def build_spa_fallback_router(ctx: CitraSenseWebApp) -> APIRouter:
    """Catchall that returns the SPA shell for unrecognized GETs."""
    router = APIRouter(tags=["spa"], include_in_schema=False)

    @router.get("/{full_path:path}", response_class=HTMLResponse)
    async def spa_shell(request: Request, full_path: str):
        # ``full_path`` is captured *without* the leading slash, e.g.
        # "api/nonexistent" for GET /api/nonexistent.
        if full_path in _BACKEND_EXACT_PATHS or any(full_path.startswith(prefix) for prefix in _BACKEND_PATH_PREFIXES):
            raise HTTPException(status_code=404, detail="Not Found")
        return ctx.templates.TemplateResponse(request, "dashboard.html")

    return router
