"""HTTP API and single-page UI for the reviewer-facing testing surface.

The MCP server assumes the reviewer wires up their own agent. This is the zero-setup path:
open a link, click, watch. Same :class:`SessionManager` underneath, so the two surfaces cannot
drift apart.

The UI exists to make one claim *visible*: the policy engine decided, not the model. Every
reply lands next to the rule id that produced it, the guardrail verdict, and a financial-tool
counter that stays at zero no matter how hard the merchant pushes.

Session calls run in a threadpool because in live mode they block on the Anthropic API, and
blocking the event loop would stall every other viewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from care_agent.server.sessions import (
    POLICY_PATH,
    REPO_ROOT,
    TRIGGERABLE_EVENTS,
    SessionLimitError,
    SessionManager,
    UnknownSessionError,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"


def _fail(exc: Exception, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": type(exc).__name__, "detail": str(exc)}, status_code=status)


async def _body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_routes(sessions: SessionManager) -> list[Route]:
    """HTTP routes backing the UI. Everything under /api requires the bearer token."""

    async def index(_: Request) -> Response:
        if not INDEX_PATH.exists():  # pragma: no cover - packaging guard
            return HTMLResponse("<h1>UI asset missing</h1>", status_code=500)
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))

    async def config(_: Request) -> JSONResponse:
        """What the UI needs to render itself — and to prove the token works."""
        return JSONResponse(
            {
                "mode": sessions.mode,
                "policy_version": sessions.policy_version,
                "models": {
                    "classifier": sessions.classifier_model,
                    "generator": sessions.generator_model,
                },
                "limits": {
                    "max_sessions": sessions.max_sessions,
                    "max_turns": sessions.max_turns,
                },
                "triggerable_events": [e.value for e in TRIGGERABLE_EVENTS],
                # Read from disk rather than handing over `policy.raw`: the loaded snapshot is
                # deep-frozen into mappingproxy objects (that immutability is the point), which
                # json cannot serialise. This also guarantees the UI shows the real file.
                "policy": json.loads(POLICY_PATH.read_text(encoding="utf-8")),
            }
        )

    async def start(request: Request) -> JSONResponse:
        data = await _body(request)
        try:
            result = await run_in_threadpool(
                sessions.start,
                merchant_name=data.get("merchant_name") or "Test Merchant",
                merchant_tier=data.get("merchant_tier") or "Silver",
                delay_minutes=int(data.get("delay_minutes", 30)),
                current_captain_id=data.get("current_captain_id", "captain-100"),
                active_system_overrides=data.get("active_system_overrides") or [],
            )
        except (SessionLimitError, ValueError) as exc:
            return _fail(exc)
        return JSONResponse(result)

    async def message(request: Request) -> JSONResponse:
        data = await _body(request)
        text = (data.get("message") or "").strip()
        if not text:
            return JSONResponse({"error": "ValueError", "detail": "message is empty"}, status_code=400)
        try:
            result = await run_in_threadpool(
                sessions.send_message, request.path_params["session_id"], text
            )
        except UnknownSessionError as exc:
            return _fail(exc, 404)
        except (SessionLimitError, ValueError) as exc:
            return _fail(exc)
        return JSONResponse(result)

    async def event(request: Request) -> JSONResponse:
        data = await _body(request)
        new_eta = data.get("new_eta")
        try:
            result = await run_in_threadpool(
                sessions.trigger_event,
                request.path_params["session_id"],
                data.get("event_type") or "",
                new_eta=int(new_eta) if new_eta not in (None, "") else None,
            )
        except UnknownSessionError as exc:
            return _fail(exc, 404)
        except ValueError as exc:
            return _fail(exc)
        return JSONResponse(result)

    async def trace(request: Request) -> JSONResponse:
        try:
            result = await run_in_threadpool(sessions.get_trace, request.path_params["session_id"])
        except UnknownSessionError as exc:
            return _fail(exc, 404)
        return JSONResponse(result)

    async def report(_: Request) -> JSONResponse:
        """The offline evaluation results, so the UI can show them without a rerun."""
        path = REPO_ROOT / "report.json"
        if not path.exists():
            return JSONResponse({"error": "missing", "detail": "run `python run_all.py`"}, status_code=404)
        return Response(path.read_text(encoding="utf-8"), media_type="application/json")

    return [
        Route("/", index),
        Route("/api/config", config),
        Route("/api/sessions", start, methods=["POST"]),
        Route("/api/sessions/{session_id}/messages", message, methods=["POST"]),
        Route("/api/sessions/{session_id}/events", event, methods=["POST"]),
        Route("/api/sessions/{session_id}/trace", trace),
        Route("/api/report", report),
    ]
