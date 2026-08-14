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
                    "max_total_turns": sessions.max_total_turns,
                    "turns_served": sessions.turns_served,
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
        """Accept the event and answer immediately (202).

        A backend event queue does not wait for the conversation to stop talking. Blocking here
        until the event had been applied meant an event fired mid-reply held the caller for as
        long as that reply took, which made an asynchronous queue behave synchronously from the
        outside. Pass `wait=true` to get the applied result instead — the MCP tool does, because
        an agent calling a tool wants the outcome, not a receipt.
        """
        data = await _body(request)
        new_eta = data.get("new_eta")
        eta = int(new_eta) if new_eta not in (None, "") else None
        apply_now = bool(data.get("wait")) or request.query_params.get("wait") == "true"
        try:
            if apply_now:
                result = await run_in_threadpool(
                    sessions.trigger_event,
                    request.path_params["session_id"],
                    data.get("event_type") or "",
                    new_eta=eta,
                )
                return JSONResponse(result)
            result = await run_in_threadpool(
                sessions.accept_event,
                request.path_params["session_id"],
                data.get("event_type") or "",
                new_eta=eta,
            )
        except UnknownSessionError as exc:
            return _fail(exc, 404)
        except ValueError as exc:
            return _fail(exc)
        return JSONResponse(result, status_code=202)

    async def trace(request: Request) -> JSONResponse:
        try:
            result = await run_in_threadpool(sessions.get_trace, request.path_params["session_id"])
        except UnknownSessionError as exc:
            return _fail(exc, 404)
        return JSONResponse(result)

    async def internals(_: Request) -> JSONResponse:
        """The literal prompts, templates and guardrail vocabularies.

        Exposed deliberately. The submission claims the model is never told a threshold, a rule
        id, a tier or an amount — that is only worth anything if a reviewer can read the actual
        strings and check. The same regexes the test suite asserts with are re-run here against
        the live text, so the claim is demonstrated rather than restated.
        """
        # `care_agent.guardrails` re-exports `normalize` as a function, so `import
        # ...guardrails.normalize as x` binds the function (the as-form does an attribute
        # lookup), not the module. Import the name straight out of the module instead.
        from care_agent.guardrails.normalize import _DIACRITICS
        import care_agent.guardrails.loop_guard as loop_guard
        import care_agent.guardrails.promise_guard as promise_guard
        from care_agent.llm import classifier, generator, templates

        def scan(text: str) -> list[dict[str, Any]]:
            checks = (
                ("rule ids", r"\bR[1-6]\b"),
                ("delay thresholds", r"\b(10|20|30|40|45|60)\b"),
                ("merchant tiers", r"(?i)\b(gold|silver|bronze)\b"),
                ("currency / compensation", r"(?i)\b(aed|usd|sar|refund|discount)\b|\$\d"),
            )
            import re as _re

            return [
                {"looking_for": label, "pattern": pattern,
                 "found": [m.group(0) for m in _re.finditer(pattern, text)]}
                for label, pattern in checks
            ]

        return JSONResponse(
            {
                "prompts": {
                    "classifier": {
                        "model": sessions.classifier_model,
                        "system": classifier.SYSTEM_PROMPT,
                        "output_schema": classifier.CLASSIFIER_SCHEMA,
                        "scan": scan(classifier.SYSTEM_PROMPT),
                    },
                    "generator": {
                        "model": sessions.generator_model,
                        "system": generator.SYSTEM_PROMPT,
                        "scan": scan(generator.SYSTEM_PROMPT),
                    },
                },
                "templates": [
                    {
                        "action": action.value,
                        "intent": spec.intent,
                        "slots": list(spec.slots),
                        "fallback": dict(spec.fallback),
                        "scan": scan(spec.intent),
                    }
                    for action, spec in templates.TEMPLATES.items()
                ],
                "guardrails": {
                    "promise_guard": {
                        "forbidden_substrings": list(promise_guard._FORBIDDEN_SUBSTRINGS),
                        "forbidden_patterns": [p.pattern for p in promise_guard._FORBIDDEN_PATTERNS],
                    },
                    "normalization": {
                        "digit_folding": "Arabic-Indic and Eastern-Arabic digits folded to ASCII",
                        "diacritics_pattern": _DIACRITICS.pattern,
                        "steps": ["NFKC", "casefold", "fold digits", "strip diacritics", "collapse whitespace"],
                    },
                    "loop_guard": {
                        "signature_flags": list(loop_guard.LOOP_FLAGS),
                        "threshold": sessions.policy.raw.get("loop_guard_threshold"),
                        "note": "keys on the classified booleans, not raw text, so rephrasing or "
                                "switching language does not reset the counter",
                    },
                    "r6_backstop": {
                        "request_markers": list(classifier._REQUEST_MARKERS),
                        "human_nouns": list(classifier._HUMAN_NOUNS),
                        "short_nouns_word_bounded": list(classifier._SHORT_HUMAN_NOUNS),
                        "note": "two signals required (a request marker AND a human noun). Runs "
                                "before any model call, so R6 never depends on the classifier.",
                    },
                },
            }
        )

    async def guard_check(request: Request) -> JSONResponse:
        """Run the promise guard against arbitrary text, so a reviewer can probe it directly."""
        from care_agent.guardrails.normalize import normalize as _norm
        from care_agent.guardrails.promise_guard import check_promise

        data = await _body(request)
        draft = data.get("text") or ""
        result = check_promise(draft)
        return JSONResponse(
            {
                "text": draft,
                "normalized": _norm(draft),
                "blocked": not result.ok,
                "reason": result.reason,
                "matched": list(result.matched),
            }
        )

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
        Route("/api/internals", internals),
        Route("/api/guard-check", guard_check, methods=["POST"]),
        Route("/api/report", report),
    ]
