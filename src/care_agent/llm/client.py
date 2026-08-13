"""Thin Claude client wrapper + a mock for tests.

Only two shapes are ever needed by this system, mirroring the two LLM edges:

* :meth:`LLMClient.structured` — text in, a JSON object matching a fixed schema out
  (the intent classifier; structured output means the model *cannot* emit free text).
* :meth:`LLMClient.text` — an already-decided action in, natural language out
  (the response generator; phrasing only).

Both implementations record per-call latency so the evaluation harness can report
``total_latency_ms`` without bespoke instrumentation.

The real client is constructed lazily so importing this module never requires the
``anthropic`` package or an API key — the whole test suite runs on :class:`MockLLMClient`.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

# Model IDs per PLAN.md §0. The classifier runs on the cheapest capable tier because it is
# the highest-volume call; the generator needs stronger multilingual phrasing.
CLASSIFIER_MODEL = "claude-haiku-4-5"
GENERATOR_MODEL = "claude-sonnet-5"
SIMULATOR_MODEL = "claude-sonnet-5"   # adversarial merchant personas
JUDGE_MODEL = "claude-opus-5"         # the eval judge, deliberately a different tier


class LLMError(RuntimeError):
    """Raised when an LLM call fails or is refused. Callers must fail safe (escalate)."""


class LLMClient(ABC):
    """Interface both edges are written against, so tests can swap in a mock."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def total_latency_ms(self) -> float:
        return sum(c["latency_ms"] for c in self.calls)

    def _record(self, kind: str, model: str, latency_ms: float) -> None:
        self.calls.append({"kind": kind, "model": model, "latency_ms": latency_ms})

    @abstractmethod
    def structured(self, *, model: str, system: str, user: str, schema: dict, max_tokens: int = 256) -> dict:
        """Return a JSON object conforming to ``schema``."""

    @abstractmethod
    def text(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        """Return a plain-text completion."""


class AnthropicClient(LLMClient):
    """Real Claude API client. Constructed lazily; credentials come from the environment."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key
        self._client = None

    def _ensure(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise LLMError("the 'anthropic' package is required for live calls") from exc
            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        return self._client

    @staticmethod
    def _first_text(response) -> str:
        # Always check stop_reason before reading content: a refusal yields empty/partial content.
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("request was refused by safety classifiers")
        for block in response.content:
            if block.type == "text":
                return block.text
        raise LLMError("response contained no text block")

    def _call(self, kind: str, model: str, max_tokens: int, **kwargs):
        client = self._ensure()
        start = time.perf_counter()
        try:
            response = client.messages.create(model=model, max_tokens=max_tokens, **kwargs)
        except Exception as exc:  # SDK raises typed errors; the caller only needs fail-safe
            self._record(kind, model, (time.perf_counter() - start) * 1000)
            raise LLMError(f"{kind} call failed: {exc}") from exc
        self._record(kind, model, (time.perf_counter() - start) * 1000)
        return response

    def structured(self, *, model: str, system: str, user: str, schema: dict, max_tokens: int = 256) -> dict:
        response = self._call(
            "structured", model, max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        raw = self._first_text(response)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"structured output was not valid JSON: {raw[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"structured output was not a JSON object: {type(data).__name__}")
        return data

    def text(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._call(
            "text", model, max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return self._first_text(response)


class MockLLMClient(LLMClient):
    """Deterministic stand-in. Queue the responses each edge should return, in order.

    A queued value may be an ``Exception`` instance, which is raised instead of returned —
    that is how failure paths are exercised without touching the network.
    """

    def __init__(
        self,
        structured_responses: list[dict | Exception] | None = None,
        text_responses: list[str | Exception] | None = None,
    ) -> None:
        super().__init__()
        self._structured = list(structured_responses or [])
        self._text = list(text_responses or [])
        self.prompts: list[dict[str, Any]] = []

    def _next(self, queue: list, kind: str):
        if not queue:
            raise AssertionError(f"MockLLMClient has no queued {kind} response")
        value = queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def structured(self, *, model: str, system: str, user: str, schema: dict, max_tokens: int = 256) -> dict:
        self.prompts.append({"kind": "structured", "model": model, "system": system, "user": user})
        self._record("structured", model, 0.0)
        return self._next(self._structured, "structured")

    def text(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        self.prompts.append({"kind": "text", "model": model, "system": system, "user": user})
        self._record("text", model, 0.0)
        return self._next(self._text, "text")
