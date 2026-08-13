"""A deterministic, offline stand-in for the Claude API.

Its only purpose is to make the agent runnable end-to-end **without an API key** — so a
reviewer can clone the repo and watch a session play out, and so demos are reproducible.

It is not a substitute for the real classifier:

* ``structured`` does crude keyword matching for the five flags. It will misread anything
  phrased indirectly — which is exactly why the real system uses a model here.
* ``text`` returns an empty string, which makes the generator fall back to its pre-approved
  per-language template. That is honest: offline, there is no phrasing model, so the agent
  sends the safe canned wording rather than pretending to have composed something.

Use :class:`~care_agent.llm.client.AnthropicClient` for real runs.
"""

from __future__ import annotations

from care_agent.guardrails.normalize import normalize
from care_agent.llm.classifier import detect_language
from care_agent.llm.client import LLMClient

# Ordered: the first matching rule wins, so a cancellation is never masked by a stray "ok".
_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("requests_cancellation", ("cancel", "الغاء", "إلغاء", "لغي", "cancel it", "elgha")),
    (
        "accepts_reassignment",
        ("another driver", "different driver", "new driver", "reassign", "send someone else",
         "another captain", "new captain", "yes please", "go ahead", "كابتن اخر", "كابتن آخر", "captain tani"),
    ),
    ("prefers_to_wait", ("wait", "keep the current", "keep him", "stay", "انتظر", "بستنى", "antazer")),
    (
        "confirms_new_eta",
        ("that works", "sounds good", "confirm", "fine", "no problem", "ok", "okay", "تمام", "ماشي", "tamam"),
    ),
)


class OfflineLLMClient(LLMClient):
    """Deterministic stand-in — keyword classification, canned-fallback generation."""

    def structured(self, *, model: str, system: str, user: str, schema: dict, max_tokens: int = 256) -> dict:
        self._record("structured", model, 0.0)
        norm = normalize(user)
        flags = {
            "requests_human": False,  # handled by the classifier's deterministic backstop
            "confirms_new_eta": False,
            "requests_cancellation": False,
            "accepts_reassignment": False,
            "prefers_to_wait": False,
        }
        for field, keywords in _KEYWORD_RULES:
            if any(normalize(k) in norm for k in keywords):
                flags[field] = True
                break
        return {**flags, "language": detect_language(user)}

    def text(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        self._record("text", model, 0.0)
        return ""  # -> generator uses its pre-approved template fallback
