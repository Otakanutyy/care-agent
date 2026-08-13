"""Unauthorized-promise guard.

Sits between the response generator and the merchant. Because no policy rule authorizes the
agent to move money (there is no cancel/credit/refund action in the engine), *any* mention of
compensation, a fee waiver, or a currency amount in a draft reply is unauthorized and is
blocked. Detection runs on the normalized (language-invariant) form, so an offer phrased in
Arabic, Franco-Arabic, or with Arabic-Indic numerals is caught the same as English.

Scope: this guard targets money/compensation — the concrete "hallucinated refund" risk the
spec calls out. It deliberately does not try to detect every conceivable off-policy *action*
claim in free text; the architecture already prevents those by never handing the generator an
action envelope that authorizes one.

The authorized set is computed from the envelope for forward-compatibility: if a future policy
action ever set an engine-approved amount in ``variables``, that specific value would be
allowed. Today it is always empty.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from care_agent.domain.models import ActionEnvelope
from care_agent.guardrails.normalize import normalize

# Unambiguous money/compensation markers, matched as substrings on normalized text.
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    # English
    "refund", "voucher", "coupon", "cashback", "reimburse", "rebate",
    "compensation", "compensate", "goodwill", "store credit", "account credit",
    "for free", "free of charge", "on the house", "no charge", "no cost",
    "fee waiver", "waive the fee", "waive your fee", "waive the delivery",
    "discount", "money back", "gift card",
    # currency words
    "dirham", "riyal", "dollar", "rial",
    # Arabic
    "استرداد", "استرجاع", "رصيد", "كريديت", "خصم", "قسيمة", "كوبون",
    "تعويض", "نعوض", "مجان", "هدية", "بلاش", "تنازل", "درهم", "ريال", "ريال", "جنيه",
    # Franco-Arabic
    "majani", "magani", "bala flous", "balash", "khasm", "ta3wid", "tacwid",
    "hadiya", "3ala 7sabna", "3al 7esab",
    # rial/currency sign glyph (kept in case NFKC leaves a raw form)
    "﷼",
)

# Riskier markers needing word boundaries / negative lookaheads to avoid false positives.
_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\baed\b"), re.compile(r"\bsar\b"), re.compile(r"\begp\b"),
    re.compile(r"\busd\b"), re.compile(r"\bqar\b"), re.compile(r"\bkwd\b"),
    re.compile(r"\bsr\b"), re.compile(r"\bdhs\b"),
    re.compile(r"\bcredit\b(?!\s*card)"),   # "credit" but not "credit card"
    re.compile(r"\bwaiv(e|er|ed|ing)\b"),
    re.compile(r"\bfree\s+(order|meal|delivery|item|drink|dessert|ride)\b"),  # not bare "free"
    re.compile(r"\$\s*\d"), re.compile(r"\d+\s*\$"),  # $50 / 50$
)


class GuardResult(BaseModel):
    model_config = {"frozen": True}

    ok: bool
    reason: str | None = None
    matched: list[str] = Field(default_factory=list)


def _authorized(envelope: ActionEnvelope | None) -> set[str]:
    """Money the envelope explicitly authorizes. Empty today (no rule authorizes money)."""
    return set()


def _find_forbidden(norm: str) -> list[str]:
    hits: list[str] = []
    for term in _FORBIDDEN_SUBSTRINGS:
        if term in norm:
            hits.append(term)
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(norm):
            hits.append(pat.pattern)
    return hits


def check_promise(draft: str, envelope: ActionEnvelope | None = None) -> GuardResult:
    """Return ok=False (blocked) if the draft promises money/compensation the envelope did
    not authorize. Language-invariant."""
    norm = normalize(draft)
    hits = _find_forbidden(norm)
    authorized = _authorized(envelope)
    violations = sorted({h for h in hits if h not in authorized})
    if violations:
        return GuardResult(ok=False, reason="unauthorized_promise", matched=violations)
    return GuardResult(ok=True)
