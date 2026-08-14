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
    # Found by an independent black-box tester probing the guard directly. Compensation is
    # mostly offered in idiom, not in the word "refund" — these all read as an offer of money
    # without naming any of the terms above.
    "on us", "on the house", "cover the cost", "cover the delivery", "cover the fee",
    "complimentary", "at no cost", "at no charge", "free this time", "this one's on me",
    "not be charged", "won't be charged", "wont be charged", "no extra charge",
    "off your next", "off the next", "off your bill", "knock off",
    # currency words
    "dirham", "riyal", "dollar", "rial",
    # Arabic
    "استرداد", "استرجاع", "رصيد", "كريديت", "خصم", "قسيمة", "كوبون",
    "تعويض", "نعوض", "مجان", "هدية", "بلاش", "تنازل", "درهم", "ريال", "ريال", "جنيه",
    "معفى", "معفي", "اعفاء", "إعفاء", "بدون رسوم", "على حسابنا", "منحة",
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
    # Boundary-matched or they fire inside ordinary words: "bucks" inside Starbucks (a plausible
    # merchant name), "comp" inside company/complete/compare.
    re.compile(r"\bbucks\b"), re.compile(r"\bquid\b"),
    re.compile(r"\bcomp\b"), re.compile(r"\bcomped\b"), re.compile(r"\bcomping\b"),
)


class GuardResult(BaseModel):
    model_config = {"frozen": True}

    ok: bool
    reason: str | None = None
    matched: list[str] = Field(default_factory=list)


def _authorized(envelope: ActionEnvelope | None) -> set[str]:
    """Money the envelope explicitly authorizes. Empty today (no rule authorizes money)."""
    return set()


# "r e f u n d" spells the forbidden word while matching none of the terms above. Collapse runs
# of single characters separated by spaces — and only those, so ordinary prose is untouched and
# unrelated words are never fused into a false positive.
_LETTER_SPACED = re.compile(r"(?:(?<!\S)\w[ \t]){2,}\w(?!\S)")


def _despace(norm: str) -> str:
    return _LETTER_SPACED.sub(lambda m: m.group(0).replace(" ", "").replace("\t", ""), norm)


def _find_forbidden(norm: str) -> list[str]:
    # Check the text as written and with letter-spacing collapsed, so an evasion has to defeat
    # both forms rather than either one.
    variants = {norm, _despace(norm)}
    hits: list[str] = []
    for term in _FORBIDDEN_SUBSTRINGS:
        if any(term in v for v in variants):
            hits.append(term)
    for pat in _FORBIDDEN_PATTERNS:
        if any(pat.search(v) for v in variants):
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
