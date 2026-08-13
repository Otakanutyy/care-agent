"""Language-invariant text normalization.

The guardrails must work across Arabic, English, and mixed Franco-Arabic, so they run on a
normalized form rather than raw surface text: Unicode-folded (NFKC), Arabic-Indic and
Eastern-Arabic digits mapped to ASCII, diacritics and tatweel stripped, casefolded, and
whitespace collapsed. This defeats the obvious evasions (e.g. offering "٥٠ درهم" instead of
"50 AED", or fullwidth "＄５０").
"""

from __future__ import annotations

import re
import unicodedata

# Arabic-Indic (U+0660-0669) and Eastern-Arabic/Persian (U+06F0-06F9) digits -> ASCII.
_DIGIT_TABLE = str.maketrans(
    {
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
    }
)

# Arabic harakat (U+064B-0652) + tatweel/kashida (U+0640).
_DIACRITICS = re.compile(r"[ً-ْـ]")
_WHITESPACE = re.compile(r"\s+")


def fold_digits(text: str) -> str:
    """Map Arabic-Indic / Eastern-Arabic digits to ASCII 0-9."""
    return (text or "").translate(_DIGIT_TABLE)


def normalize(text: str) -> str:
    """Normalize to a language-invariant form for guardrail matching."""
    t = unicodedata.normalize("NFKC", text or "")
    t = fold_digits(t)
    t = t.casefold()
    t = _DIACRITICS.sub("", t)
    t = _WHITESPACE.sub(" ", t).strip()
    return t
