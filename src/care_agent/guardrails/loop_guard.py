"""Repetition / loop guard.

Tracks how many times the merchant has pushed the *same* off-policy intent in a row, keyed
on the classifier booleans (language-invariant) rather than the raw text — so switching
language or rephrasing the same demand cannot reset the counter. When the count reaches the
policy's ``loop_guard_threshold`` the policy engine escalates (a SAFE_HALT handover) instead
of the agent rephrasing yet again.

``requests_human`` is excluded from the signature: it preempts to escalation immediately and
never enters this loop.
"""

from __future__ import annotations

from care_agent.domain.models import ClassifierFlags

LOOP_FLAGS = ("requests_cancellation", "accepts_reassignment", "prefers_to_wait", "confirms_new_eta")

Signature = tuple[bool, ...]


def intent_signature(flags: ClassifierFlags) -> Signature:
    """A language-independent fingerprint of the merchant's intent (excludes requests_human)."""
    return tuple(bool(getattr(flags, name)) for name in LOOP_FLAGS)


def next_counter(
    prev_signature: Signature | None,
    prev_count: int,
    flags: ClassifierFlags,
    is_off_policy: bool,
) -> tuple[Signature | None, int]:
    """Advance the loop counter.

    * On an on-policy turn, reset (the loop is broken).
    * On an off-policy push with the same signature as last time, increment.
    * On an off-policy push with a new signature, restart the count at 1.
    """
    if not is_off_policy:
        return None, 0
    sig = intent_signature(flags)
    if sig == prev_signature:
        return sig, prev_count + 1
    return sig, 1


def is_tripped(count: int, threshold: int) -> bool:
    return count >= threshold
