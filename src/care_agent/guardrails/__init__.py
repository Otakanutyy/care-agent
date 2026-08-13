"""Guardrail layer: unauthorized-promise block + repetition/loop guard, both language-invariant."""

from care_agent.guardrails.loop_guard import (
    intent_signature,
    is_tripped,
    next_counter,
)
from care_agent.guardrails.normalize import fold_digits, normalize
from care_agent.guardrails.promise_guard import GuardResult, check_promise

__all__ = [
    "GuardResult",
    "check_promise",
    "fold_digits",
    "intent_signature",
    "is_tripped",
    "next_counter",
    "normalize",
]
