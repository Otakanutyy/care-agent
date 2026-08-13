"""Policy loading and verification.

The policy pack (``policy/policy.json``) is the single source of truth for every
threshold, tier, cap, and rule ordering the agent uses. Nothing here calls an LLM;
this module only loads, validates, and pins an immutable snapshot for a session.
"""

from care_agent.policy.engine import OverrideEffect, decide, escalation_mode_for
from care_agent.policy.loader import (
    CheckResult,
    PolicyError,
    PolicySnapshot,
    check_invariants,
    load_policy,
    load_policy_dict,
    validate_schema,
    verify,
)

__all__ = [
    "CheckResult",
    "OverrideEffect",
    "PolicyError",
    "PolicySnapshot",
    "check_invariants",
    "decide",
    "escalation_mode_for",
    "load_policy",
    "load_policy_dict",
    "validate_schema",
    "verify",
]
