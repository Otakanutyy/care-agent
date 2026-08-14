"""The policy engine — a pure decision pipeline.

``decide(state, flags, policy) -> ActionEnvelope`` is a pure function: same inputs
always yield the same envelope, no I/O, no LLM, no mutation. It is the FSM's transition
function and the *only* originator of actions.

Evaluation order (this order is load-bearing):

    Phase 0  R6 preemption      — a merchant asking for a human short-circuits everything.
    Phase 1  override resolver  — active_system_overrides (e.g. active_outage) may
                                  suppress rules and change how/where escalation happens.
    Phase 2  first-hit R1..R5   — match on delay + tier, then branch on the classifier
                                  flags within the matched rule.

``flags`` is ``None`` when the turn is triggered by an event/tick/init (proactive) rather
than a merchant message (reactive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from care_agent.domain.models import (
    ActionEnvelope,
    ActionType,
    ClassifierFlags,
    EscalationMode,
    MerchantTier,
    SessionState,
)
from care_agent.policy.loader import PolicySnapshot

# Machine-readable reason codes (kept as constants so tests and downstream code agree).
REASON_HUMAN_REQUESTED = "human_requested"
REASON_NO_MATCHING_RULE = "no_matching_rule"
REASON_WITHIN_GRACE = "within_grace"
REASON_NOTIFY_ASK_CONFIRM = "notify_ask_confirm"
REASON_ETA_CONFIRMED = "eta_confirmed"
REASON_MERCHANT_WAITS = "merchant_waits"
REASON_GOLD_AUTO_REASSIGN = "gold_auto_reassign"
REASON_MERCHANT_ACCEPTS_REASSIGNMENT = "merchant_accepts_reassignment"
REASON_ASK_REASSIGN_OR_WAIT = "ask_reassign_or_wait"
REASON_CANCELLATION_REQUESTED = "cancellation_requested"
REASON_DELAY_OVER_THRESHOLD = "delay_over_threshold"
REASON_AMBIGUOUS_INTENT = "ambiguous_intent"
REASON_LOOP_GUARD_TRIPPED = "loop_guard_tripped"
REASON_OVERRIDE_SUPPRESSED = "override_suppressed"
REASON_UNKNOWN_ACTION = "unknown_action"

REASSIGN_TOOL_CHAIN = ["check_captain_availability", "reassign_captain"]


@dataclass(frozen=True)
class OverrideEffect:
    """The combined effect of whichever active_system_overrides are in force."""

    names: tuple[str, ...]
    suppressed_rules: frozenset[str]
    escalation_mode: EscalationMode
    notification: str | None

    @property
    def active(self) -> bool:
        return bool(self.names)


_NO_OVERRIDE = OverrideEffect((), frozenset(), EscalationMode.PER_ORDER, None)


def escalation_mode_for(state: SessionState, policy: PolicySnapshot) -> EscalationMode:
    """The escalation mode that applies given the session's active overrides
    (``attach_to_incident`` during an outage, ``per_order`` otherwise). Used by the FSM
    for escalations that don't originate from ``decide`` (e.g. reply-timeout, tool failure)."""
    return _resolve_overrides(state, policy).escalation_mode


def _resolve_overrides(state: SessionState, policy: PolicySnapshot) -> OverrideEffect:
    """Phase 1: fold the active overrides into a single effect (first-in-state wins for
    mode/notification; suppressed rules are unioned)."""
    override_map = policy.override_map
    names = tuple(o for o in state.active_system_overrides if o in override_map)
    if not names:
        return _NO_OVERRIDE

    suppressed: set[str] = set()
    mode: EscalationMode | None = None
    notification: str | None = None
    for name in names:
        spec = override_map[name]
        suppressed |= set(spec["suppressed_rules"])
        if mode is None:
            mode = EscalationMode(spec["escalation_mode"])
        if notification is None:
            notification = spec.get("notification")
    return OverrideEffect(names, frozenset(suppressed), mode or EscalationMode.PER_ORDER, notification)


def _tier_matches(rule_tier: Any, tier: MerchantTier) -> bool:
    if rule_tier == "*":
        return True
    if isinstance(rule_tier, (list, tuple)):
        return tier.value in rule_tier
    return tier.value == rule_tier


def _delay_matches(rule: Mapping[str, Any], delay: int, policy: PolicySnapshot) -> bool:
    """Membership using the externally-declared boundary inclusivity.

    Default convention is lower-inclusive, upper-exclusive with the top band open to
    infinity. The only decisions here — that the 40-minute edge belongs to R3/R4 and
    that R5 starts strictly above it — come from ``boundary_inclusivity`` in the policy
    file, not from code."""
    lo, hi = rule["delay"][0], rule["delay"][1]
    rid = rule["id"]
    binc = policy.raw["boundary_inclusivity"]

    if rid == "R5" and binc.get("R5_lower") == "exclusive":
        lower_ok = delay > lo
    else:
        lower_ok = delay >= lo

    if hi is None:
        upper_ok = True
    elif rid in ("R3", "R4") and binc.get("R3_R4_upper") == "inclusive":
        upper_ok = delay <= hi
    else:
        upper_ok = delay < hi

    return lower_ok and upper_ok


def _matched_rule(state: SessionState, policy: PolicySnapshot) -> Mapping[str, Any] | None:
    for rule in policy.rules:
        if _delay_matches(rule, state.delay_minutes, policy) and _tier_matches(rule["tier"], state.merchant_tier):
            return rule
    return None


REASON_AGENT_REPETITION = "loop_guard_agent_repetition"


def escalate_for_repetition(env: ActionEnvelope, policy: PolicySnapshot) -> ActionEnvelope:
    """Convert a stuck, endlessly-repeating reply into a human handover.

    The policy's own loop-guard threshold decides when: repeating the same non-progressing
    action more times than a merchant is allowed to push the same intent is the same failure
    seen from the other side.
    """
    return ActionEnvelope(
        action=ActionType.ESCALATE,
        rule_id=env.rule_id,
        reason=REASON_AGENT_REPETITION,
        escalation_mode=escalation_mode_for_envelope(env, policy),
        active_overrides=list(env.active_overrides),
        is_terminal=True,
    )


def escalation_mode_for_envelope(env: ActionEnvelope, policy: PolicySnapshot) -> EscalationMode:
    """Attach-to-incident during an outage, per-order otherwise."""
    for name in env.active_overrides:
        spec = policy.override_map.get(name)
        if spec:
            return EscalationMode(spec["escalation_mode"])
    return EscalationMode.PER_ORDER


def _escalate(rule_id: str | None, reason: str, overrides: OverrideEffect) -> ActionEnvelope:
    return ActionEnvelope(
        action=ActionType.ESCALATE,
        rule_id=rule_id,
        reason=reason,
        escalation_mode=overrides.escalation_mode,
        active_overrides=list(overrides.names),
        is_terminal=True,
    )


def _clarify_or_loop_guard(
    rule_id: str, state: SessionState, policy: PolicySnapshot, overrides: OverrideEffect
) -> ActionEnvelope:
    """Ambiguous / off-policy merchant intent: re-prompt, unless the loop guard has
    already been reached — then hand off to a human."""
    if state.off_policy_push_count >= policy.loop_guard_threshold:
        return _escalate(rule_id, REASON_LOOP_GUARD_TRIPPED, overrides)
    return ActionEnvelope(
        action=ActionType.CLARIFY,
        rule_id=rule_id,
        reason=REASON_AMBIGUOUS_INTENT,
        counts_toward_loop_guard=True,
        active_overrides=list(overrides.names),
    )


def _reactive_r2(
    rule_id: str, flags: ClassifierFlags, state: SessionState, policy: PolicySnapshot, overrides: OverrideEffect
) -> ActionEnvelope:
    if flags.requests_cancellation:  # financial -> human (never cancel autonomously)
        return _escalate(rule_id, REASON_CANCELLATION_REQUESTED, overrides)
    if flags.confirms_new_eta:
        return ActionEnvelope(
            action=ActionType.RESOLVE_ETA_CONFIRMED,
            rule_id=rule_id,
            reason=REASON_ETA_CONFIRMED,
            is_terminal=True,
            active_overrides=list(overrides.names),
        )
    if flags.prefers_to_wait:
        return ActionEnvelope(
            action=ActionType.ACKNOWLEDGE_WAIT,
            rule_id=rule_id,
            reason=REASON_MERCHANT_WAITS,
            active_overrides=list(overrides.names),
        )
    return _clarify_or_loop_guard(rule_id, state, policy, overrides)


def _reactive_r4(
    rule_id: str, flags: ClassifierFlags, state: SessionState, policy: PolicySnapshot, overrides: OverrideEffect
) -> ActionEnvelope:
    if flags.requests_cancellation:  # R4 cancellation branch -> mandatory escalation
        return _escalate(rule_id, REASON_CANCELLATION_REQUESTED, overrides)
    if flags.accepts_reassignment and flags.prefers_to_wait:  # conflicting -> don't guess
        return _clarify_or_loop_guard(rule_id, state, policy, overrides)
    if flags.accepts_reassignment:
        return ActionEnvelope(
            action=ActionType.REASSIGN,
            rule_id=rule_id,
            reason=REASON_MERCHANT_ACCEPTS_REASSIGNMENT,
            tool_sequence=list(REASSIGN_TOOL_CHAIN),
            active_overrides=list(overrides.names),
        )
    if flags.prefers_to_wait:
        return ActionEnvelope(
            action=ActionType.ACKNOWLEDGE_WAIT,
            rule_id=rule_id,
            reason=REASON_MERCHANT_WAITS,
            active_overrides=list(overrides.names),
        )
    return _clarify_or_loop_guard(rule_id, state, policy, overrides)


def decide(state: SessionState, flags: ClassifierFlags | None, policy: PolicySnapshot) -> ActionEnvelope:
    """Pure decision pipeline: (state, flags) -> one ActionEnvelope."""
    overrides = _resolve_overrides(state, policy)

    # --- Phase 0: R6 preemption (only a merchant message can request a human) ---
    r6_flag = policy.r6_preemption["flag"]
    if flags is not None and getattr(flags, r6_flag):
        return _escalate("R6", REASON_HUMAN_REQUESTED, overrides)

    # --- Phase 2: first-hit rule on delay + tier ---
    rule = _matched_rule(state, policy)
    if rule is None:  # coverage is validated to be total, so this is a fail-safe only
        return _escalate(None, REASON_NO_MATCHING_RULE, overrides)

    rid = rule["id"]

    # --- Phase 1 applied: override suppresses this rule (e.g. active_outage -> no reassign) ---
    if rid in overrides.suppressed_rules:
        # An override suppresses *reassignment*, not the mandatory escalation triggers. R6 is
        # already exempt (phase 0); cancellation is exempt for the same reason — it has
        # financial consequences and policy sends it to a human. Without this, an outage
        # swallows "just cancel the order" and answers with the degraded-mode notice, which the
        # live evaluation caught as the agent talking past the merchant.
        if flags is not None and flags.requests_cancellation:
            return _escalate(rid, REASON_CANCELLATION_REQUESTED, overrides)
        return ActionEnvelope(
            action=ActionType.DEGRADED_MODE_NOTICE,
            rule_id=rid,
            reason=f"{REASON_OVERRIDE_SUPPRESSED}_{rid}",
            notification=overrides.notification,
            active_overrides=list(overrides.names),
            counts_toward_loop_guard=True,
        )

    action = rule["action"]

    if action == "log_only":  # R1
        return ActionEnvelope(
            action=ActionType.LOG_ONLY, rule_id=rid, reason=REASON_WITHIN_GRACE,
            active_overrides=list(overrides.names),
        )

    if action == "escalate_immediate":  # R5 (do not offer reassignment)
        return _escalate(rid, REASON_DELAY_OVER_THRESHOLD, overrides)

    if action == "auto_reassign":  # R3 (Gold)
        if flags is not None and flags.requests_cancellation:
            return _escalate(rid, REASON_CANCELLATION_REQUESTED, overrides)
        return ActionEnvelope(
            action=ActionType.AUTO_REASSIGN,
            rule_id=rid,
            reason=REASON_GOLD_AUTO_REASSIGN,
            tool_sequence=list(REASSIGN_TOOL_CHAIN),
            active_overrides=list(overrides.names),
        )

    if action == "notify_confirm_eta":  # R2
        if flags is None:
            return ActionEnvelope(
                action=ActionType.NOTIFY_CONFIRM_ETA, rule_id=rid, reason=REASON_NOTIFY_ASK_CONFIRM,
                active_overrides=list(overrides.names),
            )
        return _reactive_r2(rid, flags, state, policy, overrides)

    if action == "ask_reassign_or_wait":  # R4 (non-Gold)
        if flags is None:
            return ActionEnvelope(
                action=ActionType.ASK_REASSIGN_OR_WAIT, rule_id=rid, reason=REASON_ASK_REASSIGN_OR_WAIT,
                active_overrides=list(overrides.names),
            )
        return _reactive_r4(rid, flags, state, policy, overrides)

    return _escalate(rid, REASON_UNKNOWN_ACTION, overrides)  # fail-safe
