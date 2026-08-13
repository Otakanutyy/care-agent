"""Step 2 tests: the pure decision pipeline.

Covers every rule (R1-R6), both sides of the 40-minute boundary, override suppression +
attach-to-incident escalation, R4's mandatory-escalation cancellation branch, boolean
precedence, conflict/all-false handling, and the loop guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from care_agent.domain.models import (
    ActionType,
    ClassifierFlags,
    EscalationMode,
    MerchantTier,
    SessionState,
)
from care_agent.policy.engine import decide
from care_agent.policy.loader import load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")


def state(delay: int, tier: MerchantTier = MerchantTier.GOLD, overrides=None, **kw) -> SessionState:
    return SessionState(
        order_id="order-1",
        merchant_name="Test Merchant",
        merchant_tier=tier,
        delay_minutes=delay,
        active_system_overrides=list(overrides or []),
        **kw,
    )


def flags(**kw) -> ClassifierFlags:
    return ClassifierFlags(**kw)


# --- R1: log only -------------------------------------------------------------


@pytest.mark.parametrize("tier", list(MerchantTier))
def test_r1_log_only(tier):
    env = decide(state(5, tier), None, POLICY)
    assert env.action is ActionType.LOG_ONLY
    assert env.rule_id == "R1"
    assert not env.is_terminal


# --- R2: notify / confirm -----------------------------------------------------


def test_r2_proactive_notifies():
    env = decide(state(15, MerchantTier.SILVER), None, POLICY)
    assert env.action is ActionType.NOTIFY_CONFIRM_ETA
    assert env.rule_id == "R2"


def test_r2_confirms_eta_resolves():
    env = decide(state(15), flags(confirms_new_eta=True), POLICY)
    assert env.action is ActionType.RESOLVE_ETA_CONFIRMED
    assert env.is_terminal


def test_r2_cancellation_escalates():
    env = decide(state(15), flags(requests_cancellation=True), POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.reason == "cancellation_requested"


def test_r2_all_false_clarifies():
    env = decide(state(15), flags(), POLICY)
    assert env.action is ActionType.CLARIFY
    assert env.counts_toward_loop_guard


# --- R3: Gold auto-reassign ---------------------------------------------------


def test_r3_gold_auto_reassign_with_tool_chain():
    env = decide(state(30, MerchantTier.GOLD), None, POLICY)
    assert env.action is ActionType.AUTO_REASSIGN
    assert env.rule_id == "R3"
    # tool chaining: availability MUST be checked before reassignment
    assert env.tool_sequence == ["check_captain_availability", "reassign_captain"]


def test_r3_cancellation_escalates():
    env = decide(state(30, MerchantTier.GOLD), flags(requests_cancellation=True), POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.reason == "cancellation_requested"


# --- R4: non-Gold ask reassign or wait ---------------------------------------


@pytest.mark.parametrize("tier", [MerchantTier.SILVER, MerchantTier.BRONZE])
def test_r4_proactive_asks(tier):
    env = decide(state(30, tier), None, POLICY)
    assert env.action is ActionType.ASK_REASSIGN_OR_WAIT
    assert env.rule_id == "R4"


def test_r4_accepts_reassignment():
    env = decide(state(30, MerchantTier.SILVER), flags(accepts_reassignment=True), POLICY)
    assert env.action is ActionType.REASSIGN
    assert env.tool_sequence == ["check_captain_availability", "reassign_captain"]


def test_r4_prefers_to_wait():
    env = decide(state(30, MerchantTier.BRONZE), flags(prefers_to_wait=True), POLICY)
    assert env.action is ActionType.ACKNOWLEDGE_WAIT


def test_r4_cancellation_mandatory_escalation():
    env = decide(state(30, MerchantTier.SILVER), flags(requests_cancellation=True), POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.reason == "cancellation_requested"


def test_r4_conflicting_reassign_and_wait_clarifies():
    env = decide(state(30, MerchantTier.SILVER), flags(accepts_reassignment=True, prefers_to_wait=True), POLICY)
    assert env.action is ActionType.CLARIFY
    assert env.counts_toward_loop_guard


# --- R5: escalate immediately -------------------------------------------------


@pytest.mark.parametrize("tier", list(MerchantTier))
def test_r5_escalates_immediately(tier):
    env = decide(state(50, tier), None, POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.rule_id == "R5"
    assert env.reason == "delay_over_threshold"
    assert env.escalation_mode is EscalationMode.PER_ORDER
    # R5 must not offer reassignment
    assert env.tool_sequence == []


# --- R6: preemption -----------------------------------------------------------


def test_r6_preempts_even_within_grace():
    # delay=5 would be R1 log-only, but requests_human must win.
    env = decide(state(5), flags(requests_human=True), POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.rule_id == "R6"
    assert env.reason == "human_requested"


def test_r6_preempts_reassignment_band():
    env = decide(state(30, MerchantTier.GOLD), flags(requests_human=True), POLICY)
    assert env.rule_id == "R6"


# --- 40-minute boundary -------------------------------------------------------


def test_boundary_40_is_r3_for_gold():
    env = decide(state(40, MerchantTier.GOLD), None, POLICY)
    assert env.rule_id == "R3"
    assert env.action is ActionType.AUTO_REASSIGN


def test_boundary_40_is_r4_for_non_gold():
    env = decide(state(40, MerchantTier.SILVER), None, POLICY)
    assert env.rule_id == "R4"


def test_boundary_41_is_r5():
    env = decide(state(41, MerchantTier.GOLD), None, POLICY)
    assert env.rule_id == "R5"
    assert env.action is ActionType.ESCALATE


def test_boundary_lower_edges():
    assert decide(state(9), None, POLICY).rule_id == "R1"
    assert decide(state(10), None, POLICY).rule_id == "R2"
    assert decide(state(19), None, POLICY).rule_id == "R2"
    assert decide(state(20, MerchantTier.GOLD), None, POLICY).rule_id == "R3"


# --- active_system_overrides (active_outage) ----------------------------------


def test_outage_suppresses_reassignment_with_degraded_notice():
    env = decide(state(30, MerchantTier.GOLD, overrides=["active_outage"]), None, POLICY)
    assert env.action is ActionType.DEGRADED_MODE_NOTICE
    assert env.notification == "degraded_mode_no_eta"
    assert "active_outage" in env.active_overrides


def test_outage_suppresses_r4_too():
    env = decide(state(30, MerchantTier.SILVER, overrides=["active_outage"]), None, POLICY)
    assert env.action is ActionType.DEGRADED_MODE_NOTICE


def test_outage_escalation_is_attach_to_incident_not_per_order():
    env = decide(state(50, overrides=["active_outage"]), None, POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.escalation_mode is EscalationMode.ATTACH_TO_INCIDENT


def test_outage_still_honors_human_request_via_incident():
    env = decide(state(30, MerchantTier.GOLD, overrides=["active_outage"]), flags(requests_human=True), POLICY)
    assert env.rule_id == "R6"
    assert env.action is ActionType.ESCALATE
    assert env.escalation_mode is EscalationMode.ATTACH_TO_INCIDENT


def test_outage_does_not_suppress_r1_or_r2():
    assert decide(state(5, overrides=["active_outage"]), None, POLICY).action is ActionType.LOG_ONLY
    assert decide(state(15, overrides=["active_outage"]), None, POLICY).action is ActionType.NOTIFY_CONFIRM_ETA


def test_unknown_override_is_ignored():
    env = decide(state(30, MerchantTier.GOLD, overrides=["not_a_real_override"]), None, POLICY)
    assert env.action is ActionType.AUTO_REASSIGN  # falls through to R3


# --- boolean precedence -------------------------------------------------------


def test_cancellation_outranks_reassignment():
    # both flags set; requests_cancellation has higher precedence -> escalate, not reassign
    env = decide(state(30, MerchantTier.SILVER), flags(requests_cancellation=True, accepts_reassignment=True), POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.reason == "cancellation_requested"


def test_human_outranks_cancellation():
    env = decide(state(30, MerchantTier.SILVER), flags(requests_human=True, requests_cancellation=True), POLICY)
    assert env.rule_id == "R6"


# --- loop guard ---------------------------------------------------------------


def test_loop_guard_clarifies_below_threshold():
    st = state(30, MerchantTier.SILVER, off_policy_push_count=POLICY.loop_guard_threshold - 1)
    env = decide(st, flags(), POLICY)
    assert env.action is ActionType.CLARIFY


def test_loop_guard_escalates_at_threshold():
    st = state(30, MerchantTier.SILVER, off_policy_push_count=POLICY.loop_guard_threshold)
    env = decide(st, flags(), POLICY)
    assert env.action is ActionType.ESCALATE
    assert env.reason == "loop_guard_tripped"


# --- purity -------------------------------------------------------------------


def test_decide_is_pure_and_repeatable():
    st = state(30, MerchantTier.SILVER)
    f = flags(accepts_reassignment=True)
    first = decide(st, f, POLICY)
    second = decide(st, f, POLICY)
    assert first == second
    # inputs untouched
    assert st.delay_minutes == 30
    assert st.off_policy_push_count == 0
