"""Step 3 tests: async events interrupting a waiting session, including the hard case of
an event landing while a tool call is in flight. The in-flight call is never cancelled; its
late result is applied if the premise still holds, or reconciled if the session moved on."""

from __future__ import annotations

from pathlib import Path

from care_agent.core.reducer import bootstrap, reduce
from care_agent.core.session import FsmState
from care_agent.domain.models import ClassifierFlags, Event, EventType, MerchantTier, SessionState
from care_agent.policy.loader import load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")


def data(delay: int, tier: MerchantTier = MerchantTier.GOLD) -> SessionState:
    return SessionState(order_id="o1", merchant_name="M", merchant_tier=tier, delay_minutes=delay)


def boot(delay: int, tier: MerchantTier = MerchantTier.GOLD):
    return bootstrap(data(delay, tier), POLICY)


def eta(new_eta: int) -> Event:
    return Event(type=EventType.SYSTEM_ETA_UPDATED, new_eta=new_eta)


def tool(outcome: str, **payload) -> Event:
    return Event(type=EventType.TOOL_RESULT, payload={"outcome": outcome, **payload})


# --- events while AWAITING_MERCHANT_REPLY -------------------------------------


def test_eta_update_pushes_across_boundary_into_r5():
    session, _ = boot(15)  # R2, AWAITING
    session, effects = reduce(session, eta(50), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert session.last_action.rule_id == "R5"


def test_eta_update_drops_delay_back_to_r1():
    session, _ = boot(30, MerchantTier.SILVER)  # R4, AWAITING
    session, effects = reduce(session, eta(5), POLICY)
    assert session.fsm_state is FsmState.MONITORING
    assert session.data.delay_minutes == 5


def test_prep_completed_resolves_while_awaiting_reply():
    session, _ = boot(15)
    session, effects = reduce(session, Event(type=EventType.ORDER_PREP_COMPLETED), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert effects[0].reason == "prep_completed"


# --- events while AWAITING_TOOL_RESULT (in-flight reassignment) ---------------


def test_captain_cancelled_keeps_reassign_in_flight():
    session, _ = boot(30, MerchantTier.GOLD)  # AWAITING_TOOL_RESULT
    session, effects = reduce(session, Event(type=EventType.CAPTAIN_CANCELLED_MID_CALL), POLICY)
    # losing the captain does not invalidate a reassignment — keep awaiting the result
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT
    assert session.pending_tool is not None
    assert effects[0].kind == "log"
    # the in-flight result then applies normally
    session, effects = reduce(session, tool("reassigned", new_captain_id="cap-9", new_eta=10), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert session.data.current_captain_id == "cap-9"


def test_captain_cancelled_does_not_discard_a_non_gold_reassignment():
    """Regression: a non-Gold merchant who already consented is still governed by the
    ask-or-wait rule. Treating that as a dead premise discarded an in-flight reassignment and
    led to a second captain being assigned later."""
    session, _ = boot(30, MerchantTier.SILVER)
    session, _ = reduce(
        session, Event(type=EventType.MERCHANT_MESSAGE, flags=ClassifierFlags(accepts_reassignment=True)), POLICY
    )
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT

    session, effects = reduce(session, Event(type=EventType.CAPTAIN_CANCELLED_MID_CALL), POLICY)
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT  # still awaiting, not re-asked
    assert effects[0].kind == "log"

    session, _ = reduce(session, tool("reassigned", new_captain_id="cap-42", new_eta=9), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert session.data.current_captain_id == "cap-42"


def test_late_successful_reassignment_is_recorded_not_dropped():
    """Even when the session has moved on, a captain that really was assigned must land in
    state — otherwise a later decision could assign a second one on top of it."""
    session, _ = boot(30, MerchantTier.GOLD)
    session, _ = reduce(session, eta(5), POLICY)  # drops to R1, premise genuinely invalidated
    assert session.fsm_state is FsmState.MONITORING

    session, effects = reduce(session, tool("reassigned", new_captain_id="cap-77"), POLICY)
    assert effects[0].kind == "reconcile"
    assert session.data.current_captain_id == "cap-77"


def test_prep_completed_during_flight_then_late_result_reconciles():
    session, _ = boot(30, MerchantTier.GOLD)  # AWAITING_TOOL_RESULT
    session, _ = reduce(session, Event(type=EventType.ORDER_PREP_COMPLETED), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    # the reassign we never cancelled now returns — reconcile, do not act
    session, effects = reduce(session, tool("reassigned", new_captain_id="cap-3"), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert effects[0].kind == "reconcile"


def test_eta_drop_during_flight_invalidates_premise():
    session, _ = boot(30, MerchantTier.GOLD)  # AWAITING_TOOL_RESULT
    session, _ = reduce(session, eta(5), POLICY)  # now R1 — reassignment no longer warranted
    assert session.fsm_state is FsmState.MONITORING
    assert session.pending_tool is None
    session, effects = reduce(session, tool("reassigned", new_captain_id="cap-3"), POLICY)
    assert effects[0].kind == "reconcile"


def test_eta_rise_during_flight_escalates_and_reconciles_late_result():
    session, _ = boot(30, MerchantTier.GOLD)  # AWAITING_TOOL_RESULT
    session, _ = reduce(session, eta(50), POLICY)  # now R5
    assert session.fsm_state is FsmState.ESCALATED
    session, effects = reduce(session, tool("reassigned", new_captain_id="cap-3"), POLICY)
    assert effects[0].kind == "reconcile"


def test_merchant_message_buffered_during_tool_flight():
    session, _ = boot(30, MerchantTier.GOLD)  # AWAITING_TOOL_RESULT
    session, effects = reduce(
        session, Event(type=EventType.MERCHANT_MESSAGE, flags=ClassifierFlags(accepts_reassignment=True)), POLICY
    )
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT  # buffered, not acted on
    assert effects[0].kind == "log"


def test_human_request_during_tool_flight_is_sacred():
    session, _ = boot(30, MerchantTier.GOLD)  # AWAITING_TOOL_RESULT
    session, effects = reduce(
        session, Event(type=EventType.MERCHANT_MESSAGE, flags=ClassifierFlags(requests_human=True)), POLICY
    )
    assert session.fsm_state is FsmState.ESCALATED
    assert session.last_action.rule_id == "R6"
