"""Step 3 tests: the FSM reducer drives sessions through the right states and effects,
and the single-writer mailbox serializes per-order events."""

from __future__ import annotations

from pathlib import Path

from care_agent.core.mailbox import Mailbox
from care_agent.core.reducer import Runtime, bootstrap, reduce
from care_agent.core.session import FsmState, LogEffect, Session
from care_agent.core.stores import InMemoryEventLog, InMemoryStateStore
from care_agent.domain.models import ClassifierFlags, Event, EventType, MerchantTier, SessionState
from care_agent.policy.loader import load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")


def data(delay: int, tier: MerchantTier = MerchantTier.GOLD, **kw) -> SessionState:
    return SessionState(order_id="o1", merchant_name="M", merchant_tier=tier, delay_minutes=delay, **kw)


def boot(delay: int, tier: MerchantTier = MerchantTier.GOLD, **kw):
    return bootstrap(data(delay, tier, **kw), POLICY)


def msg(**flags) -> Event:
    return Event(type=EventType.MERCHANT_MESSAGE, flags=ClassifierFlags(**flags))


def tool(outcome: str, **payload) -> Event:
    return Event(type=EventType.TOOL_RESULT, payload={"outcome": outcome, **payload})


def kinds(effects) -> list[str]:
    return [e.kind for e in effects]


# --- bootstrap / proactive ----------------------------------------------------


def test_bootstrap_r1_monitors():
    session, effects = boot(5)
    assert session.fsm_state is FsmState.MONITORING
    assert kinds(effects) == ["log"]


def test_bootstrap_r2_awaits_reply():
    session, effects = boot(15)
    assert session.fsm_state is FsmState.AWAITING_MERCHANT_REPLY
    assert kinds(effects) == ["send_message"]


def test_bootstrap_r3_dispatches_tool_chain():
    session, effects = boot(30, MerchantTier.GOLD)
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT
    assert kinds(effects) == ["call_tool_chain"]
    assert effects[0].tool_sequence == ["check_captain_availability", "reassign_captain"]
    assert session.pending_tool is not None


def test_bootstrap_r5_escalates():
    session, effects = boot(50)
    assert session.fsm_state is FsmState.ESCALATED
    assert effects[0].kind == "escalate"
    assert effects[0].escalation_mode.value == "per_order"


# --- reactive turns -----------------------------------------------------------


def test_r2_confirm_resolves():
    session, _ = boot(15)
    session, effects = reduce(session, msg(confirms_new_eta=True), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert set(kinds(effects)) == {"send_message", "resolve"}


def test_r2_cancel_escalates():
    session, _ = boot(15)
    session, effects = reduce(session, msg(requests_cancellation=True), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert effects[0].reason == "cancellation_requested"


def test_r4_accept_reassign_dispatches_tool():
    session, _ = boot(30, MerchantTier.SILVER)
    assert session.fsm_state is FsmState.AWAITING_MERCHANT_REPLY
    session, effects = reduce(session, msg(accepts_reassignment=True), POLICY)
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT
    assert kinds(effects) == ["call_tool_chain"]


def test_r6_from_awaiting_escalates():
    session, _ = boot(15)
    session, effects = reduce(session, msg(requests_human=True), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert session.last_action.rule_id == "R6"


# --- tool results -------------------------------------------------------------


def test_reassign_success_resolves_and_updates_captain():
    session, _ = boot(30, MerchantTier.GOLD)
    session, effects = reduce(session, tool("reassigned", new_captain_id="cap-2", new_eta=12), POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert session.data.current_captain_id == "cap-2"
    assert set(kinds(effects)) == {"send_message", "resolve"}


def test_reassign_no_captain_escalates():
    session, _ = boot(30, MerchantTier.GOLD)
    session, effects = reduce(session, tool("no_captain"), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert effects[0].reason == "tool_fail_or_no_captain"


def test_reassign_failed_escalates():
    session, _ = boot(30, MerchantTier.GOLD)
    session, effects = reduce(session, tool("failed"), POLICY)
    assert session.fsm_state is FsmState.ESCALATED


# --- timeouts -----------------------------------------------------------------


def test_reply_timeout_reprompts_then_escalates_on_second():
    session, _ = boot(15)
    session, effects = reduce(session, Event(type=EventType.MERCHANT_REPLY_TIMEOUT), POLICY)
    assert session.fsm_state is FsmState.AWAITING_MERCHANT_REPLY  # re-notified
    assert session.data.unresponsive_attempts_used == 1
    session, effects = reduce(session, Event(type=EventType.MERCHANT_REPLY_TIMEOUT), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert effects[0].reason == "unresponsive"


def test_tool_timeout_escalates():
    session, _ = boot(30, MerchantTier.GOLD)
    session, effects = reduce(session, Event(type=EventType.TOOL_TIMEOUT), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert effects[0].reason == "tool_timeout"


# --- loop guard trajectory ----------------------------------------------------


def test_loop_guard_escalates_after_repeated_off_policy():
    session, _ = boot(30, MerchantTier.SILVER)  # R4 -> AWAITING
    # three ambiguous replies re-prompt; the fourth trips the loop guard (threshold 3)
    for _ in range(POLICY.loop_guard_threshold):
        session, effects = reduce(session, msg(), POLICY)
        assert session.fsm_state is FsmState.AWAITING_MERCHANT_REPLY
        assert effects[0].kind == "send_message"
    session, effects = reduce(session, msg(), POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert effects[0].reason == "loop_guard_tripped"


# --- terminal states are inert ------------------------------------------------


def test_terminal_states_are_inert():
    session, _ = boot(15)
    session, _ = reduce(session, msg(confirms_new_eta=True), POLICY)  # RESOLVED
    assert session.fsm_state is FsmState.RESOLVED
    session, effects = reduce(session, msg(requests_cancellation=True), POLICY)
    assert session.fsm_state is FsmState.RESOLVED  # unchanged
    assert effects[0].kind == "log"


# --- Runtime + Mailbox --------------------------------------------------------


def test_runtime_persists_and_logs():
    store, log = InMemoryStateStore(), InMemoryEventLog()
    rt = Runtime(POLICY, store, log)
    mb = Mailbox(rt)
    mb.start("o1", data(15))
    mb.send("o1", msg(confirms_new_eta=True))
    assert store.get("o1").fsm_state is FsmState.RESOLVED
    assert [e.type for e in log.events("o1")] == [EventType.MERCHANT_MESSAGE]


def test_mailbox_serializes_reentrant_sends_fifo():
    """A reentrant send during processing is queued and handled after the current event —
    never interleaved. This is the single-writer guarantee in action."""

    processed: list[EventType] = []

    class ReentrantRuntime:
        mailbox: Mailbox

        def start(self, order_id, data_):
            return []

        def apply(self, order_id, event):
            processed.append(event.type)
            if event.type is EventType.TICK and len(processed) == 1:
                # inject a second event mid-processing of the first
                self.mailbox.send(order_id, Event(type=EventType.SESSION_IDLE_TIMEOUT))
            return [LogEffect(message=event.type.value)]

    rt = ReentrantRuntime()
    mb = Mailbox(rt)
    rt.mailbox = mb
    effects = mb.send("o1", Event(type=EventType.TICK))
    assert processed == [EventType.TICK, EventType.SESSION_IDLE_TIMEOUT]  # in order, not nested
    assert len(effects) == 2  # both drained under the one top-level send
