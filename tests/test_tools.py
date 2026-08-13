"""Step 4 tests: the tool broker — chaining, idempotency (no double-assign), retry cap,
partial-failure handling, key enforcement — and its results driving the FSM."""

from __future__ import annotations

from pathlib import Path

import pytest

from care_agent.core.reducer import bootstrap, reduce
from care_agent.core.session import FsmState
from care_agent.core.stores import InMemoryIdempotencyStore
from care_agent.domain.models import MerchantTier, SessionState
from care_agent.policy.loader import load_policy
from care_agent.tools.broker import ToolBroker
from care_agent.tools.stubs import ToolConfig, ToolStubs

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")

ORDER = "order-1"
KEY = "order-1:reassign:e0"


def broker(config: ToolConfig | None = None) -> tuple[ToolBroker, ToolStubs]:
    stubs = ToolStubs(config)
    return ToolBroker(stubs, InMemoryIdempotencyStore(), POLICY), stubs


# --- chaining -----------------------------------------------------------------


def test_chain_checks_availability_before_reassign():
    b, stubs = broker(ToolConfig(available=True))
    b.run_reassign_chain(ORDER, KEY)
    assert stubs.call_log.index("check_captain_availability") < stubs.call_log.index("reassign_captain")


def test_unavailable_captain_never_reassigns():
    b, stubs = broker(ToolConfig(available=False))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "no_captain"
    assert stubs.calls["reassign_captain"] == 0  # gated by the availability check


def test_check_error_reports_failed():
    b, stubs = broker(ToolConfig(check_error=True))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "failed"
    assert stubs.calls["reassign_captain"] == 0


def test_happy_path_reassigns():
    b, _ = broker(ToolConfig(available=True, new_captain_id="cap-7", estimated_eta_minutes=9))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "reassigned"
    assert result.payload["new_captain_id"] == "cap-7"
    assert result.payload["new_eta"] == 9


# --- retries + idempotency ----------------------------------------------------


def test_transient_failure_retried_within_cap():
    b, stubs = broker(ToolConfig(reassign_outcomes=["transient", "success"]))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "reassigned"
    assert stubs.calls["reassign_captain"] == 2  # retried once, within cap of 2


def test_retry_cap_exhausted_fails():
    b, stubs = broker(ToolConfig(reassign_outcomes=["transient", "transient", "transient"]))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "failed"
    assert stubs.calls["reassign_captain"] == POLICY.retry_cap  # stopped at the cap


def test_terminal_failure_not_retried():
    b, stubs = broker(ToolConfig(reassign_outcomes=["terminal"]))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "failed"
    assert stubs.calls["reassign_captain"] == 1  # no retry on a terminal failure


def test_same_key_is_idempotent_no_double_assign():
    b, stubs = broker(ToolConfig(available=True, new_captain_id="cap-1"))
    first = b.run_reassign_chain(ORDER, KEY)
    second = b.run_reassign_chain(ORDER, KEY)  # replay with the same key
    assert first.payload["new_captain_id"] == second.payload["new_captain_id"] == "cap-1"
    assert stubs.calls["reassign_captain"] == 1  # the second call did NOT re-assign


def test_reassign_requires_idempotency_key():
    b, _ = broker()
    with pytest.raises(ValueError):
        b.run_reassign_chain(ORDER, "")


# --- partial failure ----------------------------------------------------------


def test_partial_failure_when_confirmation_fails():
    b, _ = broker(ToolConfig(available=True, new_captain_id="cap-5", notify_succeeds=False))
    result = b.run_reassign_chain(ORDER, KEY)
    assert result.payload["outcome"] == "partial_failure"
    assert result.payload["new_captain_id"] == "cap-5"  # captain WAS assigned


# --- financial tools: key-enforced + idempotent -------------------------------


def test_financial_tools_require_key():
    b, _ = broker()
    with pytest.raises(ValueError):
        b.cancel_order(ORDER, reason="x", waive_fee=False, idempotency_key="")
    with pytest.raises(ValueError):
        b.issue_merchant_credit("merchant-1", amount=10, currency="AED", idempotency_key="")


def test_financial_tool_idempotent_replay():
    b, stubs = broker()
    b.issue_merchant_credit("merchant-1", amount=10, currency="AED", idempotency_key="k1")
    again = b.issue_merchant_credit("merchant-1", amount=10, currency="AED", idempotency_key="k1")
    assert again.get("idempotent_replay") is True
    assert stubs.calls["issue_merchant_credit"] == 1  # not double-issued


# --- escalation tool ----------------------------------------------------------


def test_escalate_returns_ticket():
    b, _ = broker(ToolConfig(ticket_id="T-42"))
    result = b.escalate(ORDER, reason="delay_over_threshold", context_snapshot={"order_id": ORDER})
    assert result["success"] is True
    assert result["ticket_id"] == "T-42"


# --- broker results driving the FSM (Step 3 <-> Step 4) -----------------------


def _session_awaiting_tool():
    data = SessionState(order_id=ORDER, merchant_name="M", merchant_tier=MerchantTier.GOLD, delay_minutes=30)
    session, _ = bootstrap(data, POLICY)  # R3 -> AWAITING_TOOL_RESULT
    assert session.fsm_state is FsmState.AWAITING_TOOL_RESULT
    return session


def test_broker_reassigned_result_resolves_session():
    b, _ = broker(ToolConfig(new_captain_id="cap-2"))
    session = _session_awaiting_tool()
    result = b.run_reassign_chain(ORDER, session.pending_tool.idempotency_key)
    session, _ = reduce(session, result, POLICY)
    assert session.fsm_state is FsmState.RESOLVED
    assert session.data.current_captain_id == "cap-2"


def test_broker_partial_failure_escalates_with_captain_recorded():
    b, _ = broker(ToolConfig(new_captain_id="cap-2", notify_succeeds=False))
    session = _session_awaiting_tool()
    result = b.run_reassign_chain(ORDER, session.pending_tool.idempotency_key)
    session, effects = reduce(session, result, POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert session.terminal_reason == "partial_failure"
    assert session.data.current_captain_id == "cap-2"  # human sees the captain was assigned


def test_broker_no_captain_result_escalates():
    b, _ = broker(ToolConfig(available=False))
    session = _session_awaiting_tool()
    result = b.run_reassign_chain(ORDER, session.pending_tool.idempotency_key)
    session, _ = reduce(session, result, POLICY)
    assert session.fsm_state is FsmState.ESCALATED
    assert session.terminal_reason == "tool_fail_or_no_captain"
