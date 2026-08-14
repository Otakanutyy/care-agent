"""Step 8 tests: full sessions through the assembled agent — classifier, FSM, policy engine,
tools, guardrail, generator, and escalation together. Fully mocked; no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from care_agent.agent import CareAgent
from care_agent.cli import DEMO_SCENARIO, run_scenario
from care_agent.core.session import FsmState
from care_agent.domain.models import Event, EventType, MerchantTier, SessionState
from care_agent.llm.client import LLMError, MockLLMClient
from care_agent.llm.offline import OfflineLLMClient
from care_agent.policy.loader import load_policy
from care_agent.tools.stubs import ToolConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")


def flags(**kw) -> dict:
    data = {
        "requests_human": False, "confirms_new_eta": False, "requests_cancellation": False,
        "accepts_reassignment": False, "prefers_to_wait": False, "language": "en",
    }
    data.update(kw)
    return data


def context(delay: int, tier: MerchantTier = MerchantTier.SILVER, **kw) -> SessionState:
    return SessionState(
        order_id="order-1", merchant_name="Test Merchant", merchant_tier=tier,
        delay_minutes=delay, current_captain_id="captain-1", **kw,
    )


def agent_with(client, tools: ToolConfig | None = None) -> CareAgent:
    return CareAgent(POLICY, client, tool_config=tools)


def agent_text(agent) -> str:
    return " ".join(e.text for e in agent.transcript if e.speaker == "agent").lower()


# --- happy path: R2 notify -> merchant confirms -> resolved -------------------


def test_r2_confirm_eta_happy_path():
    client = MockLLMClient(
        structured_responses=[flags(confirms_new_eta=True)],
        text_responses=["Your order is running late — does the new time work?", "Thanks for confirming!"],
    )
    agent = agent_with(client).start(context(15))
    assert agent.session.fsm_state is FsmState.AWAITING_MERCHANT_REPLY

    agent.send_message("yes that works for me")

    assert agent.session.fsm_state is FsmState.RESOLVED
    assert agent.session.terminal_reason == "eta_confirmed"
    assert agent.trajectory() == ["notify_confirm_eta", "resolve_eta_confirmed"]
    assert [e.speaker for e in agent.transcript] == ["agent", "merchant", "agent"]
    assert agent.tickets == []
    assert agent.guardrail_violations == 0


# --- escalation: R5 -----------------------------------------------------------


def test_r5_escalates_immediately_with_ticket_and_handoff():
    client = MockLLMClient(text_responses=["I'm connecting you with a colleague."])
    agent = agent_with(client).start(context(50))

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert agent.session.terminal_reason == "delay_over_threshold"
    assert len(agent.tickets) == 1
    ticket = agent.tickets[0]
    assert ticket["reason"] == "delay_over_threshold"
    assert ticket["escalation_mode"] == "per_order"
    assert ticket["ticket_id"]
    # the ops ticket carries enough context for a human to pick it up cold
    assert ticket["context_snapshot"]["order_id"] == "order-1"
    assert ticket["context_snapshot"]["delay_minutes"] == 50
    # and the merchant is told a human is taking over, rather than left in silence
    assert agent.transcript[-1].speaker == "agent"
    assert agent.transcript[-1].action == "escalate"
    # R5 must not offer reassignment
    assert agent.stubs.calls["reassign_captain"] == 0
    # the courtesy handoff message must not double-count as a second policy action
    assert agent.trajectory() == ["escalate"]


# --- R3 Gold: full tool chain -------------------------------------------------


def test_r3_gold_auto_reassign_end_to_end():
    client = MockLLMClient(text_responses=["A new driver is on the way."])
    agent = agent_with(
        client, ToolConfig(available=True, new_captain_id="captain-9", estimated_eta_minutes=8)
    ).start(context(30, MerchantTier.GOLD))

    assert agent.session.fsm_state is FsmState.MONITORING
    assert agent.session.data.current_captain_id == "captain-9"
    assert agent.trajectory() == ["auto_reassign", "notify_reassigned"]
    # availability was checked before the reassignment, exactly once each
    assert agent.stubs.call_log[:2] == ["check_captain_availability", "reassign_captain"]


def test_tool_failure_escalates():
    client = MockLLMClient(text_responses=["Connecting you with a colleague."])
    agent = agent_with(client, ToolConfig(available=False)).start(context(30, MerchantTier.GOLD))

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert agent.tickets[0]["reason"] == "tool_fail_or_no_captain"
    assert agent.stubs.calls["reassign_captain"] == 0


# --- guardrail in a live session ----------------------------------------------


def test_rogue_generator_offer_never_reaches_the_merchant():
    client = MockLLMClient(text_responses=["So sorry! Here's a $50 refund and 20% off your next order."])
    agent = agent_with(client).start(context(15))

    assert agent.guardrail_violations == 1
    assert agent.transcript[0].blocked is True
    text = agent_text(agent)
    assert "refund" not in text and "50" not in text and "20%" not in text


# --- R6 preemption via the deterministic backstop ------------------------------


def test_human_request_escalates_without_calling_the_classifier():
    client = MockLLMClient(text_responses=["Your order is late.", "Connecting you with a colleague."])
    agent = agent_with(client).start(context(15))

    agent.send_message("I want to speak to a human please")

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert agent.session.last_action.rule_id == "R6"
    assert agent.tickets[0]["reason"] == "human_requested"
    # the backstop short-circuits: no classifier call was made at all
    assert [p["kind"] for p in client.prompts] == ["text", "text"]


# --- classifier failure is fail-safe -------------------------------------------


def test_classifier_failure_escalates_rather_than_guessing():
    client = MockLLMClient(
        structured_responses=[LLMError("service unavailable")],
        text_responses=["Your order is late.", "Connecting you with a colleague."],
    )
    agent = agent_with(client).start(context(15))

    agent.send_message("something ambiguous the model cannot read")

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert agent.tickets[0]["reason"] == "classifier_unavailable"


# --- mid-call event injection ---------------------------------------------------


def test_eta_update_mid_conversation_escalates():
    client = MockLLMClient(text_responses=["Your order is late.", "Connecting you with a colleague."])
    agent = agent_with(client).start(context(15))
    assert agent.session.fsm_state is FsmState.AWAITING_MERCHANT_REPLY

    agent.send_event(Event(type=EventType.SYSTEM_ETA_UPDATED, new_eta=50))

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert agent.session.last_action.rule_id == "R5"


def test_prep_completed_mid_conversation_resolves():
    client = MockLLMClient(text_responses=["Your order is late."])
    agent = agent_with(client).start(context(15))

    agent.send_event(Event(type=EventType.ORDER_PREP_COMPLETED))

    assert agent.session.fsm_state is FsmState.RESOLVED
    assert agent.session.terminal_reason == "prep_completed"


# --- cancellation always goes to a human ----------------------------------------


def test_cancellation_request_escalates():
    client = MockLLMClient(
        structured_responses=[flags(requests_cancellation=True)],
        text_responses=["Your order is delayed.", "Connecting you with a colleague."],
    )
    agent = agent_with(client).start(context(30))

    agent.send_message("just cancel the order")

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert agent.tickets[0]["reason"] == "cancellation_requested"
    assert agent.stubs.calls["cancel_order"] == 0  # the agent never cancels on its own


# --- multilingual ----------------------------------------------------------------


def test_reply_language_follows_the_merchant():
    client = MockLLMClient(
        structured_responses=[flags(confirms_new_eta=True, language="ar")],
        text_responses=["Your order is late.", "شكرًا لتأكيدك."],
    )
    agent = agent_with(client).start(context(15))

    agent.send_message("تمام، ما في مشكلة")

    assert agent.language == "ar"
    assert agent.transcript[-1].language == "ar"
    # the generator was asked for Arabic
    assert "Arabic" in client.prompts[-1]["user"]


# --- terminal sessions are inert --------------------------------------------------


def test_terminal_session_ignores_further_input():
    client = MockLLMClient(text_responses=["Connecting you with a colleague."])
    agent = agent_with(client).start(context(50))  # escalates immediately
    before = len(agent.transcript)

    agent.send_event(Event(type=EventType.ORDER_PREP_COMPLETED))

    assert agent.session.fsm_state is FsmState.ESCALATED
    assert len(agent.transcript) == before  # nothing new was said
    assert len(agent.tickets) == 1          # and no second ticket


def test_sending_before_start_raises():
    agent = agent_with(MockLLMClient())
    with pytest.raises(RuntimeError):
        agent.send_message("hello")


# --- CLI / scenario runner --------------------------------------------------------


def test_cli_demo_scenario_runs_offline():
    agent = run_scenario(DEMO_SCENARIO, OfflineLLMClient(), policy=POLICY)
    assert agent.session.fsm_state is FsmState.MONITORING
    assert agent.session.terminal_reason is None  # reassigned, but the conversation stays open
    assert agent.session.data.current_captain_id == "captain-207"


def test_scenario_runner_handles_events_and_stops_at_terminal():
    scenario = {
        "id": "T1",
        "context": {
            "order_id": "o9", "merchant_name": "M", "merchant_tier": "Gold",
            "delay_minutes": 15, "current_captain_id": "c1", "active_system_overrides": [],
        },
        "tools": {},
        "script": [
            {"type": "event", "event": "SYSTEM_ETA_UPDATED", "new_eta": 50},  # -> R5, escalates
            {"type": "message", "text": "hello?"},                            # skipped: terminal
        ],
    }
    agent = run_scenario(scenario, OfflineLLMClient(), policy=POLICY)
    assert agent.session.fsm_state is FsmState.ESCALATED
    assert not any(e.speaker == "merchant" for e in agent.transcript)
