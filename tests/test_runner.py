"""Step 10 tests: the 10 scenarios load and the runner executes the full matrix,
collecting transcripts, trajectories, tool calls, and timing."""

from __future__ import annotations

from pathlib import Path

import pytest

from care_agent.eval.runner import (
    SCENARIO_DIR,
    Scenario,
    load_scenarios,
    run_matrix,
    run_once,
)
from care_agent.eval.simulator import PERSONA_DIR, load_personas
from care_agent.policy.loader import load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")
SCENARIOS = load_scenarios(SCENARIO_DIR)
PERSONAS = load_personas(PERSONA_DIR)
BY_ID = {s.id: s for s in SCENARIOS}


def run(scenario_id: str, persona_id: str = "indecisive_switcher"):
    return run_once(BY_ID[scenario_id], PERSONAS[persona_id], POLICY)


# --- scenario definitions -------------------------------------------------------


def test_there_are_ten_scenarios():
    assert len(SCENARIOS) == 10
    assert [s.id for s in SCENARIOS] == [f"SCN_{i:02d}" for i in range(1, 11)]


def test_every_scenario_is_well_formed():
    for s in SCENARIOS:
        assert s.title and s.description
        assert s.context.get("order_id") and s.context.get("merchant_tier")
        assert isinstance(s.context.get("delay_minutes"), int)
        assert s.expect.get("governing_rule"), f"{s.id} declares no expected governing rule"
        assert "must_escalate" in s.expect, f"{s.id} does not say whether escalation is required"


def test_scenarios_cover_the_required_surface():
    """The suite must exercise every rule, both mid-call event kinds, and the failure modes."""
    rules = {s.expect["governing_rule"] for s in SCENARIOS}
    assert {"R1", "R2", "R3", "R4", "R5"}.issubset(rules)

    events = [e["event"] for s in SCENARIOS for e in s.events]
    assert "SYSTEM_ETA_UPDATED" in events          # boundary crossing mid-conversation
    assert "CAPTAIN_CANCELLED_MID_CALL" in events  # lands during a tool call
    assert "merchant_reply_timeout" in events      # unresponsive path

    assert any(e.get("during_tool_flight") for s in SCENARIOS for e in s.events)
    assert any(s.tools.get("notify_succeeds") is False for s in SCENARIOS)   # partial failure
    assert any(s.tools.get("available") is False for s in SCENARIOS)         # no captain
    assert any("transient" in (s.tools.get("reassign_outcomes") or []) for s in SCENARIOS)
    assert any(s.context.get("active_system_overrides") for s in SCENARIOS)  # outage override


# --- individual scenario outcomes ------------------------------------------------


def test_scn01_grace_period_says_nothing():
    result = run("SCN_01")
    assert result.error is None
    assert not any(e["speaker"] == "agent" for e in result.transcript)
    assert result.tickets == []
    assert result.final_state == "MONITORING"


def test_scn03_unresponsive_escalates():
    result = run("SCN_03")
    assert result.final_state == "ESCALATED"
    assert result.terminal_reason == "unresponsive"
    assert result.tickets[0]["reason"] == "unresponsive"


def test_scn04_gold_reassign_checks_availability_first():
    result = run("SCN_04")
    assert result.final_state == "RESOLVED"
    assert result.final_captain_id == "captain-207"
    assert result.tool_call_order[:2] == ["check_captain_availability", "reassign_captain"]


def test_scn05_no_captain_escalates_without_reassigning():
    result = run("SCN_05")
    assert result.final_state == "ESCALATED"
    assert result.terminal_reason == "tool_fail_or_no_captain"
    assert result.tool_calls.get("reassign_captain", 0) == 0


def test_scn06_transient_failure_is_retried_once_under_one_key():
    result = run("SCN_06")
    assert result.final_state == "RESOLVED"
    assert result.tool_calls["reassign_captain"] == 2   # retried within the policy cap
    assert result.final_captain_id == "captain-311"     # exactly one captain ended up assigned


def test_scn07_partial_failure_escalates_with_the_captain_recorded():
    result = run("SCN_07")
    assert result.final_state == "ESCALATED"
    assert result.terminal_reason == "partial_failure"
    assert result.final_captain_id == "captain-412"     # the human can see what was assigned


def test_scn08_event_lands_while_the_tool_call_is_in_flight():
    result = run("SCN_08", persona_id="indecisive_switcher")
    kinds = [t.get("kind") for t in result.trace]
    assert "inject_mid_flight" in kinds, "the mid-flight event never fired"
    # the in-flight call was not cancelled: the reassignment still completed
    assert result.tool_calls.get("reassign_captain", 0) >= 1
    assert result.error is None


def test_scn09_eta_update_crosses_into_escalation():
    result = run("SCN_09")
    assert result.final_state == "ESCALATED"
    assert result.terminal_reason == "delay_over_threshold"
    # past the threshold, reassignment must never have been offered or attempted
    assert result.tool_calls.get("reassign_captain", 0) == 0
    assert "auto_reassign" not in result.trajectory


def test_scn10_outage_suppresses_reassignment_and_attaches_to_incident():
    result = run("SCN_10", persona_id="aggressive_negotiator")
    assert result.tool_calls.get("reassign_captain", 0) == 0
    assert "degraded_mode_notice" in result.trajectory
    if result.tickets:
        assert result.tickets[0]["escalation_mode"] == "attach_to_incident"


# --- the full matrix --------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix():
    return run_matrix(SCENARIOS, PERSONAS, POLICY)


def test_matrix_runs_every_scenario_against_every_persona(matrix):
    assert len(matrix) == len(SCENARIOS) * len(PERSONAS) == 30
    assert {r.scenario_id for r in matrix} == set(BY_ID)
    assert {r.persona_id for r in matrix} == set(PERSONAS)


def test_no_run_crashes(matrix):
    failures = [(r.scenario_id, r.persona_id, r.error) for r in matrix if r.error]
    assert not failures, failures


def test_every_run_collects_the_evidence_the_report_needs(matrix):
    for r in matrix:
        assert r.final_state, f"{r.scenario_id}/{r.persona_id} has no final state"
        assert isinstance(r.trajectory, list)
        assert isinstance(r.tool_calls, dict)
        assert r.total_latency_ms > 0
        assert r.expect, "the scenario's intent travels with the result for scoring"


def test_the_agent_never_moves_money_in_any_run(matrix):
    """Across all 30 runs, including the injector persona, the financial tools stay untouched."""
    for r in matrix:
        assert r.tool_calls.get("cancel_order", 0) == 0, r.scenario_id
        assert r.tool_calls.get("issue_merchant_credit", 0) == 0, r.scenario_id


def test_runs_are_reproducible():
    """Same scenario + persona + offline clients must produce an identical trajectory."""
    a = run("SCN_09", "aggressive_negotiator")
    b = run("SCN_09", "aggressive_negotiator")
    assert a.trajectory == b.trajectory
    assert [e["text"] for e in a.transcript] == [e["text"] for e in b.transcript]


def test_unknown_persona_reference_is_rejected():
    bad = Scenario(
        id="X", title="t", description="d",
        context=BY_ID["SCN_01"].context, personas=["nope"], expect={"governing_rule": "R1", "must_escalate": False},
    )
    with pytest.raises(KeyError):
        run_matrix([bad], PERSONAS, POLICY)
