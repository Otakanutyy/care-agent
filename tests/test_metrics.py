"""Step 11 tests: oracle-based scoring, the subjective judge, and the report.

The point of these tests is that the scorer must *fail* runs that deserve to fail — a metric
that always returns 1.0 proves nothing. Each check is exercised with a deliberately broken run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from care_agent.eval.judge import JUDGE_SCHEMA, build_prompt, judge_all, judge_run
from care_agent.eval.metrics import (
    count_guardrail_violations,
    expected_opening_action,
    idempotency_observed,
    score_all,
    score_run,
)
from care_agent.eval.report import build_report, render_markdown, write_report
from care_agent.eval.runner import RunResult
from care_agent.llm.client import LLMError, MockLLMClient
from care_agent.policy.loader import load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy(REPO_ROOT / "policy" / "policy.json")

CONTEXT = {
    "order_id": "o1", "merchant_name": "M", "merchant_tier": "Gold",
    "delay_minutes": 30, "current_captain_id": "c1", "active_system_overrides": [],
}


def make_run(**overrides) -> RunResult:
    """A clean, passing Gold auto-reassign run; override fields to break it."""
    data = dict(
        scenario_id="SCN_X", persona_id="p1", title="t",
        transcript=[{"speaker": "agent", "text": "A new driver is on the way."}],
        trajectory=["auto_reassign", "notify_reassigned"],
        tool_calls={"check_captain_availability": 1, "reassign_captain": 1, "notify_merchant": 1},
        tool_call_order=["check_captain_availability", "reassign_captain", "notify_merchant"],
        idempotency_keys=["o1:reassign:e0"],
        reassign_dispatches=1,
        final_state="RESOLVED", terminal_reason="reassigned", turns_used=1, total_latency_ms=12.0,
        expect={"_context": CONTEXT, "governing_rule": "R3", "must_escalate": False,
                "terminal_state": "RESOLVED"},
    )
    data.update(overrides)
    return RunResult(**data)


def failed_check_names(scored) -> set[str]:
    return {c.name for c in scored.checks if not c.passed}


# --- the oracle tracks the policy file, not a hard-coded answer ------------------


@pytest.mark.parametrize(
    "delay,tier,expected",
    [(5, "Gold", "log_only"), (15, "Gold", "notify_confirm_eta"), (30, "Gold", "auto_reassign"),
     (30, "Silver", "ask_reassign_or_wait"), (50, "Gold", "escalate")],
)
def test_oracle_derives_the_opening_action_from_the_policy(delay, tier, expected):
    run = make_run(expect={"_context": {**CONTEXT, "delay_minutes": delay, "merchant_tier": tier}})
    assert expected_opening_action(run, POLICY) == expected


def test_oracle_respects_the_forty_minute_boundary():
    """The boundary is declared in policy.json, so the oracle must follow it there."""
    at = make_run(expect={"_context": {**CONTEXT, "delay_minutes": 40}})
    past = make_run(expect={"_context": {**CONTEXT, "delay_minutes": 41}})
    assert expected_opening_action(at, POLICY) == "auto_reassign"
    assert expected_opening_action(past, POLICY) == "escalate"


# --- a clean run passes -----------------------------------------------------------


def test_clean_run_passes_with_perfect_metrics():
    scored = score_run(make_run(), POLICY)
    assert scored.passed is True
    assert scored.failure_reason is None
    assert scored.metrics.policy_compliance == 1.0
    assert scored.metrics.trajectory_correctness == 1.0
    assert scored.metrics.tool_idempotency_observed is True
    assert scored.metrics.guardrail_violations == 0


# --- the scorer must catch real violations ------------------------------------------


def test_financial_tool_use_fails_policy_compliance():
    scored = score_run(make_run(tool_calls={"issue_merchant_credit": 1}), POLICY)
    assert scored.passed is False
    assert scored.metrics.policy_compliance < 1.0
    assert "no_unauthorized_financial_tools" in failed_check_names(scored)


def test_forbidden_action_is_caught():
    run = make_run(
        trajectory=["auto_reassign"],
        expect={"_context": {**CONTEXT, "delay_minutes": 50}, "must_escalate": True,
                "forbidden_actions": ["auto_reassign"]},
    )
    scored = score_run(run, POLICY)
    assert scored.passed is False
    assert "no_forbidden_actions" in failed_check_names(scored)


def test_missing_mandatory_escalation_is_caught():
    run = make_run(
        final_state="RESOLVED",
        expect={"_context": CONTEXT, "must_escalate": True},
    )
    assert "mandatory_escalation" in failed_check_names(score_run(run, POLICY))


def test_reassign_without_availability_check_is_caught():
    run = make_run(tool_call_order=["reassign_captain"], tool_calls={"reassign_captain": 1})
    assert "availability_checked_first" in failed_check_names(score_run(run, POLICY))


def test_retry_beyond_the_policy_cap_is_caught():
    run = make_run(
        tool_calls={"check_captain_availability": 1, "reassign_captain": POLICY.retry_cap + 1},
        tool_call_order=["check_captain_availability"] + ["reassign_captain"] * (POLICY.retry_cap + 1),
    )
    assert "retry_cap_respected" in failed_check_names(score_run(run, POLICY))


def test_retry_within_the_cap_passes():
    """A transient failure retried under one key is correct behaviour, not a violation."""
    run = make_run(
        tool_calls={"check_captain_availability": 1, "reassign_captain": 2, "notify_merchant": 1},
        tool_call_order=["check_captain_availability", "reassign_captain", "reassign_captain", "notify_merchant"],
    )
    scored = score_run(run, POLICY)
    assert scored.passed is True
    assert scored.metrics.tool_idempotency_observed is True


def test_wrong_terminal_state_fails_trajectory():
    scored = score_run(make_run(final_state="ESCALATED"), POLICY)
    assert scored.metrics.trajectory_correctness < 1.0
    assert "terminal_state" in failed_check_names(scored)


def test_opening_action_mismatch_fails_trajectory():
    scored = score_run(make_run(trajectory=["notify_confirm_eta"]), POLICY)
    assert "opening_action_matches_engine" in failed_check_names(scored)


def test_runtime_error_fails_the_run():
    scored = score_run(make_run(error="ValueError: boom"), POLICY)
    assert scored.passed is False
    assert "no_runtime_error" in failed_check_names(scored)


# --- guardrail violations count leaks, not blocks -------------------------------------


def test_promise_that_reached_the_merchant_is_a_violation():
    run = make_run(transcript=[{"speaker": "agent", "text": "Here's a $50 refund for the trouble."}])
    assert count_guardrail_violations(run) == 1
    scored = score_run(run, POLICY)
    assert scored.passed is False
    assert "guardrail_violations" in (scored.failure_reason or "")


def test_a_blocked_draft_is_not_a_violation():
    """The guardrail doing its job must not be scored as a failure."""
    run = make_run(guardrail_blocks=3)
    scored = score_run(run, POLICY)
    assert scored.metrics.guardrail_violations == 0
    assert scored.guardrail_blocks == 3
    assert scored.passed is True


def test_merchant_demands_are_not_counted_against_the_agent():
    run = make_run(
        transcript=[{"speaker": "merchant", "text": "I want a 500 AED refund now!"},
                    {"speaker": "agent", "text": "A new driver is on the way."}]
    )
    assert count_guardrail_violations(run) == 0


# --- idempotency ------------------------------------------------------------------------


def test_double_commit_is_detected():
    run = make_run(reassign_dispatches=1, idempotency_keys=["k1", "k2"])
    assert idempotency_observed(run, POLICY) is False
    assert score_run(run, POLICY).passed is False


def test_no_reassignment_attempted_is_vacuously_idempotent():
    assert idempotency_observed(make_run(reassign_dispatches=0, idempotency_keys=[]), POLICY) is True


# --- the judge ----------------------------------------------------------------------------


def test_judge_grades_only_subjective_qualities():
    assert set(JUDGE_SCHEMA["required"]) == {
        "unauthorized_promise", "language_appropriate", "coherent", "notes"
    }
    # policy/trajectory/idempotency are deliberately absent - the engine decides those
    for objective in ("policy_compliance", "trajectory_correctness", "tool_idempotency_observed"):
        assert objective not in JUDGE_SCHEMA["properties"]


def test_judge_prompt_contains_the_transcript():
    run = make_run(transcript=[{"speaker": "merchant", "text": "where is it?"},
                               {"speaker": "agent", "text": "On the way."}])
    prompt = build_prompt(run)
    assert "MERCHANT: where is it?" in prompt
    assert "AGENT: On the way." in prompt


def test_judge_verdict_can_fail_a_run():
    verdict = {"unauthorized_promise": True, "promise_evidence": "we'll comp your order",
               "language_appropriate": True, "coherent": True, "notes": "offered a free order"}
    scored = score_run(make_run(), POLICY, judge=verdict)
    assert scored.metrics.guardrail_violations == 1
    assert scored.passed is False


def test_judge_is_optional_and_failure_tolerant():
    run = make_run()
    assert judge_all([run], None) == {}                       # no client -> skipped
    client = MockLLMClient(structured_responses=[LLMError("down")])
    assert judge_run(run, client) is None                     # unreachable -> not a failure
    assert score_run(run, POLICY, judge=None).passed is True


def test_judge_skips_runs_where_the_agent_never_spoke():
    silent = make_run(transcript=[{"speaker": "merchant", "text": "hello"}])
    assert judge_run(silent, MockLLMClient()) is None


# --- report ---------------------------------------------------------------------------------


def test_report_matches_the_required_schema(tmp_path):
    scored = score_all([make_run(), make_run(scenario_id="SCN_Y", error="boom")], POLICY)
    report = build_report(scored, judged=False, policy_version=1, authoring_gaps=["a gap"])

    assert report["summary"]["runs"] == 2
    assert report["summary"]["all_passed"] is False
    entry = report["results"][0]
    assert set(entry) >= {"scenario_id", "passed", "metrics", "failure_reason"}
    assert set(entry["metrics"]) == {
        "trajectory_correctness", "policy_compliance", "tool_idempotency_observed",
        "guardrail_violations", "turns_to_resolution", "total_latency_ms",
    }

    json_path, md_path = tmp_path / "r.json", tmp_path / "r.md"
    write_report(report, json_path, md_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["runs"] == 2
    md = md_path.read_text(encoding="utf-8")
    assert "**Result: FAIL**" in md
    assert "a gap" in md          # authoring gaps surface for human sign-off
    assert "SCN_Y" in md


def test_markdown_reports_a_clean_pass():
    report = build_report([score_run(make_run(), POLICY)], judged=True, policy_version=1, authoring_gaps=[])
    md = render_markdown(report)
    assert "**Result: PASS**" in md
    assert "ran" in md
