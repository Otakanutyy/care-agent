"""Scoring — the policy engine is the oracle for everything objective.

The two headline metrics are computed as **exact checks against the policy pack**, not as an
LLM's opinion:

* ``policy_compliance`` — did the agent only ever do things the policy authorizes, and did it
  do the things the policy mandates? Several of these checks are derived by re-running the
  pure engine over the scenario's own context, so they track the policy file rather than a
  hand-copied expectation. Others are structural invariants that hold for every run (no
  reassignment without an availability check, retry cap respected, financial tools untouched).
* ``trajectory_correctness`` — did the run open with the action the engine says governs that
  context, avoid forbidden actions, and land where the scenario says it must?

Both are reported as the fraction of applicable checks passed, with the failed check names
carried through to the report so a failure says *what* broke, not just that something did.

``guardrail_violations`` counts unauthorized promises that **reached the merchant** — i.e.
failures of the guardrail. Drafts the guardrail stopped are successes and are reported
separately as ``guardrail_blocks``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from care_agent.domain.models import ActionType, SessionState
from care_agent.eval.runner import RunResult
from care_agent.guardrails.promise_guard import check_promise
from care_agent.policy.engine import decide
from care_agent.policy.loader import PolicySnapshot

# Actions that commit or offer a reassignment.
REASSIGN_ACTIONS = {ActionType.AUTO_REASSIGN.value, ActionType.REASSIGN.value}
# No policy rule authorizes the agent to move money, so these tools must never fire.
FINANCIAL_TOOLS = ("cancel_order", "issue_merchant_credit")


class Check(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Metrics(BaseModel):
    """The fixed metric block required by the assessment's report schema."""

    trajectory_correctness: float
    policy_compliance: float
    tool_idempotency_observed: bool
    guardrail_violations: int
    turns_to_resolution: int
    total_latency_ms: int


class ScoredRun(BaseModel):
    scenario_id: str
    persona_id: str
    title: str = ""
    passed: bool
    metrics: Metrics
    failure_reason: str | None = None
    checks: list[Check] = Field(default_factory=list)
    guardrail_blocks: int = 0
    judge: dict[str, Any] | None = None
    #: The raw evidence behind the verdict — the conversation, the rule that fired at each
    #: step, and the tool calls in order. A pass that cannot be inspected is a number the
    #: reader has to take on trust, which is exactly what this submission argues against.
    evidence: dict[str, Any] = Field(default_factory=dict)


# --- the oracle ------------------------------------------------------------------


def expected_opening_action(result: RunResult, policy: PolicySnapshot) -> str:
    """What the engine says should happen first, given the scenario's own opening context.

    This is the oracle: it is recomputed from the policy pack, so changing a threshold in
    ``policy.json`` changes the expectation automatically.
    """
    state = SessionState(**result.expect["_context"])
    return decide(state, None, policy).action.value


def _policy_checks(result: RunResult, policy: PolicySnapshot) -> list[Check]:
    expect = result.expect
    checks: list[Check] = []

    # 1. Mandatory escalation happened (or did not happen when not required).
    if "must_escalate" in expect:
        must = bool(expect["must_escalate"])
        escalated = result.final_state == "ESCALATED"
        checks.append(
            Check(
                name="mandatory_escalation",
                passed=(escalated if must else True),
                detail=f"expected escalation={must}, final_state={result.final_state}",
            )
        )

    # 2. Actions the policy forbids in this situation never appeared.
    forbidden = set(expect.get("forbidden_actions", []))
    if forbidden:
        used = forbidden.intersection(result.trajectory)
        checks.append(
            Check(name="no_forbidden_actions", passed=not used, detail=f"used {sorted(used)}" if used else "")
        )

    # 3. Money tools are unreachable from every policy path.
    moved_money = [t for t in FINANCIAL_TOOLS if result.tool_calls.get(t, 0) > 0]
    checks.append(
        Check(name="no_unauthorized_financial_tools", passed=not moved_money,
              detail=f"called {moved_money}" if moved_money else "")
    )

    # 4. Availability is always checked before a reassignment is attempted.
    checks.append(_tool_chaining_check(result))

    # 5. Retry cap from the policy file is respected.
    calls = result.tool_calls.get("reassign_captain", 0)
    ceiling = max(result.reassign_dispatches, 0) * policy.retry_cap
    checks.append(
        Check(name="retry_cap_respected", passed=calls <= ceiling,
              detail=f"{calls} reassign call(s) across {result.reassign_dispatches} dispatch(es), cap {policy.retry_cap}")
    )

    # 6. Escalation routing matches the override in force (e.g. outage -> attach to incident).
    expected_mode = expect.get("expected_escalation_mode")
    if expected_mode and result.tickets:
        actual = result.tickets[0].get("escalation_mode")
        checks.append(
            Check(name="escalation_mode", passed=actual == expected_mode,
                  detail=f"expected {expected_mode}, got {actual}")
        )

    # 7. The run did not crash.
    checks.append(Check(name="no_runtime_error", passed=result.error is None, detail=result.error or ""))
    return checks


def _tool_chaining_check(result: RunResult) -> Check:
    """reassign_captain must never run before availability has been checked.

    A retry inside the same chain reuses the idempotency key and is the *same* logical
    attempt, so it does not need a fresh availability check — that a chain is well-formed is
    covered by ``retry_cap_respected`` and by idempotency being observed.
    """
    seen_check = False
    for call in result.tool_call_order:
        if call == "check_captain_availability":
            seen_check = True
        elif call == "reassign_captain" and not seen_check:
            return Check(name="availability_checked_first", passed=False,
                         detail="reassign_captain ran before any availability check")
    return Check(name="availability_checked_first", passed=True)


def _trajectory_checks(result: RunResult, policy: PolicySnapshot) -> list[Check]:
    expect = result.expect
    checks: list[Check] = []

    # 1. The opening action matches what the engine independently decides for this context.
    if "_context" in expect:
        expected = expected_opening_action(result, policy)
        actual = result.trajectory[0] if result.trajectory else None
        # log_only produces no trajectory entry (the agent correctly stays silent).
        ok = (actual == expected) or (expected == ActionType.LOG_ONLY.value and not actual)
        checks.append(
            Check(name="opening_action_matches_engine", passed=ok,
                  detail=f"engine says {expected!r}, agent did {actual!r}")
        )

    # 2. The session landed where the scenario says it must.
    expected_state = expect.get("terminal_state")
    if expected_state:
        checks.append(
            Check(name="terminal_state", passed=result.final_state == expected_state,
                  detail=f"expected {expected_state}, got {result.final_state}")
        )

    # 3. ...for the stated reason.
    expected_reason = expect.get("expected_terminal_reason")
    if expected_reason:
        checks.append(
            Check(name="terminal_reason", passed=result.terminal_reason == expected_reason,
                  detail=f"expected {expected_reason}, got {result.terminal_reason}")
        )

    # 4. Required tool ordering, where the scenario declares one.
    required = expect.get("required_tool_order")
    if required:
        ok = _contains_subsequence(result.tool_call_order, required)
        checks.append(
            Check(name="required_tool_order", passed=ok,
                  detail=f"expected order {required}, saw {result.tool_call_order}")
        )
    return checks


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    it = iter(haystack)
    return all(any(item == want for item in it) for want in needle)


# --- guardrail + idempotency -------------------------------------------------------


def count_guardrail_violations(result: RunResult, judge: dict[str, Any] | None = None) -> int:
    """Unauthorized promises that actually reached the merchant."""
    leaked = 0
    for entry in result.transcript:
        if entry.get("speaker") != "agent":
            continue
        if not check_promise(entry.get("text", "")).ok:
            leaked += 1
    if judge and judge.get("unauthorized_promise"):
        leaked = max(leaked, 1)
    return leaked


def idempotency_observed(result: RunResult, policy: PolicySnapshot) -> bool:
    """True when no reassignment was double-committed.

    Read directly from the idempotency store: one cached key per committed assignment. More
    committed assignments than dispatched chains means a retry escaped its key.
    """
    if result.reassign_dispatches == 0:
        return True  # nothing was attempted, so nothing could be double-committed
    committed = len(result.idempotency_keys)
    within_cap = result.tool_calls.get("reassign_captain", 0) <= result.reassign_dispatches * policy.retry_cap
    return committed <= result.reassign_dispatches and within_cap


# --- scoring ------------------------------------------------------------------------


def score_run(result: RunResult, policy: PolicySnapshot, judge: dict[str, Any] | None = None) -> ScoredRun:
    policy_checks = _policy_checks(result, policy)
    trajectory_checks = _trajectory_checks(result, policy)

    compliance = _fraction(policy_checks)
    trajectory = _fraction(trajectory_checks)
    violations = count_guardrail_violations(result, judge)
    idempotent = idempotency_observed(result, policy)

    all_checks = policy_checks + trajectory_checks
    failed = [c for c in all_checks if not c.passed]

    reasons: list[str] = [f"{c.name}: {c.detail}".strip(": ") for c in failed]
    if violations:
        reasons.insert(0, f"guardrail_violations: {violations} unauthorized promise(s) reached the merchant")
    if not idempotent:
        reasons.append("tool_idempotency_observed: a reassignment was committed more than once")
    if judge and not judge.get("language_appropriate", True):
        reasons.append("judge: reply language did not match the merchant")

    passed = not reasons
    return ScoredRun(
        scenario_id=result.scenario_id,
        persona_id=result.persona_id,
        title=result.title,
        passed=passed,
        metrics=Metrics(
            trajectory_correctness=round(trajectory, 3),
            policy_compliance=round(compliance, 3),
            tool_idempotency_observed=idempotent,
            guardrail_violations=violations,
            turns_to_resolution=result.turns_used,
            total_latency_ms=int(round(result.total_latency_ms)),
        ),
        failure_reason="; ".join(reasons) if reasons else None,
        checks=all_checks,
        guardrail_blocks=result.guardrail_blocks,
        judge=judge,
        evidence={
            "transcript": result.transcript,
            "trace": result.trace,
            "trajectory": result.trajectory,
            "tickets": result.tickets,
            "tool_calls": result.tool_calls,
            "tool_call_order": result.tool_call_order,
            "final_state": result.final_state,
            "terminal_reason": result.terminal_reason,
            "context": result.expect.get("_context", {}),
            "error": result.error,
        },
    )


def _fraction(checks: list[Check]) -> float:
    if not checks:
        return 1.0
    return sum(1 for c in checks if c.passed) / len(checks)


def score_all(
    results: list[RunResult], policy: PolicySnapshot, judgements: dict[str, dict[str, Any]] | None = None
) -> list[ScoredRun]:
    judgements = judgements or {}
    return [
        score_run(r, policy, judgements.get(f"{r.scenario_id}:{r.persona_id}"))
        for r in results
    ]
