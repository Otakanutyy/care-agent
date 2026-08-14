"""Report generation — the pass/fail execution report.

``report.json`` carries one entry per run in the schema the assessment specifies
(``scenario_id`` / ``passed`` / ``metrics`` / ``failure_reason``), plus the persona that drove
the run and the individual checks behind each verdict, so a failure can be read without
re-running anything. ``report.md`` is the human summary.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care_agent.eval.metrics import ScoredRun


def build_report(
    scored: list[ScoredRun],
    *,
    judged: bool,
    policy_version: int,
    authoring_gaps: list[str],
    mode: str = "offline",
) -> dict[str, Any]:
    passed = sum(1 for s in scored if s.passed)
    scenarios = sorted({s.scenario_id for s in scored})
    scenario_pass = {
        sid: all(s.passed for s in scored if s.scenario_id == sid) for sid in scenarios
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy_version": policy_version,
        # Which run produced this. Policy decisions are identical either way; the transcripts
        # differ, since offline phrases replies from templates. Stated so a reader reproducing
        # it offline is not surprised by different wording.
        "mode": mode,
        "subjective_judge_ran": judged,
        "summary": {
            "runs": len(scored),
            "passed": passed,
            "failed": len(scored) - passed,
            "scenarios": len(scenarios),
            "scenarios_passed": sum(1 for ok in scenario_pass.values() if ok),
            "guardrail_violations": sum(s.metrics.guardrail_violations for s in scored),
            "guardrail_blocks": sum(s.guardrail_blocks for s in scored),
            "all_passed": passed == len(scored),
        },
        "policy_authoring_gaps": authoring_gaps,
        "results": [
            {
                "scenario_id": s.scenario_id,
                "persona_id": s.persona_id,
                "title": s.title,
                "passed": s.passed,
                "metrics": s.metrics.model_dump(),
                "failure_reason": s.failure_reason,
                "guardrail_blocks": s.guardrail_blocks,
                "checks": [c.model_dump() for c in s.checks],
                "judge": s.judge,
                # The conversation and decision trail behind the verdict, so a reader can audit
                # a pass rather than trust it. Additive: the assessment's required fields above
                # keep their exact shape.
                "evidence": s.evidence,
            }
            for s in scored
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    verdict = "PASS" if s["all_passed"] else "FAIL"
    lines = [
        "# Care Agent - Evaluation Report",
        "",
        f"**Result: {verdict}** - {s['passed']}/{s['runs']} runs passed "
        f"({s['scenarios_passed']}/{s['scenarios']} scenarios clean).",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Policy version: {report['policy_version']}",
        f"- Unauthorized promises that reached a merchant: **{s['guardrail_violations']}**",
        f"- Unauthorized promises blocked before sending: {s['guardrail_blocks']}",
        f"- Subjective (LLM judge) pass: {'ran' if report['subjective_judge_ran'] else 'skipped - no API key'}",
        "",
        "Objective metrics are decided by the policy engine and structural invariants, not by a",
        "model. The judge only grades what has no exact answer.",
        "",
        "## Runs",
        "",
        "| Scenario | Persona | Pass | Trajectory | Policy | Idempotent | Violations | Turns | ms |",
        "|---|---|:--:|--:|--:|:--:|--:|--:|--:|",
    ]
    for r in report["results"]:
        m = r["metrics"]
        lines.append(
            f"| {r['scenario_id']} | {r['persona_id']} | {'PASS' if r['passed'] else 'FAIL'} | "
            f"{m['trajectory_correctness']:.2f} | {m['policy_compliance']:.2f} | "
            f"{'yes' if m['tool_idempotency_observed'] else 'NO'} | {m['guardrail_violations']} | "
            f"{m['turns_to_resolution']} | {m['total_latency_ms']} |"
        )

    failures = [r for r in report["results"] if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- **{r['scenario_id']} / {r['persona_id']}** - {r['failure_reason']}")

    lines += ["", "## Scenario coverage", ""]
    titles: dict[str, str] = {}
    for r in report["results"]:
        titles.setdefault(r["scenario_id"], r["title"])
    counts = Counter(r["scenario_id"] for r in report["results"])
    for sid in sorted(titles):
        ok = all(r["passed"] for r in report["results"] if r["scenario_id"] == sid)
        lines.append(f"- `{sid}` {titles[sid]} - {counts[sid]} persona run(s), {'all pass' if ok else 'FAILURES'}")

    if report["policy_authoring_gaps"]:
        lines += [
            "",
            "## Policy authoring gaps",
            "",
            "Flagged for human sign-off - the source policy does not specify these, so the",
            "behaviour below is an adopted decision rather than a stated rule:",
            "",
        ]
        lines += [f"- {gap}" for gap in report["policy_authoring_gaps"]]

    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
