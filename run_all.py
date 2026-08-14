#!/usr/bin/env python
"""Run the full evaluation suite and write the pass/fail execution report.

    python run_all.py              # reproducible: offline stand-ins, deterministic personas
    python run_all.py --live       # real Claude API for the agent, personas, and judge

Writes ``report.json`` (per-run, in the required schema) and ``report.md`` (human summary).
Exits non-zero if any run failed, so it can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):  # tolerate non-ASCII transcripts on legacy consoles
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from care_agent.eval.judge import judge_all  # noqa: E402
from care_agent.eval.metrics import score_all  # noqa: E402
from care_agent.eval.report import build_report, write_report  # noqa: E402
from care_agent.eval.runner import run_matrix  # noqa: E402
from care_agent.llm.client import AnthropicClient, LLMClient  # noqa: E402
from care_agent.llm.offline import OfflineLLMClient  # noqa: E402
from care_agent.policy.loader import load_policy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Care agent evaluation suite.")
    parser.add_argument("--live", action="store_true",
                        help="use the real Claude API for the agent, personas, and judge")
    parser.add_argument("--json", default="report.json", help="path for the JSON report")
    parser.add_argument("--md", default="report.md", help="path for the Markdown report")
    args = parser.parse_args(argv)

    policy = load_policy(REPO_ROOT / "policy" / "policy.json")

    if args.live:
        print("[live] agent, personas, and judge all use the Claude API")
        client_factory = AnthropicClient
        simulator_client: LLMClient | None = AnthropicClient()
        judge_client: LLMClient | None = AnthropicClient()
    else:
        print("[offline] deterministic stand-ins; runs are reproducible and need no API key")
        client_factory = OfflineLLMClient
        simulator_client = None
        judge_client = None

    print("Running scenario x persona matrix...")
    results = run_matrix(
        policy=policy, client_factory=client_factory, simulator_client=simulator_client
    )
    print(f"  {len(results)} runs complete")

    judgements = judge_all(results, judge_client)
    if judge_client is not None:
        print(f"  {len(judgements)} runs graded by the subjective judge")

    scored = score_all(results, policy, judgements)
    report = build_report(
        scored,
        judged=bool(judgements),
        policy_version=policy.version,
        authoring_gaps=list(policy.authoring_gaps),
        mode="live" if args.live else "offline",
    )
    write_report(report, REPO_ROOT / args.json, REPO_ROOT / args.md)

    s = report["summary"]
    print("-" * 68)
    for r in report["results"]:
        if not r["passed"]:
            print(f"  FAIL {r['scenario_id']}/{r['persona_id']}: {r['failure_reason']}")
    print(f"RESULT: {'PASS' if s['all_passed'] else 'FAIL'} "
          f"({s['passed']}/{s['runs']} runs, {s['scenarios_passed']}/{s['scenarios']} scenarios)")
    print(f"  unauthorized promises reaching a merchant : {s['guardrail_violations']}")
    print(f"  unauthorized promises blocked            : {s['guardrail_blocks']}")
    print(f"  wrote {args.json} and {args.md}")
    print("-" * 68)
    return 0 if s["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
