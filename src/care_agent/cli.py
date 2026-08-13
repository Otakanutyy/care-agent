#!/usr/bin/env python
"""Run one Care agent session from a scripted scenario and print what happened.

By default it runs fully offline (no API key needed) using a deterministic stand-in for the
model, so the state machine, policy decisions, tools, and guardrails can be watched end to
end. Pass ``--live`` to use the real Claude API instead.

    python -m care_agent.cli                        # built-in demo, offline
    python -m care_agent.cli --scenario s.json      # a scripted scenario
    python -m care_agent.cli --live                 # real Claude API (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:  # allow `python src/care_agent/cli.py`
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Never crash on a non-ASCII reply under a legacy console codepage (Arabic replies on cp1251).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from care_agent.agent import CareAgent  # noqa: E402
from care_agent.domain.models import Event, EventType, SessionState  # noqa: E402
from care_agent.llm.client import LLMClient  # noqa: E402
from care_agent.llm.offline import OfflineLLMClient  # noqa: E402
from care_agent.policy.loader import load_policy  # noqa: E402
from care_agent.tools.stubs import ToolConfig  # noqa: E402

DEMO_SCENARIO: dict[str, Any] = {
    "id": "DEMO_01",
    "description": "Silver merchant, 30-minute delay (R4): agent offers a choice, merchant accepts a new driver.",
    "context": {
        "order_id": "order-8842",
        "merchant_name": "Al Safadi Restaurant",
        "merchant_tier": "Silver",
        "delay_minutes": 30,
        "current_captain_id": "captain-114",
        "active_system_overrides": [],
    },
    "tools": {"available": True, "new_captain_id": "captain-207", "estimated_eta_minutes": 11},
    "script": [{"type": "message", "text": "this is taking forever, can you send another driver?"}],
}


def build_event(step: dict[str, Any]) -> Event:
    """Turn a scenario step into an Event."""
    name = step["event"]
    try:
        event_type = EventType(name)
    except ValueError as exc:
        valid = ", ".join(e.value for e in EventType)
        raise SystemExit(f"unknown event {name!r}; expected one of: {valid}") from exc
    return Event(
        type=event_type,
        new_eta=step.get("new_eta"),
        payload=step.get("payload", {}),
    )


def run_scenario(scenario: dict[str, Any], client: LLMClient, policy=None) -> CareAgent:
    """Run a scripted scenario to completion and return the finished agent."""
    policy = policy or load_policy(REPO_ROOT / "policy" / "policy.json")
    agent = CareAgent(policy, client, tool_config=ToolConfig(**scenario.get("tools", {})))
    agent.start(SessionState(**scenario["context"]))

    for step in scenario.get("script", []):
        if agent.is_terminal:
            break  # terminal sessions are inert; stop feeding the script
        kind = step.get("type")
        if kind == "message":
            agent.send_message(step["text"])
        elif kind == "event":
            agent.send_event(build_event(step))
        else:
            raise SystemExit(f"unknown step type {kind!r}; expected 'message' or 'event'")
    return agent


def report(agent: CareAgent, scenario: dict[str, Any]) -> None:
    line = "-" * 72
    print(line)
    print(f"Scenario: {scenario.get('id', '?')} | {scenario.get('description', '')}")
    ctx = scenario["context"]
    print(f"Order {ctx['order_id']} | tier {ctx['merchant_tier']} | delay {ctx['delay_minutes']} min"
          f" | overrides {ctx.get('active_system_overrides') or 'none'}")
    print(line)

    print("CONVERSATION")
    for entry in agent.transcript:
        who = "merchant" if entry.speaker == "merchant" else "agent   "
        tags = []
        if entry.rule_id:
            tags.append(entry.rule_id)
        if entry.action:
            tags.append(entry.action)
        if entry.blocked:
            tags.append("GUARDRAIL-BLOCKED")
        if entry.used_fallback:
            tags.append("fallback")
        suffix = f"   [{' / '.join(tags)}]" if tags else ""
        print(f"  {who} | {entry.text}{suffix}")

    print()
    print("TRAJECTORY")
    for step in agent.trace:
        detail = step.get("detail")
        detail = f"  <- {detail}" if detail else ""
        print(f"  {step['fsm_state']:<24} {step['action']}{detail}")

    print()
    print("OUTCOME")
    session = agent.session
    print(f"  final state        : {session.fsm_state.value}")
    print(f"  terminal reason    : {session.terminal_reason or '(still open)'}")
    print(f"  tool calls         : {dict(agent.stubs.calls) or 'none'}")
    print(f"  guardrail blocks   : {agent.guardrail_violations}")
    for ticket in agent.tickets:
        halt = " (guardrail halt)" if ticket["guardrail_halt"] else ""
        print(f"  escalation ticket  : {ticket['ticket_id']} | {ticket['reason']}"
              f" [{ticket['escalation_mode']}]{halt}")
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Care agent session.")
    parser.add_argument("--scenario", help="path to a scenario JSON file (default: built-in demo)")
    parser.add_argument("--live", action="store_true", help="use the real Claude API instead of the offline stand-in")
    args = parser.parse_args(argv)

    if args.scenario:
        path = Path(args.scenario)
        if not path.exists():
            print(f"[ERROR] scenario not found: {path}")
            return 1
        scenario = json.loads(path.read_text(encoding="utf-8"))
    else:
        scenario = DEMO_SCENARIO

    if args.live:
        from care_agent.llm.client import AnthropicClient

        client: LLMClient = AnthropicClient()
        print("[live] using the Claude API")
    else:
        client = OfflineLLMClient()
        print("[offline] deterministic stand-in for the model; replies use pre-approved templates")

    report(run_scenario(scenario, client), scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
