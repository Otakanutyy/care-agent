"""Evaluation runner — executes the agent across every scenario x persona pair.

A **scenario** owns the situation: the injected session context, the backend event script
(including events timed to land while a tool call is in flight), the tool outcomes for that
run, and an ``expect`` block stating the scenario's intent. A **persona** owns the merchant's
behaviour. The matrix of the two is what the assessment asks for.

The runner only *collects* — transcript, action trajectory, tool calls, tickets, guardrail
blocks, turn count, latency. Scoring lives in ``metrics.py`` (Step 11), so the thing that
produces the evidence is separate from the thing that judges it.

Runs are reproducible by default: with no LLM client the agent uses the offline stand-in and
personas walk their scripted ladder, so the same scenario yields the same trajectory every time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from care_agent.agent import CareAgent
from care_agent.domain.models import Event, EventType, SessionState
from care_agent.eval.simulator import Persona, build_simulator, load_personas
from care_agent.llm.client import LLMClient
from care_agent.llm.offline import OfflineLLMClient
from care_agent.policy.loader import PolicySnapshot, load_policy
from care_agent.tools.stubs import ToolConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "eval" / "scenarios"
POLICY_PATH = REPO_ROOT / "policy" / "policy.json"


class Scenario(BaseModel):
    """One situation the agent must handle."""

    model_config = {"extra": "forbid"}

    id: str
    title: str
    description: str
    context: dict[str, Any]
    tools: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    max_turns: int = 3
    personas: list[str] | None = None          # None = run against every persona
    expect: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Everything one scenario x persona run produced. Raw evidence, not a verdict."""

    scenario_id: str
    persona_id: str
    title: str = ""
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    trajectory: list[str] = Field(default_factory=list)
    tickets: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: dict[str, int] = Field(default_factory=dict)
    tool_call_order: list[str] = Field(default_factory=list)
    guardrail_blocks: int = 0            # drafts the guardrail stopped (a success, not a leak)
    idempotency_keys: list[str] = Field(default_factory=list)  # one per committed side effect
    reassign_dispatches: int = 0         # how many times a reassign chain was started
    final_state: str | None = None
    terminal_reason: str | None = None
    final_captain_id: str | None = None
    turns_used: int = 0
    total_latency_ms: float = 0.0
    llm_calls: int = 0
    error: str | None = None
    expect: dict[str, Any] = Field(default_factory=dict)


def load_scenario(path: str | Path) -> Scenario:
    return Scenario(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_scenarios(directory: str | Path = SCENARIO_DIR) -> list[Scenario]:
    directory = Path(directory)
    scenarios = [load_scenario(p) for p in sorted(directory.glob("*.json"))]
    if not scenarios:
        raise FileNotFoundError(f"no scenario files found in {directory}")
    return scenarios


def _build_event(spec: dict[str, Any]) -> Event:
    name = spec["event"]
    try:
        event_type = EventType(name)
    except ValueError as exc:
        valid = ", ".join(e.value for e in EventType)
        raise ValueError(f"unknown event {name!r}; expected one of: {valid}") from exc
    return Event(type=event_type, new_eta=spec.get("new_eta"), payload=spec.get("payload", {}))


def run_once(
    scenario: Scenario,
    persona: Persona,
    policy: PolicySnapshot,
    client_factory: Callable[[], LLMClient] = OfflineLLMClient,
    simulator_client: LLMClient | None = None,
) -> RunResult:
    """Run one scenario against one persona and collect the evidence.

    ``client_factory`` builds the agent's model client (offline stand-in by default).
    ``simulator_client`` drives the merchant persona; None means walk its scripted ladder.
    """
    client = client_factory()
    agent = CareAgent(policy, client, tool_config=ToolConfig(**scenario.tools))
    simulator = build_simulator(persona, simulator_client)

    result = RunResult(
        scenario_id=scenario.id,
        persona_id=persona.id,
        title=scenario.title,
        # The opening context travels with the result so the scorer can re-derive what the
        # policy engine says *should* have happened, rather than trusting a hand-written answer.
        expect={**scenario.expect, "_context": scenario.context},
    )
    started = time.perf_counter()

    try:
        agent.start(SessionState(**scenario.context))

        # Events marked for mid-flight land while the next tool chain is still running.
        for spec in scenario.events:
            if spec.get("during_tool_flight"):
                agent.inject_before_tool_result.append(_build_event(spec))

        _apply_events(agent, scenario, after_turn=0)

        for turn in range(1, scenario.max_turns + 1):
            if agent.is_terminal:
                break
            agent.send_message(simulator.next_message(agent.transcript))
            result.turns_used = turn
            _apply_events(agent, scenario, after_turn=turn)
    except Exception as exc:  # a crash is a result, not a reason to lose the whole matrix
        result.error = f"{type(exc).__name__}: {exc}"

    result.total_latency_ms = (time.perf_counter() - started) * 1000
    _collect(result, agent, client)
    return result


def _apply_events(agent: CareAgent, scenario: Scenario, after_turn: int) -> None:
    for spec in scenario.events:
        if spec.get("during_tool_flight"):
            continue  # queued separately; fires inside the tool chain
        if spec.get("after_turn", 0) != after_turn:
            continue
        if agent.is_terminal:
            return
        agent.send_event(_build_event(spec))


def _collect(result: RunResult, agent: CareAgent, client: LLMClient) -> None:
    result.transcript = [e.model_dump() for e in agent.transcript]
    result.trace = list(agent.trace)
    result.trajectory = agent.trajectory()
    result.tickets = list(agent.tickets)
    result.tool_calls = dict(agent.stubs.calls)
    result.tool_call_order = list(agent.stubs.call_log)
    result.guardrail_blocks = agent.guardrail_violations
    result.idempotency_keys = agent.idempotency.keys()
    result.reassign_dispatches = sum(1 for t in agent.trace if t.get("kind") == "tool_chain")
    result.llm_calls = len(client.calls)
    if agent.order_id is not None:
        session = agent.session
        result.final_state = session.fsm_state.value
        result.terminal_reason = session.terminal_reason
        result.final_captain_id = session.data.current_captain_id


def run_matrix(
    scenarios: Sequence[Scenario] | None = None,
    personas: dict[str, Persona] | None = None,
    policy: PolicySnapshot | None = None,
    client_factory: Callable[[], LLMClient] = OfflineLLMClient,
    simulator_client: LLMClient | None = None,
) -> list[RunResult]:
    """Run every scenario against every persona it declares."""
    scenarios = list(scenarios) if scenarios is not None else load_scenarios()
    personas = personas if personas is not None else load_personas()
    policy = policy or load_policy(POLICY_PATH)

    results: list[RunResult] = []
    for scenario in scenarios:
        wanted = scenario.personas or list(personas)
        for persona_id in wanted:
            persona = personas.get(persona_id)
            if persona is None:
                raise KeyError(f"scenario {scenario.id} references unknown persona {persona_id!r}")
            results.append(
                run_once(scenario, persona, policy, client_factory, simulator_client)
            )
    return results
