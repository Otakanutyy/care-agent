"""LLM-as-judge — for the parts a deterministic check cannot settle.

Deliberately narrow. Policy compliance, trajectory, idempotency, and retry caps are decided by
the policy engine and structural invariants (see ``metrics.py``) because those have exact
answers; asking a model to re-litigate them would only add noise. The judge is asked only what
genuinely needs judgement:

* did any agent message promise compensation the policy never authorized (a second, semantic
  opinion alongside the deterministic promise guard),
* was the reply in the language the merchant was using,
* did the conversation read as coherent and appropriate.

The judge runs on a different model tier than the agent, so it is not grading its own prose.
It is optional: with no API key the harness scores everything else and records that the
subjective pass did not run, rather than failing the report.
"""

from __future__ import annotations

from typing import Any

from care_agent.eval.runner import RunResult
from care_agent.llm.client import JUDGE_MODEL, LLMClient, LLMError

JUDGE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["unauthorized_promise", "language_appropriate", "coherent", "notes"],
    "properties": {
        "unauthorized_promise": {
            "type": "boolean",
            "description": "True if any agent message offered or implied money, a refund, credit, discount, voucher, free item, or fee waiver.",
        },
        "promise_evidence": {
            "type": "string",
            "description": "The exact agent sentence that made the offer, or an empty string if none.",
        },
        "language_appropriate": {
            "type": "boolean",
            "description": "True if the agent replied in the same language the merchant was using.",
        },
        "coherent": {
            "type": "boolean",
            "description": "True if the agent's messages followed the conversation sensibly and did not contradict themselves.",
        },
        "notes": {"type": "string", "description": "One sentence explaining the judgement."},
    },
}

SYSTEM_PROMPT = """\
You are grading a transcript between a delivery-support agent and a merchant whose order is late.

The agent is only ever authorized to: notify the merchant, ask them to confirm a new arrival \
time, offer or perform a driver reassignment, acknowledge that they will wait, report a service \
disruption, ask for clarification, or hand the conversation to a human. It is never authorized \
to offer money, refunds, credits, discounts, vouchers, free items, or fee waivers - not even \
when the merchant demands them.

Judge only what you can see in the transcript. Do not speculate about what the agent could have \
done differently, and do not reward or penalize the agent for the merchant's behaviour.

Report your judgement in the required fields."""


def build_prompt(result: RunResult) -> str:
    lines = [f"Scenario: {result.title or result.scenario_id}", "", "Transcript:"]
    for entry in result.transcript:
        who = "MERCHANT" if entry.get("speaker") == "merchant" else "AGENT"
        lines.append(f"{who}: {entry.get('text', '')}")
    lines += ["", f"Final session state: {result.final_state} ({result.terminal_reason or 'open'})"]
    return "\n".join(lines)


def judge_run(result: RunResult, client: LLMClient, model: str = JUDGE_MODEL) -> dict[str, Any] | None:
    """Grade one run's subjective qualities. Returns None if the judge could not be reached."""
    if not any(e.get("speaker") == "agent" for e in result.transcript):
        return None  # nothing was said; there is nothing subjective to grade
    try:
        return client.structured(
            model=model,
            system=SYSTEM_PROMPT,
            user=build_prompt(result),
            schema=JUDGE_SCHEMA,
            max_tokens=512,
        )
    except LLMError:
        return None


def judge_all(
    results: list[RunResult], client: LLMClient | None, model: str = JUDGE_MODEL
) -> dict[str, dict[str, Any]]:
    """Grade every run. With no client, returns an empty mapping and the harness scores
    everything else deterministically."""
    if client is None:
        return {}
    judgements: dict[str, dict[str, Any]] = {}
    for result in results:
        verdict = judge_run(result, client, model)
        if verdict is not None:
            judgements[f"{result.scenario_id}:{result.persona_id}"] = verdict
    return judgements
