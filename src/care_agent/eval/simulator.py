"""Adversarial merchant simulator — a second LLM playing a difficult merchant.

Each persona (``eval/personas/*.json``) carries two things:

* ``system_prompt`` — used by :class:`MerchantSimulator` to drive a real model, so test runs
  face genuinely unscripted pressure (the point of an adversarial harness).
* ``scripted_turns`` — a fixed ladder of escalating pressure used by
  :class:`ScriptedMerchantSimulator`. This is what makes a run **reproducible**: same
  scenario, same inputs, byte-identical trajectory. It is also the fallback when a live
  simulator call fails, so one flaky API call cannot abort an eval run.

The simulator only ever produces merchant *text*. It has no access to the agent's state,
policy, or tools — it pressures the agent exactly the way a real merchant could.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field

from care_agent.llm.client import SIMULATOR_MODEL, LLMClient, LLMError

PERSONA_DIR = Path(__file__).resolve().parents[3] / "eval" / "personas"

# Framing appended to every persona prompt so the model returns a usable chat line.
OUTPUT_CONTRACT = (
    "\n\nRemember: reply with the merchant's next chat message only — no narration, no labels, "
    "no quotation marks."
)


class Persona(BaseModel):
    """One adversarial merchant suite."""

    model_config = {"extra": "forbid"}

    id: str
    name: str
    description: str
    tactics: list[str] = Field(default_factory=list)
    max_turns: int = 6
    system_prompt: str
    opening: str | None = None
    scripted_turns: list[str] = Field(default_factory=list)


def load_persona(path: str | Path) -> Persona:
    return Persona(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_personas(directory: str | Path = PERSONA_DIR) -> dict[str, Persona]:
    """Load every persona in a directory, keyed by id."""
    directory = Path(directory)
    personas: dict[str, Persona] = {}
    for path in sorted(directory.glob("*.json")):
        persona = load_persona(path)
        personas[persona.id] = persona
    if not personas:
        raise FileNotFoundError(f"no persona files found in {directory}")
    return personas


# --- history rendering ---------------------------------------------------------


def _speaker_and_text(entry: Any) -> tuple[str, str]:
    """Accept TranscriptEntry objects or plain dicts, so eval isn't coupled to agent internals."""
    if isinstance(entry, dict):
        return entry.get("speaker", "agent"), entry.get("text", "")
    return getattr(entry, "speaker", "agent"), getattr(entry, "text", "")


def render_history(transcript: Sequence[Any]) -> str:
    """Format the conversation so far from the merchant's point of view."""
    if not transcript:
        return "(no messages yet)"
    lines = []
    for entry in transcript:
        speaker, text = _speaker_and_text(entry)
        who = "You (merchant)" if speaker == "merchant" else "Support agent"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


# --- simulators -----------------------------------------------------------------


class MerchantSimulatorProtocol(Protocol):
    persona: Persona

    def next_message(self, transcript: Sequence[Any]) -> str: ...


class ScriptedMerchantSimulator:
    """Deterministic: walks the persona's scripted ladder, one rung per turn.

    Reproducible by construction — the eval runner uses this when a run must be replayable.
    """

    def __init__(self, persona: Persona) -> None:
        self.persona = persona
        self.turn = 0

    def next_message(self, transcript: Sequence[Any] | None = None) -> str:
        turns = self.persona.scripted_turns
        if not turns:
            raise ValueError(f"persona {self.persona.id!r} has no scripted_turns")
        message = turns[self.turn] if self.turn < len(turns) else turns[-1]
        self.turn += 1
        return message


class MerchantSimulator:
    """LLM-driven: a second model plays the merchant, unscripted.

    Falls back to the persona's scripted ladder if the call fails, so a transient API error
    degrades the run's realism rather than aborting the eval.
    """

    def __init__(self, persona: Persona, client: LLMClient, model: str = SIMULATOR_MODEL) -> None:
        self.persona = persona
        self.client = client
        self.model = model
        self.turn = 0
        self._scripted = ScriptedMerchantSimulator(persona)

    def next_message(self, transcript: Sequence[Any] | None = None) -> str:
        transcript = transcript or []
        # Turn 1 may use the persona's fixed opening so every run starts from the same place.
        if self.turn == 0 and self.persona.opening:
            self.turn += 1
            self._scripted.turn = 1
            return self.persona.opening

        user = (
            f"Conversation so far:\n{render_history(transcript)}\n\n"
            "Write your next message as the merchant."
        )
        try:
            reply = self.client.text(
                model=self.model,
                system=self.persona.system_prompt + OUTPUT_CONTRACT,
                user=user,
                max_tokens=300,
            )
        except LLMError:
            return self._scripted.next_message(transcript)

        reply = (reply or "").strip().strip('"')
        if not reply:
            return self._scripted.next_message(transcript)

        self.turn += 1
        self._scripted.turn = self.turn
        return reply


def build_simulator(
    persona: Persona, client: LLMClient | None = None, model: str = SIMULATOR_MODEL
) -> MerchantSimulatorProtocol:
    """LLM-driven when a client is supplied, deterministic scripted otherwise."""
    if client is None:
        return ScriptedMerchantSimulator(persona)
    return MerchantSimulator(persona, client, model)
