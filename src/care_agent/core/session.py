"""FSM state, the per-session record the reducer threads, and the typed effects it emits.

Effects are *descriptions* of side effects, not the side effects themselves — the reducer
stays pure. Later steps (tool broker, guardrail, generator) execute them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from care_agent.domain.models import ActionEnvelope, EscalationMode, SessionState


class FsmState(str, Enum):
    INIT = "INIT"
    MONITORING = "MONITORING"                        # R1 / passive notice — no reply expected
    AWAITING_MERCHANT_REPLY = "AWAITING_MERCHANT_REPLY"
    AWAITING_TOOL_RESULT = "AWAITING_TOOL_RESULT"
    ESCALATED = "ESCALATED"                          # terminal, inert
    RESOLVED = "RESOLVED"                            # terminal, inert


TERMINAL_STATES = (FsmState.ESCALATED, FsmState.RESOLVED)


class PendingTool(BaseModel):
    """The in-flight tool chain the session is awaiting (AWAITING_TOOL_RESULT)."""

    purpose: str
    tool_sequence: list[str]
    idempotency_key: str
    dispatch_epoch: int


# --- Effects ------------------------------------------------------------------


class LogEffect(BaseModel):
    kind: Literal["log"] = "log"
    message: str


class SendMessageEffect(BaseModel):
    """Ask the response generator to phrase this already-decided envelope for the merchant."""

    kind: Literal["send_message"] = "send_message"
    envelope: ActionEnvelope


class CallToolChainEffect(BaseModel):
    """Ask the tool broker to run a chain (e.g. check_captain_availability -> reassign_captain)
    with idempotency; the broker feeds a TOOL_RESULT event back into the mailbox."""

    kind: Literal["call_tool_chain"] = "call_tool_chain"
    tool_sequence: list[str]
    idempotency_key: str
    dispatch_epoch: int
    purpose: str
    source_reason: str | None = None


class EscalateEffect(BaseModel):
    kind: Literal["escalate"] = "escalate"
    reason: str
    escalation_mode: EscalationMode | None = None
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    guardrail_halt: bool = False  # True when a guardrail forced the stop (SAFE_HALT)


class ResolveEffect(BaseModel):
    kind: Literal["resolve"] = "resolve"
    reason: str


class ReconcileEffect(BaseModel):
    """A tool result arrived after the session had moved on; log/compensate, don't apply."""

    kind: Literal["reconcile"] = "reconcile"
    detail: str


Effect = Union[
    LogEffect,
    SendMessageEffect,
    CallToolChainEffect,
    EscalateEffect,
    ResolveEffect,
    ReconcileEffect,
]


# --- Session ------------------------------------------------------------------


class Session(BaseModel):
    """Everything the reducer threads from one event to the next for a single order."""

    fsm_state: FsmState = FsmState.INIT
    data: SessionState
    epoch: int = 0                       # bumps on each state-mutating event (in-flight fence)
    pending_tool: PendingTool | None = None
    last_action: ActionEnvelope | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    terminal_reason: str | None = None
    last_off_policy_signature: tuple[bool, ...] | None = None  # loop-guard: last off-policy intent
