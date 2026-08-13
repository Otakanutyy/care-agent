"""The Care agent — everything wired together for one session.

This is the only place the pieces meet:

    merchant text ──▶ classifier ──▶ Event ──▶ mailbox ──▶ reducer ──▶ policy engine
                                                                            │
                                          effects ◀─────────────────────────┘
                                             │
              ┌──────────────────────────────┼───────────────────────────────┐
              ▼                              ▼                               ▼
      generator + guardrail            tool broker                    escalation
        (phrasing only)          (chaining, idempotency)         (ticket + handoff)

The agent owns no decisions. It classifies inbound text, hands events to the mailbox, and
executes the effects the reducer returns — nothing more. Effects that produce new events (a
tool chain returning a result) are fed back through the same mailbox, so the single-writer
ordering guarantee holds for the whole cascade.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from care_agent.core.mailbox import Mailbox
from care_agent.core.reducer import Runtime
from care_agent.core.session import (
    CallToolChainEffect,
    Effect,
    EscalateEffect,
    LogEffect,
    ReconcileEffect,
    ResolveEffect,
    Session,
    SendMessageEffect,
)
from care_agent.core.stores import InMemoryEventLog, InMemoryIdempotencyStore, InMemoryStateStore
from care_agent.domain.models import ActionEnvelope, ActionType, Event, EventType, SessionState
from care_agent.llm.classifier import CLASSIFIER_MODEL, ClassifierError, classify
from care_agent.llm.client import GENERATOR_MODEL, LLMClient
from care_agent.llm.generator import generate
from care_agent.llm.templates import has_message
from care_agent.policy.loader import PolicySnapshot
from care_agent.tools.broker import ToolBroker
from care_agent.tools.stubs import ToolConfig, ToolStubs


class TranscriptEntry(BaseModel):
    """One line of the merchant-visible conversation."""

    speaker: str            # "merchant" | "agent"
    text: str
    language: str = "en"
    action: str | None = None       # the policy action this message realizes
    rule_id: str | None = None
    blocked: bool = False           # the generator's draft was blocked by the guardrail
    used_fallback: bool = False


class CareAgent:
    """A single merchant session, end to end."""

    def __init__(
        self,
        policy: PolicySnapshot,
        client: LLMClient,
        tool_config: ToolConfig | None = None,
        classifier_model: str = CLASSIFIER_MODEL,
        generator_model: str = GENERATOR_MODEL,
    ) -> None:
        self.policy = policy
        self.client = client
        self.classifier_model = classifier_model
        self.generator_model = generator_model

        self.stubs = ToolStubs(tool_config)
        self.state_store = InMemoryStateStore()
        self.event_log = InMemoryEventLog()
        self.idempotency = InMemoryIdempotencyStore()
        self.broker = ToolBroker(self.stubs, self.idempotency, policy)
        self.runtime = Runtime(policy, self.state_store, self.event_log)
        self.mailbox = Mailbox(self.runtime)

        self.order_id: str | None = None
        self.language: str = "en"
        # Events queued here land *while a tool call is in flight* — after the chain is
        # dispatched but before its result comes back. That is the hardest ordering the spec
        # asks for, and the only way to reproduce it against a synchronous tool layer.
        self.inject_before_tool_result: list[Event] = []
        self.transcript: list[TranscriptEntry] = []
        self.trace: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self.guardrail_violations: int = 0

    # --- inputs ---------------------------------------------------------------

    def start(self, data: SessionState) -> "CareAgent":
        """Open the proactive session and run the turn-0 decision."""
        self.order_id = data.order_id
        self._dispatch(self.mailbox.start(data.order_id, data))
        return self

    def send_message(self, text: str) -> "CareAgent":
        """Feed one merchant message: classify it, then let the FSM decide."""
        self._require_started()
        self.transcript.append(TranscriptEntry(speaker="merchant", text=text, language=self.language))
        try:
            flags = classify(text, self.client, model=self.classifier_model)
        except ClassifierError as exc:
            # Never guess an intent — hand the conversation to a human.
            self._trace("classifier_unavailable", detail=str(exc))
            self._dispatch(self.mailbox.send(self.order_id, Event(type=EventType.CLASSIFIER_UNAVAILABLE)))
            return self

        self.language = flags.language
        self.transcript[-1] = self.transcript[-1].model_copy(update={"language": flags.language})
        event = Event(type=EventType.MERCHANT_MESSAGE, text=text, flags=flags)
        self._dispatch(self.mailbox.send(self.order_id, event))
        return self

    def send_event(self, event: Event) -> "CareAgent":
        """Feed a backend event, timer, or tool result into the same ordered mailbox."""
        self._require_started()
        self._dispatch(self.mailbox.send(self.order_id, event))
        return self

    # --- state ----------------------------------------------------------------

    @property
    def session(self) -> Session:
        self._require_started()
        session = self.state_store.get(self.order_id)
        if session is None:  # pragma: no cover - start() guarantees this
            raise RuntimeError(f"no session for order {self.order_id!r}")
        return session

    @property
    def is_terminal(self) -> bool:
        return self.session.terminal_reason is not None

    def trajectory(self) -> list[str]:
        """The ordered list of **policy actions** taken.

        Read from the session's own history, not from the effect trace: history speaks the
        policy engine's vocabulary (``auto_reassign``), whereas the trace speaks in execution
        terms (``tool_chain``). The evaluation compares against the engine, so the trajectory
        has to use the engine's names. ``self.trace`` remains the execution log.
        """
        return [h["action"] for h in self.session.history if h.get("action")]

    # --- effect execution ------------------------------------------------------

    def _dispatch(self, effects: list[Effect]) -> None:
        """Run effects in order. Effects that yield new events re-enter through the mailbox."""
        queue = list(effects)
        while queue:
            queue.extend(self._handle(queue.pop(0)))

    def _handle(self, effect: Effect) -> list[Effect]:
        if isinstance(effect, SendMessageEffect):
            self._say(effect.envelope)
            return []

        if isinstance(effect, CallToolChainEffect):
            self._trace("tool_chain", detail=",".join(effect.tool_sequence))
            result = self.broker.run_reassign_chain(
                self.order_id, effect.idempotency_key, effect.dispatch_epoch
            )
            follow_up: list[Effect] = []
            # An async event that arrived while the tool was running is applied first — the
            # call is never cancelled, so its result is reconciled against the newer state.
            while self.inject_before_tool_result:
                injected = self.inject_before_tool_result.pop(0)
                self._trace("inject_mid_flight", detail=injected.type.value)
                follow_up.extend(self.mailbox.send(self.order_id, injected))
            # Feed the tool result back through the mailbox — same ordering guarantee.
            follow_up.extend(self.mailbox.send(self.order_id, result))
            return follow_up

        if isinstance(effect, EscalateEffect):
            ticket = self.broker.escalate(self.order_id, effect.reason, effect.context_snapshot)
            self.tickets.append(
                {
                    "ticket_id": ticket.get("ticket_id"),
                    "reason": effect.reason,
                    "escalation_mode": effect.escalation_mode.value if effect.escalation_mode else None,
                    "guardrail_halt": effect.guardrail_halt,
                    "context_snapshot": effect.context_snapshot,
                }
            )
            self._trace("escalate", detail=effect.reason)
            # Tell the merchant a human is taking over, rather than going silent. This is a
            # courtesy message on an already-traced action, so it does not add a trajectory step.
            self._say(ActionEnvelope(action=ActionType.ESCALATE, reason=effect.reason), trace=False)
            return []

        if isinstance(effect, ResolveEffect):
            self._trace("resolve", detail=effect.reason)
            return []

        if isinstance(effect, ReconcileEffect):
            self._trace("reconcile", detail=effect.detail)
            return []

        if isinstance(effect, LogEffect):
            self._trace("log", detail=effect.message)
            return []

        return []  # pragma: no cover - all effect kinds are handled above

    def _say(self, envelope: ActionEnvelope, trace: bool = True) -> None:
        """Phrase an already-decided action and send it, guardrailed."""
        if not has_message(envelope.action):
            return
        reply = generate(envelope, self.client, language=self.language, model=self.generator_model)
        if reply.blocked:
            self.guardrail_violations += 1
        self.transcript.append(
            TranscriptEntry(
                speaker="agent",
                text=reply.text,
                language=reply.language,
                action=envelope.action.value,
                rule_id=envelope.rule_id,
                blocked=reply.blocked,
                used_fallback=reply.used_fallback,
            )
        )
        if trace:
            self._trace(
                "message",
                action=envelope.action.value,
                rule_id=envelope.rule_id,
                detail=envelope.reason,
                blocked=reply.blocked,
            )

    # --- bookkeeping ------------------------------------------------------------

    def _trace(self, kind: str, **fields: Any) -> None:
        entry: dict[str, Any] = {"kind": kind, **fields}
        if "action" not in entry:
            entry["action"] = kind
        entry["fsm_state"] = self.state_store.get(self.order_id).fsm_state.value if self.order_id else None
        self.trace.append(entry)

    def _require_started(self) -> None:
        if self.order_id is None:
            raise RuntimeError("call start() before sending messages or events")
