"""Session management for the remote testing surface.

Wraps :class:`~care_agent.agent.CareAgent` and returns plain dictionaries, so both the MCP
server and any future HTTP/UI layer share one implementation.

Every turn reports **why** it did what it did — `matched_rule`, `action`, and
`guardrail_status` come back in the payload itself, not just in a log. That is the point of
this surface: a reviewer (or a reviewer's agent) can verify that the decision came from the
policy engine rather than from the model, without reading the code.

Two production concerns are handled here because this runs on a public endpoint:

* **Spend is bounded.** Sessions and turns-per-session are capped, so a leaked URL cannot run
  up an unbounded API bill.
* **Live is opt-in.** The default is the deterministic offline stand-in; the real Claude API is
  used only when ``CARE_AGENT_MODE=live``.

State is in-memory, so sessions do not survive a restart. That is consistent with the
architecture's swappable-store decision, and `get_trace` fails cleanly on an unknown id.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from care_agent.agent import CareAgent, TranscriptEntry
from care_agent.domain.models import Event, EventType, SessionState
from care_agent.llm.client import (
    CLASSIFIER_MODEL,
    GENERATOR_MODEL,
    AnthropicClient,
    LLMClient,
)
from care_agent.llm.offline import OfflineLLMClient
from care_agent.policy.loader import PolicySnapshot, load_policy
from care_agent.tools.stubs import ToolConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "policy" / "policy.json"

DEFAULT_MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))
DEFAULT_MAX_TURNS = int(os.getenv("MAX_TURNS", "25"))
#: Total merchant turns this process will serve. Turns are what actually cost money — a session
#: sitting in a dict costs nothing — so the spend ceiling belongs here rather than on the
#: session count. Sessions are evicted oldest-first instead of refused, because a reviewer
#: arriving after someone else's testing should never be told the endpoint is full.
DEFAULT_MAX_TOTAL_TURNS = int(os.getenv("MAX_TOTAL_TURNS", "600"))

#: Events a tester may inject. Deliberately excludes the transport-internal types
#: (``merchant_message``, ``tool_result``, ``tick``, ``classifier_unavailable``) — those are
#: produced by the runtime itself, and injecting them out of band would corrupt the session
#: rather than test it.
TRIGGERABLE_EVENTS: tuple[EventType, ...] = (
    EventType.CAPTAIN_CANCELLED_MID_CALL,
    EventType.ORDER_PREP_COMPLETED,
    EventType.SYSTEM_ETA_UPDATED,
    EventType.MERCHANT_REPLY_TIMEOUT,
    EventType.SESSION_IDLE_TIMEOUT,
    EventType.TOOL_TIMEOUT,
)


class SessionLimitError(RuntimeError):
    """Raised when a cap protecting the deployment's API budget is hit."""


class UnknownSessionError(KeyError):
    """Raised when a session id is not held in memory (it may have expired on restart)."""


def build_client(mode: str | None = None) -> LLMClient:
    """Offline stand-in unless live is explicitly requested."""
    mode = (mode or os.getenv("CARE_AGENT_MODE", "offline")).lower()
    return AnthropicClient() if mode == "live" else OfflineLLMClient()


class ManagedSession:
    def __init__(self, session_id: str, agent: CareAgent) -> None:
        self.id = session_id
        self.agent = agent
        self.turns = 0
        self.created_at = time.time()
        # One writer per order. The reducer is single-threaded by design, but this surface is
        # not: a merchant message and a backend event can arrive on two HTTP connections at the
        # same instant, which is exactly the case the spec asks for. The lock makes the
        # serialisation explicit instead of leaving it to chance.
        self.lock = threading.Lock()
        #: Backend events accepted but not yet applied, so /events can answer immediately.
        self.pending_events = 0


class SessionManager:
    """Holds live agent sessions for the remote testing surface."""

    def __init__(
        self,
        policy: PolicySnapshot | None = None,
        mode: str | None = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_total_turns: int = DEFAULT_MAX_TOTAL_TURNS,
    ) -> None:
        self.policy = policy or load_policy(POLICY_PATH)
        self.mode = (mode or os.getenv("CARE_AGENT_MODE", "offline")).lower()
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.max_total_turns = max_total_turns
        self.turns_served = 0
        self.sessions_evicted = 0
        # Serving models are overridable per-deployment via env, so a cost-constrained instance
        # can run the cheaper tier without changing the code's documented defaults. Only the
        # generator is worth downshifting; the classifier is already the cheapest tier.
        self.classifier_model = os.getenv("CLASSIFIER_MODEL", CLASSIFIER_MODEL)
        self.generator_model = os.getenv("GENERATOR_MODEL", GENERATOR_MODEL)
        self._sessions: dict[str, ManagedSession] = {}

    @property
    def policy_version(self) -> Any:
        return self.policy.raw["version"]

    # --- lifecycle ---------------------------------------------------------

    def start(
        self,
        order_id: str | None = None,
        merchant_name: str = "Test Merchant",
        merchant_tier: str = "Silver",
        delay_minutes: int = 30,
        current_captain_id: str | None = "captain-100",
        active_system_overrides: list[str] | None = None,
        tools: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Make room rather than refuse. Sessions are in-memory and disposable; the oldest is the
        # least likely to still be someone's live conversation, and telling a reviewer the
        # endpoint is full reads as broken. Spend stays bounded by the turn budget below.
        while len(self._sessions) >= self.max_sessions:
            self._sessions.pop(next(iter(self._sessions)))
            self.sessions_evicted += 1

        session_id = uuid.uuid4().hex[:12]
        agent = CareAgent(
            self.policy,
            build_client(self.mode),
            tool_config=ToolConfig(**(tools or {})),
            classifier_model=self.classifier_model,
            generator_model=self.generator_model,
        )
        state = SessionState(
            order_id=order_id or f"order-{session_id}",
            merchant_name=merchant_name,
            merchant_tier=merchant_tier,
            delay_minutes=delay_minutes,
            current_captain_id=current_captain_id,
            active_system_overrides=active_system_overrides or [],
        )
        before = 0
        agent.start(state)
        self._sessions[session_id] = ManagedSession(session_id, agent)
        return {
            "session_id": session_id,
            "mode": self.mode,
            "policy_version": self.policy_version,
            "context": state.model_dump(mode="json"),
            **self._turn_report(agent, before, 0),
        }

    def _get(self, session_id: str) -> ManagedSession:
        managed = self._sessions.get(session_id)
        if managed is None:
            raise UnknownSessionError(
                f"unknown session {session_id!r} - it may have expired when the server restarted; "
                "call start_session again"
            )
        return managed

    # --- inputs ------------------------------------------------------------

    def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        managed = self._get(session_id)
        if managed.turns >= self.max_turns:
            raise SessionLimitError(f"this session has reached its {self.max_turns}-turn cap")
        if self.turns_served >= self.max_total_turns:
            raise SessionLimitError(
                f"this deployment has served its budget of {self.max_total_turns} merchant turns. "
                "Everything still works offline from the repository: `python run_all.py`."
            )
        if managed.agent.is_terminal:
            # Record it for real. The note below used to promise the input was recorded while
            # dropping it on the floor, so a human picking up the escalation could not see what
            # the merchant said after the handoff — which is exactly when they say the thing
            # that matters. No classification: the session is closed, so paying a model to
            # label text nothing will act on would be waste.
            managed.agent.transcript.append(TranscriptEntry(speaker="merchant", text=text))
            # Same key set as a live turn. A caller looping over turns should not have to
            # special-case the shape of the response just because the session has closed.
            report = self._turn_report(managed.agent, len(managed.agent.transcript), len(managed.agent.tickets))
            report["note"] = (
                "this session is terminal and inert; the message is recorded on the transcript "
                "for the human handling the handoff, but the agent will not act on it"
            )
            return {"session_id": session_id, **report}

        with managed.lock:
            before_msgs, before_tickets = len(managed.agent.transcript), len(managed.agent.tickets)
            was = managed.agent.session.fsm_state.value
            managed.agent.send_message(text)
            managed.turns += 1
            self.turns_served += 1
            return {
                "session_id": session_id,
                **self._turn_report(managed.agent, before_msgs, before_tickets, previous_fsm_state=was),
            }

    @staticmethod
    def _parse_event(event_type: str) -> EventType:
        valid = ", ".join(e.value for e in TRIGGERABLE_EVENTS)
        try:
            kind = EventType(event_type)
        except ValueError as exc:
            raise ValueError(f"unknown event_type {event_type!r}; expected one of: {valid}") from exc
        if kind not in TRIGGERABLE_EVENTS:
            raise ValueError(
                f"{event_type!r} is produced by the runtime itself and cannot be injected; "
                f"expected one of: {valid}"
            )
        return kind

    def accept_event(
        self, session_id: str, event_type: str, new_eta: int | None = None, payload: dict | None = None
    ) -> dict[str, Any]:
        """Take the event onto the queue and answer immediately.

        A backend event queue does not wait for a conversation to finish talking, and neither
        should the caller: an event fired while the agent is mid-reply blocked for as long as
        that reply took, which made an asynchronous queue behave synchronously from the outside.
        Validation still happens up front, so a malformed event is rejected rather than
        silently accepted and dropped.
        """
        managed = self._get(session_id)
        kind = self._parse_event(event_type)
        managed.pending_events += 1

        def apply() -> None:
            try:
                with managed.lock:
                    managed.agent.send_event(Event(type=kind, new_eta=new_eta, payload=payload or {}))
            finally:
                managed.pending_events -= 1

        threading.Thread(target=apply, daemon=True).start()
        return {
            "session_id": session_id,
            "accepted": True,
            "event_type": kind.value,
            "new_eta": new_eta,
            "pending_events": managed.pending_events,
            "note": "queued; read the trace to see its effect once applied",
        }

    def trigger_event(
        self, session_id: str, event_type: str, new_eta: int | None = None, payload: dict | None = None
    ) -> dict[str, Any]:
        """Apply an event and wait for the result — for callers that want the outcome."""
        managed = self._get(session_id)
        kind = self._parse_event(event_type)

        # Report from inside the lock: reading the agent while another thread is mutating it
        # would compose the answer from two different moments.
        with managed.lock:
            before_msgs, before_tickets = len(managed.agent.transcript), len(managed.agent.tickets)
            was = managed.agent.session.fsm_state.value
            managed.agent.send_event(Event(type=kind, new_eta=new_eta, payload=payload or {}))
            return {
                "session_id": session_id,
                "event_applied": kind.value,
                **self._turn_report(managed.agent, before_msgs, before_tickets, previous_fsm_state=was),
            }

    # --- inspection --------------------------------------------------------

    def get_trace(self, session_id: str) -> dict[str, Any]:
        agent = self._get(session_id).agent
        return {
            "session_id": session_id,
            "policy_version": self.policy_version,
            "transcript": [e.model_dump() for e in agent.transcript],
            "trace": agent.trace,
            "trajectory": agent.trajectory(),
            "tickets": agent.tickets,
            "tool_calls": dict(agent.stubs.calls),
            "tool_call_order": agent.stubs.call_log,
            "guardrail_blocks": agent.guardrail_violations,
            **self._state_report(agent),
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": m.id,
                "order_id": m.agent.order_id,
                "turns": m.turns,
                "fsm_state": m.agent.session.fsm_state.value,
                "terminal_reason": m.agent.session.terminal_reason,
            }
            for m in self._sessions.values()
        ]

    # --- reporting ---------------------------------------------------------

    def _state_report(self, agent: CareAgent) -> dict[str, Any]:
        session = agent.session
        return {
            "fsm_state": session.fsm_state.value,
            "terminal_reason": session.terminal_reason,
            "delay_minutes": session.data.delay_minutes,
            "current_captain_id": session.data.current_captain_id,
            "active_system_overrides": list(session.data.active_system_overrides),
        }

    def _turn_report(
        self,
        agent: CareAgent,
        before_msgs: int,
        before_tickets: int,
        previous_fsm_state: str | None = None,
    ) -> dict[str, Any]:
        """What this turn did, and — critically — which policy rule decided it."""
        new_messages = [
            e.model_dump() for e in agent.transcript[before_msgs:] if e.speaker == "agent"
        ]
        new_tickets = agent.tickets[before_tickets:]
        last = agent.session.last_action
        blocked = any(m.get("blocked") for m in new_messages)
        return {
            "agent_replies": new_messages,
            "matched_rule": last.rule_id if last else None,
            "action": last.action.value if last else None,
            "decision_reason": last.reason if last else None,
            "guardrail_status": "blocked" if blocked else "clean",
            "escalation": new_tickets[0] if new_tickets else None,
            "tool_calls": dict(agent.stubs.calls),
            "previous_fsm_state": previous_fsm_state,
            **self._state_report(agent),
        }
