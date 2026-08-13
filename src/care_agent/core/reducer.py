"""The FSM reducer: ``reduce(session, event, policy) -> (session', effects)``.

Pure and deterministic — no I/O, no LLM, no mutation of inputs. The policy engine's
``decide`` is the transition function; this module maps its :class:`ActionEnvelope` (plus
the current macro-state and the event kind) to the next :class:`FsmState` and a list of
:class:`Effect` descriptions.

Key behaviours:
* Terminal states (ESCALATED/RESOLVED) are inert — late input is logged/reconciled, never
  acted on (no zombie actions after a human handoff).
* An async event may interrupt any waiting state. It updates state, then the pipeline
  re-evaluates. If a tool is in flight, the call is never cancelled: if the reassignment
  premise still holds we keep awaiting it, otherwise we move on and reconcile the late
  result when it returns.
* A merchant message arriving during a tool call is buffered — only a human request acts.
"""

from __future__ import annotations

from typing import Any

from care_agent.core.session import (
    TERMINAL_STATES,
    CallToolChainEffect,
    Effect,
    EscalateEffect,
    FsmState,
    LogEffect,
    PendingTool,
    ReconcileEffect,
    ResolveEffect,
    SendMessageEffect,
    Session,
)
from care_agent.core.stores import EventLog, StateStore
from care_agent.domain.models import (
    ActionEnvelope,
    ActionType,
    ClassifierFlags,
    Event,
    EventType,
    SessionState,
)
from care_agent.guardrails.loop_guard import next_counter
from care_agent.policy.engine import decide, escalation_mode_for
from care_agent.policy.loader import PolicySnapshot

ASYNC_EVENTS = (
    EventType.CAPTAIN_CANCELLED_MID_CALL,
    EventType.ORDER_PREP_COMPLETED,
    EventType.SYSTEM_ETA_UPDATED,
)
REASSIGN_ACTIONS = (ActionType.AUTO_REASSIGN, ActionType.REASSIGN)
# The premise behind an in-flight reassignment is "policy still permits reassigning this
# order" — not "policy would dispatch one right now". For a non-Gold merchant who has already
# consented, the governing rule still reads ask-or-wait, and that must not be mistaken for the
# premise dying: discarding a reassignment already in flight is how you end up assigning twice.
REASSIGN_PREMISE_ACTIONS = REASSIGN_ACTIONS + (ActionType.ASK_REASSIGN_OR_WAIT,)
# Escalation reasons that represent a guardrail-forced stop (SAFE_HALT), for auditability.
GUARDRAIL_REASONS = frozenset({"loop_guard_tripped", "unauthorized_promise"})


def _snapshot(session: Session) -> dict[str, Any]:
    """The context a human ops ticket carries — enough to understand *why*, cold."""
    return {
        "order_id": session.data.order_id,
        "merchant_tier": session.data.merchant_tier.value,
        "delay_minutes": session.data.delay_minutes,
        "fsm_state": session.fsm_state.value,
        "last_action": session.last_action.model_dump() if session.last_action else None,
        "history": list(session.history),
    }


def _with_history(session: Session, record: dict[str, Any], **updates: Any) -> Session:
    return session.model_copy(update={"history": session.history + [record], **updates})


def _route(session: Session, env: ActionEnvelope) -> tuple[Session, list[Effect]]:
    """Map a decided envelope to the next state + effects."""
    data = session.data
    pending: PendingTool | None = None
    action = env.action

    if action == ActionType.LOG_ONLY:
        fsm, effects = FsmState.MONITORING, [LogEffect(message="within grace; monitoring")]
    elif action in (ActionType.NOTIFY_CONFIRM_ETA, ActionType.ASK_REASSIGN_OR_WAIT):
        fsm, effects = FsmState.AWAITING_MERCHANT_REPLY, [SendMessageEffect(envelope=env)]
    elif action == ActionType.CLARIFY:
        # Loop-guard counting happens in _on_merchant_message (it needs the flags).
        fsm, effects = FsmState.AWAITING_MERCHANT_REPLY, [SendMessageEffect(envelope=env)]
    elif action in REASSIGN_ACTIONS:
        key = f"{data.order_id}:reassign:e{session.epoch}"
        pending = PendingTool(
            purpose="reassign", tool_sequence=list(env.tool_sequence),
            idempotency_key=key, dispatch_epoch=session.epoch,
        )
        fsm = FsmState.AWAITING_TOOL_RESULT
        effects = [
            CallToolChainEffect(
                tool_sequence=list(env.tool_sequence), idempotency_key=key,
                dispatch_epoch=session.epoch, purpose="reassign", source_reason=env.reason,
            )
        ]
    elif action in (ActionType.ACKNOWLEDGE_WAIT, ActionType.DEGRADED_MODE_NOTICE):
        fsm, effects = FsmState.MONITORING, [SendMessageEffect(envelope=env)]
    elif action == ActionType.RESOLVE_ETA_CONFIRMED:
        fsm = FsmState.RESOLVED
        effects = [SendMessageEffect(envelope=env), ResolveEffect(reason=env.reason)]
    elif action == ActionType.ESCALATE:
        fsm = FsmState.ESCALATED
        effects = [
            EscalateEffect(
                reason=env.reason, escalation_mode=env.escalation_mode,
                context_snapshot=_snapshot(session),
                guardrail_halt=env.reason in GUARDRAIL_REASONS,
            )
        ]
    else:  # fail-safe
        fsm = FsmState.ESCALATED
        effects = [EscalateEffect(reason="unhandled_action", context_snapshot=_snapshot(session))]

    terminal_reason = env.reason if fsm in TERMINAL_STATES else None
    new_session = _with_history(
        session,
        {"action": action.value, "rule": env.rule_id, "fsm": fsm.value, "reason": env.reason},
        fsm_state=fsm, data=data, pending_tool=pending,
        last_action=env, terminal_reason=terminal_reason,
    )
    return new_session, effects


def _escalate(session: Session, reason: str, policy: PolicySnapshot) -> tuple[Session, list[Effect]]:
    """Escalate for reasons that don't originate from ``decide`` (timeouts, tool failure)."""
    mode = escalation_mode_for(session.data, policy)
    new_session = _with_history(
        session,
        {"action": "escalate", "fsm": FsmState.ESCALATED.value, "reason": reason},
        fsm_state=FsmState.ESCALATED, pending_tool=None, terminal_reason=reason,
    )
    return new_session, [
        EscalateEffect(
            reason=reason, escalation_mode=mode, context_snapshot=_snapshot(session),
            guardrail_halt=reason in GUARDRAIL_REASONS,
        )
    ]


def _apply_async_to_data(data: SessionState, event: Event) -> SessionState:
    if event.type == EventType.CAPTAIN_CANCELLED_MID_CALL:
        return data.model_copy(update={"current_captain_id": None})
    if event.type == EventType.SYSTEM_ETA_UPDATED and event.new_eta is not None:
        # Adopted interpretation: new_eta is the updated projected lateness in minutes.
        # (Decision #11: production derives delay from timestamps + a periodic tick; deferred.)
        return data.model_copy(update={"delay_minutes": max(0, int(event.new_eta))})
    return data


# --- Per-event handlers -------------------------------------------------------


def _on_merchant_message(session, event, policy):
    flags = event.flags or ClassifierFlags()
    if session.fsm_state == FsmState.AWAITING_TOOL_RESULT:
        # Buffer merchant intents while a tool is in flight; only a human request is sacred.
        if getattr(flags, policy.r6_preemption["flag"]):
            return _route(session, decide(session.data, flags, policy))
        s = _with_history(
            session, {"note": "buffered merchant message during tool flight"},
            epoch=session.epoch + 1,
        )
        return s, [LogEffect(message="buffered merchant message during tool flight; only human-request acts")]

    env = decide(session.data, flags, policy)
    # Advance the signature-keyed loop-guard counter (only reactive off-policy pushes count).
    is_off_policy = env.action == ActionType.CLARIFY
    sig, count = next_counter(session.last_off_policy_signature, session.data.off_policy_push_count, flags, is_off_policy)
    session = session.model_copy(
        update={
            "data": session.data.model_copy(update={"off_policy_push_count": count}),
            "last_off_policy_signature": sig,
        }
    )
    return _route(session, env)


def _on_tool_result(session, event, policy):
    if session.fsm_state != FsmState.AWAITING_TOOL_RESULT or session.pending_tool is None:
        return _reconcile(session, event)  # we already moved on
    outcome = (event.payload or {}).get("outcome")
    if outcome == "reassigned":
        new_captain = event.payload.get("new_captain_id")
        new_eta = event.payload.get("new_eta")
        data = session.data.model_copy(
            update={"current_captain_id": new_captain, "reassign_attempts_used": 0}
        )
        notify = ActionEnvelope(
            action=ActionType.NOTIFY_REASSIGNED,
            rule_id=session.last_action.rule_id if session.last_action else None,
            reason="reassigned",
            variables={"new_captain_id": new_captain, "new_eta": new_eta},
            is_terminal=True,
        )
        s = _with_history(
            session, {"action": "notify_reassigned", "fsm": FsmState.RESOLVED.value, "reason": "reassigned"},
            data=data, pending_tool=None, fsm_state=FsmState.RESOLVED,
            last_action=notify, terminal_reason="reassigned",
        )
        return s, [SendMessageEffect(envelope=notify), ResolveEffect(reason="reassigned")]
    if outcome == "partial_failure":
        # reassign committed but the confirmation dispatch failed. There is no unassign
        # tool, so the compensating action is to escalate WITH the assigned captain recorded,
        # so a human can reconcile (captain assigned, merchant not confirmed).
        new_captain = (event.payload or {}).get("new_captain_id")
        data = session.data.model_copy(update={"current_captain_id": new_captain})
        s = session.model_copy(update={"data": data, "pending_tool": None})
        return _escalate(s, "partial_failure", policy)
    if outcome in ("no_captain", "failed"):
        return _escalate(session, "tool_fail_or_no_captain", policy)
    return _escalate(session, "tool_unknown_outcome", policy)


def _on_async_event(session, event, policy):
    data = _apply_async_to_data(session.data, event)
    s2 = session.model_copy(update={"data": data, "epoch": session.epoch + 1})

    if event.type == EventType.ORDER_PREP_COMPLETED:
        # The order is ready — supersedes any in-flight reassignment; resolve.
        s3 = _with_history(
            s2, {"action": "resolve", "fsm": FsmState.RESOLVED.value, "reason": "prep_completed"},
            fsm_state=FsmState.RESOLVED, pending_tool=None, terminal_reason="prep_completed",
        )
        return s3, [ResolveEffect(reason="prep_completed")]

    env = decide(s2.data, None, policy)

    if session.fsm_state == FsmState.AWAITING_TOOL_RESULT:
        if env.action in REASSIGN_PREMISE_ACTIONS:
            # Premise still holds: keep awaiting the in-flight result, do not re-dispatch.
            s3 = _with_history(
                s2, {"note": f"{event.type.value} during tool flight; reassign premise still valid"},
            )
            return s3, [LogEffect(message=f"{event.type.value} during tool flight; reassign still warranted, awaiting result")]
        # Premise invalidated: move on per the new decision; the late tool result reconciles.
        return _route(s2, env)

    return _route(s2, env)


def _on_reply_timeout(session, event, policy):
    if session.fsm_state != FsmState.AWAITING_MERCHANT_REPLY:
        return session, [LogEffect(message="reply timeout ignored; not awaiting a reply")]
    used = session.data.unresponsive_attempts_used + 1
    data = session.data.model_copy(update={"unresponsive_attempts_used": used})
    s = session.model_copy(update={"data": data})
    if used >= policy.unresponsive_attempts:
        return _escalate(s, "unresponsive", policy)
    return _route(s, decide(data, None, policy))  # re-notify


def _on_tool_timeout(session, event, policy):
    if session.fsm_state != FsmState.AWAITING_TOOL_RESULT:
        return session, [LogEffect(message="tool timeout ignored; no tool in flight")]
    return _escalate(session, "tool_timeout", policy)


def _reeval(session, policy):
    return _route(session, decide(session.data, None, policy))


def _reconcile(session, event):
    """A tool result arrived after the session moved on. We cannot un-assign a captain, so a
    late *success* is recorded rather than dropped — otherwise session state diverges from the
    backend and a later decision could assign a second captain on top of the first."""
    payload = event.payload or {}
    updates: dict[str, Any] = {"note": "reconciled stale tool result"}
    session_updates: dict[str, Any] = {}
    if payload.get("outcome") in ("reassigned", "partial_failure") and payload.get("new_captain_id"):
        session_updates["data"] = session.data.model_copy(
            update={"current_captain_id": payload["new_captain_id"]}
        )
        updates["note"] = "recorded captain from late tool result"
    s = _with_history(session, updates, **session_updates)
    return s, [ReconcileEffect(detail=f"stale tool result reconciled in {session.fsm_state.value}: {payload}")]


def _inert(session, event):
    if event.type == EventType.TOOL_RESULT:
        s = _with_history(session, {"note": "late tool result in terminal state"})
        return s, [ReconcileEffect(detail=f"late tool result in terminal {session.fsm_state.value}: {event.payload}")]
    s = _with_history(session, {"note": f"ignored {event.type.value} in terminal state"})
    return s, [LogEffect(message=f"ignored {event.type.value} in terminal {session.fsm_state.value}")]


def reduce(session: Session, event: Event, policy: PolicySnapshot) -> tuple[Session, list[Effect]]:
    """Apply one event to a session. Pure: returns a new session and the effects to run."""
    if session.fsm_state in TERMINAL_STATES:
        return _inert(session, event)

    et = event.type
    if et == EventType.MERCHANT_MESSAGE:
        return _on_merchant_message(session, event, policy)
    if et == EventType.TOOL_RESULT:
        return _on_tool_result(session, event, policy)
    if et in ASYNC_EVENTS:
        return _on_async_event(session, event, policy)
    if et == EventType.MERCHANT_REPLY_TIMEOUT:
        return _on_reply_timeout(session, event, policy)
    if et == EventType.TOOL_TIMEOUT:
        return _on_tool_timeout(session, event, policy)
    if et == EventType.CLASSIFIER_UNAVAILABLE:
        return _escalate(session, "classifier_unavailable", policy)
    if et in (EventType.SESSION_IDLE_TIMEOUT, EventType.TICK):
        return _reeval(session, policy)
    return session, [LogEffect(message=f"unhandled event {et.value}")]


def bootstrap(data: SessionState, policy: PolicySnapshot) -> tuple[Session, list[Effect]]:
    """INIT: seed a session and run the turn-0 (proactive) evaluation."""
    return _route(Session(fsm_state=FsmState.INIT, data=data), decide(data, None, policy))


class Runtime:
    """Wires the pure reducer to the stores: bootstrap + apply-one-event, persisting state
    and appending to the event log. The mailbox drives this one event at a time."""

    def __init__(self, policy: PolicySnapshot, state_store: StateStore, event_log: EventLog) -> None:
        self.policy = policy
        self.state = state_store
        self.log = event_log

    def start(self, order_id: str, data: SessionState) -> list[Effect]:
        session, effects = bootstrap(data, self.policy)
        self.state.put(order_id, session)
        return effects

    def apply(self, order_id: str, event: Event) -> list[Effect]:
        session = self.state.get(order_id)
        if session is None:
            raise KeyError(f"no session for order {order_id!r}; call start() first")
        new_session, effects = reduce(session, event, self.policy)
        self.state.put(order_id, new_session)
        self.log.append(order_id, event)
        return effects
