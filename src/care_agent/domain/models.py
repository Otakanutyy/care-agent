"""Typed domain models.

These are plain data carriers with validation — no behaviour, no LLM, no I/O:

* :class:`SessionState` — everything the policy engine reads about one order.
* :class:`ClassifierFlags` — the fixed booleans (+ language) the classifier emits.
* :class:`ActionEnvelope` — the single typed decision the policy engine emits.
* :class:`Event` / :class:`EventType` — the inputs the FSM reducer consumes (Step 3).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MerchantTier(str, Enum):
    GOLD = "Gold"
    SILVER = "Silver"
    BRONZE = "Bronze"


class ActionType(str, Enum):
    """Every action the policy engine can decide. Note that ``cancel_order`` and
    ``issue_merchant_credit`` are intentionally absent — no policy rule authorizes the
    agent to cancel or move money, so a cancellation request always escalates to a human."""

    LOG_ONLY = "log_only"                        # R1
    NOTIFY_CONFIRM_ETA = "notify_confirm_eta"    # R2 proactive
    AUTO_REASSIGN = "auto_reassign"              # R3 (Gold)
    ASK_REASSIGN_OR_WAIT = "ask_reassign_or_wait"  # R4 proactive
    REASSIGN = "reassign"                        # R4: merchant accepted reassignment
    NOTIFY_REASSIGNED = "notify_reassigned"      # post-reassign: inform merchant of new captain+ETA
    ACKNOWLEDGE_WAIT = "acknowledge_wait"        # merchant chose to wait as-is
    RESOLVE_ETA_CONFIRMED = "resolve_eta_confirmed"  # R2: merchant confirmed new ETA
    DEGRADED_MODE_NOTICE = "degraded_mode_notice"    # override (e.g. active_outage)
    CLARIFY = "clarify"                          # ambiguous/off-policy merchant intent
    ESCALATE = "escalate"                        # R5, R6, R4-cancel, loop guard, fail-safe


class EscalationMode(str, Enum):
    PER_ORDER = "per_order"
    ATTACH_TO_INCIDENT = "attach_to_incident"


class EventType(str, Enum):
    MERCHANT_MESSAGE = "merchant_message"
    TOOL_RESULT = "tool_result"
    CAPTAIN_CANCELLED_MID_CALL = "CAPTAIN_CANCELLED_MID_CALL"
    ORDER_PREP_COMPLETED = "ORDER_PREP_COMPLETED"
    SYSTEM_ETA_UPDATED = "SYSTEM_ETA_UPDATED"
    MERCHANT_REPLY_TIMEOUT = "merchant_reply_timeout"
    SESSION_IDLE_TIMEOUT = "session_idle_timeout"
    TOOL_TIMEOUT = "tool_timeout"
    TICK = "tick"
    # The classifier could not read a merchant message. We never guess an intent: the session
    # fails safe to a human instead.
    CLASSIFIER_UNAVAILABLE = "classifier_unavailable"


class ClassifierFlags(BaseModel):
    """Fixed booleans the intent classifier emits (plus the detected language tag).
    The policy engine reads only the booleans; ``language`` is for the response generator."""

    model_config = {"frozen": True, "extra": "forbid"}

    requests_human: bool = False
    confirms_new_eta: bool = False
    requests_cancellation: bool = False
    accepts_reassignment: bool = False
    prefers_to_wait: bool = False
    language: str = "en"


class SessionState(BaseModel):
    """Everything the policy engine needs to know about one order at decision time."""

    model_config = {"extra": "forbid"}

    order_id: str
    merchant_name: str
    merchant_tier: MerchantTier
    delay_minutes: int = Field(ge=0)
    current_captain_id: str | None = None
    active_system_overrides: list[str] = Field(default_factory=list)

    # Accumulated counters read by the pipeline (maintained by the FSM in later steps).
    unresponsive_attempts_used: int = 0
    reassign_attempts_used: int = 0
    off_policy_push_count: int = 0


class ActionEnvelope(BaseModel):
    """The single typed decision the policy engine emits each turn. This is the *only*
    thing that authorizes downstream effects; the response generator may phrase it but
    cannot add to it."""

    model_config = {"frozen": True, "extra": "forbid"}

    action: ActionType
    rule_id: str | None = None            # "R1".."R5", "R6", or None (fail-safe)
    reason: str                            # machine-readable reason code
    escalation_mode: EscalationMode | None = None  # set only when action == ESCALATE
    tool_sequence: list[str] = Field(default_factory=list)  # e.g. check -> reassign
    notification: str | None = None        # template key (e.g. degraded_mode_no_eta)
    variables: dict[str, Any] = Field(default_factory=dict)  # engine-set values only
    counts_toward_loop_guard: bool = False
    active_overrides: list[str] = Field(default_factory=list)
    is_terminal: bool = False              # ESCALATE / RESOLVE close the loop


class Event(BaseModel):
    """An input to the FSM reducer. Kept intentionally light here; the reducer (Step 3)
    is the consumer and may refine payload handling per event type."""

    model_config = {"extra": "forbid"}

    type: EventType
    event_id: str | None = None
    text: str | None = None        # merchant message (raw)
    flags: ClassifierFlags | None = None  # classifier output for a MERCHANT_MESSAGE (filled by the edge)
    new_eta: int | None = None     # SYSTEM_ETA_UPDATED payload
    payload: dict[str, Any] = Field(default_factory=dict)  # e.g. TOOL_RESULT outcome fields
