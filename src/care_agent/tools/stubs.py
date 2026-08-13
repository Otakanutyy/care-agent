"""Simulated tool stubs.

These stand in for Careem's real dispatch/notification/ops backends. What is being tested
is how the agent behaves across a range of stub *outcomes* (available / not available /
transient failure / terminal failure / partial failure), not any matching logic — so each
run's outcomes are declared by a :class:`ToolConfig` (decision E: scenario-declared).

Two deviations from the spec signatures, both deliberate:
* ``reassign_captain`` already takes an ``idempotency_key`` (from the spec). We additionally
  require one on ``cancel_order`` and ``issue_merchant_credit`` — the spec omits it, but
  double-executing a refund/credit is the most damaging thing here (audit gap #9).
* ``escalate_to_human_ops`` is not in the spec's tool list at all; it must be added because
  R5/R6 both mandate escalation (audit gap #1).
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Per-scenario declaration of what each tool does this run."""

    # check_captain_availability
    available: bool = True
    estimated_eta_minutes: int = 12
    check_error: bool = False

    # reassign_captain — one entry consumed per call: "success" | "transient" | "terminal"
    reassign_outcomes: list[str] = Field(default_factory=lambda: ["success"])
    new_captain_id: str = "captain-2"

    # post-reassign confirmation dispatch (the partial-failure point)
    notify_succeeds: bool = True

    # escalate_to_human_ops
    ticket_id: str = "TICKET-1"

    # financial tools (never triggered by policy; present for completeness + key enforcement)
    cancel_success: bool = True
    refund_issued: bool = True
    credit_success: bool = True
    transaction_id: str = "TXN-1"


class ToolStubs:
    """Stateful simulator. Tracks per-tool call counts and an ordered call log so tests can
    assert idempotency (no double-assign) and chaining order (check before reassign)."""

    def __init__(self, config: ToolConfig | None = None) -> None:
        self.config = config or ToolConfig()
        self.calls: Counter[str] = Counter()
        self.call_log: list[str] = []
        self._reassign_index = 0

    def _record(self, name: str) -> None:
        self.calls[name] += 1
        self.call_log.append(name)

    # --- spec tools ---

    def check_captain_availability(self, order_id: str) -> dict:
        self._record("check_captain_availability")
        if self.config.check_error:
            return {"available": False, "error": "check_failed"}
        return {"available": self.config.available, "estimated_eta_minutes": self.config.estimated_eta_minutes}

    def reassign_captain(self, order_id: str, idempotency_key: str) -> dict:
        self._record("reassign_captain")
        outcomes = self.config.reassign_outcomes or ["success"]
        idx = self._reassign_index
        self._reassign_index += 1
        outcome = outcomes[idx] if idx < len(outcomes) else outcomes[-1]
        if outcome == "success":
            return {"success": True, "new_captain_id": self.config.new_captain_id}
        if outcome == "transient":
            return {"success": False, "transient": True, "error": "timeout"}
        return {"success": False, "transient": False, "error": "no_captain_available"}

    def cancel_order(self, order_id: str, reason: str, waive_fee: bool, idempotency_key: str) -> dict:
        self._record("cancel_order")
        return {"success": self.config.cancel_success, "refund_issued": self.config.refund_issued}

    def issue_merchant_credit(self, merchant_id: str, amount: float, currency: str, idempotency_key: str) -> dict:
        self._record("issue_merchant_credit")
        return {"success": self.config.credit_success, "transaction_id": self.config.transaction_id}

    # --- added tool (escalation) ---

    def escalate_to_human_ops(self, order_id: str, reason: str, context_snapshot: dict) -> dict:
        self._record("escalate_to_human_ops")
        return {"success": True, "ticket_id": self.config.ticket_id}

    # --- post-reassign confirmation dispatch (system notification, not the chat reply) ---

    def notify_merchant(self, order_id: str, new_captain_id: str | None = None) -> dict:
        self._record("notify_merchant")
        return {"success": self.config.notify_succeeds}
