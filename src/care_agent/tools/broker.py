"""The tool broker: executes the reassignment chain safely and enforces the side-effect
boundary.

Responsibilities (kept out of the pure reducer):
* **Chaining** — ``check_captain_availability`` MUST pass before ``reassign_captain`` runs.
* **Retries with idempotency** — retry a transient reassign failure with the *same* key up to
  ``retry_cap``; cache the success so a replay never double-assigns.
* **Partial-failure handling** — if the reassignment commits but the confirmation dispatch
  fails, there is no unassign tool, so the broker reports ``partial_failure`` and the reducer
  escalates with the assigned captain recorded (the closest thing to a rollback here).
* **Key enforcement** — every side-effecting tool (reassign + the two financial tools) refuses
  to run without an idempotency key.

The broker returns a ``TOOL_RESULT`` :class:`Event` that the orchestrator feeds back into the
mailbox, closing the loop with the FSM.
"""

from __future__ import annotations

from care_agent.core.stores import IdempotencyStore
from care_agent.domain.models import Event, EventType
from care_agent.policy.loader import PolicySnapshot
from care_agent.tools.idempotency import IdempotentExecutor
from care_agent.tools.stubs import ToolStubs


class ToolBroker:
    def __init__(self, stubs: ToolStubs, idempotency_store: IdempotencyStore, policy: PolicySnapshot) -> None:
        self.stubs = stubs
        self.idem = idempotency_store
        self.policy = policy
        self.executor = IdempotentExecutor(idempotency_store)

    # --- the reassignment chain --------------------------------------------------

    def run_reassign_chain(self, order_id: str, idempotency_key: str, dispatch_epoch: int | None = None) -> Event:
        """check availability -> reassign (idempotent, retried) -> confirm. Returns a
        TOOL_RESULT event with outcome in {reassigned, no_captain, failed, partial_failure}."""
        check = self.stubs.check_captain_availability(order_id)
        if check.get("error"):
            return self._result("failed", dispatch_epoch, reason="check_failed")
        if not check.get("available", False):
            return self._result("no_captain", dispatch_epoch)

        eta = check.get("estimated_eta_minutes")
        reassign = self._reassign_with_retries(order_id, idempotency_key)
        if not reassign.get("success"):
            return self._result("failed", dispatch_epoch, reason="reassign_failed")

        new_captain = reassign.get("new_captain_id")
        confirm = self.stubs.notify_merchant(order_id, new_captain_id=new_captain)
        if not confirm.get("success"):
            return self._result("partial_failure", dispatch_epoch, new_captain_id=new_captain, new_eta=eta)
        return self._result("reassigned", dispatch_epoch, new_captain_id=new_captain, new_eta=eta)

    def _reassign_with_retries(self, order_id: str, key: str) -> dict:
        if not key:
            raise ValueError("reassign_captain requires an idempotency_key")
        cached = self.idem.get(key)
        if cached is not None and cached.get("success"):
            return {**cached, "idempotent_replay": True}  # replay -> no double-assign

        result: dict = {"success": False}
        for _ in range(self.policy.retry_cap):
            result = self.stubs.reassign_captain(order_id, key)
            if result.get("success"):
                self.idem.set(key, result)  # cache successes only
                return result
            if not result.get("transient", False):
                break  # terminal failure -> do not retry
        return result  # cap exhausted or terminal

    # --- financial tools (key-enforced; not driven by policy) --------------------

    def cancel_order(self, order_id: str, reason: str, waive_fee: bool, idempotency_key: str) -> dict:
        if not idempotency_key:
            raise ValueError("cancel_order requires an idempotency_key")
        return self.executor.run(
            idempotency_key, lambda: self.stubs.cancel_order(order_id, reason, waive_fee, idempotency_key)
        )

    def issue_merchant_credit(self, merchant_id: str, amount: float, currency: str, idempotency_key: str) -> dict:
        if not idempotency_key:
            raise ValueError("issue_merchant_credit requires an idempotency_key")
        return self.executor.run(
            idempotency_key, lambda: self.stubs.issue_merchant_credit(merchant_id, amount, currency, idempotency_key)
        )

    # --- escalation --------------------------------------------------------------

    def escalate(self, order_id: str, reason: str, context_snapshot: dict) -> dict:
        return self.stubs.escalate_to_human_ops(order_id, reason, context_snapshot)

    # --- helpers -----------------------------------------------------------------

    @staticmethod
    def _result(outcome: str, dispatch_epoch: int | None, **payload) -> Event:
        body = {"outcome": outcome, **payload}
        if dispatch_epoch is not None:
            body["dispatch_epoch"] = dispatch_epoch
        return Event(type=EventType.TOOL_RESULT, payload=body)
