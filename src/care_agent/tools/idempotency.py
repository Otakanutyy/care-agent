"""Idempotency: deterministic key generation + a cache-on-success-only executor.

Two rules from the gap audit:
* Cache **only successful** results. A transient failure is never cached, so a retry with
  the same key genuinely re-attempts; a *success* is cached, so a duplicate/replayed call
  returns the cached result instead of assigning a second captain / issuing a second credit.
* The key is stable across retries of the same logical attempt and distinct across genuinely
  new attempts — here, keyed by (order_id, purpose, epoch). The reducer mints reassignment
  keys in this same format; the executor mints keys for tools it drives directly.
"""

from __future__ import annotations

from typing import Callable

from care_agent.core.stores import IdempotencyStore


def make_key(order_id: str, purpose: str, epoch: int) -> str:
    """Deterministic idempotency key. Matches the format the FSM reducer uses for reassign."""
    return f"{order_id}:{purpose}:e{epoch}"


class IdempotentExecutor:
    """Runs a side-effecting call at most once per key (caching successes only)."""

    def __init__(self, store: IdempotencyStore) -> None:
        self.store = store

    def run(self, key: str, fn: Callable[[], dict]) -> dict:
        if not key:
            raise ValueError("an idempotency key is required for side-effecting tools")
        cached = self.store.get(key)
        if cached is not None and cached.get("success"):
            return {**cached, "idempotent_replay": True}
        result = fn()
        if result.get("success"):
            self.store.set(key, result)  # cache successes only
        return result
