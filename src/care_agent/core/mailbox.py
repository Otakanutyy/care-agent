"""Single-writer, ordered per-session mailbox.

Every input for one order is applied one at a time by a single consumer, in arrival order.
In this synchronous build that guarantee is inherent (one thread) — the mailbox's job is to
preserve per-order FIFO ordering and to be *reentrancy-safe*: if applying an event enqueues
another event for the same order (e.g. the tool broker feeds a TOOL_RESULT back in), that
event is queued and processed after the current one, never interleaved. The same partition
key + single-consumer model is what scales to multiple workers in production (one owner per
order, guarded by a lease/fence — described in the write-up).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

from care_agent.core.session import Effect
from care_agent.domain.models import Event, SessionState


class SupportsApply(Protocol):
    def start(self, order_id: str, data: SessionState) -> list[Effect]: ...
    def apply(self, order_id: str, event: Event) -> list[Effect]: ...


class Mailbox:
    """Serializes event application per order_id over an injected runtime."""

    def __init__(self, runtime: SupportsApply) -> None:
        self._runtime = runtime
        self._queues: dict[str, deque[Event]] = defaultdict(deque)
        self._draining: set[str] = set()

    def start(self, order_id: str, data: SessionState) -> list[Effect]:
        return self._runtime.start(order_id, data)

    def send(self, order_id: str, event: Event) -> list[Effect]:
        """Enqueue an event and drain this order's queue in order. Returns the effects
        produced. If called reentrantly while the order is already draining, the event is
        queued for the active drain loop and an empty list is returned."""
        self._queues[order_id].append(event)
        if order_id in self._draining:
            return []

        self._draining.add(order_id)
        produced: list[Effect] = []
        try:
            queue = self._queues[order_id]
            while queue:
                next_event = queue.popleft()
                produced.extend(self._runtime.apply(order_id, next_event))
        finally:
            self._draining.discard(order_id)
        return produced
