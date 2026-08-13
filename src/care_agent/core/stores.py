"""Storage interfaces + in-memory implementations.

Everything the FSM persists sits behind a small interface so the in-memory versions used
for the assessment can be swapped for durable, shared backends (Redis/DB) in production
without touching the reducer. That swap is the production story described in the write-up;
here the defaults are plain dicts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from care_agent.core.session import Session
from care_agent.domain.models import Event


class StateStore(ABC):
    """Persists the current :class:`Session` per order_id."""

    @abstractmethod
    def get(self, order_id: str) -> Session | None: ...

    @abstractmethod
    def put(self, order_id: str, session: Session) -> None: ...


class EventLog(ABC):
    """Append-only per-session event log (the transcript + crash-recovery source)."""

    @abstractmethod
    def append(self, order_id: str, event: Event) -> None: ...

    @abstractmethod
    def events(self, order_id: str) -> list[Event]: ...


class IdempotencyStore(ABC):
    """Keyed cache for side-effecting tool results. Used by the tool broker (Step 4);
    the interface lives here so it is swappable alongside the other stores."""

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def keys(self) -> list[str]:
        """Keys with a cached (successful) result — one per committed side effect.
        The evaluation harness reads this to observe idempotency directly rather than
        inferring it from call counts."""


# --- In-memory implementations ------------------------------------------------


class InMemoryStateStore(StateStore):
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, order_id: str) -> Session | None:
        return self._sessions.get(order_id)

    def put(self, order_id: str, session: Session) -> None:
        self._sessions[order_id] = session


class InMemoryEventLog(EventLog):
    def __init__(self) -> None:
        self._log: dict[str, list[Event]] = {}

    def append(self, order_id: str, event: Event) -> None:
        self._log.setdefault(order_id, []).append(event)

    def events(self, order_id: str) -> list[Event]:
        return list(self._log.get(order_id, []))


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def keys(self) -> list[str]:
        return list(self._cache)
