"""Keep the order of recorded conversation events while a confirmed turn can still get a late final.

A final can reach the agent after the turn its audio belongs to was committed. The turn's message
and the reply to it are already on their way, so the late final would be recorded behind the reply.
While such a turn is still open, the recording events (`conversation_item_added`,
`function_tools_executed`) are queued in arrival order and a late final is inserted right after the
turn's last item. Conversation state (chat context, playback, generation) is never delayed: only
the emission of those events is.

A turn is released when the provider says no more finals can come for its audio, when the hold
time is over, or when the session closes. Nothing is dropped and nothing is emitted twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..log import logger

# longest time the recording of events is held back for one turn
RECORD_HOLD_MAX_SECONDS = 3.0

GATED_EVENTS = ("conversation_item_added", "function_tools_executed")


@dataclass
class _Turn:
    turn_id: int
    message_ids: set[int] = field(default_factory=set)
    message_pushed: bool = False
    pending_late: list[tuple[str, Any]] = field(default_factory=list)
    marker: _Marker | None = None
    timer: asyncio.TimerHandle | None = None
    last_item: Any = None
    released: bool = False


class _Marker:
    """A position in the queue: nothing behind it is emitted while it is the first entry."""

    def __init__(self, turn: _Turn) -> None:
        self.turn = turn


class RecordGate:
    def __init__(self, emit: Callable[[str, Any], None]) -> None:
        self._emit = emit
        self._queue: list[tuple[str, Any] | _Marker] = []
        self._turns: dict[int, _Turn] = {}
        self._flushing = False

    @property
    def active(self) -> bool:
        return bool(self._queue) or any(not t.released for t in self._turns.values())

    def open_turn(self, turn_id: int) -> None:
        if turn_id in self._turns:
            return
        turn = _Turn(turn_id)
        self._turns[turn_id] = turn
        try:
            turn.timer = asyncio.get_running_loop().call_later(
                RECORD_HOLD_MAX_SECONDS, self.release_turn, turn_id
            )
        except RuntimeError:
            turn.timer = None

    def bind_message(self, turn_id: int, message: Any) -> None:
        turn = self._turns.get(turn_id)
        if turn is not None:
            turn.message_ids.add(id(message))

    def is_open(self, turn_id: int) -> bool:
        turn = self._turns.get(turn_id)
        return turn is not None and not turn.released

    def last_item(self, turn_id: int) -> Any:
        turn = self._turns.get(turn_id)
        return turn.last_item if turn is not None else None

    def push(self, name: str, arg: Any) -> None:
        """Emit now, or queue behind an open turn."""
        item = getattr(arg, "item", None)
        owner = next(
            (
                t
                for t in self._turns.values()
                if item is not None and id(item) in t.message_ids and not t.released
            ),
            None,
        )
        self._queue.append((name, arg))
        if owner is not None and not owner.message_pushed:
            owner.message_pushed = True
            owner.last_item = item
            for late in owner.pending_late:
                self._queue.append(late)
                owner.last_item = late[1].item
            owner.pending_late.clear()
            owner.marker = _Marker(owner)
            self._queue.append(owner.marker)
        self._flush()

    def insert_late(self, turn_id: int, name: str, arg: Any) -> None:
        """Record a late final right after the turn's last item."""
        turn = self._turns.get(turn_id)
        if turn is None or turn.released:
            self._queue.append((name, arg))
            self._flush()
            return
        if turn.marker is None:
            turn.pending_late.append((name, arg))
            turn.last_item = getattr(arg, "item", None)
            return
        index = self._queue.index(turn.marker)
        self._queue.insert(index, (name, arg))
        turn.last_item = getattr(arg, "item", None)
        self._flush()

    def release_turn(self, turn_id: int) -> None:
        turn = self._turns.get(turn_id)
        if turn is None or turn.released:
            return
        turn.released = True
        if turn.timer is not None:
            turn.timer.cancel()
            turn.timer = None
        if turn.marker is not None and turn.marker in self._queue:
            self._queue.remove(turn.marker)
        for late in turn.pending_late:
            self._queue.append(late)
        turn.pending_late.clear()
        self._flush()
        if not self._queue:
            self._turns = {k: t for k, t in self._turns.items() if not t.released}

    def release_all(self) -> None:
        for turn_id in list(self._turns):
            self.release_turn(turn_id)
        self._queue = [e for e in self._queue if not isinstance(e, _Marker)]
        self._flush()

    def _flush(self) -> None:
        if self._flushing:
            return
        self._flushing = True
        try:
            while self._queue and not isinstance(self._queue[0], _Marker):
                name, arg = self._queue.pop(0)
                try:
                    self._emit(name, arg)
                except Exception:
                    logger.exception("error while emitting a held recording event")
        finally:
            self._flushing = False
