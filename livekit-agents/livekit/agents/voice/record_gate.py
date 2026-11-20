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

# longest time an agent message waits for a customer word that ended before the agent started
SPEECH_ROW_HOLD_MAX_SECONDS = 1.0
# a word that ended longer than this before the agent's first audio is not waited for
SPEECH_ROW_HOLD_WINDOW_SECONDS = 3.0


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
    message: Any = None
    restore_end: float | None = None


class _Marker:
    """A position in the queue: nothing behind it is emitted while it is the first entry."""

    def __init__(self, turn: _Turn) -> None:
        self.turn = turn


class _Hold(_Marker):
    """A position in the queue held for a customer row that belongs ahead of the entry behind it."""

    def __init__(self, first_audio_at: float) -> None:
        self.turn = None  # type: ignore[assignment]
        self.first_audio_at = first_audio_at
        self.timer: asyncio.TimerHandle | None = None


class RecordGate:
    def __init__(self, emit: Callable[[str, Any], None]) -> None:
        self._emit = emit
        self._queue: list[tuple[str, Any] | _Marker] = []
        self._turns: dict[int, _Turn] = {}
        self._flushing = False
        self._hold: _Hold | None = None

    @property
    def hold_active(self) -> bool:
        return self._hold is not None

    @property
    def hold_first_audio_at(self) -> float | None:
        return self._hold.first_audio_at if self._hold is not None else None

    def hold(self, name: str, arg: Any, first_audio_at: float, seconds: float) -> None:
        """Queue an agent message behind a hold: a customer row may still come ahead of it.

        The customer row goes in front with ``push_before_hold``; when none comes the hold
        is released after ``seconds`` and the message goes out as it is. Only the emission of
        the recording event waits.
        """
        hold = _Hold(first_audio_at)
        self._queue.append(hold)
        self._queue.append((name, arg))
        if self._hold is None:
            self._hold = hold
        try:
            hold.timer = asyncio.get_running_loop().call_later(seconds, self.release_hold, hold)
        except RuntimeError:
            hold.timer = None
        self._flush()

    def push_before_hold(self, name: str, arg: Any) -> None:
        """Record a customer row ahead of the held agent message, then release the hold."""
        hold = self._hold
        if hold is None or hold not in self._queue:
            self._queue.append((name, arg))
            self._flush()
            return
        self._queue.insert(self._queue.index(hold), (name, arg))
        self.release_hold(hold)

    def release_hold(self, hold: _Hold | None = None) -> None:
        hold = hold or self._hold
        if hold is None:
            return
        if hold.timer is not None:
            hold.timer.cancel()
            hold.timer = None
        if hold in self._queue:
            self._queue.remove(hold)
        if self._hold is hold:
            self._hold = next((e for e in self._queue if isinstance(e, _Hold)), None)
        self._flush()

    @property
    def active(self) -> bool:
        # a released turn whose own message has not been added yet still owns a waiting late
        # final: the message must come through the gate to take it along
        return bool(self._queue) or any(
            not t.released or t.pending_late for t in self._turns.values()
        )

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
            turn.message = message
            if turn.restore_end is not None:
                self._lower_end(message, turn.restore_end)

    @staticmethod
    def _lower_end(message: Any, end: float) -> None:
        metrics = getattr(message, "metrics", None)
        if isinstance(metrics, dict) and metrics.get("stopped_speaking_at", end) > end:
            metrics["stopped_speaking_at"] = end

    def lower_end(self, turn_id: int, end: float) -> None:
        """Bring a message end that was stretched over a late final's audio back down.

        Only ever lowers the value on the same metrics object; if the message is not there yet
        the value is applied when it is bound.
        """
        turn = self._turns.get(turn_id)
        if turn is None:
            return
        turn.restore_end = end if turn.restore_end is None else min(turn.restore_end, end)
        if turn.message is not None:
            self._lower_end(turn.message, turn.restore_end)

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
                if item is not None and id(item) in t.message_ids and not t.message_pushed
            ),
            None,
        )
        self._queue.append((name, arg))
        if owner is not None:
            owner.message_pushed = True
            owner.last_item = item
            for late in owner.pending_late:
                self._queue.append(late)
                owner.last_item = late[1].item
            owner.pending_late.clear()
            if not owner.released:
                owner.marker = _Marker(owner)
                self._queue.append(owner.marker)
        self._flush()

    def insert_late(self, turn_id: int, name: str, arg: Any) -> None:
        """Record a late final right after the turn's last item."""
        turn = self._turns.get(turn_id)
        if turn is None or (turn.released and turn.message_pushed):
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
        # a late final still waiting for the turn's own message stays behind it: the message
        # is on its way (it is added by a task), and the pair must come out in this order
        self._flush()
        if not self._queue:
            self._turns = {
                k: t for k, t in self._turns.items() if not (t.released and not t.pending_late)
            }

    def release_all(self) -> None:
        for turn_id in list(self._turns):
            self.release_turn(turn_id)
        # the session ends: a turn whose own message never came leaves its late finals last
        for turn in self._turns.values():
            self._queue.extend(turn.pending_late)
            turn.pending_late.clear()
        for entry in self._queue:
            if isinstance(entry, _Hold) and entry.timer is not None:
                entry.timer.cancel()
                entry.timer = None
        self._hold = None
        self._queue = [e for e in self._queue if not isinstance(e, _Marker)]
        self._flush()

    def _flush(self) -> None:
        if self._flushing:
            return
        self._flushing = True
        try:
            while self._queue:
                head = self._queue[0]
                if isinstance(head, _Marker):
                    break
                self._queue.pop(0)
                name, arg = head
                try:
                    self._emit(name, arg)
                except Exception:
                    logger.exception("error while emitting a held recording event")
        finally:
            self._flushing = False
