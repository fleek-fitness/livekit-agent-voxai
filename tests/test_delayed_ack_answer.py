"""A delayed ignore-word acknowledgement that answers a finished agent question is a reply turn.

Reproduces the sequence that leads to dead air: the customer starts "예, 예" while the AI
is still asking a question, the AI finishes normally, and the final transcript arrives afterwards.
Ignore words still never interrupt the AI while it is speaking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent
from livekit.agents.voice.agent_activity import (
    DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY,
    AgentActivity,
)
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = pytest.mark.unit

SPEED = 5.0
SESSION_TIMEOUT = 60.0
IGNORE_WORDS = ["네", "예", "안녕하세요"]


class _Agent(Agent):
    def __init__(self, *, allow_delayed_ack: bool) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self._allow_delayed_ack = allow_delayed_ack

    def delayed_ack_answer_allowed(self) -> bool:
        return self._allow_delayed_ack


async def _run(
    actions: FakeActions, *, allow_delayed_ack: bool
) -> tuple[_Agent, list[ConversationItemAddedEvent], list[Any]]:
    session = create_session(
        actions,
        speed_factor=SPEED,
        # the consumer's interruption gate: VAD alone never interrupts, words have to
        turn_handling={"interruption": {"min_words": 1}},
        extra_kwargs={"interruption_ignore_words": IGNORE_WORDS},
    )
    agent = _Agent(allow_delayed_ack=allow_delayed_ack)
    items: list[ConversationItemAddedEvent] = []
    playbacks: list[Any] = []
    session.on("conversation_item_added", items.append)
    session.output.audio.on("playback_finished", playbacks.append)
    await asyncio.wait_for(run_session(session, agent, drain_delay=2.0), timeout=SESSION_TIMEOUT)
    return agent, items, playbacks


def _assistant_messages(agent: Agent) -> list[Any]:
    return [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "assistant"
    ]


def _question_then_ack(*, ack_start: float, ack_end: float, ack_text: str) -> FakeActions:
    """AI asks a 2 s question (audible roughly 3.5 s - 5.5 s); the customer acks over its end."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Are both rooms still vacant?")
    actions.add_llm("Are both rooms still vacant?")
    actions.add_tts(2.0)
    actions.add_user_speech(ack_start, ack_end, ack_text, stt_delay=0.3)
    actions.add_llm("Great, I will register them as they are.")
    actions.add_tts(1.0)
    return actions


def _decisions(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [r for r in caplog.records if r.message == "delayed_ack_as_answer"]


async def test_delayed_ack_after_finished_question_is_answered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    # ack starts while the AI still speaks, the AI finishes normally, the customer keeps
    # talking well past the AI's end, and the final arrives afterwards
    actions = _question_then_ack(ack_start=4.5, ack_end=7.0, ack_text="예, 예")
    agent, _, _ = await _run(actions, allow_delayed_ack=True)

    assert len(_assistant_messages(agent)) == 2
    assert _assistant_messages(agent)[0].interrupted is False
    (decision,) = _decisions(caplog)
    assert decision.decision is True
    assert not [r for r in caplog.records if r.message.startswith("consuming delayed")]


async def test_delayed_ack_without_opt_in_is_recorded_without_reply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    actions = _question_then_ack(ack_start=4.5, ack_end=7.0, ack_text="예, 예")
    agent, items, _ = await _run(actions, allow_delayed_ack=False)

    # old behaviour: no reply, and the user message is recorded with the consumed marker
    assert len(_assistant_messages(agent)) == 1
    consumed = [
        ev.item
        for ev in items
        if ev.item.type == "message"
        and ev.item.role == "user"
        and ev.item.extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY)
    ]
    assert len(consumed) == 1
    assert not _decisions(caplog)  # the agent never opted in: no predicate evaluation at all


async def test_consumed_marker_absent_when_answered() -> None:
    actions = _question_then_ack(ack_start=4.5, ack_end=7.0, ack_text="예, 예")
    _, items, _ = await _run(actions, allow_delayed_ack=True)

    ack_messages = [ev.item for ev in items if ev.item.type == "message" and ev.item.role == "user"]
    assert ack_messages
    assert not any(m.extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY) for m in ack_messages)


@pytest.mark.parametrize("ack_text", ["예", "안녕하세요"])
async def test_echo_like_ack_ending_with_agent_speech_is_not_answered(
    ack_text: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    # the customer's utterance (a backchannel or a greeting said over the AI's greeting) ends
    # before the AI does: it must not trigger a second reply
    actions = _question_then_ack(ack_start=4.0, ack_end=5.2, ack_text=ack_text)
    agent, _, _ = await _run(actions, allow_delayed_ack=True)

    assert len(_assistant_messages(agent)) == 1
    (decision,) = _decisions(caplog)
    assert decision.decision is False
    assert decision.reasons == ["user_audio_ended_with_agent"]


async def test_ignore_word_during_agent_speech_does_not_interrupt() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story for you ... the end.")
    actions.add_tts(6.0)
    # ack fully inside the AI's playout
    actions.add_user_speech(4.5, 5.5, "예", stt_delay=0.3)
    agent, _, playbacks = await _run(actions, allow_delayed_ack=True)

    assert playbacks[-1].interrupted is False
    messages = _assistant_messages(agent)
    assert len(messages) == 1
    assert messages[0].interrupted is False


def _stub_activity(**overrides: Any) -> AgentActivity:
    """An activity in the post-condition state of that timeline (predicate inputs only)."""
    finished_speech = SimpleNamespace(done=lambda: True, interrupted=False)
    activity = object.__new__(AgentActivity)
    state: dict[str, Any] = {
        "_agent": SimpleNamespace(delayed_ack_answer_allowed=lambda: True),
        "_session": SimpleNamespace(agent_state="listening", user_state="listening"),
        "_delayed_interruption_speech": finished_speech,
        "_last_finished_speech": (finished_speech, 100.0),
        "_last_user_speech_end_time": 101.0,
        "_speech_q": [],
        "_current_speech": None,
        "_background_speeches": set(),
    }
    state.update(overrides)
    for key, value in state.items():
        setattr(activity, key, value)
    return activity


def _info(transcript: str = "예, 예") -> Any:
    return SimpleNamespace(new_transcript=transcript)


def test_predicate_true_for_completed_speech_and_later_user_audio() -> None:
    assert _stub_activity()._delayed_ack_is_answer(_info()) is True


def _interrupted() -> dict[str, Any]:
    speech = SimpleNamespace(done=lambda: True, interrupted=True)
    return {"_delayed_interruption_speech": speech, "_last_finished_speech": (speech, 100.0)}


_NEGATIVE_CASES = {
    "agent_speech_interrupted": _interrupted(),
    "agent_speech_not_finished": {
        "_delayed_interruption_speech": SimpleNamespace(done=lambda: False, interrupted=False)
    },
    "speech_not_tracked": {"_delayed_interruption_speech": None},
    "queued_speech": {"_speech_q": [(0, 0, object())]},
    "current_speech_running": {
        "_current_speech": SimpleNamespace(done=lambda: False, interrupted=False)
    },
    "background_speech": {"_background_speeches": {object()}},
    "agent_thinking": {"_session": SimpleNamespace(agent_state="thinking", user_state="listening")},
    "user_still_speaking": {
        "_session": SimpleNamespace(agent_state="listening", user_state="speaking")
    },
    "user_ended_with_agent": {"_last_user_speech_end_time": 100.2},
    "user_end_unknown": {"_last_user_speech_end_time": None},
    "agent_vetoes": {"_agent": SimpleNamespace(delayed_ack_answer_allowed=lambda: False)},
}


@pytest.mark.parametrize("name", list(_NEGATIVE_CASES))
def test_predicate_false(name: str) -> None:
    assert _stub_activity(**_NEGATIVE_CASES[name])._delayed_ack_is_answer(_info()) is False


@pytest.mark.parametrize("transcript", ["", "   ", "...", "?"])
def test_predicate_false_for_empty_transcript(transcript: str) -> None:
    assert _stub_activity()._delayed_ack_is_answer(_info(transcript)) is False


def test_predicate_false_when_hook_raises() -> None:
    def boom() -> bool:
        raise RuntimeError("boom")

    activity = _stub_activity(_agent=SimpleNamespace(delayed_ack_answer_allowed=boom))
    assert activity._delayed_ack_is_answer(_info()) is False


def test_predicate_logs_decision_and_reasons(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    _stub_activity(_last_user_speech_end_time=100.2)._delayed_ack_is_answer(_info())
    record = next(r for r in caplog.records if r.message == "delayed_ack_as_answer")
    assert record.decision is False
    assert record.reasons == ["user_audio_ended_with_agent"]
    assert not hasattr(record, "user_input")


def _end_of_speech_activity() -> AgentActivity:
    activity = object.__new__(AgentActivity)
    activity._audio_recognition = None
    activity._session = MagicMock()
    activity._user_silence_event = asyncio.Event()
    activity._stt_eos_received = False
    activity._dynamic_interruption = MagicMock()
    activity._paused_speech = None
    activity._last_user_speech_end_time = None
    return activity


def test_vad_end_of_speech_records_user_audio_end() -> None:
    activity = _end_of_speech_activity()
    vad_event = SimpleNamespace(silence_duration=0.5, inference_duration=0.0)
    before = time.time()
    activity.on_end_of_speech(vad_event)  # type: ignore[arg-type]
    assert activity._last_user_speech_end_time is not None
    assert activity._last_user_speech_end_time <= time.time() - 0.5 + 0.05
    assert activity._last_user_speech_end_time >= before - 0.5 - 0.05


def test_stt_only_end_of_speech_is_not_user_audio_timing() -> None:
    # turn_detection="stt": the end-of-speech event carries only its arrival time
    activity = _end_of_speech_activity()
    activity.on_end_of_speech(None)
    assert activity._last_user_speech_end_time is None

    # and it never erases a VAD end already recorded for the same segment
    activity._last_user_speech_end_time = 123.0
    activity.on_end_of_speech(None)
    assert activity._last_user_speech_end_time == 123.0
