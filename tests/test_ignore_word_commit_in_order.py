"""An ignore-word turn that ends while the agent still speaks is recorded where it was spoken.

Sequence: the agent asks a long interruptible question, the customer says "네." over it (an
ignore word, so it does not interrupt), and the turn ends while the question is still playing.
The ack must become its own user message right then, without a reply and without interrupting
the agent. Before the fix the turn was dropped but its transcript stayed in the recognition
buffer, so it was prepended, with its start time, to the customer's next turn.

An ack the customer keeps talking through ("네. 그런데 ...") is still one turn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent
from livekit.agents.voice.agent_activity import DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = pytest.mark.unit

SPEED = 5.0
SESSION_TIMEOUT = 120.0
IGNORE_WORDS = ["네", "예"]

# The fake session replays the customer's audio and transcripts on the wall clock, while the
# agent's speech starts and ends whenever its fake LLM/TTS pipeline gets scheduled. Every
# scenario keeps real-time slack (SLACK_REAL seconds) between the events whose order matters
# and polls for the record it expects. Times are scripted seconds (SPEED pass per real second).
#
#   AI_START ... AI_END   the question is audible (about; a pipeline lag shifts both later)
#   ack                   starts SLACK after AI_START; its turn ends well before AI_END
#   answer                starts SLACK after AI_END
SLACK_REAL = 1.5
SLACK = SLACK_REAL * SPEED
QUESTION_DURATION = 20.0
AI_START = 3.3  # the question becomes audible about here (the greeting turn ends at 2.5)
AI_END = AI_START + QUESTION_DURATION
ACK_START = AI_START + SLACK
ACK_END = ACK_START + 0.6
ANSWER_START = AI_END + SLACK
ANSWER_END = ANSWER_START + 2.0
WAIT_TIMEOUT = 40.0  # real seconds to wait for an expected record

GREETING = "Tell me about the rooms."
QUESTION = "Are both rooms still vacant?"
ACK = "네."
ANSWER = "그건 괜찮아요."
REPLY_TO_ANSWER = "Great, I will register them."
# what the customer's answer turn looked like before the fix
MERGED = f"{ACK} {ANSWER}"
REPLY_TO_MERGED = "Sorry, are both rooms still vacant?"
CONTINUATION = "그런데 하나 더 물어볼게요."
ACK_THEN_CONTINUATION = f"{ACK} {CONTINUATION}"
REPLY_TO_CONTINUATION = "Sure, go ahead."


async def _run(
    actions: FakeActions,
    *,
    until: Callable[[Agent, list[ConversationItemAddedEvent]], bool],
    settle: float = 1.5,
) -> tuple[Agent, list[ConversationItemAddedEvent]]:
    session = create_session(
        actions,
        speed_factor=SPEED,
        # the consumer's interruption gate: VAD alone never interrupts, words have to
        turn_handling={"interruption": {"min_words": 1}},
        extra_kwargs={"interruption_ignore_words": IGNORE_WORDS},
    )
    agent = Agent(instructions="You are a helpful assistant.")
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)
    await asyncio.wait_for(
        run_session(
            session,
            agent,
            drain_delay=settle,
            until=lambda: until(agent, items),
            until_timeout=WAIT_TIMEOUT,
        ),
        timeout=SESSION_TIMEOUT,
    )
    return agent, items


def _messages(agent: Agent) -> list[Any]:
    return [
        item
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role in ("user", "assistant")
    ]


def _assistant_messages(agent: Agent) -> list[Any]:
    return [m for m in _messages(agent) if m.role == "assistant"]


def _has_two_assistant_messages(agent: Agent, _items: list[ConversationItemAddedEvent]) -> bool:
    return len(_assistant_messages(agent)) >= 2


def _question_with_greeting() -> FakeActions:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, GREETING)
    actions.add_llm(QUESTION)
    actions.add_tts(QUESTION_DURATION)
    return actions


def _ack_during_question_then_answer() -> FakeActions:
    actions = _question_with_greeting()
    actions.add_user_speech(ACK_START, ACK_END, ACK)
    actions.add_user_speech(ANSWER_START, ANSWER_END, ANSWER)
    actions.add_llm(REPLY_TO_ANSWER, input=ANSWER)
    actions.add_tts(1.0, input=REPLY_TO_ANSWER)
    actions.add_llm(REPLY_TO_MERGED, input=MERGED)
    actions.add_tts(1.0, input=REPLY_TO_MERGED)
    return actions


async def test_ignore_word_during_agent_speech_is_recorded_in_spoken_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    agent, items = await _run(_ack_during_question_then_answer(), until=_has_two_assistant_messages)

    messages = _messages(agent)
    assert [(m.role, m.text_content) for m in messages] == [
        ("user", GREETING),
        ("assistant", QUESTION),
        ("user", ACK),
        ("user", ANSWER),
        ("assistant", REPLY_TO_ANSWER),
    ]
    question, ack, answer = messages[1], messages[2], messages[3]

    # the ack neither interrupted the question nor got a reply of its own
    assert question.interrupted is False
    assert len(_assistant_messages(agent)) == 2
    assert ack.extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY) is True
    assert not answer.extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY)

    # the ack was recorded while the question was still playing, at its own speech timing,
    # and the answer kept its own start instead of inheriting the ack's
    ack_events = [ev.item for ev in items if ev.item is ack]
    assert ack_events, "the ack was not emitted as a conversation item"
    assert [ev.item for ev in items].index(ack) < [ev.item for ev in items].index(question)
    assert question.created_at < ack.created_at < answer.created_at
    ack_started = ack.metrics["started_speaking_at"]
    answer_started = answer.metrics["started_speaking_at"]
    assert ack_started < ack.metrics["stopped_speaking_at"] < answer_started
    # scripted gap between the two starts is (ANSWER_START - ACK_START) / SPEED real seconds
    assert answer_started - ack_started > 0.5 * (ANSWER_START - ACK_START) / SPEED

    (record,) = [
        r for r in caplog.records if r.message == "consuming ignored interruption transcript"
    ]
    assert record.reason == "interruption_ignore_words"
    assert record.user_input == ACK
    assert not [
        r for r in caplog.records if r.message == "consuming delayed interruption transcript"
    ]


def _ack_then_continuation_during_question() -> FakeActions:
    actions = _question_with_greeting()
    # the customer keeps talking right after the ack, and the ack's final arrives while they
    # still speak: the new speech cancels the pending end of turn, so none falls between them
    actions.add_user_speech(ACK_START, ACK_END, ACK, stt_delay=1.5)
    actions.add_user_speech(ACK_END + 0.2, ACK_END + 2.2, CONTINUATION)
    actions.add_llm(REPLY_TO_CONTINUATION, input=ACK_THEN_CONTINUATION)
    actions.add_tts(1.0, input=REPLY_TO_CONTINUATION)
    return actions


async def test_ack_continued_by_more_words_stays_one_turn() -> None:
    agent, items = await _run(
        _ack_then_continuation_during_question(), until=_has_two_assistant_messages
    )

    messages = _messages(agent)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0].text_content == GREETING
    assert messages[2].text_content == ACK_THEN_CONTINUATION
    assert messages[3].text_content == REPLY_TO_CONTINUATION
    # the words after the ack make it a real turn: it interrupts the question and gets a reply
    assert messages[1].interrupted is True
    assert QUESTION.startswith(messages[1].text_content)
    assert not any(
        ev.item.extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY)
        for ev in items
        if ev.item.type == "message"
    )


def _ack_only_during_question(ack: str = ACK) -> FakeActions:
    actions = _question_with_greeting()
    actions.add_user_speech(ACK_START, ACK_END, ack)
    actions.add_llm(REPLY_TO_MERGED, input=ack)
    actions.add_tts(1.0, input=REPLY_TO_MERGED)
    return actions


async def test_ignore_word_alone_never_interrupts_or_replies() -> None:
    ack = ACK

    def _question_finished(agent: Agent, _items: list[ConversationItemAddedEvent]) -> bool:
        return bool(_assistant_messages(agent))

    agent, _ = await _run(_ack_only_during_question(ack), until=_question_finished, settle=3.0)

    messages = _messages(agent)
    assert [(m.role, m.text_content) for m in messages] == [
        ("user", GREETING),
        ("assistant", QUESTION),
        ("user", ack),
    ]
    assert messages[1].interrupted is False
    assert messages[2].extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY) is True


SINGLE_CHARACTER = "3"
SINGLE_THEN_ANSWER = f"{SINGLE_CHARACTER} {ANSWER}"
REPLY_TO_SINGLE_THEN_ANSWER = "Three, noted."


def _single_character_during_question_then_answer() -> FakeActions:
    actions = _question_with_greeting()
    actions.add_user_speech(ACK_START, ACK_END, SINGLE_CHARACTER)
    actions.add_user_speech(ANSWER_START, ANSWER_END, ANSWER)
    actions.add_llm(REPLY_TO_SINGLE_THEN_ANSWER, input=SINGLE_THEN_ANSWER)
    actions.add_tts(1.0, input=REPLY_TO_SINGLE_THEN_ANSWER)
    actions.add_llm(REPLY_TO_ANSWER, input=ANSWER)
    actions.add_tts(1.0, input=REPLY_TO_ANSWER)
    return actions


async def test_single_character_over_agent_speech_is_not_consumed() -> None:
    # a single character only loosely matches the ignore words: it may be a real answer, so
    # it keeps accumulating into the next turn instead of being recorded without a reply
    agent, items = await _run(
        _single_character_during_question_then_answer(), until=_has_two_assistant_messages
    )

    messages = _messages(agent)
    assert [(m.role, m.text_content) for m in messages] == [
        ("user", GREETING),
        ("assistant", QUESTION),
        ("user", SINGLE_THEN_ANSWER),
        ("assistant", REPLY_TO_SINGLE_THEN_ANSWER),
    ]
    assert not any(
        ev.item.extra.get(DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY)
        for ev in items
        if ev.item.type == "message"
    )
