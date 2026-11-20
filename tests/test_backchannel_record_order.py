"""A backchannel spoken over an agent speech is recorded right behind that speech's message.

The agent's own message is recorded when its speech ends or is interrupted, so a backchannel
spoken while it played would otherwise be recorded ahead of it. Only the recording event waits;
the chat context is updated at once, and nothing is dropped when the session ends first.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import AgentSession, llm
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


def _recorder(session: AgentSession) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []

    @session.on("conversation_item_added")
    def _on(ev: ConversationItemAddedEvent) -> None:
        seen.append((ev.item.role, ev.item.text_content or ""))

    return seen


def _msg(role: str, text: str) -> llm.ChatMessage:
    return llm.ChatMessage(role=role, content=[text])  # type: ignore[arg-type]


async def test_backchannel_over_a_speech_is_recorded_behind_the_speech_message() -> None:
    session = AgentSession()
    seen = _recorder(session)
    speech = SpeechHandle.create()

    session._conversation_item_added_after_speech(_msg("user", "예, 예, 예"), speech)

    assert seen == []  # waits for the speech
    assert any(i.text_content == "예, 예, 예" for i in session._chat_ctx.items)  # state is not held

    session._conversation_item_added(_msg("assistant", "안내 말씀"))  # the speech ends: its message
    speech._mark_done()
    await asyncio.sleep(0)  # done callbacks run on the next loop turn

    assert seen == [("assistant", "안내 말씀"), ("user", "예, 예, 예")]


async def test_paused_and_resumed_speech_is_one_speech() -> None:
    session = AgentSession()
    seen = _recorder(session)
    speech = SpeechHandle.create()

    session._conversation_item_added_after_speech(_msg("user", "예"), speech)  # while paused
    # the speech resumes and plays on: same handle, still not done
    session._conversation_item_added_after_speech(_msg("user", "네"), speech)
    assert seen == []

    session._conversation_item_added(_msg("assistant", "끝까지 말함"))
    speech._mark_done()
    await asyncio.sleep(0)  # done callbacks run on the next loop turn

    assert seen == [("assistant", "끝까지 말함"), ("user", "예"), ("user", "네")]


async def test_without_a_running_speech_it_is_recorded_at_once() -> None:
    session = AgentSession()
    seen = _recorder(session)
    done = SpeechHandle.create()
    done._mark_done()

    session._conversation_item_added_after_speech(_msg("user", "네"), None)
    session._conversation_item_added_after_speech(_msg("user", "예"), done)

    assert seen == [("user", "네"), ("user", "예")]


async def test_a_backchannel_still_waiting_when_the_session_ends_is_recorded() -> None:
    session = AgentSession()
    seen = _recorder(session)
    speech = SpeechHandle.create()

    session._conversation_item_added_after_speech(_msg("user", "네, 네"), speech)
    assert seen == []

    session._flush_deferred_backchannels()  # what closing the session does

    assert seen == [("user", "네, 네")]
    assert session._deferred_backchannels == []


async def test_other_speeches_are_not_flushed_early() -> None:
    session = AgentSession()
    seen = _recorder(session)
    first, second = SpeechHandle.create(), SpeechHandle.create()

    session._conversation_item_added_after_speech(_msg("user", "예"), first)
    session._conversation_item_added_after_speech(_msg("user", "네"), second)
    first._mark_done()
    await asyncio.sleep(0)

    assert seen == [("user", "예")]
    second._mark_done()
    await asyncio.sleep(0)
    assert seen == [("user", "예"), ("user", "네")]
