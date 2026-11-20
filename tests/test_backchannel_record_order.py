"""A backchannel spoken over an agent speech is recorded right behind that speech's message.

The agent's own message is recorded when its speech ends or is interrupted, so a backchannel
spoken while it played would otherwise be recorded ahead of it. Only the recording event waits;
the chat context is updated at once, and nothing is dropped when the session ends first.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

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


# --- the real chain: overlap verdict -> held transcript trim -> activity hook -> session ---------


def _real_chain(speech: SpeechHandle):
    """A real AudioRecognition gate and trim, a real activity hook and a real session."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from livekit.agents.voice.agent_activity import AgentActivity

    from .test_held_ignore_word_recorded import _recognition

    session = AgentSession()
    seen = _recorder(session)
    activity = AgentActivity.__new__(AgentActivity)
    activity._agent = SimpleNamespace(_chat_ctx=llm.ChatContext.empty(), stt=None)  # type: ignore[assignment]
    activity._session = session
    ar = _recognition()
    # every hook is a mock except the one the trim calls to record the word: the real activity's
    ar._hooks.on_held_ignore_word_discarded = activity.on_held_ignore_word_discarded
    # the speech is the running speech of the session
    session._activity = SimpleNamespace(current_speech=speech)  # type: ignore[assignment]
    return session, activity, ar, seen, MagicMock


async def _backchannel_over(
    ar,
    *,
    start: float,
    end: float,
    arrived: float,
    text: str,
    span: tuple[float, float] | None = None,
) -> None:
    from .test_held_ignore_word_recorded import (
        _backchannel_verdict,
        _final,
        _open_overlap,
    )

    _open_overlap(ar)
    _backchannel_verdict(ar)
    ev = _final(text, start=start, end=end, created_at=arrived)
    if span is not None:
        ev = replace(ev, transcript_span=span)
    await ar._on_stt_event(ev)


async def test_trimmed_backchannel_over_a_playing_speech_is_recorded_behind_its_message() -> None:
    from .test_held_ignore_word_recorded import (
        ACK_ARRIVED,
        ACK_END,
        ACK_START,
        AGENT_END,
        AGENT_START,
    )

    speech = SpeechHandle.create()
    speech._first_audio_at = AGENT_START
    session, _activity, ar, seen, _ = _real_chain(speech)

    await _backchannel_over(
        ar,
        start=ACK_START,
        end=ACK_END,
        arrived=ACK_ARRIVED,
        text=" 예.",
        span=(ACK_START, ACK_END),
    )
    ar._on_end_of_agent_speech(ended_at=AGENT_END)  # the trim records the word

    assert seen == []  # the speech is still not done, so its message is not recorded yet

    session._conversation_item_added(_msg("assistant", "안내 말씀"))  # the speech ends
    speech._mark_done()
    await asyncio.sleep(0)

    assert seen == [("assistant", "안내 말씀"), ("user", "예.")]


async def test_two_backchannels_over_one_paused_and_resumed_speech_are_recorded_once_each() -> None:
    from .test_held_ignore_word_recorded import (
        ACK_ARRIVED,
        ACK_END,
        ACK_START,
        AGENT_END,
        AGENT_START,
    )

    speech = SpeechHandle.create()
    speech._first_audio_at = AGENT_START  # the first audio; a resume does not change it
    session, _activity, ar, seen, _ = _real_chain(speech)

    await _backchannel_over(
        ar,
        start=ACK_START,
        end=ACK_END,
        arrived=ACK_ARRIVED,
        text=" 예.",
        span=(ACK_START, ACK_END),
    )
    ar._on_end_of_agent_speech(ended_at=AGENT_END)
    # the same speech resumes and the customer says it again
    ar2 = _real_chain(speech)[2]
    ar2._hooks.on_held_ignore_word_discarded = _activity.on_held_ignore_word_discarded
    await _backchannel_over(
        ar2,
        start=ACK_START + 1,
        end=ACK_END + 1,
        arrived=ACK_ARRIVED + 1,
        span=(ACK_START + 1, ACK_END + 1),
        text=" 네.",
    )
    ar2._on_end_of_agent_speech(ended_at=AGENT_END)
    assert seen == []

    session._conversation_item_added(_msg("assistant", "끝까지 말함"))
    speech._mark_done()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert seen == [("assistant", "끝까지 말함"), ("user", "예."), ("user", "네.")]
    assert session._deferred_backchannels == []


async def test_a_word_said_before_the_speech_audio_is_not_deferred() -> None:
    from .test_held_ignore_word_recorded import ACK_ARRIVED, ACK_END, ACK_START, AGENT_END

    speech = SpeechHandle.create()
    speech._first_audio_at = ACK_END + 1.0  # the audio started after the word ended
    session, _activity, ar, seen, _ = _real_chain(speech)

    await _backchannel_over(
        ar,
        start=ACK_START,
        end=ACK_END,
        arrived=ACK_ARRIVED,
        text=" 네.",
        span=(ACK_START, ACK_END),
    )
    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert seen == [("user", "네.")]  # recorded at once, ahead of the speech's message


async def test_close_after_the_running_speech_ends_records_the_backchannel_once() -> None:
    from livekit.agents.voice.events import CloseReason

    from .test_agent_session import _CancellingCloseActivity, _create_closing_test_session

    speech = SpeechHandle.create()
    holder: dict[str, AgentSession] = {}

    class _Activity(_CancellingCloseActivity):
        current_speech = speech

        async def interrupt(self, *, force: bool) -> None:
            # closing interrupts the running speech: its message is recorded, then it is done
            holder["s"]._conversation_item_added(_msg("assistant", "끊긴 말"))
            speech._mark_done()

        async def drain(self) -> None:
            return None

    session = _create_closing_test_session(_Activity())
    holder["s"] = session
    session._chat_ctx = llm.ChatContext.empty()  # type: ignore[attr-defined]
    session._conversation_item_added_after_speech(_msg("user", "네, 네"), speech)

    await asyncio.wait_for(session._aclose_impl(reason=CloseReason.USER_INITIATED), 5)

    recorded = [
        (c.args[1].item.role, c.args[1].item.text_content)
        for c in session.emit.call_args_list
        if c.args[0] == "conversation_item_added"
    ]
    assert recorded == [("assistant", "끊긴 말"), ("user", "네, 네")]
    names = [c.args[0] for c in session.emit.call_args_list]
    assert names.index("close") == len(names) - 1  # both were recorded before the close event
    await asyncio.sleep(0)
    assert [c.args[0] for c in session.emit.call_args_list].count("conversation_item_added") == 2


async def test_close_records_a_backchannel_whose_speech_never_finishes() -> None:
    from livekit.agents.voice.events import CloseReason

    from .test_agent_session import _CancellingCloseActivity, _create_closing_test_session

    class _Activity(_CancellingCloseActivity):
        async def drain(self) -> None:
            return None

    session = _create_closing_test_session(_Activity())
    session._chat_ctx = llm.ChatContext.empty()  # type: ignore[attr-defined]
    stuck = SpeechHandle.create()  # never done, and not the activity's current speech
    session._conversation_item_added_after_speech(_msg("user", "예"), stuck)

    await asyncio.wait_for(session._aclose_impl(reason=CloseReason.USER_INITIATED), 5)

    names = [c.args[0] for c in session.emit.call_args_list]
    assert names.count("conversation_item_added") == 1
    assert names.index("conversation_item_added") < names.index("close")
    stuck._mark_done()  # a late finish records nothing more
    await asyncio.sleep(0)
    assert [c.args[0] for c in session.emit.call_args_list].count("conversation_item_added") == 1


def _consuming_activity(speech: SpeechHandle):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from livekit.agents.voice.agent_activity import AgentActivity

    session = AgentSession()
    session._activity = SimpleNamespace(current_speech=speech)  # type: ignore[assignment]
    seen = _recorder(session)
    activity = AgentActivity.__new__(AgentActivity)
    activity._agent = SimpleNamespace(_chat_ctx=llm.ChatContext.empty(), stt=None)  # type: ignore[assignment]
    activity._session = session
    activity._cancel_preemptive_generation = MagicMock()  # type: ignore[method-assign]
    activity._clear_user_speech_interruption_context = MagicMock()  # type: ignore[method-assign]
    activity._init_metrics_from_end_of_turn = MagicMock(return_value={})  # type: ignore[method-assign]
    return session, activity, seen


def _turn(span: tuple[float, float] | None):
    from types import SimpleNamespace

    return SimpleNamespace(new_transcript="네", transcript_confidence=0.9, transcript_span=span)


async def test_ignored_turn_said_over_the_speech_waits_for_its_message() -> None:
    speech = SpeechHandle.create()
    speech._first_audio_at = 10.0
    session, activity, seen = _consuming_activity(speech)

    activity._consume_delayed_interruption_turn(  # said while the audio played (11.0 - 11.3)
        _turn((11.0, 11.3)), reason="interruption_ignore_words", during_agent_speech=True
    )
    assert seen == []

    session._conversation_item_added(_msg("assistant", "말씀"))
    speech._mark_done()
    await asyncio.sleep(0)

    assert seen == [("assistant", "말씀"), ("user", "네")]


@pytest.mark.parametrize("span", [(9.0, 9.3), None])
async def test_ignored_turn_said_before_the_audio_or_without_a_span_is_recorded_at_once(
    span: tuple[float, float] | None,
) -> None:
    speech = SpeechHandle.create()
    speech._first_audio_at = 10.0  # the audio started after the word (or the word has no span)
    session, activity, seen = _consuming_activity(speech)

    activity._consume_delayed_interruption_turn(
        _turn(span), reason="interruption_ignore_words", during_agent_speech=True
    )

    assert seen == [("user", "네")]


async def test_trim_judges_by_the_span_not_by_the_arrival_time() -> None:
    from .test_held_ignore_word_recorded import ACK_ARRIVED, ACK_END, ACK_START, AGENT_END

    # the word ended (span end) before the speech's first audio, but its final arrived after it
    speech = SpeechHandle.create()
    speech._first_audio_at = ACK_ARRIVED - 0.5
    assert ACK_END < speech._first_audio_at < ACK_ARRIVED
    session, _activity, ar, seen, _ = _real_chain(speech)

    await _backchannel_over(
        ar,
        start=ACK_START,
        end=ACK_END,
        arrived=ACK_ARRIVED,
        text=" 네.",
        span=(ACK_START, ACK_END),
    )
    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert seen == [("user", "네.")]


@pytest.mark.parametrize("unknown", ["not_given", "none"])
async def test_trim_without_a_proven_span_is_recorded_at_once(unknown: str) -> None:
    from livekit.agents.types import NOT_GIVEN

    from .test_held_ignore_word_recorded import (
        ACK_ARRIVED,
        ACK_END,
        ACK_START,
        AGENT_END,
        AGENT_START,
        _backchannel_verdict,
        _final,
        _open_overlap,
    )

    speech = SpeechHandle.create()
    speech._first_audio_at = AGENT_START
    session, _activity, ar, seen, _ = _real_chain(speech)
    _open_overlap(ar)
    _backchannel_verdict(ar)
    ev = _final(" 네.", start=ACK_START, end=ACK_END, created_at=ACK_ARRIVED)
    ev = replace(ev, transcript_span=NOT_GIVEN if unknown == "not_given" else None)
    await ar._on_stt_event(ev)

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert seen == [
        ("user", "네.")
    ]  # the arrival time is after the first audio, but proves nothing


# --- a word that ended before the agent's first audio, still unconfirmed when its row is due ---


def _speaking(session: AgentSession, first_audio: float) -> SpeechHandle:
    speech = SpeechHandle.create()
    speech._first_audio_at = first_audio
    session._activity = type("A", (), {"current_speech": speech})()  # type: ignore[assignment]
    return speech


def _user(text: str, span: tuple[float, float] | None) -> llm.ChatMessage:
    msg = _msg("user", text)
    msg.metrics = (  # type: ignore[typeddict-item]
        {} if span is None else {"started_speaking_at": span[0], "stopped_speaking_at": span[1]}
    )
    return msg


def _held(first_audio: float = 10.0, word: tuple[float, float] = (9.4, 9.7)):
    session = AgentSession()
    seen = _recorder(session)
    _speaking(session, first_audio=first_audio)
    session._note_user_word_ended(*word)
    session._conversation_item_added(_msg("assistant", "안내"))
    return session, seen


async def test_agent_row_waits_for_a_word_that_ended_before_its_first_audio() -> None:
    session, seen = _held()
    assert seen == []  # held, not recorded
    session._conversation_item_added(_user("네", (9.4, 9.7)))

    assert seen == [("user", "네"), ("assistant", "안내")]
    assert not session._record_gate.hold_active


async def test_agent_row_goes_out_after_the_hold_when_no_word_is_confirmed() -> None:
    session, seen = _held()
    await asyncio.sleep(1.1)
    assert seen == [("assistant", "안내")]  # once, after at most 1 s
    session._conversation_item_added(_user("네", (9.4, 9.7)))
    assert seen == [("assistant", "안내"), ("user", "네")]


async def test_the_hold_is_for_the_segment_not_for_any_row() -> None:
    session, seen = _held()
    session._conversation_item_added(_user("다른 말", None))  # span unknown
    session._conversation_item_added(_user("먼 말", (3.0, 3.5)))  # another, earlier segment
    assert seen == []  # still waiting for the segment
    session._conversation_item_added(_user("네", (9.4, 9.7)))
    assert seen == [
        ("user", "네"),
        ("assistant", "안내"),
        ("user", "다른 말"),
        ("user", "먼 말"),
    ]


async def test_a_newer_segment_does_not_replace_the_waiting_one() -> None:
    session = AgentSession()
    seen = _recorder(session)
    _speaking(session, first_audio=10.0)
    session._note_user_word_ended(9.4, 9.7)
    session._note_user_word_ended(11.0, 11.2)  # after the agent started
    session._conversation_item_added(_msg("assistant", "안내"))
    session._conversation_item_added(_user("네", (9.4, 9.7)))
    assert seen == [("user", "네"), ("assistant", "안내")]


async def test_an_empty_segment_of_another_start_does_not_release_the_hold() -> None:
    session, seen = _held()
    session._clear_pending_user_word(2.0)  # some other segment had no text
    assert seen == []
    session._clear_pending_user_word(9.4)  # the waited-for segment had no text
    assert seen == [("assistant", "안내")]


async def test_a_segment_ending_when_the_agent_starts_is_not_waited_for() -> None:
    session = AgentSession()
    seen = _recorder(session)
    _speaking(session, first_audio=10.0)
    session._note_user_word_ended(9.8, 10.0)  # same instant: said over the agent, not before
    session._conversation_item_added(_msg("assistant", "안내"))
    assert seen == [("assistant", "안내")]


async def test_a_segment_long_before_the_agent_is_still_waited_for() -> None:
    session = AgentSession()
    seen = _recorder(session)
    _speaking(session, first_audio=20.0)
    session._note_user_word_ended(9.4, 9.7)  # confirmation is late, the agent started 10 s later
    session._conversation_item_added(_msg("assistant", "안내"))
    assert seen == []
    session._conversation_item_added(_user("네", (9.4, 9.7)))
    assert seen == [("user", "네"), ("assistant", "안내")]


async def test_no_hold_without_a_pending_word() -> None:
    session = AgentSession()
    seen = _recorder(session)
    _speaking(session, first_audio=10.0)
    session._conversation_item_added(_msg("assistant", "하나"))
    assert seen == [("assistant", "하나")]


async def test_a_word_said_over_the_agent_does_not_jump_ahead_of_the_held_row() -> None:
    session, seen = _held()
    session._conversation_item_added(_user("네", (10.2, 10.4)))  # said while the agent talked
    await asyncio.sleep(1.1)
    assert seen == [("assistant", "안내"), ("user", "네")]


async def test_close_records_a_held_row_once() -> None:
    session, seen = _held()
    session._record_gate.release_all()
    await asyncio.sleep(1.1)
    assert seen == [("assistant", "안내")]


async def test_a_late_final_of_the_same_turn_follows_its_row_ahead_of_the_held_row() -> None:
    session = AgentSession()
    seen = _recorder(session)
    gate = session._record_gate
    _speaking(session, first_audio=10.0)
    session._note_user_word_ended(9.4, 9.7)
    first = _user("네", (9.4, 9.7))
    gate.open_turn(1)
    gate.bind_message(1, first)
    session._conversation_item_added(_msg("assistant", "안내"))  # held
    session._conversation_item_added(first)  # the turn's own row, ahead of the held one
    late = _user("그리고", (9.7, 9.9))
    session._chat_ctx.insert(late)
    session._late_final_item_added(late, turn_id=1)
    gate.release_turn(1)

    assert seen == [("user", "네"), ("user", "그리고"), ("assistant", "안내")]
    await asyncio.sleep(1.1)
    session._record_gate.release_all()
    assert seen == [("user", "네"), ("user", "그리고"), ("assistant", "안내")]  # once each
