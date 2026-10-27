"""An ignore word that the adaptive transcript gate trims is still recorded where it was spoken.

Sequence: the agent reads a long notice, the customer says "네." over it, the
adaptive detector judges the overlap a backchannel (is_interruption=false), and the final arrives
while the transcript gate still holds STT events. When the agent stops, the gate trims every held
event that ended during the agent's speech, so the "네." never reached on_end_of_turn and was
lost from the record. The trim now hands such an ignore-word final to the hooks, which record it
as a user message at its speech time, without a turn or a reply.
"""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from livekit.agents import llm
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.agent_activity import (
    DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY,
    AgentActivity,
)
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit

IGNORE_WORDS = ["네", "예"]

# Synthetic wall-clock seconds; STT input opens at INPUT_START.
INPUT_START = 100.0
AGENT_START = 146.17
OVERLAP_START = 148.91
VERDICT_AT = 149.46
ACK_START = INPUT_START + 47.42
ACK_END = INPUT_START + 47.72
ACK_ARRIVED = 151.63
AGENT_END = 161.05


class _RecordingChan:
    closed = False

    def __init__(self) -> None:
        self.sent: list[object] = []

    def send_nowait(self, item: object) -> None:
        self.sent.append(item)


def _recognition() -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._session.options.interruption_ignore_words = IGNORE_WORDS
    ar._stt_pipeline = SimpleNamespace(input_started_at=INPUT_START)  # type: ignore[assignment]
    ar._stt_aligned_transcript = True
    ar._stt_request_ids = []
    ar._turn_detection_mode = "vad"
    ar._user_turn_committed = False
    ar._end_of_turn_task = None
    ar._turn_transcript_received = False
    ar._transcription_timeout_handle = None
    ar._vad_speech_started = False
    ar._interruption_enabled = True
    ar._interruption_ch = _RecordingChan()  # type: ignore[assignment]
    ar._agent_speaking = False
    ar._agent_speech_started_at = None
    ar._transcript_gate_active = False
    ar._active_vad_speech_started_at = None
    ar._endpointing = MagicMock()
    ar._backchannel_boundary = None
    ar._backchannel_boundary_timer = None
    ar._backchannel_boundary_callback = None
    ar._overlap_in_current_turn = False
    ar._overlap_open = False
    ar._turn_backchannel_over_agent = False
    ar._transcript_buffer = deque()
    ar._tasks = set()
    ar._user_silence_ev = asyncio.Event()
    ar._user_silence_ev.set()
    ar._late_ack_waiting = False
    ar._hooks = MagicMock()
    ar._hooks.interruption_by_audio_activity_enabled = False
    ar._process_stt_event = MagicMock()  # type: ignore[method-assign]
    return ar


def _final(text: str, *, start: float, end: float, created_at: float) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(
                language="ko",
                text=text,
                start_time=start - INPUT_START,
                end_time=end - INPUT_START,
                confidence=0.9,
            )
        ],
        created_at=created_at,
    )


def _open_overlap(ar: AudioRecognition) -> None:
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    ar._on_start_of_speech(started_at=OVERLAP_START)
    ar._active_vad_speech_started_at = OVERLAP_START
    assert ar._transcript_gate_active


def _backchannel_verdict(ar: AudioRecognition) -> None:
    ar._active_vad_speech_started_at = None  # the customer stopped before the verdict
    ar._on_overlap_speech_event(
        OverlappingSpeechEvent(
            is_interruption=False,
            agent_ended=False,
            detected_at=VERDICT_AT,
            overlap_started_at=OVERLAP_START,
        )
    )


def _recorded(ar: AudioRecognition) -> list[tuple[str, float, float]]:
    hook = ar._hooks.on_held_ignore_word_discarded  # type: ignore[attr-defined]
    return [
        (
            call.args[0].alternatives[0].text,
            call.kwargs["started_speaking_at"],
            call.kwargs["stopped_speaking_at"],
        )
        for call in hook.call_args_list
    ]


async def test_ack_held_after_backchannel_verdict_is_recorded_when_agent_stops() -> None:
    ar = _recognition()
    _open_overlap(ar)
    _backchannel_verdict(ar)

    # the final arrives after the verdict: the gate still holds it
    await ar._on_stt_event(_final(" 네.", start=ACK_START, end=ACK_END, created_at=ACK_ARRIVED))
    assert len(ar._transcript_buffer) == 1
    assert _recorded(ar) == []

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [(" 네.", ACK_START, ACK_END)]
    # recorded, not replayed as a turn
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]
    assert not ar._transcript_buffer
    assert not ar._transcript_gate_active


async def test_ack_held_before_backchannel_verdict_is_recorded_at_the_verdict() -> None:
    ar = _recognition()
    _open_overlap(ar)
    # a fast final: held before the verdict, trimmed by the verdict itself
    await ar._on_stt_event(
        _final("네.", start=OVERLAP_START, end=OVERLAP_START + 0.3, created_at=OVERLAP_START + 0.4)
    )

    _backchannel_verdict(ar)

    assert _recorded(ar) == [("네.", OVERLAP_START, OVERLAP_START + 0.3)]
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    assert not ar._transcript_buffer

    ar._on_end_of_agent_speech(ended_at=AGENT_END)
    assert len(_recorded(ar)) == 1  # not recorded twice


async def test_trimmed_non_ignore_word_backchannel_is_still_dropped() -> None:
    ar = _recognition()
    _open_overlap(ar)
    _backchannel_verdict(ar)
    await ar._on_stt_event(
        _final("아 그렇구나", start=ACK_START, end=ACK_END, created_at=ACK_ARRIVED)
    )

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == []
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    assert not ar._transcript_buffer


async def test_ack_still_spoken_at_agent_end_is_replayed_not_recorded() -> None:
    ar = _recognition()
    _open_overlap(ar)
    _backchannel_verdict(ar)
    # spoken right as the agent stops: it ends after the trim boundary and stays a turn
    late_ack = _final("네.", start=AGENT_END - 0.2, end=AGENT_END + 0.1, created_at=AGENT_END + 0.3)
    ar._transcript_buffer.append(late_ack)

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == []
    ar._process_stt_event.assert_called_once_with(late_ack)  # type: ignore[attr-defined]


async def test_ack_without_word_timing_uses_its_arrival_time() -> None:
    ar = _recognition()
    ar._stt_aligned_transcript = False
    _open_overlap(ar)
    _backchannel_verdict(ar)
    untimed = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text="네.")],
        created_at=ACK_ARRIVED,
    )
    ar._transcript_buffer.append(untimed)

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", ACK_ARRIVED, ACK_ARRIVED)]


def _activity(chat_ctx: llm.ChatContext) -> tuple[AgentActivity, list[llm.ChatItem]]:
    activity = AgentActivity.__new__(AgentActivity)
    added: list[llm.ChatItem] = []
    activity._agent = SimpleNamespace(_chat_ctx=chat_ctx, stt=NOT_GIVEN)  # type: ignore[assignment]
    activity._session = MagicMock()
    activity._session._text_only = False
    activity._session.stt = None
    activity._session._conversation_item_added.side_effect = added.append
    return activity, added


def test_recorded_ack_sits_between_question_and_answer_without_reply() -> None:
    chat_ctx = llm.ChatContext.empty()
    question = chat_ctx.add_message(
        role="assistant", content="사과가 몇 개 필요하신가요?", created_at=AGENT_START
    )
    activity, added = _activity(chat_ctx)
    activity._create_speech_task = MagicMock()  # type: ignore[method-assign]
    ack = _final(" 네.", start=ACK_START, end=ACK_END, created_at=ACK_ARRIVED)

    activity.on_held_ignore_word_discarded(
        ack, started_speaking_at=ACK_START, stopped_speaking_at=ACK_END
    )
    answer = chat_ctx.add_message(role="user", content="3개가 필요해요.", created_at=AGENT_END + 8)

    assert [item.id for item in chat_ctx.items] == [question.id, added[0].id, answer.id]
    recorded = added[0]
    assert isinstance(recorded, llm.ChatMessage)
    assert recorded.role == "user"
    assert recorded.text_content == "네."
    assert recorded.created_at == ACK_START
    assert recorded.metrics["started_speaking_at"] == ACK_START
    assert recorded.metrics["stopped_speaking_at"] == ACK_END
    assert recorded.extra[DELAYED_INTERRUPTION_CONSUMED_EXTRA_KEY] is True
    assert recorded.transcript_confidence == 0.9
    activity._create_speech_task.assert_not_called()  # type: ignore[attr-defined]


def test_ack_recorded_before_the_question_lands_still_sorts_after_it() -> None:
    # the agent's message is added when its playout finishes, which can be after the trim
    chat_ctx = llm.ChatContext.empty()
    activity, added = _activity(chat_ctx)

    activity.on_held_ignore_word_discarded(
        _final("네.", start=ACK_START, end=ACK_END, created_at=ACK_ARRIVED),
        started_speaking_at=ACK_START,
        stopped_speaking_at=ACK_END,
    )
    question = chat_ctx.add_message(
        role="assistant", content="사과가 몇 개 필요하신가요?", created_at=AGENT_START
    )

    assert [item.id for item in chat_ctx.items] == [question.id, added[0].id]


async def test_unaligned_word_times_are_not_taken_as_session_times() -> None:
    # e.g. a StreamAdapter: word times are relative to each recognized chunk, and only
    # speech_end_time is a wall-clock time
    ar = _recognition()
    ar._stt_aligned_transcript = False
    _open_overlap(ar)
    _backchannel_verdict(ar)
    chunked = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text="네.", start_time=0.2, end_time=0.5)],
        created_at=ACK_ARRIVED,
        speech_end_time=ACK_END,
    )
    await ar._on_stt_event(chunked)

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", ACK_END, ACK_END)]
