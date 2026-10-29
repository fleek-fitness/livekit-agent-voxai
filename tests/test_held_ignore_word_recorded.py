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
from livekit.agents.voice.audio_recognition import AudioRecognition, _HeldSegments

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
    ar._speech_start_time = None
    ar._last_speaking_time = None
    ar._audio_transcript = ""
    ar._audio_interim_transcript = ""
    ar._audio_preflight_transcript = ""
    ar._user_turn_start = None
    ar._user_turn_span = None
    ar._stt_model = None
    ar._stt_provider = None
    ar._session._room_io = None
    ar._interruption_enabled = True
    ar._interruption_ch = _RecordingChan()  # type: ignore[assignment]
    ar._agent_speaking = False
    ar._agent_speech_started_at = None
    ar._transcript_gate_active = False
    ar._held_segments = _HeldSegments()
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


@pytest.mark.parametrize("advance_speaking_clock", [False, True])
async def test_untimed_ack_retires_its_clock_before_the_next_stt_turn(
    advance_speaking_clock: bool,
) -> None:
    ar = _recognition()
    ar._stt_aligned_transcript = False
    _open_overlap(ar)
    _backchannel_verdict(ar)
    ar._speech_start_time = OVERLAP_START
    ar._last_speaking_time = OVERLAP_START + 0.3
    ar._user_turn_start = OVERLAP_START
    old_span = MagicMock()
    ar._user_turn_span = old_span
    await ar._on_stt_event(
        SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language="ko", text="네.")],
            created_at=ACK_ARRIVED,
        )
    )

    # The shared speaking clock can advance after the final was held. It is not an
    # utterance identity: the gate still owns this single, completed segment.
    if advance_speaking_clock:
        ar._last_speaking_time = ACK_ARRIVED

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", OVERLAP_START, OVERLAP_START + 0.3)]
    assert ar._speech_start_time is None
    assert ar._last_speaking_time is None
    assert ar._user_turn_start is None
    old_span.end.assert_called_once()
    # Exercise the actual STT onset branch: it only sets an unset turn clock.
    ar._turn_detection_mode = "stt"
    ar._vad = None
    answer_start = AGENT_END + 8
    AudioRecognition._process_stt_event(
        ar, SpeechEvent(type=SpeechEventType.START_OF_SPEECH, speech_start_time=answer_start)
    )
    assert ar._speech_start_time == answer_start
    # This is the anchor _run_eou_detection passes to _compute_end_of_turn_metrics.
    assert ar._user_turn_start == answer_start


@pytest.mark.parametrize("new_speech,buffered_text", [(True, ""), (False, "실제 답")])
async def test_recording_ack_preserves_newer_or_accumulated_turn(
    new_speech: bool, buffered_text: str
) -> None:
    ar = _recognition()
    _open_overlap(ar)
    _backchannel_verdict(ar)
    await ar._on_stt_event(_final("네.", start=ACK_START, end=ACK_END, created_at=ACK_ARRIVED))
    ar._speech_start_time = AGENT_END - 1 if new_speech else ACK_START
    ar._last_speaking_time = AGENT_END + 1 if new_speech else ACK_END
    ar._audio_transcript = buffered_text
    original_start = ar._speech_start_time
    original_end = ar._last_speaking_time

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", ACK_START, ACK_END)]
    assert ar._speech_start_time == original_start
    assert ar._last_speaking_time == original_end
    assert ar._audio_transcript == buffered_text


async def test_late_untimed_final_cannot_claim_a_second_completed_vad_segment() -> None:
    ar = _recognition()
    ar._stt_aligned_transcript = False
    _open_overlap(ar)
    _backchannel_verdict(ar)
    next_start, next_end = 153.0, 154.0
    ar._on_start_of_speech(started_at=next_start)
    ar._speech_start_time = next_start
    ar._last_speaking_time = next_end
    ar._active_vad_speech_started_at = None
    delayed_final = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text="네.")],
        created_at=155.0,
    )
    await ar._on_stt_event(delayed_final)

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", 155.0, 155.0)]
    assert ar._speech_start_time == next_start
    assert ar._last_speaking_time == next_end


async def test_aligned_ack_timing_does_not_have_to_equal_its_vad_owner() -> None:
    ar = _recognition()
    _open_overlap(ar)
    _backchannel_verdict(ar)
    ar._speech_start_time = OVERLAP_START
    ar._user_turn_start = OVERLAP_START
    ar._user_turn_span = MagicMock()
    ar._last_speaking_time = OVERLAP_START + 0.4
    stt_start, stt_end = OVERLAP_START - 0.1, OVERLAP_START + 0.3
    await ar._on_stt_event(_final("네.", start=stt_start, end=stt_end, created_at=ACK_ARRIVED))

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", stt_start, stt_end)]
    assert ar._speech_start_time is None
    assert ar._user_turn_start is None
    answer_start = AGENT_END + 8
    ar._ensure_user_turn_span(start_time=answer_start)
    assert ar._user_turn_start == answer_start


@pytest.mark.parametrize(
    "retained_type", [SpeechEventType.FINAL_TRANSCRIPT, SpeechEventType.INTERIM_TRANSCRIPT]
)
async def test_same_segment_retained_answer_keeps_its_span(retained_type: SpeechEventType) -> None:
    ar = _recognition()
    _open_overlap(ar)
    _backchannel_verdict(ar)
    ar._speech_start_time = OVERLAP_START
    ar._user_turn_start = OVERLAP_START
    span = MagicMock()
    ar._user_turn_span = span
    ar._last_speaking_time = AGENT_END - 0.5
    ar._backchannel_boundary = (0.0, 1.0)
    await ar._on_stt_event(
        _final("네.", start=OVERLAP_START, end=OVERLAP_START + 0.3, created_at=ACK_ARRIVED)
    )
    retained = SpeechEvent(
        type=retained_type,
        alternatives=[SpeechData(language="ko", text="그런데 두 개 필요해요.")],
        speech_start_time=OVERLAP_START,
        speech_end_time=AGENT_END - 0.5,
        created_at=AGENT_END,
    )
    ar._transcript_buffer.append(retained)

    ar._on_end_of_agent_speech(ended_at=AGENT_END)

    assert _recorded(ar) == [("네.", OVERLAP_START, OVERLAP_START + 0.3)]
    ar._process_stt_event.assert_called_once_with(retained)  # type: ignore[attr-defined]
    # the ack's final alone settled the segment; its retained answer makes it a turn candidate
    # again, so the replayed answer starts its turn where the segment started
    assert ar._speech_start_time == OVERLAP_START
    assert ar._user_turn_start == OVERLAP_START
    assert ar._user_turn_span is not None


def _user_speech_starts(ar: AudioRecognition, at: float) -> None:
    # the VAD start-of-speech handler, then the hooks forwarding it to _on_start_of_speech
    if not ar._vad_speech_started:
        ar._speech_start_time = at
        ar._vad_speech_started = True
    ar._active_vad_speech_started_at = at
    ar._ensure_user_turn_span(start_time=at)
    ar._on_start_of_speech(started_at=at)
    ar._speaking = True
    ar._last_speaking_time = at


def _user_speech_ends(ar: AudioRecognition, at: float) -> None:
    # the VAD end-of-speech handler
    ar._on_end_of_speech(ended_at=at)
    ar._active_vad_speech_started_at = None
    ar._vad_speech_started = False
    ar._speaking = False
    ar._last_speaking_time = at


def _verdict(ar: AudioRecognition, *, is_interruption: bool, at: float, started: float) -> None:
    ar._on_overlap_speech_event(
        OverlappingSpeechEvent(
            is_interruption=is_interruption,
            agent_ended=False,
            detected_at=at,
            overlap_started_at=started,
        )
    )


async def _consumed_ack(ar: AudioRecognition, start: float) -> None:
    # "네." over the agent: the final is held, the verdict calls it a backchannel and trims it
    _user_speech_starts(ar, start)
    _user_speech_ends(ar, start + 0.3)
    if ar._stt_aligned_transcript:
        ack = _final("네.", start=start, end=start + 0.3, created_at=start + 0.4)
    else:
        # no speech timing: held with the completed VAD segment's clock
        ack = SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language="ko", text="네.")],
            created_at=start + 0.4,
        )
    ack.request_id = f"ack-{start}"
    await ar._on_stt_event(ack)
    _verdict(ar, is_interruption=False, at=start + 0.5, started=start)


@pytest.mark.parametrize("aligned", [True, False])
async def test_second_ack_retires_its_clock_before_the_answer_in_the_same_agent_speech(
    aligned: bool,
) -> None:
    first_ack, second_ack, answer_start = 148.91, 153.00, 169.05
    answer_text = "두 개 필요해요."
    ar = _recognition()
    ar._stt_aligned_transcript = aligned
    activity, added = _activity(llm.ChatContext.empty())
    activity._create_speech_task = MagicMock()  # type: ignore[method-assign]
    ar._hooks.on_held_ignore_word_discarded.side_effect = activity.on_held_ignore_word_discarded
    ar._on_start_of_agent_speech(started_at=AGENT_START)

    await _consumed_ack(ar, first_ack)
    await _consumed_ack(ar, second_ack)

    assert _recorded(ar) == [
        ("네.", first_ack, first_ack + 0.3),
        ("네.", second_ack, second_ack + 0.3),
    ]
    retired_clock = (ar._speech_start_time, ar._last_speaking_time, ar._user_turn_span)
    retired_request_ids = list(ar._stt_request_ids)
    assert [item.text_content for item in added] == ["네.", "네."]
    activity._create_speech_task.assert_not_called()  # type: ignore[attr-defined]
    # Neither acknowledgement interrupted the agent or requested a turn/reply.
    assert ar._agent_speaking
    assert ar._hooks.interruption_by_audio_activity_enabled is False
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]

    # The real answer also starts during the same agent utterance and remains held.
    _user_speech_starts(ar, answer_start)
    answer = _final(
        answer_text, start=answer_start, end=answer_start + 1.2, created_at=answer_start + 1.4
    )
    await ar._on_stt_event(answer)
    # This is the anchor used by end-of-turn metrics, not just the VAD clock.
    assert ar._user_turn_start == answer_start
    assert retired_clock == (None, None, None)
    assert retired_request_ids == []
    assert ar._hooks.interruption_by_audio_activity_enabled is False
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]

    # Playout completion releases only the answer; the acknowledgements never replay.
    ar._on_end_of_agent_speech(ended_at=answer_start + 1.5)
    ar._process_stt_event.assert_called_once()  # type: ignore[attr-defined]
    (replayed,) = ar._process_stt_event.call_args.args  # type: ignore[attr-defined]
    assert replayed.alternatives[0].text == answer_text
    assert len(_recorded(ar)) == 2
    assert ar._speech_start_time == answer_start
    assert ar._user_turn_start == answer_start
    assert ar._hooks.interruption_by_audio_activity_enabled is False
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]


async def test_ack_consumed_while_the_answer_is_held_keeps_the_answer_clock() -> None:
    first_ack, second_ack, answer_start = 148.91, 153.00, 153.60
    ar = _recognition()
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    await _consumed_ack(ar, first_ack)

    # the second ack's verdict lands only after the answer started and its interim is held
    _user_speech_starts(ar, second_ack)
    _user_speech_ends(ar, second_ack + 0.3)
    await ar._on_stt_event(
        _final("네.", start=second_ack, end=second_ack + 0.3, created_at=second_ack + 0.4)
    )
    _user_speech_starts(ar, answer_start)
    held_answer = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text="그런데 두 개")],
        created_at=answer_start + 0.5,
    )
    await ar._on_stt_event(held_answer)
    # the second ack still owns the turn clock while its verdict is pending
    assert ar._user_turn_start == second_ack
    _verdict(ar, is_interruption=False, at=answer_start + 0.6, started=second_ack)

    assert _recorded(ar)[-1] == ("네.", second_ack, second_ack + 0.3)
    # the held answer is untouched: still buffered, still speaking, clock not cleared
    assert list(ar._transcript_buffer) == [held_answer]
    assert ar._speaking
    assert ar._speech_start_time == answer_start
    assert ar._last_speaking_time == answer_start
    # the consumed ack hands the turn clock to the answer instead of lending its own start
    assert ar._user_turn_span is not None
    assert ar._user_turn_start == answer_start
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]


async def _ack_judged_before_its_final(
    ar: AudioRecognition, start: float, *, interim: str | None = None
) -> None:
    # "네." over the agent: the verdict lands while it is still spoken, its final only after
    _user_speech_starts(ar, start)
    if interim is not None:
        # the STT first hears another word and its final corrects it to "네."
        await ar._on_stt_event(
            SpeechEvent(
                type=SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        language="ko",
                        text=interim,
                        start_time=start - INPUT_START if ar._stt_aligned_transcript else 0.0,
                        end_time=start + 0.2 - INPUT_START if ar._stt_aligned_transcript else 0.0,
                    )
                ],
                created_at=start + 0.3,
            )
        )
    _verdict(ar, is_interruption=False, at=start + 0.6, started=start)
    _user_speech_ends(ar, start + 0.39)
    if ar._stt_aligned_transcript:
        ack = _final("네.", start=start, end=start + 0.39, created_at=start + 0.79)
    else:
        ack = SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language="ko", text="네.")],
            created_at=start + 0.79,
        )
    await ar._on_stt_event(ack)


@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("answer_held", [True, False])
async def test_ack_judged_before_its_final_does_not_lend_its_turn_start_to_the_answer(
    aligned: bool, answer_held: bool
) -> None:
    await _ack_ack_answer(aligned=aligned, answer_held=answer_held)


@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("answer_held", [True, False])
async def test_ack_corrected_from_an_interim_does_not_lend_its_turn_start_to_the_answer(
    aligned: bool, answer_held: bool
) -> None:
    # the held interim "내" is superseded by its own final "네.", so it is not a pending answer
    await _ack_ack_answer(aligned=aligned, answer_held=answer_held, first_interim="내")


async def _ack_ack_answer(
    *, aligned: bool, answer_held: bool, first_interim: str | None = None
) -> None:
    first_ack, second_ack = 155.24, 163.30
    agent_end = 171.05 if answer_held else 166.00
    answer_start = 169.05 if answer_held else 172.40
    answer_text = "3개가 필요해요."
    ar = _recognition()
    ar._stt_aligned_transcript = aligned
    activity, added = _activity(llm.ChatContext.empty())
    activity._create_speech_task = MagicMock()  # type: ignore[method-assign]
    ar._hooks.on_held_ignore_word_discarded.side_effect = activity.on_held_ignore_word_discarded
    ar._on_start_of_agent_speech(started_at=AGENT_START)

    await _ack_judged_before_its_final(ar, first_ack, interim=first_interim)
    # the second "네." starts while the first one's final is still held
    _user_speech_starts(ar, second_ack)
    assert ar._user_turn_start == second_ack
    if aligned:
        second = _final("네.", start=second_ack, end=second_ack + 0.3, created_at=second_ack + 0.5)
    else:
        second = SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language="ko", text="네.")],
            created_at=second_ack + 0.5,
        )
    await ar._on_stt_event(second)
    _verdict(ar, is_interruption=False, at=second_ack + 0.55, started=second_ack)
    _user_speech_ends(ar, second_ack + 0.3)

    # the first acknowledgement is recorded at its own speech time
    assert _recorded(ar) == [("네.", first_ack, first_ack + 0.39)]

    if answer_held:
        # the real answer starts during the same agent utterance and remains held
        _user_speech_starts(ar, answer_start)
        assert ar._user_turn_start == answer_start
        await ar._on_stt_event(
            _final(answer_text, start=answer_start, end=answer_start + 1.2, created_at=170.45)
            if aligned
            else SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(language="ko", text=answer_text)],
                created_at=170.45,
            )
        )
        ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
        ar._on_end_of_agent_speech(ended_at=agent_end)
    else:
        ar._on_end_of_agent_speech(ended_at=agent_end)
        _user_speech_starts(ar, answer_start)

    second_timing = (second_ack, second_ack + 0.3) if aligned else (second_ack + 0.5,) * 2
    assert _recorded(ar) == [
        ("네.", first_ack, first_ack + 0.39),
        ("네.", *second_timing),
    ]
    assert [item.text_content for item in added] == ["네.", "네."]
    # the answer owns its turn start: not the first or second acknowledgement's
    assert ar._speech_start_time == answer_start
    assert ar._user_turn_start == answer_start
    if answer_held:
        # playout completion releases only the answer; the acknowledgements never replay
        ar._process_stt_event.assert_called_once()  # type: ignore[attr-defined]
        (replayed,) = ar._process_stt_event.call_args.args  # type: ignore[attr-defined]
        assert replayed.alternatives[0].text == answer_text
    else:
        ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    # neither acknowledgement interrupted the agent or requested a turn/reply
    assert ar._hooks.interruption_by_audio_activity_enabled is False
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]
    activity._create_speech_task.assert_not_called()  # type: ignore[attr-defined]


async def test_pending_interim_after_the_ack_final_keeps_the_held_answer() -> None:
    first_ack, next_start = 155.24, 156.40
    ar = _recognition()
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    await _ack_judged_before_its_final(ar, first_ack)
    # the customer goes on after "네.": this interim follows the final, so it is still pending
    held_answer = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text="그런데 두 개")],
        created_at=first_ack + 1.0,
    )
    await ar._on_stt_event(held_answer)
    held_span = ar._user_turn_span
    held_request_ids = list(ar._stt_request_ids)

    _user_speech_starts(ar, next_start)

    # the held segment is not settled: nothing is retired and the pending answer stays held
    assert held_answer in ar._transcript_buffer
    assert ar._speech_start_time is not None
    assert ar._user_turn_span is held_span
    assert ar._user_turn_start == first_ack
    assert ar._stt_request_ids == held_request_ids
    assert _recorded(ar) == []
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]


async def test_dropped_backchannel_does_not_lend_its_turn_start_to_the_answer() -> None:
    backchannel_start, answer_start = 155.24, 165.00
    ar = _recognition()
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    _user_speech_starts(ar, backchannel_start)
    _user_speech_ends(ar, backchannel_start + 0.6)
    await ar._on_stt_event(
        _final(
            "아 그렇구나",
            start=backchannel_start,
            end=backchannel_start + 0.6,
            created_at=backchannel_start + 0.8,
        )
    )
    _verdict(ar, is_interruption=False, at=backchannel_start + 0.9, started=backchannel_start)

    # not an ignore word: dropped as before, but its clock is retired with it
    assert _recorded(ar) == []
    assert ar._speech_start_time is None
    assert ar._user_turn_start is None

    ar._on_end_of_agent_speech(ended_at=AGENT_END + 0.5)
    _user_speech_starts(ar, answer_start)
    assert ar._user_turn_start == answer_start
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]


def _ack_final(aligned: bool, *, start: float, end: float, created_at: float) -> SpeechEvent:
    if aligned:
        return _final("네.", start=start, end=end, created_at=created_at)
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="ko", text="네.")],
        created_at=created_at,
    )


async def _ack_judged_without_its_final(ar: AudioRecognition, start: float) -> None:
    # "네." over the agent, judged a backchannel; its final has not arrived yet
    _user_speech_starts(ar, start)
    _verdict(ar, is_interruption=False, at=start + 0.6, started=start)
    _user_speech_ends(ar, start + 0.39)


@pytest.mark.parametrize("aligned", [True, False])
async def test_ack_final_after_the_next_overlap_does_not_lend_its_turn_start(
    aligned: bool,
) -> None:
    first_ack, second_ack, answer_start = 155.24, 156.00, 165.00
    first_arrived, second_arrived = 156.20, 156.50
    ar = _recognition()
    ar._stt_aligned_transcript = aligned
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    await _ack_judged_without_its_final(ar, first_ack)

    # the second "네." starts before the first one's final arrives
    _user_speech_starts(ar, second_ack)
    assert ar._user_turn_start == second_ack
    await ar._on_stt_event(
        _ack_final(aligned, start=first_ack, end=first_ack + 0.39, created_at=first_arrived)
    )
    _user_speech_ends(ar, second_ack + 0.3)
    await ar._on_stt_event(
        _ack_final(aligned, start=second_ack, end=second_ack + 0.3, created_at=second_arrived)
    )
    _verdict(ar, is_interruption=False, at=156.60, started=second_ack)

    # an untimed first final keeps its arrival time: it never takes the second ack's clock
    first_timing = (first_ack, first_ack + 0.39) if aligned else (first_arrived,) * 2
    assert _recorded(ar) == [("네.", *first_timing), ("네.", second_ack, second_ack + 0.3)]
    assert ar._speech_start_time is None
    assert ar._user_turn_start is None

    ar._on_end_of_agent_speech(ended_at=AGENT_END)
    _user_speech_starts(ar, answer_start)
    assert ar._user_turn_start == answer_start
    ar._process_stt_event.assert_not_called()  # type: ignore[attr-defined]
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize("aligned", [True, False])
async def test_ack_final_after_an_interrupting_answer_started_keeps_the_answer_start(
    aligned: bool,
) -> None:
    ack_start, answer_start = 155.24, 156.00
    ar = _recognition()
    ar._stt_aligned_transcript = aligned
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    await _ack_judged_without_its_final(ar, ack_start)
    _user_speech_starts(ar, answer_start)
    await ar._on_stt_event(
        _ack_final(aligned, start=ack_start, end=ack_start + 0.39, created_at=156.20)
    )

    _verdict(ar, is_interruption=True, at=156.50, started=answer_start)

    replayed = [call.args[0] for call in ar._process_stt_event.call_args_list]  # type: ignore[attr-defined]
    if aligned:
        # timed: it ended before the answer, so it is trimmed and recorded
        assert _recorded(ar) == [("네.", ack_start, ack_start + 0.39)]
        assert replayed == []
    else:
        # untimed: it arrived after the answer started, so it stays in the answer's turn
        assert _recorded(ar) == []
        assert [ev.alternatives[0].text for ev in replayed] == ["네."]
    assert ar._user_turn_start == answer_start
