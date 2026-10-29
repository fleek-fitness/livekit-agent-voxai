"""Which overlap segment owns the user turn clock, for every way its transcript can arrive.

Each case replays one event order through AudioRecognition: agent speech, VAD onsets and ends,
held STT interims and finals, overlap verdicts, and the end of agent speech. The checks are the
turn start the next real answer gets and the acknowledgements recorded at their spoken time.

A segment judged a backchannel must never lend its start to the next turn, whether its final is
late, corrected from another interim, missing, or untimed. A segment whose text is still going to
reach a turn keeps its start.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit

IGNORE_WORDS = ["네", "예"]
INPUT_START = 100.0
AGENT_START = 146.17
AGENT_END = 161.05
ANSWER = 165.00


class _RecordingChan:
    closed = False

    def __init__(self) -> None:
        self.sent: list[object] = []

    def send_nowait(self, item: object) -> None:
        self.sent.append(item)


def _recognition(*, aligned: bool, transcription_timeout: float | None = None) -> AudioRecognition:
    session = MagicMock()
    session.options.interruption_ignore_words = IGNORE_WORDS
    session.options.interruption = {}
    session.options.transcription_timeout = transcription_timeout
    session._room_io = None
    session.amd = None
    ar = AudioRecognition(
        session,
        hooks=MagicMock(),
        endpointing=MagicMock(),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection="vad",
        stt_aligned_transcript=aligned,
    )
    ar._interruption_enabled = True
    ar._interruption_ch = _RecordingChan()  # type: ignore[assignment]
    ar._stt_pipeline = SimpleNamespace(input_started_at=INPUT_START)  # type: ignore[assignment]
    ar._hooks.interruption_by_audio_activity_enabled = False
    ar._process_stt_event = MagicMock()  # type: ignore[method-assign]
    return ar


@dataclass
class Sos:
    at: float


@dataclass
class Eos:
    at: float


@dataclass
class Text:
    text: str
    start: float
    end: float
    arrived: float
    final: bool = True


@dataclass
class Verdict:
    at: float
    started: float
    kind: str = "backchannel"  # or "interruption" / "agent_ended"


@dataclass
class AgentEnd:
    at: float = AGENT_END


Step = Sos | Eos | Text | Verdict | AgentEnd


async def _play(ar: AudioRecognition, steps: list[Step]) -> None:
    ar._on_start_of_agent_speech(started_at=AGENT_START)
    for step in steps:
        if isinstance(step, Sos):
            # the VAD start-of-speech handler, then the hooks forwarding it
            if not ar._vad_speech_started:
                ar._speech_start_time = step.at
                ar._vad_speech_started = True
            ar._active_vad_speech_started_at = step.at
            ar._ensure_user_turn_span(start_time=step.at)
            ar._on_start_of_speech(started_at=step.at)
            ar._speaking = True
            ar._last_speaking_time = step.at
        elif isinstance(step, Eos):
            # the hooks forwarding VAD end-of-speech, then the VAD handler
            ar._on_end_of_speech(ended_at=step.at)
            ar._active_vad_speech_started_at = None
            ar._vad_speech_started = False
            ar._speaking = False
            ar._last_speaking_time = step.at
            ar._arm_transcription_timeout(step.at - AGENT_START, delay=0.0)
        elif isinstance(step, Text):
            aligned = ar._stt_aligned_transcript
            await ar._on_stt_event(
                SpeechEvent(
                    type=(
                        SpeechEventType.FINAL_TRANSCRIPT
                        if step.final
                        else SpeechEventType.INTERIM_TRANSCRIPT
                    ),
                    alternatives=[
                        SpeechData(
                            language="ko",
                            text=step.text,
                            start_time=step.start - INPUT_START if aligned else 0.0,
                            end_time=step.end - INPUT_START if aligned else 0.0,
                            confidence=0.9,
                        )
                    ],
                    created_at=step.arrived,
                )
            )
        elif isinstance(step, Verdict):
            ar._on_overlap_speech_event(
                OverlappingSpeechEvent(
                    is_interruption=step.kind == "interruption",
                    agent_ended=step.kind == "agent_ended",
                    detected_at=step.at,
                    overlap_started_at=step.started,
                )
            )
        else:
            ar._on_end_of_agent_speech(ended_at=step.at)


async def _play_text(ar: AudioRecognition, text: Text) -> None:
    """Deliver one STT text to a recognition whose agent speech was already played."""
    started = ar._on_start_of_agent_speech
    ar._on_start_of_agent_speech = lambda **_: None  # type: ignore[method-assign]
    try:
        await _play(ar, [text])
    finally:
        ar._on_start_of_agent_speech = started  # type: ignore[method-assign]


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


def _replayed(ar: AudioRecognition) -> list[str]:
    return [
        call.args[0].alternatives[0].text
        for call in ar._process_stt_event.call_args_list  # type: ignore[attr-defined]
        if call.args[0].alternatives
    ]


def _ack(start: float, *, verdict_first: bool = True, text: str = "네.") -> list[Step]:
    """A short acknowledgement over the agent, judged a backchannel, final held."""
    end = round(start + 0.39, 2)
    final = Text(text, start, end, round(start + 0.79, 2))
    verdict = Verdict(round(start + 0.6, 2), start)
    if verdict_first:
        return [Sos(start), verdict, Eos(end), final]
    return [Sos(start), Eos(end), final, verdict]


@dataclass
class Case:
    steps: list[Step]
    turn_start: float | None
    recorded: list[tuple[str, float, float]]
    # recorded timing when the STT gives no speech times, if it differs
    untimed_recorded: list[tuple[str, float, float]] | None = None
    replayed: tuple[str, ...] = ()
    untimed_replayed: tuple[str, ...] | None = None
    # the VAD clock, when it differs from the turn start
    speech_start: float | None = None


A1, A2 = 155.24, 156.00

CASES: dict[str, Case] = {
    # interim differs from final
    "interim_corrected_before_final": Case(
        [
            Sos(A1),
            Text("내", A1, A1 + 0.2, A1 + 0.3, final=False),
            *_ack(A1)[1:],
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A1, A1 + 0.39)],
    ),
    "interim_ye_corrected": Case(
        [
            Sos(A1),
            Text("예요", A1, A1 + 0.2, A1 + 0.3, final=False),
            *_ack(A1)[1:],
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A1, A1 + 0.39)],
    ),
    "interim_nene_corrected": Case(
        [
            Sos(A1),
            Text("네네 내", A1, A1 + 0.2, A1 + 0.3, final=False),
            *_ack(A1)[1:],
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A1, A1 + 0.39)],
    ),
    "several_interims_then_one_final": Case(
        [
            Sos(A1),
            Text("내", A1, A1 + 0.1, A1 + 0.2, final=False),
            Text("내 네", A1, A1 + 0.2, A1 + 0.3, final=False),
            Text("그래", A1, A1 + 0.3, A1 + 0.35, final=False),
            *_ack(A1)[1:],
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A1, A1 + 0.39)],
    ),
    # the corrected final arrives only after the next overlap started
    "interim_corrected_after_next_overlap": Case(
        [
            Sos(A1),
            Text("내", A1, A1 + 0.2, A1 + 0.3, final=False),
            Verdict(A1 + 0.6, A1),
            Eos(A1 + 0.39),
            Sos(A2),
            Text("네.", A1, A1 + 0.39, 156.20),
            Eos(A2 + 0.3),
            Text("네.", A2, A2 + 0.3, 156.50),
            Verdict(156.60, A2),
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A1, A1 + 0.39), ("네.", A2, A2 + 0.3)],
        # an untimed final that arrives while the next segment is spoken keeps its arrival time
        untimed_recorded=[("네.", 156.20, 156.20), ("네.", A2, A2 + 0.3)],
    ),
    # final late or never
    "final_after_next_overlap": Case(
        [
            Sos(A1),
            Verdict(A1 + 0.6, A1),
            Eos(A1 + 0.39),
            Sos(A2),
            Text("네.", A1, A1 + 0.39, 156.20),
            Eos(A2 + 0.3),
            Text("네.", A2, A2 + 0.3, 156.50),
            Verdict(156.60, A2),
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A1, A1 + 0.39), ("네.", A2, A2 + 0.3)],
        untimed_recorded=[("네.", 156.20, 156.20), ("네.", A2, A2 + 0.3)],
    ),
    "final_never_then_answer_after_agent_end": Case(
        [Sos(A1), Verdict(A1 + 0.6, A1), Eos(A1 + 0.39), AgentEnd(), Sos(ANSWER)],
        ANSWER,
        [],
    ),
    "final_never_then_answer_during_agent_speech": Case(
        [
            Sos(A1),
            Verdict(A1 + 0.6, A1),
            Eos(A1 + 0.39),
            Sos(158.00),
            Text("두 개요", 158.00, 159.00, 159.20),
            AgentEnd(),
            Eos(161.30),
        ],
        158.00,
        [],
        replayed=("두 개요",),
    ),
    "first_final_missing_second_arrives": Case(
        [
            Sos(A1),
            Verdict(A1 + 0.6, A1),
            Eos(A1 + 0.39),
            Sos(A2),
            Eos(A2 + 0.3),
            Text("네.", A2, A2 + 0.3, 156.50),
            Verdict(156.60, A2),
            AgentEnd(),
            Sos(ANSWER),
        ],
        ANSWER,
        [("네.", A2, A2 + 0.3)],
        # untimed, the final could be either segment's: it keeps its arrival time
        untimed_recorded=[("네.", 156.50, 156.50)],
    ),
    # overlap judgment before / after the final
    "verdict_before_final": Case(
        [*_ack(A1), AgentEnd(), Sos(ANSWER)], ANSWER, [("네.", A1, A1 + 0.39)]
    ),
    "verdict_after_final": Case(
        [*_ack(A1, verdict_first=False), AgentEnd(), Sos(ANSWER)], ANSWER, [("네.", A1, A1 + 0.39)]
    ),
    "two_acks_then_answer": Case(
        [*_ack(A1), *_ack(158.00), AgentEnd(), Sos(ANSWER)],
        ANSWER,
        [("네.", A1, A1 + 0.39), ("네.", 158.00, 158.39)],
    ),
    # a backchannel that is not an ignore word is dropped, and so is its clock
    "non_ignore_backchannel_dropped": Case(
        [*_ack(A1, text="아 그렇구나"), AgentEnd(), Sos(ANSWER)], ANSWER, []
    ),
    # real answer while an acknowledgement is held
    "answer_held_while_ack_held": Case(
        [
            *_ack(A1),
            Sos(158.00),
            Text("두 개요", 158.00, 159.00, 159.20),
            AgentEnd(),
            Eos(161.30),
        ],
        158.00,
        [("네.", A1, A1 + 0.39)],
        replayed=("두 개요",),
    ),
    "answer_interrupts_after_ack": Case(
        [
            *_ack(A1),
            Sos(158.00),
            Text("잠깐만요", 158.00, 158.50, 158.60),
            Verdict(158.70, 158.00, kind="interruption"),
        ],
        158.00,
        [("네.", A1, A1 + 0.39)],
        replayed=("잠깐만요",),
    ),
    "answer_interrupts_before_the_ack_final": Case(
        [
            Sos(A1),
            Verdict(A1 + 0.6, A1),
            Eos(A1 + 0.39),
            Sos(A2),
            Text("네.", A1, A1 + 0.39, 156.20),
            Verdict(156.50, A2, kind="interruption"),
        ],
        A2,
        [("네.", A1, A1 + 0.39)],
        # untimed, it arrived after the answer started: it stays in the answer's turn
        untimed_recorded=[],
        untimed_replayed=("네.",),
    ),
    # the customer goes on after the acknowledgement; the trim drops that interim, so its
    # segment cannot own the next turn either
    "real_interim_after_ack_final_then_answer": Case(
        [
            *_ack(A1),
            Text("그런데 두 개", A1 + 0.8, A1 + 0.95, A1 + 1.0, final=False),
            Sos(156.40),
            AgentEnd(157.00),
        ],
        156.40,
        [("네.", A1, A1 + 0.39)],
    ),
    # the backchannel verdict lands only after the agent stopped and the answer began
    "verdict_after_agent_end_while_answering": Case(
        [Sos(A1), Eos(A1 + 0.39), AgentEnd(), Sos(ANSWER), Verdict(ANSWER + 0.1, A1)],
        ANSWER,
        [],
    ),
    # no verdict ever arrives for a segment that ended before the agent stopped: like any VAD
    # segment without a transcript, it keeps the first onset as the turn start
    "no_verdict_segment_keeps_the_first_onset": Case(
        [Sos(A1), Eos(A1 + 0.39), AgentEnd(), Sos(ANSWER)],
        A1,
        [],
        speech_start=ANSWER,
    ),
    # AI speech ends while the customer is still speaking: inconclusive, the segment keeps
    # its start and its final becomes the turn
    "agent_ends_mid_segment": Case(
        [Sos(160.50), AgentEnd(), Verdict(161.10, 160.50, kind="agent_ended"), Eos(161.30)],
        160.50,
        [],
    ),
}


@pytest.mark.parametrize("aligned", [True, False], ids=["aligned", "untimed"])
@pytest.mark.parametrize("name", list(CASES))
async def test_turn_owner(name: str, aligned: bool) -> None:
    case = CASES[name]
    ar = _recognition(aligned=aligned)

    await _play(ar, case.steps)

    recorded = case.recorded
    if not aligned and case.untimed_recorded is not None:
        recorded = case.untimed_recorded
    assert _recorded(ar) == [(t, pytest.approx(s), pytest.approx(e)) for t, s, e in recorded]
    assert ar._user_turn_start == case.turn_start
    if case.turn_start is not None:
        assert ar._speech_start_time == (case.speech_start or case.turn_start)
    replayed = case.replayed
    if not aligned and case.untimed_replayed is not None:
        replayed = case.untimed_replayed
    assert tuple(_replayed(ar)) == replayed
    # no acknowledgement interrupted the agent or requested a turn
    ar._hooks.on_end_of_turn.assert_not_called()  # type: ignore[attr-defined]


async def test_late_final_answer_gives_the_turn_clock_back_to_the_earlier_segment() -> None:
    # A is judged a backchannel before its final arrives, so its clock is retired and the next
    # segment B takes it. A's untimed final then turns out to be a real answer: A is the earliest
    # segment that can start a turn again, and the clock and the span follow it.
    ar = _recognition(aligned=False)

    await _play(
        ar,
        [
            Sos(148.00),
            Verdict(148.60, 148.00),
            Eos(149.00),
            Sos(150.00),
            Text("그런데 말이죠", 148.00, 149.00, 150.20),
            AgentEnd(150.30),
        ],
    )

    assert ar._user_turn_start == 148.00
    assert ar._speech_start_time == 148.00
    assert _replayed(ar) == ["그런데 말이죠"]


@pytest.mark.parametrize("aligned", [True, False], ids=["aligned", "untimed"])
async def test_final_after_the_gate_closed_gets_the_retired_turn_clock_back(aligned: bool) -> None:
    # The overlap is judged a backchannel and the agent stops before its final arrives. The gate
    # is off, so the final is not held; it still has to start the turn at its own speech time.
    ar = _recognition(aligned=aligned)
    clocks: list[tuple[float | None, float | None]] = []
    ar._process_stt_event.side_effect = lambda ev: clocks.append(  # type: ignore[attr-defined]
        (ar._user_turn_start, ar._speech_start_time)
    )

    await _play(ar, [Sos(160.00), Eos(160.50), Verdict(160.60, 160.00), AgentEnd(AGENT_END)])
    assert ar._user_turn_start is None
    await _play_text(ar, Text("두 개요", 160.00, 160.50, 161.20))

    assert clocks == [(160.00, 160.00)]


async def test_final_after_the_gate_closed_is_processed_with_the_segment_speech_end() -> None:
    # the real STT handler, not the stub: the end of speech must be the segment's end, not the
    # arrival of the final, or the transcription delay collapses to zero
    ar = _recognition(aligned=False)
    del ar._process_stt_event  # use the class method
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    ar._vad = MagicMock()  # a VAD anchors the end of speech
    ar._vad_base_turn_detection = True
    ar._endpointing.max_delay = 3.0
    ar._check_user_turn_limit = MagicMock()  # type: ignore[method-assign]

    await _play(ar, [Sos(160.00), Eos(160.50), Verdict(160.60, 160.00), AgentEnd(AGENT_END)])
    await _play_text(ar, Text("두 개요", 160.00, 160.50, 161.20))

    assert ar._speech_start_time == 160.00
    assert ar._last_speaking_time == 160.50
    ar._run_eou_detection.assert_called_once()


async def test_held_final_is_processed_with_the_segment_speech_end() -> None:
    # same as above while the gate is still open: the held final is replayed when the agent
    # stops, and the retired segment gets its speech end back along with its start
    ar = _recognition(aligned=False)
    del ar._process_stt_event
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    ar._vad = MagicMock()
    ar._vad_base_turn_detection = True
    ar._endpointing.max_delay = 3.0
    ar._check_user_turn_limit = MagicMock()  # type: ignore[method-assign]
    # a final that arrives within the end boundary is replayed instead of trimmed
    ar._backchannel_boundary = (0.0, 1.0)
    ends: list[float | None] = []
    process = ar._process_stt_event
    ar._process_stt_event = lambda ev: (  # type: ignore[method-assign]
        ends.append(ar._last_speaking_time),
        process(ev),
    )[1]

    await _play(
        ar,
        [
            Sos(160.00),
            Eos(160.50),
            Verdict(160.60, 160.00),
            Text("두 개요", 160.00, 160.50, 160.80),
            AgentEnd(AGENT_END),
        ],
    )

    assert ends == [160.50]


async def test_replayed_ack_keeps_its_speech_time() -> None:
    # an ignore word inside the end boundary is not trimmed: it reaches the turn, so it needs the
    # clock its segment lost to the backchannel verdict
    ar = _recognition(aligned=False)
    clocks: list[tuple[float | None, float | None]] = []
    ar._process_stt_event.side_effect = lambda ev: clocks.append(  # type: ignore[attr-defined]
        (ar._user_turn_start, ar._speech_start_time)
    )
    ar._backchannel_boundary = (0.0, 1.0)

    await _play(
        ar,
        [
            Sos(160.00),
            Eos(160.50),
            Verdict(160.60, 160.00),
            Text("네.", 160.00, 160.50, 160.80),
            AgentEnd(AGENT_END),
        ],
    )

    assert clocks == [(160.00, 160.00)]


async def test_final_after_the_gate_closed_keeps_a_newer_onset_clock() -> None:
    ar = _recognition(aligned=False)
    clocks: list[tuple[float | None, float | None]] = []
    ar._process_stt_event.side_effect = lambda ev: clocks.append(  # type: ignore[attr-defined]
        (ar._user_turn_start, ar._speech_start_time)
    )

    await _play(
        ar,
        [Sos(160.00), Eos(160.50), Verdict(160.60, 160.00), AgentEnd(AGENT_END), Sos(ANSWER)],
    )
    await _play_text(ar, Text("두 개요", ANSWER, ANSWER + 0.5, ANSWER + 0.9))

    assert clocks == [(ANSWER, ANSWER)]


async def test_transcription_timeout_is_not_reported_for_a_retired_backchannel() -> None:
    ar = _recognition(aligned=True, transcription_timeout=0.05)
    await _play(ar, [Sos(A1), Verdict(A1 + 0.6, A1), Eos(A1 + 0.39)])

    await asyncio.sleep(0.1)

    ar._hooks.on_transcription_timeout.assert_not_called()  # type: ignore[attr-defined]
    assert ar._user_turn_start is None


async def test_transcription_timeout_is_not_reported_when_the_next_segment_holds_a_final() -> None:
    # A's backchannel verdict arrives after B's final is held: handing the clock to B must not
    # forget that B's text was received
    ar = _recognition(aligned=True, transcription_timeout=0.05)
    await _play(
        ar,
        [
            Sos(150.00),
            Eos(150.39),
            Text("네.", 150.00, 150.39, 150.79),
            Sos(152.00),
            Text("두 개요", 152.00, 153.00, 153.20),
            Verdict(153.30, 150.00),
            Eos(153.40),
        ],
    )

    await asyncio.sleep(0.1)

    ar._hooks.on_transcription_timeout.assert_not_called()  # type: ignore[attr-defined]
    assert ar._user_turn_start == 152.00


async def test_transcription_timeout_is_still_reported_for_an_untranscribed_turn() -> None:
    ar = _recognition(aligned=True, transcription_timeout=0.05)
    await _play(ar, [Sos(160.50), AgentEnd(), Eos(161.30)])

    await asyncio.sleep(0.1)

    ar._hooks.on_transcription_timeout.assert_called_once()  # type: ignore[attr-defined]
    assert ar._hooks.on_transcription_timeout.call_args.kwargs["turn_start"] == 160.50  # type: ignore[attr-defined]
