"""A late ack-only final is merged into the user turn that is already in progress.

Sequence: the agent asks an interruptible question, the user says "ok" over it (an ignore
word, so it does not interrupt), the STT is slow for that segment and delivers the final only
after the agent finished and after the user started a newer utterance. That final must neither
start a preemptive generation on its own nor let the turn commit before the newer utterance's
own final arrives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, stt, vad
from livekit.agents.llm import ChatContext
from livekit.agents.voice._ignore_words import matches_ignore_words, matches_ignore_words_exactly
from livekit.agents.voice.agent_activity import (
    ActivityClosedError,
    AgentActivity,
    _matches_ignore_words,
)
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = pytest.mark.unit

SPEED = 5.0
SESSION_TIMEOUT = 120.0
IGNORE_WORDS = ["네", "예", "ok"]

# The fake session replays the customer's audio and transcripts on the wall clock, while the
# agent's speech starts and ends whenever its fake LLM/TTS pipeline gets scheduled. Every
# scenario below therefore keeps real-time slack (SLACK_REAL seconds) between the events whose
# order matters, and the tests poll for the record they expect instead of asserting at a fixed
# instant. Times are scripted seconds (SPEED of them pass per real second).
#
#   AI_START ... AI_END   the question is audible (about; a pipeline lag shifts both later)
#   ack                   starts and ends well inside that interval
#   newer utterance       starts SLACK after AI_END and lasts NEWER_LEN
#   ack final             lands in the middle of the newer utterance
#   own final             lands SLACK after the newer utterance's VAD end of speech
SLACK_REAL = 1.5
SLACK = SLACK_REAL * SPEED
QUESTION_DURATION = 14.0
AI_START = 3.3  # the question becomes audible about here (the greeting turn ends at 2.5)
AI_END = AI_START + QUESTION_DURATION
ACK_START = AI_START + SLACK
ACK_END = ACK_START + 1.0
NEWER_START = AI_END + SLACK
NEWER_LEN = 10.0
NEWER_END = NEWER_START + NEWER_LEN
VAD_MIN_SILENCE = 0.5  # the fake VAD's end of speech comes this long after the audio ends
ACK_FINAL_AT = NEWER_START + NEWER_LEN / 2
OWN_FINAL_AT = NEWER_END + VAD_MIN_SILENCE + SLACK
# real seconds: wider than any scripted gap between audio and final, so a late final is
# still timed against the audio, and the merged turn waits for the own final
MAX_ENDPOINTING_DELAY = 10.0
WAIT_TIMEOUT = 60.0  # real seconds to wait for an expected record

QUESTION = "Is the second option ok?"
ACK = "네."
ANSWER = "그건 괜찮아요."
MERGED = f"{ACK} {ANSWER}"
REPLY_TO_MERGED = "Great, I will note that down."
REPLY_TO_ACK = "Is there anything else?"


class _Agent(Agent):
    def __init__(self, *, allow_delayed_ack: bool = False) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self._allow_delayed_ack = allow_delayed_ack

    def delayed_ack_answer_allowed(self) -> bool:
        return self._allow_delayed_ack


@pytest.fixture
def preemptive_inputs(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Transcripts every preemptive generation was started from."""
    seen: list[str] = []
    original = AgentActivity.on_preemptive_generation

    def spy(self: AgentActivity, info: Any) -> None:
        seen.append(info.new_transcript)
        original(self, info)

    monkeypatch.setattr(AgentActivity, "on_preemptive_generation", spy)
    yield seen


async def _run(
    actions: FakeActions,
    *,
    allow_delayed_ack: bool = False,
    max_delay: float = MAX_ENDPOINTING_DELAY,
    until: Callable[[_Agent, list[ConversationItemAddedEvent]], bool] | None = None,
    settle: float = 1.5,
) -> tuple[_Agent, list[ConversationItemAddedEvent]]:
    """Replay `actions`, wait (polling) until `until(agent, items)` holds, then settle.

    `max_delay` is the endpointing max delay in real seconds.
    """
    session = create_session(
        actions,
        speed_factor=SPEED,
        turn_handling={  # type: ignore[arg-type]
            "interruption": {"min_words": 1},
            "endpointing": {"max_delay": max_delay},
        },
        extra_kwargs={"interruption_ignore_words": IGNORE_WORDS},
    )
    agent = _Agent(allow_delayed_ack=allow_delayed_ack)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)
    await asyncio.wait_for(
        run_session(
            session,
            agent,
            drain_delay=settle,
            until=(lambda: until(agent, items)) if until else None,
            until_timeout=WAIT_TIMEOUT,
        ),
        timeout=SESSION_TIMEOUT,
    )
    return agent, items


def _user_texts(items: list[ConversationItemAddedEvent]) -> list[str]:
    return [
        ev.item.text_content or ""
        for ev in items
        if ev.item.type == "message" and ev.item.role == "user"
    ]


def _assistant_texts(agent: Agent) -> list[str]:
    return [
        item.text_content or ""
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "assistant"
    ]


def _merges(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.message == "late_ack_merged_into_turn"]


def _late_ack_actions(*, newer_final: bool = True) -> FakeActions:
    """The question is audible from about AI_START to AI_END.

    The ack (ACK_START-ACK_END) falls inside that interval but its final only arrives at
    ACK_FINAL_AT, while the newer utterance (NEWER_START-NEWER_END) is in progress. The newer
    utterance's own final arrives at OWN_FINAL_AT, after its VAD end of speech.
    """
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello there.")
    actions.add_llm(QUESTION)
    actions.add_tts(QUESTION_DURATION)
    actions.add_user_speech(ACK_START, ACK_END, ACK, stt_delay=ACK_FINAL_AT - ACK_END)
    actions.add_llm(REPLY_TO_MERGED, input=MERGED)
    actions.add_tts(1.0)
    # what a turn made of the ack alone would be answered with
    actions.add_llm(REPLY_TO_ACK, input=ACK)
    actions.add_tts(1.0)
    newer_text = ANSWER if newer_final else ""
    actions.add_user_speech(NEWER_START, NEWER_END, newer_text, stt_delay=OWN_FINAL_AT - NEWER_END)
    return actions


def _has_assistant_messages(
    count: int,
) -> Callable[[_Agent, list[ConversationItemAddedEvent]], bool]:
    return lambda agent, _items: len(_assistant_texts(agent)) >= count


async def test_late_ack_is_merged_into_the_newer_utterance_turn(
    caplog: pytest.LogCaptureFixture, preemptive_inputs: list[str]
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    agent, items = await _run(_late_ack_actions(), until=_has_assistant_messages(2))

    # "Hello there." greeting turn, then exactly one turn for the ack plus the real answer
    assert _user_texts(items) == ["Hello there.", MERGED]
    assert _assistant_texts(agent) == [QUESTION, REPLY_TO_MERGED]
    assert ACK not in preemptive_inputs  # never a preemptive generation from the ack alone
    assert MERGED in preemptive_inputs  # positive control: the spy is live, the merged turn fired
    assert len(_merges(caplog)) == 1


async def test_late_ack_merge_log_carries_no_transcript_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    await _run(_late_ack_actions(), until=lambda *_: bool(_merges(caplog)))

    (record,) = _merges(caplog)
    assert ACK not in str(record.__dict__)
    assert not hasattr(record, "user_input")


async def test_ack_spanning_agent_end_without_newer_utterance_keeps_delayed_ack_path(
    caplog: pytest.LogCaptureFixture, preemptive_inputs: list[str]
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello there.")
    actions.add_llm(QUESTION)
    actions.add_tts(QUESTION_DURATION)
    # starts during the agent's speech and keeps going well past its end; no newer utterance
    ack_end = AI_END + 0.3 * SPEED + SLACK
    actions.add_user_speech(ACK_START, ack_end, "예, 예", stt_delay=0.3)
    actions.add_llm("Great, I will register it.")
    actions.add_tts(1.0)
    agent, items = await _run(actions, allow_delayed_ack=True, until=_has_assistant_messages(2))

    assert len(_assistant_texts(agent)) == 2
    decisions = [r for r in caplog.records if r.message == "delayed_ack_as_answer"]
    assert [d.decision for d in decisions] == [True]
    assert not _merges(caplog)


async def test_ack_final_on_time_behaves_as_before(
    caplog: pytest.LogCaptureFixture, preemptive_inputs: list[str]
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello there.")
    actions.add_llm(QUESTION)
    actions.add_tts(QUESTION_DURATION)
    # inside the agent's speech, final delivered while the agent still speaks (a pipeline lag
    # only moves the agent's end later, which keeps this order)
    actions.add_user_speech(ACK_START, ACK_END, ACK, stt_delay=2.0)
    agent, items = await _run(actions)

    assert not _merges(caplog)
    assert _assistant_texts(agent) == [QUESTION]  # the ack alone is ignored, as before
    assert _user_texts(items).count(ACK) <= 1  # whether it is recorded is pre-existing behaviour


async def test_late_ack_is_committed_after_bounded_wait_when_no_final_follows(
    caplog: pytest.LogCaptureFixture, preemptive_inputs: list[str]
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    bounded_wait = 3.0  # real seconds
    # the newer utterance never yields a final: the ack must not be dropped
    agent, items = await _run(
        _late_ack_actions(newer_final=False),
        max_delay=bounded_wait,
        until=lambda agent, items: ACK in _user_texts(items) and len(_assistant_texts(agent)) >= 2,
    )

    assert len(_merges(caplog)) == 1
    assert _user_texts(items) == ["Hello there.", ACK]
    assert _assistant_texts(agent) == [QUESTION, REPLY_TO_ACK]
    assert ACK not in preemptive_inputs

    # the ack final arrived mid-utterance; the commit waited for the bounded delay after the
    # utterance's end of speech instead of firing at the normal endpointing delay
    ack_commit = next(
        ev
        for ev in items
        if ev.item.type == "message" and ev.item.role == "user" and ev.item.text_content == ACK
    )
    merge_logged_at = _merges(caplog)[0].created
    assert ack_commit.created_at - merge_logged_at >= bounded_wait


def test_ignore_word_matcher_stays_importable_from_agent_activity() -> None:
    assert _matches_ignore_words("네.", IGNORE_WORDS)
    assert not _matches_ignore_words("그건 괜찮아요", IGNORE_WORDS)


# direct checks of the late-ack predicate and the agent-speech / VAD bookkeeping

AGENT_START = 100.0
AGENT_END = 104.0
NEWER_SEGMENT_START = 105.0
INPUT_STARTED = 50.0


def _recognition(**overrides: Any) -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    state: dict[str, Any] = {
        "_vad_base_turn_detection": True,
        "_vad": object(),
        "_agent_speaking": False,
        "_last_agent_speech_interval": (AGENT_START, AGENT_END),
        "_post_agent_vad_segment_start": NEWER_SEGMENT_START,
        "_agent_overlap_segment_unfinalized": True,
        "_audio_transcript": "",
        "_late_ack_waiting": False,
        "_session": SimpleNamespace(
            options=SimpleNamespace(interruption_ignore_words=IGNORE_WORDS)
        ),
        "_endpointing": MagicMock(max_delay=6.0),
        "_user_silence_ev": asyncio.Event(),
        "_stt_pipeline": None,
    }
    state.update(overrides)
    for key, value in state.items():
        setattr(ar, key, value)
    return ar


def _final(text: str = ACK, *, start_time: float = 0.0) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="ko", text=text, start_time=start_time)],  # type: ignore[arg-type]
    )


def _timestamped(**overrides: Any) -> AudioRecognition:
    pipeline = SimpleNamespace(input_started_at=INPUT_STARTED)
    return _recognition(_stt_pipeline=pipeline, **overrides)


def test_predicate_fallback_without_stt_timestamps() -> None:
    assert _recognition()._is_late_ack_final(_final(), ACK) is True
    unfinalized = _recognition(_agent_overlap_segment_unfinalized=False)
    assert unfinalized._is_late_ack_final(_final(), ACK) is False


_FAR_PAST = 0.4  # the provider clock restarted near zero after a reconnect
_FUTURE = time.time() + 1000.0 - INPUT_STARTED


@pytest.mark.parametrize(
    "start_time, unfinalized, expected",
    [
        # reliable and inside the agent's last speech, before the newer segment: merge
        (AGENT_START + 1.0 - INPUT_STARTED, True, True),
        # reliable and inside, but the VAD bookkeeping saw no unfinalized overlap segment
        (AGENT_START + 1.0 - INPUT_STARTED, False, False),
        # reliable but outside the interval: no merge
        (AGENT_START - 1.0 - INPUT_STARTED, True, False),
        (AGENT_END + 0.5 - INPUT_STARTED, True, False),
        # unreliable (far in the past, or in the future): decided by the bookkeeping alone
        (_FAR_PAST, True, True),
        (_FUTURE, True, True),
        (_FAR_PAST, False, False),
        (_FUTURE, False, False),
    ],
)
def test_predicate_stt_timestamps(start_time: float, unfinalized: bool, expected: bool) -> None:
    ar = _timestamped(_agent_overlap_segment_unfinalized=unfinalized)
    assert ar._is_late_ack_final(_final(start_time=start_time), ACK) is expected


@pytest.mark.parametrize(
    "start_time, label",
    [
        (AGENT_START + 1.0 - INPUT_STARTED, "reliable"),
        (_FAR_PAST, "unreliable"),
        (_FUTURE, "unreliable"),
        (0.0, "absent"),
    ],
)
def test_merge_log_reports_the_timestamp_reliability(
    start_time: float, label: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    assert _timestamped()._is_late_ack_final(_final(start_time=start_time), ACK) is True
    (record,) = _merges(caplog)
    assert record.stt_timestamp == label


def test_predicate_timestamp_must_precede_the_newer_segment() -> None:
    ar = _timestamped(_post_agent_vad_segment_start=AGENT_START + 0.5)
    # the newer segment starts before the agent stopped according to the bookkeeping: never late
    assert ar._is_late_ack_final(_final(start_time=AGENT_START + 1.0 - INPUT_STARTED), ACK) is False


_PREDICATE_NEGATIVES = {
    "agent_speaking": {"_agent_speaking": True},
    "no_agent_interval": {"_last_agent_speech_interval": None},
    "no_newer_segment": {"_post_agent_vad_segment_start": None},
    "not_vad_turn_detection": {"_vad_base_turn_detection": False},
    "no_vad": {"_vad": None},
    "no_ignore_words": {
        "_session": SimpleNamespace(options=SimpleNamespace(interruption_ignore_words=None))
    },
}


@pytest.mark.parametrize("name", list(_PREDICATE_NEGATIVES))
def test_predicate_false(name: str) -> None:
    assert _recognition(**_PREDICATE_NEGATIVES[name])._is_late_ack_final(_final(), ACK) is False


@pytest.mark.parametrize("text", [ANSWER, "네, 그건 괜찮아요", "Is the second option ok?"])
def test_predicate_false_for_text_that_is_more_than_an_ack(text: str) -> None:
    assert _recognition()._is_late_ack_final(_final(text), text) is False


def test_agent_speech_end_records_the_interval() -> None:
    ar = _recognition(
        _agent_speaking=True,
        _agent_speech_started_at=AGENT_START,
        _last_agent_speech_interval=None,
        _interruption_enabled=False,
        _interruption_ch=None,
        _backchannel_boundary_timer=None,
        _backchannel_boundary_callback=None,
        _overlap_open=False,
        _transcript_gate_active=False,
        _transcript_buffer=[],
    )
    ar._on_end_of_agent_speech(ended_at=AGENT_END)
    assert ar._last_agent_speech_interval == (AGENT_START, AGENT_END)
    assert ar._agent_speech_started_at is None  # existing semantics unchanged

    # a second end without active speech leaves the recorded interval alone
    ar._on_end_of_agent_speech(ended_at=AGENT_END + 5)
    assert ar._last_agent_speech_interval == (AGENT_START, AGENT_END)


def test_vad_segments_are_classified_against_the_agent_speech() -> None:
    ar = _recognition(
        _post_agent_vad_segment_start=None,
        _vad_segment_overlaps_agent=False,
        _final_since_agent_overlap_segment=True,
        _late_ack_wait_timer=None,
    )
    ar._on_vad_segment_started(AGENT_START + 1.0)
    assert ar._vad_segment_overlaps_agent is True
    assert ar._final_since_agent_overlap_segment is False
    assert ar._post_agent_vad_segment_start is None

    ar._vad_segment_overlaps_agent = False
    ar._on_vad_segment_started(NEWER_SEGMENT_START)
    assert ar._post_agent_vad_segment_start == NEWER_SEGMENT_START
    # a later segment of the same unanswered stretch keeps the earliest start
    ar._on_vad_segment_started(NEWER_SEGMENT_START + 3)
    assert ar._post_agent_vad_segment_start == NEWER_SEGMENT_START


def test_start_of_agent_speech_drops_the_stale_newer_segment() -> None:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._post_agent_vad_segment_start = NEWER_SEGMENT_START
    ar._interruption_enabled = False
    ar._interruption_ch = None
    ar._endpointing = MagicMock()
    ar._backchannel_boundary = None
    ar._user_silence_ev = asyncio.Event()
    ar._user_silence_ev.set()
    ar._transcript_gate_active = False
    ar._on_start_of_agent_speech(started_at=time.time())
    assert ar._post_agent_vad_segment_start is None


def test_predicate_false_when_real_text_is_already_in_the_turn() -> None:
    assert _recognition(_audio_transcript=ANSWER)._is_late_ack_final(_final(), ACK) is False
    # a second late ack after a merged one is still only ack text
    waiting = _recognition(_audio_transcript=ACK, _late_ack_waiting=True)
    assert waiting._is_late_ack_final(_final(), ACK) is True


async def test_aclose_inside_the_bounded_wait_commits_the_merged_ack() -> None:
    ar = _recognition(
        _audio_transcript=ACK,
        _late_ack_waiting=True,
        _end_of_turn_task=None,
        _closing=asyncio.Event(),
        _commit_user_turn_atask=None,
        _stt_pipeline=None,
        _tasks=set(),
        _stt_consumer_atask=None,
        _vad_atask=None,
        _interruption_atask=None,
        _turn_detector_stream=None,
        _backchannel_boundary_timer=None,
        _late_ack_wait_timer=None,
        _user_turn_span=None,
    )
    ar._user_silence_ev.set()  # the user is not speaking
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    ar._hooks = MagicMock()
    ar._interruption_ch = None
    ar._turn_detector_prediction_fut = None

    await ar._aclose()

    ar._run_eou_detection.assert_called_once()
    assert ar._run_eou_detection.call_args.kwargs["trigger"] == "vad"
    assert ar._late_ack_waiting is False


# the agent starts speaking while a merged ack waits for the segment's own final


async def _eventually(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    """Poll until `predicate()` holds; a loaded machine only delays it, never breaks it."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


def _waiting_recognition(max_delay: float) -> AudioRecognition:
    ar = _recognition(
        _endpointing=MagicMock(max_delay=max_delay),
        _audio_transcript=ACK,
        _late_ack_waiting=True,
        _late_ack_wait_timer=None,
        _interruption_enabled=False,
        _interruption_ch=None,
        _backchannel_boundary=None,
        _backchannel_boundary_timer=None,
        _backchannel_boundary_callback=None,
        _overlap_open=False,
        _transcript_gate_active=False,
        _transcript_buffer=deque(),
        _agent_speech_started_at=None,
        _last_agent_speech_interval=None,
        _hooks=MagicMock(),
    )
    ar._user_silence_ev.set()  # the user is not speaking
    ar._run_eou_detection = MagicMock()  # type: ignore[method-assign]
    return ar


async def test_agent_speech_during_the_wait_defers_the_commit_until_playout_ends() -> None:
    max_delay = 0.6
    ar = _waiting_recognition(max_delay=max_delay)
    ar._arm_late_ack_wait()

    ar._on_start_of_agent_speech(started_at=time.time())
    await asyncio.sleep(max_delay * 2)  # the bounded wait would have expired during the playout
    ar._run_eou_detection.assert_not_called()  # type: ignore[attr-defined]
    assert ar._late_ack_waiting is True
    assert ar._audio_transcript == ACK  # kept, not dropped

    ar._on_end_of_agent_speech(ended_at=time.time())
    ar._run_eou_detection.assert_not_called()  # type: ignore[attr-defined]
    await _eventually(lambda: ar._run_eou_detection.called)  # type: ignore[attr-defined]
    ar._run_eou_detection.assert_called_once()  # type: ignore[attr-defined]
    assert ar._run_eou_detection.call_args.kwargs["trigger"] == "vad"  # type: ignore[attr-defined]
    assert ar._late_ack_waiting is False


def _driven_recognition(max_delay: float = 1.0) -> AudioRecognition:
    """A recognition whose real final-transcript and end-of-turn code runs against a mocked hook.

    `max_delay` (real seconds) is wide enough that "nothing committed yet" checks right after an
    event cannot be overtaken by the bounded wait on a loaded machine.
    """
    ar = _waiting_recognition(max_delay=max_delay)
    del ar._run_eou_detection  # use the real one: the commit goes through `on_end_of_turn`
    session = MagicMock()
    session.amd = None
    session.options.enable_adaptive_endpointing = False
    session.options.turn_handling = {"user_turn_limit": {}}
    session.options.interruption_ignore_words = IGNORE_WORDS
    for key, value in {
        "_session": session,
        "_stt": object(),
        "_turn_detection_mode": "vad",
        "_turn_detector": None,
        "_turn_detector_stream": None,
        "_turn_detector_prediction_fut": None,
        "_turn_detector_flushed": False,
        "_endpointing": MagicMock(min_delay=0.01, max_delay=max_delay),
        "_final_transcript_received": asyncio.Event(),
        "_final_transcript_confidence": [0.9],
        "_final_transcript_clock_suppressed": False,
        "_last_final_transcript_time": None,
        "_last_speaking_time": None,
        "_speech_start_time": None,
        "_user_turn_start": None,
        "_user_turn_span": None,
        "_user_turn_committed": False,
        "_vad_speech_started": False,
        "_end_of_turn_task": None,
        "_closing": asyncio.Event(),
        "_stt_request_ids": [],
        "_last_language": None,
        "_last_emitted_prediction": None,
        "_turn_backchannel_over_agent": False,
        "_overlap_in_current_turn": False,
        "_transcription_timeout_handle": None,
        "_turn_speech_duration": 0.0,
        "_turn_transcript_received": False,
        "_audio_interim_transcript": "",
        "_audio_preflight_transcript": "",
        "_post_agent_vad_segment_start": None,
    }.items():
        setattr(ar, key, value)
    ar._hooks.on_end_of_turn.return_value = True
    ar._hooks.retrieve_chat_ctx.return_value = ChatContext.empty()
    ar._ensure_user_turn_span = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(is_recording=MagicMock(return_value=False))
    )
    return ar


def _committed_transcripts(ar: AudioRecognition) -> list[str]:
    return [c.args[0].new_transcript for c in ar._hooks.on_end_of_turn.call_args_list]  # type: ignore[attr-defined]


@pytest.mark.parametrize("agent_speaking", [True, False], ids=["agent_speaking", "agent_stopped"])
async def test_real_final_while_waiting_commits_one_merged_turn_via_the_normal_path(
    agent_speaking: bool,
) -> None:
    ar = _driven_recognition()
    ar._agent_speaking = agent_speaking
    if agent_speaking:
        ar._on_start_of_agent_speech(started_at=time.time())
    else:
        ar._arm_late_ack_wait()
    assert ar._late_ack_waiting is True

    ar._process_stt_event(_final(ANSWER))

    assert ar._late_ack_waiting is False
    assert ar._late_ack_wait_timer is None
    await _eventually(lambda: bool(_committed_transcripts(ar)))
    await asyncio.sleep(0.2)  # a duplicate commit would show up by now
    assert _committed_transcripts(ar) == [MERGED]  # exactly one commit, ack first
    assert ar._audio_transcript == ""


# exact matching of final transcripts


@pytest.mark.parametrize("text", ["2", "왜", "ㄴ", "", "  ", "...", "네 왜"])
def test_exact_matcher_rejects_wildcards_and_non_acks(text: str) -> None:
    assert matches_ignore_words_exactly(text, IGNORE_WORDS) is False
    ar = _recognition()
    assert ar._is_late_ack_final(_final(text or "."), text) is False


def test_exact_matcher_rejects_a_prefix_of_an_ignore_word() -> None:
    assert matches_ignore_words_exactly("네에", ["네에요"]) is False
    assert matches_ignore_words_exactly("네에요", ["네에요"]) is True
    session = SimpleNamespace(options=SimpleNamespace(interruption_ignore_words=["네에요"]))
    assert _recognition(_session=session)._is_late_ack_final(_final("네에"), "네에") is False


@pytest.mark.parametrize("text", ["네.", "네 네", "예, 예", "OK", "ok. ok"])
def test_exact_matcher_accepts_ack_only_text(text: str) -> None:
    assert matches_ignore_words_exactly(text, IGNORE_WORDS) is True
    assert _recognition()._is_late_ack_final(_final(text), text) is True


# full-width and other Unicode punctuation is stripped like ASCII punctuation
@pytest.mark.parametrize("text", ["네～", "네…", "네〜", "네。", "네！", "네？", "「네」", "네·"])
def test_matchers_strip_unicode_punctuation(text: str) -> None:
    assert matches_ignore_words_exactly(text, IGNORE_WORDS) is True
    assert matches_ignore_words(text, IGNORE_WORDS) is True
    assert _recognition()._is_late_ack_final(_final(text), text) is True


def test_unicode_punctuation_does_not_make_other_text_an_ack() -> None:
    assert matches_ignore_words_exactly("네… 왜", IGNORE_WORDS) is False
    assert matches_ignore_words_exactly("…", IGNORE_WORDS) is False
    assert matches_ignore_words_exactly("ＯＫ", IGNORE_WORDS) is True  # NFKC folds full-width


def test_exact_matcher_needs_ignore_words() -> None:
    assert matches_ignore_words_exactly("네", None) is False
    assert matches_ignore_words_exactly("네", []) is False
    assert matches_ignore_words_exactly("네", ["", "..."]) is False


# consecutive late acks of one overlap segment


@pytest.mark.parametrize(
    "start_time, expected",
    [
        (0.0, True),  # no timestamp
        (AGENT_START + 1.0 - INPUT_STARTED, True),  # reliable, inside the agent interval
        (AGENT_END + 0.5 - INPUT_STARTED, False),  # reliable, outside: narrowing still applies
        (_FAR_PAST, True),  # unreliable
    ],
)
def test_predicate_further_ack_while_waiting(start_time: float, expected: bool) -> None:
    ar = _timestamped(
        _agent_overlap_segment_unfinalized=False, _audio_transcript=ACK, _late_ack_waiting=True
    )
    assert ar._is_late_ack_final(_final("예.", start_time=start_time), "예.") is expected


def _two_acks_recognition() -> AudioRecognition:
    ar = _driven_recognition()
    ar._audio_transcript = ""
    ar._late_ack_waiting = False
    ar._last_agent_speech_interval = (AGENT_START, AGENT_END)
    ar._post_agent_vad_segment_start = NEWER_SEGMENT_START
    ar._agent_overlap_segment_unfinalized = True
    ar._final_transcript_confidence = []
    ar._user_silence_ev.set()  # the newer segment already ended without its own final
    return ar


async def test_consecutive_late_acks_then_the_real_final_commit_one_turn() -> None:
    ar = _two_acks_recognition()

    ar._process_stt_event(_final(ACK))
    ar._process_stt_event(_final("예."))
    assert ar._late_ack_waiting is True
    ar._hooks.on_preemptive_generation.assert_not_called()  # type: ignore[attr-defined]
    await asyncio.sleep(0.1)
    assert _committed_transcripts(ar) == []

    ar._process_stt_event(_final(ANSWER))
    assert ar._late_ack_waiting is False
    await _eventually(lambda: bool(_committed_transcripts(ar)))
    await asyncio.sleep(0.2)
    assert _committed_transcripts(ar) == [f"{ACK} 예. {ANSWER}"]


async def test_consecutive_late_acks_without_a_real_final_commit_after_the_bounded_wait() -> None:
    ar = _two_acks_recognition()

    ar._process_stt_event(_final(ACK))
    ar._process_stt_event(_final("예."))
    ar._hooks.on_preemptive_generation.assert_not_called()  # type: ignore[attr-defined]
    await asyncio.sleep(0.1)
    assert _committed_transcripts(ar) == []  # still inside max_delay

    await _eventually(lambda: bool(_committed_transcripts(ar)))
    assert _committed_transcripts(ar) == [f"{ACK} 예."]  # committed, never dropped


# closing while a commit task is pending, and a stale overlap flag across agent speeches


async def test_aclose_with_a_pending_commit_task_leaves_the_whole_turn_to_it() -> None:
    ar = _driven_recognition()
    ar._last_agent_speech_interval = (AGENT_START, AGENT_END)
    ar._post_agent_vad_segment_start = NEWER_SEGMENT_START
    ar._last_final_transcript_time = time.time() - 5.0  # old: the commit waits for a transcript
    for key, value in {
        "_commit_user_turn_atask": None,
        "_tasks": set(),
        "_stt_consumer_atask": None,
        "_vad_atask": None,
        "_interruption_atask": None,
        "_turn_detector_stream": None,
        "_stt_pipeline": None,
        "_sample_rate": None,
    }.items():
        setattr(ar, key, value)

    # the ack is already merged and waiting (see `_driven_recognition`); the session then
    # scheduled the final user turn commit
    assert ar._late_ack_waiting and ar._audio_transcript == ACK
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=2.0)
    await asyncio.sleep(0.05)
    assert _committed_transcripts(ar) == []

    close = asyncio.create_task(ar._aclose())
    await asyncio.sleep(0.05)
    assert _committed_transcripts(ar) == []  # no ack-only commit from the close fallback

    ar._process_stt_event(_final(ANSWER))  # the following transcript
    await asyncio.wait_for(close, timeout=5.0)

    assert _committed_transcripts(ar) == [MERGED]
    assert fut.done()


async def _vad_event(ar: AudioRecognition, type: vad.VADEventType) -> None:
    await ar._on_vad_event(
        vad.VADEvent(
            type=type, samples_index=0, timestamp=0.0, speech_duration=0.0, silence_duration=0.0
        )
    )


async def test_overlap_flag_from_an_earlier_agent_speech_does_not_leak_into_the_next() -> None:
    ar = _driven_recognition()
    ar._last_agent_speech_interval = None
    ar._post_agent_vad_segment_start = None
    ar._agent_overlap_segment_unfinalized = False
    ar._audio_transcript = ""
    ar._late_ack_waiting = False
    ar._final_transcript_confidence = []
    ar._vad_base_turn_detection = True
    ar._vad = object()

    # agent speech 1: the user speaks over it and no final arrives for that segment
    ar._agent_speaking = True
    ar._on_start_of_agent_speech(started_at=time.time())
    await _vad_event(ar, vad.VADEventType.START_OF_SPEECH)
    await _vad_event(ar, vad.VADEventType.END_OF_SPEECH)
    assert ar._agent_overlap_segment_unfinalized is True
    ar._on_end_of_agent_speech(ended_at=time.time())

    # agent speech 2 starts and ends without any user overlap
    ar._on_start_of_agent_speech(started_at=time.time())
    assert ar._agent_overlap_segment_unfinalized is False
    ar._on_end_of_agent_speech(ended_at=time.time())

    # the user answers "ok" in a new segment; no STT timestamps
    await asyncio.sleep(0.02)
    await _vad_event(ar, vad.VADEventType.START_OF_SPEECH)
    ar._process_stt_event(_final(ACK))
    assert ar._late_ack_waiting is False
    ar._hooks.on_preemptive_generation.assert_called_once()  # type: ignore[attr-defined]
    await _vad_event(ar, vad.VADEventType.END_OF_SPEECH)
    await _eventually(lambda: bool(_committed_transcripts(ar)))
    assert _committed_transcripts(ar) == [ACK]  # the normal path, not a bounded wait


async def test_user_segment_open_when_agent_speech_starts_overlaps_that_speech() -> None:
    ar = _driven_recognition()
    ar._late_ack_waiting = False
    ar._audio_transcript = ""
    ar._last_agent_speech_interval = None
    await _vad_event(ar, vad.VADEventType.START_OF_SPEECH)  # user speaks first
    ar._on_start_of_agent_speech(started_at=time.time())  # the agent starts over them
    assert ar._vad_segment_overlaps_agent is True
    await _vad_event(ar, vad.VADEventType.END_OF_SPEECH)
    assert ar._agent_overlap_segment_unfinalized is True


# wait_for_idle must not return while a merged late ack still waits for its turn


def _activity_over(ar: AudioRecognition) -> AgentActivity:
    activity = object.__new__(AgentActivity)
    state: dict[str, Any] = {
        "_audio_recognition": ar,
        "_closed": False,
        "_session": SimpleNamespace(_closing=False, _user_turn_claims=0, _idle_holds=0),
        "_current_speech": None,
        "_speech_q": [],
        "_user_turn_completed_atask": None,
        "_agent": SimpleNamespace(label="test"),
    }
    for key, value in state.items():
        setattr(activity, key, value)
    return activity


async def test_wait_for_idle_holds_until_the_bounded_wait_commits_the_turn() -> None:
    ar = _driven_recognition()  # a merged ack waits; the newer segment ended without a final
    ar._arm_late_ack_wait()  # max_delay is 1.0 s
    waiter = asyncio.create_task(_activity_over(ar).wait_for_idle())

    await asyncio.sleep(0.1)
    assert not waiter.done()
    assert _committed_transcripts(ar) == []

    await asyncio.wait_for(waiter, timeout=5.0)
    assert _committed_transcripts(ar) == [ACK]  # idle only after the turn was committed


async def test_wait_for_idle_holds_until_a_real_final_commits_the_turn() -> None:
    ar = _driven_recognition()
    ar._arm_late_ack_wait()
    waiter = asyncio.create_task(_activity_over(ar).wait_for_idle())
    await asyncio.sleep(0.05)
    assert not waiter.done()

    ar._process_stt_event(_final(ANSWER))
    await asyncio.wait_for(waiter, timeout=5.0)
    assert _committed_transcripts(ar) == [MERGED]


async def test_wait_for_idle_without_a_late_ack_is_unchanged() -> None:
    ar = _driven_recognition()
    ar._clear_late_ack_wait()
    ar._audio_transcript = ""
    await asyncio.wait_for(_activity_over(ar).wait_for_idle(), timeout=1.0)
    assert _committed_transcripts(ar) == []


async def test_close_during_the_wait_releases_wait_for_idle() -> None:
    ar = _driven_recognition()
    for key, value in {
        "_commit_user_turn_atask": None,
        "_tasks": set(),
        "_stt_consumer_atask": None,
        "_vad_atask": None,
        "_interruption_atask": None,
        "_turn_detector_stream": None,
        "_stt_pipeline": None,
    }.items():
        setattr(ar, key, value)
    activity = _activity_over(ar)
    waiter = asyncio.create_task(activity.wait_for_idle())
    await asyncio.sleep(0.05)
    assert not waiter.done()

    activity._session._closing = True
    await asyncio.wait_for(ar._aclose(), timeout=5.0)

    with pytest.raises(ActivityClosedError):
        await asyncio.wait_for(waiter, timeout=5.0)
    assert ar._late_ack_waiting is False


# hard bound on the idle wait


async def test_close_during_the_wait_releases_wait_for_idle_promptly() -> None:
    max_delay = 3.0
    ar = _driven_recognition(max_delay=max_delay)
    for key, value in {
        "_commit_user_turn_atask": None,
        "_tasks": set(),
        "_stt_consumer_atask": None,
        "_vad_atask": None,
        "_interruption_atask": None,
        "_turn_detector_stream": None,
        "_stt_pipeline": None,
    }.items():
        setattr(ar, key, value)
    ar._arm_late_ack_wait()
    activity = _activity_over(ar)
    waiter = asyncio.create_task(activity.wait_for_idle())
    await asyncio.sleep(0.03)

    activity._session._closing = True
    started = time.perf_counter()
    await asyncio.wait_for(ar._aclose(), timeout=5.0)
    with pytest.raises(ActivityClosedError):
        await asyncio.wait_for(waiter, timeout=5.0)
    assert time.perf_counter() - started < max_delay / 2  # released, not waited out


async def test_wait_for_idle_returns_about_max_delay_after_an_intervening_agent_speech() -> None:
    max_delay = 0.8
    ar = _driven_recognition(max_delay=max_delay)
    ar._arm_late_ack_wait()
    waiter = asyncio.create_task(_activity_over(ar).wait_for_idle())
    await asyncio.sleep(0.03)

    ar._on_start_of_agent_speech(started_at=time.time())  # e.g. an end_call farewell
    await asyncio.sleep(max_delay * 1.5)  # longer than max_delay: nothing may commit meanwhile
    assert not waiter.done()
    assert _committed_transcripts(ar) == []

    ended = time.perf_counter()
    ar._on_end_of_agent_speech(ended_at=time.time())
    await asyncio.wait_for(waiter, timeout=15.0)
    elapsed = time.perf_counter() - ended
    # the bounded wait restarted when the speech ended, then the turn committed; a loaded
    # machine can only stretch the upper side
    assert max_delay * 0.8 <= elapsed <= max_delay + 10.0
    assert _committed_transcripts(ar) == [ACK]


def _closable(ar: AudioRecognition) -> AudioRecognition:
    for key, value in {
        "_commit_user_turn_atask": None,
        "_tasks": set(),
        "_stt_consumer_atask": None,
        "_vad_atask": None,
        "_interruption_atask": None,
        "_turn_detector_stream": None,
        "_stt_pipeline": None,
    }.items():
        setattr(ar, key, value)
    return ar


async def test_wait_for_idle_never_blocks_beyond_the_cap(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    ar = _closable(_driven_recognition(max_delay=0.05))  # cap = 2 * 0.05 + 1.0 s
    assert ar._late_ack_waiting  # and nothing is armed: a missing re-arm

    started = time.perf_counter()
    await asyncio.wait_for(_activity_over(ar).wait_for_idle(), timeout=15.0)
    elapsed = time.perf_counter() - started

    assert 1.0 <= elapsed < 5.0  # a single cap, not one per internal check
    (record,) = [r for r in caplog.records if r.message == "late_ack_idle_wait_timeout"]
    assert record.cap == pytest.approx(1.1)
    # the merged ack keeps its state: still waiting, text kept, nothing committed yet
    assert ar._late_ack_waiting is True
    assert ar._audio_transcript == ACK
    assert _committed_transcripts(ar) == []

    # and the close fallback still commits it, exactly once
    await asyncio.wait_for(ar._aclose(), timeout=5.0)
    assert _committed_transcripts(ar) == [ACK]


async def test_agent_speech_longer_than_the_cap_does_not_hold_idle_or_lose_the_ack() -> None:
    ar = _driven_recognition(max_delay=0.05)  # cap = 1.1 s
    ar._arm_late_ack_wait()
    waiter = asyncio.create_task(_activity_over(ar).wait_for_idle())
    await asyncio.sleep(0.02)
    ar._on_start_of_agent_speech(started_at=time.time())  # a long speech

    started = time.perf_counter()
    await asyncio.wait_for(waiter, timeout=15.0)
    assert 1.0 <= time.perf_counter() - started < 5.0  # returned at the cap, mid-speech
    assert _committed_transcripts(ar) == []
    assert ar._late_ack_waiting is True

    ar._on_end_of_agent_speech(ended_at=time.time())
    await _eventually(lambda: bool(_committed_transcripts(ar)))  # max_delay after the speech
    assert _committed_transcripts(ar) == [ACK]
    assert ar._late_ack_waiting is False


async def test_a_dropped_end_of_turn_still_releases_the_wait() -> None:
    ar = _driven_recognition()
    ar._hooks.on_end_of_turn.return_value = False  # the turn is not committed
    ar._run_eou_detection(ar._hooks.retrieve_chat_ctx(), trigger="manual")
    await _eventually(lambda: bool(_committed_transcripts(ar)))
    await _eventually(lambda: not ar._late_ack_waiting)
    assert _committed_transcripts(ar) == [ACK]
