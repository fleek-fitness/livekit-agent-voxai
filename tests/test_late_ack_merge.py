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
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, stt
from livekit.agents.llm import ChatContext
from livekit.agents.voice._ignore_words import matches_ignore_words_exactly
from livekit.agents.voice.agent_activity import AgentActivity, _matches_ignore_words
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = pytest.mark.unit

SPEED = 5.0
SESSION_TIMEOUT = 60.0
IGNORE_WORDS = ["네", "예", "ok"]

QUESTION = "Is the second option ok?"
ACK = "네."
ANSWER = "그건 괜찮아요."
MERGED = f"{ACK} {ANSWER}"


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
    max_delay: float | None = None,
) -> tuple[Agent, list[ConversationItemAddedEvent]]:
    turn_handling: dict[str, Any] = {"interruption": {"min_words": 1}}
    if max_delay is not None:
        turn_handling["endpointing"] = {"max_delay": max_delay / SPEED}
    session = create_session(
        actions,
        speed_factor=SPEED,
        turn_handling=turn_handling,  # type: ignore[arg-type]
        extra_kwargs={"interruption_ignore_words": IGNORE_WORDS},
    )
    agent = _Agent(allow_delayed_ack=allow_delayed_ack)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)
    await asyncio.wait_for(run_session(session, agent, drain_delay=2.0), timeout=SESSION_TIMEOUT)
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
    """The agent speaks the question audibly from about 3.5 s to 7.5 s.

    The ack (4.5-5.0) falls inside that interval but its final only arrives at 9.0, while the
    newer utterance (8.5-9.5, VAD end of speech at 10.0) is in progress. The newer utterance's
    own final arrives at 10.1, after that VAD end of speech.
    """
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello there.")
    actions.add_llm(QUESTION)
    actions.add_tts(4.0)
    actions.add_user_speech(4.5, 5.0, ACK, stt_delay=4.0)
    actions.add_llm("Great, I will note that down.", input=MERGED)
    actions.add_tts(1.0)
    # what a turn made of the ack alone would be answered with
    actions.add_llm("Is there anything else?", input=ACK)
    actions.add_tts(1.0)
    if newer_final:
        actions.add_user_speech(8.5, 9.5, ANSWER, stt_delay=0.6)
    else:
        actions.add_user_speech(8.5, 9.5, "", stt_delay=0.6)
    return actions


async def test_late_ack_is_merged_into_the_newer_utterance_turn(
    caplog: pytest.LogCaptureFixture, preemptive_inputs: list[str]
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    agent, items = await _run(_late_ack_actions())

    # "Hello there." greeting turn, then exactly one turn for the ack plus the real answer
    assert _user_texts(items) == ["Hello there.", MERGED]
    assert _assistant_texts(agent) == [QUESTION, "Great, I will note that down."]
    assert ACK not in preemptive_inputs  # never a preemptive generation from the ack alone
    assert MERGED in preemptive_inputs  # positive control: the spy is live, the merged turn fired
    assert len(_merges(caplog)) == 1


async def test_late_ack_merge_log_carries_no_transcript_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    await _run(_late_ack_actions())

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
    actions.add_tts(2.0)
    # starts during the agent's speech and keeps going past its end; no newer utterance follows
    actions.add_user_speech(4.5, 7.0, "예, 예", stt_delay=0.3)
    actions.add_llm("Great, I will register it.")
    actions.add_tts(1.0)
    agent, items = await _run(actions, allow_delayed_ack=True)

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
    actions.add_tts(4.0)
    # inside the agent's speech, final delivered while the agent still speaks
    actions.add_user_speech(4.5, 5.0, ACK, stt_delay=0.3)
    agent, items = await _run(actions)

    assert not _merges(caplog)
    assert _assistant_texts(agent) == [QUESTION]  # the ack alone is ignored, as before
    assert _user_texts(items).count(ACK) <= 1  # whether it is recorded is pre-existing behaviour


async def test_late_ack_is_committed_after_bounded_wait_when_no_final_follows(
    caplog: pytest.LogCaptureFixture, preemptive_inputs: list[str]
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    # the newer utterance never yields a final: the ack must not be dropped
    agent, items = await _run(_late_ack_actions(newer_final=False), max_delay=3.0)

    assert len(_merges(caplog)) == 1
    assert _user_texts(items) == ["Hello there.", ACK]
    assert _assistant_texts(agent) == [QUESTION, "Is there anything else?"]
    assert ACK not in preemptive_inputs

    # the commit waited for the bounded delay (3 s of script time) instead of firing at the
    # normal endpointing delay (0.5 s) after the newer utterance's end of speech
    ack_commit = next(
        ev
        for ev in items
        if ev.item.type == "message" and ev.item.role == "user" and ev.item.text_content == ACK
    )
    merge_logged_at = _merges(caplog)[0].created
    assert ack_commit.created_at - merge_logged_at >= 1.5 / SPEED


def test_ignore_word_matcher_stays_importable_from_agent_activity() -> None:
    assert _matches_ignore_words("네.", IGNORE_WORDS)
    assert not _matches_ignore_words("그건 괜찮아요", IGNORE_WORDS)


# direct checks of the late-ack predicate and the agent-speech / VAD bookkeeping

AGENT_START = 100.0
AGENT_END = 104.0
NEWER_START = 105.0
INPUT_STARTED = 50.0


def _recognition(**overrides: Any) -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    state: dict[str, Any] = {
        "_vad_base_turn_detection": True,
        "_vad": object(),
        "_agent_speaking": False,
        "_last_agent_speech_interval": (AGENT_START, AGENT_END),
        "_post_agent_vad_segment_start": NEWER_START,
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
    ar._on_vad_segment_started(NEWER_START)
    assert ar._post_agent_vad_segment_start == NEWER_START
    # a later segment of the same unanswered stretch keeps the earliest start
    ar._on_vad_segment_started(NEWER_START + 3)
    assert ar._post_agent_vad_segment_start == NEWER_START


def test_start_of_agent_speech_drops_the_stale_newer_segment() -> None:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._post_agent_vad_segment_start = NEWER_START
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
    ar = _waiting_recognition(max_delay=0.2)
    ar._arm_late_ack_wait()

    ar._on_start_of_agent_speech(started_at=time.time())
    await asyncio.sleep(0.4)  # the bounded wait would have expired during the playout
    ar._run_eou_detection.assert_not_called()  # type: ignore[attr-defined]
    assert ar._late_ack_waiting is True
    assert ar._audio_transcript == ACK  # kept, not dropped

    ar._on_end_of_agent_speech(ended_at=time.time())
    await asyncio.sleep(0.1)
    ar._run_eou_detection.assert_not_called()  # type: ignore[attr-defined]
    await asyncio.sleep(0.3)  # the bound restarted when the playout ended
    ar._run_eou_detection.assert_called_once()  # type: ignore[attr-defined]
    assert ar._run_eou_detection.call_args.kwargs["trigger"] == "vad"  # type: ignore[attr-defined]
    assert ar._late_ack_waiting is False


def _driven_recognition() -> AudioRecognition:
    """A recognition whose real final-transcript and end-of-turn code runs against a mocked hook."""
    ar = _waiting_recognition(max_delay=0.2)
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
        "_endpointing": MagicMock(min_delay=0.01, max_delay=0.2),
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
    await asyncio.sleep(0.5)  # well past the bounded wait and the endpointing delay
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
    ar._post_agent_vad_segment_start = NEWER_START
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
    await asyncio.sleep(0.5)
    assert _committed_transcripts(ar) == [f"{ACK} 예. {ANSWER}"]


async def test_consecutive_late_acks_without_a_real_final_commit_after_the_bounded_wait() -> None:
    ar = _two_acks_recognition()

    ar._process_stt_event(_final(ACK))
    ar._process_stt_event(_final("예."))
    ar._hooks.on_preemptive_generation.assert_not_called()  # type: ignore[attr-defined]
    await asyncio.sleep(0.1)
    assert _committed_transcripts(ar) == []  # still inside max_delay

    await asyncio.sleep(0.5)
    assert _committed_transcripts(ar) == [f"{ACK} 예."]  # committed, never dropped
