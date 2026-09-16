from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from livekit.agents.metrics import (
    TTSMetrics,
)
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
)
from livekit.agents.voice.endpointing import BaseEndpointing

pytestmark = pytest.mark.unit


def _recognition() -> AudioRecognition:
    return AudioRecognition(
        SimpleNamespace(options=SimpleNamespace(interruption={}), amd=None),
        hooks=SimpleNamespace(),
        endpointing=BaseEndpointing(min_delay=0.1, max_delay=0.5),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection="vad",
    )


def test_final_transcript_clock_does_not_advance_after_endpointing_window() -> None:
    recognition = _recognition()
    recognition._last_speaking_time = time.time() - 1.0
    recognition._speaking = False

    assert recognition._should_advance_final_transcript_clock() is False


def test_final_transcript_clock_advances_while_user_is_speaking() -> None:
    recognition = _recognition()
    recognition._last_speaking_time = time.time() - 1.0
    recognition._speaking = True

    assert recognition._should_advance_final_transcript_clock() is True


def test_suppressed_final_transcript_still_updates_recency_clock() -> None:
    recognition = _recognition()
    recognition._last_speaking_time = time.time() - 1.0
    recognition._speaking = False

    recognition._record_final_transcript_time()

    assert recognition._final_transcript_clock_suppressed is True
    assert recognition._last_final_transcript_time is not None


def _opts(
    *,
    min_interruption_words: int = 0,
    dyn_min_words: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        interruption={"min_words": dyn_min_words or min_interruption_words},
        interruption_ignore_words=["네", "예"],
        enable_dynamic_interruption=False,
        enable_adaptive_endpointing=False,
    )


def _activity_for_end_of_turn(
    *,
    min_interruption_words: int = 0,
    dyn_min_words: int = 0,
) -> tuple[AgentActivity, object, list[object]]:
    activity = AgentActivity.__new__(AgentActivity)
    chat_items: list[object] = []
    opts = _opts(min_interruption_words=min_interruption_words, dyn_min_words=dyn_min_words)
    activity._opts = opts
    activity._agent = SimpleNamespace(
        stt=SimpleNamespace(metrics_metadata=None),
        _chat_ctx=SimpleNamespace(items=chat_items),
    )
    activity._session = SimpleNamespace(
        options=opts,
        _text_only=False,
        _closing=False,
        _amd=None,
        _conversation_item_added=lambda *a, **k: None,
    )
    activity._turn_detection = "vad"
    activity._rt_turn_detection_enabled = False
    activity._false_interruption_timer = None
    activity._paused_speech = None
    activity._current_speech = None
    activity._scheduling_paused = False
    activity._new_turns_blocked = False
    activity._preemptive_generation = None
    activity._user_turn_completed_atask = None
    activity._user_speech_started_during_interruptible_agent_speech = True
    activity._dynamic_interruption = SimpleNamespace(
        get_current_min_interruption_words=lambda: dyn_min_words,
    )
    task = object()

    def create_speech_task(coro, *, speech_handle=None, name=None):  # noqa: ANN001
        coro.close()
        return task

    activity._create_speech_task = create_speech_task
    return activity, task, chat_items


def _info(
    *,
    text: str,
    suppressed: bool = False,
    skip_reply: bool = False,
) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        new_transcript=text,
        transcript_confidence=1.0,
        transcript_clock_suppressed=suppressed,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None if suppressed else 400.0,
            stopped_speaking_at=None if suppressed else 456.0,
            transcription_delay=None if suppressed else 0.1,
            end_of_turn_delay=None if suppressed else 0.2,
        ),
    )


def test_vad_end_does_not_create_response_latency_anchor() -> None:
    updates = []
    activity = AgentActivity.__new__(AgentActivity)
    activity._session = SimpleNamespace(
        options=SimpleNamespace(interruption={"false_interruption_timeout": None}),
        _update_user_state=lambda state, **kwargs: updates.append((state, kwargs)),
        _user_speaking_span=None,
    )
    activity._audio_recognition = None
    activity._user_silence_event = SimpleNamespace(set=lambda: None)
    activity._dynamic_interruption = SimpleNamespace(on_user_speech_ended=lambda: None)
    activity._paused_speech = None
    activity._stt_eos_received = False
    activity._interruption_detection_enabled = False

    activity.on_end_of_speech(None)

    assert updates[-1][0] == "listening"
    assert isinstance(updates[-1][1]["last_speaking_time"], float)


def test_vad_start_keeps_context_when_speech_was_already_interrupted() -> None:
    updates = []
    activity = AgentActivity.__new__(AgentActivity)
    activity._current_speech = SimpleNamespace(interrupted=True, allow_interruptions=True)
    activity._session = SimpleNamespace(
        agent_state="listening",
        _user_speaking_span=None,
        _update_user_state=lambda state, **kwargs: updates.append((state, kwargs)),
    )
    activity._audio_recognition = None
    activity._user_silence_event = SimpleNamespace(clear=lambda: None)
    activity._stt_eos_received = True
    activity._dynamic_interruption = SimpleNamespace(on_user_speech_started=lambda **kwargs: None)
    activity._false_interruption_timer = None
    activity._interruption_detected = True
    activity._capture_dynamic_continuity_at_speech_start = lambda: None
    activity._pause_enabled = lambda: False

    activity.on_start_of_speech(None, speech_start_time=123.0)

    assert activity._user_speech_started_during_interruptible_agent_speech is True
    assert activity._stt_eos_received is False
    assert updates[-1][0] == "speaking"


def test_suppressed_transcript_commits_without_latency_anchor() -> None:
    activity, task, _ = _activity_for_end_of_turn()

    committed = activity.on_end_of_turn(_info(text="네", suppressed=True))

    assert committed is True
    assert activity._user_turn_completed_atask is task
    assert activity._user_speech_started_during_interruptible_agent_speech is False


def test_suppressed_transcript_bypasses_interruption_filters() -> None:
    activity, task, chat_items = _activity_for_end_of_turn(dyn_min_words=3)
    activity._current_speech = SimpleNamespace(allow_interruptions=True, interrupted=False)

    committed = activity.on_end_of_turn(_info(text="네", suppressed=True))

    assert committed is True
    assert activity._user_turn_completed_atask is task
    assert activity._user_speech_started_during_interruptible_agent_speech is False
    assert chat_items == []


def test_committed_turn_waits_for_reply_before_setting_response_latency_anchor() -> None:
    activity, task, _ = _activity_for_end_of_turn()
    activity._user_speech_started_during_interruptible_agent_speech = False

    committed = activity.on_end_of_turn(_info(text="예약 문의요"))

    assert committed is True
    assert activity._user_turn_completed_atask is task


def test_skip_reply_turn_does_not_set_response_latency_anchor() -> None:
    activity, task, _ = _activity_for_end_of_turn()
    activity._user_speech_started_during_interruptible_agent_speech = False

    committed = activity.on_end_of_turn(_info(text="예약 문의요", skip_reply=True))

    assert committed is True
    assert activity._user_turn_completed_atask is task


def test_non_interruptible_no_reply_turn_does_not_set_response_latency_anchor() -> None:
    activity, task, _ = _activity_for_end_of_turn()
    activity._current_speech = SimpleNamespace(allow_interruptions=False, interrupted=False)
    activity._user_speech_started_during_interruptible_agent_speech = False

    committed = activity.on_end_of_turn(_info(text="예약 문의요"))

    assert committed is True
    assert activity._user_turn_completed_atask is task


def test_non_suppressed_delayed_ignored_backchannel_is_consumed_without_reply_or_anchor() -> None:
    activity, _, chat_items = _activity_for_end_of_turn()

    committed = activity.on_end_of_turn(_info(text="네"))

    assert committed is True
    assert activity._user_turn_completed_atask is None
    assert activity._user_speech_started_during_interruptible_agent_speech is False
    assert len(chat_items) == 1
    assert chat_items[0].role == "user"


def test_delayed_short_transcript_consumed_via_min_interruption_words_branch() -> None:
    activity, _, chat_items = _activity_for_end_of_turn(dyn_min_words=3)
    activity._current_speech = SimpleNamespace(allow_interruptions=False, interrupted=False)

    committed = activity.on_end_of_turn(_info(text="아"))

    assert committed is True
    assert activity._user_turn_completed_atask is None
    assert activity._user_speech_started_during_interruptible_agent_speech is False
    assert len(chat_items) == 1
    assert chat_items[0].text_content == "아"


def test_delayed_empty_transcript_consumed_via_empty_interruption_branch() -> None:
    activity, _, chat_items = _activity_for_end_of_turn(dyn_min_words=2)
    activity._current_speech = SimpleNamespace(allow_interruptions=False, interrupted=False)

    committed = activity.on_end_of_turn(_info(text=""))

    assert committed is True
    assert activity._user_turn_completed_atask is None
    assert activity._user_speech_started_during_interruptible_agent_speech is False
    assert chat_items == []


class _TestSpeechHandle:
    def __init__(self, speech_id: str = "test-speech") -> None:
        self.id = speech_id
        self.interrupted = False
        self.done_callbacks = []

    async def interrupt(self) -> None:
        self.interrupted = True

    def add_done_callback(self, callback) -> None:  # noqa: ANN001
        self.done_callbacks.append(callback)


async def _async_noop(*args: object, **kwargs: object) -> None:
    return None


def _build_user_turn_completed_activity(
    emitted: list[tuple[str, object]],
) -> tuple[AgentActivity, _TestSpeechHandle]:
    activity = AgentActivity.__new__(AgentActivity)
    activity._agent = SimpleNamespace(
        chat_ctx=SimpleNamespace(copy=lambda: SimpleNamespace(items=[])),
        on_user_turn_completed=_async_noop,
        _chat_ctx=SimpleNamespace(items=[]),
        llm=SimpleNamespace(),
        stt=SimpleNamespace(model="test-stt", provider="test-provider", metrics_metadata=None),
    )
    activity._session = SimpleNamespace(
        emit=lambda evt, payload: emitted.append((evt, payload)),
        _text_only=False,
        _closing=False,
        _conversation_item_added=lambda *a, **k: None,
        llm=None,
    )
    activity._rt_session = None
    activity._current_speech = None
    activity._scheduling_paused = False
    activity._new_turns_blocked = False
    activity._preemptive_generation = None
    activity._turn_detection = "vad"
    activity._dynamic_interruption = SimpleNamespace(reset_collisions=lambda: None)
    speech_handle = _TestSpeechHandle()
    activity._generate_reply = lambda **kwargs: speech_handle
    activity._interrupt_background_speeches = lambda force=False: []
    return activity, speech_handle


def test_metrics_collected_emitted_when_clock_not_suppressed() -> None:
    emitted: list[tuple[str, object]] = []
    activity, speech_handle = _build_user_turn_completed_activity(emitted)

    async def run() -> None:
        activity._user_turn_completed_atask = asyncio.current_task()
        await activity._user_turn_completed_task(None, _info(text="안녕하세요"))

    asyncio.run(run())

    assert any(evt == "metrics_collected" for evt, _ in emitted), emitted


def test_metrics_collected_suppressed_when_clock_suppressed() -> None:
    emitted: list[tuple[str, object]] = []
    activity, _ = _build_user_turn_completed_activity(emitted)

    async def run() -> None:
        activity._user_turn_completed_atask = asyncio.current_task()
        await activity._user_turn_completed_task(None, _info(text="네", suppressed=True))

    asyncio.run(run())

    assert not any(evt == "metrics_collected" for evt, _ in emitted), emitted


class _UsageCollector:
    def collect(self, metrics: object) -> None:
        return None


def _tts_metrics(*, speech_id: str, timestamp: float = 10.0) -> TTSMetrics:
    return TTSMetrics(
        label="test-tts",
        request_id=f"tts-{speech_id}",
        timestamp=timestamp,
        ttfb=0.2,
        duration=1.0,
        audio_duration=1.0,
        cancelled=False,
        characters_count=10,
        streamed=True,
        speech_id=speech_id,
    )
