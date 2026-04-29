import time
from types import SimpleNamespace

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition, _EndOfTurnInfo


def test_final_transcript_clock_does_not_advance_after_endpointing_window() -> None:
    recognition = AudioRecognition(
        SimpleNamespace(),
        hooks=SimpleNamespace(),
        stt=None,
        vad=SimpleNamespace(),
        turn_detection="vad",
        min_endpointing_delay=0.1,
        max_endpointing_delay=0.5,
    )
    recognition._last_speaking_time = time.time() - 1.0
    recognition._speaking = False

    assert recognition._should_advance_final_transcript_clock() is False


def test_final_transcript_clock_advances_while_user_is_speaking() -> None:
    recognition = AudioRecognition(
        SimpleNamespace(),
        hooks=SimpleNamespace(),
        stt=None,
        vad=SimpleNamespace(),
        turn_detection="vad",
        min_endpointing_delay=0.1,
        max_endpointing_delay=0.5,
    )
    recognition._last_speaking_time = time.time() - 1.0
    recognition._speaking = True

    assert recognition._should_advance_final_transcript_clock() is True


def _activity_for_end_of_turn() -> tuple[AgentActivity, object]:
    activity = AgentActivity.__new__(AgentActivity)
    activity._agent = SimpleNamespace(stt=object())
    activity._session = SimpleNamespace(
        options=SimpleNamespace(
            min_interruption_words=0,
            interruption_ignore_words=["네", "예"],
        ),
        _closing=False,
    )
    activity._turn_detection = "vad"
    activity._current_speech = None
    activity._scheduling_paused = False
    activity._preemptive_generation = None
    activity._last_eou_timestamp = 123.0
    activity._user_turn_completed_atask = None
    activity._user_speech_started_during_interruptible_agent_speech = True
    activity._dynamic_interruption = SimpleNamespace(
        get_current_min_interruption_words=lambda: 0,
    )
    task = object()

    def create_speech_task(coro, *, speech_handle=None, name=None):  # noqa: ANN001
        coro.close()
        return task

    activity._create_speech_task = create_speech_task
    return activity, task


def test_vad_end_does_not_create_response_latency_anchor() -> None:
    updates = []
    activity = AgentActivity.__new__(AgentActivity)
    activity._session = SimpleNamespace(
        options=SimpleNamespace(false_interruption_timeout=None),
        _update_user_state=lambda state, **kwargs: updates.append((state, kwargs)),
    )
    activity._user_silence_event = SimpleNamespace(set=lambda: None)
    activity._dynamic_interruption = SimpleNamespace(on_user_speech_ended=lambda: None)
    activity._paused_speech = None
    activity._stt_eos_received = False
    activity._last_eou_timestamp = None

    activity.on_end_of_speech(None)

    assert activity._last_eou_timestamp is None
    assert updates[-1][0] == "listening"
    assert isinstance(updates[-1][1]["last_speaking_time"], float)


def test_suppressed_transcript_commits_without_latency_anchor() -> None:
    activity, task = _activity_for_end_of_turn()

    committed = activity.on_end_of_turn(
        _EndOfTurnInfo(
            skip_reply=False,
            new_transcript="네",
            transcript_confidence=1.0,
            transcript_clock_suppressed=True,
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        )
    )

    assert committed is True
    assert activity._user_turn_completed_atask is task
    assert activity._last_eou_timestamp is None
    assert activity._user_speech_started_during_interruptible_agent_speech is False


def test_committed_turn_sets_response_latency_anchor_from_reliable_stop_time() -> None:
    activity, task = _activity_for_end_of_turn()
    activity._user_speech_started_during_interruptible_agent_speech = False

    committed = activity.on_end_of_turn(
        _EndOfTurnInfo(
            skip_reply=False,
            new_transcript="예약 문의요",
            transcript_confidence=1.0,
            transcript_clock_suppressed=False,
            started_speaking_at=400.0,
            stopped_speaking_at=456.0,
            transcription_delay=0.1,
            end_of_turn_delay=0.2,
        )
    )

    assert committed is True
    assert activity._user_turn_completed_atask is task
    assert activity._last_eou_timestamp == 456.0


def test_non_suppressed_delayed_ignored_backchannel_is_consumed_without_reply_or_anchor() -> None:
    activity, _ = _activity_for_end_of_turn()

    committed = activity.on_end_of_turn(
        _EndOfTurnInfo(
            skip_reply=False,
            new_transcript="네",
            transcript_confidence=1.0,
            transcript_clock_suppressed=False,
            started_speaking_at=400.0,
            stopped_speaking_at=456.0,
            transcription_delay=0.1,
            end_of_turn_delay=0.2,
        )
    )

    assert committed is True
    assert activity._user_turn_completed_atask is None
    assert activity._last_eou_timestamp is None
    assert activity._user_speech_started_during_interruptible_agent_speech is False
