from __future__ import annotations

import time
from types import SimpleNamespace

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.dynamic_interruption import DynamicInterruptionManager


def _opts(
    *,
    min_words: int = 2,
    enable_dynamic_interruption: bool = False,
    conversation_continuity_threshold: float = 8.0,
    enable_adaptive_endpointing: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        interruption={"min_words": min_words},
        enable_dynamic_interruption=enable_dynamic_interruption,
        conversation_continuity_threshold=conversation_continuity_threshold,
        enable_adaptive_endpointing=enable_adaptive_endpointing,
    )


def _activity(opts: SimpleNamespace) -> AgentActivity:
    activity = AgentActivity.__new__(AgentActivity)
    activity._opts = opts
    activity._dynamic_interruption = DynamicInterruptionManager(opts)
    return activity


def _audio_recognition(opts: SimpleNamespace, multiplier: float) -> AudioRecognition:
    audio_recognition = AudioRecognition.__new__(AudioRecognition)
    audio_recognition._session = SimpleNamespace(options=opts)
    audio_recognition._hooks = SimpleNamespace(
        _dynamic_interruption=SimpleNamespace(get_endpointing_multiplier=lambda: multiplier)
    )
    audio_recognition._endpointing = SimpleNamespace(max_delay=10.0)
    return audio_recognition


def test_dynamic_interruption_disabled_is_noop() -> None:
    opts = _opts(min_words=3)
    activity = _activity(opts)
    activity._dynamic_interruption.conversation_state.last_user_speech_end_time = time.time()
    activity._dynamic_interruption._current_speech_within_continuity = True

    assert activity._get_dynamic_min_interruption_words() == 3


def test_dynamic_interruption_enabled_consults_continuity_threshold() -> None:
    opts = _opts(
        min_words=2,
        enable_dynamic_interruption=True,
        conversation_continuity_threshold=2.0,
    )
    activity = _activity(opts)

    activity._dynamic_interruption.conversation_state.last_user_speech_end_time = time.time() - 1.0
    activity._capture_dynamic_continuity_at_speech_start()
    assert activity._get_dynamic_min_interruption_words() == 0

    activity._dynamic_interruption.conversation_state.last_user_speech_end_time = time.time() - 3.0
    activity._capture_dynamic_continuity_at_speech_start()
    assert activity._get_dynamic_min_interruption_words() == 2


def test_adaptive_endpointing_disabled_no_multiplier() -> None:
    audio_recognition = _audio_recognition(
        _opts(enable_adaptive_endpointing=False),
        multiplier=2.0,
    )

    assert audio_recognition._apply_adaptive_endpointing_multiplier(2.0) == 2.0


def test_adaptive_endpointing_enabled_applies_multiplier() -> None:
    audio_recognition = _audio_recognition(
        _opts(enable_adaptive_endpointing=True),
        multiplier=2.0,
    )

    assert audio_recognition._apply_adaptive_endpointing_multiplier(2.0) == 4.0
