from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents.voice.agent_activity import AgentActivity, _PausedSpeechInfo
from livekit.agents.voice.generation import perform_audio_forwarding
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit


class _TrackingAudioOutput(FakeAudioOutput):
    def __init__(self) -> None:
        super().__init__(sample_rate=16_000, can_pause=True)
        self.resume_calls = 0
        self.frames: list[rtc.AudioFrame] = []

    def resume(self) -> None:
        self.resume_calls += 1
        super().resume()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)
        await super().capture_frame(frame)


class _Timer:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
        samples_per_channel=160,
    )


async def _one_frame():
    yield _frame()


async def test_paused_handle_blocks_late_audio_forwarding_until_resume() -> None:
    handle = SpeechHandle.create()
    handle._playout_allowed.clear()
    output = _TrackingAudioOutput()

    task, audio_out = perform_audio_forwarding(
        speech_handle=handle,
        audio_output=output,
        tts_output=_one_frame(),
    )
    await asyncio.sleep(0)

    assert task.done() is False
    assert output.resume_calls == 0
    assert output.frames == []

    handle._playout_allowed.set()
    await task

    assert output.resume_calls == 1
    assert len(output.frames) == 1
    if not audio_out.first_frame_fut.done():
        audio_out.first_frame_fut.cancel()


async def test_interrupted_paused_handle_never_resumes_audio() -> None:
    handle = SpeechHandle.create()
    handle._playout_allowed.clear()
    output = _TrackingAudioOutput()

    task, audio_out = perform_audio_forwarding(
        speech_handle=handle,
        audio_output=output,
        tts_output=_one_frame(),
    )
    await asyncio.sleep(0)

    handle.interrupt()
    await task

    assert output.resume_calls == 0
    assert output.frames == []
    audio_out.first_frame_fut.cancel()


async def test_false_interruption_reopens_current_handle_playout() -> None:
    handle = SpeechHandle.create()
    output = _TrackingAudioOutput()
    emitted: list[object] = []
    resume_event = asyncio.Event()

    def _emit(*args: object) -> None:
        emitted.append(args)
        resume_event.set()

    activity = AgentActivity.__new__(AgentActivity)
    activity._session = SimpleNamespace(
        agent_state="speaking",
        options=SimpleNamespace(interruption={"resume_false_interruption": True}),
        output=SimpleNamespace(audio=output),
        _loop=asyncio.get_running_loop(),
        _update_agent_state=lambda *args, **kwargs: None,
        emit=_emit,
    )
    activity._current_speech = handle
    activity._paused_speech = None
    activity._false_interruption_timer = None
    activity._audio_recognition = None
    activity._interruption_detection_enabled = False

    activity._update_paused_speech(handle, timeout=0)
    assert handle._playout_allowed.is_set() is False

    activity._start_false_interruption_timer(0)
    await asyncio.wait_for(resume_event.wait(), timeout=1)

    assert handle._playout_allowed.is_set() is True
    assert output.resume_calls == 1
    assert activity._paused_speech is None
    assert len(emitted) == 1


def test_confirmed_pause_cancels_and_blocks_false_resume_timer() -> None:
    handle = SpeechHandle.create()
    timer = _Timer()
    activity = AgentActivity.__new__(AgentActivity)
    activity._paused_speech = _PausedSpeechInfo(
        handle=handle,
        agent_state="speaking",
        timeout=0.6,
    )
    activity._false_interruption_timer = timer

    assert activity.confirm_paused_speech(handle) is True
    assert timer.cancelled is True
    assert activity._false_interruption_timer is None
    assert activity._paused_speech.confirmed is True

    activity._start_false_interruption_timer(0.6)
    assert activity._false_interruption_timer is None


def test_confirm_pause_rejects_a_different_handle() -> None:
    paused_handle = SpeechHandle.create()
    next_handle = SpeechHandle.create()
    timer = _Timer()
    activity = AgentActivity.__new__(AgentActivity)
    activity._paused_speech = _PausedSpeechInfo(
        handle=paused_handle,
        agent_state="speaking",
        timeout=0.6,
    )
    activity._false_interruption_timer = timer

    assert activity.confirm_paused_speech(next_handle) is False
    assert timer.cancelled is False
    assert activity._paused_speech.confirmed is False
