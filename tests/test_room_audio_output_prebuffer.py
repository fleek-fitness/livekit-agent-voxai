"""Playout prebuffer of _ParticipantAudioOutput.

Opt-in: prebuffer_ms=0 forwards every frame immediately. With it on, the first
prebuffer_ms of an answer is collected before playback starts; flush releases a short
answer, clear_buffer drops what is held, and a dry source collects again.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.room_io import _output as output_module
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

pytestmark = pytest.mark.unit

SR = 24000
FRAME_SAMPLES = SR // 20  # 50ms, matches AudioByteStream config


class _FakeAudioSource:
    def __init__(self, *_args, **_kwargs) -> None:
        self.captured: list[rtc.AudioFrame] = []
        self.queued_duration = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.captured.append(frame)
        self.queued_duration += frame.duration

    def clear_queue(self) -> None:
        self.queued_duration = 0.0

    async def wait_for_playout(self) -> None:
        return None


def _make(monkeypatch, *, prebuffer_ms: int):
    monkeypatch.setattr(output_module.rtc, "AudioSource", _FakeAudioSource)
    out = _ParticipantAudioOutput(
        room=object(),
        sample_rate=SR,
        num_channels=1,
        track_publish_options=None,
        prebuffer_ms=prebuffer_ms,
    )
    out._subscribed_fut.set_result(None)
    return out, out._audio_source


def _frame(ms: int = 50) -> rtc.AudioFrame:
    n = SR * ms // 1000
    data = np.full(n, 1000, dtype=np.int16).tobytes()
    return rtc.AudioFrame(data=data, sample_rate=SR, num_channels=1, samples_per_channel=n)


def _played(source: _FakeAudioSource) -> float:
    return sum(f.duration for f in source.captured)


async def _settle() -> None:
    await asyncio.sleep(0.02)


async def test_off_forwards_immediately(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=0)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await _settle()
    assert source.captured
    await utils.aio.cancel_and_wait(task)


async def test_holds_until_threshold_then_plays(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=300)
    task = asyncio.create_task(out._forward_audio())
    for _ in range(5):  # 250ms < 300ms
        await out.capture_frame(_frame())
    await _settle()
    assert source.captured == []

    for _ in range(2):  # past 300ms, with room for the byte stream's partial frame
        await out.capture_frame(_frame())
    await _settle()
    assert _played(source) >= 0.3
    await utils.aio.cancel_and_wait(task)


async def test_flush_releases_short_answer(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=300)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    await _settle()
    assert source.captured == []

    out.flush()
    await _settle()
    assert _played(source) == pytest.approx(0.1, abs=0.001)
    await utils.aio.cancel_and_wait(task)


async def test_clear_buffer_drops_held_audio(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=300)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    await _settle()
    out.clear_buffer()
    await _settle()
    assert source.captured == []
    await utils.aio.cancel_and_wait(task)


async def test_dry_source_collects_again(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=300)
    task = asyncio.create_task(out._forward_audio())
    for _ in range(8):
        await out.capture_frame(_frame())
    await _settle()
    first = _played(source)
    assert first >= 0.3

    source.queued_duration = 0.0  # playback drained mid-answer
    await out.capture_frame(_frame())
    await _settle()
    assert _played(source) == first  # held again

    for _ in range(8):
        await out.capture_frame(_frame())
    await _settle()
    assert _played(source) > first
    await utils.aio.cancel_and_wait(task)


async def test_stalled_provider_is_released_after_bounded_wait(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=300)
    out._prebuffer_max_wait_s = 0.05
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await asyncio.sleep(0.15)
    assert source.captured
    await utils.aio.cancel_and_wait(task)
