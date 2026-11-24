"""Playout prebuffer of _ParticipantAudioOutput.

Opt-in: prebuffer_ms=0 forwards every frame immediately. With it on, the first prebuffer_ms
of audio *length* is collected before playback starts (so the wait is short when audio
arrives in a burst and as long as the provider's pacing when it trickles in, capped by a
bounded wait). The source here drains on the real clock like rtc.AudioSource, so "dry
mid-answer" happens by time passing, not by editing a value.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.room_io import _output as output_module
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

pytestmark = pytest.mark.unit

SR = 24000
PRE_MS = 100


class _ClockAudioSource:
    """rtc.AudioSource stand-in: the queue drains with the monotonic clock."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.captured: list[tuple[float, rtc.AudioFrame]] = []
        self._end = 0.0

    @property
    def queued_duration(self) -> float:
        return max(0.0, self._end - time.monotonic())

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        now = time.monotonic()
        self.captured.append((now, frame))
        self._end = max(self._end, now) + frame.duration

    def clear_queue(self) -> None:
        self._end = 0.0

    async def wait_for_playout(self) -> None:
        await asyncio.sleep(self.queued_duration)


def _make(monkeypatch, *, prebuffer_ms: int = PRE_MS):
    monkeypatch.setattr(output_module.rtc, "AudioSource", _ClockAudioSource)
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


def _played(source: _ClockAudioSource) -> float:
    return sum(f.duration for _, f in source.captured)


async def _settle(seconds: float = 0.02) -> None:
    await asyncio.sleep(seconds)


async def _stop(out, task) -> None:
    if out._flush_task and not out._flush_task.done():
        await asyncio.wait_for(out._flush_task, 2)
    await utils.aio.cancel_and_wait(task)


async def test_off_forwards_immediately(monkeypatch) -> None:
    out, source = _make(monkeypatch, prebuffer_ms=0)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await _settle()
    assert source.captured
    await _stop(out, task)


async def test_burst_starts_without_waiting(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    task = asyncio.create_task(out._forward_audio())
    t0 = time.monotonic()
    for _ in range(4):  # 200ms arrives at once
        await out.capture_frame(_frame())
    await _settle()
    assert source.captured
    assert source.captured[0][0] - t0 < 0.05
    await _stop(out, task)


async def test_trickle_waits_for_audio_length_not_longer(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    task = asyncio.create_task(out._forward_audio())
    t0 = time.monotonic()
    await out.capture_frame(_frame())
    await _settle(0.06)
    assert source.captured == []  # 50ms collected < 100ms
    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    await _settle()
    assert source.captured
    assert source.captured[0][0] - t0 >= 0.06
    await _stop(out, task)


async def test_short_answer_flush_is_prompt_and_complete(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame(30))
    await _settle()
    assert source.captured == []
    t0 = time.monotonic()
    out.flush()
    await _settle()
    assert source.captured and source.captured[0][0] - t0 < 0.05
    assert _played(source) == pytest.approx(0.03, abs=0.001)
    await _stop(out, task)


async def test_mid_answer_exhaustion_collects_again_without_loss(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    task = asyncio.create_task(out._forward_audio())
    for _ in range(4):
        await out.capture_frame(_frame())
    await _settle()
    first = _played(source)
    assert first >= PRE_MS / 1000

    await asyncio.sleep(source.queued_duration + 0.05)  # really runs dry
    assert source.queued_duration == 0.0
    await out.capture_frame(_frame())  # one late frame, 50ms
    await _settle(0.03)
    assert _played(source) == first  # held, not played as a short blip

    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    out.flush()
    await _settle()
    assert _played(source) == pytest.approx(0.35, abs=0.001)  # 200ms + 150ms, none lost
    await _stop(out, task)


async def test_stalled_provider_is_released_after_bounded_wait(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    out._prebuffer_max_wait_s = 0.05
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await asyncio.sleep(0.15)
    assert _played(source) > 0  # released although under 100ms
    await _stop(out, task)


async def test_pause_then_one_frame_then_resume_does_not_bypass_buffer(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    task = asyncio.create_task(out._forward_audio())
    out.pause()
    await out.capture_frame(_frame())
    await _settle()
    out.resume()
    await _settle(0.05)
    assert source.captured == []  # 50ms < 100ms: still collecting after resume

    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    out.flush()
    await _settle()
    assert _played(source) == pytest.approx(0.15, abs=0.001)  # nothing lost
    await _stop(out, task)


async def test_pause_during_hold_waits_for_resume(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await _settle()
    out.pause()
    await out.capture_frame(_frame())
    await out.capture_frame(_frame())  # enough, but paused
    await _settle()
    assert source.captured == []
    out.resume()
    out.flush()
    await _settle()
    assert _played(source) == pytest.approx(0.15, abs=0.001)
    await _stop(out, task)


async def test_clear_and_flush_during_hold_then_next_answer_is_clean(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    finished = []
    out.on("playback_finished", finished.append)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await _settle()
    out.clear_buffer()
    out.flush()
    await _settle(0.05)
    assert source.captured == []  # held audio dropped, none leaked
    assert len(finished) == 1 and finished[0].interrupted
    assert out._buffered_duration == 0.0 and not out._flush_requested

    # next answer behaves like a fresh one: held, then played whole
    await out.capture_frame(_frame())
    await _settle(0.03)
    assert source.captured == []
    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    out.flush()
    await _settle()
    assert _played(source) == pytest.approx(0.15, abs=0.001)
    await _stop(out, task)


async def test_pause_flush_clear_then_immediate_resume_drops_old_frame(monkeypatch) -> None:
    out, source = _make(monkeypatch)
    finished = []
    out.on("playback_finished", finished.append)
    task = asyncio.create_task(out._forward_audio())
    await out.capture_frame(_frame())
    await _settle()  # the frame is being held, then pause arrives during the hold
    out.pause()
    await _settle()
    # flush releases the hold; the forwarder now waits for resume. clear and resume then
    # arrive back to back, before the playout task has moved the interruption generation.
    out.flush()
    await _settle()
    out.clear_buffer()
    out.resume()
    await _settle(0.05)
    assert source.captured == []  # nothing of the old answer reaches the source
    assert len(finished) == 1 and finished[0].interrupted

    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    await out.capture_frame(_frame())
    out.flush()
    await _settle()
    assert _played(source) == pytest.approx(0.15, abs=0.001)  # next answer plays whole
    await _stop(out, task)
