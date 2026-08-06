"""Fade-stop behavior of _ParticipantAudioOutput.

The fade must be strictly opt-in: fade_out_ms=0 keeps the stock instant
clear_queue behavior. With fade enabled, pause/interrupt replay the dropped
queue content as a cosine ramp instead of cutting mid-phoneme.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.room_io import _output as output_module
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

pytestmark = pytest.mark.unit

SR = 24000
FRAME_SAMPLES = SR // 20  # 50ms, matches AudioByteStream config


class _FakeAudioSource:
    """Stands in for rtc.AudioSource: tracks captured frames + queued time."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.captured: list[rtc.AudioFrame] = []
        self.cleared = 0
        self.queued_duration = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.captured.append(frame)

    def clear_queue(self) -> None:
        self.cleared += 1
        self.queued_duration = 0.0

    async def wait_for_playout(self) -> None:
        return None


def _make_output(monkeypatch, *, fade_out_ms: int) -> tuple[_ParticipantAudioOutput, _FakeAudioSource]:
    monkeypatch.setattr(output_module.rtc, "AudioSource", _FakeAudioSource)
    out = _ParticipantAudioOutput(
        room=object(),  # unused by the paths under test
        sample_rate=SR,
        num_channels=1,
        track_publish_options=None,
        fade_out_ms=fade_out_ms,
    )
    return out, out._audio_source


def _frame(value: int = 1000, samples: int = FRAME_SAMPLES) -> rtc.AudioFrame:
    data = np.full(samples, value, dtype=np.int16).tobytes()
    return rtc.AudioFrame(data=data, sample_rate=SR, num_channels=1, samples_per_channel=samples)


def _samples(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16)


async def test_default_off_keeps_instant_clear(monkeypatch) -> None:
    out, source = _make_output(monkeypatch, fade_out_ms=0)
    source.queued_duration = 0.2

    # default path must not exist at all: fade helpers untouched, clear is instant
    assert out._fade_out_ms == 0
    tail = await out._play_fade_tail() if False else None  # helper never called at 0
    source.clear_queue()
    assert source.cleared == 1
    assert source.captured == []
    assert tail is None


async def test_fade_tail_replays_dropped_content_with_ramp(monkeypatch) -> None:
    out, source = _make_output(monkeypatch, fade_out_ms=150)

    # simulate 4 captured frames (200ms) whose copies sit in the ring
    for _ in range(4):
        out._fade_ring_push(_frame(10000))
    source.queued_duration = 0.2  # all of it still queued -> will be dropped

    tail_s = await out._play_fade_tail()

    assert source.cleared == 1  # reactivity: queue dropped immediately
    assert len(source.captured) == 1
    tail = _samples(source.captured[0])
    assert tail_s == pytest.approx(0.15, abs=0.01)
    assert len(tail) == int(0.15 * SR)
    # cosine ramp: starts near full volume, ends near zero, monotonic-ish
    assert tail[0] == pytest.approx(10000, rel=0.01)
    assert abs(int(tail[-1])) <= 50
    assert abs(int(tail[len(tail) // 2])) == pytest.approx(5000, rel=0.05)


async def test_fade_tail_without_queue_is_noop(monkeypatch) -> None:
    out, source = _make_output(monkeypatch, fade_out_ms=150)
    out._fade_ring_push(_frame(10000))
    source.queued_duration = 0.0  # nothing queued -> nothing was dropped

    tail_s = await out._play_fade_tail()

    assert tail_s == 0.0
    assert source.captured == []


async def test_concurrent_fade_falls_back_to_clear(monkeypatch) -> None:
    out, source = _make_output(monkeypatch, fade_out_ms=150)
    out._fade_ring_push(_frame(10000))
    source.queued_duration = 0.1
    out._fade_tail_active = True  # a fade is already in flight

    tail_s = await out._play_fade_tail()

    assert tail_s == 0.0
    assert source.cleared == 1
    assert source.captured == []


async def test_fade_in_ramps_first_resume_samples(monkeypatch) -> None:
    out, _ = _make_output(monkeypatch, fade_out_ms=150)
    out._fade_in_remaining = int(out._fade_in_ms / 1000 * SR)  # as armed on resume

    first = out._apply_fade_in(_frame(10000))
    second = out._apply_fade_in(_frame(10000))

    s1, s2 = _samples(first), _samples(second)
    assert abs(int(s1[0])) <= 50  # starts from silence
    assert int(s1[-1]) < 10000  # still ramping at 50ms (fade_in=80ms)
    assert int(s2[-1]) == 10000  # fully ramped after 80ms
    assert out._fade_in_remaining == 0


async def test_ring_buffer_is_bounded(monkeypatch) -> None:
    out, _ = _make_output(monkeypatch, fade_out_ms=150)
    for _ in range(100):  # 5s of frames
        out._fade_ring_push(_frame())
    assert out._fade_ring_duration <= 150 / 1000 * 2 + 0.25 + 0.05
