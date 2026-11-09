"""text-to-dialogue sends char times relative to each message; words must stay ordered across messages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.plugins.elevenlabs.tts import _accumulate_alignment

pytestmark = pytest.mark.unit


class _Emitter:
    def __init__(self) -> None:
        self.words = []

    def push_timed_transcript(self, words) -> None:
        self.words.extend(words)


def _stream():
    return SimpleNamespace(
        _text_buffer="", _start_times_ms=[], _durations_ms=[], _alignment_offset_ms=0
    )


def test_chunk_relative_messages_become_stream_relative() -> None:
    stream, emitter = _stream(), _Emitter()
    # Each message restarts its char times at 0, as the text-to-dialogue API sends them.
    _accumulate_alignment(
        stream, emitter, list("안녕 "), [0, 80, 160], [80, 80, 80], chunk_relative=True
    )
    _accumulate_alignment(
        stream,
        emitter,
        list("하세요 "),
        [0, 160, 320, 480],
        [160, 160, 160, 80],
        chunk_relative=True,
    )
    _accumulate_alignment(stream, emitter, list("네 "), [0, 80], [80, 80], chunk_relative=True)
    starts = [w.start_time for w in emitter.words]
    ends = [w.end_time for w in emitter.words]
    assert starts == sorted(starts)
    assert all(end >= start for start, end in zip(starts, ends, strict=True))
    assert stream._alignment_offset_ms == 240 + 560 + 160


def test_stream_relative_messages_keep_their_times() -> None:
    stream, emitter = _stream(), _Emitter()
    _accumulate_alignment(stream, emitter, list("안녕 "), [0, 80, 160], [80, 80, 80])
    _accumulate_alignment(
        stream, emitter, list("하세요 "), [240, 400, 560, 720], [160, 160, 160, 80]
    )
    starts = [w.start_time for w in emitter.words]
    assert starts == sorted(starts)
    assert stream._alignment_offset_ms == 0  # websocket path never offsets
