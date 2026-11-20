"""Public playhead read of _ParticipantAudioOutput (consumers must not read private fields)."""

from __future__ import annotations

import pytest

from livekit.agents.voice.room_io import _output as output_module
from livekit.agents.voice.room_io._output import PlaybackState, _ParticipantAudioOutput

pytestmark = pytest.mark.unit


class _FakeAudioSource:
    def __init__(self, *_args, **_kwargs) -> None:
        self.queued_duration = 0.0

    def clear_queue(self) -> None:
        self.queued_duration = 0.0


def _make_output(monkeypatch) -> _ParticipantAudioOutput:
    monkeypatch.setattr(output_module.rtc, "AudioSource", _FakeAudioSource)
    return _ParticipantAudioOutput(
        room=object(),
        sample_rate=24000,
        num_channels=1,
        track_publish_options=None,
        fade_out_ms=0,
    )


def test_playback_state_reports_the_playhead(monkeypatch) -> None:
    out = _make_output(monkeypatch)
    assert out.playback_state() == PlaybackState(position=0.0, run_offset=0.0, paused=False)

    out._source_pushed_duration = 5.0
    out._audio_source.queued_duration = 1.5
    out._run_offset = 2.0
    state = out.playback_state()
    assert state.position == pytest.approx(3.5)
    assert state.run_offset == 2.0 and state.paused is False


def test_playback_state_never_goes_negative_and_shows_pause(monkeypatch) -> None:
    out = _make_output(monkeypatch)
    out._source_pushed_duration = 1.0
    out._audio_source.queued_duration = 1.4
    out.pause()
    state = out.playback_state()
    assert state.position == 0.0 and state.paused is True
    out.resume()
    assert out.playback_state().paused is False


def test_playback_state_matches_what_playback_progressed_will_report(monkeypatch) -> None:
    out = _make_output(monkeypatch)
    out._source_pushed_duration = 4.0
    out._audio_source.queued_duration = 1.0
    expected = out._source_pushed_duration - out._audio_source.queued_duration
    assert out.playback_state().position == pytest.approx(expected)
