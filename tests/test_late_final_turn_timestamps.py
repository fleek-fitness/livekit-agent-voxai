"""A provider final that lands after its VAD burst's turn committed keeps its own speech time.

One VAD burst can carry two sentences that the STT finalizes separately. When the
first final commits the turn at the burst's end, the second final arrives with the
turn anchors already reset and no new VAD onset to set them again.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import llm, stt
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _recognition() -> tuple[AudioRecognition, MagicMock]:
    session = MagicMock()
    session.amd = None
    session._room_io = None
    session.options.interruption = {}
    session.options.transcription_timeout = None
    session.options.turn_handling = {"user_turn_limit": {}}
    hooks = MagicMock()
    hooks.retrieve_chat_ctx.return_value = llm.ChatContext.empty()
    hooks.on_end_of_turn.return_value = True
    recognition = AudioRecognition(
        session,
        hooks=hooks,
        endpointing=MagicMock(min_delay=0.1, max_delay=3.0),
        stt=MagicMock(),
        vad=MagicMock(),
        interruption_detection=None,
        turn_detection="vad",
        stt_aligned_transcript=True,
    )
    return recognition, hooks


def _final(text: str, start: float, end: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(text=text, language="en", start_time=start, end_time=end)],
    )


async def test_second_final_of_one_burst_keeps_its_speech_time() -> None:
    recognition, hooks = _recognition()
    origin = time.time() - 5.0
    recognition._stt_pipeline = MagicMock(input_started_at=origin)

    async def vad_event(kind: VADEventType, at: float) -> None:
        ev = VADEvent(
            type=kind,
            samples_index=0,
            timestamp=origin + at,
            speech_duration=0.0,
            silence_duration=0.0,
        )
        with patch("time.time", return_value=origin + at):
            await recognition._on_vad_event(ev)

    await vad_event(VADEventType.START_OF_SPEECH, 0.5)
    await recognition._on_stt_event(_final("My favorite tree is maple.", 0.5, 1.9))
    await vad_event(VADEventType.END_OF_SPEECH, 3.9)
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task

    await recognition._on_stt_event(_final("What is the launch code for that tree?", 2.2, 3.9))
    await recognition._end_of_turn_task

    assert hooks.on_end_of_turn.call_count == 2
    first, second = (call.args[0] for call in hooks.on_end_of_turn.call_args_list)
    assert first.new_transcript == "My favorite tree is maple."
    assert first.metrics.started_speaking_at == pytest.approx(origin + 0.5, abs=0.001)
    assert second.new_transcript == "What is the launch code for that tree?"
    assert second.metrics.started_speaking_at == pytest.approx(origin + 2.2, abs=0.001)
    assert second.metrics.stopped_speaking_at == pytest.approx(origin + 3.9, abs=0.001)
    assert second.metrics.end_of_turn_delay is not None
    assert second.metrics.transcription_delay is not None
