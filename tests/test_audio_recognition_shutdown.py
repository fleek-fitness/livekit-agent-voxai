import asyncio
from contextlib import suppress
from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition


@pytest.mark.asyncio
async def test_aclose_ignores_pre_cancelled_end_of_turn_task() -> None:
    """Regression for shutdown after user speech interrupts EOU detection."""

    recognition = AudioRecognition(
        MagicMock(),
        hooks=MagicMock(),
        stt=None,
        vad=None,
        turn_detection="manual",
        min_endpointing_delay=0.1,
        max_endpointing_delay=0.1,
    )

    async def pending_eou_task() -> None:
        await asyncio.sleep(60)

    task = asyncio.create_task(pending_eou_task())
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert task.done()
    assert task.cancelled()
    recognition._end_of_turn_task = task

    await asyncio.wait_for(recognition.aclose(), timeout=0.1)
