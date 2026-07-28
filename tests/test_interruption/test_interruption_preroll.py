from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from time import perf_counter_ns
from unittest.mock import AsyncMock, patch

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionResponse,
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechStartedSentinel,
)

pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000


def _frame(samples: list[int]) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.asarray(samples, dtype=np.int16).tobytes(),
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(samples),
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0)


def _detector() -> AdaptiveInterruptionDetector:
    return AdaptiveInterruptionDetector(
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=AsyncMock(spec=aiohttp.ClientSession),
        audio_prefix_duration=4 / _SAMPLE_RATE,
        detection_interval=1 / _SAMPLE_RATE,
    )


@pytest.mark.asyncio
async def test_first_overlap_contains_pre_agent_pcm_without_early_inference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.agents")
    detector = _detector()
    stream = detector.stream()
    predict = AsyncMock(
        return_value=InterruptionResponse(
            created_at=perf_counter_ns(),
            is_bargein=False,
            prediction_duration=0.001,
            probabilities=np.asarray([0.1], dtype=np.float32),
        )
    )
    with patch.object(stream, "predict", predict):
        try:
            # Four prefix samples and four user-speech samples arrive before agent playout.
            stream.push_frame(_frame([1, 2, 3, 4]))
            stream.push_frame(_frame([10, 11, 12, 13]))
            await _wait_until(lambda: len(stream._audio_buffer) == 8)
            predict.assert_not_awaited()

            stream.push_frame(_AgentSpeechStartedSentinel())
            await _wait_until(lambda: stream._agent_speech_started)
            np.testing.assert_array_equal(
                stream._audio_buffer.read(),
                [1, 2, 3, 4, 10, 11, 12, 13],
            )
            predict.assert_not_awaited()

            stream.push_frame(
                _OverlapSpeechStartedSentinel(
                    speech_duration=4 / _SAMPLE_RATE,
                    started_at=100.0,
                )
            )
            stream.push_frame(_frame([20, 21]))
            await _wait_until(lambda: predict.await_count == 1)

            assert predict.await_args is not None
            model_input = predict.await_args.args[0]
            np.testing.assert_array_equal(
                model_input,
                [1, 2, 3, 4, 10, 11, 12, 13, 20, 21],
            )
            overlap_log = next(
                record
                for record in caplog.records
                if record.message == "overlap speech started, starting interruption inference"
            )
            log_fields = vars(overlap_log)
            assert log_fields["buffer_samples_before_trim"] == 8
            assert log_fields["buffer_samples_after_trim"] == 8
            assert log_fields["prefix_samples_target"] == 4
            assert log_fields["prefix_samples_available"] == 4
            assert log_fields["prefix_underflow_samples"] == 0
        finally:
            await stream.aclose()


@pytest.mark.asyncio
async def test_agent_boundaries_reset_inference_state_but_keep_pcm_history() -> None:
    detector = _detector()
    stream = detector.stream()

    try:
        stream.push_frame(_frame([1, 2, 3]))
        stream.push_frame(_AgentSpeechStartedSentinel())
        stream.push_frame(
            _OverlapSpeechStartedSentinel(
                speech_duration=2 / _SAMPLE_RATE,
                started_at=100.0,
            )
        )
        await _wait_until(lambda: stream._overlap_started)

        stream.push_frame(_AgentSpeechEndedSentinel())
        stream.push_frame(_frame([4, 5]))
        await _wait_until(lambda: len(stream._audio_buffer) == 5)

        assert not stream._agent_speech_started
        assert not stream._overlap_started
        np.testing.assert_array_equal(stream._audio_buffer.read(), [1, 2, 3, 4, 5])
    finally:
        await stream.aclose()
