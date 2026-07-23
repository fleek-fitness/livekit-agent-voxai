"""Tests for interruption detection failover (retry + error-emission) behavior.

Covers:
- HTTP stream: timeout, 429, non-retryable errors
- WS stream: connection timeout, connection 429, cache-based inference timeout
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents._exceptions import APIError
from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionDetectionError,
    InterruptionDetectionStateChangedEvent,
    InterruptionHttpStream,
    InterruptionWebSocketStream,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit

MAX_RETRY = 2
CONN_OPTIONS = APIConnectOptions(max_retry=MAX_RETRY, retry_interval=0.0, timeout=1.0)


def _make_audio_frame(*, num_samples: int = 1600, sample_rate: int = 16000) -> rtc.AudioFrame:
    data = np.zeros(num_samples, dtype=np.int16).tobytes()
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )


def _create_detector(
    mock_session: AsyncMock, *, use_proxy: bool, inference_timeout: float = 0.1
) -> AdaptiveInterruptionDetector:
    detector = AdaptiveInterruptionDetector(
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=mock_session,
        inference_timeout=inference_timeout,
        transport="websocket" if use_proxy else "http",
    )
    return detector


def _collect_errors(
    detector: AdaptiveInterruptionDetector,
) -> list[InterruptionDetectionError]:
    errors: list[InterruptionDetectionError] = []
    detector.on("error", lambda e: errors.append(e))
    return errors


def _collect_states(
    detector: AdaptiveInterruptionDetector,
) -> list[InterruptionDetectionStateChangedEvent]:
    states: list[InterruptionDetectionStateChangedEvent] = []
    detector.on("state_changed", states.append)
    return states


async def _feed_audio_continuously(
    stream: InterruptionHttpStream | InterruptionWebSocketStream,
    stop_event: asyncio.Event,
) -> None:
    """Feed overlap audio frames until stop_event is set."""
    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.5, started_at=time.time()))
    while not stop_event.is_set():
        try:
            stream.push_frame(_make_audio_frame())
        except RuntimeError:
            break
        await asyncio.sleep(0.001)


async def _wait_for_stream_failure(
    stream: InterruptionHttpStream | InterruptionWebSocketStream,
) -> Exception | None:
    """Wait for the stream's background task to complete and return the exception."""
    stop = asyncio.Event()
    feed_task = asyncio.create_task(_feed_audio_continuously(stream, stop))

    try:
        # wait for the internal _task to complete (it will fail after retries)
        try:
            await stream._task
        except Exception as exc:
            return exc
        return None
    finally:
        stop.set()
        await feed_task
        await stream.aclose()


def _mock_request_info() -> MagicMock:
    ri = MagicMock()
    ri.real_url = "http://localhost:9999/bargein"
    ri.method = "GET"
    ri.url = "http://localhost:9999/bargein"
    ri.headers = {}
    return ri


# ---------------------------------------------------------------------------
# HTTP stream tests
# ---------------------------------------------------------------------------


class TestHttpTimeout:
    @pytest.mark.asyncio
    async def test_retries_then_emits_unrecoverable(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="livekit.agents")
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("test timeout"))
        mock_session.post.return_value = mock_ctx

        detector = _create_detector(mock_session, use_proxy=False)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        stream.push_frame(_AgentSpeechStartedSentinel())
        stream.push_frame(
            _OverlapSpeechStartedSentinel(speech_duration=0.5, started_at=time.time())
        )
        stream.push_frame(_make_audio_frame())
        try:
            with pytest.raises(APIError) as exc_info:
                await asyncio.wait_for(stream._task, timeout=1.0)
            exc: Exception | None = exc_info.value
        finally:
            await stream.aclose()

        assert exc is not None, f"Expected exception, got None. Errors: {errors}"
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


# there is no 429 in HTTP when hosted on LiveKit Cloud, so this is actually redundant
class TestHttp429:
    @pytest.mark.asyncio
    async def test_retries_then_emits_unrecoverable(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_resp = MagicMock()
        mock_resp.status = 429
        mock_resp.raise_for_status = Mock(
            side_effect=aiohttp.ClientResponseError(
                request_info=_mock_request_info(),
                history=(),
                status=429,
                message="Too Many Requests",
            )
        )
        mock_resp.text = AsyncMock(return_value="rate limited")
        mock_resp.json = AsyncMock(return_value={})
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value = mock_ctx

        detector = _create_detector(mock_session, use_proxy=False)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None, f"Expected exception, got None. Errors: {errors}"
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1


class TestHttpNonRetryable:
    @pytest.mark.asyncio
    async def test_immediate_fallback(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=APIError("fatal error", retryable=False))
        mock_session.post.return_value = mock_ctx

        detector = _create_detector(mock_session, use_proxy=False)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1


class TestHttpLifecycle:
    @pytest.mark.asyncio
    async def test_reopens_active_after_stream_close(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        detector = _create_detector(mock_session, use_proxy=False)

        first_stream = detector.stream(conn_options=CONN_OPTIONS)
        await first_stream.aclose()
        assert detector.state == "closed"

        second_stream = detector.stream(conn_options=CONN_OPTIONS)
        try:
            assert detector.state == "active"
        finally:
            await second_stream.aclose()

    @pytest.mark.asyncio
    async def test_stream_close_preserves_terminal_fallback(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        detector = _create_detector(mock_session, use_proxy=False)
        stream = detector.stream(conn_options=CONN_OPTIONS)
        detector.fail(RuntimeError("terminal"))

        await stream.aclose()

        assert detector.state == "fallback"


# ---------------------------------------------------------------------------
# WebSocket stream tests
# ---------------------------------------------------------------------------


class TestWsConnectionTimeout:
    @pytest.mark.asyncio
    async def test_retries_then_emits_unrecoverable(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=asyncio.TimeoutError("connect timeout"))

        detector = _create_detector(mock_session, use_proxy=True)
        errors = _collect_errors(detector)
        states = _collect_states(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == MAX_RETRY
        assert len(unrecoverable_errors) == 1
        assert states[0].state == "reconnecting"
        assert len([event for event in states if event.state == "reconnecting"]) == MAX_RETRY
        assert states[-1].state in ("fallback", "closed")


class TestWsConnection429:
    @pytest.mark.asyncio
    async def test_retries_then_emits_unrecoverable(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=_mock_request_info(),
                history=(),
                status=429,
                message="Too Many Requests",
            )
        )

        detector = _create_detector(mock_session, use_proxy=True)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == MAX_RETRY
        assert len(unrecoverable_errors) == 1


class TestWsHandshake:
    @pytest.mark.asyncio
    async def test_becomes_active_only_after_session_created(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        mock_ws.send_str = AsyncMock()
        mock_ws.closed = False
        mock_ws.close_code = None
        mock_ws.close = AsyncMock(return_value=True)
        responses = [
            aiohttp.WSMessage(
                type=aiohttp.WSMsgType.TEXT,
                data='{"type":"session.created"}',
                extra=None,
            )
        ]

        async def _receive() -> aiohttp.WSMessage:
            if responses:
                return responses.pop()
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        mock_ws.receive = _receive
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        detector = _create_detector(mock_session, use_proxy=True)
        states = _collect_states(detector)

        assert detector.state == "connecting"
        stream = detector.stream(conn_options=CONN_OPTIONS)
        try:
            await asyncio.wait_for(_wait_until(lambda: detector.state == "active"), timeout=1.0)
            assert [event.state for event in states] == ["active"]
        finally:
            await stream.aclose()


class TestWsLifecycle:
    @pytest.mark.asyncio
    async def test_end_input_stops_timeout_watcher_and_completes_stream(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
        mock_ws.closed = False
        mock_ws.close_code = None
        mock_ws.close = AsyncMock(return_value=True)
        session_close_sent = asyncio.Event()

        async def _send_str(data: str) -> None:
            if '"session.close"' in data:
                session_close_sent.set()

        receive_count = 0

        async def _receive() -> aiohttp.WSMessage:
            nonlocal receive_count
            receive_count += 1
            if receive_count == 1:
                return aiohttp.WSMessage(
                    type=aiohttp.WSMsgType.TEXT,
                    data='{"type":"session.created"}',
                    extra=None,
                )
            await session_close_sent.wait()
            if receive_count == 2:
                return aiohttp.WSMessage(
                    type=aiohttp.WSMsgType.TEXT,
                    data='{"type":"session.closed"}',
                    extra=None,
                )
            return aiohttp.WSMessage(
                type=aiohttp.WSMsgType.CLOSED,
                data=None,
                extra=None,
            )

        mock_ws.send_str = AsyncMock(side_effect=_send_str)
        mock_ws.receive = _receive
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        detector = _create_detector(
            mock_session,
            use_proxy=True,
            inference_timeout=0.05,
        )
        stream = detector.stream(conn_options=CONN_OPTIONS)

        stream.end_input()
        try:
            await asyncio.wait_for(stream._task, timeout=1.0)
        finally:
            await stream.aclose()

        assert session_close_sent.is_set()


class TestWsCacheTimeout:
    @pytest.mark.asyncio
    async def test_times_out_without_another_audio_frame(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        inference_timeout = 0.05

        def _make_mock_ws() -> MagicMock:
            mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
            mock_ws.send_str = AsyncMock()
            mock_ws.closed = False
            mock_ws.close_code = None

            async def _slow_send_bytes(*_args: object, **_kwargs: object) -> None:
                await asyncio.sleep(inference_timeout / 2)

            mock_ws.send_bytes = _slow_send_bytes

            received_handshake = False

            async def _receive_hang() -> aiohttp.WSMessage:
                nonlocal received_handshake
                if not received_handshake:
                    received_handshake = True
                    return aiohttp.WSMessage(
                        type=aiohttp.WSMsgType.TEXT,
                        data='{"type":"session.created"}',
                        extra=None,
                    )
                await asyncio.sleep(3600)
                return aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

            mock_ws.receive = _receive_hang
            mock_ws.close = AsyncMock(return_value=True)
            return mock_ws

        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: _make_mock_ws())

        detector = _create_detector(
            mock_session, use_proxy=True, inference_timeout=inference_timeout
        )
        errors = _collect_errors(detector)
        # A zero retry budget makes the assertion about the independent timeout
        # deterministic: no second audio frame is needed to discover the failure.
        stream = detector.stream(
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)
        )

        stream.push_frame(_AgentSpeechStartedSentinel())
        stream.push_frame(
            _OverlapSpeechStartedSentinel(speech_duration=0.5, started_at=time.time())
        )
        stream.push_frame(_make_audio_frame())
        try:
            with pytest.raises(APIError) as exc_info:
                await asyncio.wait_for(stream._task, timeout=1.0)
            exc: Exception | None = exc_info.value
        finally:
            await stream.aclose()

        assert exc is not None
        assert isinstance(exc, APIError)

        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(unrecoverable_errors) == 1
        assert "timed out" in str(unrecoverable_errors[0].error)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0)
