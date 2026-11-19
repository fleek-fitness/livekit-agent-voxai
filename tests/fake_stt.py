from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from typing import Literal

from pydantic import BaseModel

from livekit import rtc
from livekit.agents import NOT_GIVEN, LanguageCode, NotGivenOr, utils
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.audio import AudioBuffer


class RecognizeSentinel:
    pass


class FakeFinal(BaseModel):
    """One extra final of a speech, with its own audio span and arrival time (scripted)."""

    text: str
    start_time: float
    end_time: float
    arrive_at: float
    finalize_onset: bool = False


class FakeUserSpeech(BaseModel):
    type: Literal["user_speech"] = "user_speech"
    start_time: float
    end_time: float
    transcript: str  # empty string fires VAD SOS/EOS only — no STT events
    stt_delay: float
    final: bool = True
    # how the adapter reports the call-clock span of the final text:
    # "not_given" = provider without spans, "given" = span of the speech, "none" = unprovable
    span: Literal["not_given", "given", "none"] = "not_given"
    # the final answers the finalize sent for this speech's VAD segment
    finalize_onset: bool = False
    # an empty speech: the finalize for this VAD segment ended without text
    empty_finalize: bool = False
    # when set, the main final covers [start_time, main_end] and arrives at its own time
    main_end: float | None = None
    main_arrive_at: float | None = None
    extra_finals: list[FakeFinal] = []

    def speed_up(self, factor: float) -> FakeUserSpeech:
        obj = copy.deepcopy(self)
        obj.start_time /= factor
        obj.end_time /= factor
        obj.stt_delay /= factor
        for attr in ("main_end", "main_arrive_at"):
            if getattr(obj, attr) is not None:
                setattr(obj, attr, getattr(obj, attr) / factor)
        for extra in obj.extra_finals:
            extra.start_time /= factor
            extra.end_time /= factor
            extra.arrive_at /= factor
        return obj


class FakeSTT(STT):
    def __init__(
        self,
        *,
        fake_exception: Exception | None = None,
        fake_transcript: str | None = None,
        fake_timeout: float | None = None,
        fake_user_speeches: list[FakeUserSpeech] | None = None,
        fake_require_audio: bool = False,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
        )

        self._fake_exception = fake_exception
        self._fake_transcript = fake_transcript
        self._fake_timeout = fake_timeout
        self._fake_require_audio = fake_require_audio

        if fake_user_speeches is not None:
            fake_user_speeches = sorted(fake_user_speeches, key=lambda x: x.start_time)
            for prev, next in zip(fake_user_speeches[:-1], fake_user_speeches[1:], strict=False):
                if prev.end_time > next.start_time:
                    raise ValueError("fake user speeches overlap")
        self._fake_user_speeches = fake_user_speeches
        self._done_fut = asyncio.Future[None]()

        self.streams: list[FakeRecognizeStream] = []
        self._recognize_ch = utils.aio.Chan[RecognizeSentinel]()
        self._stream_ch = utils.aio.Chan[FakeRecognizeStream]()

    def update_options(
        self,
        *,
        fake_exception: NotGivenOr[Exception | None] = NOT_GIVEN,
        fake_transcript: NotGivenOr[str | None] = NOT_GIVEN,
        fake_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(fake_exception):
            self._fake_exception = fake_exception

        if utils.is_given(fake_transcript):
            self._fake_transcript = fake_transcript

        if utils.is_given(fake_timeout):
            self._fake_timeout = fake_timeout

    @property
    def recognize_ch(self) -> utils.aio.ChanReceiver[RecognizeSentinel]:
        return self._recognize_ch

    @property
    def stream_ch(self) -> utils.aio.ChanReceiver[FakeRecognizeStream]:
        return self._stream_ch

    @property
    def fake_user_speeches(self) -> list[FakeUserSpeech]:
        return self._fake_user_speeches

    @property
    def fake_user_speeches_done(self) -> asyncio.Future[None]:
        return self._done_fut

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        if self._fake_timeout is not None:
            await asyncio.sleep(self._fake_timeout)

        if self._fake_exception is not None:
            raise self._fake_exception

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(text=self._fake_transcript or "", language=LanguageCode(language or ""))
            ],
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        self._recognize_ch.send_nowait(RecognizeSentinel())
        return await super().recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> FakeRecognizeStream:
        stream = FakeRecognizeStream(
            stt=self,
            conn_options=conn_options,
        )
        self.streams.append(stream)
        self._stream_ch.send_nowait(stream)
        return stream


class FakeRecognizeStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: FakeSTT,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options)
        self._attempt = 0
        self._start_fut = asyncio.Future[float]()
        self._stt: FakeSTT = stt
        # onsets of the VAD segments the agent asked to finalize (flush_vad), in order
        self.flushed_onsets: list[float] = []

    def flush_vad(self, speech_start_time: float) -> None:
        self.flushed_onsets.append(speech_start_time)
        super().flush_vad(speech_start_time)

    def _flushed_onset_near(self, epoch: float) -> float | None:
        """The onset the agent flushed for the segment that started near `epoch`."""
        near = [o for o in self.flushed_onsets if abs(o - epoch) < 1.0]
        return min(near, key=lambda o: abs(o - epoch)) if near else None

    @property
    def attempt(self) -> int:
        return self._attempt

    def send_fake_transcript(
        self,
        transcript: str,
        is_final: bool = True,
        *,
        transcript_span: NotGivenOr[tuple[float, float] | None] = NOT_GIVEN,
        finalize_onset: float | None = None,
    ) -> None:
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT
                if is_final
                else SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[SpeechData(text=transcript, language=LanguageCode(""))],
                transcript_span=transcript_span,
                finalize_onset=finalize_onset,
            )
        )

    async def _run(self) -> None:
        self._attempt += 1
        assert isinstance(self._stt, FakeSTT)

        if self._stt._fake_timeout is not None:
            await asyncio.sleep(self._stt._fake_timeout)

        if self._stt._fake_require_audio:
            got_audio = False
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    got_audio = True
                elif isinstance(data, self._FlushSentinel) and got_audio:
                    if self._stt._fake_transcript is not None:
                        self.send_fake_transcript(self._stt._fake_transcript)
                    got_audio = False
        else:
            if self._stt._fake_transcript is not None:
                self.send_fake_transcript(self._stt._fake_transcript)

            await self._fake_user_speech_task()

            async for _ in self._input_ch:
                pass

        if self._stt._fake_exception is not None:
            raise self._stt._fake_exception

    async def _fake_user_speech_task(self) -> None:
        if not self._stt._fake_user_speeches:
            return

        # start from when the first frame is pushed
        await self._input_ch.recv()
        start_time = time.perf_counter()
        # the call clock (epoch seconds) of scripted time zero, as a provider adapter would map it
        origin = time.time()

        def curr_time() -> float:
            return time.perf_counter() - start_time

        async def wait_until(t: float) -> None:
            if curr_time() < t:
                await asyncio.sleep(t - curr_time())

        def span_of(
            speech: FakeUserSpeech, a: float, b: float
        ) -> NotGivenOr[tuple[float, float] | None]:
            if speech.span == "given":
                return (origin + a, origin + b)
            return None if speech.span == "none" else NOT_GIVEN

        for fake_speech in self._stt._fake_user_speeches:
            if not fake_speech.transcript:
                # empty transcript: VAD SOS/EOS fire but STT emits no text — still
                # advance the clock past this speech so subsequent speeches (and the
                # done signal) are timed correctly relative to it.
                await wait_until(fake_speech.end_time + fake_speech.stt_delay)
                if fake_speech.empty_finalize:
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.VAD_FINALIZE_EMPTY,
                            finalize_onset=self._flushed_onset_near(
                                origin + fake_speech.start_time
                            ),
                        )
                    )
                continue
            main_end = (
                fake_speech.main_end if fake_speech.main_end is not None else fake_speech.end_time
            )
            interim_transcript_time = main_end + fake_speech.stt_delay * 0.5
            await wait_until(interim_transcript_time)
            self.send_fake_transcript(" ".join(fake_speech.transcript.split()[:2]), is_final=False)

            final_transcript_time = (
                fake_speech.main_arrive_at
                if fake_speech.main_arrive_at is not None
                else main_end + fake_speech.stt_delay
            )
            extras = sorted(fake_speech.extra_finals, key=lambda f: f.arrive_at)
            if fake_speech.final:
                await wait_until(final_transcript_time)
                self.send_fake_transcript(
                    fake_speech.transcript,
                    is_final=True,
                    transcript_span=span_of(fake_speech, fake_speech.start_time, main_end),
                    finalize_onset=(
                        self._flushed_onset_near(origin + fake_speech.start_time)
                        if fake_speech.finalize_onset
                        else None
                    ),
                )
            for extra in extras:
                await wait_until(extra.arrive_at)
                self.send_fake_transcript(
                    extra.text,
                    is_final=True,
                    transcript_span=span_of(fake_speech, extra.start_time, extra.end_time),
                    finalize_onset=(
                        self._flushed_onset_near(origin + fake_speech.start_time)
                        if extra.finalize_onset
                        else None
                    ),
                )

        with contextlib.suppress(asyncio.InvalidStateError):
            self._stt._done_fut.set_result(None)
