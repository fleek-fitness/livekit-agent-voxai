"""Preemptive generations started by a preflight transcript.

Covers the options that cancel them once an interim transcript moves past them, keep their
synthesis from starting early, and compare their transcript ignoring spacing and
punctuation, plus the event that reports how each one differed from the turn's final
transcript.

STT events are fed to an ``AudioRecognition`` wired to a started session's activity; the
end of the turn is committed explicitly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings, stt
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _normalized_preflight_match,
    _preflight_transcript_diff,
)
from livekit.agents.voice.events import (
    PreemptiveGenerationOutcomeEvent,
    PreflightFinalCompareEvent,
)
from livekit.agents.voice.turn import PreemptiveGenerationOptions

from .fake_io import FakeAudioOutput
from .fake_tts import FakeTTS
from .test_preemptive_pause_deadlock import _end_of_turn, _GatedLLM, _wait_until
from .test_transcription_delay_anchor import _make_recognition

pytestmark = pytest.mark.unit

# long enough for a generation to reach the TTS node if it were going to
SETTLE = 0.2


class _SynthesisCountingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="take a booking")
        self.synthesis_count = 0

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        self.synthesis_count += 1
        async for _ in text:
            pass
        return
        yield


@dataclass
class _Harness:
    session: AgentSession[None]
    agent: _SynthesisCountingAgent
    activity: AgentActivity
    recognition: AudioRecognition
    outcomes: list[PreemptiveGenerationOutcomeEvent] = field(default_factory=list)
    compares: list[PreflightFinalCompareEvent] = field(default_factory=list)

    async def preflight(self, text: str) -> None:
        await self._push(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, text)

    async def interim(self, text: str) -> None:
        await self._push(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)

    async def final(self, text: str) -> None:
        await self._push(stt.SpeechEventType.FINAL_TRANSCRIPT, text)

    async def _push(self, kind: stt.SpeechEventType, text: str) -> None:
        await self.recognition._on_stt_event(
            stt.SpeechEvent(
                type=kind,
                alternatives=[stt.SpeechData(language="ko", text=text, confidence=1.0)],
            )
        )

    @property
    def generation_count(self) -> int:
        return self.activity._preemptive_generation_count

    async def end_turn(self, transcript: str) -> None:
        assert self.activity.on_end_of_turn(_end_of_turn(transcript))
        task = self.activity._user_turn_completed_atask
        assert task is not None
        await task


@asynccontextmanager
async def _harness(options: PreemptiveGenerationOptions) -> AsyncIterator[_Harness]:
    llm = _GatedLLM()
    session: AgentSession[None] = AgentSession(
        llm=llm, tts=FakeTTS(), turn_handling={"preemptive_generation": options}
    )
    session.output.audio = FakeAudioOutput()
    agent = _SynthesisCountingAgent()
    await session.start(agent=agent)
    activity = session._activity
    assert activity is not None

    recognition = _make_recognition(vad=None, input_started_at=time.time() - 10.0)
    recognition._hooks = activity
    recognition._vad_base_turn_detection = True  # a final may start a generation

    harness = _Harness(session, agent, activity, recognition)
    session.on("preemptive_generation_outcome", harness.outcomes.append)
    session.on("preflight_final_compare", harness.compares.append)
    try:
        yield harness
    finally:
        llm.release()
        await asyncio.wait_for(session.aclose(), 5.0)


@pytest.mark.parametrize("enabled", [True, False])
async def test_interim_that_moves_past_the_preflight_cancels_it(enabled: bool) -> None:
    async with _harness({"cancel_on_preflight_stale": enabled}) as h:
        await h.preflight("내일 세 시")
        candidate = h.activity._preemptive_generation
        assert candidate is not None

        await h.interim("내일 세 시 예약할게요")

        if enabled:
            assert h.activity._preemptive_generation is None
            assert candidate.speech_handle.interrupted
            assert [(o.outcome, o.reason, o.trigger_source) for o in h.outcomes] == [
                ("discarded", "preflight_stale", "preflight")
            ]
            assert h.recognition._audio_preflight_transcript == ""
        else:
            assert h.activity._preemptive_generation is candidate
            assert not candidate.speech_handle.interrupted
            assert h.outcomes == []
            assert h.recognition._audio_preflight_transcript == "내일 세 시"


async def test_interim_is_compared_together_with_the_earlier_finals() -> None:
    async with _harness({"cancel_on_preflight_stale": True}) as h:
        await h.final("내일")
        await h.preflight("세 시")
        candidate = h.activity._preemptive_generation
        assert candidate is not None and candidate.info.new_transcript == "내일 세 시"

        await h.interim("세 시")
        assert h.activity._preemptive_generation is candidate

        await h.interim("세 시 반")
        assert h.activity._preemptive_generation is None
        assert h.outcomes[-1].reason == "preflight_stale"


@pytest.mark.parametrize("normalize", [True, False])
async def test_interim_differing_only_in_punctuation_keeps_the_candidate(normalize: bool) -> None:
    async with _harness(
        {"cancel_on_preflight_stale": True, "normalize_preflight_transcript": normalize}
    ) as h:
        await h.preflight("네")
        candidate = h.activity._preemptive_generation

        await h.interim("네.")

        assert h.activity._preemptive_generation is candidate
        assert h.outcomes == []


async def test_final_after_a_stale_cancel_starts_one_generation() -> None:
    async with _harness({"cancel_on_preflight_stale": True}) as h:
        await h.preflight("내일 세 시")
        await h.interim("내일 네 시")
        assert h.activity._preemptive_generation is None

        # the final goes back to the preflight words: it still has to start a generation
        await h.final("내일 세 시")

        candidate = h.activity._preemptive_generation
        assert candidate is not None
        assert candidate.info.trigger_source == "final_transcript"
        assert h.generation_count == 2


async def test_equivalent_interim_then_same_final_reuses_the_candidate() -> None:
    async with _harness({"cancel_on_preflight_stale": True}) as h:
        await h.preflight("내일 세 시")
        candidate = h.activity._preemptive_generation
        await h.interim("내일 세 시.")
        await h.final("내일 세 시")

        assert h.activity._preemptive_generation is candidate
        assert h.generation_count == 1

        await h.end_turn("내일 세 시")
        assert [(o.outcome, o.trigger_source) for o in h.outcomes] == [("reused", "preflight")]


@pytest.mark.parametrize(
    ("preflight", "final"),
    [("네", "네."), ("6 개요", "6개요.")],
)
async def test_normalized_final_reuses_the_preflight_candidate(preflight: str, final: str) -> None:
    async with _harness({"normalize_preflight_transcript": True}) as h:
        await h.preflight(preflight)
        candidate = h.activity._preemptive_generation
        assert candidate is not None

        await h.final(final)
        assert h.activity._preemptive_generation is candidate
        assert h.generation_count == 1

        await h.end_turn(final)
        assert [(o.outcome, o.speech_id) for o in h.outcomes] == [
            ("reused", candidate.speech_handle.id)
        ]


@pytest.mark.parametrize(
    ("normalize", "preflight", "final"),
    [
        (True, "네", "네?"),
        (True, "3.5", "35"),
        (False, "네", "네."),
    ],
)
async def test_final_that_does_not_match_starts_a_new_candidate(
    normalize: bool, preflight: str, final: str
) -> None:
    async with _harness({"normalize_preflight_transcript": normalize}) as h:
        await h.preflight(preflight)
        candidate = h.activity._preemptive_generation

        await h.final(final)

        replacement = h.activity._preemptive_generation
        assert replacement is not None and replacement is not candidate
        assert replacement.info.trigger_source == "final_transcript"
        assert replacement.info.new_transcript == final


@pytest.mark.parametrize(
    ("preflight_tts", "preemptive_tts", "preflight_early", "final_early"),
    [
        (False, True, False, True),
        (None, True, True, True),
        (None, False, False, False),
        (True, False, True, False),
    ],
)
async def test_preflight_tts_only_applies_to_preflight_candidates(
    preflight_tts: bool | None,
    preemptive_tts: bool,
    preflight_early: bool,
    final_early: bool,
) -> None:
    async with _harness({"preflight_tts": preflight_tts, "preemptive_tts": preemptive_tts}) as h:
        await h.preflight("내일")
        await asyncio.sleep(SETTLE)
        assert h.agent.synthesis_count == int(preflight_early)

        await h.final("모레")
        await asyncio.sleep(SETTLE)
        assert h.agent.synthesis_count == int(preflight_early) + int(final_early)
        assert [o.early_tts for o in h.outcomes] == [preflight_early]

        # a candidate that did not synthesize early does so once it is scheduled
        await h.end_turn("모레")
        await _wait_until(lambda: h.agent.synthesis_count == int(preflight_early) + 1)
        assert [o.early_tts for o in h.outcomes] == [preflight_early, final_early]


async def test_preflight_tts_does_not_enable_synthesis_when_generation_is_disabled() -> None:
    async with _harness({"enabled": False, "preflight_tts": True}) as h:
        assert not h.activity._early_tts_enabled("preflight")


async def test_no_early_tts_without_audio_output() -> None:
    async with _harness({"preemptive_tts": True}) as h:
        h.session.output.set_audio_enabled(False)
        await h.preflight("내일")
        candidate = h.activity._preemptive_generation
        assert candidate is not None and not candidate.early_tts


async def test_compare_event_covers_a_candidate_cancelled_as_stale() -> None:
    async with _harness({"cancel_on_preflight_stale": True}) as h:
        await h.preflight("네")
        stale = h.activity._preemptive_generation
        assert stale is not None
        await h.interim("아니")
        await h.final("네.")
        assert h.compares == []  # nothing is reported before the turn is committed

        await h.end_turn("네.")

        assert [(c.speech_id, c.diff) for c in h.compares] == [
            (stale.speech_handle.id, "punctuation_only")
        ]
        assert h.activity._preflight_candidates == []


async def test_compare_event_is_emitted_for_every_preflight_candidate() -> None:
    async with _harness({}) as h:
        await h.preflight("내일")
        first = h.activity._preemptive_generation
        await h.preflight("내일 세 시")
        second = h.activity._preemptive_generation
        assert first is not None and second is not None
        await h.final("내일 세 시")

        await h.end_turn("내일 세 시")

        assert [(c.speech_id, c.diff) for c in h.compares] == [
            (first.speech_handle.id, "changed"),
            (second.speech_handle.id, "same"),
        ]


async def test_no_compare_event_without_a_preflight_candidate() -> None:
    async with _harness({}) as h:
        await h.final("내일")
        await h.end_turn("내일")
        assert h.compares == []


async def test_compare_candidates_are_dropped_when_the_turn_is_cleared() -> None:
    async with _harness({}) as h:
        await h.preflight("내일")
        h.activity.clear_user_turn()
        assert h.activity._preflight_candidates == []
        assert h.compares == []


@pytest.mark.parametrize(
    ("preflight", "final", "diff"),
    [
        ("네", "네", "same"),
        ("6 개요", "6개요", "whitespace_only"),
        ("네.", "네", "punctuation_only"),
        ("6 개요", "6개요.", "punctuation_only"),
        ("Hello, there", "hello there.", "punctuation_only"),
        ("café", "café", "punctuation_only"),
        ("3.5", "3.5.", "punctuation_only"),
        ("네", "네?", "changed"),
        ("네", "네？", "changed"),
        ("3.5", "35", "changed"),
        ("1,000", "1000", "changed"),
        ("3:30", "330", "changed"),
        ("5%", "5", "changed"),
        ("내일 세 시", "내일 세 시 예약할게요", "changed"),
        (".", "!", "changed"),
    ],
)
def test_preflight_transcript_diff(preflight: str, final: str, diff: str) -> None:
    assert _preflight_transcript_diff(preflight, final) == diff
    assert _normalized_preflight_match(preflight, final) is (diff != "changed")
