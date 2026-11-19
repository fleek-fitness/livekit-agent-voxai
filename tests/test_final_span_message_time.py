"""A customer message takes its time from the audio span its own final reports.

Each case replays one provider-shaped event order through a real AgentSession: VAD segments
(some with no text), the agent's speech, and finals that carry `transcript_span` (the call-clock
span of the audio the text came from), `finalize_onset` (the final answers the finalize sent for
one VAD segment) or a `VAD_FINALIZE_EMPTY` event. The checks compare the stored user messages
(count, text, start and end) with the scripted segments.

Provider shapes (the event shape only; the adapters live in the agent server):
  soniox_auto    spans on finals, nothing for an empty segment
  soniox_manual  spans, a finalize onset on the final, a VAD_FINALIZE_EMPTY for an empty segment
  rtzr           spans on finals, nothing for an empty segment
  rtzrx          spans, a finalize onset on the final, a VAD_FINALIZE_EMPTY for an empty segment
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent
from livekit.agents.voice.agent_activity import LATE_FINAL_EXTRA_KEY
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session
from .fake_stt import FakeFinal

pytestmark = pytest.mark.unit

SPEED = 5.0
SESSION_TIMEOUT = 120.0
WAIT_TIMEOUT = 40.0
TOLERANCE = 0.8  # scripted seconds; 0.16 real seconds, far below any gap the cases rely on

SLACK = 1.5 * SPEED
QUESTION_DURATION = 20.0
AI_START = 3.3
AI_END = AI_START + QUESTION_DURATION
ANSWER_START = AI_END + SLACK
ANSWER_END = ANSWER_START + 0.9

QUESTION = "Do you have any other concerns?"
GREETING = "Tell me about the rooms."
ANSWER = "아니요, 없습니다"
REPLY = "Understood, thank you."

SHAPES = {
    # (span, finalize_onset on the final, VAD_FINALIZE_EMPTY for a segment with no text)
    "soniox_auto": ("given", False, False),
    "soniox_manual": ("given", True, True),
    "rtzr": ("given", False, False),
    "rtzrx": ("given", True, True),
}


@pytest.fixture(autouse=True)
def flush_stt_on_vad_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the agent server does when VAD ends a segment: finalize that segment's audio."""
    from livekit.agents.voice.agent_activity import AgentActivity
    from tests.fake_stt import FakeSTT

    original = AgentActivity.on_end_of_speech

    def on_end_of_speech(self: AgentActivity, ev: Any) -> None:
        onset = self._audio_recognition._vad_finalize_start if self._audio_recognition else None
        if onset is not None and isinstance(self.stt, FakeSTT):
            for stream in self.stt.streams:
                stream.flush_vad(onset)
        original(self, ev)

    monkeypatch.setattr(AgentActivity, "on_end_of_speech", on_end_of_speech)


class _Run:
    def __init__(self, agent: Agent, items: list[ConversationItemAddedEvent], origin: float):
        self.agent = agent
        self.items = items
        self.origin = origin

    def users(self) -> list[Any]:
        return [
            ev.item for ev in self.items if ev.item.type == "message" and ev.item.role == "user"
        ]

    def messages(self) -> list[Any]:
        return [
            item
            for item in self.agent.chat_ctx.items
            if item.type == "message" and item.role in ("user", "assistant")
        ]

    def span(self, message: Any) -> tuple[float, float] | None:
        """The message's recorded time in scripted seconds from the audio origin."""
        metrics = message.metrics
        if "started_speaking_at" not in metrics or "stopped_speaking_at" not in metrics:
            return None
        return (
            (metrics["started_speaking_at"] - self.origin) * SPEED,
            (metrics["stopped_speaking_at"] - self.origin) * SPEED,
        )

    def recorded(self) -> list[tuple[str, str]]:
        """Messages in the order their records were emitted (the order that is stored)."""
        return [
            (ev.item.role, ev.item.text_content)
            for ev in self.items
            if ev.item.type == "message" and ev.item.role in ("user", "assistant")
        ]

    def shape(self) -> list[tuple[str, str]]:
        return [(m.role, m.text_content) for m in self.messages()]


class _Greeter(Agent):
    def __init__(self, question: str | None) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self._question = question

    async def on_enter(self) -> None:
        if self._question:
            self.session.say(self._question)


async def _run(
    actions: FakeActions,
    *,
    until: Callable[[_Run], bool],
    ignore_words: list[str] | None = None,
    say_first: str | None = None,
    settle: float = 1.5,
) -> _Run:
    extra: dict[str, Any] = {}
    handling: dict[str, Any] = {}
    if ignore_words:
        extra["interruption_ignore_words"] = ignore_words
        handling = {"interruption": {"min_words": 1}}
    session = create_session(
        actions, speed_factor=SPEED, turn_handling=handling, extra_kwargs=extra
    )
    agent = _Greeter(say_first)
    items: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", items.append)
    run = _Run(agent, items, 0.0)

    origin_box: list[float] = []

    async def go() -> None:
        origin_box.append(
            await run_session(
                session,
                agent,
                drain_delay=settle,
                until=lambda: until(run),
                until_timeout=WAIT_TIMEOUT,
            )
        )

    await asyncio.wait_for(go(), timeout=SESSION_TIMEOUT)
    run.origin = origin_box[0]
    return run


def _assistants(run: _Run) -> list[Any]:
    return [m for m in run.messages() if m.role == "assistant"]


def _approx(actual: tuple[float, float] | None, expected: tuple[float, float]) -> bool:
    return (
        actual is not None
        and abs(actual[0] - expected[0]) <= TOLERANCE
        and abs(actual[1] - expected[1]) <= TOLERANCE
    )


# --- 1 and 2: an empty segment before the question's answer ---------------------------------


def _noise_then_answer(shape: str, *, with_noise: bool) -> FakeActions:
    span, onset, empty_event = SHAPES[shape]
    actions = FakeActions()
    if with_noise:
        # the customer's audio detector fires before the greeting, with no text
        actions.add_user_speech(0.2, 0.5, "", empty_finalize=empty_event)
    actions.add_user_speech(
        ANSWER_START, ANSWER_END, ANSWER, span=span, finalize_onset=onset, stt_delay=1.0
    )
    actions.add_llm(REPLY, input=ANSWER)
    actions.add_tts(1.0, input=REPLY)
    actions.add_tts(QUESTION_DURATION, input=QUESTION)
    return actions


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_answer_after_empty_segment_keeps_its_own_span(shape: str) -> None:
    run = await _run(
        _noise_then_answer(shape, with_noise=True),
        until=lambda r: len(_assistants(r)) >= 2,
        say_first=QUESTION,
    )

    assert run.shape() == [("assistant", QUESTION), ("user", ANSWER), ("assistant", REPLY)]
    (answer,) = run.users()
    assert _approx(run.span(answer), (ANSWER_START, ANSWER_END)), run.span(answer)
    assert run.messages()[0].interrupted is False


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_empty_segment_leaves_nothing_behind(shape: str) -> None:
    with_noise = await _run(
        _noise_then_answer(shape, with_noise=True),
        until=lambda r: len(_assistants(r)) >= 2,
        say_first=QUESTION,
    )
    without = await _run(
        _noise_then_answer(shape, with_noise=False),
        until=lambda r: len(_assistants(r)) >= 2,
        say_first=QUESTION,
    )

    assert with_noise.shape() == without.shape()
    (a,), (b,) = with_noise.users(), without.users()
    assert with_noise.span(a) is not None and without.span(b) is not None
    assert abs(with_noise.span(a)[0] - without.span(b)[0]) <= TOLERANCE  # type: ignore[index]
    assert abs(with_noise.span(a)[1] - without.span(b)[1]) <= TOLERANCE  # type: ignore[index]


# --- 3, 4 and 7: a final that arrives after its turn was committed ---------------------------


# a late final inside the recording hold (the turn commits about 5 scripted seconds after the
# speech ends and is held for 3 real seconds, 15 scripted)
LATE_WITHIN_HOLD = 10.0
LATE_AFTER_HOLD = 30.0


def _late_final_actions(
    shape: str,
    *,
    next_speech: str | None = None,
    late_text: str = "없습니다",
    late_delay: float = LATE_WITHIN_HOLD,
) -> FakeActions:
    span, onset, _ = SHAPES[shape]
    first_end = ANSWER_START + 0.31
    late_start, late_end = ANSWER_START + 0.41, ANSWER_START + 0.86
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, GREETING)
    actions.add_llm(QUESTION, input=GREETING)
    actions.add_tts(QUESTION_DURATION, input=QUESTION)
    actions.add_user_speech(
        ANSWER_START,
        late_end,
        "아니요",
        span=span,
        finalize_onset=False,
        main_end=first_end,
        main_arrive_at=first_end + 0.5,
        extra_finals=[
            FakeFinal(
                text=late_text,
                start_time=late_start,
                end_time=late_end,
                # well after the VAD end, the pause and the turn commit
                arrive_at=late_end + late_delay,
                finalize_onset=onset,
            )
        ],
    )
    actions.add_llm(REPLY, input="아니요")
    actions.add_tts(30.0, input=REPLY)
    if next_speech:
        actions.add_user_speech(late_end + 3 * SPEED, late_end + 3 * SPEED + 0.5, next_speech)
    return actions


@pytest.mark.parametrize("shape", ["soniox_auto", "soniox_manual", "rtzr", "rtzrx"])
async def test_late_final_is_recorded_once_in_its_own_turn(shape: str) -> None:
    run = await _run(
        _late_final_actions(shape),
        until=lambda r: any(u.text_content == "없습니다" for u in r.users()),
        settle=2.0,
    )

    texts = [t for role, t in run.shape() if role == "user"]
    assert texts == [GREETING, "아니요", "없습니다"]
    late = [u for u in run.users() if u.text_content == "없습니다"]
    assert len(late) == 1
    assert late[0].extra.get(LATE_FINAL_EXTRA_KEY) is True
    assert _approx(
        run.span(late[0]),
        (ANSWER_START + 0.41, ANSWER_START + 0.86),
    ), run.span(late[0])
    # one reply to the first final; the late final neither answered nor cut it off
    replies = [m for m in _assistants(run) if m.text_content == REPLY]
    assert len(replies) <= 1
    assert all(not m.interrupted for m in replies)
    # recorded right after its own turn, before the reply to it, in the stored order
    recorded = run.recorded()
    assert recorded.index(("user", "아니요")) + 1 == recorded.index(("user", "없습니다"))
    if ("assistant", REPLY) in recorded:
        assert recorded.index(("user", "없습니다")) < recorded.index(("assistant", REPLY))
    chat = [(m.role, m.text_content) for m in run.messages()]
    assert chat.index(("user", "없습니다")) < chat.index(("assistant", REPLY))


async def test_late_final_after_the_hold_is_recorded_where_it_arrives() -> None:
    run = await _run(
        _late_final_actions("soniox_manual", late_delay=LATE_AFTER_HOLD),
        until=lambda r: any(u.text_content == "없습니다" for u in r.users()),
        settle=2.0,
    )

    recorded = run.recorded()
    assert [t for role, t in recorded if role == "user"].count("없습니다") == 1
    late = next(u for u in run.users() if u.text_content == "없습니다")
    assert _approx(run.span(late), (ANSWER_START + 0.41, ANSWER_START + 0.86))
    # no second answer and no cut-off of the reply
    assert all(not m.interrupted for m in _assistants(run) if m.text_content == REPLY)


async def test_late_final_does_not_join_the_next_open_segment() -> None:
    run = await _run(
        _late_final_actions("soniox_manual", next_speech="네"),
        until=lambda r: (
            any(u.text_content == "네" for u in r.users())
            and any(u.text_content == "없습니다" for u in r.users())
        ),
        settle=2.0,
    )

    texts = [u.text_content for u in run.users()]
    assert texts.count("없습니다") == 1
    assert "네" in texts
    assert all("없습니다 네" != t and "네 없습니다" != t for t in texts)
    nexts = [u for u in run.users() if u.text_content == "네"]
    assert len(nexts) == 1


async def test_late_final_with_the_same_text_as_another_turn_is_kept() -> None:
    run = await _run(
        _late_final_actions("soniox_manual", next_speech="네", late_text="네"),
        until=lambda r: sum(u.text_content == "네" for u in r.users()) >= 2,
        settle=2.0,
    )

    assert [u.text_content for u in run.users()].count("네") == 2


# --- 5: a final with no span ---------------------------------------------------------------


@pytest.mark.parametrize("shape", ["rtzrx", "soniox_manual"])
async def test_final_without_span_uses_its_proven_vad_segment(shape: str) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.2, 0.5, "", empty_finalize=True)
    actions.add_user_speech(
        ANSWER_START,
        ANSWER_END,
        ANSWER,
        span="none",
        finalize_onset=True,
        stt_delay=1.0,
    )
    actions.add_llm(REPLY, input=ANSWER)
    actions.add_tts(1.0, input=REPLY)
    actions.add_tts(QUESTION_DURATION, input=QUESTION)
    run = await _run(actions, until=lambda r: len(_assistants(r)) >= 2, say_first=QUESTION)

    (answer,) = run.users()
    assert _approx(run.span(answer), (ANSWER_START, ANSWER_END)), run.span(answer)


async def test_final_without_span_or_onset_is_unknown_but_kept() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.2, 0.5, "")
    actions.add_user_speech(ANSWER_START, ANSWER_END, ANSWER, span="none", stt_delay=1.0)
    actions.add_llm(REPLY, input=ANSWER)
    actions.add_tts(1.0, input=REPLY)
    actions.add_tts(QUESTION_DURATION, input=QUESTION)
    run = await _run(actions, until=lambda r: len(_assistants(r)) >= 2, say_first=QUESTION)

    assert run.shape() == [("assistant", QUESTION), ("user", ANSWER), ("assistant", REPLY)]
    (answer,) = run.users()
    assert run.span(answer) is None


# --- 8: an acknowledgement over the question keeps its own place and span --------------------


async def test_backchannel_over_question_keeps_place_and_span() -> None:
    ack_start = AI_START + SLACK
    ack_end = ack_start + 0.6
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, GREETING)
    actions.add_llm(QUESTION, input=GREETING)
    actions.add_tts(QUESTION_DURATION, input=QUESTION)
    actions.add_user_speech(ack_start, ack_end, "네.", span="given", stt_delay=1.0)
    actions.add_user_speech(ANSWER_START, ANSWER_END, ANSWER, span="given", stt_delay=1.0)
    actions.add_llm(REPLY, input=ANSWER)
    actions.add_tts(1.0, input=REPLY)
    run = await _run(
        actions,
        until=lambda r: len(_assistants(r)) >= 2,
        ignore_words=["네", "예"],
    )

    assert run.shape() == [
        ("user", GREETING),
        ("assistant", QUESTION),
        ("user", "네."),
        ("user", ANSWER),
        ("assistant", REPLY),
    ]
    ack = next(u for u in run.users() if u.text_content == "네.")
    answer = next(u for u in run.users() if u.text_content == ANSWER)
    assert _approx(run.span(ack), (ack_start, ack_end)), run.span(ack)
    assert _approx(run.span(answer), (ANSWER_START, ANSWER_END)), run.span(answer)
    assert run.messages()[1].interrupted is False


# --- the message end stretches to the end of its VAD segment (never past one second) ----------


@pytest.mark.parametrize(
    ("short_by", "stretched"),
    [(3.0, True), (7.0, False)],  # scripted seconds; one real second is SPEED scripted seconds
)
async def test_message_end_stretches_to_vad_segment_end_within_one_second(
    short_by: float, stretched: bool
) -> None:
    start, end = ANSWER_START, ANSWER_START + 8.0
    actions = FakeActions()
    actions.add_user_speech(
        start,
        end,
        ANSWER,
        span="given",
        main_end=end - short_by,  # the last token ends before the speech does (Soniox shape)
        main_arrive_at=end + 1.0,
        stt_delay=1.0,
    )
    actions.add_llm(REPLY, input=ANSWER)
    actions.add_tts(1.0, input=REPLY)
    actions.add_tts(QUESTION_DURATION, input=QUESTION)
    run = await _run(actions, until=lambda r: len(_assistants(r)) >= 2, say_first=QUESTION)

    (answer,) = run.users()
    got = run.span(answer)
    assert got is not None
    assert abs(got[0] - start) <= TOLERANCE
    expected_end = end if stretched else end - short_by
    assert abs(got[1] - expected_end) <= TOLERANCE, (got, expected_end)
