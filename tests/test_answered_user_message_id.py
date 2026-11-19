from __future__ import annotations

import pytest

from livekit.agents import Agent, MetricsCollectedEvent
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import EOUMetrics

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def _run(actions: FakeActions, **session_kwargs):
    session = create_session(actions, **session_kwargs)
    items: list[ChatMessage] = []
    eous: list[EOUMetrics] = []

    session.on(
        "conversation_item_added",
        lambda ev: items.append(ev.item) if ev.item.type == "message" else None,
    )

    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        if isinstance(ev.metrics, EOUMetrics):
            eous.append(ev.metrics)

    session.on("metrics_collected", _on_metrics)
    await run_session(session, Agent(instructions="test"), drain_delay=2.0)
    return items, eous


@pytest.mark.parametrize(
    "preemptive",
    [
        pytest.param({"enabled": False}, id="no_preemptive"),
        pytest.param({"preemptive_tts": True}, id="preemptive_tts"),
        pytest.param({"preemptive_tts": False}, id="preemptive_llm_only"),
    ],
)
async def test_eou_and_reply_name_the_stored_user_message(preemptive: dict) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing great, thank you!")
    actions.add_tts(2.0)

    items, eous = await _run(actions, turn_handling={"preemptive_generation": preemptive})

    (user,) = [m for m in items if m.role == "user"]
    (reply,) = [m for m in items if m.role == "assistant"]
    (eou,) = eous
    # the id the conversation history keeps, also when a preemptive reply is reused
    assert eou.user_message_id == user.id
    assert reply.metrics["answered_user_message_id"] == user.id


@pytest.mark.parametrize(
    "preemptive",
    [
        pytest.param({"enabled": False}, id="no_preemptive"),
        pytest.param({"preemptive_tts": True}, id="preemptive_tts"),
    ],
)
async def test_interrupting_turn_keeps_each_eou_on_its_own_user_message(
    preemptive: dict,
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Tell me a long story", stt_delay=0.1)
    actions.add_llm("Once upon a time there was a very long story.")
    actions.add_tts(8.0)
    # B talks over A's reply
    actions.add_user_speech(5.0, 6.5, "Actually, stop", stt_delay=0.1)
    actions.add_llm("Okay, stopping.")
    actions.add_tts(1.0)

    items, eous = await _run(actions, turn_handling={"preemptive_generation": preemptive})

    users = [m for m in items if m.role == "user"]
    replies = [m for m in items if m.role == "assistant"]
    assert len(users) == 2
    # each user message ends exactly one turn, and each reply answers its own turn in order
    assert [e.user_message_id for e in eous] == [m.id for m in users]
    assert len(replies) == 2
    assert replies[0].interrupted
    assert [r.metrics["answered_user_message_id"] for r in replies] == [m.id for m in users]
