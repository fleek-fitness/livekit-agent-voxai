"""A tool that answers early and then returns nothing asks for no deferred reply.

``None`` and ``StopResponse`` after a first ``ctx.update()`` (or after an interrupted turn
released the tool) still record a final output, but the output carries ``reply_required=False``,
so no extra reply is generated. A returned string asks for one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, RunContext, StopResponse, function_tool, llm
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT
from livekit.agents.voice.tool_executor import _ToolExecutor

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

TOOL_DURATION = INTERRUPTION_TIMEOUT + 3


def _outputs(chat_ctx: llm.ChatContext, call_id: str) -> list[llm.FunctionCallOutput]:
    return [i for i in chat_ctx.items if i.type == "function_call_output" and i.call_id == call_id]


def _count_deferred_replies(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    delivered: list[int] = []
    original = _ToolExecutor._deliver_reply

    async def counting(self: _ToolExecutor, session: Any) -> None:
        delivered.append(1)
        await original(self, session)

    monkeypatch.setattr(_ToolExecutor, "_deliver_reply", counting)
    return delivered


def _make_agent(result: Callable[[], Any], *, early_update: bool) -> Agent:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self, ctx: RunContext) -> str | None:
            """Look up the order."""
            if early_update:
                await ctx.update("working on it")
            await asyncio.sleep(TOOL_DURATION)
            return result()

    return ToolAgent()


def _raise_stop() -> Any:
    raise StopResponse()


async def _run(agent: Agent, *, interrupt: bool) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(4.0)
    session = create_session(actions)

    if interrupt:

        def _interrupt_while_speaking(ev: object) -> None:
            if getattr(ev, "new_state", None) == "speaking":
                session.off("agent_state_changed", _interrupt_while_speaking)
                asyncio.get_running_loop().call_later(0.5, lambda: session.interrupt(force=True))

        session.on("agent_state_changed", _interrupt_while_speaking)

    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 8), timeout=90)


@pytest.mark.parametrize("interrupt", [False, True], ids=["after_update", "after_release"])
@pytest.mark.parametrize(
    ("result", "replies"),
    [
        pytest.param(lambda: None, 0, id="none"),
        pytest.param(_raise_stop, 0, id="stop_response"),
        pytest.param(lambda: "order 42 shipped", 1, id="string"),
    ],
)
async def test_final_return_asks_for_a_reply_only_with_content(
    monkeypatch: pytest.MonkeyPatch,
    result: Callable[[], Any],
    replies: int,
    interrupt: bool,
) -> None:
    delivered = _count_deferred_replies(monkeypatch)
    agent = _make_agent(result, early_update=not interrupt)

    await _run(agent, interrupt=interrupt)

    finals = _outputs(agent.chat_ctx, "1_final")
    assert len(finals) == 1
    assert finals[0].reply_required is (replies == 1)
    assert len(delivered) == replies
