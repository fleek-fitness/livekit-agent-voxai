"""A paused speech that is then cancelled keeps the result of the tool still running.

The speech is paused (a barge-in candidate) and then cancelled by the caller's transcript while a
tool runs past the interruption timeout. The tool is released as an async tool, so its call is
answered, it runs once, and its final result reaches the chat context.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, function_tool, llm
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

TOOL_DURATION = INTERRUPTION_TIMEOUT + 3


def _outputs(chat_ctx: llm.ChatContext, call_id: str) -> list[llm.FunctionCallOutput]:
    return [
        item
        for item in chat_ctx.items
        if item.type == "function_call_output" and item.call_id == call_id
    ]


async def test_paused_speech_cancelled_while_tool_runs() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(4.0)
    runs: list[int] = []

    class SlowToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self) -> str:
            """Look up the order."""
            runs.append(1)
            await asyncio.sleep(TOOL_DURATION)
            return "order 42 shipped"

    session = create_session(actions, can_pause_audio=True)
    agent = SlowToolAgent()
    fired = False

    def _pause_then_cancel(ev: object) -> None:
        nonlocal fired
        if getattr(ev, "new_state", None) != "speaking" or fired:
            return
        fired = True

        def _go() -> None:
            activity = session._activity
            handle = session.current_speech
            assert activity is not None and handle is not None
            assert session.output.audio is not None
            session.output.audio.pause()
            activity._update_paused_speech(handle, timeout=0)
            handle.interrupt()

        asyncio.get_running_loop().call_later(0.5, _go)

    session.on("agent_state_changed", _pause_then_cancel)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 8), timeout=90)

    # the tool ran once, its call is answered, and its real result is kept
    assert len(runs) == 1
    assert len(_outputs(agent.chat_ctx, "1")) == 1
    final = _outputs(agent.chat_ctx, "1_final")
    assert len(final) == 1
    assert "order 42 shipped" in final[0].output
