"""``function_tools_executed`` carries the wall-clock window of each tool callable.

The window starts right before the callable is awaited and ends right after it returns, so
time spent waiting before it runs, or between two tool rounds, is not part of it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.events import FunctionToolsExecutedEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class _SleepyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")

    @function_tool
    async def slow(self) -> str:
        """Takes one second."""
        await asyncio.sleep(1.0)
        return "slow done"

    @function_tool
    async def slower(self) -> str:
        """Takes two seconds."""
        await asyncio.sleep(2.0)
        return "slower done"


def _collect(session) -> list[FunctionToolsExecutedEvent]:
    events: list[FunctionToolsExecutedEvent] = []
    session.on("function_tools_executed", events.append)
    return events


async def test_window_is_the_callable_only_across_two_rounds() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "do it")
    actions.add_llm(
        content="Checking.",
        tool_calls=[FunctionToolCall(name="slow", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)
    actions.add_llm(
        content="Checking more.",
        input="slow done",
        tool_calls=[FunctionToolCall(name="slower", arguments="{}", call_id="2")],
    )
    actions.add_tts(1.0)
    actions.add_llm(content="All done.", input="slower done")
    actions.add_tts(1.0)

    session = create_session(actions)
    events = _collect(session)
    await asyncio.wait_for(run_session(session, _SleepyAgent(), drain_delay=15), timeout=60)

    windows = {}
    for ev in events:
        windows.update(ev.function_call_exec_windows)
    assert set(windows) == {"1", "2"}
    (s1, e1), (s2, e2) = windows["1"], windows["2"]
    assert e1 - s1 == pytest.approx(1.0, abs=0.05)
    assert e2 - s2 == pytest.approx(2.0, abs=0.05)
    # the second round's LLM and speech between the rounds is a gap, not tool time
    assert s2 - e1 > 0.5


async def test_parallel_calls_each_get_their_own_overlapping_window() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "do both")
    actions.add_llm(
        content="Checking.",
        tool_calls=[
            FunctionToolCall(name="slow", arguments="{}", call_id="1"),
            FunctionToolCall(name="slower", arguments="{}", call_id="2"),
        ],
    )
    actions.add_tts(1.0)
    actions.add_llm(content="Both done.", input="slow done\nslower done")
    actions.add_tts(1.0)

    session = create_session(actions)
    events = _collect(session)
    await asyncio.wait_for(run_session(session, _SleepyAgent(), drain_delay=10), timeout=60)

    windows = {}
    for ev in events:
        windows.update(ev.function_call_exec_windows)
    assert set(windows) == {"1", "2"}
    (s1, e1), (s2, e2) = windows["1"], windows["2"]
    assert e1 - s1 == pytest.approx(1.0, abs=0.05)
    assert e2 - s2 == pytest.approx(2.0, abs=0.05)
    assert s2 < e1  # they ran at the same time


async def test_call_that_never_ran_has_no_window() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "do it")
    actions.add_llm(
        content="Checking.",
        tool_calls=[FunctionToolCall(name="nonexistent", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)
    actions.add_llm(content="Sorry.", input="Unknown function")
    actions.add_tts(1.0)

    session = create_session(actions)
    events = _collect(session)
    await asyncio.wait_for(run_session(session, _SleepyAgent(), drain_delay=8), timeout=60)

    assert events
    assert all(ev.function_call_exec_windows == {} for ev in events)
