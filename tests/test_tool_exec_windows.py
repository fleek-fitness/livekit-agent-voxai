"""``function_tools_executed`` carries the wall-clock window of each tool callable.

The window starts right before the callable is awaited and ends right after it returns, so
time spent waiting before it runs, or between two tool rounds, is not part of it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, RunContext, function_tool
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


def _ended(session) -> dict:
    from livekit.agents.voice.events import ToolCallEnded, ToolExecutionUpdatedEvent

    ended: dict = {}

    def _on(ev: ToolExecutionUpdatedEvent) -> None:
        if isinstance(ev.update, ToolCallEnded):
            ended[ev.update.call_id] = ev.update

    session.on("tool_execution_updated", _on)
    return ended


def _async_actions(tool: str, args: str = "{}") -> FakeActions:
    from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm(
        content="Checking.",
        tool_calls=[FunctionToolCall(name=tool, arguments=args, call_id="1")],
    )
    actions.add_tts(1.0)
    actions.add_llm(
        content="Working on it.",
        input=UPDATE_TEMPLATE.format(function_name=tool, call_id="1", message="started"),
    )
    actions.add_tts(1.0)
    return actions


class _AsyncAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")

    @function_tool
    async def long_job(self, context: RunContext) -> str:
        """Updates after one second, finishes after five."""
        await asyncio.sleep(1.0)
        await context.update("started")
        await asyncio.sleep(4.0)
        return "done"

    @function_tool
    async def endless_job(self, context: RunContext) -> str:
        """Updates after one second, then never finishes on its own."""
        await asyncio.sleep(1.0)
        await context.update("started")
        await asyncio.sleep(1000)
        return "never"

    @function_tool
    async def needs_arg(self, must_have: str) -> str:
        """Needs an argument the model does not send."""
        return must_have


async def test_async_tool_window_runs_to_its_real_end_after_the_first_update() -> None:
    session = create_session(_async_actions("long_job"))
    events = _collect(session)
    ended = _ended(session)
    await asyncio.wait_for(run_session(session, _AsyncAgent(), drain_delay=15), timeout=60)

    first_start, first_end = events[0].function_call_exec_windows["1"]
    # the first notice only knows the tool is still going
    assert first_end - first_start == pytest.approx(1.0, abs=0.1)
    start, end = ended["1"].exec_window
    assert start == first_start
    assert end - start == pytest.approx(5.0, abs=0.1)
    assert ended["1"].status == "done"


async def test_cancelled_async_tool_window_ends_at_the_cancel() -> None:
    session = create_session(_async_actions("endless_job"))
    ended = _ended(session)

    async def _cancel_later() -> None:
        await asyncio.sleep(7.0)
        await session._activity._tool_executor.cancel_all()

    task = asyncio.create_task(_cancel_later())
    await asyncio.wait_for(run_session(session, _AsyncAgent(), drain_delay=15), timeout=60)
    await task

    start, end = ended["1"].exec_window
    assert ended["1"].status == "cancelled"
    assert 3.0 < end - start < 10.0  # not the full sleep, not just the first second


async def test_call_cancelled_before_the_callable_has_no_window() -> None:
    session = create_session(_async_actions("needs_arg"))
    ended = _ended(session)
    await asyncio.wait_for(run_session(session, _AsyncAgent(), drain_delay=10), timeout=60)

    assert ended["1"].exec_window is None


async def _drive_realtime_tool(tool_seconds: float, *, interrupt_after: float | None):
    from livekit import rtc
    from livekit.agents import AgentSession, llm, utils

    from .fake_io import FakeAudioOutput
    from .fake_realtime import FakeRealtimeModel, _audio_frame, fake_capabilities

    started = asyncio.Event()

    class RtAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            started.set()
            await asyncio.sleep(tool_seconds)
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities())
    async with AgentSession(llm=model) as session:
        events = _collect(session)
        session.output.audio = FakeAudioOutput(can_pause=True)
        await session.start(RtAgent())
        rt_session = model.active_session

        handle = session.generate_reply()
        while not rt_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("let me check")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()
        function_ch.send_nowait(llm.FunctionCall(call_id="1", name="lookup", arguments="{}"))
        function_ch.close()
        rt_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        await asyncio.wait_for(started.wait(), timeout=5)
        if interrupt_after is not None:
            handle.interrupt(force=True)
            await asyncio.sleep(interrupt_after)
        else:
            await asyncio.sleep(tool_seconds + 1.0)
        return events


async def test_realtime_normal_path_carries_the_window() -> None:
    events = await _drive_realtime_tool(2.0, interrupt_after=None)

    windows = {}
    for ev in events:
        windows.update(ev.function_call_exec_windows)
    start, end = windows["1"]
    assert end - start == pytest.approx(2.0, abs=0.1)


async def test_realtime_interrupted_path_carries_the_window() -> None:
    from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT

    # the tool outlives the interruption timeout, so the released placeholder is committed
    events = await _drive_realtime_tool(
        INTERRUPTION_TIMEOUT + 3, interrupt_after=INTERRUPTION_TIMEOUT + 0.5
    )

    windows = {}
    for ev in events:
        windows.update(ev.function_call_exec_windows)
    assert "1" in windows


async def test_event_names_the_speech_that_ran_the_tools() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "do it")
    actions.add_llm(
        content="Checking.",
        tool_calls=[FunctionToolCall(name="slow", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)
    actions.add_llm(content="Done.", input="slow done")
    actions.add_tts(1.0)

    session = create_session(actions)
    events = _collect(session)
    speech_ids: list[str] = []
    session.on("speech_created", lambda ev: speech_ids.append(ev.speech_handle.id))
    await asyncio.wait_for(run_session(session, _SleepyAgent(), drain_delay=10), timeout=60)

    assert events
    assert all(ev.speech_id in speech_ids for ev in events)
