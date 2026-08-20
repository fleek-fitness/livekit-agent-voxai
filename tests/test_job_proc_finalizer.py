from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import JobExecutorType
from livekit.agents.ipc.job_proc_lazy_main import _JobProc
from livekit.agents.utils import http_context

pytestmark = pytest.mark.unit


def _create_job_proc(session_end_fnc: AsyncMock) -> _JobProc:
    proc = _JobProc(
        MagicMock(),
        MagicMock(),
        session_end_fnc,
        session_end_timeout=1.0,
        executor_type=JobExecutorType.PROCESS,
    )
    proc._job_ctx = MagicMock()
    proc._job_ctx._on_session_end = AsyncMock()
    proc._session_cleanup_done = asyncio.Event()
    return proc


async def test_session_finalizer_is_created_once_and_waits_for_cleanup() -> None:
    session_end_fnc = AsyncMock()
    proc = _create_job_proc(session_end_fnc)

    first = proc._ensure_session_finalizer_task()
    second = proc._ensure_session_finalizer_task()

    assert first is second
    await asyncio.sleep(0)
    session_end_fnc.assert_not_awaited()

    proc._session_cleanup_done.set()
    await proc._await_session_finalizer()

    session_end_fnc.assert_awaited_once_with(proc._job_ctx)
    proc._job_ctx._on_session_end.assert_awaited_once_with()


async def test_cancelling_waiter_does_not_cancel_owned_finalizer() -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    callback_count = 0

    async def session_end_fnc(job_ctx: object) -> None:
        nonlocal callback_count
        callback_count += 1
        callback_started.set()
        await release_callback.wait()

    proc = _create_job_proc(AsyncMock(side_effect=session_end_fnc))
    proc._session_cleanup_done.set()

    waiter = asyncio.create_task(proc._await_session_finalizer())
    await callback_started.wait()
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    finalizer = proc._ensure_session_finalizer_task()
    assert not finalizer.done()

    release_callback.set()
    await proc._await_session_finalizer()

    assert callback_count == 1
    proc._job_ctx._on_session_end.assert_awaited_once_with()


async def test_child_cancelled_callback_does_not_skip_internal_finalizer() -> None:
    session_end_fnc = AsyncMock(side_effect=asyncio.CancelledError)
    proc = _create_job_proc(session_end_fnc)
    proc._session_cleanup_done.set()

    await proc._await_session_finalizer()

    session_end_fnc.assert_awaited_once_with(proc._job_ctx)
    proc._job_ctx._on_session_end.assert_awaited_once_with()


async def test_parent_cancellation_is_replayed_after_complete_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_close_started = asyncio.Event()

    async def close_session() -> None:
        session_close_started.set()
        await asyncio.Event().wait()

    async def entrypoint(job_ctx: object) -> None:
        return None

    session_end_fnc = AsyncMock()
    proc = _JobProc(
        MagicMock(),
        entrypoint,
        session_end_fnc,
        session_end_timeout=1.0,
        executor_type=JobExecutorType.PROCESS,
    )
    proc._client = MagicMock()
    proc._client.send = AsyncMock()
    proc._room = MagicMock()
    proc._room.disconnect = AsyncMock()
    proc._shutdown_fut = asyncio.Future()
    proc._shutdown_fut.set_result(MagicMock(reason="test shutdown", user_initiated=False))
    proc._session_cleanup_done = asyncio.Event()

    shutdown_callback = AsyncMock()
    proc._job_ctx = MagicMock()
    proc._job_ctx.job.id = "00000000-0000-0000-0000-000000000001"
    proc._job_ctx.job.agent_name = "test-agent"
    proc._job_ctx.job.room.name = "test-room"
    proc._job_ctx._primary_agent_session.aclose = close_session
    proc._job_ctx._shutdown_callbacks = [shutdown_callback]
    proc._job_ctx._pending_tasks = []
    proc._job_ctx._on_session_end = AsyncMock()

    close_http_ctx = AsyncMock()
    monkeypatch.setattr(http_context, "_new_session_ctx", MagicMock())
    monkeypatch.setattr(http_context, "_close_http_ctx", close_http_ctx)

    job_task = asyncio.create_task(proc._run_job_task())
    await session_close_started.wait()
    job_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await job_task

    session_end_fnc.assert_awaited_once_with(proc._job_ctx)
    proc._job_ctx._on_session_end.assert_awaited_once_with()
    assert proc._client.send.await_count == 2
    proc._room.disconnect.assert_awaited_once_with()
    shutdown_callback.assert_awaited_once_with("test shutdown")
    proc._job_ctx._on_cleanup.assert_called_once_with()
    close_http_ctx.assert_awaited_once_with()
