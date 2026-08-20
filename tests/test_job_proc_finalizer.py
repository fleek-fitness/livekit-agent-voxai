from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import JobExecutorType
from livekit.agents.ipc.job_proc_lazy_main import _JobProc

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
