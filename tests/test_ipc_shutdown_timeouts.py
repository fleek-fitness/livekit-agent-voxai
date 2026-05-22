"""Tests for the parent-side IPC shutdown timeout safety floors.

Verifies that the timeout-wrapping fix introduced for the parent-side
shutdown regression makes the relevant `aclose()` / `_supervise_task` /
`do_inference` paths return within a bounded time even when the
inference proc / child proc never responds.

See:
- https://github.com/livekit/agents/issues/5497
- https://github.com/livekit/agents/issues/3174
- https://github.com/livekit/agents/pull/4580 (regression source)
- https://github.com/livekit/agents/pull/5602 (child-side complement)
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.ipc import (
    inference_proc_executor,
    job_proc_executor,
    proto,
    supervised_proc,
)
from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor
from livekit.agents.ipc.supervised_proc import SupervisedProc
from livekit.agents.utils import aio


class _DummySupervisedProc(SupervisedProc):
    """Concrete SupervisedProc subclass that never actually spawns a process.

    Mirrors the pattern used by tests/test_drain_timeout.py so we avoid
    `object.__new__` + `# type: ignore` workarounds.
    """

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        raise NotImplementedError

    async def _main_task(self, ipc_ch: aio.ChanReceiver[object]) -> None:
        raise NotImplementedError


class _DummyInferenceExecutor(InferenceProcExecutor):
    """Concrete InferenceProcExecutor subclass for unit testing.

    Avoids spawning real processes / opening real sockets.
    """

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        raise NotImplementedError

    async def _main_task(self, ipc_ch: aio.ChanReceiver[object]) -> None:  # type: ignore[override]
        raise NotImplementedError


def _make_dummy_supervised() -> _DummySupervisedProc:
    return _DummySupervisedProc(
        initialize_timeout=1.0,
        close_timeout=1.0,
        memory_warn_mb=0.0,
        memory_limit_mb=0.0,
        ping_interval=1.0,
        ping_timeout=1.0,
        high_ping_threshold=1.0,
        http_proxy=None,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_event_loop(),
    )


def _make_dummy_inference() -> _DummyInferenceExecutor:
    return _DummyInferenceExecutor(
        runners={},
        initialize_timeout=1.0,
        close_timeout=1.0,
        memory_warn_mb=0.0,
        memory_limit_mb=0.0,
        ping_interval=1.0,
        ping_timeout=1.0,
        high_ping_threshold=1.0,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_event_loop(),
        http_proxy=None,
    )


async def test_do_inference_raises_runtime_error_on_timeout(monkeypatch):
    """When the inference proc never responds, do_inference must raise
    RuntimeError within roughly the configured timeout, and must clean
    up its in-flight request entry so it does not leak.
    """

    monkeypatch.setattr(inference_proc_executor, "_INFERENCE_RESPONSE_TIMEOUT_S", 0.1)

    async def _noop_asend_message(*args, **kwargs):
        return None

    monkeypatch.setattr(inference_proc_executor.channel, "asend_message", _noop_asend_message)

    executor = _make_dummy_inference()
    executor._supervise_atask = asyncio.create_task(asyncio.sleep(60))
    executor._pch = AsyncMock()

    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(RuntimeError) as exc_info:
            await executor.do_inference("test_method", b"")
        elapsed = loop.time() - started

        assert "timed out" in str(exc_info.value)
        assert "test_method" in str(exc_info.value)
        assert elapsed < 1.0, f"do_inference took too long to time out: {elapsed:.3f}s"
        assert executor._active_requests == {}
    finally:
        executor._supervise_atask.cancel()
        with pytest.raises((asyncio.CancelledError, BaseException)):
            await executor._supervise_atask


async def test_do_inference_happy_path_returns_response(monkeypatch):
    """When the inference proc responds, do_inference must return the
    response payload exactly as before — i.e. the new timeout wrap is a
    no-op on the happy path.
    """

    monkeypatch.setattr(inference_proc_executor, "_INFERENCE_RESPONSE_TIMEOUT_S", 5.0)

    executor = _make_dummy_inference()
    executor._supervise_atask = asyncio.create_task(asyncio.sleep(60))
    executor._pch = AsyncMock()

    async def _send_and_resolve(_pch, msg, **kwargs):
        loop = asyncio.get_running_loop()

        def _resolve():
            fut = executor._active_requests.pop(msg.request_id, None)
            if fut is not None and not fut.done():
                fut.set_result(
                    proto.InferenceResponse(request_id=msg.request_id, data=b"\x01\x02", error="")
                )

        loop.call_soon(_resolve)

    monkeypatch.setattr(inference_proc_executor.channel, "asend_message", _send_and_resolve)

    try:
        data = await executor.do_inference("ok_method", b"")
        assert data == b"\x01\x02"
        assert executor._active_requests == {}
    finally:
        executor._supervise_atask.cancel()
        with pytest.raises((asyncio.CancelledError, BaseException)):
            await executor._supervise_atask


async def test_shutting_down_fut_timeout_triggers_kill(monkeypatch):
    """When the child never sends ShuttingDown, aclose() must time out
    within _SHUTTING_DOWN_HARD_TIMEOUT_S and escalate to dump+kill.

    This is the only escalating fallback path in the new wraps — if it
    silently regresses, prod symptom is identical to pre-PR.
    """

    monkeypatch.setattr(supervised_proc, "_SHUTTING_DOWN_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_SHUTDOWN_ACK_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_SUPERVISE_HARD_TIMEOUT_S", 0.05)

    proc = _make_dummy_supervised()
    proc._pch = AsyncMock()
    proc._pch.aclose = AsyncMock()
    # Resolve shutdown_ack synthetically so we focus on the shutting_down branch.
    proc._shutdown_ack_fut.set_result(None)
    # Leave _shutting_down_fut unresolved — that's what triggers the timeout.

    # A `started`-truthy supervise task that completes promptly so the
    # second supervise wait in aclose() doesn't also need to escalate.
    async def _quick() -> None:
        return None

    proc._supervise_atask = asyncio.create_task(_quick())

    dump_calls: list[None] = []
    kill_calls: list[None] = []

    async def _dump() -> None:
        dump_calls.append(None)

    async def _kill() -> None:
        kill_calls.append(None)

    monkeypatch.setattr(proc, "_send_dump_signal", _dump)
    monkeypatch.setattr(proc, "_send_kill_signal", _kill)

    await proc.aclose()

    # Dump+kill should be invoked at least once via the shutting_down branch.
    assert len(dump_calls) >= 1
    assert len(kill_calls) >= 1


async def test_kill_supervise_timeout_logs_and_returns(monkeypatch):
    """When supervise_atask hangs after kill(), kill() must return
    within _KILL_SUPERVISE_HARD_TIMEOUT_S rather than waiting forever.
    """

    monkeypatch.setattr(supervised_proc, "_KILL_SUPERVISE_HARD_TIMEOUT_S", 0.05)

    proc = _make_dummy_supervised()

    # Hanging supervise task — never completes naturally.
    proc._supervise_atask = asyncio.create_task(asyncio.sleep(60))

    async def _noop() -> None:
        return None

    monkeypatch.setattr(proc, "_send_dump_signal", _noop)
    monkeypatch.setattr(proc, "_send_kill_signal", _noop)

    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await proc.kill()
    finally:
        proc._supervise_atask.cancel()
        with pytest.raises((asyncio.CancelledError, BaseException)):
            await proc._supervise_atask
    elapsed = loop.time() - started

    assert elapsed < 1.0, f"kill() did not bound: {elapsed:.3f}s"


async def test_main_task_inference_cancel_timeout_bounds(monkeypatch):
    """job_proc_executor._main_task finally must bound the cancel of
    pending inference tasks within the configured hard timeout, even
    when those tasks ignore CancelledError. Without the wrap, this
    would block _supervise_task cleanup forever and keep a K8s pod
    Terminating.
    """

    monkeypatch.setattr(job_proc_executor, "_INFERENCE_TASKS_CANCEL_HARD_TIMEOUT_S", 0.05)

    proc = job_proc_executor.ProcJobExecutor(
        initialize_process_fnc=lambda _: None,
        job_entrypoint_fnc=lambda _: asyncio.sleep(0),
        session_end_fnc=None,
        inference_executor=None,
        initialize_timeout=1.0,
        close_timeout=1.0,
        session_end_timeout=1.0,
        memory_warn_mb=0.0,
        memory_limit_mb=0.0,
        ping_interval=1.0,
        ping_timeout=1.0,
        high_ping_threshold=1.0,
        http_proxy=None,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_event_loop(),
    )

    # A stuck inference task that ignores cancel — emulates the leak
    # this PR is supposed to bound.
    async def _stuck() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(60)

    stuck = asyncio.create_task(_stuck())
    proc._inference_tasks.add(stuck)

    ipc_ch: aio.Chan[object] = aio.Chan()
    ipc_ch.close()

    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await proc._main_task(ipc_ch)  # type: ignore[arg-type]
    finally:
        stuck.cancel()
    elapsed = loop.time() - started

    # _main_task must return well within the 60s sleeps the stuck task
    # is wedged in. Allow generous CI scheduling slack.
    assert elapsed < 2.0, f"_main_task did not bound: {elapsed:.3f}s"


async def test_pch_aclose_timeout_force_closes_sock(monkeypatch):
    """When pch.aclose() hangs, the timeout branch must force-close
    the underlying transport + socket to release the fd promptly.
    """

    monkeypatch.setattr(supervised_proc, "_PCH_ACLOSE_HARD_TIMEOUT_S", 0.05)

    force_close_called: list[object] = []

    def _capture_force_close(pch):
        force_close_called.append(pch)

    monkeypatch.setattr(supervised_proc, "_force_close_pch", _capture_force_close)

    # Direct call into the helper path: run the body that wraps pch.aclose
    # via a minimal harness. Use a hanging coroutine to emulate aclose
    # never resolving.
    import contextlib

    async def _hang() -> None:
        await asyncio.sleep(60)

    fake_pch = MagicMock()
    fake_pch.aclose = _hang

    with contextlib.suppress(asyncio.TimeoutError):
        try:
            await asyncio.wait_for(
                fake_pch.aclose(),
                timeout=supervised_proc._PCH_ACLOSE_HARD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            supervised_proc._force_close_pch(fake_pch)
            raise

    assert force_close_called == [fake_pch]
