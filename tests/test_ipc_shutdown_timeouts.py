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
import contextlib
import multiprocessing as mp
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.ipc import (
    inference_proc_executor,
    job_proc_executor,
    job_thread_executor,
    proto,
    supervised_proc,
)
from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor
from livekit.agents.ipc.job_executor import JobStatus
from livekit.agents.ipc.job_thread_executor import ThreadJobExecutor
from livekit.agents.ipc.supervised_proc import SupervisedProc
from livekit.agents.utils import aio
from livekit.agents.utils.aio import duplex_unix


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
    """When `_supervise_task()` reaches its tail-end `pch.aclose()` and
    that aclose hangs, the production path must time out within
    `_PCH_ACLOSE_HARD_TIMEOUT_S` and invoke `_force_close_pch` so the fd
    is released.

    Drives the real `SupervisedProc._supervise_task()` body — not a
    reimplementation of the wait_for+force_close pattern — so a future
    edit that drops the wrap or moves the call site will break this test
    instead of silently passing.
    """

    monkeypatch.setattr(supervised_proc, "_PCH_ACLOSE_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_HELPERS_CANCEL_HARD_TIMEOUT_S", 0.05)

    force_close_called: list[object] = []
    monkeypatch.setattr(
        supervised_proc, "_force_close_pch", lambda pch: force_close_called.append(pch)
    )

    proc = _make_dummy_supervised()

    # Short-circuit the helper tasks that `_supervise_task` spawns so it
    # reaches the pch.aclose tail without spinning forever.
    async def _noop_main(ipc_ch: aio.ChanReceiver[object]) -> None:
        return None

    async def _noop_read(ipc_ch: aio.Chan[object], pong_timeout: aio.Sleep) -> None:
        return None

    async def _noop_ping(pong_timeout: aio.Sleep) -> None:
        return None

    monkeypatch.setattr(proc, "_main_task", _noop_main)
    monkeypatch.setattr(proc, "_read_ipc_task", _noop_read)
    monkeypatch.setattr(proc, "_ping_pong_task", _noop_ping)

    # Pre-resolve `_initialize_fut` and `_join_fut` so `_supervise_task`
    # walks past them straight into the cleanup phase.
    proc._initialize_fut.set_result(None)
    proc._join_fut = asyncio.Future[None]()
    proc._join_fut.set_result(None)

    fake_proc = MagicMock()
    fake_proc.exitcode = 0
    fake_proc.close = MagicMock()
    proc._proc = fake_proc

    # `_pch.aclose` hangs — this is the regression site the wrap is
    # supposed to defend.
    fake_pch = MagicMock(spec=duplex_unix._AsyncDuplex)

    async def _hang_aclose() -> None:
        await asyncio.sleep(60)

    fake_pch.aclose = _hang_aclose
    proc._pch = fake_pch

    loop = asyncio.get_running_loop()
    started = loop.time()
    await proc._supervise_task()
    elapsed = loop.time() - started

    assert force_close_called == [fake_pch], (
        f"_force_close_pch not invoked via production path: {force_close_called!r}"
    )
    # Bounded by HELPERS_CANCEL + PCH_ACLOSE timeouts (both 0.05s); generous
    # slack for CI scheduling.
    assert elapsed < 2.0, f"_supervise_task did not bound pch.aclose: {elapsed:.3f}s"


async def test_thread_main_task_does_not_false_success_while_thread_runs(monkeypatch):
    """Regression guard for `_main_task`'s `_join_fut` wait.

    The PR's first attempt bounded the wait with `_JOIN_FUT_HARD_TIMEOUT_S`
    (30s) and unconditionally set `JobStatus.SUCCESS` on fall-through.
    That falsely reported normal long-running calls (typical voice agents
    run for minutes) as SUCCESS while the underlying thread was still
    alive. The wait must stay unbounded; bounding lives in `aclose()` via
    the shielded `_main_atask`.
    """

    loop = asyncio.get_running_loop()
    executor = ThreadJobExecutor(
        initialize_process_fnc=lambda _: None,
        job_entrypoint_fnc=lambda _: asyncio.sleep(0),
        session_end_fnc=None,
        inference_executor=None,
        initialize_timeout=1.0,
        close_timeout=1.0,
        session_end_timeout=1.0,
        ping_interval=10.0,
        high_ping_threshold=10.0,
        http_proxy=None,
        loop=loop,
    )

    # Short-circuit ping/monitor helpers that `_main_task` spawns —
    # otherwise they'd try to read from a non-existent `_pch`.
    async def _noop_ping() -> None:
        await asyncio.sleep(60)

    async def _noop_monitor() -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(executor, "_ping_task", _noop_ping)
    monkeypatch.setattr(executor, "_monitor_task", _noop_monitor)

    executor._pch = AsyncMock()
    executor._initialize_fut.set_result(None)
    executor._join_fut = asyncio.Future[None]()  # unresolved — thread "still running"

    main_task = asyncio.create_task(executor._main_task())

    # Wait longer than the old `_JOIN_FUT_HARD_TIMEOUT_S=30.0`s would have
    # been monkeypatched to (had it survived). 0.2s is plenty of real wall
    # clock to expose a still-bounded wait misfiring.
    await asyncio.sleep(0.2)

    try:
        assert not main_task.done(), (
            "regression: _main_task returned while _join_fut still pending"
            " — bounded wait re-introduced on the normal path"
        )
        assert executor._job_status is None, (
            f"regression: _job_status set to {executor._job_status!r} while"
            " thread is still running — false SUCCESS emission"
        )

        # Resolving the join future should now let _main_task complete
        # cleanly with SUCCESS, proving the wait was the only blocker.
        executor._join_fut.set_result(None)
        await asyncio.wait_for(main_task, timeout=2.0)
        assert executor._job_status == JobStatus.SUCCESS, (
            f"_main_task did not converge to SUCCESS after join: {executor._job_status!r}"
        )
    finally:
        if not main_task.done():
            main_task.cancel()
            with contextlib.suppress(BaseException):
                await main_task


async def test_supervised_aclose_idempotent_under_ack_timeout(monkeypatch):
    """Repeated `SupervisedProc.aclose()` after a first ack timeout must
    not raise `CancelledError`.

    Before the shield fix, `wait_for(self._shutdown_ack_fut, ...)` would
    cancel `_shutdown_ack_fut` on timeout. The next aclose() call (pool
    double-close, launch-failure cleanup) would then re-await the cancelled
    future and immediately raise `CancelledError`, breaking idempotency.
    """

    monkeypatch.setattr(supervised_proc, "_SHUTDOWN_ACK_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_SHUTTING_DOWN_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_SUPERVISE_HARD_TIMEOUT_S", 0.05)

    proc = _make_dummy_supervised()
    proc._pch = AsyncMock()
    proc._pch.aclose = AsyncMock()

    async def _noop() -> None:
        return None

    proc._supervise_atask = asyncio.create_task(_noop())
    monkeypatch.setattr(proc, "_send_dump_signal", _noop)
    monkeypatch.setattr(proc, "_send_kill_signal", _noop)

    # First aclose: shutdown_ack times out → escalates to dump+kill.
    await proc.aclose()

    # The shield must keep `_shutdown_ack_fut` pending across the timeout.
    assert not proc._shutdown_ack_fut.cancelled(), (
        "shield missing: first aclose() cancelled _shutdown_ack_fut"
    )
    assert not proc._shutdown_ack_fut.done(), (
        "unexpected: _shutdown_ack_fut resolved by something other than producer"
    )

    # Second aclose: must not raise CancelledError. (Re-arm supervise_atask
    # so the path is fully re-entered; the supervise wait is already
    # shielded so it's not the regression target here.)
    proc._supervise_atask = asyncio.create_task(_noop())
    try:
        await proc.aclose()
    except asyncio.CancelledError as exc:  # pragma: no cover — regression
        pytest.fail(f"second aclose() raised CancelledError (shield regression): {exc!r}")


async def test_thread_aclose_idempotent_under_ack_timeout(monkeypatch):
    """ThreadJobExecutor mirror of `test_supervised_aclose_idempotent_under_ack_timeout`.

    The same shield-on-`_shutdown_ack_fut` invariant must hold for the
    threaded executor; both call sites had identical unshielded `wait_for`.
    """

    monkeypatch.setattr(supervised_proc, "_SHUTDOWN_ACK_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_SHUTTING_DOWN_HARD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(supervised_proc, "_SUPERVISE_HARD_TIMEOUT_S", 0.05)

    loop = asyncio.get_running_loop()
    executor = ThreadJobExecutor(
        initialize_process_fnc=lambda _: None,
        job_entrypoint_fnc=lambda _: asyncio.sleep(0),
        session_end_fnc=None,
        inference_executor=None,
        initialize_timeout=1.0,
        close_timeout=1.0,
        session_end_timeout=1.0,
        ping_interval=10.0,
        high_ping_threshold=10.0,
        http_proxy=None,
        loop=loop,
    )

    executor._pch = AsyncMock()
    executor._pch.aclose = AsyncMock()

    async def _noop() -> None:
        return None

    executor._main_atask = asyncio.create_task(_noop())  # marks `started`

    # First aclose: ack times out (future unresolved).
    await executor.aclose()

    assert not executor._shutdown_ack_fut.cancelled(), (
        "shield missing: first aclose() cancelled _shutdown_ack_fut (thread)"
    )
    assert not executor._shutdown_ack_fut.done()

    executor._main_atask = asyncio.create_task(_noop())
    try:
        await executor.aclose()
    except asyncio.CancelledError as exc:  # pragma: no cover — regression
        pytest.fail(f"second thread aclose() raised CancelledError (shield regression): {exc!r}")


async def test_thread_start_exception_bounds_pch_aclose(monkeypatch):
    """When `ThreadJobExecutor._start()` raises after `pch` is created,
    the exception-cleanup `pch.aclose()` must be bounded by
    `_START_CLEANUP_PCH_ACLOSE_HARD_TIMEOUT_S` and fall back to
    `_force_close_pch` on timeout — matching the process-side behavior.

    Before this fix, the thread `_start` exception path had a bare
    `await pch.aclose()` that would hang indefinitely if the duplex
    flush stalled (the symptom this PR is supposed to defend against).
    """

    monkeypatch.setattr(job_thread_executor, "_START_CLEANUP_PCH_ACLOSE_HARD_TIMEOUT_S", 0.05)

    fake_pch = MagicMock(spec=duplex_unix._AsyncDuplex)

    async def _hang_aclose() -> None:
        await asyncio.sleep(60)

    fake_pch.aclose = _hang_aclose

    async def _fake_open(sock):
        return fake_pch

    monkeypatch.setattr(duplex_unix._AsyncDuplex, "open", staticmethod(_fake_open))

    class _BoomThread:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated thread construction failure")

    monkeypatch.setattr(job_thread_executor.threading, "Thread", _BoomThread)

    force_close_called: list[object] = []
    monkeypatch.setattr(
        job_thread_executor,
        "_force_close_pch",
        lambda pch: force_close_called.append(pch),
    )

    loop = asyncio.get_running_loop()
    executor = ThreadJobExecutor(
        initialize_process_fnc=lambda _: None,
        job_entrypoint_fnc=lambda _: asyncio.sleep(0),
        session_end_fnc=None,
        inference_executor=None,
        initialize_timeout=1.0,
        close_timeout=1.0,
        session_end_timeout=1.0,
        ping_interval=10.0,
        high_ping_threshold=10.0,
        http_proxy=None,
        loop=loop,
    )

    started = loop.time()
    with pytest.raises(RuntimeError, match="simulated thread construction failure"):
        await executor.start()
    elapsed = loop.time() - started

    assert force_close_called == [fake_pch], (
        "_force_close_pch was not invoked from _start() exception cleanup"
    )
    # _start cleanup is bounded by 0.05s pch.aclose timeout; generous slack.
    assert elapsed < 1.0, f"_start exception cleanup did not bound: {elapsed:.3f}s"
