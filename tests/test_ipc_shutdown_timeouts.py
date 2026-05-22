"""Tests for the parent-side IPC shutdown timeout safety floors.

Verifies that the timeout-wrapping fix introduced for the
1.4.x/1.5.x parent-side shutdown regression makes
`InferenceProcExecutor.do_inference` return within a bounded time
even when the inference proc never responds.

See:
- https://github.com/livekit/agents/issues/5497
- https://github.com/livekit/agents/issues/3174
- https://github.com/livekit/agents/pull/4580 (regression source)
- https://github.com/livekit/agents/pull/5602 (child-side complement)
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.ipc import inference_proc_executor


async def _noop_asend_message(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_do_inference_raises_runtime_error_on_timeout(monkeypatch):
    """When the inference proc never responds, ``do_inference`` must raise
    ``RuntimeError`` within roughly the configured timeout, and must clean
    up its in-flight request entry so it does not leak.
    """

    # Patch the module timeout down to something well under any plausible
    # CI delay so the test finishes quickly.
    monkeypatch.setattr(
        inference_proc_executor,
        "_INFERENCE_RESPONSE_TIMEOUT_S",
        0.1,
    )

    # The `channel.asend_message` call is the only outbound IPC the
    # function under test performs before awaiting the response future.
    # Replace it with a noop so we never touch a real socket.
    monkeypatch.setattr(
        inference_proc_executor.channel,
        "asend_message",
        _noop_asend_message,
    )

    executor = object.__new__(inference_proc_executor.InferenceProcExecutor)
    executor._active_requests = {}
    executor._pch = None
    # `started` is `return self._supervise_atask is not None`, so a
    # truthy sentinel here is enough to satisfy the property without
    # needing a real asyncio.Task.
    executor._supervise_atask = object()  # type: ignore[assignment]
    # `logging_extra()` is called inside the timeout branch.
    executor.logging_extra = lambda: {}  # type: ignore[method-assign]

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(RuntimeError) as exc_info:
        await executor.do_inference("test_method", b"")
    elapsed = loop.time() - started

    assert "timed out" in str(exc_info.value)
    assert "test_method" in str(exc_info.value)
    assert elapsed < 1.0, f"do_inference took too long to time out: {elapsed:.3f}s"
    # The in-flight request entry must be cleaned up on timeout so the
    # executor does not leak fut references across hangs.
    assert executor._active_requests == {}


@pytest.mark.asyncio
async def test_do_inference_happy_path_returns_response(monkeypatch):
    """When the inference proc responds, ``do_inference`` must return the
    response payload exactly as before — i.e. the new timeout wrap is a
    no-op on the happy path.
    """

    monkeypatch.setattr(
        inference_proc_executor,
        "_INFERENCE_RESPONSE_TIMEOUT_S",
        5.0,
    )

    executor = object.__new__(inference_proc_executor.InferenceProcExecutor)
    executor._active_requests = {}
    executor._pch = None
    executor._supervise_atask = object()  # type: ignore[assignment]
    executor.logging_extra = lambda: {}  # type: ignore[method-assign]

    async def _send_and_resolve(_pch, msg, **kwargs):
        # Simulate the inference proc responding by completing the
        # matching future on the next tick. In production, `_main_task`
        # is the one that pops from `_active_requests` when the response
        # arrives; mirror that so the dict ends up clean.
        loop = asyncio.get_running_loop()

        def _resolve():
            fut = executor._active_requests.pop(msg.request_id, None)
            if fut is not None and not fut.done():
                from livekit.agents.ipc import proto

                fut.set_result(
                    proto.InferenceResponse(request_id=msg.request_id, data=b"\x01\x02", error="")
                )

        loop.call_soon(_resolve)

    monkeypatch.setattr(
        inference_proc_executor.channel,
        "asend_message",
        _send_and_resolve,
    )

    data = await executor.do_inference("ok_method", b"")
    assert data == b"\x01\x02"
    assert executor._active_requests == {}
