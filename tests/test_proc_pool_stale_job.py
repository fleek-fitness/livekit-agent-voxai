"""A real job that waited too long for a process is dropped instead of launched.

LiveKit re-sends a job that this worker has not started to another worker, and the first worker
is never told. Launching it anyway would run the same job twice.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from livekit.agents.ipc import proc_pool
from livekit.agents.ipc.proc_pool import MAX_JOB_PROCESS_WAIT_S, ProcPool

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class FakeProc:
    id = "PCEXEC_stale"

    def __init__(self, launch_error: Exception | None = None) -> None:
        self.launched: list[object] = []
        self._launch_error = launch_error

    async def launch_job(self, info) -> None:
        if self._launch_error is not None:
            raise self._launch_error
        self.launched.append(info)

    async def aclose(self) -> None:
        pass

    def logging_extra(self) -> dict:
        return {}


def make_pool(monkeypatch, clock: FakeClock, waits: list[float], procs: list[FakeProc]):
    """Pool whose n-th acquire takes waits[n] seconds of fake time and returns procs[n]."""
    monkeypatch.setattr(proc_pool, "time", SimpleNamespace(monotonic=clock.monotonic))
    acquired = iter(zip(waits, procs, strict=True))

    async def acquire(job_id: str) -> FakeProc:
        wait, proc = next(acquired)
        clock.now += wait
        return proc

    events: list[tuple] = []
    pool = SimpleNamespace(
        _jobs_waiting_for_process=0,
        _acquire_proc=acquire,
        _warmed_proc_queue=asyncio.Queue(),
        emit=lambda *args: events.append(args),
        _close_tasks=set(),
        events=events,
    )
    return pool


def job_info(job_id: str = "AJ_stale"):
    return SimpleNamespace(job=SimpleNamespace(id=job_id))


async def test_slow_acquire_drops_job_and_returns_proc_to_warm_queue(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [MAX_JOB_PROCESS_WAIT_S + 0.5], [proc])

    assert await ProcPool.launch_job(pool, job_info("AJ_slow"), reassignable=True) is None

    assert proc.launched == []
    assert pool._warmed_proc_queue.qsize() == 1
    assert pool._warmed_proc_queue.get_nowait() is proc
    assert pool.events == []  # no process_job_launched
    assert pool._jobs_waiting_for_process == 0
    assert len(pool._close_tasks) == 0

    dropped = [r for r in caplog.records if "dropping job" in r.getMessage()]
    assert len(dropped) == 1
    assert dropped[0].levelno == logging.WARNING
    assert dropped[0].job_id == "AJ_slow"
    assert dropped[0].waited_s == pytest.approx(MAX_JOB_PROCESS_WAIT_S + 0.5)
    assert "reassigned" in dropped[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_fast_acquire_launches_job(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [1.0], [proc])
    info = job_info()

    await ProcPool.launch_job(pool, info, reassignable=True)

    assert proc.launched == [info]
    assert pool._warmed_proc_queue.empty()
    assert pool.events == [("process_job_launched", proc)]
    assert not [r for r in caplog.records if "dropping job" in r.getMessage()]


async def test_wait_exactly_at_the_limit_still_launches(monkeypatch):
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [MAX_JOB_PROCESS_WAIT_S], [proc])

    await ProcPool.launch_job(pool, job_info(), reassignable=True)

    assert len(proc.launched) == 1
    assert pool._warmed_proc_queue.empty()


async def test_job_not_marked_reassignable_is_never_dropped(monkeypatch):
    # simulated, reloaded and hand-connected jobs never get reassigned by the server
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [MAX_JOB_PROCESS_WAIT_S * 10], [proc])
    info = job_info("mock-job-1")

    await ProcPool.launch_job(pool, info)

    assert proc.launched == [info]
    assert pool._warmed_proc_queue.empty()
    assert pool.events == [("process_job_launched", proc)]


async def test_wait_is_measured_from_entry_across_launch_retries(monkeypatch):
    failing = FakeProc(launch_error=RuntimeError("launch unavailable"))
    second = FakeProc()
    # each acquire is quick, but together with the failed first attempt the job is too old
    pool = make_pool(monkeypatch, FakeClock(), [6.0, 6.0], [failing, second])
    assert 6.0 < MAX_JOB_PROCESS_WAIT_S < 12.0

    await ProcPool.launch_job(pool, job_info(), reassignable=True)
    await asyncio.gather(*pool._close_tasks)

    assert second.launched == []
    assert pool._warmed_proc_queue.get_nowait() is second
    assert pool._jobs_waiting_for_process == 0


async def test_job_assigned_by_the_server_is_launched_as_reassignable() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from livekit.agents.worker import AgentServer
    from livekit.protocol import agent, models

    server = AgentServer(
        ws_url="ws://127.0.0.1:1",
        api_key="devkey",
        api_secret="devsecret",
        num_idle_processes=0,
    )
    server._proc_pool = MagicMock()
    server._proc_pool.launch_job = AsyncMock()
    server._refresh_worker_load = AsyncMock()  # type: ignore[method-assign]
    server._queue_msg = AsyncMock()  # type: ignore[method-assign]
    server._is_available = lambda: True  # type: ignore[method-assign]
    # run() initialises these in production
    server._reserved_slots = 0
    server._loop = asyncio.get_running_loop()
    server._pending_assignments = {}
    server._job_lifecycle_tasks = set()

    async def request_fnc(req) -> None:
        await req.accept()

    server._request_fnc = request_fnc

    job = agent.Job(id="AJ_assigned", room=models.Room(name="room"))
    await server._answer_availability(agent.AvailabilityRequest(job=job))
    while "AJ_assigned" not in server._pending_assignments:
        await asyncio.sleep(0)
    server._handle_assignment(agent.JobAssignment(job=job, url="ws://127.0.0.1:1", token="t"))
    await asyncio.gather(*server._job_lifecycle_tasks)

    server._proc_pool.launch_job.assert_awaited_once()
    (running_info,), kwargs = server._proc_pool.launch_job.call_args
    assert running_info.job.id == "AJ_assigned"
    assert running_info.fake_job is False
    assert kwargs == {"reassignable": True}
    assert server._reserved_slots == 0
