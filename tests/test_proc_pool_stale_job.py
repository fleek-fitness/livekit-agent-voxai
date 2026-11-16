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


def job_info(job_id: str = "AJ_stale", *, fake_job: bool = False):
    return SimpleNamespace(job=SimpleNamespace(id=job_id), fake_job=fake_job)


async def test_slow_acquire_drops_job_and_returns_proc_to_warm_queue(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [MAX_JOB_PROCESS_WAIT_S + 0.5], [proc])

    assert await ProcPool.launch_job(pool, job_info("AJ_slow")) is None

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

    await ProcPool.launch_job(pool, info)

    assert proc.launched == [info]
    assert pool._warmed_proc_queue.empty()
    assert pool.events == [("process_job_launched", proc)]
    assert not [r for r in caplog.records if "dropping job" in r.getMessage()]


async def test_wait_exactly_at_the_limit_still_launches(monkeypatch):
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [MAX_JOB_PROCESS_WAIT_S], [proc])

    await ProcPool.launch_job(pool, job_info())

    assert len(proc.launched) == 1
    assert pool._warmed_proc_queue.empty()


async def test_fake_job_is_never_dropped(monkeypatch):
    proc = FakeProc()
    pool = make_pool(monkeypatch, FakeClock(), [MAX_JOB_PROCESS_WAIT_S * 10], [proc])
    info = job_info("mock-job-1", fake_job=True)

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

    await ProcPool.launch_job(pool, job_info())
    await asyncio.gather(*pool._close_tasks)

    assert second.launched == []
    assert pool._warmed_proc_queue.get_nowait() is second
    assert pool._jobs_waiting_for_process == 0
