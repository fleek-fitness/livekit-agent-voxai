import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from livekit.agents.ipc import proc_pool, startup_timing as timing
from livekit.agents.ipc.proc_pool import ProcPool

pytestmark = pytest.mark.unit


def records(caplog):
    return [
        json.loads(r.getMessage().split(" ", 1)[1])
        for r in caplog.records
        if r.getMessage().startswith("agent_startup_timing ")
    ]


def test_safe_correlation_and_monotonic_duration(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="livekit.agents")
    ticks = iter([4.0, 4.1])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(ticks))
    with timing.startup_timing(
        "pool_acquire", job_id="AJ_test1", attempt=2, warm_at_request=False
    ) as fields:
        fields["process_id"] = "PCEXEC_synthetic"
        fields["room"] = "synthetic-room-sensitive-payload"
    r = records(caplog)[0]
    assert r["duration_ms"] == 100.0
    assert r["startup_job_id"] == "AJ_test1"
    assert r["attempt"] == 2
    assert r["warm_at_request"] is False
    assert "room" not in r
    assert "synthetic-room-sensitive-payload" not in caplog.text


@pytest.mark.parametrize("error", [RuntimeError("secret=synthetic"), asyncio.CancelledError()])
def test_exception_and_cancellation_preserved(error, caplog):
    caplog.set_level(logging.INFO, logger="livekit.agents")
    with pytest.raises(type(error)) as exc:
        with timing.startup_timing("process_initialize", job_id="invalid/synthetic-sensitive-id"):
            raise error
    assert exc.value is error
    r = records(caplog)[0]
    assert r["outcome"] == "failed_or_cancelled"
    assert "startup_job_id" not in r
    assert "synthetic" not in caplog.text


@pytest.mark.asyncio
async def test_concurrent_acquire_is_per_attempt_and_wait_count_balanced(caplog):
    caplog.set_level(logging.INFO, logger="livekit.agents")

    class Proc:
        id = "PCEXEC_test"

        async def launch_job(self, info):
            await asyncio.sleep(0)

    async def acquire(job):
        await asyncio.sleep(0)
        return Proc()

    pool = SimpleNamespace(
        _jobs_waiting_for_process=0,
        _acquire_proc=acquire,
        _warmed_proc_queue=asyncio.Queue(),
        emit=lambda *args: None,
    )
    launch = ProcPool.launch_job
    await asyncio.gather(
        *(
            launch(pool, SimpleNamespace(job=SimpleNamespace(id=job)))
            for job in ["AJ_one", "AJ_two"]
        )
    )
    assert pool._jobs_waiting_for_process == 0
    assert len(records(caplog)) == 4
    assert {r["startup_job_id"] for r in records(caplog)} == {"AJ_one", "AJ_two"}


@pytest.mark.asyncio
async def test_acquire_cancellation_balances_wait_count(caplog):
    caplog.set_level(logging.INFO, logger="livekit.agents")

    async def acquire(job):
        raise asyncio.CancelledError()

    pool = SimpleNamespace(
        _jobs_waiting_for_process=0, _acquire_proc=acquire, _warmed_proc_queue=asyncio.Queue()
    )
    with pytest.raises(asyncio.CancelledError):
        await ProcPool.launch_job(pool, SimpleNamespace(job=SimpleNamespace(id="AJ_cancel")))
    assert pool._jobs_waiting_for_process == 0
    assert records(caplog)[0]["outcome"] == "failed_or_cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_initialize_slot_is_released_without_changing_pool_policy(
    monkeypatch, caplog, cancel
):
    import multiprocessing

    caplog.set_level(logging.INFO, logger="livekit.agents")
    entered = asyncio.Event()
    release = asyncio.Event()

    class Proc:
        id = "PCEXEC_test"
        pid = 123
        closed = False

        async def start(self):
            pass

        async def initialize(self):
            entered.set()
            await release.wait()

        async def join(self):
            pass

        async def aclose(self):
            self.closed = True

        def logging_extra(self):
            return {}

    proc = Proc()
    monkeypatch.setattr(proc_pool.job_proc_executor, "ProcJobExecutor", lambda **kwargs: proc)
    pool = ProcPool(
        initialize_process_fnc=lambda _: None,
        job_entrypoint_fnc=lambda _: None,
        session_end_fnc=None,
        simulation_end_fnc=None,
        num_idle_processes=1,
        initialize_timeout=10,
        close_timeout=10,
        session_end_timeout=10,
        inference_executor=None,
        job_executor_type=proc_pool.JobExecutorType.PROCESS,
        mp_ctx=multiprocessing.get_context("spawn"),
        memory_warn_mb=600,
        memory_limit_mb=0,
        http_proxy=None,
        loop=asyncio.get_running_loop(),
    )
    pool._init_sem = asyncio.Semaphore(1)
    task = asyncio.create_task(pool._proc_spawn_task())
    await entered.wait()
    assert pool._init_sem.locked()
    if cancel:
        task.cancel()
    else:
        release.set()
    await task
    assert not pool._init_sem.locked()
    assert pool.target_idle_processes == 1
    assert proc.closed is cancel
    assert pool._warmed_proc_queue.qsize() == (0 if cancel else 1)
    await asyncio.gather(*pool._monitor_tasks)
    assert [r["phase"] for r in records(caplog)] == [
        "initialize_slot_wait",
        "process_start",
        "process_initialize",
    ]
    assert records(caplog)[-1]["outcome"] == ("failed_or_cancelled" if cancel else "succeeded")


@pytest.mark.asyncio
async def test_launch_retry_logs_distinct_attempts_and_preserves_retry_count(caplog):
    caplog.set_level(logging.INFO, logger="livekit.agents")
    attempts = 0

    class Proc:
        id = "PCEXEC_retry"

        async def launch_job(self, info):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("launch unavailable")

        async def aclose(self):
            pass

        def logging_extra(self):
            return {}

    async def acquire(job):
        return Proc()

    pool = SimpleNamespace(
        _jobs_waiting_for_process=0,
        _acquire_proc=acquire,
        _warmed_proc_queue=asyncio.Queue(),
        emit=lambda *args: None,
        _close_tasks=set(),
    )
    await ProcPool.launch_job(pool, SimpleNamespace(job=SimpleNamespace(id="AJ_retry")))
    await asyncio.gather(*pool._close_tasks)
    launches = [r for r in records(caplog) if r["phase"] == "job_launch"]
    assert [r["attempt"] for r in launches] == [1, 2, 3]
    assert [r["outcome"] for r in launches] == [
        "failed_or_cancelled",
        "failed_or_cancelled",
        "succeeded",
    ]
    assert pool._jobs_waiting_for_process == 0
