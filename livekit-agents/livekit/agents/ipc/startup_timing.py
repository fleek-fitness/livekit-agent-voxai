from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..log import logger
from ..version import __version__


@contextmanager
def startup_timing(
    phase: str,
    *,
    job_id: str | None = None,
    process_id: str | None = None,
    child_pid: int | None = None,
    attempt: int | None = None,
    warm_at_request: bool | None = None,
) -> Iterator[dict[str, Any]]:
    """Log observed boundaries only; never serialize a Job or logging_extra()."""
    started = time.monotonic()
    fields: dict[str, Any] = {"process_id": process_id, "child_pid": child_pid}
    outcome = "succeeded"
    try:
        yield fields
    except BaseException:
        outcome = "failed_or_cancelled"
        raise
    finally:
        # Explicit allowlist also protects fields filled after process acquisition.
        record: dict[str, Any] = {
            "phase": phase,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            "outcome": outcome,
            "sdk_version": __version__,
            "attempt": attempt,
            "warm_at_request": warm_at_request,
        }
        for key, value in [("startup_job_id", job_id), ("process_id", fields.get("process_id"))]:
            prefix = "AJ_" if key == "startup_job_id" else "PCEXEC_"
            if isinstance(value, str) and re.fullmatch(prefix + r"[A-Za-z0-9]{1,64}", value):
                record[key] = value
        if isinstance(fields.get("child_pid"), int):
            record["child_pid"] = fields["child_pid"]
        try:
            # JSON in the message survives consumers that overwrite job_id extras.
            logger.info("agent_startup_timing %s", json.dumps(record, separators=(",", ":")))
        except Exception:
            pass  # Observability must not change process/job lifecycle outcomes.
