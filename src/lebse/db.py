"""Thin psql runner for a CourtListener replica. Execution is not unit-tested (it needs a live DB);
the SQL it runs is built and tested in :mod:`lebse.sql`."""

from __future__ import annotations

import os
import subprocess

# Planner hints: use the large OS page cache and avoid JIT for these short analytical queries.
_SET = "SET effective_cache_size='200GB'; SET work_mem='512MB'; SET jit=off; "
SEP = "\x01"


def dsn() -> str:
    return os.environ.get("LEBSE_DSN", "dbname=courtlistener")


def rows(sql: str, timeout: int = 600) -> list[str]:
    """Run ``sql`` and return non-empty output lines (fields separated by ``SEP``)."""
    proc = subprocess.run(
        ["psql", dsn(), "-v", "ON_ERROR_STOP=1", "-tAF", SEP, "-c", _SET + sql],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:800])
    return [ln for ln in proc.stdout.split("\n") if ln.strip()]
