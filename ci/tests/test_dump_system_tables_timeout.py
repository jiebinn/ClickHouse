"""
Regression test for the per-table wall-clock cap in
ClickHouseProc.dump_system_tables (ci/jobs/scripts/clickhouse_proc.py).

Background
----------
After all functional tests pass, the job's "Collect logs" phase dumps ~12
system tables one by one with `clickhouse local ... select * from system.<t>
into outfile ...`. On amd_tsan + s3 runs the JSON-typed `system.minio_audit_logs`
table can grow huge and `clickhouse local` can hang reading it. Because the dump
command had NO per-command timeout, a single stuck table consumed the rest of
the 9000s job budget; the job watchdog then SIGKILLed everything and the job
finished with a job-level ERROR (exit code 125 / -15) and results:[] - no
individual test failed. Observed on PRs #107307, #103402, #103706 and on master.

The fix wraps each dump in `timeout --signal=TERM --kill-after=60 <N>` so one
stuck table is bounded and reported as a failed dump instead of killing the job.

These tests exercise the `timeout` wrapper mechanism (the same coreutil and
flags the fix builds into `dump_prefix`), proving both directions with a small
cap: a hanging command is killed within the cap and returns 124; a fast command
passes through untouched. They also assert the production code still builds the
timeout prefix from the class constant.
"""

import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "ci" / "jobs" / "scripts" / "clickhouse_proc.py"


def _prefix(timeout_s):
    # Mirror the exact prefix the fix builds in dump_system_tables.
    return f"timeout --signal=TERM --kill-after=60 {timeout_s} "


def test_hanging_dump_is_bounded_by_timeout():
    # A dump that would hang forever must be killed within the cap and report
    # timeout's exit code 124 - not run unbounded until the job watchdog fires.
    cap = 2
    start = time.monotonic()
    res = subprocess.run(
        _prefix(cap) + "sleep 30",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - start
    assert res.returncode == 124, (
        f"expected timeout exit 124, got {res.returncode}; the dump was not bounded"
    )
    assert elapsed < 15, (
        f"command took {elapsed:.1f}s (>= 15s); timeout did not enforce the {cap}s cap"
    )


def test_without_timeout_the_dump_runs_unbounded():
    # Demonstrate the pre-fix behavior: without the timeout prefix the same
    # hanging command runs to completion (here a short sleep stands in for a
    # dump that would hang for the rest of the 9000s budget).
    start = time.monotonic()
    res = subprocess.run(
        "sleep 3",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - start
    assert res.returncode == 0
    assert elapsed >= 3, (
        f"sleep returned after {elapsed:.1f}s; expected it to run unbounded (>= 3s)"
    )


def test_fast_dump_passes_through_untouched():
    # A dump that finishes well within the cap must succeed unchanged.
    res = subprocess.run(
        _prefix(60) + "echo ok",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "ok"


def test_production_code_bounds_each_dump_with_timeout():
    # Guard against a regression that drops the timeout wrapper: the source must
    # build the timeout prefix from the class constant and apply it to the dump.
    src = _SRC.read_text()
    assert "DUMP_SYSTEM_TABLE_TIMEOUT" in src
    assert "timeout --signal=TERM --kill-after=60" in src
    assert "{dump_prefix}clickhouse local" in src
    # 124 is the exit code coreutils `timeout` returns on expiry.
    assert "res == 124" in src
