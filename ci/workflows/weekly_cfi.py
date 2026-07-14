from praktika import Workflow

from ci.defs.defs import BASE_BRANCH, DOCKERS, SECRETS, ArtifactConfigs
from ci.defs.job_configs import JobConfigs

# Weekly Control-Flow Integrity (CFI) check.
# Builds ClickHouse with Clang CFI (cfi-vcall, cfi-derived-cast) on top of a release
# build (ThinLTO + -fwhole-program-vtables already enabled) and runs integration and
# stress tests under the CFI binary. A CFI violation traps (SIGILL) and aborts the
# server, which surfaces as a job failure; -DSPLIT_DEBUG_SYMBOLS=ON keeps the resulting
# core-dump stack trace pointing at the offending virtual call or bad cast.
#
# The stateless functional suite is intentionally excluded: its exact-output assertions
# produce build-profile false positives (e.g. query-profiler symbolization) and pick up
# master-flaky tests, none of which are CFI violations, so they would keep the job red
# for non-CFI reasons. Stress + integration exercise the full server under CFI, so a
# real violation still surfaces as a crash. Adding stateless later would need a curated
# no-cfi skip set, the way asan/tsan/msan carry per-build skips.
#
# Runs every Monday at 03:00 UTC.

# NOTE: event temporarily set to PULL_REQUEST to re-validate the CFI job on this fork
# PR head after merging latest master. Revert to Workflow.Event.SCHEDULE with
# `branches=[BASE_BRANCH]` and re-enable `cron_schedules` before merging.
workflow = Workflow.Config(
    name="WeeklyCFI",
    event=Workflow.Event.PULL_REQUEST,
    base_branches=[BASE_BRANCH],
    jobs=[
        *JobConfigs.cfi_build_job,
        *JobConfigs.cfi_integration_jobs,
        *JobConfigs.cfi_stress_job,
    ],
    artifacts=[
        *ArtifactConfigs.clickhouse_binaries,
        *ArtifactConfigs.clickhouse_debians,
    ],
    dockers=DOCKERS,
    secrets=SECRETS,
    enable_cache=True,
    enable_report=True,
    enable_cidb=True,
    pre_hooks=["python3 ./ci/jobs/scripts/workflow_hooks/store_data.py"],
)

WORKFLOWS = [
    workflow,
]
