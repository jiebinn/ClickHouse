from praktika import Workflow

from ci.defs.defs import BASE_BRANCH, DOCKERS, SECRETS, ArtifactConfigs
from ci.defs.job_configs import JobConfigs

# Weekly Control-Flow Integrity (CFI) check.
# Builds ClickHouse with Clang CFI (cfi-vcall, cfi-derived-cast) on top of a release
# build (ThinLTO + -fwhole-program-vtables already enabled) and runs stateless,
# integration, and stress tests. A CFI violation aborts the server with a diagnostic
# message, which surfaces as a test failure here.
#
# Runs every Monday at 03:00 UTC.

# NOTE: event is temporarily set to PULL_REQUEST to validate the CFI build+test path
# (-fvisibility=default, -fno-sanitize-trap=cfi) against this fork PR head. A schedule
# workflow's workflow_dispatch checks out only base-repo refs, so it cannot exercise a
# fork head; PULL_REQUEST wires CHECKOUT_REF to the PR head sha. Revert to
# Workflow.Event.SCHEDULE with `branches=[BASE_BRANCH]` and re-enable `cron_schedules`
# before merging.
workflow = Workflow.Config(
    name="WeeklyCFI",
    event=Workflow.Event.PULL_REQUEST,
    base_branches=[BASE_BRANCH],
    jobs=[
        *JobConfigs.cfi_build_job,
        *JobConfigs.cfi_stateless_jobs,
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
