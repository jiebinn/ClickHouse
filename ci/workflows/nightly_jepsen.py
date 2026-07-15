from praktika import Job, Workflow

from ci.defs.defs import (
    BASE_BRANCH,
    DOCKERS,
    SECRETS,
    ArtifactConfigs,
    ArtifactNames,
    BuildTypes,
    RunnerLabels,
)
from ci.defs.job_configs import JobConfigs

binary_build_job = Job.Config.get_job(
    JobConfigs.build_jobs, f"Build ({BuildTypes.AMD_BINARY})"
).set_provides(ArtifactNames.CH_AMD_BINARY, reset=True)

# TODO: add alert on workflow failure

# NOTE: event temporarily set to PULL_REQUEST to validate that the newly added
# server Jepsen job is wired into the workflow correctly on this PR head. Revert
# to Workflow.Event.SCHEDULE with `branches=[BASE_BRANCH]` and re-enable
# `cron_schedules` before merging.
workflow = Workflow.Config(
    name="NightlyJepsen",
    event=Workflow.Event.PULL_REQUEST,
    base_branches=[BASE_BRANCH],
    jobs=[
        binary_build_job,
        # Validation-only: run just the server Jepsen job for a faster loop
        # (keeper Jepsen already passes). Keeper is dropped from this workflow,
        # so override server's `requires` to not depend on it (job_configs keeps
        # the keeper dependency for the real nightly). Before merging: restore
        # `JobConfigs.jepsen_keeper` here and drop this `set_requires` override.
        JobConfigs.jepsen_server.set_requires("Build (amd_binary)", reset=True),
    ],
    artifacts=[
        *ArtifactConfigs.clickhouse_binaries,
    ],
    dockers=DOCKERS,
    secrets=SECRETS,
    enable_cache=True,
    enable_report=True,
    enable_cidb=True,
    # cron_schedules=["13 4 * * *"],  # temporarily disabled for PR validation; restore before merge
    pre_hooks=["python3 ./ci/jobs/scripts/workflow_hooks/store_data.py"],
)

WORKFLOWS = [
    workflow,
]
