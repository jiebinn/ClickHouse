from praktika import Job, Workflow

from ci.defs.defs import (
    BASE_BRANCH,
    DOCKERS,
    SECRETS,
    ArtifactConfigs,
    ArtifactNames,
    BuildTypes,
)
from ci.defs.job_configs import JobConfigs

binary_build_job = Job.Config.get_job(
    JobConfigs.build_jobs, f"Build ({BuildTypes.AMD_BINARY})"
).set_provides(ArtifactNames.CH_AMD_BINARY, reset=True)

# TODO: add alert on workflow failure

# NOTE: event temporarily PULL_REQUEST for one final on-PR validation of the
# merge-target config (keeper + server serialized via run_after). Revert to
# Workflow.Event.SCHEDULE + branches=[BASE_BRANCH] before merging.
workflow = Workflow.Config(
    name="NightlyJepsen",
    event=Workflow.Event.PULL_REQUEST,
    base_branches=[BASE_BRANCH],
    jobs=[
        binary_build_job,
        JobConfigs.jepsen_keeper,
        # Serialize server Jepsen after keeper: both use the single shared
        # jepsen_group autoscaling group and must not run concurrently. This is
        # an ordering dependency only (not an artifact requirement), so express
        # it here with run_after rather than in the job's `requires`.
        JobConfigs.jepsen_server.set_run_after(JobConfigs.jepsen_keeper.name),
    ],
    artifacts=[
        *ArtifactConfigs.clickhouse_binaries,
    ],
    dockers=DOCKERS,
    secrets=SECRETS,
    enable_cache=True,
    enable_report=True,
    enable_cidb=True,
    # cron_schedules=["13 4 * * *"],  # temporarily disabled for final PR validation; restore before merge
    pre_hooks=["python3 ./ci/jobs/scripts/workflow_hooks/store_data.py"],
)

WORKFLOWS = [
    workflow,
]
