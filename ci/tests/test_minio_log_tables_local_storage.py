"""
Guard test for the storage placement of the MinIO webhook log tables created in
ClickHouseProc.create_minio_log_tables (ci/jobs/scripts/clickhouse_proc.py).

Background
----------
`system.minio_audit_logs` and `system.minio_server_logs` are diagnostic tables
that capture the MinIO audit/server webhook stream during a run. They are plain
`ENGINE = MergeTree` tables, so unless pinned they inherit the server's default
merge_tree storage policy. On s3 storage runs that default is S3
(s3_storage_policy_for_merge_tree_by_default.xml sets
<merge_tree><storage_policy>s3</storage_policy>), and in the private/cloud repo
the `default` policy itself is cloud-based (object storage). Either way the
tables would live ON S3, which is doubly bad for tables that record S3 activity:

  * every audit-event insert writes parts to S3, which generates more audit
    events - a feedback loop that inflates the table, and
  * the post-run `select * ... into outfile` dump in dump_system_tables reads it
    all back from S3. On amd_tsan the JSON-typed audit table grew to ~700k rows /
    ~1.5 GB and the dump exceeded DUMP_SYSTEM_TABLE_TIMEOUT, turning the
    "Scraping system tables" step red (observed on master, e.g. commit 961ded3).

The fix pins each table to a dedicated LOCAL disk with an explicit
`SETTINGS disk = disk(type = 'local', path = '/var/lib/clickhouse/disks/...')`.
That path is inside custom_local_disks_base_directory (set by
custom_disks_base_path.xml, which install.sh always installs), so the disk is
created locally regardless of what `default` maps to - correct in both the
public repo (where `default` is local) and the private/cloud repo (where
`default` is object storage). This test guards against a future edit dropping
that pin, or reverting to `storage_policy = 'default'`, and silently putting the
tables back on S3.
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "ci" / "jobs" / "scripts" / "clickhouse_proc.py"


def _create_statements(src):
    # Every `CREATE TABLE system.minio_*_logs ...` SQL string built in the
    # source, up to the closing double quote of the clickhouse-client --query.
    return re.findall(r"CREATE TABLE system\.minio_\w+_logs[^\"]*", src)


def test_minio_log_tables_are_pinned_to_local_storage():
    src = _SRC.read_text()
    statements = _create_statements(src)
    # Both the audit and server log tables must be created.
    assert len(statements) == 2, (
        f"expected 2 minio log table CREATE statements, found {len(statements)}: {statements}"
    )
    for stmt in statements:
        # The captured text is a Python source fragment, so the SQL string
        # quotes are backslash-escaped (\\'local\\'); drop the backslashes to
        # compare against the SQL as clickhouse-client will see it.
        sql = stmt.replace("\\", "")
        # A CREATE that ends at `ORDER BY tuple()` with no explicit disk would
        # inherit the server default (S3 on s3 runs, cloud in private). Require an
        # explicit local disk right after the ORDER BY, pinned inside
        # custom_local_disks_base_directory so the placement is explicit, local,
        # and independent of what `default` resolves to.
        assert re.search(
            r"ORDER BY tuple\(\)\s+SETTINGS\s+disk = disk\(\s*type = 'local',\s*"
            r"path = '/var/lib/clickhouse/disks/[^']+'\s*\)",
            sql,
        ), (
            "minio log table must be pinned to an explicit local disk under "
            "/var/lib/clickhouse/disks/ so its dump is not read back from S3 "
            f"(and not left on the cloud `default` policy in private); got: {sql}"
        )
