#!/usr/bin/env bash
# Tags: no-fasttest, no-parallel, no-random-settings
# Tag no-fasttest: needs s3
# Tag no-parallel: SYSTEM RELOAD CONFIG is global server state.
# Tag no-random-settings: SYSTEM RELOAD CONFIG combined with multipart S3 upload can exceed the
# flaky-check timeout in debug builds when random settings inflate per-step latency. The bug under
# test is about settings priority and is independent of these random settings.

# Regression test for the S3 settings-priority bug where, for S3/S3Queue *engine tables*, the
# generic top-level <s3> section wrongly overrode the more-specific URL-scoped endpoint block on
# the object-storage refresh path (S3ObjectStorage::applyNewSettings). Introduced by
# https://github.com/ClickHouse/ClickHouse/pull/100975, which reordered applyNewSettings to apply
# the config-prefix settings on top of the endpoint settings. That is correct for a disk (the
# disk's own config is more specific than the global <s3> section), but backwards for engine
# tables, whose config_prefix IS the global top-level <s3> — less specific than an endpoint block.
#
# Config (tests/config/config.d/s3_settings_override.xml):
#   - top-level <s3> max_single_part_upload_size = 33554432 (equal to the built-in default, present
#     only so it is "changed" and participates in priority resolution).
#   - <test_04498> endpoint block for this test's URL: max_single_part_upload_size = 10000 (small).
#
# The endpoint block (more specific) must win, forcing a multipart upload. With the bug the
# top-level default wins and the ~50 KB insert is uploaded as a single PutObject instead.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

$CLICKHOUSE_CLIENT --query "DROP TABLE IF EXISTS t_04498_s3_engine_endpoint_override"

# An S3 engine table at the endpoint that carries the small max_single_part_upload_size.
$CLICKHOUSE_CLIENT --query "
CREATE TABLE t_04498_s3_engine_endpoint_override (s String)
ENGINE = S3('http://localhost:11111/test/04498_s3_engine_endpoint_settings_override/data.native', 'Native')
"

# Force the object-storage settings refresh (S3ObjectStorage::applyNewSettings) — the code path
# that merges the endpoint block with the global <s3> section.
# In case of listen_try we can have 'Address already in use'.
$CLICKHOUSE_CLIENT --query "SYSTEM RELOAD CONFIG" |& grep -v -e 'Address already in use'

# randomString(100) is incompressible, so the payload stays well above the 10000-byte multipart
# threshold. A small row count keeps the test fast under the flaky check (debug build).
# s3_check_objects_after_upload is disabled because the size verification has been observed to be
# flaky against the local S3 mock; this test only needs the upload path taken, not integrity.
$CLICKHOUSE_CLIENT --query "
INSERT INTO t_04498_s3_engine_endpoint_override SELECT randomString(100) FROM numbers(500)
SETTINGS s3_check_objects_after_upload = 0, s3_truncate_on_insert = 1
"

$CLICKHOUSE_CLIENT --query "SYSTEM FLUSH LOGS query_log"

# With the endpoint block's max_single_part_upload_size = 10000 winning, the data is uploaded via
# multipart (CreateMultipartUpload + UploadPart + CompleteMultipartUpload). If the top-level default
# (32Mi) had incorrectly taken priority, it would be a single PutObject.
$CLICKHOUSE_CLIENT --query "
SELECT
    ProfileEvents['S3CreateMultipartUpload'] >= 1 AS has_multipart_create,
    ProfileEvents['S3UploadPart'] >= 1 AS has_upload_parts,
    ProfileEvents['S3CompleteMultipartUpload'] >= 1 AS has_multipart_complete
FROM system.query_log
WHERE event_date >= yesterday() AND event_time >= now() - 600
    AND type = 'QueryFinish'
    AND current_database = currentDatabase()
    AND query LIKE '%t_04498_s3_engine_endpoint_override%'
    AND query LIKE '%INSERT%'
    AND query NOT LIKE '%system.query_log%'
ORDER BY query_start_time DESC
LIMIT 1
"

$CLICKHOUSE_CLIENT --query "DROP TABLE t_04498_s3_engine_endpoint_override"
