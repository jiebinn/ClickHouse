#!/usr/bin/env bash
# Tags: no-fasttest, no-random-settings
# Tag no-fasttest: needs s3
# Tag no-random-settings: a randomized s3_max_single_part_upload_size would override the endpoint value.

# The URL-scoped endpoint block in <s3> (max_single_part_upload_size = 10000) must win over the
# top-level <s3> section, forcing the insert into a multipart upload. See s3_settings_override.xml.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

$CLICKHOUSE_CLIENT --query "DROP TABLE IF EXISTS t_04498_s3_engine_endpoint_override"

$CLICKHOUSE_CLIENT --query "
CREATE TABLE t_04498_s3_engine_endpoint_override (s String)
ENGINE = S3('http://localhost:11111/test/04498_s3_engine_endpoint_settings_override/data.native', 'Native')
"

$CLICKHOUSE_CLIENT --query "
INSERT INTO t_04498_s3_engine_endpoint_override SELECT randomString(100) FROM numbers(500)
SETTINGS s3_truncate_on_insert = 1
"

$CLICKHOUSE_CLIENT --query "SYSTEM FLUSH LOGS query_log"

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
