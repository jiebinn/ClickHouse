#!/usr/bin/env bash
# Tags: no-parallel, no-fasttest, no-object-storage, no-random-settings
# no-parallel: the failpoint below is server-global.
# no-object-storage: the test builds its own cache-over-s3 disk.
# no-random-settings: randomized buffer sizes change how much data spans file segments.

# Regression for a LOGICAL_ERROR (server abort in debug/sanitizer builds) when the
# file_cache_dynamic_resize_fail_to_evict failpoint fired during ordinary space
# reservation. That failpoint models an eviction failure and is only meaningful for the
# dynamic cache resize feature, so it must not fire on the reserve path. See issue #88945.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

set -e

CACHE_NAME="03908_cache_evict_${CLICKHOUSE_DATABASE}"
QID="03908_${CLICKHOUSE_DATABASE}"

$CLICKHOUSE_CLIENT -q "DROP TABLE IF EXISTS t_03908"

# A per-database unique cache name/path means the cache starts empty. Write-through
# caching (cache_on_write_operations) makes each INSERT populate the cache directly, so
# the second INSERT deterministically has to evict to fit under the small max_size --
# without it, cache population would depend on non-deterministic background read-caching.
$CLICKHOUSE_CLIENT -q "
    CREATE TABLE t_03908 (c0 Int)
    ENGINE = MergeTree()
    ORDER BY tuple()
    SETTINGS min_bytes_for_wide_part = 0,
             disk = disk(
                type = cache,
                name = '$CACHE_NAME',
                max_size = '25Ki',
                path = '$CACHE_NAME/',
                cache_on_write_operations = 1,
                disk = 's3_disk')"

$CLICKHOUSE_CLIENT --enable_filesystem_cache_on_write_operations=1 -q "INSERT INTO t_03908 SELECT number FROM numbers(1399)"

$CLICKHOUSE_CLIENT -q "SYSTEM ENABLE FAILPOINT file_cache_dynamic_resize_fail_to_evict"
# The failpoint is server-global, so disable it on every exit path (set -e). Otherwise a
# failure below would leave it armed for later stateless tests on the same server.
trap '$CLICKHOUSE_CLIENT -q "SYSTEM DISABLE FAILPOINT file_cache_dynamic_resize_fail_to_evict"' EXIT

# This second insert forces cache eviction on the reserve path. Before the fix the
# failpoint fired here and aborted the server with a LOGICAL_ERROR. Now the failpoint
# is confined to dynamic resize, so eviction proceeds normally and the insert succeeds.
$CLICKHOUSE_CLIENT --query_id "$QID" --enable_filesystem_cache_on_write_operations=1 -q "INSERT INTO t_03908 SELECT number FROM numbers(1770)"

$CLICKHOUSE_CLIENT -q "SYSTEM FLUSH LOGS query_log"

# The eviction really happened on the reserve path (not merely "the insert succeeded"):
# the second INSERT evicted at least one file segment.
$CLICKHOUSE_CLIENT -q "
    SELECT 'evicted on reserve path: ' || toString(ProfileEvents['FilesystemCacheEvictedFileSegments'] > 0)
    FROM system.query_log
    WHERE query_id = '$QID' AND current_database = currentDatabase() AND type = 'QueryFinish'
    ORDER BY event_time_microseconds DESC LIMIT 1"

# The server is still alive and the data is queryable.
$CLICKHOUSE_CLIENT -q "SELECT count() FROM t_03908"

$CLICKHOUSE_CLIENT -q "DROP TABLE t_03908"
