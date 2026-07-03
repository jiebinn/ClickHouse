#!/usr/bin/env bash
# Tags: no-fasttest
# Tag no-fasttest: needs Parquet

# Tests that the Query Condition Cache works for local Parquet files backed by the
# `File` table engine (previously the cache was populated only for object storage).
# The first query with a given predicate is a cache miss and records the row groups
# that contain no matching rows; a second query with the same predicate is a cache
# hit and reuses that information.
#
# We assert on the QueryConditionCacheMisses / QueryConditionCacheHits profile events,
# which is the reliable signal that the cache was populated and reused. `read_rows`
# alone does not distinguish here: Parquet bloom-filter pruning already skips the
# absent (odd) value on the very first query, so both queries read the same number of
# rows regardless of the cache.
#
# The table gets a unique UUID per test run, so the cache keys are isolated and the
# test is parallel-safe without a global cache reset.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

${CLICKHOUSE_CLIENT} --query "DROP TABLE IF EXISTS t_query_condition_cache"
${CLICKHOUSE_CLIENT} --query "CREATE TABLE t_query_condition_cache (b UInt64) ENGINE = File(Parquet)"

# All values are even, so the odd predicate value 3 never matches - the first query
# scans the surviving row groups, records them as non-matching, and the second query
# then skips them via the cache.
${CLICKHOUSE_CLIENT} --query "
    INSERT INTO t_query_condition_cache
    SELECT number * 2 FROM numbers(1000000)
    SETTINGS output_format_parquet_row_group_size = 100000
"

qid_first="${CLICKHOUSE_TEST_UNIQUE_NAME}_first"
qid_second="${CLICKHOUSE_TEST_UNIQUE_NAME}_second"

echo "first query result (expect 0):"
${CLICKHOUSE_CLIENT} --query_id="$qid_first" --query "
    SELECT count() FROM t_query_condition_cache WHERE b = 3 SETTINGS use_query_condition_cache = 1
"
echo "second query result (expect 0):"
${CLICKHOUSE_CLIENT} --query_id="$qid_second" --query "
    SELECT count() FROM t_query_condition_cache WHERE b = 3 SETTINGS use_query_condition_cache = 1
"

${CLICKHOUSE_CLIENT} --query "SYSTEM FLUSH LOGS query_log"

echo "first query was a cache miss (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT ProfileEvents['QueryConditionCacheMisses'] > 0
    FROM system.query_log
    WHERE query_id = '$qid_first' AND current_database = currentDatabase() AND type = 'QueryFinish'
    ORDER BY event_time_microseconds DESC LIMIT 1
"
echo "first query was not a cache hit (expect 0):"
${CLICKHOUSE_CLIENT} --query "
    SELECT ProfileEvents['QueryConditionCacheHits'] > 0
    FROM system.query_log
    WHERE query_id = '$qid_first' AND current_database = currentDatabase() AND type = 'QueryFinish'
    ORDER BY event_time_microseconds DESC LIMIT 1
"
echo "second query was a cache hit (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT ProfileEvents['QueryConditionCacheHits'] > 0
    FROM system.query_log
    WHERE query_id = '$qid_second' AND current_database = currentDatabase() AND type = 'QueryFinish'
    ORDER BY event_time_microseconds DESC LIMIT 1
"

# Correctness: a predicate that actually matches must still return the right rows
# once the cache is populated (no over-skipping of matching row groups).
echo "matching predicate, run 1 (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT count() FROM t_query_condition_cache WHERE b = 400000 SETTINGS use_query_condition_cache = 1
"
echo "matching predicate, run 2 (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT count() FROM t_query_condition_cache WHERE b = 400000 SETTINGS use_query_condition_cache = 1
"

${CLICKHOUSE_CLIENT} --query "DROP TABLE t_query_condition_cache"
