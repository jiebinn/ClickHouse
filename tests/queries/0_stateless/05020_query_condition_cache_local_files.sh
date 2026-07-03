#!/usr/bin/env bash
# Tags: no-fasttest
# Tag no-fasttest: needs Parquet

# Tests that the Query Condition Cache works for local Parquet files read via the
# `file()` table function / `File` engine (not only for object storage). A first
# query populates the cache with the row groups that contain no matching rows; a
# second query with the same predicate then skips those row groups (or the whole
# file). The assertions are isolated per query_id, so the test is parallel-safe
# despite the cache being global.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

file_name="${CLICKHOUSE_TEST_UNIQUE_NAME}.parquet"
file_path="${USER_FILES_PATH}/${file_name}"
rm -f "$file_path"

# 1M rows, 10 row groups of 100k rows each. Values are all even (number * 2), so an
# odd predicate value never matches, yet it falls inside the first row group's
# [min, max] range - single-query min/max pruning cannot skip that row group, but
# the Query Condition Cache can once it learns the row group has no matching rows.
${CLICKHOUSE_CLIENT} --query "
    INSERT INTO FUNCTION file('${file_name}', 'Parquet')
    SELECT number * 2 AS b FROM numbers(1000000)
    SETTINGS output_format_parquet_row_group_size = 100000, engine_file_truncate_on_insert = 1
"

${CLICKHOUSE_CLIENT} --query "SYSTEM DROP QUERY CONDITION CACHE"

qid_populate="${CLICKHOUSE_TEST_UNIQUE_NAME}_populate"
qid_cached="${CLICKHOUSE_TEST_UNIQUE_NAME}_cached"
qid_nocache="${CLICKHOUSE_TEST_UNIQUE_NAME}_nocache"

echo "first query (populates the cache), result:"
${CLICKHOUSE_CLIENT} --query_id="$qid_populate" --query "
    SELECT count() FROM file('${file_name}', 'Parquet') WHERE b = 3 SETTINGS use_query_condition_cache = 1
"

echo "second query (uses the cache), result:"
${CLICKHOUSE_CLIENT} --query_id="$qid_cached" --query "
    SELECT count() FROM file('${file_name}', 'Parquet') WHERE b = 3 SETTINGS use_query_condition_cache = 1
"

echo "second query without the cache, result:"
${CLICKHOUSE_CLIENT} --query_id="$qid_nocache" --query "
    SELECT count() FROM file('${file_name}', 'Parquet') WHERE b = 3 SETTINGS use_query_condition_cache = 0
"

${CLICKHOUSE_CLIENT} --query "SYSTEM FLUSH LOGS"

echo "rows read by the second query with the cache (expect 0 - whole file skipped):"
${CLICKHOUSE_CLIENT} --query "
    SELECT read_rows FROM system.query_log
    WHERE query_id = '$qid_cached' AND current_database = currentDatabase() AND type = 'QueryFinish'
    ORDER BY event_time_microseconds DESC LIMIT 1
"

echo "second query without the cache read more than zero rows (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT read_rows > 0 FROM system.query_log
    WHERE query_id = '$qid_nocache' AND current_database = currentDatabase() AND type = 'QueryFinish'
    ORDER BY event_time_microseconds DESC LIMIT 1
"

# Correctness: a predicate that actually matches must still return the right rows
# after the cache is populated (no over-skipping of matching row groups).
${CLICKHOUSE_CLIENT} --query "SYSTEM DROP QUERY CONDITION CACHE"
echo "matching predicate, run 1 (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT count() FROM file('${file_name}', 'Parquet') WHERE b = 400000 SETTINGS use_query_condition_cache = 1
"
echo "matching predicate, run 2 (expect 1):"
${CLICKHOUSE_CLIENT} --query "
    SELECT count() FROM file('${file_name}', 'Parquet') WHERE b = 400000 SETTINGS use_query_condition_cache = 1
"

rm -f "$file_path"
