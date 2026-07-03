#!/usr/bin/env bash
# Tags: no-fasttest
# - no-fasttest: requires `IcebergLocal` (USE_AVRO build option)
#
# Regression test: manifests written by ClickHouse declared timestamp partition fields as a
# bare Avro `long`, without the `timestamp-micros` logical type that Iceberg requires. On
# read such a value parses into an `Int64` field instead of a `DateTime64` one, so partition
# pruning compared it against a `Decimal64` predicate constant that never matches: an
# equality predicate on the partition column silently returned no rows, and a
# monotonic-function predicate (e.g. `toString`) threw `Bad get: has Int64, requested
# Decimal64` from the Iceberg iterator.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

TABLE="t_${CLICKHOUSE_DATABASE}_${RANDOM}_dt64_pruning"
TABLE_PATH="${USER_FILES_PATH}/${TABLE}/"

trap 'rm -rf "${TABLE_PATH}" 2>/dev/null' EXIT

${CLICKHOUSE_CLIENT} --query "DROP TABLE IF EXISTS ${TABLE}"
${CLICKHOUSE_CLIENT} --query "
    CREATE TABLE ${TABLE} (t DateTime64(6, 'UTC'), v String)
    ENGINE = IcebergLocal('${TABLE_PATH}', 'Parquet')
    PARTITION BY (t)
"
${CLICKHOUSE_CLIENT} --allow_insert_into_iceberg=1 --query "
    INSERT INTO ${TABLE} VALUES ('2024-01-01 00:00:00.123456', 'a'), ('2024-06-15 12:30:00.654321', 'b')
"

echo "--- full read ---"
${CLICKHOUSE_CLIENT} --query "SELECT t, v FROM ${TABLE} ORDER BY t FORMAT TSV"

echo "--- equality on the partition column ---"
${CLICKHOUSE_CLIENT} --query "
    SELECT v FROM ${TABLE} WHERE t = toDateTime64('2024-06-15 12:30:00.654321', 6, 'UTC') FORMAT TSV"

echo "--- monotonic function of the partition column ---"
${CLICKHOUSE_CLIENT} --query "
    SELECT count() FROM ${TABLE} WHERE toString(t) = '2024-06-15 12:30:00.654321' FORMAT TSV"

${CLICKHOUSE_CLIENT} --query "DROP TABLE IF EXISTS ${TABLE}"
