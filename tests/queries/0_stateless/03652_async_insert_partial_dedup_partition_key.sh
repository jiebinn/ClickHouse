#!/usr/bin/env bash
# Tags: no-fasttest
# no-fasttest: needs async insert busy-timeout flushing to coalesce several tokens into one batch.

# Regression test for https://github.com/ClickHouse/ClickHouse/issues/110604
# Partial async-insert deduplication into a partitioned MergeTree with a dedup log
# used to abort with "Invalid partition key size: 0" (LOGICAL_ERROR): writeTempPart
# moves the partition value out of the block on the first write, and the dedup-conflict
# retry re-wrote the same block whose partition value was already emptied.

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

$CLICKHOUSE_CLIENT -q "DROP TABLE IF EXISTS minimal"
$CLICKHOUSE_CLIENT -q "
CREATE TABLE minimal (id String, event_time DateTime64(6, 'UTC'))
ENGINE = MergeTree PARTITION BY toYYYYMMDD(event_time) ORDER BY (event_time, id)
SETTINGS non_replicated_deduplication_window = 10000"

# Seed the dedup log with token m-X so that a later coalesced async batch containing
# m-X becomes a PARTIAL duplicate (only some tokens are duplicates).
$CLICKHOUSE_CLIENT --insert_deduplicate=1 --insert_deduplication_token='m-X' \
    -q "INSERT INTO minimal VALUES ('r1', '2026-07-15 10:00:00')"

# Fire several async inserts concurrently so the busy-timeout flush coalesces them into
# a single batch; one of them (m-X) is a duplicate -> partial dedup -> retry path.
for t in X Y Z W V; do
    $CLICKHOUSE_CLIENT --async_insert=1 --wait_for_async_insert=1 \
        --async_insert_busy_timeout_ms=1000 --async_insert_use_adaptive_busy_timeout=0 \
        --insert_deduplicate=1 --insert_deduplication_token="m-$t" \
        -q "INSERT INTO minimal VALUES ('m_$t', '2026-07-15 10:00:0${#t}')" &
done
wait

# Before the fix the server aborted; now the insert succeeds and m-X is deduplicated.
$CLICKHOUSE_CLIENT -q "SELECT count() FROM minimal"
$CLICKHOUSE_CLIENT -q "SELECT id FROM minimal ORDER BY id"

$CLICKHOUSE_CLIENT -q "DROP TABLE minimal"
