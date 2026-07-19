#!/usr/bin/env bash

# Regression test: `max_streams_per_hierarchical_merge = 0` must disable the
# hierarchical merge on every `SortingStep` path, leaving a single flat
# `MergingSortedTransform N -> 1`. With the default value the hierarchy (a layer
# of parallel `MergingSortedTransform × M` mergers) must appear once there are
# more input streams than the threshold.
#
# The read-in-order case additionally guards the `FinishSorting` `SortingStep`
# settings propagation: with `query_plan_read_in_order = 0` and the old analyzer,
# the plan is built by `InterpreterSelectQuery::executeOrderOptimized`, whose
# `SortingStep` previously ignored `max_streams_per_hierarchical_merge` and left
# it at the hardcoded default of 16.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

# `grep -c "MergingSortedTransform ×"` is 1 only when a layer of parallel
# mergers exists, i.e. when the hierarchical merge was built.
has_hierarchy() { grep -c "MergingSortedTransform ×"; }

# `max_threads` is pinned so the CI randomizer cannot set it to 1, which would
# remove multi-stream merging entirely and make even the enabled case flat.
# `max_threads_min_free_memory_per_thread = 0` disables the free-memory limiter in
# `getMaxThreadsForAvailableMemory`: with the default `1_GiB` a memory-limited runner
# would silently lower `max_threads` below the threshold, so the hierarchy (which needs
# more streams than the threshold) would not appear and the enabled checks would flip to 0.
# `02343_aggregation_pipeline.sql` disables the same limiter for stable `EXPLAIN PIPELINE`.

echo '-- full sort --'
# disabled: single flat merge (0), default 16: hierarchy over 32 streams (1)
# `max_rows_to_read = 0` overrides the stateless test profile default (20M), which
# `EXPLAIN PIPELINE` would otherwise trip while building the `numbers_mt(100000000)` pipe.
$CLICKHOUSE_CLIENT -q "EXPLAIN PIPELINE SELECT number FROM numbers_mt(100000000) ORDER BY number
    SETTINGS max_threads = 32, max_threads_min_free_memory_per_thread = 0, max_streams_per_hierarchical_merge = 0, max_rows_to_read = 0" | has_hierarchy
$CLICKHOUSE_CLIENT -q "EXPLAIN PIPELINE SELECT number FROM numbers_mt(100000000) ORDER BY number
    SETTINGS max_threads = 32, max_threads_min_free_memory_per_thread = 0, max_streams_per_hierarchical_merge = 16, max_rows_to_read = 0" | has_hierarchy

echo '-- read in order (FinishSorting) --'
# Everything runs in as few client invocations as possible: the flaky check runs
# the test many times and per-invocation client startup dominates under
# sanitizers (the test used to time out in the TSan flaky check).
$CLICKHOUSE_CLIENT -q "
    DROP TABLE IF EXISTS t_rio;
    CREATE TABLE t_rio (a UInt64) ENGINE = MergeTree ORDER BY a SETTINGS index_granularity = 8192;
    -- Stop merges so the number of parts (and therefore read streams) is stable.
    SYSTEM STOP MERGES t_rio;
"
# 20 small parts: more parts than the default threshold of 16, and small enough
# that the correctness check below stays fast under sanitizers.
INSERTS=""
for _ in $(seq 1 20); do INSERTS="$INSERTS INSERT INTO t_rio SELECT number FROM numbers(10000);"; done
$CLICKHOUSE_CLIENT -q "$INSERTS"

RIO_SETTINGS="enable_analyzer = 0, optimize_read_in_order = 1, query_plan_read_in_order = 0,
    read_in_order_two_level_merge_threshold = 1, max_threads = 32, max_threads_min_free_memory_per_thread = 0"

# disabled: single flat merge (0), default 16: hierarchy over 20 parts (1)
$CLICKHOUSE_CLIENT -q "EXPLAIN PIPELINE SELECT a FROM t_rio ORDER BY a
    SETTINGS $RIO_SETTINGS, max_streams_per_hierarchical_merge = 0" | has_hierarchy
$CLICKHOUSE_CLIENT -q "EXPLAIN PIPELINE SELECT a FROM t_rio ORDER BY a
    SETTINGS $RIO_SETTINGS, max_streams_per_hierarchical_merge = 16" | has_hierarchy

echo '-- results identical --'
# The opt-out must not change results: disabled vs enabled produce the same order.
DISABLED=$($CLICKHOUSE_CLIENT -q "SELECT groupArray(a) FROM (SELECT a FROM t_rio ORDER BY a
    SETTINGS optimize_read_in_order = 1, max_streams_per_hierarchical_merge = 0)")
ENABLED=$($CLICKHOUSE_CLIENT -q "SELECT groupArray(a) FROM (SELECT a FROM t_rio ORDER BY a
    SETTINGS optimize_read_in_order = 1, max_streams_per_hierarchical_merge = 16)")
[ "$DISABLED" = "$ENABLED" ] && echo 1 || echo 0

$CLICKHOUSE_CLIENT -q "DROP TABLE t_rio"
