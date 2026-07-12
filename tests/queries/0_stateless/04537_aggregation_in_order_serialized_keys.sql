-- Regression test for a quadratic blowup in aggregation in order that showed up as a
-- "Hung check failed, possible deadlock found" in the server-side AST fuzzer stress test.
--
-- When the data is sorted by a prefix of the GROUP BY keys (here `s`, while grouping by `s, n`),
-- `AggregatingInOrderTransform` aggregates each run of equal `s` with a hash table over the full
-- key. A multi-column numbers/strings key selects the `prealloc_serialized` aggregation method,
-- whose state serializes the whole input block up front on construction. Because a fresh state is
-- built for every run of equal `s` (one per row here), a single block became
-- O(number_of_runs * block_size), i.e. quadratic, and did not respect `max_execution_time` because
-- one `consume` call processes a whole block without yielding.
--
-- The fix makes the per-run in-order path use the plain `serialized` method (lazy, per-row key
-- serialization), restoring linear time. After the fix the query below returns instantly; before
-- it, it exceeds `max_execution_time` and throws.

DROP TABLE IF EXISTS t_agg_in_order_serialized;
CREATE TABLE t_agg_in_order_serialized (s String, n UInt64) ENGINE = MergeTree ORDER BY s;

INSERT INTO t_agg_in_order_serialized SELECT toString(number), number FROM numbers(300000);

SELECT count() FROM (SELECT s, n FROM t_agg_in_order_serialized GROUP BY s, n)
SETTINGS optimize_aggregation_in_order = 1, optimize_read_in_order = 1,
         max_threads = 1, max_block_size = 16384, max_execution_time = 20;

DROP TABLE t_agg_in_order_serialized;

-- Multi-stream variant (`max_threads > 1`). Reading several parts in order produces more than one
-- stream, so the pipeline becomes:
--     AggregatingInOrderTransform -> FinishAggregatingInOrderTransform -> MergingAggregatedBucketTransform
-- The `serialized` fallback must be scoped to the per-run `AggregatingInOrderTransform` path only;
-- the whole-block `MergingAggregatedBucketTransform` merge (via `Aggregator::mergeBlocks`) keeps the
-- `prealloc_serialized` method, where it is a win. This test exercises that merge path and checks
-- that the multi-stream result is identical to regular aggregation and stays linear in time.
-- (The per-run path is still quadratic before the fix here too, so the `max_execution_time` guard
-- also catches a regression in this shape.)

DROP TABLE IF EXISTS t_agg_in_order_serialized_mt;
CREATE TABLE t_agg_in_order_serialized_mt (s String, n UInt64) ENGINE = MergeTree ORDER BY s;
SYSTEM STOP MERGES t_agg_in_order_serialized_mt;

INSERT INTO t_agg_in_order_serialized_mt SELECT toString(number), number FROM numbers(0, 100000);
INSERT INTO t_agg_in_order_serialized_mt SELECT toString(number), number FROM numbers(100000, 100000);
INSERT INTO t_agg_in_order_serialized_mt SELECT toString(number), number FROM numbers(200000, 100000);

SELECT count() FROM (SELECT s, n FROM t_agg_in_order_serialized_mt GROUP BY s, n)
SETTINGS optimize_aggregation_in_order = 1, optimize_read_in_order = 1,
         max_threads = 4, max_block_size = 16384, max_execution_time = 60;

-- The multi-stream in-order result must be byte-for-byte identical to regular aggregation.
SELECT
(
    SELECT sum(cityHash64(s, n, c)) FROM
    (
        SELECT s, n, count() AS c FROM t_agg_in_order_serialized_mt GROUP BY s, n
        SETTINGS optimize_aggregation_in_order = 1, optimize_read_in_order = 1,
                 max_threads = 4, max_block_size = 16384
    )
)
=
(
    SELECT sum(cityHash64(s, n, c)) FROM
    (
        SELECT s, n, count() AS c FROM t_agg_in_order_serialized_mt GROUP BY s, n
        SETTINGS optimize_aggregation_in_order = 0, optimize_read_in_order = 0
    )
);

DROP TABLE t_agg_in_order_serialized_mt;
