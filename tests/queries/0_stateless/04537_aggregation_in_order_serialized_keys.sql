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
-- The fix makes aggregation in order use the plain `serialized` method (lazy, per-row key
-- serialization), restoring linear time. After the fix the query below returns instantly; before
-- it, it exceeds `max_execution_time` and throws.

DROP TABLE IF EXISTS t_agg_in_order_serialized;
CREATE TABLE t_agg_in_order_serialized (s String, n UInt64) ENGINE = MergeTree ORDER BY s;

INSERT INTO t_agg_in_order_serialized SELECT toString(number), number FROM numbers(300000);

SELECT count() FROM (SELECT s, n FROM t_agg_in_order_serialized GROUP BY s, n)
SETTINGS optimize_aggregation_in_order = 1, optimize_read_in_order = 1,
         max_threads = 1, max_block_size = 16384, max_execution_time = 20;

DROP TABLE t_agg_in_order_serialized;
