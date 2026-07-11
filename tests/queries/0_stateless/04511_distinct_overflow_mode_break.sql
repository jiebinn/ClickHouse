-- https://github.com/ClickHouse/ClickHouse/issues/110045
-- distinct_overflow_mode = 'break' must return the partial result accumulated up to the
-- limit trip (including the chunk that crossed the limit) and stop reading the source.

SET max_threads = 1;
SET max_block_size = 10;
SET distinct_overflow_mode = 'break';
SET max_rows_in_distinct = 25;

SELECT '-- DistinctTransform: the chunk that crosses the limit is returned, not dropped';
SELECT count() FROM (SELECT DISTINCT number FROM numbers(1000));

SELECT '-- DistinctTransform: limit tripping on the very first chunk must not produce an empty result';
SELECT count() FROM (SELECT DISTINCT number FROM numbers(1000)) SETTINGS max_rows_in_distinct = 1;

SELECT '-- DistinctTransform: reading stops after the limit trips (hangs on infinite source otherwise)';
SELECT count() FROM (SELECT DISTINCT number FROM system.numbers);

SELECT '-- DistinctTransform: max_bytes_in_distinct also returns a partial non-empty result';
SELECT count() > 0, count() < 1000 FROM (SELECT DISTINCT number FROM numbers(1000)) SETTINGS max_rows_in_distinct = 0, max_bytes_in_distinct = 1000;

DROP TABLE IF EXISTS t_distinct_break;
CREATE TABLE t_distinct_break (a UInt64, b UInt64) ENGINE = MergeTree ORDER BY a SETTINGS index_granularity = 10;
INSERT INTO t_distinct_break SELECT number % 10, number FROM numbers(1000);

SET optimize_distinct_in_order = 1;

SELECT '-- DistinctSortedTransform (distinct over globally sorted stream): the chunk that crosses the limit is returned, not dropped';
-- query_plan_remove_redundant_sorting would drop the inner ORDER BY under count() and the plan would not use DistinctSortedTransform
SELECT count() FROM (SELECT DISTINCT a, b FROM (SELECT a, b FROM t_distinct_break ORDER BY a)) SETTINGS query_plan_remove_redundant_sorting = 0;

SELECT '-- DistinctSortedStreamTransform (in-order pre-distinct): partial result on limit trip';
SELECT count() FROM (SELECT DISTINCT a FROM t_distinct_break) SETTINGS max_rows_in_distinct = 3;

DROP TABLE t_distinct_break;
