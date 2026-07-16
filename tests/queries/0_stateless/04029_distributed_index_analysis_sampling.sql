-- Tags: no-random-settings, no-random-merge-tree-settings

-- Verify that the SAMPLE clause and a WHERE predicate are both passed to remote
-- replicas during distributed index analysis, so they can prune granules by the
-- AND-combination of both filters. DistributedIndexAnalyzer combines the WHERE
-- filter_ast and the sampling_filter via makeASTForLogicalAnd; without a WHERE the
-- else-if branch just forwards the sampling_filter.
SET explain_query_plan_default = 'legacy';

SET send_logs_level = 'error';

DROP TABLE IF EXISTS t_dia_sampling;

-- A minmax skip index on `value` makes the WHERE predicate participate in index
-- analysis, so the combined filter reaching the replicas is observable in the plan.
CREATE TABLE t_dia_sampling (key UInt64, value UInt64, INDEX idx_value value TYPE minmax GRANULARITY 1)
ENGINE = MergeTree ORDER BY intHash32(key) SAMPLE BY intHash32(key)
SETTINGS index_granularity = 256,
  min_bytes_for_wide_part = '1G',
  index_granularity_bytes = '10M',
  distributed_index_analysis_min_parts_to_activate = 0,
  distributed_index_analysis_min_indexes_bytes_to_activate = 0;

SYSTEM STOP MERGES t_dia_sampling;
INSERT INTO t_dia_sampling SELECT number, number FROM numbers(100000) SETTINGS max_block_size = 10000, min_insert_block_size_rows = 10000, max_insert_threads = 1;
INSERT INTO t_dia_sampling SELECT number + 100000, number FROM numbers(100000) SETTINGS max_block_size = 10000, min_insert_block_size_rows = 10000, max_insert_threads = 1;

SET cluster_for_parallel_replicas = 'test_cluster_one_shard_two_replicas';
SET max_parallel_replicas = 2;
SET use_query_condition_cache = 0;
SET allow_experimental_parallel_reading_from_replicas = 0;
SET allow_experimental_analyzer = 1;

-- Results must be identical with and without distributed index analysis.
SELECT count() FROM t_dia_sampling SAMPLE 0.1 SETTINGS distributed_index_analysis = 0;
SELECT count() FROM t_dia_sampling SAMPLE 0.1 SETTINGS distributed_index_analysis = 1;

-- SAMPLE + WHERE: exercises the AND-combination branch (filter_ast && sampling_filter).
SELECT count() FROM t_dia_sampling SAMPLE 0.1 WHERE value < 50000 SETTINGS distributed_index_analysis = 0;
SELECT count() FROM t_dia_sampling SAMPLE 0.1 WHERE value < 50000 SETTINGS distributed_index_analysis = 1;

-- The WHERE predicate must reach index analysis under distributed_index_analysis = 1:
-- the minmax skip index on `value` must appear in the plan. If the combined filter
-- dropped the WHERE part, the skip index would not be used at all.
SELECT countIf(explain ILIKE '%idx_value%') > 0 AS skip_index_used
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM t_dia_sampling SAMPLE 0.1 WHERE value < 50000
    SETTINGS distributed_index_analysis = 1
);

-- And the skip index must prune ranges: SAMPLE + WHERE must read fewer ranges than
-- SAMPLE alone. The Ranges count is the number of granule ranges left after all
-- index conditions (primary key + skip index) are applied.
SELECT
(
    SELECT max(toUInt64OrZero(extract(explain, 'Ranges: (\d+)'))) FROM (
        EXPLAIN indexes = 1 SELECT * FROM t_dia_sampling SAMPLE 0.1 WHERE value < 50000
        SETTINGS distributed_index_analysis = 1)
) <
(
    SELECT max(toUInt64OrZero(extract(explain, 'Ranges: (\d+)'))) FROM (
        EXPLAIN indexes = 1 SELECT * FROM t_dia_sampling SAMPLE 0.1
        SETTINGS distributed_index_analysis = 1)
) AS where_prunes_ranges;

DROP TABLE t_dia_sampling;
