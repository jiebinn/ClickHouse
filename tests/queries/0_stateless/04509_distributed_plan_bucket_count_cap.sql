-- Tags: no-fasttest
-- no-fasttest: make_distributed_plan is an experimental feature

-- An out-of-range distributed-plan bucket count must be rejected with INVALID_SETTING_VALUE, not size a
-- vector to the raw value and abort (std::length_error) while building the plan. The reader path
-- (tryMakeDistributedRead) is validated in optimizeTreeSecondPass, the shuffle-join path in makeDistributedPlan.

DROP TABLE IF EXISTS t_bucket_cap;
CREATE TABLE t_bucket_cap (x UInt64) ENGINE = MergeTree ORDER BY tuple();
INSERT INTO t_bucket_cap SELECT number FROM numbers(200000);

-- Reader path.
SELECT sumOrNull(x) FROM t_bucket_cap
SETTINGS make_distributed_plan = 1, distributed_plan_execute_locally = 1, distributed_plan_max_rows_to_broadcast = 0,
         distributed_plan_default_reader_bucket_count = 9223372036854775807; -- { serverError INVALID_SETTING_VALUE }

-- Shuffle-join path.
SELECT count() FROM t_bucket_cap AS a, t_bucket_cap AS b WHERE a.x = b.x
SETTINGS make_distributed_plan = 1, distributed_plan_execute_locally = 1, enable_parallel_replicas = 0,
         query_plan_use_new_logical_join_step = 1,
         distributed_plan_default_shuffle_join_bucket_count = 9223372036854775807; -- { serverError INVALID_SETTING_VALUE }

-- A bucket count at the cap is accepted: the distributed plan builds (setupDistributedReadBuckets runs).
SELECT count() > 0 FROM (
    EXPLAIN distributed = 1
    SELECT sumOrNull(x) FROM t_bucket_cap
    SETTINGS make_distributed_plan = 1, distributed_plan_max_rows_to_broadcast = 0,
             distributed_plan_default_reader_bucket_count = 256);

DROP TABLE t_bucket_cap;
