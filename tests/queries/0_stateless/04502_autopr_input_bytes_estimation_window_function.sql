-- Verify that a query plan containing a Window step is considered "simple enough" for the automatic
-- parallel replicas optimization, so that runtime dataflow statistics are collected for it.
-- Before the Window step supported dataflow statistics collection, any plan containing a window
-- function was rejected outright (`optimizeTree: Some steps in the plan don't support dataflow
-- statistics collection ... Unsupported steps: Window_...`) and no statistics were gathered.

DROP TABLE IF EXISTS t;

CREATE TABLE t(key UInt64, value UInt64) ENGINE = MergeTree ORDER BY key;

SET enable_parallel_replicas=1, automatic_parallel_replicas_mode=2, parallel_replicas_local_plan=1, parallel_replicas_index_analysis_only_on_coordinator=1,
    parallel_replicas_for_non_replicated_merge_tree=1, max_parallel_replicas=3, cluster_for_parallel_replicas='test_cluster_one_shard_three_replicas_localhost';

SET enable_analyzer=1;
SET max_threads=4;
SET max_bytes_before_external_group_by=0, max_bytes_ratio_before_external_group_by=0;
SET automatic_parallel_replicas_min_bytes_per_replica=0;

INSERT INTO t SELECT number, number * 2 FROM numbers(1e6);

-- The window function forces a Window step into the plan. With automatic_parallel_replicas_mode=2
-- statistics collection is enforced, so the reading step is instrumented and input bytes are recorded.
-- The window itself is computed on the coordinator, but the plan must still be recognized as simple enough.
SELECT key, sum(value) OVER (PARTITION BY key % 10 ORDER BY key) AS s
FROM t
FORMAT Null SETTINGS log_comment='04502_autopr_window_function_query';

-- Regression guard for the output side. The window is computed on the coordinator, so replicas only
-- ship the pre-window columns (recorded as input bytes) and the columns the window appends are never
-- sent to the initiator. The Window step must therefore NOT be instrumented as the replica-output
-- boundary. This query builds a very wide window result (`groupArray` over a 50-row frame, so each
-- output row carries an array of up to 51 values), which is far larger than the read `key`/`value`
-- columns. RuntimeDataflowStatisticsOutputBytes must stay bounded by the input bytes (in practice it is
-- zero, since the top of the replicas plan is the reading step): if the window result were mistakenly
-- counted as replica output, the recorded output bytes would balloon past the input bytes and this
-- check would fail.
SELECT key, groupArray(value) OVER (PARTITION BY key % 10 ORDER BY key ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) AS a
FROM t
FORMAT Null SETTINGS log_comment='04502_autopr_window_wide_output';

SET enable_parallel_replicas=0, automatic_parallel_replicas_mode=0;

SYSTEM FLUSH LOGS query_log;

SELECT log_comment, ProfileEvents['RuntimeDataflowStatisticsInputBytes'] > 0 AS stats_collected
FROM system.query_log
WHERE (event_date >= yesterday()) AND (event_time >= (NOW() - toIntervalMinute(15))) AND (current_database = currentDatabase()) AND (log_comment = '04502_autopr_window_function_query') AND (type = 'QueryFinish')
ORDER BY log_comment
FORMAT TSVWithNames;

SELECT
    maxIf(ProfileEvents['RuntimeDataflowStatisticsInputBytes'], log_comment = '04502_autopr_window_wide_output') > 0 AS stats_collected,
    maxIf(ProfileEvents['RuntimeDataflowStatisticsOutputBytes'], log_comment = '04502_autopr_window_wide_output')
        <= maxIf(ProfileEvents['RuntimeDataflowStatisticsInputBytes'], log_comment = '04502_autopr_window_wide_output') AS window_result_not_counted_as_output
FROM system.query_log
WHERE (event_date >= yesterday()) AND (event_time >= (NOW() - toIntervalMinute(15))) AND (current_database = currentDatabase()) AND (log_comment = '04502_autopr_window_wide_output') AND (type = 'QueryFinish')
FORMAT TSVWithNames;

DROP TABLE t;
