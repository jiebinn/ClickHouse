-- Tags: long, no-parallel, no-parallel-replicas
-- Tag no-parallel: Messes with internal cache
-- Tag long: needs ~1M rows for the QCC to populate.
--
-- TopK WHERE reads also consult the `topk_reuse_predicate_only_hash` in addition to the
-- TopK-salted hash, so a plain `SELECT ... WHERE <predicate>` run can prime entries that a
-- later `SELECT ... WHERE <predicate> ORDER BY ... LIMIT n` reuses. TopK-salted entries must
-- not be shared the other way around.

SET allow_experimental_analyzer = 1;
SET use_query_condition_cache = 1;
SET use_top_k_dynamic_filtering = 1;
SET use_skip_indexes_for_top_k = 1;
SET query_plan_max_limit_for_top_k_optimization = 1000;
SET optimize_move_to_prewhere = 0;
SET enable_parallel_replicas = 0;
SET automatic_parallel_replicas_mode = 0;
SET parallel_replicas_local_plan = 1;
SET max_threads = 1;

DROP TABLE IF EXISTS tab;

CREATE TABLE tab (id UInt32, v1 UInt32, v2 UInt32) ENGINE = MergeTree ORDER BY id
SETTINGS index_granularity = 64,
         min_bytes_for_wide_part = 0,
         min_bytes_for_full_part_storage = 0,
         add_minmax_index_for_numeric_columns = 0;

INSERT INTO tab SELECT rand(), number, number FROM numbers(1_000_000);

SELECT '--- QCC starts empty';
SYSTEM CLEAR QUERY CONDITION CACHE;
SELECT count() FROM system.query_condition_cache;

SELECT '--- Plain WHERE primes the predicate-only cache entry';
SELECT v1 FROM tab WHERE v2 = 10000 FORMAT Null;
SELECT count() FROM system.query_condition_cache;

SELECT '--- TopK reuses the predicate-only entry on read and may add its own TopK-salted entry';
SELECT v1 FROM tab WHERE v2 = 10000 ORDER BY v1 ASC LIMIT 5 FORMAT Null;
SELECT count() FROM system.query_condition_cache;

SELECT '--- TopK still returns the planted row';
SELECT v1 FROM tab WHERE v2 = 10000 ORDER BY v1 ASC LIMIT 5;

SELECT '--- Reverse: TopK-only entry must not poison plain WHERE';
SYSTEM CLEAR QUERY CONDITION CACHE;
SELECT v1 FROM tab WHERE v2 = 10000 ORDER BY v1 ASC LIMIT 5 FORMAT Null;
SELECT count() FROM system.query_condition_cache;
SELECT v1 FROM tab WHERE v2 = 10000;

DROP TABLE tab;
