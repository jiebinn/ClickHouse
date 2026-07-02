-- Tags: no-old-analyzer
-- no-old-analyzer: Not supported

-- Bare `WHERE col` and `WHERE NOT col` on non-Nullable Int / UInt / Bool
-- columns are truthy tests and partition rows into defaults vs non-defaults.

DROP TABLE IF EXISTS t_truthy;

CREATE TABLE t_truthy
(
    id UInt64,
    u  UInt32,
    i  Int32,
    b  Bool,
    n  Nullable(UInt32)
)
ENGINE = MergeTree ORDER BY id
SETTINGS index_granularity = 100,
         ratio_of_defaults_for_sparse_serialization = 0.5,
         compute_exact_num_defaults_for_sparse_columns = 1,
         serialization_info_version = 'with_types',
         min_bytes_for_wide_part = 0;

SYSTEM STOP MERGES t_truthy;

-- 1000 rows / 10 granules. Non-defaults at rows 0, 200, 400, 600, 800: granules
-- 0, 2, 4, 6, 8 mixed; the other five are all-default.
INSERT INTO t_truthy
SELECT number,
       if(number % 200 = 0, 1, 0)::UInt32,
       if(number % 200 = 0, 1, 0)::Int32,
       (number % 200 = 0)::Bool,
       if(number % 200 = 0, 1, NULL)
FROM numbers(1000)
SETTINGS optimize_on_insert = 0;

-- Aggregate projections optimizer rejects non-`UInt8` filter columns unconditionally
-- (see `projectionsCommon.cpp`), so `SELECT count() WHERE u` is disabled here; the
-- pipeline path (non-aggregate reads and `Bool` counts) still runs.
SET optimize_use_projections = 0, optimize_use_implicit_projections = 0;

-- Result correctness: bare form matches the explicit form under both modes.
SELECT 'sum u off',        sum(id) FROM t_truthy WHERE u     SETTINGS use_sparsity_info_for_pruning = 'off';
SELECT 'sum u planning',   sum(id) FROM t_truthy WHERE u     SETTINGS use_sparsity_info_for_pruning = 'planning';
SELECT 'sum u data_read',  sum(id) FROM t_truthy WHERE u     SETTINGS use_sparsity_info_for_pruning = 'data_read';
SELECT 'sum i planning',   sum(id) FROM t_truthy WHERE i     SETTINGS use_sparsity_info_for_pruning = 'planning';
SELECT 'sum b planning',   count() FROM t_truthy WHERE b     SETTINGS use_sparsity_info_for_pruning = 'planning';
SELECT 'sum notu off',     sum(id) FROM t_truthy WHERE NOT u SETTINGS use_sparsity_info_for_pruning = 'off';
SELECT 'sum notu planning', sum(id) FROM t_truthy WHERE NOT u SETTINGS use_sparsity_info_for_pruning = 'planning';
SELECT 'sum notu data_read', sum(id) FROM t_truthy WHERE NOT u SETTINGS use_sparsity_info_for_pruning = 'data_read';

-- `WHERE u` drops 5 of 10 granules; `WHERE NOT u` drops none (no all-non-default granule).
SELECT 'positive', trimLeft(explain) FROM (
    EXPLAIN indexes = 1 SELECT id FROM t_truthy WHERE u
    SETTINGS use_sparsity_info_for_pruning = 'planning'
) WHERE trimLeft(explain) LIKE 'Sparsity%'
     OR trimLeft(explain) LIKE 'Granules: %/%';

SELECT 'negative', trimLeft(explain) FROM (
    EXPLAIN indexes = 1 SELECT id FROM t_truthy WHERE NOT u
    SETTINGS use_sparsity_info_for_pruning = 'planning'
) WHERE trimLeft(explain) LIKE 'Sparsity%'
     OR trimLeft(explain) LIKE 'Granules: %/%';

-- Nullable stays unclassified: no `Sparsity` step in the plan.
SELECT 'nullable', trimLeft(explain) FROM (
    EXPLAIN indexes = 1 SELECT id FROM t_truthy WHERE n
    SETTINGS use_sparsity_info_for_pruning = 'planning'
) WHERE trimLeft(explain) LIKE 'Sparsity%';

DROP TABLE t_truthy;
