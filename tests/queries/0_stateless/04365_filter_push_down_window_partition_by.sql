-- Filter push down below WindowStep on PARTITION BY columns (issue #110109).
-- A predicate referencing only the window PARTITION BY columns is safe to apply before
-- the window, so it must reach storage and enable primary key / granule pruning.

SET enable_analyzer = 1;
SET query_plan_filter_push_down = 1;

DROP TABLE IF EXISTS t_04365;
CREATE TABLE t_04365 (key String, ts DateTime, val UInt64)
ENGINE = MergeTree ORDER BY (key, ts)
AS SELECT toString(number % 100) AS key, toDateTime(number) AS ts, number AS val
FROM numbers(100000);
OPTIMIZE TABLE t_04365 FINAL;

-- Helper output: `pruned` = 1 when the primary key pruned some granules
-- (read granules < total granules), robust to the absolute granule count which
-- depends on randomized index_granularity. `pruned` = 0 when the full table is scanned.

-- QUALIFY form: partition-key conjunct pushed below Window -> PK prunes granules.
SELECT extract(explain, 'Granules: (\d+)')::UInt64 < extract(explain, 'Granules: \d+/(\d+)')::UInt64 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
) WHERE explain ILIKE '%Granules:%';

-- Outer WHERE around a windowed subquery: same pruning.
SELECT extract(explain, 'Granules: (\d+)')::UInt64 < extract(explain, 'Granules: \d+/(\d+)')::UInt64 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn FROM t_04365
    ) WHERE key = '5'
) WHERE explain ILIKE '%Granules:%';

-- Multiple windows sharing the partition key: the common conjunct still reaches storage.
SELECT extract(explain, 'Granules: (\d+)')::UInt64 < extract(explain, 'Granules: \d+/(\d+)')::UInt64 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT key, ts,
        row_number() OVER (PARTITION BY key ORDER BY ts) AS rn,
        sum(val)     OVER (PARTITION BY key ORDER BY ts) AS s
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
) WHERE explain ILIKE '%Granules:%';

-- Mixed predicate: the partition-key conjunct pushes (granules pruned), the window-result
-- conjunct stays above.
SELECT extract(explain, 'Granules: (\d+)')::UInt64 < extract(explain, 'Granules: \d+/(\d+)')::UInt64 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5' AND ts > toDateTime(10)
) WHERE explain ILIKE '%Granules:%';

-- Negative: a predicate on the window result only must NOT be pushed (no pruning).
SELECT extract(explain, 'Granules: (\d+)')::UInt64 < extract(explain, 'Granules: \d+/(\d+)')::UInt64 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1
) WHERE explain ILIKE '%Granules:%';

-- No PARTITION BY: nothing to push, must not crash and must not prune.
SELECT extract(explain, 'Granules: (\d+)')::UInt64 < extract(explain, 'Granules: \d+/(\d+)')::UInt64 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
) WHERE explain ILIKE '%Granules:%';

-- Correctness is preserved: the optimized query returns the same rows as forcing the
-- filter before the window explicitly.
SELECT count(), sum(val) FROM (
    SELECT key, ts, val, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
);
SELECT count(), sum(val) FROM (
    SELECT key, ts, val, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM (SELECT * FROM t_04365 WHERE key = '5') QUALIFY rn = 1
);

DROP TABLE t_04365;
