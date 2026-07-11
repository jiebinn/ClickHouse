-- Filter push down below WindowStep on PARTITION BY columns (issue #110109).
-- A predicate referencing only the window PARTITION BY columns is safe to apply before
-- the window, so it must reach storage as a primary key condition and enable pruning.

SET enable_analyzer = 1;
SET query_plan_filter_push_down = 1;

DROP TABLE IF EXISTS t_04365;
CREATE TABLE t_04365 (key String, ts DateTime, val UInt64)
ENGINE = MergeTree ORDER BY (key, ts)
AS SELECT toString(number % 100) AS key, toDateTime(number) AS ts, number AS val
FROM numbers(100000);
OPTIMIZE TABLE t_04365 FINAL;

-- Helper output: `pushed` = 1 when the partition-key predicate reached ReadFromMergeTree
-- as a primary key Condition (so it was pushed below the window). Matching the
-- ReadFromMergeTree "Condition:" line is robust to randomized index_granularity /
-- plan-shape settings (no arithmetic on the granule counts, which can differ or be
-- absent under some randomized settings).

-- QUALIFY form: partition-key conjunct pushed below Window -> PK condition on storage.
SELECT count() > 0 AS pushed FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
) WHERE explain ILIKE '%Condition:%key%';

-- Outer WHERE around a windowed subquery: same push down.
SELECT count() > 0 AS pushed FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn FROM t_04365
    ) WHERE key = '5'
) WHERE explain ILIKE '%Condition:%key%';

-- Multiple windows sharing the partition key: the common conjunct still reaches storage.
SELECT count() > 0 AS pushed FROM (
    EXPLAIN indexes = 1
    SELECT key, ts,
        row_number() OVER (PARTITION BY key ORDER BY ts) AS rn,
        sum(val)     OVER (PARTITION BY key ORDER BY ts) AS s
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
) WHERE explain ILIKE '%Condition:%key%';

-- Mixed predicate: the partition-key conjunct pushes (PK condition on storage), the
-- window-result conjunct stays above.
SELECT count() > 0 AS pushed FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5' AND ts > toDateTime(10)
) WHERE explain ILIKE '%Condition:%key%';

-- Negative: a predicate on the window result only must NOT be pushed (no PK condition).
SELECT count() > 0 AS pushed FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (PARTITION BY key ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1
) WHERE explain ILIKE '%Condition:%key%';

-- No PARTITION BY: nothing to push, must not crash and must not push a PK condition.
SELECT count() > 0 AS pushed FROM (
    EXPLAIN indexes = 1
    SELECT key, ts, row_number() OVER (ORDER BY ts) AS rn
    FROM t_04365 QUALIFY rn = 1 AND key = '5'
) WHERE explain ILIKE '%Condition:%key%';

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
