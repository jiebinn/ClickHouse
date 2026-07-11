-- Filter push down below LimitByStep / NegativeLimitByStep on LIMIT BY key columns (issue #110112).
-- A predicate referencing only the LIMIT BY key columns is safe to apply before the
-- LIMIT BY (it only removes whole groups), so it must reach storage as the driving
-- primary key condition. This covers positive LIMIT BY, negative LIMIT BY, and the
-- mixed-sign decompositions (which the planner lowers to combinations of the two steps).

SET enable_analyzer = 1;
SET query_plan_filter_push_down = 1;

DROP TABLE IF EXISTS t_04366;
CREATE TABLE t_04366 (key String, ts DateTime, val UInt64)
ENGINE = MergeTree ORDER BY (key, ts)
AS SELECT toString(number % 100) AS key, toDateTime(number) AS ts, number AS val
FROM numbers(100000);
OPTIMIZE TABLE t_04366 FINAL;

-- `pushed` = 1 when the pushed LIMIT BY-key predicate reaches storage and becomes the
-- driving primary key condition (`Condition: (key in ...)`). This anchors on the
-- PrimaryKey condition text, not on arbitrary `Granules:` lines (which can come from
-- unrelated index sections). = 0 when the filter stays above and the PK sees no condition.

-- LIMIT n BY: key-column conjunct pushed below LimitBy -> PK condition on key.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
    ) WHERE key = '5'
);

-- LIMIT n OFFSET m BY: the per-group offset is unaffected by dropping whole groups.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 2 OFFSET 1 BY key
    ) WHERE key = '5'
);

-- LIMIT -n BY: negative LIMIT BY (NegativeLimitByStep) is equally safe.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT -1 BY key
    ) WHERE key = '5'
);

-- LIMIT -n OFFSET -m BY: both negative -> single NegativeLimitByStep.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT -2 OFFSET -1 BY key
    ) WHERE key = '5'
);

-- LIMIT -n OFFSET m BY: mixed sign -> LimitBy(offset) then NegativeLimitBy; the filter
-- must push below both.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT -2 OFFSET 1 BY key
    ) WHERE key = '5'
);

-- LIMIT n OFFSET -m BY: mixed sign -> NegativeLimitBy(offset) then LimitBy; the filter
-- must push below both.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 2 OFFSET -1 BY key
    ) WHERE key = '5'
);

-- Mixed predicate: the key conjunct pushes (PK condition on key), the non-key conjunct
-- stays above.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
    ) WHERE key = '5' AND val > 10
);

-- Negative: a predicate on a non-key column (ts is in the PK but NOT a LIMIT BY key) must
-- NOT be pushed below LimitBy, so the primary key gets no condition on it.
SELECT countIf(match(explain, 'Condition: \(key in ')) > 0 AS pushed
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
    ) WHERE ts > toDateTime(99990)
);

-- Correctness: pushing the key filter below LIMIT BY must not change the result.
SELECT count(), sum(val) FROM (
    SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
) WHERE key = '5';

-- Same result when the filter is forced before the LIMIT BY.
SELECT count(), sum(val) FROM (
    SELECT key, ts, val FROM t_04366 WHERE key = '5' ORDER BY key, ts LIMIT 1 BY key
);

DROP TABLE t_04366;
