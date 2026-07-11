-- Filter push down below LimitByStep on LIMIT BY key columns (issue #110112).
-- A predicate referencing only the LIMIT BY key columns is safe to apply before the
-- LIMIT BY (it only removes whole groups), so it must reach storage and enable
-- primary key / granule pruning.

SET enable_analyzer = 1;
SET query_plan_filter_push_down = 1;

DROP TABLE IF EXISTS t_04366;
CREATE TABLE t_04366 (key String, ts DateTime, val UInt64)
ENGINE = MergeTree ORDER BY (key, ts)
AS SELECT toString(number % 100) AS key, toDateTime(number) AS ts, number AS val
FROM numbers(100000);
OPTIMIZE TABLE t_04366 FINAL;

-- Helper output: `pruned` = 1 when the primary key pruned some granules
-- (read granules < total granules), robust to the absolute granule count which
-- depends on randomized index_granularity. `pruned` = 0 when the full table is scanned.
-- Only well-formed `Granules: N/M` rows are considered (match filter), so extra
-- `Granules:` lines from other index sections cannot break the parse.

-- Outer WHERE around a LIMIT n BY subquery: key-column conjunct pushed below LimitBy -> PK prunes.
SELECT sum(toUInt64OrZero(extract(explain, 'Granules: (\d+)/')) < toUInt64OrZero(extract(explain, 'Granules: \d+/(\d+)'))) > 0 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
    ) WHERE key = '5'
) WHERE match(explain, 'Granules: \d+/\d+');

-- LIMIT n OFFSET m BY: the per-group offset is unaffected by dropping whole groups.
SELECT sum(toUInt64OrZero(extract(explain, 'Granules: (\d+)/')) < toUInt64OrZero(extract(explain, 'Granules: \d+/(\d+)'))) > 0 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 2 OFFSET 1 BY key
    ) WHERE key = '5'
) WHERE match(explain, 'Granules: \d+/\d+');

-- Mixed predicate: the key conjunct pushes (granules pruned), the non-key conjunct stays above.
SELECT sum(toUInt64OrZero(extract(explain, 'Granules: (\d+)/')) < toUInt64OrZero(extract(explain, 'Granules: \d+/(\d+)'))) > 0 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
    ) WHERE key = '5' AND val > 10
) WHERE match(explain, 'Granules: \d+/\d+');

-- Negative: a predicate on a non-key column (ts is in the PK but NOT a LIMIT BY key) must
-- NOT be pushed below LimitBy, so the primary key does not prune on it.
SELECT sum(toUInt64OrZero(extract(explain, 'Granules: (\d+)/')) < toUInt64OrZero(extract(explain, 'Granules: \d+/(\d+)'))) > 0 AS pruned
FROM (
    EXPLAIN indexes = 1
    SELECT * FROM (
        SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
    ) WHERE ts > toDateTime(99990)
) WHERE match(explain, 'Granules: \d+/\d+');

-- Correctness: pushing the key filter below LIMIT BY must not change the result.
SELECT count(), sum(val) FROM (
    SELECT key, ts, val FROM t_04366 ORDER BY key, ts LIMIT 1 BY key
) WHERE key = '5';

-- Same result when the filter is forced before the LIMIT BY.
SELECT count(), sum(val) FROM (
    SELECT key, ts, val FROM t_04366 WHERE key = '5' ORDER BY key, ts LIMIT 1 BY key
);

DROP TABLE t_04366;
