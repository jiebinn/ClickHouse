-- Binary arithmetic functions (plus/minus/multiply/...) resolve their interval/tuple sub-function
-- builders at build time and cache the Date/Time overflow behavior, so executeImpl never consults
-- the query context. This keeps a stored expression (e.g. a sorting key) executable during a merge
-- after the context that built it has been destroyed. Related: issue #54890.

SET session_timezone = 'UTC';

-- Interval arithmetic must resolve the same sub-function for plain, Nullable and LowCardinality
-- operands (the argument types are normalized the same way executeImpl sees them).
SELECT toDate('2020-01-01') + INTERVAL 1 DAY;
SELECT toNullable(toDate('2020-01-01')) + INTERVAL 1 DAY;
SELECT toLowCardinality(toDate('2020-01-01')) + INTERVAL 1 DAY;
SELECT materialize(toDate('2020-01-01')) - INTERVAL 1 MONTH;
SELECT toDateTime('2020-01-01 00:00:00', 'UTC') + INTERVAL 1 HOUR;
SELECT materialize(toNullable(toDateTime('2020-01-01 00:00:00', 'UTC'))) + INTERVAL 90 MINUTE;

-- Date plus a tuple of intervals.
SELECT toDate('2020-01-01') + (INTERVAL 1 DAY, INTERVAL 1 MONTH);

-- Interval plus interval (merge into a tuple of intervals).
SELECT INTERVAL 1 DAY + INTERVAL 1 HOUR;

-- Tuple arithmetic and tuple-and-number arithmetic, plain and materialized.
SELECT (1, 2, 3) + (4, 5, 6);
SELECT (1, 2) - (3, 4);
SELECT (10, 20) * 2;
SELECT materialize((1, 2)) + (3, 4);

-- Array arithmetic re-evaluates the operation on the element types (a separate code path that
-- delegates to a sibling function built for those element types).
SELECT [1, 2, 3] + [4, 5, 6];
SELECT materialize([1, 2, 3]) + materialize([4, 5, 6]);
SELECT [1, 2, 3] * 2;
SELECT [toDate('2020-01-01'), toDate('2020-06-01')] + [toIntervalDay(1), toIntervalDay(2)];
SELECT [[1, 2], [3]] + [[10, 20], [30]];

-- Plain numeric arithmetic (no special case applies; the context is not touched at all).
SELECT number + 1, number * 2, number - 3 FROM numbers(3);

-- Regression: a sorting key that contains arithmetic is stored in the table metadata with the
-- expression built by the CREATE query. It must still be executable during a background merge,
-- when that query context no longer exists.
DROP TABLE IF EXISTS t_arith_key;
CREATE TABLE t_arith_key (a Int64, b Int64, d Date) ENGINE = MergeTree ORDER BY (a + b);
INSERT INTO t_arith_key VALUES (3, 4, '2020-01-01');
INSERT INTO t_arith_key VALUES (1, 1, '2020-01-02');
OPTIMIZE TABLE t_arith_key FINAL;
SELECT a, b, d FROM t_arith_key ORDER BY a, b;
DROP TABLE t_arith_key;
