-- Test: `defaultImplementationForNulls` no longer pre-allocates the result null map,
-- so with a single non-const Nullable argument the argument's null map is shared with
-- the result, and with no non-const Nullable arguments the null map stays empty.
-- Regression check for the case where no per-row null map exists at all:
-- a const non-NULL Nullable argument combined with a non-const non-Nullable argument,
-- with short_circuit_function_evaluation_for_nulls_threshold = 0 (which makes the
-- short-circuit condition true for a zero null ratio).

SELECT 'no per-row null map, threshold=0';
SELECT concat(toNullable('a'), materialize('b'))
FROM numbers(4)
SETTINGS short_circuit_function_evaluation_for_nulls = 1, short_circuit_function_evaluation_for_nulls_threshold = 0.0;

SELECT 'no per-row null map, short circuit disabled';
SELECT concat(toNullable('a'), materialize('b'))
FROM numbers(4)
SETTINGS short_circuit_function_evaluation_for_nulls = 0;

-- Single non-const Nullable argument: the result shares the argument's null map.
-- Both the argument and the result are returned, so corruption of the shared
-- null map by either column would be visible.
SELECT 'shared null map, single nullable argument';
SELECT materialize(if(number % 2 = 0, 'AbC', NULL)::Nullable(String)) AS x, lower(x), x IS NULL
FROM numbers(4) ORDER BY number;

-- Two non-const Nullable arguments: the shared map must be copied before merging
-- the second argument's null map, leaving the first argument's null map intact.
SELECT 'two nullable arguments';
SELECT
    materialize(if(number % 2 = 0, 'a', NULL)::Nullable(String)) AS x,
    materialize(if(number % 3 = 0, 'b', NULL)::Nullable(String)) AS y,
    concat(x, y), x IS NULL, y IS NULL
FROM numbers(6) ORDER BY number;
