-- A correlated subquery over a relation whose header carries a column name twice
-- (`SELECT number, *` yields two columns named `number`) is decorrelated into an
-- ANY RIGHT JOIN. With join runtime filters enabled the filter is built on that
-- duplicated key column, changing a downstream join input's column multiplicity and
-- aborting with `Block structure mismatch in JoinStep` in debug/sanitizer builds.

SET enable_analyzer = 1;
SET allow_experimental_correlated_subqueries = 1;
SET enable_join_runtime_filters = 1;
-- The runtime filter is only built for hash-family algorithms (supportsRuntimeFilter),
-- and CI randomizes join_algorithm. Pin it so the repro statements below deterministically
-- exercise the crashing runtime-filter path (they fail on the pre-fix build) instead of
-- silently passing under e.g. full_sorting_merge / partial_merge.
SET join_algorithm = 'hash';
-- The filter is only added for the RIGHT-build side (can_use_runtime_filter). CI randomizes
-- query_plan_optimize_join_order_randomize, which feeds random cardinalities to the join-order
-- optimizer and can swap ANY RIGHT JOIN into an ANY LEFT JOIN (build side becomes the left
-- input), for which no runtime filter is built. Pin query_plan_join_swap_table='false' so the
-- right table stays the build side and the filter is deterministically produced.
SET query_plan_join_swap_table = 'false';

-- { echoOn }
WITH t AS (SELECT number, * FROM numbers(3))
SELECT *, (SELECT t.number WHERE t.number >= 0) AS r FROM t
ORDER BY 1
SETTINGS correlated_subqueries_default_join_kind = 'right', correlated_subqueries_use_in_memory_buffer = 0;

WITH t AS (SELECT number, * FROM numbers(3))
SELECT *, (SELECT t.number WHERE t.number >= 0) AS r FROM t
ORDER BY 1
SETTINGS correlated_subqueries_default_join_kind = 'left', correlated_subqueries_use_in_memory_buffer = 0;

WITH t AS (SELECT number, *, * FROM numbers(3))
SELECT *, (SELECT t.number WHERE t.number >= 0) AS r FROM t
ORDER BY 1
SETTINGS correlated_subqueries_default_join_kind = 'right', correlated_subqueries_use_in_memory_buffer = 0;
-- { echoOff }

-- A runtime filter is still built for a normal join without duplicated column names.
-- Pin join_algorithm='hash': the filter is only added for hash-family algorithms
-- (supportsRuntimeFilter), and CI randomizes join_algorithm, so an unpinned run may
-- pick e.g. full_sorting_merge and build no filter, making this assertion flap.
SELECT countIf(explain LIKE '%BuildRuntimeFilter%') > 0
FROM (
    EXPLAIN PLAN
    SELECT * FROM (SELECT number AS a FROM numbers(100)) AS l
    ANY RIGHT JOIN (SELECT number AS b FROM numbers(3)) AS r ON l.a = r.b
    SETTINGS enable_join_runtime_filters = 1, join_algorithm = 'hash', query_plan_join_swap_table = 'false'
);

-- The filter must still be built when the build side has a duplicated NON-key column
-- (here the key `b` is unique and `c` is duplicated): only a duplicated KEY column breaks the
-- name-keyed filter machinery, so the guard is scoped to the join keys and must not disable the
-- filter for an unrelated duplicated column.
SELECT countIf(explain LIKE '%BuildRuntimeFilter%') > 0
FROM (
    EXPLAIN PLAN
    SELECT * FROM (SELECT number AS a FROM numbers(100)) AS l
    ANY RIGHT JOIN (SELECT number AS b, number + 1 AS c, c FROM numbers(3)) AS r ON l.a = r.b
    SETTINGS enable_join_runtime_filters = 1, join_algorithm = 'hash', query_plan_join_swap_table = 'false'
);
