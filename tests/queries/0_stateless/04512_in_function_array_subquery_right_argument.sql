-- A subquery on the right side of IN whose single column is an Array exactly one dimension
-- deeper than the left argument is the set of the elements of those arrays, exactly like an
-- array literal or an array-returning function on the right side of IN. Previously the whole
-- array was treated as a single opaque set value, so the left argument was coerced into the
-- array type, giving a confusing "Array does not start with '[' character" or a type-mismatch
-- error. Flattening the column with arrayJoin makes `x IN (SELECT groupArray(x) ...)` behave
-- like `x IN (SELECT arrayJoin(groupArray(x)) ...)`.
-- https://github.com/ClickHouse/ClickHouse/issues/37066

SET enable_analyzer = 1;

-- The two cases from the issue: scalar IN (subquery returning Array of the scalar type).
SELECT count() FROM numbers(10) WHERE number IN (SELECT groupArray(number) FROM numbers(10));
SELECT count() FROM numbers(10) WHERE toString(number) IN (SELECT groupArray(toString(number)) FROM numbers(10));

-- Membership is by element, exactly like arrayJoin of the same subquery.
SELECT 5 IN (SELECT groupArray(number) FROM numbers(10)) AS in_res, 5 IN (SELECT arrayJoin(groupArray(number)) FROM numbers(10)) AS aj_res;
SELECT 42 IN (SELECT groupArray(number) FROM numbers(10)) AS in_res, 42 IN (SELECT arrayJoin(groupArray(number)) FROM numbers(10)) AS aj_res;

-- NOT IN is the negation.
SELECT count() FROM numbers(10) WHERE number NOT IN (SELECT groupArray(number) FROM numbers(5));

-- GLOBAL IN.
SELECT 3 GLOBAL IN (SELECT groupArray(number) FROM numbers(10)) AS present, 100 GLOBAL IN (SELECT groupArray(number) FROM numbers(10)) AS absent;

-- Multiple rows: the set is the union of the elements of all the arrays.
SELECT x, x IN (SELECT [number, number + 10] FROM numbers(2)) AS in_res
FROM (SELECT arrayJoin([0, 1, 5, 10, 11]) AS x)
ORDER BY x;

-- String elements.
SELECT 'b' IN (SELECT ['a', 'b', 'c']) AS present, 'z' IN (SELECT ['a', 'b', 'c']) AS absent;

-- The left argument may be an Array with an Array(Array(...)) subquery on the right (depth + 1).
SELECT [1, 2] IN (SELECT [[1, 2], [3, 4]]) AS present, [9, 9] IN (SELECT [[1, 2], [3, 4]]) AS absent;

-- Equal depths (Array(T) on both sides) stay a one-element set, i.e. equality, not flattening.
SELECT [0, 1, 2] IN (SELECT groupArray(number) FROM numbers(3)) AS eq_in, [0, 1] IN (SELECT groupArray(number) FROM numbers(3)) AS ne_in;

-- Empty arrays produce an empty set.
SELECT 1 IN (SELECT emptyArrayUInt64());

-- Regression guard: an ordinary scalar subquery is unchanged.
SELECT 3 IN (SELECT number FROM numbers(10)) AS present, 99 IN (SELECT number FROM numbers(10)) AS absent;

-- Regression guard: a multi-column subquery (tuple IN) is unaffected.
SELECT (1, 2) IN (SELECT 1, 2) AS present, (1, 3) IN (SELECT 1, 2) AS absent;

-- Nullable elements: with `transform_null_in = 1` a NULL element of the array is a member of the
-- set, so a NULL left argument matches it, exactly like the scalar-subquery case. With the default
-- `transform_null_in = 0` a comparison against NULL is unknown (NULL), as usual for IN.
SELECT x, x IN (SELECT [NULL, toNullable(1)]) AS in_res
FROM (SELECT arrayJoin([NULL, toNullable(1), toNullable(2)]) AS x)
ORDER BY x
SETTINGS transform_null_in = 1;
SELECT x, x IN (SELECT [NULL, toNullable(1)]) AS in_res
FROM (SELECT arrayJoin([NULL, toNullable(1), toNullable(2)]) AS x)
ORDER BY x
SETTINGS transform_null_in = 0;

-- The same behavior must not depend on `rewrite_in_to_join`, which rewrites a non-constant
-- `x IN (subquery)` into a correlated `EXISTS` before the regular IN handling runs. Without
-- flattening the array subquery there too, the rewrite would compare `x = <array>` and keep the
-- reported bug alive whenever this setting is enabled.
SET allow_experimental_correlated_subqueries = 1;
SET rewrite_in_to_join = 1;
SELECT count() FROM numbers(10) WHERE number IN (SELECT groupArray(number) FROM numbers(10));
SELECT count() FROM numbers(10) WHERE number NOT IN (SELECT groupArray(number) FROM numbers(5));

-- Under `rewrite_in_to_join` the non-constant `x IN (subquery)` is lowered to a correlated EXISTS
-- that compares with `equals`, which does not honor `transform_null_in` for a NULL set element.
-- This is a pre-existing property of the rewrite that applies to a scalar subquery too, so the
-- flattened array subquery behaves exactly like the equivalent `SELECT arrayJoin(...)` scalar
-- subquery and adds no new inconsistency: the NULL row is 0 under the rewrite (it honors
-- `transform_null_in` only on the regular IN path above, tested with the same data).
SELECT x, x IN (SELECT [NULL, toNullable(1)]) AS in_res
FROM (SELECT arrayJoin([NULL, toNullable(1), toNullable(2)]) AS x)
ORDER BY x
SETTINGS transform_null_in = 1;
SELECT x, x IN (SELECT arrayJoin([NULL, toNullable(1)])) AS in_res
FROM (SELECT arrayJoin([NULL, toNullable(1), toNullable(2)]) AS x)
ORDER BY x
SETTINGS transform_null_in = 1;
