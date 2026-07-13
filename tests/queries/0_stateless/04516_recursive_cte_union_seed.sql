-- Test recursive CTE whose seed (non-recursive) term is itself a UNION of several SELECTs.
SET enable_analyzer = 1;

-- Seed is a UNION DISTINCT of two SELECTs; recursion widens from every seed row.
SELECT sum(n), count()
FROM (
    WITH RECURSIVE t AS (
        (SELECT toUInt64(1) AS n UNION DISTINCT SELECT toUInt64(2))
        UNION ALL
        SELECT n + 1 FROM t WHERE n < 4
    )
    SELECT n FROM t ORDER BY n
)
ORDER BY 1;

-- Seed union has heterogeneous types, forcing supertype inference over the union seed.
SELECT sum(n), count()
FROM (
    WITH RECURSIVE t AS (
        (SELECT toUInt8(1) AS n UNION DISTINCT SELECT toUInt64(300))
        UNION ALL
        SELECT n + 1 FROM t WHERE n < 3
    )
    SELECT n FROM t ORDER BY n
)
ORDER BY 1;
