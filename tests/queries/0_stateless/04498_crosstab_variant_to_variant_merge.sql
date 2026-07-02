SET enable_analyzer = 1;

-- Chained set operations (EXCEPT / INTERSECT) and nested UNION ALL build a Variant that is itself cast to
-- another Variant (Variant -> Variant). When one branch produces the window state representation and another
-- the aggregation representation of the same aggregate function, they collapse into one variant slot by name.
-- The Variant -> Variant cast must convert the subcolumn to the destination representation instead of copying
-- it verbatim, otherwise a later read of the state (here via the -Merge combinator) reads the bytes with the
-- wrong layout and the server aborts. See createVariantToVariantWrapper.

SELECT round(cramersVBiasCorrectedMerge(s.`AggregateFunction(cramersVBiasCorrected, UInt8, UInt8)`), 4)
FROM
(
    SELECT cramersVBiasCorrectedState(toUInt8(number % 10), toUInt8(number % 6)) OVER () AS s FROM numbers(100) LIMIT 1
    EXCEPT DISTINCT
    SELECT cramersVBiasCorrectedState(toUInt8(number % 10), toInt128OrDefault(number % 65535)) AS s FROM numbers(100)
    EXCEPT DISTINCT
    SELECT cramersVBiasCorrectedState(toUInt8(number % 10), toUInt8(number % 65535)) AS s FROM numbers(100)
);

SELECT round(cramersVMerge(s.`AggregateFunction(cramersV, UInt8, UInt8)`), 4)
FROM
(
    SELECT cramersVState(toUInt8(number % 10), toUInt8(number % 6)) OVER () AS s FROM numbers(100) LIMIT 1
    EXCEPT DISTINCT
    SELECT cramersVState(toUInt8(number % 10), toInt128OrDefault(number % 65535)) AS s FROM numbers(100)
    EXCEPT DISTINCT
    SELECT cramersVState(toUInt8(number % 10), toUInt8(number % 65535)) AS s FROM numbers(100)
);

SELECT round(contingencyMerge(s.`AggregateFunction(contingency, UInt8, UInt8)`), 4)
FROM
(
    SELECT contingencyState(toUInt8(number % 10), toUInt8(number % 6)) OVER () AS s FROM numbers(100) LIMIT 1
    EXCEPT DISTINCT
    SELECT contingencyState(toUInt8(number % 10), toInt128OrDefault(number % 65535)) AS s FROM numbers(100)
    EXCEPT DISTINCT
    SELECT contingencyState(toUInt8(number % 10), toUInt8(number % 65535)) AS s FROM numbers(100)
);

SELECT round(theilsUMerge(s.`AggregateFunction(theilsU, UInt8, UInt8)`), 4)
FROM
(
    SELECT theilsUState(toUInt8(number % 10), toUInt8(number % 6)) OVER () AS s FROM numbers(100) LIMIT 1
    EXCEPT DISTINCT
    SELECT theilsUState(toUInt8(number % 10), toInt128OrDefault(number % 65535)) AS s FROM numbers(100)
    EXCEPT DISTINCT
    SELECT theilsUState(toUInt8(number % 10), toUInt8(number % 65535)) AS s FROM numbers(100)
);

-- Nested UNION ALL: the inner UNION ALL already yields a Variant, the outer UNION ALL casts it Variant -> Variant.
SELECT round(cramersVBiasCorrectedMerge(s.`AggregateFunction(cramersVBiasCorrected, UInt8, UInt8)`), 4)
FROM
(
    SELECT s FROM
    (
        SELECT cramersVBiasCorrectedState(toUInt8(number % 10), toUInt8(number % 6)) OVER () AS s FROM numbers(100) LIMIT 1
        UNION ALL
        SELECT cramersVBiasCorrectedState(toUInt8(number % 10), toInt128OrDefault(number % 65535)) AS s FROM numbers(100)
    )
    UNION ALL
    SELECT cramersVBiasCorrectedState(toUInt8(number % 10), toUInt8(number % 6)) AS s FROM numbers(100)
);
