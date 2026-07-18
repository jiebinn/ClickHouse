-- A -Tuple + RESPECT NULLS -State must be named after its resolved variant (any_respect_nullsTuple),
-- so its serialized type identifies the correct state layout on a round-trip.

SELECT toTypeName(anyTupleState(tuple(1::UInt32)) RESPECT NULLS);
SELECT toTypeName(anyTupleStateDistinct(tuple(1::UInt32)) RESPECT NULLS);
SELECT toTypeName(anyTupleState(tuple(1::UInt32)));
SELECT toTypeName(anyTupleState(tuple(1::UInt32)) RESPECT NULLS) = toTypeName(anyRespectNullsTupleState(tuple(1::UInt32)));

-- The window-variant state round-trips through its own (now correct) type name without confusion.
SELECT finalizeAggregation(CAST(anyTupleState(tuple(v)) RESPECT NULLS AS AggregateFunction(any_respect_nullsTuple, Tuple(Nullable(UInt32))))) FROM (SELECT 5::Nullable(UInt32) AS v);

-- Distributed round-trip (shard -> initiator re-resolution) over the full combinator chain no longer
-- reconstructs a mismatched state layout; the server survives and returns a result.
SELECT finalizeAggregation(anyTupleStateDistinct(tuple(number)) RESPECT NULLS) AS k FROM remote('127.0.0.{1,2}', numbers(3)) GROUP BY ALL WITH ROLLUP ORDER BY k;
