-- indexHint ignores its arguments and always returns UInt8. A Nothing-typed
-- argument must not force the declared return type to Nothing (which mismatched
-- the UInt8 column produced at execution and propagated up, tripping a chassert
-- inside anyLast_respect_nulls). See indexHint.h useDefaultImplementationForNothing.

SELECT toTypeName(indexHint(assumeNotNull(materialize(NULL))));
SELECT indexHint(assumeNotNull(materialize(NULL)));

-- The original fuzzer-found chain: a Nothing arg flows through indexHint into an
-- aggregate state combinator. Must not abort (LOGICAL_ERROR 'returns_nullable_type').
SELECT anyLastRespectNullsStateOrDefaultDistinct(divide(toFixedString(NULL, JSONExtractBoolCaseInsensitive(indexHint(assumeNotNull(toString(materialize(NULL), 1025))), toString(NULL))), ';--')) GROUP BY ALL FORMAT Null;

-- indexHint keeps working normally.
SELECT toTypeName(indexHint(1)), indexHint(1), indexHint(NULL);

-- The sibling ignore() has the same declared-result invariant (always UInt8), so a
-- Nothing-typed argument must not rewrite its declared return type to Nothing either.
-- See ignore.cpp useDefaultImplementationForNothing. Only the declared type is checked:
-- unlike indexHint, ignore evaluates its arguments, and a non-empty Nothing column
-- cannot be materialized, so the value form is not evaluable and is not the bug here.
SELECT toTypeName(ignore(assumeNotNull(materialize(NULL))));
SELECT toTypeName(ignore(1)), ignore(1), ignore(NULL);
