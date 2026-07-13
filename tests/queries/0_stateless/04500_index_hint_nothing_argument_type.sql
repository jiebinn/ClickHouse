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
