-- Test that toStartOfInterval with extreme DateTime64 values returns well-defined results without overflow.
-- Use reinterpret to inject raw internal values because CAST clamps to the valid DateTime64 range.
-- The start of the interval is computed by floor division, so no intermediate value can exceed the input.

-- Millisecond interval with scale=6 (microseconds)
SELECT toStartOfInterval(reinterpret(toInt64(9223372036854775806), 'DateTime64(6, \'UTC\')'), toIntervalMillisecond(1));

-- Millisecond interval with scale=9 (nanoseconds)
SELECT toStartOfInterval(reinterpret(toInt64(9223372036854775806), 'DateTime64(9, \'UTC\')'), toIntervalMillisecond(1));

-- Microsecond interval with scale=9 (nanoseconds)
SELECT toStartOfInterval(reinterpret(toInt64(9223372036854775806), 'DateTime64(9, \'UTC\')'), toIntervalMicrosecond(1));
