-- Test that ORDER BY ... WITH FILL respects max_execution_time.
-- A single WITH FILL range can expand into billions of rows inside one transform() call, so without
-- enforcing the time limit inside the generation loop the query runs unbounded and ignores the limit.
-- Ref: https://github.com/ClickHouse/ClickHouse/issues/61713

SET max_execution_time = 1;
SET timeout_overflow_mode = 'throw';

-- This fills ~3 billion rows (year 2000 to 2100, step 1 second) from a single source row.
-- With the fix it stops with a timeout after ~1 second instead of running for minutes.
SELECT ts
FROM (SELECT toDateTime('2050-06-15 12:00:00') AS ts)
ORDER BY ts WITH FILL FROM toDateTime('2000-01-01 00:00:00') TO toDateTime('2100-01-01 00:00:00') STEP 1
FORMAT Null; -- { serverError TIMEOUT_EXCEEDED }
