-- Test that ORDER BY ... WITH FILL respects max_execution_time and stops promptly.
-- A single WITH FILL range can expand into billions of rows inside one transform() call, and the
-- pipeline executor only enforces max_execution_time between work() calls, so without checking the
-- time limit inside the generation loops the query runs unbounded and ignores the limit.
-- Ref: https://github.com/ClickHouse/ClickHouse/issues/61713

SET max_execution_time = 1;

-- `timeout_overflow_mode = 'throw'`: the query must stop with a timeout after ~1 second instead of
-- filling ~3 billion rows (year 2000 to 2100, step 1 second) and running for minutes.
SELECT ts
FROM (SELECT toDateTime('2050-06-15 12:00:00') AS ts)
ORDER BY ts WITH FILL FROM toDateTime('2000-01-01 00:00:00') TO toDateTime('2100-01-01 00:00:00') STEP 1
SETTINGS timeout_overflow_mode = 'throw'
FORMAT Null; -- { serverError TIMEOUT_EXCEEDED }

-- `timeout_overflow_mode = 'break'`: the soft timeout does not throw and, during a single long
-- transform() call, does not set `isCancelled` either (the executor observes it only between work()
-- calls, and the cancellation checker leaves break-mode queries running). So besides the inner
-- generation loops, the outer loops over ranges and over input rows must also stop on the time limit;
-- otherwise, after the first range hits the limit, WITH FILL keeps refilling every remaining range
-- (each up to DEFAULT_BLOCK_SIZE rows), running far past max_execution_time.
--
-- `max_memory_usage` bounds the pre-fix runaway so it fails fast (memory limit) instead of hanging,
-- while the fixed version stops after ~1 second and returns a partial result well within the limit.

-- Many sorting-prefix groups, each expanding to a huge suffix: exercises the outer loop over ranges in
-- transform() (it must stop instead of refilling the remaining groups).
SELECT g, x
FROM (SELECT number AS g, 0::UInt64 AS x FROM numbers(100000))
ORDER BY g, x WITH FILL FROM 0 TO 1000000000000 STEP 1
SETTINGS timeout_overflow_mode = 'break', use_with_fill_by_sorting_prefix = 1, max_memory_usage = 4000000000
FORMAT Null;

-- Many input rows in a single range, each separated by a huge gap: exercises the outer loop over input
-- rows in transformRange() (it must stop instead of refilling the gaps after the remaining rows).
SELECT x
FROM (SELECT (number * 10000000000)::UInt64 AS x FROM numbers(100000))
ORDER BY x WITH FILL FROM 0 TO 1000000000000000 STEP 1
SETTINGS timeout_overflow_mode = 'break', max_memory_usage = 4000000000
FORMAT Null;
