-- Tags: no-asan, no-tsan, no-msan, no-ubsan, no-sanitize-coverage, no-llvm-coverage, no-cfi
-- no-cfi: CFI (cfi-vcall) routes indirect calls through jump-table thunks that carry no
-- source line info, so the query profiler's symbolized `lines` lack file:line:column frames
-- and the last assertion fails. This is a symbolization artifact, not a CFI violation.

SET log_queries = 1;
SET log_query_threads = 1;
SET query_profiler_real_time_period_ns = 100000000;
SELECT sleep(1);
SYSTEM FLUSH LOGS trace_log;

SELECT COUNT(*) > 1 FROM system.trace_log WHERE event_date >= yesterday() AND event_time >= now() - 600 AND build_id IS NOT NULL;
SELECT countIf(arrayExists(x -> x LIKE '%:%:%', lines)) > 1 FROM system.trace_log WHERE event_date >= yesterday() AND event_time >= now() - 600;

