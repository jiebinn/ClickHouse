-- Tags: no-parallel, no-fasttest, no-object-storage, no-random-settings
-- no-parallel: the failpoint below is server-global.
-- no-object-storage: the test builds its own cache-over-s3 disk.

-- Regression for a LOGICAL_ERROR (server abort in debug/sanitizer builds) when the
-- file_cache_dynamic_resize_fail_to_evict failpoint fired during ordinary space
-- reservation. That failpoint models an eviction failure and is only meaningful for the
-- dynamic cache resize feature, so it must not fire on the reserve path. See issue #88945.

DROP TABLE IF EXISTS t_03908;

CREATE TABLE t_03908 (c0 Int)
ENGINE = MergeTree()
ORDER BY tuple()
SETTINGS min_bytes_for_wide_part = 0,
         disk = disk(
            type = cache,
            name = '03908_cache_evict_failure',
            max_size = '25Ki',
            path = '03908_cache_evict_failure/',
            disk = 's3_disk');

INSERT INTO t_03908 SELECT number FROM numbers(1399);

SYSTEM ENABLE FAILPOINT file_cache_dynamic_resize_fail_to_evict;

-- This second insert forces cache eviction on the reserve path. Before the fix the
-- failpoint fired here and aborted the server with a LOGICAL_ERROR. Now the failpoint
-- is confined to dynamic resize, so eviction proceeds normally and the insert succeeds.
INSERT INTO t_03908 SELECT number FROM numbers(1770);

SYSTEM DISABLE FAILPOINT file_cache_dynamic_resize_fail_to_evict;

-- The server is still alive and the data is queryable.
SELECT count() FROM t_03908;

DROP TABLE t_03908;
