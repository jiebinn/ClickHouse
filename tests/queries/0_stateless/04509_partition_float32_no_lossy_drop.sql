-- Regression test for https://github.com/ClickHouse/ClickHouse/pull/108055
-- `ALTER ... PARTITION` target resolution (MergeTreeData::getPartitionIDFromQuery) must stay
-- fail-closed for float literals that are not exactly representable in the partition-key type:
-- it goes through `convertFieldToTypeOrThrow`, which must NOT silently round the literal for a
-- destructive statement. On a `Float32` partition key, `DROP PARTITION 0.1` is rejected (0.1 is not
-- exactly representable in Float32) instead of being rounded and dropping a partition; the user must
-- pass an exactly-representable value.

DROP TABLE IF EXISTS t_partition_float32;
CREATE TABLE t_partition_float32 (x Float32, v UInt32) ENGINE = MergeTree ORDER BY v PARTITION BY x
    SETTINGS allow_floating_point_partition_key = 1;
INSERT INTO t_partition_float32 VALUES (toFloat32(0.5), 1), (toFloat32(0.1), 2);

SELECT '-- initial rows (two partitions)';
SELECT v, x FROM t_partition_float32 ORDER BY v;

SELECT '-- DROP PARTITION with an unrepresentable Float64 literal is rejected (fail-closed)';
ALTER TABLE t_partition_float32 DROP PARTITION 0.1; -- { serverError ARGUMENT_OUT_OF_BOUND }

SELECT '-- nothing was dropped';
SELECT count() FROM t_partition_float32;

SELECT '-- an exactly-representable literal resolves the partition';
ALTER TABLE t_partition_float32 DROP PARTITION 0.5;
SELECT v, x FROM t_partition_float32 ORDER BY v;

SELECT '-- the remaining partition is dropped with the exactly-representable value';
ALTER TABLE t_partition_float32 DROP PARTITION 0.10000000149011612;
SELECT count() FROM t_partition_float32;

DROP TABLE t_partition_float32;
