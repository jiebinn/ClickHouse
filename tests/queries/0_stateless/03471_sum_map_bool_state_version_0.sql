-- Regression test for "Cannot sum Bools" LOGICAL_ERROR.
-- Version-0 sumMap-family state used to deserialize Bool values into bool-tagged Fields,
-- which threw in FieldVisitorSum on merge when two states shared a map key.

DROP TABLE IF EXISTS t_sum_map_bool_v0;
CREATE TABLE t_sum_map_bool_v0 (s AggregateFunction(0, sumMap, Array(UInt8), Array(Bool))) ENGINE = TinyLog;
INSERT INTO t_sum_map_bool_v0 SELECT sumMapState([1], [true]);
INSERT INTO t_sum_map_bool_v0 SELECT sumMapState([1], [false]);
SELECT sumMapMerge(s) FROM t_sum_map_bool_v0;
DROP TABLE t_sum_map_bool_v0;

DROP TABLE IF EXISTS t_sum_map_of_bool_v0;
CREATE TABLE t_sum_map_of_bool_v0 (s AggregateFunction(0, sumMap, Array(UInt8), Array(Bool))) ENGINE = TinyLog;
INSERT INTO t_sum_map_of_bool_v0 SELECT sumMapState([1], [true]);
INSERT INTO t_sum_map_of_bool_v0 SELECT sumMapState([1], [true]);
SELECT sumMapMerge(s) FROM t_sum_map_of_bool_v0;
DROP TABLE t_sum_map_of_bool_v0;

DROP TABLE IF EXISTS t_sum_map_overflow_bool_v0;
CREATE TABLE t_sum_map_overflow_bool_v0 (s AggregateFunction(0, sumMapWithOverflow, Array(UInt8), Array(Bool))) ENGINE = TinyLog;
INSERT INTO t_sum_map_overflow_bool_v0 SELECT sumMapWithOverflowState([1], [true]);
INSERT INTO t_sum_map_overflow_bool_v0 SELECT sumMapWithOverflowState([1], [false]);
SELECT sumMapWithOverflowMerge(s) FROM t_sum_map_overflow_bool_v0;
DROP TABLE t_sum_map_overflow_bool_v0;

DROP TABLE IF EXISTS t_sum_map_nullable_bool_v0;
CREATE TABLE t_sum_map_nullable_bool_v0 (s AggregateFunction(0, sumMap, Array(UInt8), Array(Nullable(Bool)))) ENGINE = TinyLog;
INSERT INTO t_sum_map_nullable_bool_v0 SELECT sumMapState([1], [true::Nullable(Bool)]);
INSERT INTO t_sum_map_nullable_bool_v0 SELECT sumMapState([1], [false::Nullable(Bool)]);
SELECT sumMapMerge(s) FROM t_sum_map_nullable_bool_v0;
DROP TABLE t_sum_map_nullable_bool_v0;

-- Bool as the map KEY across version-0 states must dedup (previously two 'true' keys stayed separate).
DROP TABLE IF EXISTS t_sum_map_bool_key_v0;
CREATE TABLE t_sum_map_bool_key_v0 (s AggregateFunction(0, sumMap, Array(Bool), Array(UInt32))) ENGINE = TinyLog;
INSERT INTO t_sum_map_bool_key_v0 SELECT sumMapState(CAST([true], 'Array(Bool)'), CAST([10], 'Array(UInt32)'));
INSERT INTO t_sum_map_bool_key_v0 SELECT sumMapState(CAST([true], 'Array(Bool)'), CAST([10], 'Array(UInt32)'));
SELECT sumMapMerge(s) FROM t_sum_map_bool_key_v0;
DROP TABLE t_sum_map_bool_key_v0;
