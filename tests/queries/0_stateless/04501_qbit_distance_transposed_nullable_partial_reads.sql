-- A Nullable(QBit) column must get the same partial-read optimization as a plain QBit column: the transposed distance
-- functions read only the requested bit-plane subcolumns instead of the whole Nullable(QBit) column. The bit planes are
-- read as Nullable(FixedString), so the per-row null map reaches the function through its default Nullable handling and
-- the result stays Nullable(Float64), identical to the unoptimised full-column read.
-- Companion to 03375 / 04489 (plain QBit partial reads) and 03374 (Nullable(QBit) storage).

SET explain_query_plan_default = 'legacy';
SET enable_analyzer = 1;

DROP TABLE IF EXISTS qbit_nullable;
CREATE TABLE qbit_nullable (id UInt32, vec Nullable(QBit(BFloat16, 16))) ENGINE = Memory;

SET optimize_qbit_distance_function_reads = true;

SELECT '-- optimization enabled on Nullable(QBit): reads Nullable(FixedString) planes vec.1 .. vec.4, result Nullable(Float64)';
EXPLAIN actions=1
WITH arrayMap(i -> i * 2, range(16)) AS reference_vec
SELECT id, L2DistanceTransposed(vec, reference_vec, 4) AS dist FROM qbit_nullable;

SELECT '-- optimization disabled: reads the whole Nullable(QBit) column';
EXPLAIN actions=1
WITH arrayMap(i -> i * 2, range(16)) AS reference_vec
SELECT id, L2DistanceTransposed(vec, reference_vec, 4) AS dist FROM qbit_nullable SETTINGS optimize_qbit_distance_function_reads = false;

DROP TABLE qbit_nullable;


-- Correctness: the optimization must not change the values, and NULL rows must stay NULL. For each function the two
-- blocks below (optimization on, then off) must be identical.

DROP TABLE IF EXISTS qbit_ns;
CREATE TABLE qbit_ns (id UInt32, vec Nullable(QBit(Float32, 4))) ENGINE = Memory;
INSERT INTO qbit_ns VALUES (1, [1, 2, 3, 4]), (2, NULL), (3, [0.5, 0.5, 0.5, 0.5]), (4, NULL), (5, [9, 8, 7, 6]);

SELECT '-- non-strided L2DistanceTransposed, precision 8, optimization on then off';
SELECT id, round(L2DistanceTransposed(vec, [1., 2., 3., 4.], 8), 6) FROM qbit_ns ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 1;
SELECT id, round(L2DistanceTransposed(vec, [1., 2., 3., 4.], 8), 6) FROM qbit_ns ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 0;

SELECT '-- non-strided cosineDistanceTransposed, precision 16, optimization on then off';
SELECT id, round(cosineDistanceTransposed(vec, [1., 2., 3., 4.], 16), 6) FROM qbit_ns ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 1;
SELECT id, round(cosineDistanceTransposed(vec, [1., 2., 3., 4.], 16), 6) FROM qbit_ns ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 0;

SELECT '-- non-strided dotProductTransposed, full precision 32, optimization on then off';
SELECT id, round(dotProductTransposed(vec, [1., 2., 3., 4.], 32), 6) FROM qbit_ns ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 1;
SELECT id, round(dotProductTransposed(vec, [1., 2., 3., 4.], 32), 6) FROM qbit_ns ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 0;

-- The result type must stay Nullable(Float64) whether or not the optimization fires.
SELECT '-- result type is Nullable(Float64) with the optimization on and off';
SELECT DISTINCT toTypeName(L2DistanceTransposed(vec, [1., 2., 3., 4.], 8)) FROM qbit_ns SETTINGS optimize_qbit_distance_function_reads = 1;
SELECT DISTINCT toTypeName(L2DistanceTransposed(vec, [1., 2., 3., 4.], 8)) FROM qbit_ns SETTINGS optimize_qbit_distance_function_reads = 0;

DROP TABLE qbit_ns;


-- The strided QBit form (Matryoshka-style partial-dimension reads) must behave the same way for Nullable columns.

DROP TABLE IF EXISTS qbit_strided;
CREATE TABLE qbit_strided (id UInt32, vec Nullable(QBit(Float32, 16, 8))) ENGINE = Memory;
INSERT INTO qbit_strided VALUES (1, range(16)), (2, NULL), (3, arrayMap(x -> 0.5, range(16))), (4, NULL);

SELECT '-- strided dotProductTransposed, used_dims 8, precision 16, optimization on then off';
SELECT id, round(dotProductTransposed(vec, range(8)::Array(Float32), 16, 8), 6) FROM qbit_strided ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 1;
SELECT id, round(dotProductTransposed(vec, range(8)::Array(Float32), 16, 8), 6) FROM qbit_strided ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 0;

SELECT '-- strided L2DistanceTransposed, all dimensions, precision 8, optimization on then off';
SELECT id, round(L2DistanceTransposed(vec, range(16)::Array(Float32), 8), 6) FROM qbit_strided ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 1;
SELECT id, round(L2DistanceTransposed(vec, range(16)::Array(Float32), 8), 6) FROM qbit_strided ORDER BY id SETTINGS optimize_qbit_distance_function_reads = 0;

DROP TABLE qbit_strided;
