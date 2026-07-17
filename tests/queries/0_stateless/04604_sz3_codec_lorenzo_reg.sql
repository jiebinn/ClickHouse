-- Tags: no-fasttest, no-random-settings
-- no-fasttest: needs sz3 library
-- no-random-settings: a randomized `max_compress_block_size` that is not a multiple of the float width
-- is rejected by the SZ3 codec at write time

-- Regression test for the SZ3 `ALGO_LORENZO_REG` decompression path.
-- `RegressionPredictor::load` (used only by `ALGO_LORENZO_REG`) decremented `remaining_length` by the
-- uncompressed regression-coefficient count after the Huffman `decode`, instead of the bytes `decode`
-- actually consumed. The understated bound then made the following `encoder.load` reject valid data with
-- "SZ3 Huffman: encoded length exceeds compressed buffer", so a column written with
-- `CODEC(SZ3('ALGO_LORENZO_REG', ...))` could be inserted but failed every later read with CORRUPTED_DATA.

SET allow_experimental_codecs = 1;

DROP TABLE IF EXISTS tab_sz3_lorenzo_reg;

CREATE TABLE tab_sz3_lorenzo_reg (
    key  UInt64,
    orig Float64,
    val  Float64 CODEC(SZ3('ALGO_LORENZO_REG', 'ABS', 0.01))
) ENGINE = MergeTree ORDER BY key;

-- A smooth signal with a linear trend, so the regression predictor is actually selected for the blocks
-- (as opposed to the lossless fallback that SZ3 uses for tiny or incompressible inputs).
-- Two parts, so the OPTIMIZE FINAL below actually merges and re-reads the SZ3 column during the merge.
INSERT INTO tab_sz3_lorenzo_reg
SELECT number, number * 0.5 + sin(number * 0.1), number * 0.5 + sin(number * 0.1)
FROM numbers(50000);
INSERT INTO tab_sz3_lorenzo_reg
SELECT number, number * 0.5 + sin(number * 0.1), number * 0.5 + sin(number * 0.1)
FROM numbers(50000, 50000);

SELECT 'ALGO_LORENZO_REG data round-trips within the error bound (the read failed with CORRUPTED_DATA before the fix)';
SELECT count() = 100000, max(abs(orig - val)) <= 0.011 FROM tab_sz3_lorenzo_reg;

SELECT 'The same holds after a merge recompresses the data';
OPTIMIZE TABLE tab_sz3_lorenzo_reg FINAL;
-- A merge decompresses and recompresses the column through the lossy codec, so every pass adds up to
-- the ABS bound (0.01) on top of the previous one. The initial write, a possible concurrent background
-- merge of the two parts, and the OPTIMIZE FINAL rewrite give at most three lossy passes.
SELECT count() = 100000, max(abs(orig - val)) <= 0.031 FROM tab_sz3_lorenzo_reg;

DROP TABLE tab_sz3_lorenzo_reg;
