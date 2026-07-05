-- Tags: no-random-settings, no-random-merge-tree-settings, no-parallel-replicas

-- The implicit min-max skip index over the persistent virtual columns _block_number / _block_offset
-- must not prevent column ALTERs (RENAME / ADD / MODIFY) on an unrelated column.
-- Before the fix, ALTER validation re-resolved the implicit index against physical columns only,
-- so _block_number could not be resolved and the ALTER threw UNKNOWN_IDENTIFIER.

SET enable_analyzer = 1;
SET use_skip_indexes = 1;

-- RENAME COLUMN, index over _block_number
DROP TABLE IF EXISTS t_imv_alter;
CREATE TABLE t_imv_alter (date1 Date, value1 String)
ENGINE = MergeTree ORDER BY tuple()
SETTINGS enable_block_number_column = 1, add_minmax_index_for_block_number_column = 1, index_granularity = 1;
INSERT INTO t_imv_alter SELECT toDate('2018-10-01') + number % 3, toString(number) FROM numbers(9);
ALTER TABLE t_imv_alter RENAME COLUMN date1 TO renamed_date1;
SELECT name, expr FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = 't_imv_alter' ORDER BY name;
SELECT count() FROM t_imv_alter;

-- RENAME COLUMN, index over _block_offset
DROP TABLE IF EXISTS t_imv_alter_off;
CREATE TABLE t_imv_alter_off (date1 Date, value1 String)
ENGINE = MergeTree ORDER BY tuple()
SETTINGS enable_block_offset_column = 1, add_minmax_index_for_block_offset_column = 1, index_granularity = 1;
INSERT INTO t_imv_alter_off SELECT toDate('2018-10-01') + number % 3, toString(number) FROM numbers(9);
ALTER TABLE t_imv_alter_off RENAME COLUMN date1 TO renamed_date1;
SELECT name, expr FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = 't_imv_alter_off' ORDER BY name;

-- ADD / MODIFY COLUMN, both indices enabled; verify the index still prunes granules afterwards
DROP TABLE IF EXISTS t_imv_alter_both;
CREATE TABLE t_imv_alter_both (a UInt64)
ENGINE = MergeTree ORDER BY a
SETTINGS enable_block_number_column = 1, enable_block_offset_column = 1,
         add_minmax_index_for_block_number_column = 1, add_minmax_index_for_block_offset_column = 1,
         index_granularity = 1;
SYSTEM STOP MERGES t_imv_alter_both;
INSERT INTO t_imv_alter_both SELECT number FROM numbers(10);
INSERT INTO t_imv_alter_both SELECT number + 10 FROM numbers(10);
SYSTEM START MERGES t_imv_alter_both;
OPTIMIZE TABLE t_imv_alter_both FINAL;

ALTER TABLE t_imv_alter_both ADD COLUMN b Int32;
ALTER TABLE t_imv_alter_both MODIFY COLUMN a UInt64 CODEC(ZSTD);
ALTER TABLE t_imv_alter_both RENAME COLUMN b TO b2;
SELECT name, expr FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = 't_imv_alter_both' ORDER BY name;
-- The implicit indices must still PRUNE granules after the ALTERs, not merely stay attached.
-- EXPLAIN indexes = 1 prints the index name even when it reads every granule (Granules: N/N),
-- so a name match alone would pass a regression that stopped pruning. Assert on the Granules line.
-- 20 rows in two blocks, index_granularity = 1: _block_number = 1 matches one block (10/20 granules read),
-- _block_offset = 0 matches one row per block (2/20). A stopped-pruning regression reads 20/20 and fails these.
SELECT count() > 0 FROM (EXPLAIN indexes = 1 SELECT * FROM t_imv_alter_both WHERE _block_number = 1) WHERE explain ILIKE '%Granules: 10/20%';
SELECT count() > 0 FROM (EXPLAIN indexes = 1 SELECT * FROM t_imv_alter_both WHERE _block_offset = 0) WHERE explain ILIKE '%Granules: 2/20%';

DROP TABLE IF EXISTS t_imv_alter;
DROP TABLE IF EXISTS t_imv_alter_off;
DROP TABLE IF EXISTS t_imv_alter_both;
