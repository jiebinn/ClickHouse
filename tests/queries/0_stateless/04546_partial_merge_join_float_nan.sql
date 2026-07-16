-- Test compareTrackAt with Float64/NaN/Inf keys in partial_merge join.
-- ColumnVector<Float64>::compareTrackAt (ColumnVector.h:157) uses
-- CompareHelper<Float64>::compare with NaN handling (nan_direction_hint).
-- Exercises multi-row skipping on Float keys and NaN=NaN matching.

SET join_algorithm = 'partial_merge';

DROP TABLE IF EXISTS t_04546_left;
DROP TABLE IF EXISTS t_04546_right;

CREATE TABLE t_04546_left (key Float64, val String) ENGINE = MergeTree() ORDER BY key;
CREATE TABLE t_04546_right (key Float64, val String) ENGINE = MergeTree() ORDER BY key;

-- Left: -inf, 1, 2, 3, 7, 8, 9, inf, nan (sorted)
-- Right: 0, 3, 5, 10, nan (sorted)
INSERT INTO t_04546_left VALUES (-inf, 'L-inf'), (1, 'L1'), (2, 'L2'), (3, 'L3'), (7, 'L7'), (8, 'L8'), (9, 'L9'), (inf, 'Linf'), (nan, 'Lnan');
INSERT INTO t_04546_right VALUES (0, 'R0'), (3, 'R3'), (5, 'R5'), (10, 'R10'), (nan, 'Rnan');

-- INNER JOIN: matching keys 3 and NaN.
-- isNaN() prefix pins NaN rows last so the ORDER BY is deterministic regardless
-- of the sort's nan_direction_hint (which CI randomizes).
SELECT l.key, l.val, r.val
FROM t_04546_left l INNER JOIN t_04546_right r ON l.key = r.key
ORDER BY isNaN(l.key), l.key, l.val;

-- LEFT JOIN: all left rows, right values for matches
SELECT l.key, l.val, r.val
FROM t_04546_left l LEFT JOIN t_04546_right r ON l.key = r.key
ORDER BY isNaN(l.key), l.key, l.val;

-- NaN matches NaN in merge join (compareAt returns 0 for NaN=NaN)
SELECT count() FROM t_04546_left l INNER JOIN t_04546_right r ON l.key = r.key WHERE isNaN(l.key);

DROP TABLE t_04546_left;
DROP TABLE t_04546_right;
