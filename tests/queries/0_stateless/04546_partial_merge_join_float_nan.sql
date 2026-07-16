-- Test compareTrackAt with Float64 / +-Inf keys in partial_merge join.
-- ColumnVector<Float64>::compareTrackAt (ColumnVector.h:157) uses
-- CompareHelper<Float64>::compare, which drives multi-row skipping over the
-- sorted Float keys. NaN keys must follow equi-join semantics (NaN != NaN),
-- so a NaN key never produces a match; only finite keys and +-Inf match.

SET join_algorithm = 'partial_merge';

DROP TABLE IF EXISTS t_04546_left;
DROP TABLE IF EXISTS t_04546_right;

CREATE TABLE t_04546_left (key Float64, val String) ENGINE = MergeTree() ORDER BY key;
CREATE TABLE t_04546_right (key Float64, val String) ENGINE = MergeTree() ORDER BY key;

-- Left:  -inf, 1, 2, 3, 7, 8, 9, inf, nan
-- Right:  0, 3, 5, 10, inf
-- Finite match: 3. +-Inf: +inf matches, -inf does not. NaN must never match.
INSERT INTO t_04546_left VALUES (-inf, 'L-inf'), (1, 'L1'), (2, 'L2'), (3, 'L3'), (7, 'L7'), (8, 'L8'), (9, 'L9'), (inf, 'Linf'), (nan, 'Lnan');
INSERT INTO t_04546_right VALUES (0, 'R0'), (3, 'R3'), (5, 'R5'), (10, 'R10'), (inf, 'Rinf');

-- INNER JOIN: matching keys are 3 and +inf; NaN does not match.
-- isNaN() prefix pins any NaN row last so the ORDER BY is deterministic
-- regardless of the sort's nan_direction_hint (which CI randomizes).
SELECT l.key, l.val, r.val
FROM t_04546_left l INNER JOIN t_04546_right r ON l.key = r.key
ORDER BY isNaN(l.key), l.key, l.val;

-- LEFT JOIN: all left rows, right values for matches (3 and +inf); NaN unmatched.
SELECT l.key, l.val, r.val
FROM t_04546_left l LEFT JOIN t_04546_right r ON l.key = r.key
ORDER BY isNaN(l.key), l.key, l.val;

-- NaN follows equi-join semantics: NaN != NaN, so a NaN key never matches (expect 0).
SELECT count() FROM t_04546_left l INNER JOIN t_04546_right r ON l.key = r.key WHERE isNaN(l.key);

DROP TABLE t_04546_left;
DROP TABLE t_04546_right;
