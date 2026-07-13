-- Tests the Enum dictionary optimization of string search functions (issue #73114):
-- for an Enum haystack and a constant needle, the search runs over the distinct enum names only
-- and the results are mapped back per row. This must produce exactly the same results as running
-- the search over the equivalent String column (toString(c)).

DROP TABLE IF EXISTS t_es8;
DROP TABLE IF EXISTS t_es16;
DROP TABLE IF EXISTS t_es8_few;

-- Enum8 with names of different sizes (including empty and one with spaces) and negative/zero/positive values.
CREATE TABLE t_es8 (c Enum8('' = -128, 'a' = -5, 'A' = 0, 'AB' = 1, 'aBc' = 2, 'ABCD' = 3, 'xAyz' = 100, 'Foo A Bar' = 127), s UInt8)
ENGINE = MergeTree ORDER BY tuple();
INSERT INTO t_es8
SELECT ['', 'a', 'A', 'AB', 'aBc', 'ABCD', 'xAyz', 'Foo A Bar'][(number % 8) + 1], number % 4
FROM numbers(1000);

-- Enum16 spanning a wide Int16 range (exercises the large plain lookup array).
CREATE TABLE t_es16 (c Enum16('' = -30000, 'a' = -1, 'A' = 0, 'AB' = 7, 'aBc' = 300, 'ABCD' = 1000, 'xAyz' = 5000, 'Foo A Bar' = 30000))
ENGINE = MergeTree ORDER BY tuple();
INSERT INTO t_es16
SELECT ['', 'a', 'A', 'AB', 'aBc', 'ABCD', 'xAyz', 'Foo A Bar'][(number % 8) + 1]
FROM numbers(1000);

-- Fewer rows than distinct enum names: the optimization is not applied (non-transform path).
CREATE TABLE t_es8_few (c Enum8('' = -128, 'a' = -5, 'A' = 0, 'AB' = 1, 'aBc' = 2, 'ABCD' = 3, 'xAyz' = 100, 'Foo A Bar' = 127))
ENGINE = MergeTree ORDER BY tuple();
INSERT INTO t_es8_few SELECT ['A', 'aBc', ''][(number % 3) + 1] FROM numbers(3);

SELECT 'Enum8 transform path';
SELECT 'like', sum(like(c, '%A%') != like(toString(c), '%A%')) FROM t_es8;
SELECT 'like ESCAPE', sum((c LIKE 'A%' ESCAPE '!') != (toString(c) LIKE 'A%' ESCAPE '!')) FROM t_es8;
SELECT 'notLike', sum(notLike(c, '%A%') != notLike(toString(c), '%A%')) FROM t_es8;
SELECT 'ilike', sum(ilike(c, '%a%') != ilike(toString(c), '%a%')) FROM t_es8;
SELECT 'match', sum(match(c, 'A') != match(toString(c), 'A')) FROM t_es8;
SELECT 'position', sum(position(c, 'A') != position(toString(c), 'A')) FROM t_es8;
SELECT 'positionCaseInsensitive', sum(positionCaseInsensitive(c, 'a') != positionCaseInsensitive(toString(c), 'a')) FROM t_es8;
SELECT 'positionUTF8', sum(positionUTF8(c, 'A') != positionUTF8(toString(c), 'A')) FROM t_es8;
SELECT 'countSubstrings', sum(countSubstrings(c, 'A') != countSubstrings(toString(c), 'A')) FROM t_es8;
SELECT 'countSubstringsCaseInsensitive', sum(countSubstringsCaseInsensitive(c, 'a') != countSubstringsCaseInsensitive(toString(c), 'a')) FROM t_es8;
SELECT 'hasToken', sum(hasToken(c, 'A') != hasToken(toString(c), 'A')) FROM t_es8;
SELECT 'hasTokenCaseInsensitive', sum(hasTokenCaseInsensitive(c, 'a') != hasTokenCaseInsensitive(toString(c), 'a')) FROM t_es8;
-- *OrNull variant exercises the Null execution error policy (the null map is mapped back too).
SELECT 'hasTokenOrNull', sum(hasTokenOrNull(c, 'A') != hasTokenOrNull(toString(c), 'A')) FROM t_es8;

SELECT 'Enum8 fallback (non-const needle)';
SELECT 'position non-const needle', sum(position(c, substring(toString(c), 1, 1)) != position(toString(c), substring(toString(c), 1, 1))) FROM t_es8;
SELECT 'Enum8 fallback (per-row start position)';
SELECT 'position per-row start', sum(position(c, 'A', s + 1) != position(toString(c), 'A', s + 1)) FROM t_es8;
SELECT 'position const start', sum(position(c, 'A', 1) != position(toString(c), 'A', 1)) FROM t_es8;

SELECT 'Enum16 transform path (wide range)';
SELECT 'like', sum(like(c, '%A%') != like(toString(c), '%A%')) FROM t_es16;
SELECT 'position', sum(position(c, 'A') != position(toString(c), 'A')) FROM t_es16;
SELECT 'match', sum(match(c, 'A') != match(toString(c), 'A')) FROM t_es16;
SELECT 'countSubstrings', sum(countSubstrings(c, 'A') != countSubstrings(toString(c), 'A')) FROM t_es16;

SELECT 'Enum8 non-transform path (few rows)';
SELECT 'like', sum(like(c, '%A%') != like(toString(c), '%A%')) FROM t_es8_few;
SELECT 'position', sum(position(c, 'A') != position(toString(c), 'A')) FROM t_es8_few;

DROP TABLE t_es8;
DROP TABLE t_es16;
DROP TABLE t_es8_few;
