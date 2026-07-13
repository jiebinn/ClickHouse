-- Set operators (UNION / UNION ALL / UNION DISTINCT / EXCEPT / INTERSECT) inside a subquery must work
-- in the predicate and in the assignment expressions of DELETE and ALTER UPDATE mutations.
-- Regression test for https://github.com/ClickHouse/ClickHouse/issues/72853

DROP TABLE IF EXISTS t_union_mut;
CREATE TABLE t_union_mut (c0 Int) ENGINE = MergeTree ORDER BY tuple();

SET mutations_sync = 1;

-- DELETE with UNION DISTINCT in the predicate.
INSERT INTO t_union_mut VALUES (0), (1), (2);
DELETE FROM t_union_mut WHERE c0 IN ((SELECT 1) UNION DISTINCT (SELECT 0));
SELECT 'delete union distinct', arraySort(groupArray(c0)) FROM t_union_mut;
TRUNCATE TABLE t_union_mut;

-- DELETE with UNION ALL in the predicate.
INSERT INTO t_union_mut VALUES (0), (1), (2);
DELETE FROM t_union_mut WHERE c0 IN ((SELECT 1) UNION ALL (SELECT 0));
SELECT 'delete union all', arraySort(groupArray(c0)) FROM t_union_mut;
TRUNCATE TABLE t_union_mut;

-- DELETE with EXCEPT in the predicate.
INSERT INTO t_union_mut VALUES (0), (1), (2);
DELETE FROM t_union_mut WHERE c0 IN ((SELECT 1) EXCEPT (SELECT 0));
SELECT 'delete except', arraySort(groupArray(c0)) FROM t_union_mut;
TRUNCATE TABLE t_union_mut;

-- DELETE with INTERSECT in the predicate.
INSERT INTO t_union_mut VALUES (0), (1), (2);
DELETE FROM t_union_mut WHERE c0 IN ((SELECT 1) INTERSECT (SELECT 1));
SELECT 'delete intersect', arraySort(groupArray(c0)) FROM t_union_mut;
TRUNCATE TABLE t_union_mut;

-- ALTER UPDATE with UNION in the predicate.
INSERT INTO t_union_mut VALUES (0), (1), (2);
ALTER TABLE t_union_mut UPDATE c0 = 9 WHERE c0 IN ((SELECT 1) UNION DISTINCT (SELECT 0));
SELECT 'update union predicate', arraySort(groupArray(c0)) FROM t_union_mut;
TRUNCATE TABLE t_union_mut;

-- ALTER UPDATE with UNION in the assignment expression.
INSERT INTO t_union_mut VALUES (0), (1), (2);
ALTER TABLE t_union_mut UPDATE c0 = (SELECT sum(x) FROM ((SELECT 1 AS x) UNION DISTINCT (SELECT 2 AS x))) WHERE c0 = 1;
SELECT 'update union assignment', arraySort(groupArray(c0)) FROM t_union_mut;
TRUNCATE TABLE t_union_mut;

-- The same must also work with the old analyzer.
SET enable_analyzer = 0;
INSERT INTO t_union_mut VALUES (0), (1), (2);
DELETE FROM t_union_mut WHERE c0 IN ((SELECT 1) UNION DISTINCT (SELECT 0));
SELECT 'delete union distinct, old analyzer', arraySort(groupArray(c0)) FROM t_union_mut;

DROP TABLE t_union_mut;
