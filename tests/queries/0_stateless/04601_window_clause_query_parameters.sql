-- Query parameters inside the definitions of a named WINDOW clause were invisible to AST visitors,
-- because `ParserWindowList` did not attach the parsed definition to the children of `ASTWindowListElement`.
-- They were never substituted, and an unset Identifier parameter produced an `ASTIdentifier`
-- with an empty name part, causing the exception `Logical error: '!part.empty()'`
-- during the `EXPLAIN SYNTAX` round trip in debug builds.

SET enable_analyzer = 1;

DROP TABLE IF EXISTS t_window_params;
CREATE TABLE t_window_params (p UInt8, x UInt8) ENGINE = MergeTree ORDER BY tuple();
INSERT INTO t_window_params VALUES (0, 1), (0, 2), (1, 10);

SET param_part_col = 'p';
SET param_str = 'abc';

-- A String parameter in PARTITION BY of a named window
SELECT count() OVER w AS c FROM t_window_params WINDOW w AS (PARTITION BY {str:String}) ORDER BY c;

-- An Identifier parameter in PARTITION BY of a named window
SELECT sum(x) OVER w AS s FROM t_window_params WINDOW w AS (PARTITION BY {part_col:Identifier}) ORDER BY s;

-- Round trip: parameters in an unused window definition must be substituted as well
EXPLAIN SYNTAX query_tree_passes = 1, oneline = 1 SELECT 1 WINDOW w1 AS (), w2 AS (PARTITION BY {part_col:Identifier}, {str:String} ORDER BY {part_col:Identifier});

-- An unset parameter inside a WINDOW definition is an error (previously it was silently kept unsubstituted)
SELECT 1 WINDOW w_bad AS (PARTITION BY {unset_window_param:String}); -- { serverError UNKNOWN_QUERY_PARAMETER }
SELECT 1 WINDOW w_bad AS (PARTITION BY {unset_window_param:Identifier}.c); -- { serverError UNKNOWN_QUERY_PARAMETER }
EXPLAIN SYNTAX query_tree_passes = 1, oneline = 1 SELECT 1 WINDOW w1 AS (), w2 AS (PARTITION BY {unset_window_param:Identifier}.c); -- { serverError UNKNOWN_QUERY_PARAMETER }

DROP TABLE t_window_params;
