-- Tags: no-parallel

DROP TABLE IF EXISTS src_04538;
DROP TABLE IF EXISTS dst_04538;
DROP TABLE IF EXISTS mv_04538;
DROP TABLE IF EXISTS mv_04538_dup;

CREATE TABLE src_04538 (x UInt8) ENGINE = Memory;
CREATE TABLE dst_04538 (x UInt8) ENGINE = Memory;

-- Bug: previously failed to parse because COMMENT was consumed as an implicit alias for src_04538
CREATE MATERIALIZED VIEW mv_04538 TO dst_04538 AS SELECT x FROM src_04538 COMMENT 'trailing comment test';

SELECT comment FROM system.tables WHERE name = 'mv_04538' AND database = currentDatabase();

DROP TABLE mv_04538;

-- Explicit alias form should still work
CREATE MATERIALIZED VIEW mv_04538 TO dst_04538 AS SELECT x FROM src_04538 AS s COMMENT 'explicit alias comment';

SELECT comment FROM system.tables WHERE name = 'mv_04538' AND database = currentDatabase();

DROP TABLE mv_04538;

-- Specifying a comment in both the pre-AS and post-SELECT position should throw a clear error
CREATE MATERIALIZED VIEW mv_04538_dup (x UInt8) ENGINE = Memory COMMENT 'pre-as comment' AS SELECT x FROM src_04538 COMMENT 'post-select comment'; -- { serverError SYNTAX_ERROR }

-- Explicit AS alias with the word "comment" should still work (not affected by restricted_keywords fix)
SELECT 1 AS comment;

DROP TABLE src_04538;
DROP TABLE dst_04538;
