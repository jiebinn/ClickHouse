-- Regression test for issue #80893: a constant LowCardinality aggregation key
-- lost its constness in the distributed merging-aggregator and arrived as
-- Const(LowCardinality(...)), aborting with a LOGICAL_ERROR in
-- HashMethodSingleLowCardinalityColumn. Must not crash.

SET allow_suspicious_low_cardinality_types = 1;

DROP TABLE IF EXISTS t0;
CREATE TABLE t0 (c0 LowCardinality(Bool)) ENGINE = Memory;
INSERT INTO t0 VALUES (TRUE);

SELECT c0 FROM remote('127.0.0.1', currentDatabase(), 't0') AS tx GROUP BY ALL QUALIFY c0;

DROP TABLE t0;
