-- Tags: no-parallel-replicas

-- Regression test for a query-plan optimizer abort: the same text-index predicate placed in both
-- PREWHERE and WHERE, walked through a Merge -> Distributed -> MergeTree plan, made
-- optimizeDirectReadFromTextIndex register the synthetic __text_index_..._has_<hash> read column
-- twice for the same reading step (the step is optimized on more than one pass), aborting with
-- "Column ... already added for reading". See https://github.com/ClickHouse/ClickHouse/issues/110697

SET enable_analyzer = 1;
SET allow_experimental_full_text_index = 1;

DROP TABLE IF EXISTS logs;
DROP TABLE IF EXISTS logs_dist;
DROP TABLE IF EXISTS logs_merge;

CREATE TABLE logs
(
    ts DateTime,
    attributes Map(String, String),
    INDEX attributes_vals_idx mapValues(attributes) TYPE text(tokenizer = 'array') GRANULARITY 1,
    INDEX attributes_keys_idx mapKeys(attributes) TYPE text(tokenizer = 'array') GRANULARITY 1
)
ENGINE = MergeTree ORDER BY ts;

INSERT INTO logs VALUES (1, {'ip':'192.168.1.1'}), (2, {'ip':'10.0.0.1'});

CREATE TABLE logs_dist AS logs ENGINE = Distributed(test_shard_localhost, currentDatabase(), logs);
CREATE TABLE logs_merge AS logs ENGINE = Merge(currentDatabase(), '^logs_dist$');

-- The trigger: identical text-index predicate in PREWHERE and WHERE over the Merge table.
-- force_data_skipping_indices guarantees the text-index direct-read path engages.
SELECT count() FROM logs_merge
PREWHERE has(mapValues(attributes), toNullable('192.168.1.1'))
WHERE has(mapValues(attributes), toNullable('192.168.1.1'))
SETTINGS force_data_skipping_indices = 'attributes_vals_idx';

-- Direct read from the text index must remain engaged (optimization preserved, not disabled).
SELECT count() > 0 FROM
(
    EXPLAIN actions = 1
    SELECT count() FROM logs_merge
    PREWHERE has(mapValues(attributes), toNullable('192.168.1.1'))
    WHERE has(mapValues(attributes), toNullable('192.168.1.1'))
)
WHERE explain ILIKE '%__text_index_attributes_vals_idx_has%';

-- Non-equal predicates in PREWHERE and WHERE still return correct results.
SELECT count() FROM logs_merge
PREWHERE has(mapValues(attributes), toNullable('192.168.1.1'))
WHERE has(mapValues(attributes), toNullable('10.0.0.1'))
SETTINGS force_data_skipping_indices = 'attributes_vals_idx';

DROP TABLE logs_merge;
DROP TABLE logs_dist;
DROP TABLE logs;
