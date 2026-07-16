#!/usr/bin/env bash

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

# `SHOW CREATE TABLE` for a non-existent table must suggest a similarly-named table,
# just like `SELECT` does. Previously it threw the bare `UNKNOWN_TABLE` /
# `CANNOT_GET_CREATE_TABLE_QUERY` error with no "Maybe you meant ...?" hint.
#
# The test database name is dynamic, so it is normalized to `{db}` in the output.
DB="${CLICKHOUSE_DATABASE}"

# Atomic database (the default): table metadata lives on disk, so a missing table is
# reported with `CANNOT_GET_CREATE_TABLE_QUERY`.
${CLICKHOUSE_CLIENT} -q "CREATE TABLE ${DB}.some_target_table (x UInt8) ENGINE = MergeTree ORDER BY x"

# grep -m1: with --send_logs_level the server can also echo the exception as a log event,
# so the hint may appear more than once; take a single match to stay deterministic.
${CLICKHOUSE_CLIENT} -q "SHOW CREATE TABLE ${DB}.some_target_tabl" 2>&1 \
    | grep -oF -m1 "Maybe you meant ${DB}.some_target_table?" | sed "s/${DB}/{db}/g"

# The unqualified form (resolved against the current database, which is ${CLICKHOUSE_DATABASE})
# behaves the same.
${CLICKHOUSE_CLIENT} -q "SHOW CREATE TABLE some_target_tabl" 2>&1 \
    | grep -oF -m1 "Maybe you meant ${DB}.some_target_table?" | sed "s/${DB}/{db}/g"

# `SHOW CREATE VIEW` and `SHOW CREATE DICTIONARY` go through the same code path.
${CLICKHOUSE_CLIENT} -q "SHOW CREATE VIEW ${DB}.some_target_tabl" 2>&1 \
    | grep -oF -m1 "Maybe you meant ${DB}.some_target_table?" | sed "s/${DB}/{db}/g"

# Memory-engine database: a missing table is reported with `UNKNOWN_TABLE` (a different
# error code), which must carry the hint as well.
${CLICKHOUSE_CLIENT} -q "CREATE DATABASE ${DB}_mem ENGINE = Memory"
${CLICKHOUSE_CLIENT} -q "CREATE TABLE ${DB}_mem.mem_target (x UInt8) ENGINE = Memory"
${CLICKHOUSE_CLIENT} -q "SHOW CREATE TABLE ${DB}_mem.mem_targe" 2>&1 \
    | grep -oF -m1 "Maybe you meant ${DB}_mem.mem_target?" | sed "s/${DB}/{db}/g"
${CLICKHOUSE_CLIENT} -q "DROP DATABASE ${DB}_mem"

# When no similarly-named table exists, there must be no hint (0 matching lines).
${CLICKHOUSE_CLIENT} -q "SHOW CREATE TABLE ${DB}.nothing_is_close_to_this_xyz" 2>&1 \
    | grep -c -F "Maybe you meant" || true
