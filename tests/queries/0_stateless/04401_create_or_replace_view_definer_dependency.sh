#!/usr/bin/env bash

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

user="user04401_${CLICKHOUSE_DATABASE}"
db=${CLICKHOUSE_DATABASE}

${CLICKHOUSE_CLIENT} --query "DROP USER IF EXISTS $user"
${CLICKHOUSE_CLIENT} --query "CREATE USER $user"

${CLICKHOUSE_CLIENT} --query "CREATE TABLE $db.src (x Int64) ENGINE = MergeTree() ORDER BY x"

# CREATE OR REPLACE VIEW builds the view under an internal _tmp_replace_* name, then renames it
# to the target. The definer dependency must move with the rename, not linger on the temp name.
${CLICKHOUSE_CLIENT} --query "CREATE OR REPLACE VIEW $db.v DEFINER = $user SQL SECURITY DEFINER AS SELECT x FROM $db.src"

# The user is a definer only of $db.v; dropping $db.v must free the user for DROP USER.
${CLICKHOUSE_CLIENT} --query "DROP VIEW $db.v"
${CLICKHOUSE_CLIENT} --query "DROP USER $user" && echo "OK"

${CLICKHOUSE_CLIENT} --query "DROP TABLE $db.src"
