#!/usr/bin/env bash

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

# `SHOW CREATE DICTIONARY` is authorized with the `SHOW DICTIONARIES` grant, which does not
# imply `SHOW TABLES`. The "Maybe you meant ...?" hint for a missing dictionary must work for
# a user with dictionary-only visibility, and it must suggest only dictionaries to such a
# user - never tables, whose existence the user is not allowed to see.
#
# The test database name is dynamic, so it is normalized to `{db}` in the output.
DB="${CLICKHOUSE_DATABASE}"
USER="user_${CLICKHOUSE_TEST_UNIQUE_NAME}"

${CLICKHOUSE_CLIENT} -q "CREATE TABLE ${DB}.secret_target (key UInt64, val UInt64) ENGINE = Memory"
${CLICKHOUSE_CLIENT} -q "CREATE DICTIONARY ${DB}.dict_target (key UInt64 DEFAULT 0, val UInt64 DEFAULT 0) PRIMARY KEY key SOURCE(CLICKHOUSE(TABLE 'secret_target' DB '${DB}')) LIFETIME(MIN 0 MAX 0) LAYOUT(FLAT())"

${CLICKHOUSE_CLIENT} -q "DROP USER IF EXISTS ${USER}"
${CLICKHOUSE_CLIENT} -q "CREATE USER ${USER}"
${CLICKHOUSE_CLIENT} -q "GRANT SHOW DICTIONARIES ON ${DB}.* TO ${USER}"

# The dictionary-only user gets a hint about the similarly-named dictionary.
${CLICKHOUSE_CLIENT} --user "${USER}" -q "SHOW CREATE DICTIONARY ${DB}.dict_targe" 2>&1 \
    | grep -oF -m1 "Maybe you meant ${DB}.dict_target?" | sed "s/${DB}/{db}/g"

# But tables stay invisible to that user: a typo close to the table's name must produce
# no hint at all (0 matching lines), otherwise the hint would leak the table's existence.
${CLICKHOUSE_CLIENT} --user "${USER}" -q "SHOW CREATE DICTIONARY ${DB}.secret_targe" 2>&1 \
    | grep -c -F "Maybe you meant" || true

# Sanity check: for a user with full visibility the same typo does produce the table hint.
${CLICKHOUSE_CLIENT} -q "SHOW CREATE DICTIONARY ${DB}.secret_targe" 2>&1 \
    | grep -oF -m1 "Maybe you meant ${DB}.secret_target?" | sed "s/${DB}/{db}/g"

${CLICKHOUSE_CLIENT} -q "DROP USER ${USER}"
