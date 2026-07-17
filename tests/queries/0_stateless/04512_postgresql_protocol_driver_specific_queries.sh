#!/usr/bin/env bash
# Tags: no-fasttest
# Tag no-fasttest: Requires postgresql-client

# Some PostgreSQL drivers (e.g. Skunk) send session-management commands such as
# `RESET ALL` and `UNLISTEN *` during connection setup or cleanup. ClickHouse has
# no equivalent for these, but it must accept them as no-ops instead of failing
# with a syntax error, so that such drivers can connect over the wire protocol.
# See https://github.com/ClickHouse/ClickHouse/issues/12476

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

${CLICKHOUSE_CLIENT} -q "
DROP USER IF EXISTS postgresql_user_04512;
CREATE USER postgresql_user_04512 HOST IP '127.0.0.1' IDENTIFIED WITH no_password;
"

psql --host localhost --port "${CLICKHOUSE_PORT_POSTGRESQL}" "${CLICKHOUSE_DATABASE}" --user postgresql_user_04512 --no-align 2>&1 <<'EOF'
RESET ALL;
RESET search_path;
UNLISTEN *;
UNLISTEN some_channel;
LISTEN some_channel;
NOTIFY some_channel;
DISCARD ALL;
DISCARD PLANS;
DISCARD SEQUENCES;
DISCARD TEMP;
DISCARD TEMPORARY;
SELECT 1 AS connection_is_still_usable;
EOF

${CLICKHOUSE_CLIENT} -q "DROP USER IF EXISTS postgresql_user_04512;"
