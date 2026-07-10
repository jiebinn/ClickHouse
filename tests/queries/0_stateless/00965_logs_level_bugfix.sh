#!/usr/bin/env bash

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

# Match the bracketed priority marker anywhere: the log prefix omits host_name/query_id
# when empty, so the marker column is not fixed and a positional awk '{print $8}' is fragile.
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="trace" --query="SELECT 1" 2>&1 | grep -oE -m1 '<Trace>'
echo "."
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="debug" --query="SELECT 1" 2>&1 | grep -oE -m1 '<Debug>'
echo "."
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="information" --query="SELECT 1" 2>&1 | grep -oE -m1 '<Information>'
echo "."
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="error" --query="SELECT throwIf(1)" 2>&1 | grep -oE -m1 '<Error>'
echo "-"
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="debug" --query="SELECT 1" 2>&1 | grep -oE -m1 '<Trace>'
echo "."
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="information" --query="SELECT 1" 2>&1 | grep -oE -m1 '<Debug>|<Trace>'
echo "."
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="error" --query="SELECT throwIf(1)" 2>&1 | grep -oE -m1 '<Debug>|<Trace>|<Information>'
echo "."
${CLICKHOUSE_CLIENT_BINARY} --send_logs_level="None" --query="SELECT throwIf(1)" 2>&1 | grep -oE -m1 '<Debug>|<Trace>|<Information>|<Error>'
