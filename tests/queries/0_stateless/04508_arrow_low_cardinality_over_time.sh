#!/usr/bin/env bash
# Tags: no-fasttest

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

# Writing a LowCardinality(Time) column as an Arrow dictionary used to abort with
# "Cannot fill arrow array time32 with LowCardinality(...) data" because the Arrow
# dictionary dispatch had no Time32 entry.
$CLICKHOUSE_LOCAL -q "select '12:00:00'::LowCardinality(Time) as a format Arrow settings allow_suspicious_low_cardinality_types = 1, output_format_arrow_low_cardinality_as_dictionary = 1" > /dev/null && echo "Time ok"
$CLICKHOUSE_LOCAL -q "select '12:00:00'::LowCardinality(Nullable(Time)) as a format Arrow settings allow_suspicious_low_cardinality_types = 1, output_format_arrow_low_cardinality_as_dictionary = 1" > /dev/null && echo "Nullable(Time) ok"
