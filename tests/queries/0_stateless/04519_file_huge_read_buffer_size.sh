#!/usr/bin/env bash

# Regression test for https://github.com/ClickHouse/ClickHouse/issues/83577
# An out-of-range `max_read_buffer_size` used to be passed straight to the allocator when
# `StorageFile` created its read buffer, tripping the allocator size guard with a
# `LOGICAL_ERROR` "Too large size passed to allocator". The read buffer size is now
# clamped, so reading the file must succeed.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

FILE_NAME="${CLICKHOUSE_TEST_UNIQUE_NAME}.csv"
DATA_FILE="${CLICKHOUSE_USER_FILES:?}/${FILE_NAME}"

${CLICKHOUSE_CLIENT} --query "select number from numbers(1000) format CSV" > "${DATA_FILE}"

${CLICKHOUSE_CLIENT} --query "
select sum(c1) from file('${FILE_NAME}', 'CSV', 'c1 UInt64')
settings max_read_buffer_size = 10000000000000000000, storage_file_read_method = 'pread'"

rm -f "${DATA_FILE}"
