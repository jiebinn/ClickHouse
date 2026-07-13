#!/usr/bin/env bash
# Regression test: serializing a shared/const quantileTDigest state as a parallel GROUP BY key
# used to mutate the shared state via compress() (data race -> heap-buffer-overflow in RadixSort).
# The race is probabilistic per run, so loop the query enough times that the broken code trips
# ASan reliably; the fixed serialize() is const and never touches the shared centroids buffer.

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

# A single tdigest state (const, one row) broadcast across many rows and grouped by, so the
# Aggregator serializes the SAME shared state pointer from several threads concurrently.
# 200k rows reliably trips the pre-fix race within the first few loop iterations under ASan,
# while keeping a whole run (~15s) well under the 180s flaky-check per-run ceiling.
query="SELECT count()
FROM
(
    SELECT k
    FROM
    (
        SELECT number, (SELECT quantileTDigestState(number) FROM numbers(5000)) AS k
        FROM numbers_mt(200000)
    )
    GROUP BY k
) SETTINGS max_threads = 16"

for _ in {1..30}; do
    ${CLICKHOUSE_CLIENT} --query "$query"
done
