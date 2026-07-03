#!/usr/bin/env bash
# A long OR chain of raw `match()` predicates on the same expression. When `multiMatchAny` is not
# used (here the chain contains raw `match()` regexps, so it is kept off the Vectorscan path), the
# rewrite would merge the whole chain into a single combined `match(s, '(p0)|(p1)|...')` regexp.
# Each individual pattern compiles in RE2 on its own, but the merged alternation expands past RE2's
# default 8 MiB program budget (`RE2::Options::kDefaultMaxMem`) and `match` would throw
# `CANNOT_COMPILE_REGEXP`. With `optimize_or_like_chain` now enabled by default, the rewrite must not
# turn such a previously-working query into an exception: when the combined regexp does not compile it
# has to keep the original branches. Note the blow-up here comes from bounded repetition (`z{1000}`),
# not from the text length, so a combined-regexp *length* cap would not catch it — only pre-compiling
# the merged regexp does. Verify the query succeeds and returns the same result as the un-rewritten OR
# chain, for both the new and the old analyzer.
#
# RE2 rejects a single repetition larger than `{1000}`, so each branch spells out 100 consecutive
# `z{1000}` tokens (~100000 RE2 instructions per branch, still far below the 8 MiB per-pattern budget),
# and the 20 branches merge into a ~2000000-instruction alternation, ~2x over the budget. A handful of
# large branches reproduces the same over-budget combined program that thousands of `z{1000}` branches
# would, but keeps the OR-chain AST small so a single run stays fast: the flaky check runs each new
# test many times and fails a single run over 180s, and a chain of thousands of branches took minutes
# per run under sanitizers.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

${CLICKHOUSE_CLIENT} -q "
    DROP TABLE IF EXISTS t_or_like_combined;
    CREATE TABLE t_or_like_combined (s String) ENGINE = Memory;
    -- Two rows match (the first two patterns), one row matches nothing.
    INSERT INTO t_or_like_combined SELECT concat('p', toString(number), repeat('z', 100000)) FROM numbers(2);
    INSERT INTO t_or_like_combined VALUES ('nothing matches here');
"

# Build an OR chain of 20 raw match() predicates. Each branch is `p<i>` followed by 100 `z{1000}`
# tokens, so it compiles on its own (~100000 RE2 instructions), but the combined alternation is
# ~2000000 instructions, far above RE2's ~8 MiB program budget.
z_repeat=$(printf 'z{1000}%.0s' $(seq 1 100))
predicate=""
for i in $(seq 0 19); do
    if [ -n "$predicate" ]; then
        predicate="$predicate OR "
    fi
    predicate="${predicate}match(s, 'p${i}${z_repeat}')"
done

query="SELECT count() FROM t_or_like_combined WHERE ${predicate}"

# Reference (rewrite disabled), then the rewrite enabled for both the old and the new analyzer.
# All three must succeed (no CANNOT_COMPILE_REGEXP) and return the same count.
${CLICKHOUSE_CLIENT} -q "${query} SETTINGS optimize_or_like_chain = 0"
${CLICKHOUSE_CLIENT} -q "${query} SETTINGS optimize_or_like_chain = 1, enable_analyzer = 0"
${CLICKHOUSE_CLIENT} -q "${query} SETTINGS optimize_or_like_chain = 1, enable_analyzer = 1"

${CLICKHOUSE_CLIENT} -q "DROP TABLE t_or_like_combined"
