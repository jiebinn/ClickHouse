#!/usr/bin/env bash

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

# The Web UI (`/play`) prunes truly-blank saved tabs on page load in `reconcileStartup`, so that
# empty scratch tabs left over from a previous session are not restored. This is client-side logic
# that only runs in a browser, so there is no server-side behavior to exercise; the functional
# suite has no browser to drive it. Guard the contract at the served-HTML level instead, asserting
# that the shipped page still contains the load-bearing invariants — a later refactor of
# `reconcileStartup` that silently drops one of them (as several rounds of review nearly did) then
# fails here loudly.

URL="${CLICKHOUSE_PORT_HTTP_PROTO}://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT_HTTP}"
PLAY="$(${CLICKHOUSE_CURL} -sS "${URL}/play")"

# The startup reconciliation is present in the shipped page.
grep -oF 'reconcileStartup' <<< "$PLAY" | head -n1

# Contract: a mixed workspace restores only the non-blank tabs. A saved tab is kept on startup
# only when its query is non-blank (whitespace does not count) ...
grep -oF "(r.query || '').trim() !== ''" <<< "$PLAY" | head -n1

# ... OR it carries a genuine run result, so a run-backed tab whose editor was cleared after a run
# still survives the reload.
grep -oF 'r.result && r.result.ran' <<< "$PLAY" | head -n1

# The dropped blank tabs are recorded so a stale `?tab=` reload echo cannot resurrect them.
grep -oF 'pruned_blank_ids' <<< "$PLAY" | head -n1

# Contract: an all-blank workspace falls back to a single fresh tab — pruning leaves `savedTabs`
# empty, and this gate then routes control to the fresh-bootstrap-tab branch.
grep -oF 'if (savedTabs.length)' <<< "$PLAY" | head -n1
