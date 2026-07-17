#!/usr/bin/env bash
# Tags: zookeeper, no-parallel, no-shared-merge-tree, no-replicated-database
# no-parallel: uses a PAUSEABLE_ONCE failpoint that fires exactly once globally.
# no-shared-merge-tree: the failpoint is injected in ReplicatedMergeTreePartCheckThread; SharedMergeTree uses a different code path.
# no-replicated-database: uses explicit ReplicatedMergeTree ZooKeeper paths.
#
# Regression test for the race in ReplicatedMergeTreePartCheckThread::cancelRemovedPartsCheck.
# The function snapshots the parts to remove under parts_mutex, releases the lock to remove them
# from ZooKeeper, then re-locks and asserts the recheck invariant. Two concurrent calls with
# overlapping drop ranges could interleave: the second erases the snapshotted part from
# parts_queue during the first call's lock gap, so the first re-locks, finds fewer parts than it
# snapshotted, and throws LOGICAL_ERROR "Unexpected number of parts to remove from parts_queue"
# (server abort in debug builds).
#
# Two foreground MOVE PARTITION TO TABLE queries both reach cancelRemovedPartsCheck directly
# (movePartitionToTable), and neither registers in currently_executing_drop_replace_ranges, so
# they are not serialized against each other by the existing drop-replace machinery.
#
# We drive the race deterministically:
#   1. Seed a part into the check queue: INSERT, remove the part's node from ZooKeeper, then
#      CHECK TABLE. The part is younger than the (now removed) ZooKeeper node, so the check thread
#      re-enqueues it for a delayed recheck -> it sits in parts_queue.
#   2. Start the first MOVE PARTITION in the background; it enters cancelRemovedPartsCheck,
#      snapshots the seeded part, and pauses in the lock gap (before the ZooKeeper removal).
#   3. Run a second MOVE PARTITION over the same partition. Without the fix it runs the full cancel
#      and erases the seeded part from parts_queue; with the fix it blocks on the new mutex until
#      the first cancel finishes.
#   4. Resume the first MOVE. Without the fix it re-locks, finds the part gone, and throws the
#      logical error (server abort in debug). With the fix the first MOVE succeeds.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

ZK_PATH="/clickhouse/tables/${CLICKHOUSE_TEST_ZOOKEEPER_PREFIX}/cancel_race"

function cleanup()
{
    ${CLICKHOUSE_CLIENT} -q "SYSTEM DISABLE FAILPOINT rmt_cancel_removed_parts_check_pause_in_gap" 2>/dev/null ||:
    ${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS cancel_race SYNC" 2>/dev/null ||:
    ${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS cancel_race_dst1 SYNC" 2>/dev/null ||:
    ${CLICKHOUSE_CLIENT} -q "DROP TABLE IF EXISTS cancel_race_dst2 SYNC" 2>/dev/null ||:
}
trap cleanup EXIT
cleanup

${CLICKHOUSE_CLIENT} -q "
    CREATE TABLE cancel_race (a UInt32)
    ENGINE = ReplicatedMergeTree('/clickhouse/tables/$CLICKHOUSE_TEST_ZOOKEEPER_PREFIX/cancel_race', 'r1')
    PARTITION BY a ORDER BY a;
    CREATE TABLE cancel_race_dst1 (a UInt32)
    ENGINE = ReplicatedMergeTree('/clickhouse/tables/$CLICKHOUSE_TEST_ZOOKEEPER_PREFIX/cancel_race_dst1', 'r1')
    PARTITION BY a ORDER BY a;
    CREATE TABLE cancel_race_dst2 (a UInt32)
    ENGINE = ReplicatedMergeTree('/clickhouse/tables/$CLICKHOUSE_TEST_ZOOKEEPER_PREFIX/cancel_race_dst2', 'r1')
    PARTITION BY a ORDER BY a;
"

# --- Step 1: seed a part into the part-check queue ---
${CLICKHOUSE_CLIENT} -q "INSERT INTO cancel_race VALUES (0)"
PART=$(${CLICKHOUSE_CLIENT} -q "SELECT name FROM system.parts WHERE database = currentDatabase() AND table = 'cancel_race' AND active LIMIT 1")

# Remove the part's node from ZooKeeper; CHECK TABLE then re-enqueues it for a delayed recheck.
${CLICKHOUSE_KEEPER_CLIENT} -q "rm '${ZK_PATH}/replicas/r1/parts/${PART}'" >/dev/null 2>&1
${CLICKHOUSE_CLIENT} -q "CHECK TABLE cancel_race SETTINGS check_query_single_value_result = 0" >/dev/null 2>&1 ||:

# --- Step 2: pause the first MOVE inside cancelRemovedPartsCheck's lock gap ---
${CLICKHOUSE_CLIENT} -q "SYSTEM ENABLE FAILPOINT rmt_cancel_removed_parts_check_pause_in_gap"

${CLICKHOUSE_CLIENT} -q "ALTER TABLE cancel_race MOVE PARTITION 0 TO TABLE cancel_race_dst1" &
MOVE1_PID=$!

# Wait until the first MOVE has snapshotted parts_to_remove and paused in the gap.
${CLICKHOUSE_CLIENT} -q "SYSTEM WAIT FAILPOINT rmt_cancel_removed_parts_check_pause_in_gap PAUSE"

# --- Step 3: start a second overlapping MOVE in the background. Without the fix it runs the full
#     cancel and erases the seeded part from parts_queue; with the fix it blocks on the new mutex
#     (held by the paused first MOVE), so it must run in the background or the test would deadlock. ---
${CLICKHOUSE_CLIENT} -q "ALTER TABLE cancel_race MOVE PARTITION 0 TO TABLE cancel_race_dst2" 2>/dev/null &
MOVE2_PID=$!
sleep 2

# --- Step 4: resume the first MOVE. Without the fix it re-locks, finds the part gone, and throws
#     the logical error (server abort in debug). With the fix the first MOVE succeeds. ---
${CLICKHOUSE_CLIENT} -q "SYSTEM NOTIFY FAILPOINT rmt_cancel_removed_parts_check_pause_in_gap"

# The second MOVE either moved the partition first or fails with a benign concurrent-ALTER error;
# either outcome is fine. The invariant under test is that the first MOVE does not hit the race.
wait $MOVE2_PID 2>/dev/null ||:

if wait $MOVE1_PID; then
    echo "OK"
else
    echo "FAIL: first MOVE PARTITION failed (cancelRemovedPartsCheck race)"
fi
