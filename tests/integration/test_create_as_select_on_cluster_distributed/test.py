"""
Regression test for https://github.com/ClickHouse/ClickHouse/issues/90651

`CREATE TABLE ... ON CLUSTER ... AS SELECT ... FROM <Distributed> ORDER BY ...`
used to fail on a cluster with two or more shards with

    Code: 1001 ... std::bad_function_call            (25.8)
    Code: 10 ... Not found column __table1.<col> in block. There are only columns: <col>:
        While executing Remote. (NOT_FOUND_COLUMN_IN_BLOCK)   (later versions)

Root cause: the `ON CLUSTER` DDL is executed by the DDL worker, whose query
context did not carry a client version (it stayed 0.0.0). When that context
reads the `Distributed` table it forwards the sub-query to the remote shard as
if the initiator were a pre-23.3 server, so the shard disabled the analyzer for
"compatibility". The initiator kept using the analyzer, so the initiator and the
shard produced different column names (`__table1.x` vs `x`) and distributed
execution failed.

The query is only affected when at least one shard is read remotely, and the
outcome depended on whether `allow_experimental_analyzer` happened to be marked
as explicitly changed, so it could look intermittent. The loop below runs the
statement several times to be robust to that.
"""

import pytest

from helpers.cluster import ClickHouseCluster

cluster = ClickHouseCluster(__file__)

node1 = cluster.add_instance(
    "node1",
    main_configs=["configs/remote_servers.xml"],
    with_zookeeper=True,
    macros={"shard": 1, "replica": 1},
)
node2 = cluster.add_instance(
    "node2",
    main_configs=["configs/remote_servers.xml"],
    with_zookeeper=True,
    macros={"shard": 2, "replica": 1},
)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        for node in (node1, node2):
            node.query(
                "CREATE TABLE local (num UInt64, d Date) "
                "ENGINE = MergeTree ORDER BY num"
            )
            node.query(
                "CREATE TABLE dist (num Int32, d Date) "
                "ENGINE = Distributed(test_cluster, currentDatabase(), local, cityHash64(num))"
            )
        node1.query("INSERT INTO local SELECT number, today() FROM numbers(50000)")
        node2.query(
            "INSERT INTO local SELECT number + 50000, today() FROM numbers(50000)"
        )
        yield cluster
    finally:
        cluster.shutdown()


@pytest.mark.parametrize(
    "select",
    [
        # Minimal reproducer: a raw column from a Distributed table plus a step
        # (ORDER BY) that forces the coordinator to merge the shard streams.
        "SELECT num FROM dist ORDER BY num",
        # A window function forces the same coordinator merge.
        "SELECT num, count() OVER (PARTITION BY num % 4) AS c FROM dist ORDER BY num",
        # The shape from the original report: cross join + window + ORDER BY.
        "SELECT num, jn.n, uniq(num) OVER (PARTITION BY jn.n) AS m "
        "FROM dist CROSS JOIN (SELECT number AS n FROM numbers(4)) AS jn "
        "WHERE num != 0 ORDER BY jn.n",
    ],
)
def test_create_as_select_on_cluster_over_distributed(started_cluster, select):
    # The failure looked intermittent, so run the statement several times.
    for i in range(10):
        table = f"res_{abs(hash(select)) % 100000}_{i}"
        node1.query(f"DROP TABLE IF EXISTS {table} ON CLUSTER test_cluster SYNC")
        # This must not throw NOT_FOUND_COLUMN_IN_BLOCK / bad_function_call.
        node1.query(
            f"CREATE TABLE {table} ON CLUSTER test_cluster ENGINE = Memory AS {select}"
        )
        # Every shard executed the same statement locally and must hold data.
        assert int(node1.query(f"SELECT count() FROM {table}")) > 0
        assert int(node2.query(f"SELECT count() FROM {table}")) > 0
        node1.query(f"DROP TABLE IF EXISTS {table} ON CLUSTER test_cluster SYNC")
