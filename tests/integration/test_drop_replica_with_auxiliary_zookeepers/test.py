import time

import pytest

import helpers.client as client
from helpers.client import QueryRuntimeException
from helpers.cluster import ClickHouseCluster
from helpers.test_tools import TSV

cluster = ClickHouseCluster(__file__)
node1 = cluster.add_instance(
    "node1",
    main_configs=["configs/zookeeper_config.xml", "configs/remote_servers.xml"],
    with_zookeeper=True,
    use_keeper=False,
    stay_alive=True,
)
node2 = cluster.add_instance(
    "node2",
    main_configs=["configs/zookeeper_config.xml", "configs/remote_servers.xml"],
    with_zookeeper=True,
    use_keeper=False,
    stay_alive=True,
)


def create_aux_root(zk):
    # The zookeeper_aux auxiliary keeper is chrooted at /aux_root; the root must
    # exist before ClickHouse connects to it.
    zk.ensure_path("/aux_root")


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.add_zookeeper_startup_command(create_aux_root)
        cluster.start()

        yield cluster

    except Exception as ex:
        print(ex)

    finally:
        cluster.shutdown()


def drop_table(nodes, table_name):
    for node in nodes:
        node.query("DROP TABLE IF EXISTS {} NO DELAY".format(table_name))


def test_drop_replica_in_auxiliary_zookeeper(started_cluster):
    drop_table([node1, node2], "test_auxiliary_zookeeper")
    for node in [node1, node2]:
        node.query(
            """
                CREATE TABLE test_auxiliary_zookeeper(a Int32)
                ENGINE = ReplicatedMergeTree('zookeeper2:/clickhouse/tables/test/test_auxiliary_zookeeper', '{replica}')
                ORDER BY a;
            """.format(
                replica=node.name
            )
        )

    # stop node2 server
    node2.stop_clickhouse()
    time.sleep(5)

    # check is_active
    retries = 0
    max_retries = 5
    zk = cluster.get_kazoo_client("zoo1")
    while True:
        if (
            zk.exists(
                "/clickhouse/tables/test/test_auxiliary_zookeeper/replicas/node2/is_active"
            )
            is None
        ):
            break
        else:
            retries += 1
            if retries > max_retries:
                raise Exception("Failed to stop server.")
            time.sleep(1)

    # drop replica node2
    node1.query("SYSTEM DROP REPLICA 'node2'")

    assert zk.exists("/clickhouse/tables/test/test_auxiliary_zookeeper")
    assert (
        zk.exists("/clickhouse/tables/test/test_auxiliary_zookeeper/replicas/node2")
        is None
    )


def test_drop_replica_from_zkpath_in_auxiliary_zookeeper(started_cluster):
    # SYSTEM DROP REPLICA ... FROM ZKPATH 'aux:/path' must operate on the named
    # auxiliary keeper, not the default one. zookeeper_aux is chrooted at
    # /aux_root, so the table's znodes live under a namespace that does not exist
    # on the default keeper; routing the command to the wrong keeper would fail
    # to find the path.
    table_zk_path = "zookeeper_aux:/clickhouse/tables/test/test_from_zkpath_aux"
    inner_path = "/aux_root/clickhouse/tables/test/test_from_zkpath_aux"

    # The previous test leaves node2 stopped; make sure it is running here.
    node2.start_clickhouse()
    drop_table([node1, node2], "test_auxiliary_zookeeper")
    drop_table([node1, node2], "test_from_zkpath_aux")

    for node in [node1, node2]:
        node.query(
            """
                CREATE TABLE test_from_zkpath_aux(a Int32)
                ENGINE = ReplicatedMergeTree('{zk_path}', '{replica}')
                ORDER BY a;
            """.format(
                zk_path=table_zk_path, replica=node.name
            )
        )

    # stop node2 server so its replica can be dropped remotely
    node2.stop_clickhouse()
    time.sleep(5)

    # wait until node2 is no longer active in the auxiliary keeper
    retries = 0
    max_retries = 5
    zk = cluster.get_kazoo_client("zoo1")
    while zk.exists(inner_path + "/replicas/node2/is_active") is not None:
        retries += 1
        if retries > max_retries:
            raise Exception("Failed to stop server.")
        time.sleep(1)

    # Targeting the default keeper (the pre-fix behaviour) would not find this
    # path, so a successful drop proves the command was routed to zookeeper_aux.
    node1.query(
        "SYSTEM DROP REPLICA 'node2' FROM ZKPATH '{zk_path}'".format(
            zk_path=table_zk_path
        )
    )

    assert zk.exists(inner_path)
    assert zk.exists(inner_path + "/replicas/node2") is None

    node2.start_clickhouse()
    drop_table([node1, node2], "test_from_zkpath_aux")
