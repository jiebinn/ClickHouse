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
        node.query("""
                CREATE TABLE test_auxiliary_zookeeper(a Int32)
                ENGINE = ReplicatedMergeTree('zookeeper2:/clickhouse/tables/test/test_auxiliary_zookeeper', '{replica}')
                ORDER BY a;
            """.format(replica=node.name))

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
        node.query("""
                CREATE TABLE test_from_zkpath_aux(a Int32)
                ENGINE = ReplicatedMergeTree('{zk_path}', '{replica}')
                ORDER BY a;
            """.format(zk_path=table_zk_path, replica=node.name))

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


def test_drop_replica_from_zkpath_with_trailing_slashes(started_cluster):
    # A valid non-root ZKPATH with extra trailing slashes must be normalized so the
    # interpreter finds the "<path>/replicas" node. Before normalization was applied
    # in the parser, "<path>//" left a leftover slash and probed "<path>//replicas",
    # which does not exist, wrongly failing with "does not look like a table path".
    zk_path = "zookeeper_aux:/clickhouse/tables/test/test_zkpath_slashes"
    inner_path = "/aux_root/clickhouse/tables/test/test_zkpath_slashes"

    node2.start_clickhouse()
    drop_table([node1, node2], "test_zkpath_slashes")

    for node in [node1, node2]:
        node.query("""
                CREATE TABLE test_zkpath_slashes(a Int32)
                ENGINE = ReplicatedMergeTree('{zk_path}', '{replica}')
                ORDER BY a;
            """.format(zk_path=zk_path, replica=node.name))

    node2.stop_clickhouse()
    time.sleep(5)

    retries = 0
    max_retries = 5
    zk = cluster.get_kazoo_client("zoo1")
    while zk.exists(inner_path + "/replicas/node2/is_active") is not None:
        retries += 1
        if retries > max_retries:
            raise Exception("Failed to stop server.")
        time.sleep(1)

    # Trailing slashes must be collapsed; the drop must still find the table path.
    node1.query(
        "SYSTEM DROP REPLICA 'node2' FROM ZKPATH '{zk_path}//'".format(zk_path=zk_path)
    )

    assert zk.exists(inner_path)
    assert zk.exists(inner_path + "/replicas/node2") is None

    node2.start_clickhouse()
    drop_table([node1, node2], "test_zkpath_slashes")


def test_drop_replica_from_zkpath_not_blocked_by_default_keeper_table(started_cluster):
    # The self-protection guard must be keeper-aware: a local table on the DEFAULT
    # keeper that happens to share the same path string and replica name must not
    # block a drop that targets that path on an AUXILIARY keeper (a physically
    # different znode under /aux_root).
    #
    # node1 hosts a default-keeper decoy at the shared path with replica name
    # "shared_replica"; node2 hosts the auxiliary-keeper target at the same path
    # string with the same replica name. They live on different servers and
    # different keepers, so the interserver endpoints do not collide. Dropping
    # "shared_replica" via the auxiliary keeper from node1 hits node1's guard,
    # which used to match the decoy purely on the path string.
    shared_path = "/clickhouse/tables/test/test_shared_path"
    aux_zk_path = "zookeeper_aux:" + shared_path
    aux_inner_path = "/aux_root" + shared_path

    node2.start_clickhouse()
    drop_table([node1, node2], "test_shared_default")
    drop_table([node1, node2], "test_shared_aux")

    # Default-keeper decoy on node1 (stays alive throughout).
    node1.query("""
            CREATE TABLE test_shared_default(a Int32)
            ENGINE = ReplicatedMergeTree('{path}', 'shared_replica')
            ORDER BY a;
        """.format(path=shared_path))

    # Auxiliary-keeper target on node2, same path string and replica name.
    node2.query("""
            CREATE TABLE test_shared_aux(a Int32)
            ENGINE = ReplicatedMergeTree('{path}', 'shared_replica')
            ORDER BY a;
        """.format(path=aux_zk_path))

    # Stop node2 so its auxiliary-keeper replica can be dropped remotely.
    node2.stop_clickhouse()
    time.sleep(5)

    retries = 0
    max_retries = 5
    zk = cluster.get_kazoo_client("zoo1")
    while zk.exists(aux_inner_path + "/replicas/shared_replica/is_active") is not None:
        retries += 1
        if retries > max_retries:
            raise Exception("Failed to stop server.")
        time.sleep(1)

    # node1 still hosts the default-keeper decoy with the same path+replica string.
    # Before the guard became keeper-aware this drop was rejected with
    # TABLE_WAS_NOT_DROPPED ("There is a local table ..."); now it must succeed
    # against the auxiliary keeper and leave the default-keeper decoy untouched.
    node1.query(
        "SYSTEM DROP REPLICA 'shared_replica' FROM ZKPATH '{zk_path}'".format(
            zk_path=aux_zk_path
        )
    )

    # The auxiliary-keeper replica is gone.
    assert zk.exists(aux_inner_path + "/replicas/shared_replica") is None
    # The default-keeper decoy on node1 is untouched.
    assert zk.exists(shared_path + "/replicas/shared_replica") is not None

    node2.start_clickhouse()
    drop_table([node1, node2], "test_shared_default")
    drop_table([node1, node2], "test_shared_aux")
