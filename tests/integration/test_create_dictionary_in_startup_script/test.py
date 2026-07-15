
import pytest

from helpers.cluster import ClickHouseCluster

cluster = ClickHouseCluster(__file__)
node = cluster.add_instance(
    "node",
    main_configs=[
        "config/startup_scripts.xml",
    ],
    user_configs=[
        "config/users.xml",
    ],
    stay_alive=True,
    with_minio=True,
    macros={"shard": 1, "replica": 1},
)


@pytest.fixture(scope="module")
def start_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


def test_create_dictionary_in_startup_script(start_cluster):
    STATE_SUCCESS = "1"
    assert node.query(
        """
        SELECT value
        FROM system.metrics
        WHERE name = 'StartupScriptsExecutionState'
        """
    ).strip() == STATE_SUCCESS

    assert int(
        node.query(
            """
            SELECT count()
            FROM system.custom_metrics
            """
        ).strip()
    ) > 0
