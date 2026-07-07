#!/usr/bin/env python3

import pytest

from helpers.cluster import ClickHouseCluster


cluster = ClickHouseCluster(__file__, keeper_config_dir="configs/")

node = cluster.add_instance(
    "node",
    stay_alive=True,
    with_zookeeper=True,
)


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        cluster.wait_zookeeper_to_start()
        yield cluster
    finally:
        cluster.shutdown()


def test_standalone_keeper_respects_disabled_psi_metrics(started_cluster):
    zoo1_container = started_cluster.get_container_id("zoo1")

    has_pressure_dir = started_cluster.exec_in_container(
        zoo1_container,
        ["bash", "-c", "test -d /proc/pressure && echo yes || true"],
    ).strip()
    if has_pressure_dir != "yes":
        pytest.skip("/proc/pressure is not available in this environment")

    pressure_fds = started_cluster.exec_in_container(
        zoo1_container,
        [
            "bash",
            "-c",
            """
            set -e
            pid="$(pgrep -f '[c]lickhouse.*keeper_config1.xml' | head -n1)"
            test -n "$pid"
            for fd in /proc/"$pid"/fd/*; do
                readlink "$fd" || true
            done | grep -F '/proc/pressure/' || true
            """,
        ],
        user="root",
    ).strip()

    assert pressure_fds == ""
