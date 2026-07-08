import pytest

from helpers.iceberg_utils import (
    create_iceberg_table,
    get_uuid_str
)


def _metadata_dir(table_name):
    return f"/var/lib/clickhouse/user_files/iceberg_data/default/{table_name}/metadata"


@pytest.mark.parametrize("format_version", [1, 2])
@pytest.mark.parametrize("storage_type", ["local"])
def test_writes_with_compression_metadata(started_cluster_iceberg_no_spark, format_version, storage_type):
    instance = started_cluster_iceberg_no_spark.instances["node1"]
    TABLE_NAME = "test_writes_with_compression_metadata_" + storage_type + "_" + get_uuid_str()

    create_iceberg_table(storage_type, instance, TABLE_NAME, started_cluster_iceberg_no_spark, "(x String, y Int64)", format_version, use_version_hint=True, compression_method="gzip")

    assert instance.query(f"SELECT * FROM {TABLE_NAME} ORDER BY ALL") == ''
    instance.query(f"INSERT INTO {TABLE_NAME} VALUES ('123', 1);", settings={"iceberg_metadata_compression_method": "gzip"})
    assert instance.query(f"SELECT * FROM {TABLE_NAME} ORDER BY ALL") == '123\t1\n'

    # gzip metadata must use the Iceberg spec extension `gz`, not the HTTP
    # Content-Encoding token `gzip`, otherwise Spark / Hadoop-catalog readers
    # cannot locate the metadata file (issue #109801).
    listing = instance.exec_in_container(
        ["bash", "-c", f"ls {_metadata_dir(TABLE_NAME)}"]
    )
    assert ".gz.metadata.json" in listing
    assert ".gzip.metadata.json" not in listing
