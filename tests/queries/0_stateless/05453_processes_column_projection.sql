SELECT count() FROM system.processes WHERE query_id = currentQueryID();
SELECT user = currentUser(), memory_usage >= 0 FROM system.processes WHERE query_id = currentQueryID();
SELECT current_database = currentDatabase() FROM system.processes WHERE query_id = currentQueryID();
SELECT Settings['max_threads'] FROM system.processes WHERE query_id = currentQueryID() SETTINGS max_threads = 7;
SELECT length(`Settings.Names`) = length(`Settings.Values`), has(`Settings.Names`, 'max_threads')
FROM system.processes WHERE query_id = currentQueryID() SETTINGS max_threads = 7;
SELECT ProfileEvents['SelectedRows'] >= 10 FROM system.processes
WHERE query_id = currentQueryID() AND (SELECT sum(number) FROM numbers(10)) = 45;
SELECT length(`ProfileEvents.Names`) = length(`ProfileEvents.Values`), has(`ProfileEvents.Names`, 'SelectedRows')
FROM system.processes WHERE query_id = currentQueryID() AND (SELECT sum(number) FROM numbers(10)) = 45;
SELECT length(thread_ids) > 0, peak_threads_usage > 0 FROM system.processes WHERE query_id = currentQueryID();
SELECT peak_memory_usage >= memory_usage, spillable_memory_bytes >= 0 FROM system.processes WHERE query_id = currentQueryID();
SELECT toTypeName(address), port > 0, toTypeName(initial_address), initial_port > 0 FROM system.processes WHERE query_id = currentQueryID();
SELECT is_initial_query, initial_user = currentUser(), initial_query_id = query_id, interface = 'TCP'
FROM system.processes WHERE query_id = currentQueryID();
SELECT elapsed >= 0, is_cancelled, is_all_data_sent, read_rows >= 0, read_bytes >= 0,
       total_rows_approx >= 0, written_rows, written_bytes, query_kind, is_internal
FROM system.processes WHERE query_id = currentQueryID();
SELECT _database, _table, quota_key, distributed_depth FROM system.processes WHERE query_id = currentQueryID();
SELECT length(query) > 0, normalized_query_hash != 0, client_revision > 0,
       client_version_major > 0, client_version_minor >= 0, client_version_patch >= 0
FROM system.processes WHERE query_id = currentQueryID();
