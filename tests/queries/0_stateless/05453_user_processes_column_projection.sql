SELECT count() FROM system.user_processes WHERE user = currentUser();
SELECT user = currentUser() FROM system.user_processes WHERE user = currentUser();
SELECT memory_usage >= 0 FROM system.user_processes WHERE user = currentUser();
SELECT peak_memory_usage >= memory_usage FROM system.user_processes WHERE user = currentUser();
SELECT ProfileEvents['SelectedRows'] >= 1000
FROM system.user_processes
WHERE user = currentUser() AND (SELECT sum(number) FROM numbers_mt(1000)) > 0;
SELECT `ProfileEvents.Names` = mapKeys(ProfileEvents), `ProfileEvents.Values` = mapValues(ProfileEvents)
FROM system.user_processes WHERE user = currentUser();
SELECT _database, _table FROM system.user_processes WHERE user = currentUser();
