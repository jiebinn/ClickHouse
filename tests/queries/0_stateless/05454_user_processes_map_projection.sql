SELECT memory_usage >= 0, user = currentUser(), peak_memory_usage >= memory_usage
FROM system.user_processes WHERE user = currentUser();
SELECT ProfileEvents.keys = mapKeys(ProfileEvents), ProfileEvents.values = mapValues(ProfileEvents)
FROM system.user_processes WHERE user = currentUser();
SELECT user = currentUser()
FROM system.user_processes
WHERE user = currentUser() AND ProfileEvents['SelectedRows'] >= 1000
    AND (SELECT sum(number) FROM numbers_mt(1000)) > 0;
SELECT count() FROM system.user_processes WHERE _database = 'system' AND _table = 'user_processes' AND user = currentUser();
SELECT count() FROM system.user_processes WHERE user = currentUser() AND 0;
SELECT count() FROM (SELECT * EXCEPT ProfileEvents FROM system.user_processes WHERE user = currentUser());
