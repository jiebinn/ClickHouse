SET system_events_show_zero_values = 1;
SELECT event FROM system.events WHERE event = 'SelectedRows';
SELECT name FROM system.events WHERE name IN ('Query', 'SelectQuery') ORDER BY name;
SELECT event FROM system.events WHERE startsWith(event, 'PerfAlignment') OR event = 'SelectedRows' ORDER BY event;
SELECT event FROM system.events
WHERE (event = 'Query' OR value > 0) AND event IN ('Query', 'SelectQuery') ORDER BY event;
SELECT event FROM system.events
WHERE event IN (SELECT arrayJoin(['Query', 'SelectQuery'])) ORDER BY event;
SELECT count() FROM system.events WHERE event = 'hcc_nonexistent_event_name_20261027';
SELECT length(description) > 0 FROM system.events WHERE event = 'Query';
SELECT _database, _table FROM system.events WHERE event = 'Query';
SELECT event FROM system.events WHERE event != 'Query' AND event IN ('Query', 'SelectQuery');
SELECT value FROM system.events WHERE event = 'PerfAlignmentFaults';
SET system_events_show_zero_values = 0;
SELECT count() FROM system.events WHERE event = 'PerfAlignmentFaults';
SET system_events_show_zero_values = 1;
SET short_circuit_function_evaluation = 'disable';
SELECT count() FROM system.events WHERE event = 'SelectedRows' AND rowNumberInAllBlocks() = 0;
SELECT count() FROM system.events
WHERE event = 'SelectedRows' AND arraySum(arrayMap(x -> rowNumberInAllBlocks() + x, [toUInt64(0)])) = 0;
