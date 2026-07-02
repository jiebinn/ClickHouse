-- Tags: no-parallel
-- Empty ZKPATH is rejected at parse time, so the error is raised in the client, not the server.
SYSTEM DROP REPLICA 'r1' FROM ZKPATH ''; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP REPLICA 'r1' FROM ZKPATH '/'; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP DATABASE REPLICA 'r1' FROM ZKPATH ''; -- { clientError BAD_ARGUMENTS }
