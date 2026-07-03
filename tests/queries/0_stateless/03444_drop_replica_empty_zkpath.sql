SYSTEM DROP REPLICA 'r1' FROM ZKPATH ''; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP REPLICA 'r1' FROM ZKPATH '/'; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP REPLICA 'r1' FROM ZKPATH '//'; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP REPLICA 'r1' FROM ZKPATH 'aux:/'; -- { clientError BAD_ARGUMENTS } -- 'aux:/' must not be misparsed as default keeper '/aux:'
SYSTEM DROP REPLICA 'r1' FROM ZKPATH 'aux://'; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP DATABASE REPLICA 'r1' FROM ZKPATH ''; -- { clientError BAD_ARGUMENTS }
SYSTEM DROP DATABASE REPLICA 'r1' FROM ZKPATH 'aux:/'; -- { clientError BAD_ARGUMENTS }
