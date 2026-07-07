#!/usr/bin/env python3
"""executable UDF that does CPU work per row and exits with code 3 after EOF.

Used to validate that with `check_exit_code=false` the source reaps via the
non-blocking, no-status-check path (`tryReapWithoutStatusCheck`), so:
  - a non-zero child exit does not raise an exception, and
  - rusage is still captured (`ExecutableUserDefinedFunctionUserTimeMicroseconds > 0`),
  - and `CHILD_WAS_NOT_EXITED_NORMALLY` is NOT logged.
"""

import os
import sys
import time


def _cpu_work(seed: int) -> int:
    acc = 0
    base = seed & 0xFFFF
    for _ in range(300_000):
        acc = (acc + base) % 1000003
    return acc


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        n = int(line)
    except ValueError:
        n = 0
    sys.stdout.write(f"{_cpu_work(n)}\n")
    sys.stdout.flush()

# Flush all output, then close stdout so ClickHouse observes EOF and proceeds to
# reap the child while this process is still alive. Sleep before exiting so the
# child is provably still running (not yet a zombie) when cleanup runs its reap:
# this makes the "reap loses rusage" race deterministic. A single non-blocking
# wait4(WNOHANG) then returns 0 and loses the rusage, so the reap must poll for a
# bounded interval to still capture the child's CPU rusage. The 2s delay is chosen
# to sit inside command_termination_timeout (5s here) yet above a 1s window: a
# reap budget that follows command_termination_timeout captures this child's rusage
# in cleanup, whereas a shorter fixed budget would give up and let the destructor's
# waitForPid reap it (which collects no rusage), so a correct reap succeeds and
# cleanup never SIGTERM-bounds.
sys.stdout.flush()
os.close(1)
time.sleep(2.0)
os._exit(3)
