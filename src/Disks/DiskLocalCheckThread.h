#pragma once

#include <Interpreters/Context_fwd.h>

#include <Core/BackgroundSchedulePoolTaskHolder.h>

#include <Common/AggregatedMetrics.h>
#include <Common/Logger_fwd.h>

namespace DB
{
class DiskLocal;

class DiskLocalCheckThread : WithContext
{
    void run();

public:
    DiskLocalCheckThread(DiskLocal * disk_, ContextPtr context_, UInt64 local_disk_check_period_ms);
    ~DiskLocalCheckThread();

    void startup();
    void shutdown();

private:
    DiskLocal * disk;
    const size_t check_period_ms;
    const LoggerPtr log;

    AggregatedMetrics::GlobalSum is_readonly;
    AggregatedMetrics::GlobalSum is_broken;
    /// Declared last so it is destroyed first: the explicit destructor (and the holder's own
    /// destructor) deactivates the task and waits for an in-flight `run` to finish before the
    /// metrics handles above are destroyed, avoiding a use-after-destruction race on shutdown.
    BackgroundSchedulePoolTaskHolder task;
};

}
