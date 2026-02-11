# UniqExactSet::merge → ThreadPool scheduleImpl / worker Flow

This document describes the call path from `runner.enqueueAndKeepTrack(thread_func, Priority{})` inside `UniqExactSet::merge()` to `ThreadPoolImpl::scheduleImpl` and `ThreadFromThreadPool::worker()`, and how the interaction between the **mutex** and **condition_variable** affects performance.

---

## 1. Call chain overview (Mermaid flowchart)

```mermaid
flowchart TB
    subgraph UniqExactSet["UniqExactSet.h"]
        A["merge(other, thread_pool, is_cancelled)"]
        B["ThreadPoolCallbackRunnerLocal runner(*thread_pool)"]
        C["runner.enqueueAndKeepTrack(thread_func, Priority{})"]
    end

    subgraph Runner["threadPoolCallbackRunner.h"]
        D["enqueueAndKeepTrack()"]
        E["enqueueAndGiveOwnership()"]
        F["pool.scheduleOrThrowOnError(task_func, priority)"]
    end

    subgraph ThreadPool["ThreadPool.cpp"]
        G["scheduleOrThrowOnError()"]
        H["scheduleImpl()"]
    end

    A --> B --> C
    C --> D --> E --> F
    F --> G --> H
```

---

## 2. scheduleImpl and worker: lock / condition variable interaction

The sequence diagram below shows contention on the **same mutex** between `scheduleImpl` (producer) and `worker()` (consumer), and how `new_job_or_shutdown` notify/wait ties them together.

```mermaid
sequenceDiagram
    participant M as parent_pool.mutex
    participant CV as new_job_or_shutdown
    participant S as scheduleImpl (caller thread)
    participant W1 as worker-1
    participant W2 as worker-2

    Note over S: enqueueAndKeepTrack → scheduleOrThrowOnError → scheduleImpl

    S->>S: ScopedDecrement(available_threads)
    S->>S: Optional: lock-free creation of new_thread (remaining_pool_capacity CAS)

    rect rgb(255, 230, 200)
        Note over S,M: scheduleImpl acquires the lock
        S->>M: std::unique_lock lock(mutex)  [blocks until acquired]
        Note over S: If a worker holds the lock then scheduleImpl waits and adds latency
        M-->>S: Lock acquired

        S->>S: job_finished.wait(lock, pred)  [waits if queue full or no free thread]
        S->>S: jobs.emplace(job, priority, ...)
        S->>S: ++scheduled_jobs
        S->>S: If adding_new_thread: (*thread_slot)->start(thread_slot)
        S->>M: lock destructor then release mutex
    end

    S->>CV: new_job_or_shutdown.notify_one()
    Note over S,CV: Wakes at most one worker blocked in wait

    rect rgb(200, 230, 255)
        Note over W1,M: worker loop: acquire lock → wait → get job
        W1->>M: std::unique_lock lock(parent_pool.mutex)  [may contend with scheduleImpl]
        M-->>W1: Lock acquired
        W1->>W1: If job_is_done: --scheduled_jobs and job_finished.notify_all()
        W1->>CV: new_job_or_shutdown.wait(lock, predicate)
        Note over W1: If predicate false: release mutex and sleep. After notify re-acquire lock
        CV-->>W1: Woken by notify_one then re-check predicate
        W1->>W1: job_data = jobs.top() and jobs.pop()
        W1->>M: lock destructor then release mutex
    end

    W1->>W1: Execute job_data->job() outside the lock

    Note over W2: Other workers may still be blocked in wait
```

---

## 3. How the lock and condition variable affect each other (performance)

```mermaid
flowchart LR
    subgraph scheduleImpl["scheduleImpl while holding lock"]
        S1["std::unique_lock lock(mutex)"]
        S2["job_finished.wait(lock, pred)"]
        S3["jobs.emplace(...)"]
        S4["++scheduled_jobs"]
        S5["start(thread_slot)"]
        S6["Release lock"]
    end

    subgraph worker["worker while holding lock"]
        W1["std::unique_lock lock(parent_pool.mutex)"]
        W2["Handle job_is_done, --scheduled_jobs"]
        W3["new_job_or_shutdown.wait(lock, predicate)"]
        W4["jobs.top(); jobs.pop()"]
        W5["Release lock"]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
    W1 --> W2 --> W3 --> W4 --> W5

    S6 -->|"notify_one()"| N["new_job_or_shutdown"]
    N -->|"Wake one worker in wait"| W3

    M["Same mutex"]
    S1 -.->|"Contention"| M
    W1 -.->|"Contention"| M
```

### 3.1 Key code locations (lock and wait)

| Location | Code | Purpose |
|----------|------|---------|
| **scheduleImpl** (ThreadPool.cpp ~line 320) | `std::unique_lock lock(mutex);` | Protects `jobs`, `scheduled_jobs`, `threads`; must hold lock before enqueueing |
| **scheduleImpl** (~line 435) | `new_job_or_shutdown.notify_one();` | Called **after** releasing the lock; wakes one worker blocked in `wait` |
| **worker** (~line 719) | `std::unique_lock lock(parent_pool.mutex);` | Contends for the **same** `parent_pool.mutex` as scheduleImpl |
| **worker** (~line 756) | `parent_pool.new_job_or_shutdown.wait(lock, [this]{ return !jobs.empty() \|\| shutdown \|\| ... });` | When predicate is false: **atomically release lock** and sleep; after notify **re-acquire lock** first, then re-check predicate |

### 3.2 Why this interaction causes performance issues

1. **Single mutex contention**
   - Under the lock, `scheduleImpl` does: `job_finished.wait`, `jobs.emplace`, `++scheduled_jobs`, and possibly `start(thread_slot)`.
   - Under the lock, each `worker` does: finish previous job bookkeeping, `--scheduled_jobs`, `job_finished.notify_all`, `new_job_or_shutdown.wait`, and `jobs.top()` / `jobs.pop()`.
   - All queue and counter updates use this one mutex, so **scheduleImpl and every worker contend on the same mutex**. Under high concurrency, lock contention lengthens hold times and increases:
   - Latency of `enqueueAndKeepTrack` (caller must acquire the lock before enqueueing).
   - Latency from “worker is notified” to “worker gets the job” (worker must re-acquire the lock before `jobs.pop()`).

2. **notify_one and wait interaction**
   - `wait(lock, predicate)` **releases the lock and then sleeps** when the predicate is false, so it never sleeps while holding the lock (avoids deadlock).
   - `scheduleImpl` calls `notify_one()` **after** releasing the lock; the woken worker returns from `wait` and **competes again for the lock**. If another `scheduleImpl` call acquires the lock first, that worker blocks again on `lock(mutex)`, adding “woken then immediately blocked on lock” latency.

3. **Bulk enqueue in UniqExactSet::merge**
   - Merge loops over buckets and calls `runner.enqueueAndKeepTrack(thread_func, Priority{})` each time, so each call enters `scheduleImpl` and holds the lock.
   - With many threads (e.g. `min(thread_pool->getMaxThreads(), rhs.NUM_BUCKETS)` large), **many enqueue callers** and **many workers** contend on the same mutex, amplifying the above latency and contention.

---

## 4. Combined flow (including mutex / wait)

```mermaid
flowchart TB
    subgraph Entry["Entry"]
        A["UniqExactSet::merge()"]
        A --> B["for i in 0..min(max_threads, NUM_BUCKETS)"]
        B --> C["runner.enqueueAndKeepTrack(thread_func, Priority{})"]
    end

    subgraph Runner["ThreadPoolCallbackRunnerLocal"]
        C --> D["enqueueAndGiveOwnership()"]
        D --> E["pool.scheduleOrThrowOnError(task_func, priority)"]
    end

    subgraph Schedule["scheduleImpl (ThreadPool.cpp)"]
        E --> F["ScopedDecrement(available_threads)"]
        F --> G["Optional: lock-free create ThreadFromThreadPool"]
        G --> H["std::unique_lock lock(mutex)"]
        H --> I["job_finished.wait(lock, pred)"]
        I --> J["jobs.emplace(job, ...)"]
        J --> K["++scheduled_jobs"]
        K --> L["(if new thread) start(thread_slot)"]
        L --> M["Release lock"]
        M --> N["new_job_or_shutdown.notify_one()"]
    end

    subgraph Worker["ThreadFromThreadPool::worker()"]
        N --> O["std::unique_lock lock(parent_pool.mutex)"]
        O --> P["If job_is_done: --scheduled_jobs, job_finished.notify_all()"]
        P --> Q["new_job_or_shutdown.wait(lock, predicate)"]
        Q --> R{"predicate?"}
        R -->|jobs non-empty / shutdown / thread excess| S["job_data = jobs.top(); jobs.pop()"]
        R -->|else| Q
        S --> T["Release lock"]
        T --> U["Execute job_data->job()"]
        U --> O
    end

    H -.->|"Same mutex"| O
    M -.->|"notify_one wakes"| Q
```

---

## 5. Summary

| Mechanism | Role | Performance impact |
|-----------|------|--------------------|
| `std::unique_lock lock(mutex)` (scheduleImpl) | Protects `jobs`, `scheduled_jobs`, `threads` when enqueueing | Longer hold time causes more blocking of workers and other scheduleImpl callers on this lock |
| `std::unique_lock lock(parent_pool.mutex)` (worker) | Get job, update scheduled_jobs, wait for new job | Contends with scheduleImpl on the same mutex; becomes a hot spot under high concurrency |
| `new_job_or_shutdown.wait(lock, predicate)` | When no job: release lock and sleep; after notify, re-acquire lock and re-check | Avoids busy-wait, but after wakeup the thread may block again while re-acquiring the lock |
| `new_job_or_shutdown.notify_one()` (after scheduleImpl releases lock) | Wake one worker blocked in `wait` | Wakeup order is independent of who acquires the lock next; the first woken worker may acquire the lock later than others |

Overall, **scheduleImpl and worker cooperate via the same mutex and the same condition_variable**. In bulk enqueue scenarios like `UniqExactSet::merge`, lock contention and the ordering of notify/wait together affect latency and throughput. The code instruments this via `ProfileEvents::GlobalThreadPoolLockWaitMicroseconds` / `LocalThreadPoolLockWaitMicroseconds` and TRACE logs (e.g. `scheduleImpl_mutex lock time`, `worker_get_job_mutex lock time`, `new_job_or_shutdown.wait() time`).
