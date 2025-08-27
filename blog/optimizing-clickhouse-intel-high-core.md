# Optimizing ClickHouse for Intel's Next‑Generation High Core Count Processors

Modern Intel high core count processors unlock massive parallelism for analytical databases, but they also expose new bottlenecks that don't appear on smaller systems. This post presents a systematic methodology for optimizing ClickHouse on many-core Intel Xeon processors, based on 22 merged upstream contributions that collectively deliver 20-40% performance improvements on high core count systems.

Each optimization methodology is illustrated with detailed technical examples, performance measurements, and code changes that you can apply to your own systems.

**Target Hardware**: Intel Xeon with 80-480 vCPUs, SMT/Hyper-Threading enabled, high memory bandwidth
**Test Environment**: 2×240 vCPU systems, ClickBench workload
**Measurement Tools**: perf, Intel VTune, pipeline visualization

---

## Methodology 1: Cache Line Contention and False Sharing Elimination

**Core Principle**: On many-core systems, cache coherence traffic becomes a major bottleneck. Hot data structures accessed by multiple threads must be carefully aligned and partitioned to avoid false sharing.

**Detection Method**: Use `perf c2c` or Intel VTune to identify cache line bouncing patterns. Look for high LLC miss rates and `native_queued_spin_lock_slowpath` hotspots.

### Example 1.1: ProfileEvents Counter Alignment (PR #82697)

**Problem Identified**: ClickBench Q3 showed 36.6% of CPU cycles spent in `ProfileEvents::increment` on a 2×240 vCPU system. Performance profiling revealed severe cache line contention.

**Root Cause Analysis**: Multiple ProfileEvents counters were packed into the same cache line without alignment. When different threads updated different counters simultaneously, the entire cache line bounced between cores.

**Technical Solution**: 
```cpp
// Before: Counters packed without alignment
struct Counter {
    std::atomic<Value> value;  // 8 bytes, no alignment
};

// After: Cache line aligned counters  
struct alignas(64) Counter {
    std::atomic<Value> value;  // 8 bytes, 64-byte aligned
    char padding[56];          // Pad to full cache line
};
```

**Performance Impact**:
- `ProfileEvents::increment` hotspot reduced from 36.6% to 8.5%
- 20-40% QPS improvement on systems with 100-480 vCPUs
- ClickBench Q3 achieved 27.4% performance improvement

![PR82697 figure 1](assets/prs/82697_img1.png)
*Before: 36.6% cycles in ProfileEvents::increment*

![PR82697 figure 2](assets/prs/82697_img2.png)
*After: 8.5% cycles in ProfileEvents::increment*

![PR82697 figure 3](assets/prs/82697_img3.png)
*Performance scaling by core count*

![PR82697 figure 4](assets/prs/82697_img4.png)
*Overall ClickBench improvement*

### Example 1.2: QueryConditionCache Lock Optimization (PR #80247)

**Problem Identified**: After resolving jemalloc page faults, a new 76% hotspot emerged in `native_queued_spin_lock_slowpath` from `QueryConditionCache::write` on 2×240 vCPU systems.

**Root Cause Analysis**: All threads were attempting to lock `entry->mutex` unnecessarily due to missing checks for `mark_ranges` and `has_final_mark`. This caused serialization on the query hot path.

**Technical Solution**:
```cpp
// Before: Always lock for cache updates
void updateCache() {
    std::lock_guard<std::mutex> lock(entry->mutex);
    // Always rebuild cache
    rebuildConditions();
}

// After: Check if update is needed before locking
bool needsUpdate(const MarkRanges& ranges, bool has_final) {
    return ranges != cached_ranges || has_final != cached_final;
}

void updateCache() {
    if (!needsUpdate(mark_ranges, has_final_mark))
        return;  // Skip unnecessary work
    
    std::lock_guard<std::mutex> lock(entry->mutex);
    if (!needsUpdate(mark_ranges, has_final_mark))  // Double-check
        return;
    rebuildConditions();
}
```

**Performance Impact**:
- `native_queued_spin_lock_slowpath` reduced from 76% to 1%
- Q10 and Q11 QPS increased by 85% and 89% respectively
- Overall geometric mean improvement: 8.1%

### Example 1.3: Memory Tracker Shared Mutex (PR #72375)

**Problem Identified**: The `overcommit_m` mutex in OvercommitTracker caused excessive `native_queued_spin_lock_slowpath` in ClickBench Q8, Q42 on high core count systems.

**Root Cause Analysis**: Most operations only read memory usage but used exclusive locking, creating unnecessary contention.

**Technical Solution**:
```cpp
// Before: Exclusive mutex for all operations
class OvercommitTracker {
    mutable std::mutex overcommit_m;
    
    size_t getMemoryUsage() const {
        std::lock_guard<std::mutex> lock(overcommit_m);  // Exclusive
        return memory_usage;
    }
};

// After: Reader/writer lock for better concurrency
class OvercommitTracker {
    mutable std::shared_mutex overcommit_m;
    
    size_t getMemoryUsage() const {
        std::shared_lock<std::shared_mutex> lock(overcommit_m);  // Shared
        return memory_usage;
    }
    
    void updateMemoryUsage(size_t delta) {
        std::unique_lock<std::shared_mutex> lock(overcommit_m);  // Exclusive
        memory_usage += delta;
    }
};
```

**Performance Impact**:
- Overall geometric mean: 6.8% improvement
- Q8: 77% improvement, Q24: 19.5%, Q26: 19.5%, Q42: 11.4%
- No regressions observed

---

## Methodology 2: Intelligent Thread Scheduling and SMT Optimization

**Core Principle**: Many-core systems require careful balance between physical cores and SMT threads. The optimal strategy depends on workload characteristics, memory bandwidth, and data access patterns.

**Detection Method**: Monitor CPU utilization patterns, context switches, and memory bandwidth saturation. Use `lstopo` to understand NUMA topology and SMT pairing.

### Example 2.1: SMT Threshold Tuning (PR #69548 + #70111)

**Problem Identified**: Default SMT thresholds were too conservative, leaving performance on the table for memory-bound workloads on high core count systems.

**Root Cause Analysis**: The original threshold of 32 cores for SMT activation was based on older processors. Modern Intel Xeon with improved memory controllers and larger caches can benefit from SMT at higher core counts.

**Technical Solution**:
```cpp
// Before: Conservative SMT threshold
constexpr size_t SMT_THRESHOLD = 32;

bool use_smt = (physical_cores <= SMT_THRESHOLD);

// After: Data-driven SMT threshold  
constexpr size_t SMT_THRESHOLD = 64;  // Increased based on testing

bool use_smt = (physical_cores <= SMT_THRESHOLD) && 
               (memory_bandwidth_sufficient() || workload_is_compute_bound());
```

**Experimental Results** (80×2 vCPU system):

| vCPUs | use_vCPUs | use_physical_cores | Ratio |
|-------|-----------|-------------------|-------|
| 16    | 5.79      | 4.71              | 81.3% |
| 32    | 8.38      | 7.40              | 88.3% |
| 48    | 9.48      | 8.81              | 93.0% |
| 64    | 9.93      | 9.76              | 98.2% |
| 80    | 9.63      | 10.25             | 106.3%|
| 96    | 10.13     | 10.92             | 107.9%|
| 112   | 10.28     | 11.59             | 112.7%|

![PR69548 figure 1](assets/prs/69548_img1.png)
*SMT performance crossover analysis*

**Key Insight**: SMT becomes beneficial when physical cores > 64, providing 12-16% improvement on memory-bound analytical workloads.

### Example 2.2: Concurrent Read Threshold Optimization (PR #69547)

**Problem Identified**: Default `min_marks_for_concurrent_read=24` created too many small tasks, overwhelming the scheduler on high core count systems.

**Root Cause Analysis**: Pipeline visualization showed excessive task creation overhead. CPU utilization was low due to coordination costs exceeding useful work for small data ranges.

**Technical Solution**:
```cpp
// Before: Fixed small threshold
constexpr size_t MIN_MARKS_FOR_CONCURRENT_READ = 24;

// After: Adaptive threshold based on system size
size_t getOptimalConcurrentReadThreshold() {
    size_t num_cores = std::thread::hardware_concurrency();
    if (num_cores <= 32) return 24;
    if (num_cores <= 64) return 16;  // Sweet spot for mid-range
    return 12;  // Aggressive parallelization for large systems
}
```

**Performance Results**:
- Overall geometric mean: 4.3% improvement
- Q20: 12.6% improvement with 15% higher CPU utilization
- 10+ queries achieved >10% improvement

![PR69547 figure 1](assets/prs/69547_img1.png)
*Before: Fragmented pipeline with low CPU utilization*

![PR69547 figure 2](assets/prs/69547_img2.png)
*After: Compact pipeline with higher CPU utilization*

**Tuning Guidelines**:
1. Monitor pipeline visualization for task granularity
2. Adjust threshold based on: core count, memory bandwidth, typical data size
3. Validate with CPU utilization metrics - target >80% on analytical workloads

---

## Methodology 3: Parallel Hash Operations and Merge Optimization

**Core Principle**: Hash-based operations (GROUP BY, DISTINCT, JOIN) are memory-intensive and benefit greatly from parallelization. However, naive parallelization can create bottlenecks during merge phases.

**Detection Method**: Use pipeline visualization to identify serial merge bottlenecks. Monitor hash table conversion overhead and memory allocation patterns.

### Example 3.1: Parallel Hash Set Conversion (PR #50748)

**Problem Identified**: ClickBench Q5 showed severe performance degradation as core count increased from 80 to 112 threads. Pipeline analysis revealed a serial bottleneck in hash set conversion.

**Root Cause Analysis**: When merging hash sets of different levels (singleLevel + twoLevel), the singleLevel sets had to be converted to twoLevel serially before merging, creating a critical section.

**Technical Solution**:
```cpp
// Before: Serial conversion during merge
void mergeHashSets(HashSet& lhs, HashSet& rhs) {
    if (lhs.isSingleLevel() && rhs.isTwoLevel()) {
        lhs.convertToTwoLevel();  // Serial conversion!
    }
    // Then merge...
}

// After: Parallel conversion before merge
class ParallelHashSetConverter {
    ThreadPool conversion_pool;
    
    void convertAllToTwoLevel(std::vector<HashSet*>& sets) {
        std::vector<std::future<void>> futures;
        
        for (auto* set : sets) {
            if (set->isSingleLevel()) {
                futures.push_back(conversion_pool.scheduleOrThrowOnError([set]() {
                    set->convertToTwoLevel();
                }));
            }
        }
        
        // Wait for all conversions to complete
        for (auto& future : futures) {
            future.get();
        }
    }
};
```

**Performance Impact**:
- Q5: 264% performance improvement on 2×112 vCPU system
- 24 queries achieved >5% improvement
- Overall geometric mean: 7.4% improvement

![PR50748 figure 1](assets/prs/50748_img1.png)
*Performance degradation with increased core count (before)*

![PR50748 figure 2](assets/prs/50748_img2.png)
*Pipeline visualization showing serial bottleneck (max_threads=80)*

![PR50748 figure 3](assets/prs/50748_img3.png)
*Pipeline visualization showing serial bottleneck (max_threads=112)*

![PR50748 figure 4](assets/prs/50748_img4.png)
*Performance improvement after parallel conversion*

### Example 3.2: Parallel Merge with Key (PR #68441)

**Problem Identified**: GROUP BY operations with large hash tables were merging serially, underutilizing available cores.

**Root Cause Analysis**: The merge-with-key implementation only parallelized the initial aggregation but not the final merge phase, creating a bottleneck for queries with high cardinality.

**Technical Solution**:
```cpp
// Before: Serial merge with key
void mergeWithKey(const std::vector<HashTable>& tables) {
    HashTable result;
    for (const auto& table : tables) {
        result.merge(table);  // Serial merge
    }
}

// After: Parallel merge with cancellation support
class ParallelMergeWithKey {
    ThreadPool merge_pool;
    std::atomic<bool> cancelled{false};
    
    void parallelMerge(std::vector<HashTable>& tables) {
        // Convert to two-level if size threshold exceeded
        if (getTotalSize(tables) > PARALLEL_MERGE_THRESHOLD) {
            convertToTwoLevel(tables);
            
            // Merge each bucket in parallel
            std::vector<std::future<void>> futures;
            for (size_t bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
                futures.push_back(merge_pool.scheduleOrThrowOnError([&, bucket]() {
                    if (cancelled.load()) return;
                    mergeBucket(tables, bucket);
                }));
            }
            
            waitForCompletion(futures);
        } else {
            // Keep single-level for small tables
            serialMerge(tables);
        }
    }
};
```

**Performance Impact**:
- Q8: 10.3% improvement, Q9: 7.6% improvement
- No regressions on other queries
- Better CPU utilization during merge phase

![PR68441 figure 1](assets/prs/68441_img1.png)
*Pipeline showing parallel merge optimization*

---

## Methodology 4: SIMD Optimization and Vectorization

**Core Principle**: Modern Intel processors provide powerful SIMD instructions (AVX2, AVX-512) that can process multiple data elements simultaneously. Effective vectorization requires careful algorithm design and compiler optimization.

**Detection Method**: Use Intel VTune to analyze SIMD utilization. Look for scalar operations in hot loops and opportunities for data-parallel processing.

### Example 4.1: Auto-Vectorization for Binary Operations (PR #57343)

**Problem Identified**: Arithmetic operations like addition and multiplication were not fully utilizing available SIMD lanes, leaving performance on the table.

**Root Cause Analysis**: The compiler's auto-vectorization was being inhibited by complex control flow and non-obvious data dependencies in the binary operation implementations.

**Technical Solution**:
```cpp
// Before: Scalar implementation with complex branching
template<typename T>
void binaryOperation(const T* a, const T* b, T* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (likely(isValid(a[i]) && isValid(b[i]))) {
            result[i] = a[i] + b[i];  // Scalar operation
        } else {
            result[i] = handleSpecialCase(a[i], b[i]);
        }
    }
}

// After: Vectorization-friendly implementation
template<typename T>
void binaryOperationVectorized(const T* a, const T* b, T* result, size_t size) {
    size_t vectorized_size = size - (size % SIMD_WIDTH);
    
    // Vectorized main loop - compiler can auto-vectorize
    #pragma GCC ivdep  // Assume no dependencies
    for (size_t i = 0; i < vectorized_size; i += SIMD_WIDTH) {
        // Simple loop body enables vectorization
        for (size_t j = 0; j < SIMD_WIDTH; ++j) {
            result[i + j] = a[i + j] + b[i + j];
        }
    }
    
    // Handle remainder scalarly
    for (size_t i = vectorized_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}
```

**Performance Impact**:
- Q29: 5% improvement on 2×80 vCPU system
- Overall runtime improvement: 1.6% across 43 queries
- Better utilization of AVX2/AVX-512 units

### Example 4.2: Optimized SIMD String Search (PR #46289)

**Problem Identified**: String search operations (LIKE, substring matching) were major bottlenecks in text-heavy queries, with Q20 showing significant performance issues.

**Root Cause Analysis**: The existing SIMD string searcher only checked the first character, leading to many false positives and expensive full string comparisons.

**Technical Solution**:
```cpp
// Before: Single character SIMD search
class StringSearcher {
    __m256i first_char_pattern;
    
    size_t search(const char* haystack, size_t size) {
        for (size_t i = 0; i < size; i += 32) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(haystack + i));
            __m256i matches = _mm256_cmpeq_epi8(chunk, first_char_pattern);
            
            uint32_t mask = _mm256_movemask_epi8(matches);
            while (mask) {
                size_t pos = __builtin_ctz(mask);
                if (fullStringMatch(haystack + i + pos)) {  // Expensive!
                    return i + pos;
                }
                mask &= mask - 1;
            }
        }
        return std::string::npos;
    }
};

// After: Two-character SIMD search with early rejection
class OptimizedStringSearcher {
    __m256i first_char_pattern;
    __m256i second_char_pattern;
    
    size_t search(const char* haystack, size_t size) {
        for (size_t i = 0; i < size; i += 32) {
            __m256i chunk1 = _mm256_loadu_si256((__m256i*)(haystack + i));
            __m256i chunk2 = _mm256_loadu_si256((__m256i*)(haystack + i + 1));
            
            __m256i first_matches = _mm256_cmpeq_epi8(chunk1, first_char_pattern);
            __m256i second_matches = _mm256_cmpeq_epi8(chunk2, second_char_pattern);
            __m256i combined = _mm256_and_si256(first_matches, second_matches);
            
            uint32_t mask = _mm256_movemask_epi8(combined);
            while (mask) {
                size_t pos = __builtin_ctz(mask);
                if (fullStringMatch(haystack + i + pos)) {  // Much fewer calls!
                    return i + pos;
                }
                mask &= mask - 1;
            }
        }
        return std::string::npos;
    }
};
```

**Performance Impact**:
- Q20: 35% performance improvement
- String search related queries: ~10% improvement
- Overall geometric mean: 4.1% improvement
- Significant reduction in false positive string comparisons

**Additional Optimizations**:
- Fast path for single-character needles
- Optimized handling of short strings
- Better cache utilization through reduced memory access

---

## Methodology 5: Micro-Optimization of Hot Loops

**Core Principle**: In tight inner loops executed billions of times, even small inefficiencies compound dramatically. Focus on eliminating redundant operations, reducing function call overhead, and optimizing branch patterns.

**Detection Method**: Use CPU profiling to identify functions consuming >5% of cycles. Analyze assembly output to spot redundant operations and branch mispredictions.

### Example 5.1: Sparse Column Filter Optimization (PR #64426)

**Problem Identified**: ClickBench Q10 spent over 30% of CPU cycles in `ColumnSparse::filter`, making it the dominant hotspot.

**Root Cause Analysis**: Assembly analysis revealed that `isDefault()` was being called twice in each loop iteration - once explicitly and once inside the `++offset_it` operator.

**Technical Solution**:
```cpp
// Before: Redundant isDefault() calls
void filter(const IColumn::Filter& filter) {
    for (auto offset_it = offsets.begin(); offset_it != offsets.end(); ++offset_it) {
        if (!offset_it.isDefault()) {  // First call
            // Process non-default value
            processValue(*offset_it);
        }
        ++offset_it;  // Contains another isDefault() call internally!
    }
}

// Iterator's operator++ implementation (called above)
Iterator& operator++() {
    ++current_offset;
    if (isDefault()) {  // Second call - redundant!
        skipToNextNonDefault();
    }
    return *this;
}

// After: Eliminate redundant isDefault() calls
void filter(const IColumn::Filter& filter) {
    for (auto offset_it = offsets.begin(); offset_it != offsets.end();) {
        bool is_default = offset_it.isDefault();  // Single call
        
        if (!is_default) {
            processValue(*offset_it);
            offset_it.increaseCurrentOffset();  // Direct increment
        } else {
            offset_it.increaseCurrentRow();     // Skip default
        }
    }
}
```

**Performance Impact**:
- Q10: 9.6% QPS improvement
- Total CPU cycles reduced to 79.2% of original
- `ColumnSparse::filter` cycles reduced to 46.4% of original
- Q7, Q11, Q20: 7.3%, 8.7%, 7.3% improvements respectively
- Overall geometric mean: 2.4% improvement

![PR64426 figure 1](assets/prs/64426_img1.png)
*Performance metrics showing cycle reduction*

**Key Lessons**:
1. Profile at instruction level for hot loops
2. Examine operator implementations for hidden costs
3. Consider manual loop unrolling for critical paths
4. Validate optimizations with cycle-accurate measurements

---

## Summary: Systematic Optimization Methodology

This analysis of 22 merged ClickHouse optimizations reveals five key methodologies for maximizing performance on Intel high core count processors:

### 1. **Cache Line Contention Elimination**
- **Impact**: 20-40% improvement on 100+ core systems
- **Key Techniques**: Alignment, lock-free algorithms, reader/writer locks
- **Detection**: `perf c2c`, cache miss analysis

### 2. **Intelligent Thread Scheduling**
- **Impact**: 4-16% improvement through SMT optimization
- **Key Techniques**: Adaptive thresholds, workload-aware scheduling
- **Detection**: Pipeline visualization, CPU utilization analysis

### 3. **Parallel Hash Operations**
- **Impact**: Up to 264% improvement on hash-heavy workloads
- **Key Techniques**: Parallel conversion, bucket-level parallelization
- **Detection**: Serial bottleneck identification in merge phases

### 4. **SIMD Optimization**
- **Impact**: 5-35% improvement on compute-intensive operations
- **Key Techniques**: Auto-vectorization, algorithm redesign for SIMD
- **Detection**: VTune SIMD analysis, assembly inspection

### 5. **Micro-Optimization**
- **Impact**: 2-10% improvement through hot loop optimization
- **Key Techniques**: Redundancy elimination, branch optimization
- **Detection**: Instruction-level profiling, cycle analysis

### Performance Validation Framework

**Hardware Requirements**:
- Intel Xeon with 80+ cores, SMT capability
- High memory bandwidth (>200 GB/s)
- NUMA-aware configuration

**Measurement Protocol**:
1. Baseline with `perf stat` and VTune
2. Apply optimization
3. Measure with same tools + ClickBench
4. Validate with multiple runs (CV < 5%)

**Key Metrics**:
- QPS improvement (geometric mean across queries)
- CPU cycle reduction in hot functions
- Cache miss rate changes
- Context switch overhead

### Deployment Recommendations

**Configuration Tuning**:
```bash
# SMT optimization
max_threads = physical_cores * (1.5 if memory_bound else 1.0)

# Concurrency tuning  
min_marks_for_concurrent_read = max(12, 48 / log2(num_cores))

# Memory optimization
max_memory_usage = 0.8 * available_memory
```

**Monitoring Checklist**:
- [ ] LLC miss rate < 10%
- [ ] Lock contention < 5% of cycles
- [ ] CPU utilization > 80% during queries
- [ ] Context switches < 1000/sec per core
- [ ] Memory bandwidth utilization > 60%

### Future Optimization Opportunities

1. **Intel IAA Integration**: Leverage In-Memory Analytics Accelerator for compression/decompression
2. **AMX Utilization**: Explore Advanced Matrix Extensions for ML workloads
3. **NUMA-Aware Scheduling**: Optimize data placement and thread affinity
4. **Dynamic CPU Dispatch**: Runtime optimization based on detected CPU features

---

## References and Resources

- **Source Code**: All optimizations available in ClickHouse main branch
- **Slide Deck**: [Shanghai Meetup Presentation](https://github.com/ClickHouse/clickhouse-presentations/blob/master/2025-meetup-Shanghai-1/Talk%204%20-%20Intel%20-%20Shanghai%20Meetup_01Mar25.pdf)
- **Pull Requests**: 22 merged PRs with detailed performance analysis
- **Intel Optimization Guide**: [ClickHouse on Intel Architecture](https://www.intel.com/content/www/us/en/developer/articles/guide/clickhouse-iaa-iavx512-4th-gen-xeon-scalable.html)

### Acknowledgments

Special thanks to the ClickHouse community for rigorous code review and performance validation. These optimizations represent collaborative effort between Intel and ClickHouse teams to unlock the full potential of modern many-core processors.

---

*For questions about implementation details or performance reproduction, please refer to the individual PR discussions linked throughout this post.*
