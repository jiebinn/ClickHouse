#pragma once

#include <Common/ThreadGroupSwitcher.h>
#include <Common/HashTable/HashSet.h>
#include <Common/ThreadPool.h>
#include <Common/scope_guard_safe.h>
#include <Common/setThreadName.h>
#include <Common/threadPoolCallbackRunner.h>
#include <Common/VectorWithMemoryTracking.h>

#include <atomic>
#include <memory>
#include <utility>

namespace DB
{

namespace ErrorCodes
{
extern const int TOO_LARGE_ARRAY_SIZE;
}

enum class SetLevelHint
{
    singleLevel,
    twoLevel,
    unknown,
};

template <typename SingleLevelSet, typename TwoLevelSet>
class UniqExactSet
{
    static_assert(std::is_same_v<typename SingleLevelSet::value_type, typename TwoLevelSet::value_type>);
    static_assert(std::is_same_v<typename SingleLevelSet::Cell::State, HashTableNoState>);

    /// The two-level set together with a flag that records whether it has ever been handed out for sharing.
    /// `getTwoLevelSet` may return this pointee to another `UniqExactSet` (see the fast path in `merge`), so several
    /// destinations can hold the same instance and read it concurrently. `is_shared` is set (once, monotonically) the
    /// moment the pointee escapes, and `doDeepCopyIfNeeded` forks before any in-place mutation once it is set. This
    /// replaces the previous `shared_ptr::use_count()` guard, which is not a synchronization primitive: another holder
    /// dropping its reference (count 2 -> 1) does not establish a happens-before with the memory of the pointee, so a
    /// concurrent reader could still race an in-place writer that observed count == 1.
    struct SharedTwoLevelSet
    {
        TwoLevelSet set;
        std::atomic<bool> is_shared{false};

        SharedTwoLevelSet() = default;
        explicit SharedTwoLevelSet(size_t size_hint) : set(size_hint) {}
        template <typename Source>
        explicit SharedTwoLevelSet(const Source & src) : set(src) {}
    };

public:
    using value_type = typename SingleLevelSet::value_type;

    template <typename Arg, SetLevelHint hint>
    auto ALWAYS_INLINE insert(Arg && arg)
    {
        if constexpr (hint == SetLevelHint::singleLevel)
        {
            asSingleLevel().insert(std::forward<Arg>(arg));
        }
        else if constexpr (hint == SetLevelHint::twoLevel)
        {
            asTwoLevel().insert(std::forward<Arg>(arg));
        }
        else
        {
            if (isSingleLevel())
            {
                auto && [_, inserted] = asSingleLevel().insert(std::forward<Arg>(arg));
                if (inserted && worthConvertingToTwoLevel(asSingleLevel().size()))
                    convertToTwoLevel();
            }
            else
            {
                asTwoLevel().insert(std::forward<Arg>(arg));
            }
        }
    }

    /// In merge, if one of the lhs and rhs is twolevelset and the other is singlelevelset, then the singlelevelset will need to convertToTwoLevel().
    /// It's not in parallel and will cost extra large time if the thread_num is large.
    /// This method will convert all the SingleLevelSet to TwoLevelSet in parallel if the hashsets are not all singlelevel or not all twolevel.
    /// Accepts a container of places and an accessor that returns `UniqExactSet *` for each element.
    /// This avoids building an intermediate vector of pointers.
    template <typename Places, typename Accessor>
    static void parallelizeMergePrepare(const Places & places, Accessor && accessor, ThreadPool & thread_pool, std::atomic<bool> & is_cancelled)
    {
        UInt64 single_level_set_num = 0;
        UInt64 all_single_hash_size = 0;

        for (size_t i = 0; i < places.size(); ++i)
        {
            if (accessor(places[i])->isSingleLevel())
                single_level_set_num ++;
        }

        if (single_level_set_num == places.size())
        {
            for (size_t i = 0; i < places.size(); ++i)
                all_single_hash_size += accessor(places[i])->size();
        }

        /// If all the hashtables are mixed by singleLevel and twoLevel, or all singleLevel (larger than 6000 for average value), they could be converted into
        /// twoLevel hashtables in parallel and then merge together. please refer to the following PR for more details.
        /// https://github.com/ClickHouse/ClickHouse/pull/50748
        /// https://github.com/ClickHouse/ClickHouse/pull/52973
        if ((single_level_set_num > 0 && single_level_set_num < places.size()) || ((all_single_hash_size/places.size()) > 6000))
        {
            try
            {
                auto data_vec_atomic_index = std::make_shared<std::atomic_uint32_t>(0);
                auto thread_func = [&places, &accessor, data_vec_atomic_index, &is_cancelled, thread_group = getCurrentThreadGroup()]()
                {
                    ThreadGroupSwitcher switcher(thread_group, ThreadName::UNIQ_EXACT_CONVERT);

                    while (true)
                    {
                        if (is_cancelled.load(std::memory_order_seq_cst))
                            return;

                        const auto i = data_vec_atomic_index->fetch_add(1);
                        if (i >= places.size())
                            return;
                        if (accessor(places[i])->isSingleLevel())
                            accessor(places[i])->convertToTwoLevel();
                    }
                };
                for (size_t i = 0; i < std::min<size_t>(thread_pool.getMaxThreads(), single_level_set_num); ++i)
                    thread_pool.scheduleOrThrowOnError(thread_func);

                thread_pool.wait();
            }
            catch (...)
            {
                thread_pool.wait();
                throw;
            }
        }
    }

    /// Batch merge multiple UniqExactSet into the first one in parallel.
    /// Each thread processes one bucket at a time across all hash tables,
    /// reducing thread pool overhead from O(N) to O(1) compared to pairwise merge.
    /// Accepts a container of places and an accessor that returns `UniqExactSet *` for each element.
    template <typename Places, typename Accessor>
    static void parallelizeMergeMulti(const Places & places, Accessor && accessor, ThreadPool & thread_pool, std::atomic<bool> & is_cancelled)
    {
        if (places.size() <= 1)
            return;

        auto * first = accessor(places[0]);

        /// If not all are two-level, fall back to pairwise merge with thread pool.
        bool all_two_level = true;
        for (size_t i = 0; i < places.size(); ++i)
        {
            if (!accessor(places[i])->isTwoLevel())
            {
                all_two_level = false;
                break;
            }
        }

        if (!all_two_level)
        {
            for (size_t j = 1; j < places.size(); ++j)
            {
                if (is_cancelled.load(std::memory_order_seq_cst))
                    return;
                first->merge(*accessor(places[j]), &thread_pool, &is_cancelled);
            }
            return;
        }

        /// All sets are two-level, perform parallel bucket-wise merge.
        auto & first_two_level = first->asTwoLevelChecked();
        constexpr size_t NUM_BUCKETS = TwoLevelSet::NUM_BUCKETS;

        /// Pre-fetch all two-level set pointers to avoid concurrent access to getTwoLevelSet().
        VectorWithMemoryTracking<TwoLevelSet *> two_level_ptrs;
        two_level_ptrs.reserve(places.size());
        for (size_t i = 0; i < places.size(); ++i)
            two_level_ptrs.emplace_back(&accessor(places[i])->asTwoLevelChecked());

        ThreadPoolCallbackRunnerLocal<void> runner(thread_pool, ThreadName::UNIQ_EXACT_MERGER);
        try
        {
            auto next_bucket_to_merge = std::make_shared<std::atomic_uint32_t>(0);

            auto thread_func = [&two_level_ptrs, &first_two_level, next_bucket_to_merge, &is_cancelled]()
            {
                while (true)
                {
                    if (is_cancelled.load(std::memory_order_seq_cst))
                        return;

                    const auto bucket = next_bucket_to_merge->fetch_add(1);
                    if (bucket >= NUM_BUCKETS)
                        return;

                    for (size_t j = 1; j < two_level_ptrs.size(); ++j)
                    {
                        if (is_cancelled.load(std::memory_order_seq_cst))
                            return;

                        first_two_level.impls[bucket].merge(two_level_ptrs[j]->impls[bucket]);
                    }
                }
            };

            const size_t max_threads_to_enqueue = std::min<size_t>(thread_pool.getMaxThreads(), NUM_BUCKETS);
            for (size_t i = 0; i < max_threads_to_enqueue
                 && next_bucket_to_merge->load(std::memory_order_relaxed) < NUM_BUCKETS; ++i)
                runner.enqueueAndKeepTrack(thread_func, Priority{});
        }
        catch (...)
        {
            is_cancelled.store(true);
            throw;
        }
        runner.waitForAllToFinishAndRethrowFirstError();
    }

    auto merge(const UniqExactSet & other, ThreadPool * thread_pool = nullptr, std::atomic<bool> * is_cancelled = nullptr)
    {
        if (size() == 0 && worthConvertingToTwoLevel(other.size()))
        {
            two_level_set = other.getTwoLevelSet();
            return;
        }

        if (isSingleLevel() && other.isTwoLevel())
            convertToTwoLevel();

        if (isSingleLevel())
        {
            asSingleLevel().merge(other.asSingleLevel());
        }
        else
        {
            auto & lhs = asTwoLevelChecked();

            if (other.isSingleLevel())
                return lhs.merge(other.asSingleLevel());

            /// `rhs_ptr` keeps the pointee alive; `rhs` is its two-level set. `getTwoLevelSet` marked the pointee shared,
            /// so no other holder can mutate it in place while we read its buckets here.
            const auto rhs_ptr = other.getTwoLevelSet();
            const auto & rhs = rhs_ptr->set;
            if (!thread_pool)
            {
                for (size_t i = 0; i < rhs.NUM_BUCKETS; ++i)
                {
                    lhs.impls[i].merge(rhs.impls[i]);
                }
            }
            else
            {

                /// Usage of lhs and rhs is fine. The references belong to *this and will outlive `runner`, so the order of destruction is ok
                ThreadPoolCallbackRunnerLocal<void> runner(*thread_pool, ThreadName::UNIQ_EXACT_MERGER);
                try
                {
                    auto next_bucket_to_merge = std::make_shared<std::atomic_uint32_t>(0);

                    auto thread_func = [&lhs, &rhs, next_bucket_to_merge, is_cancelled]()
                    {
                        while (true)
                        {
                            if (is_cancelled->load())
                                return;

                            const auto bucket = next_bucket_to_merge->fetch_add(1);
                            if (bucket >= rhs.NUM_BUCKETS)
                                return;
                            lhs.impls[bucket].merge(rhs.impls[bucket]);
                        }
                    };

                    const size_t max_threads_to_enqueue = std::min<size_t>(thread_pool->getMaxThreads(), rhs.NUM_BUCKETS);
                    for (size_t i = 0; i < max_threads_to_enqueue
                         && next_bucket_to_merge->load(std::memory_order_relaxed) < rhs.NUM_BUCKETS; ++i)
                        runner.enqueueAndKeepTrack(thread_func, Priority{});
                }
                catch (...)
                {
                    is_cancelled->store(true);
                    throw;
                }
                runner.waitForAllToFinishAndRethrowFirstError();
            }
        }
    }

    void read(ReadBuffer & in)
    {
        size_t new_size = 0;
        readVarUInt(new_size, in);
        if (new_size > 100'000'000'000)
            throw DB::Exception(
                DB::ErrorCodes::TOO_LARGE_ARRAY_SIZE, "The size of serialized hash table is suspiciously large: {}", new_size);

        if (worthConvertingToTwoLevel(new_size))
        {
            two_level_set = std::make_shared<SharedTwoLevelSet>(new_size);
            for (size_t i = 0; i < new_size; ++i)
            {
                typename SingleLevelSet::Cell x;
                x.read(in);
                asTwoLevel().insert(x.getValue());
            }
        }
        else
        {
            asSingleLevel().reserve(new_size);

            for (size_t i = 0; i < new_size; ++i)
            {
                typename SingleLevelSet::Cell x;
                x.read(in);
                asSingleLevel().insert(x.getValue());
            }
        }
    }

    void write(WriteBuffer & out) const
    {
        if (isSingleLevel())
            asSingleLevel().write(out);
        else
            /// We have to preserve compatibility with the old implementation that used only single level hash sets.
            asTwoLevel().writeAsSingleLevel(out);
    }

    size_t size() const { return isSingleLevel() ? asSingleLevel().size() : asTwoLevel().size(); }

    /// To convert set to two level before merging (we cannot just call convertToTwoLevel() on the right hand side set, because it is const).
    /// This is `const` and may run concurrently for the same source set (ROLLUP/CUBE/GROUPING SETS merge one state into several
    /// destinations at once), so it must not mutate the buckets of `*this`. Handing out the pointee makes it reachable from another
    /// `UniqExactSet`, so mark it shared before returning: from then on `doDeepCopyIfNeeded` forks before any in-place mutation of it,
    /// keeping the shared instance read-only. Concurrent calls only store the same `true` into the flag, which is safe.
    std::shared_ptr<SharedTwoLevelSet> getTwoLevelSet() const
    {
        if (two_level_set)
        {
            two_level_set->is_shared.store(true, std::memory_order_release);
            return two_level_set;
        }
        /// Freshly built from the single-level set: solely owned by the caller, so it stays unshared and can be mutated in place.
        return std::make_shared<SharedTwoLevelSet>(asSingleLevel());
    }

    static bool worthConvertingToTwoLevel(size_t size) { return size > 100'000; }

    void convertToTwoLevel()
    {
        /// Already two-level: nothing to convert. (Rebuilding from the now-empty single-level set here would
        /// discard the existing data -- the single-level set is cleared once the pointee is built.)
        if (two_level_set)
            return;

        /// Build a fresh, solely-owned pointee from the single-level set. It is not marked shared, so it can be
        /// mutated in place until it is handed out via `getTwoLevelSet`.
        two_level_set = std::make_shared<SharedTwoLevelSet>(asSingleLevel());
        single_level_set.clear();
    }

    bool isSingleLevel() const { return !two_level_set; }
    bool isTwoLevel() const { return !!two_level_set; }

private:
    SingleLevelSet & asSingleLevel() { return single_level_set; }
    const SingleLevelSet & asSingleLevel() const { return single_level_set; }

    TwoLevelSet & asTwoLevelChecked()
    {
        doDeepCopyIfNeeded();
        return two_level_set->set;
    }

    TwoLevelSet & asTwoLevel() { return two_level_set->set; }
    const TwoLevelSet & asTwoLevel() const { return two_level_set->set; }

    /// Fork a private copy before mutating a `two_level_set` that may be shared (adopted by another `UniqExactSet`
    /// via the fast path in `merge`, or handed out for reading). The fork decision uses the pointee's `is_shared`
    /// flag, not `shared_ptr::use_count()`: `use_count` is not a synchronization primitive across threads, whereas
    /// `is_shared` only ever goes false -> true (release) before the pointee escapes, so a shared pointee is never
    /// mutated in place. The forked copy is solely owned and unshared, so later mutations stay in place.
    void doDeepCopyIfNeeded()
    {
        if (two_level_set && two_level_set->is_shared.load(std::memory_order_acquire))
        {
            const auto & src = two_level_set->set;
            auto copy = std::make_shared<SharedTwoLevelSet>(src.size());
            for (size_t i = 0; i < TwoLevelSet::NUM_BUCKETS; ++i)
                copy->set.impls[i].merge(src.impls[i]);
            two_level_set = std::move(copy);
        }
    }

    SingleLevelSet single_level_set;
    std::shared_ptr<SharedTwoLevelSet> two_level_set;
};
}
