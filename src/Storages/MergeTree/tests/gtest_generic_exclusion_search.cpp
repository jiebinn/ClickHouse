#include <gtest/gtest.h>

#include <Storages/MergeTree/GenericExclusionSearch.h>

#include <limits>
#include <random>

using namespace DB;

namespace
{

MarkRangeCheck oracleFromFlags(const std::vector<bool> & matching_marks)
{
    return [&matching_marks](const MarkRange & range)
    {
        bool can_be_true = false;
        bool can_be_false = false;
        for (size_t mark = range.begin; mark != range.end; ++mark)
        {
            if (matching_marks[mark])
                can_be_true = true;
            else
                can_be_false = true;
        }
        return BoolMask(can_be_true, can_be_false);
    };
}

/// The expected search result for a precise oracle: the runs of matching marks, where two runs
/// separated by a gap of at most `max_gap` marks are merged into one range.
MarkRanges runsOfMatchingMarks(const std::vector<bool> & matching_marks, size_t max_gap)
{
    MarkRanges expected;
    for (size_t mark = 0; mark != matching_marks.size(); ++mark)
    {
        if (!matching_marks[mark])
            continue;

        if (!expected.empty() && mark - expected.back().end <= max_gap)
            expected.back().end = mark + 1;
        else
            expected.emplace_back(mark, mark + 1);
    }
    return expected;
}

MarkRanges makeRanges(const std::vector<MarkRange> & ranges)
{
    return MarkRanges(ranges.begin(), ranges.end());
}

bool covers(const MarkRanges & ranges, size_t mark)
{
    for (const auto & range : ranges)
        if (range.begin <= mark && mark < range.end)
            return true;
    return false;
}

bool isContained(const MarkRange & inner, const MarkRanges & outer)
{
    for (const auto & range : outer)
        if (range.begin <= inner.begin && inner.end <= range.end)
            return true;
    return false;
}

std::vector<bool> randomFlags(size_t size, double density, std::mt19937 & rng)
{
    std::bernoulli_distribution distribution(density);
    std::vector<bool> flags(size);
    for (size_t i = 0; i != size; ++i)
        flags[i] = distribution(rng);
    return flags;
}

}

TEST(GenericExclusionSearch, UnlimitedMatchesBruteForce)
{
    /// The exhaustive search must reproduce the brute-force expectation for arbitrary patterns,
    /// split factors, and seek thresholds.
    std::mt19937 rng(42);

    for (size_t coarse_index_granularity : {2, 3, 8})
    {
        for (size_t min_marks_for_seek : {0, 2, 1000})
        {
            for (double density : {0.05, 0.5, 0.95})
            {
                auto matching = randomFlags(200, density, rng);
                GenericExclusionSearchSettings settings{.coarse_index_granularity = coarse_index_granularity, .max_steps = 0, .min_marks_for_seek = min_marks_for_seek};

                auto result = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracleFromFlags(matching), settings, true);

                EXPECT_EQ(result.ranges, runsOfMatchingMarks(matching, min_marks_for_seek));
                EXPECT_EQ(result.exact_ranges, runsOfMatchingMarks(matching, 0));
                EXPECT_FALSE(result.reached_step_limit);
            }
        }
    }
}

TEST(GenericExclusionSearch, ExactRangesNotMergedAcrossSeekGap)
{
    /// Two matching runs separated by a gap smaller than the seek threshold: the ranges to read are
    /// merged across the gap, but the exact ranges must stay separate, because the gap marks do not
    /// match.
    std::vector<bool> matching(100);
    for (size_t mark = 10; mark != 14; ++mark)
        matching[mark] = true;
    for (size_t mark = 20; mark != 24; ++mark)
        matching[mark] = true;

    for (size_t max_steps : {0, 1000})
    {
        GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 7};

        auto result = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracleFromFlags(matching), settings, true);

        EXPECT_EQ(result.ranges, makeRanges({{10, 24}}));
        EXPECT_EQ(result.exact_ranges, makeRanges({{10, 14}, {20, 24}}));
    }
}

TEST(GenericExclusionSearch, MinMarksForSeekMergingForRes)
{
    /// A gap of exactly `min_marks_for_seek` marks is merged in the ranges to read, a gap of one
    /// more mark is not, and the exact ranges are unaffected either way.
    std::vector<bool> close_runs(50);
    close_runs[5] = close_runs[6] = true;
    close_runs[10] = close_runs[11] = true;

    std::vector<bool> distant_runs(50);
    distant_runs[5] = distant_runs[6] = true;
    distant_runs[11] = distant_runs[12] = true;

    for (size_t max_steps : {0, 1000})
    {
        GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 3};

        auto merged = genericExclusionSearch(makeRanges({{0, close_runs.size()}}), oracleFromFlags(close_runs), settings, true);
        EXPECT_EQ(merged.ranges, makeRanges({{5, 12}}));
        EXPECT_EQ(merged.exact_ranges, makeRanges({{5, 7}, {10, 12}}));

        auto separate = genericExclusionSearch(makeRanges({{0, distant_runs.size()}}), oracleFromFlags(distant_runs), settings, true);
        EXPECT_EQ(separate.ranges, makeRanges({{5, 7}, {11, 13}}));
        EXPECT_EQ(separate.exact_ranges, makeRanges({{5, 7}, {11, 13}}));
    }
}

TEST(GenericExclusionSearch, GapMergeAcrossInitialRanges)
{
    /// The gap between two disjoint initial ranges, as the query condition cache produces them, may
    /// be absorbed by the seek merging of the ranges to read, but never by the exact ranges.
    std::vector<bool> matching(12, true);
    matching[5] = matching[6] = false;

    std::vector<bool> all_matching(10, true);

    for (size_t max_steps : {0, 1000})
    {
        GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 2};

        auto result = genericExclusionSearch(makeRanges({{0, 5}, {7, 12}}), oracleFromFlags(matching), settings, true);

        EXPECT_EQ(result.ranges, makeRanges({{0, 12}}));
        EXPECT_EQ(result.exact_ranges, makeRanges({{0, 5}, {7, 12}}));

        /// Touching initial ranges that both fully match merge in both outputs, because there is no
        /// gap mark between them.
        auto touching = genericExclusionSearch(makeRanges({{0, 5}, {5, 10}}), oracleFromFlags(all_matching), settings, true);

        EXPECT_EQ(touching.ranges, makeRanges({{0, 10}}));
        EXPECT_EQ(touching.exact_ranges, makeRanges({{0, 10}}));
    }
}

TEST(GenericExclusionSearch, MoreInitialRangesThanBudget)
{
    /// Every initial range receives its one check even when there are more of them than the budget,
    /// so the number of checks is bounded by the number of initial ranges rather than the budget.
    std::vector<MarkRange> initial;
    for (size_t i = 0; i != 10; ++i)
        initial.emplace_back(i * 20, i * 20 + 10);

    GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = 3, .min_marks_for_seek = 0};

    /// Fully matching and fully non-matching ranges are classified by their single check, so no
    /// range is ever accepted because of the budget and the limit flag stays unset.
    std::vector<bool> alternating(200);
    for (size_t i = 0; i != 10; i += 2)
        for (size_t mark = i * 20; mark != i * 20 + 10; ++mark)
            alternating[mark] = true;

    auto classified = genericExclusionSearch(makeRanges(initial), oracleFromFlags(alternating), settings, true);
    EXPECT_EQ(classified.num_steps, 10u);
    EXPECT_EQ(classified.ranges, makeRanges({{0, 10}, {40, 50}, {80, 90}, {120, 130}, {160, 170}}));
    EXPECT_EQ(classified.exact_ranges, classified.ranges);
    EXPECT_FALSE(classified.reached_step_limit);

    /// Ambiguous ranges would need splitting, which the budget cannot cover, so they are all
    /// accepted whole and the limit flag is set.
    std::vector<bool> halves(200);
    for (size_t i = 0; i != 10; ++i)
        for (size_t mark = i * 20; mark != i * 20 + 5; ++mark)
            halves[mark] = true;

    auto truncated = genericExclusionSearch(makeRanges(initial), oracleFromFlags(halves), settings, true);
    EXPECT_EQ(truncated.num_steps, 10u);
    EXPECT_EQ(truncated.ranges, makeRanges(initial));
    EXPECT_TRUE(truncated.exact_ranges.empty());
    EXPECT_TRUE(truncated.reached_step_limit);
}

TEST(GenericExclusionSearch, CoarseGranularityLargerThanRange)
{
    /// When the split factor exceeds the range size, a single split produces one subrange per mark,
    /// and the budget gate must account for that fan-out exactly.
    std::vector<bool> matching(50);
    matching[25] = true;
    auto oracle = oracleFromFlags(matching);

    /// The fan-out of the only possible split (50 subranges) does not fit into the budget, so the
    /// range is accepted whole after its single check.
    GenericExclusionSearchSettings small{.coarse_index_granularity = 64, .max_steps = 10, .min_marks_for_seek = 0};
    auto blocked = genericExclusionSearch(makeRanges({{0, 50}}), oracle, small, true);
    EXPECT_EQ(blocked.ranges, makeRanges({{0, 50}}));
    EXPECT_EQ(blocked.num_steps, 1u);
    EXPECT_TRUE(blocked.reached_step_limit);

    /// The same truncating run without exact range collection returns the same ranges to read.
    auto blocked_no_exact = genericExclusionSearch(makeRanges({{0, 50}}), oracle, small, false);
    EXPECT_EQ(blocked_no_exact.ranges, blocked.ranges);
    EXPECT_TRUE(blocked_no_exact.exact_ranges.empty());
    EXPECT_TRUE(blocked_no_exact.reached_step_limit);

    /// A budget that covers the fan-out lets the split happen and the search finds the single
    /// matching mark.
    GenericExclusionSearchSettings ample{.coarse_index_granularity = 64, .max_steps = 60, .min_marks_for_seek = 0};
    auto refined = genericExclusionSearch(makeRanges({{0, 50}}), oracle, ample, true);
    EXPECT_EQ(refined.ranges, makeRanges({{25, 26}}));
    EXPECT_EQ(refined.exact_ranges, makeRanges({{25, 26}}));
    EXPECT_EQ(refined.num_steps, 51u);
    EXPECT_FALSE(refined.reached_step_limit);
}

TEST(GenericExclusionSearch, EqualSizeRangesCheckedLeftToRight)
{
    /// Among ranges of equal size the leftmost one is checked first, which keeps the search
    /// deterministic independently of the heap implementation.
    std::vector<bool> matching(200);
    std::vector<MarkRange> checked_order;
    MarkRangeCheck recording_oracle = [&](const MarkRange & range)
    {
        checked_order.push_back(range);
        return oracleFromFlags(matching)(range);
    };

    GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = 100, .min_marks_for_seek = 0};
    auto result = genericExclusionSearch(makeRanges({{0, 8}, {100, 108}}), recording_oracle, settings, true);

    EXPECT_TRUE(result.ranges.empty());
    ASSERT_EQ(checked_order.size(), 2u);
    EXPECT_EQ(checked_order[0], MarkRange(0, 8));
    EXPECT_EQ(checked_order[1], MarkRange(100, 108));
}

TEST(GenericExclusionSearch, StepLimitBoundsCheckCount)
{
    /// Alternating marks keep every multi-mark range ambiguous, so the exhaustive search descends
    /// all the way to single marks, which is its worst case.
    std::vector<bool> matching(4096);
    for (size_t mark = 0; mark < matching.size(); mark += 2)
        matching[mark] = true;

    auto oracle = oracleFromFlags(matching);

    size_t unlimited_checks = 0;
    MarkRangeCheck counting_unlimited_oracle = [&](const MarkRange & range)
    {
        ++unlimited_checks;
        return oracle(range);
    };

    GenericExclusionSearchSettings unlimited{.coarse_index_granularity = 8, .max_steps = 0, .min_marks_for_seek = 0};
    auto unlimited_result = genericExclusionSearch(makeRanges({{0, matching.size()}}), counting_unlimited_oracle, unlimited, true);
    size_t unlimited_steps = unlimited_result.num_steps;
    EXPECT_EQ(unlimited_checks, unlimited_steps);

    /// The last two budgets probe the exact boundary: the precise number of checks the exhaustive
    /// search needs must complete untruncated, and one check less must not.
    std::vector<size_t> budgets{1, 5, 17, 100, 1000, 100000, unlimited_steps, unlimited_steps - 1};

    for (size_t max_steps : budgets)
    {
        size_t checks = 0;
        MarkRangeCheck counting_oracle = [&](const MarkRange & range)
        {
            ++checks;
            return oracle(range);
        };

        GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 0};
        auto result = genericExclusionSearch(makeRanges({{0, matching.size()}}), counting_oracle, settings, true);

        EXPECT_EQ(checks, result.num_steps);
        EXPECT_LE(checks, max_steps);
        EXPECT_EQ(result.reached_step_limit, max_steps < unlimited_steps);
        if (!result.reached_step_limit)
            EXPECT_EQ(result.ranges, unlimited_result.ranges);
    }
}

TEST(GenericExclusionSearch, StepLimitInvariantsWithMultipleInitialRanges)
{
    /// The budget and multiple initial input ranges (the query condition cache scenario) interact:
    /// whatever the budget, the invariants of the outputs must hold, and marks outside the initial
    /// ranges must not reappear unless the seek merging deliberately bridges a gap.
    std::mt19937 rng(13);
    auto matching = randomFlags(300, 0.3, rng);
    auto oracle = oracleFromFlags(matching);
    auto initial = makeRanges({{0, 50}, {60, 140}, {141, 200}, {260, 300}});

    for (size_t min_marks_for_seek : {0, 3})
    {
        for (size_t max_steps = 1; max_steps != 60; ++max_steps)
        {
            GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = min_marks_for_seek};
            auto result = genericExclusionSearch(initial, oracle, settings, true);

            EXPECT_NO_THROW(assertSortedAndNonIntersecting(result.ranges));
            EXPECT_NO_THROW(assertSortedAndNonIntersecting(result.exact_ranges));

            /// Every matching mark inside the initial ranges must stay covered.
            for (size_t mark = 0; mark != matching.size(); ++mark)
                if (matching[mark] && covers(initial, mark))
                    EXPECT_TRUE(covers(result.ranges, mark)) << "mark " << mark << ", budget " << max_steps;

            /// Without seek merging, nothing outside the initial ranges may appear in the result.
            if (min_marks_for_seek == 0)
                for (const auto & range : result.ranges)
                    EXPECT_TRUE(isContained(range, initial)) << "(" << range.begin << ", " << range.end << "), budget " << max_steps;

            /// Exact ranges never bridge gaps, so they always stay inside the initial ranges and
            /// consist of matching marks only.
            for (const auto & range : result.exact_ranges)
            {
                EXPECT_TRUE(isContained(range, result.ranges)) << "budget " << max_steps;
                EXPECT_TRUE(isContained(range, initial)) << "budget " << max_steps;
                for (size_t mark = range.begin; mark != range.end; ++mark)
                    EXPECT_TRUE(matching[mark]) << "mark " << mark << ", budget " << max_steps;
            }
        }
    }
}

TEST(GenericExclusionSearch, StepLimitBoundsWithUnevenSplits)
{
    /// A range of nine marks splits with factor eight into five subranges (four of two marks and a
    /// last one of a single mark), so the budget gate must account for a fan-out of five, not eight.
    /// This pins the consistency of the gate's subrange count with what the split actually emits.
    std::vector<bool> matching(9);
    for (size_t mark = 0; mark < matching.size(); mark += 2)
        matching[mark] = true;

    size_t checks = 0;
    auto oracle = oracleFromFlags(matching);
    MarkRangeCheck counting_oracle = [&](const MarkRange & range)
    {
        ++checks;
        return oracle(range);
    };

    /// A budget of six covers the initial check plus the five subranges, so the split happens.
    GenericExclusionSearchSettings six{.coarse_index_granularity = 8, .max_steps = 6, .min_marks_for_seek = 0};
    auto split_result = genericExclusionSearch(makeRanges({{0, 9}}), counting_oracle, six, true);
    EXPECT_EQ(checks, 6u);
    EXPECT_EQ(split_result.num_steps, 6u);
    EXPECT_EQ(split_result.exact_ranges, makeRanges({{0, 1}}));
    EXPECT_TRUE(split_result.reached_step_limit);

    /// One check less does not cover the fan-out, so the range is accepted whole after one check.
    checks = 0;
    GenericExclusionSearchSettings five{.coarse_index_granularity = 8, .max_steps = 5, .min_marks_for_seek = 0};
    auto blocked_result = genericExclusionSearch(makeRanges({{0, 9}}), counting_oracle, five, true);
    EXPECT_EQ(checks, 1u);
    EXPECT_EQ(blocked_result.ranges, makeRanges({{0, 9}}));
    EXPECT_TRUE(blocked_result.exact_ranges.empty());
    EXPECT_TRUE(blocked_result.reached_step_limit);
}

TEST(GenericExclusionSearch, LimitedIsAlwaysCoarserThanUnlimited)
{
    /// For any budget, truncation may only coarsen the result: everything the exhaustive search
    /// selects stays selected, exact ranges may only shrink, and the limited search never spends
    /// more checks than the exhaustive one, because it visits a subset of the same recursion tree.
    std::mt19937 rng(37);

    for (double density : {0.2, 0.7})
    {
        auto matching = randomFlags(250, density, rng);
        auto oracle = oracleFromFlags(matching);

        for (size_t min_marks_for_seek : {0, 2})
        {
            GenericExclusionSearchSettings unlimited{.coarse_index_granularity = 8, .max_steps = 0, .min_marks_for_seek = min_marks_for_seek};
            auto exhaustive = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracle, unlimited, true);

            for (size_t max_steps = 1; max_steps != 70; ++max_steps)
            {
                GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = min_marks_for_seek};
                auto limited = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracle, settings, true);

                EXPECT_LE(limited.num_steps, exhaustive.num_steps);

                for (size_t mark = 0; mark != matching.size(); ++mark)
                    if (covers(exhaustive.ranges, mark))
                        EXPECT_TRUE(covers(limited.ranges, mark)) << "mark " << mark << ", budget " << max_steps;

                for (const auto & range : limited.exact_ranges)
                    for (size_t mark = range.begin; mark != range.end; ++mark)
                        EXPECT_TRUE(covers(exhaustive.exact_ranges, mark)) << "mark " << mark << ", budget " << max_steps;
            }
        }
    }
}

TEST(GenericExclusionSearch, StepLimitProducesSupersetAndValidExactRanges)
{
    /// Whatever the budget, the outputs must keep their contracts: sorted and non-intersecting, all
    /// matching marks covered, exact ranges contained in the result and fully matching.
    std::mt19937 rng(7);
    auto matching = randomFlags(300, 0.3, rng);
    auto oracle = oracleFromFlags(matching);

    for (size_t min_marks_for_seek : {0, 3})
    {
        for (size_t max_steps = 1; max_steps != 80; ++max_steps)
        {
            GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = min_marks_for_seek};
            auto result = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracle, settings, true);

            EXPECT_NO_THROW(assertSortedAndNonIntersecting(result.ranges));
            EXPECT_NO_THROW(assertSortedAndNonIntersecting(result.exact_ranges));

            for (size_t mark = 0; mark != matching.size(); ++mark)
                if (matching[mark])
                    EXPECT_TRUE(covers(result.ranges, mark)) << "mark " << mark << ", budget " << max_steps;

            for (const auto & range : result.exact_ranges)
            {
                EXPECT_TRUE(isContained(range, result.ranges)) << "(" << range.begin << ", " << range.end << "), budget " << max_steps;
                for (size_t mark = range.begin; mark != range.end; ++mark)
                    EXPECT_TRUE(matching[mark]) << "mark " << mark << ", budget " << max_steps;
            }
        }
    }
}

TEST(GenericExclusionSearch, BudgetSharedAcrossInitialRanges)
{
    /// One large initial range with a small matching run inside, followed by non-matching
    /// single-mark initial ranges. The budget must be shared across all of them: the large range
    /// gets subdivided while the single-mark ranges still receive their checks and are excluded.
    std::vector<bool> matching(1024);
    for (size_t mark = 100; mark != 110; ++mark)
        matching[mark] = true;

    std::vector<MarkRange> initial{{0, 1000}};
    for (size_t i = 0; i != 10; ++i)
        initial.emplace_back(1002 + 2 * i, 1003 + 2 * i);

    GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = 60, .min_marks_for_seek = 0};
    auto result = genericExclusionSearch(makeRanges(initial), oracleFromFlags(matching), settings, true);

    EXPECT_LE(result.num_steps, 60u);

    /// The non-matching single-mark ranges were checked and dropped.
    for (size_t i = 0; i != 10; ++i)
        EXPECT_FALSE(covers(result.ranges, 1002 + 2 * i));

    /// The large range was refined instead of being accepted as a whole.
    EXPECT_LT(result.ranges.getNumberOfMarks(), 200u);
    for (size_t mark = 100; mark != 110; ++mark)
        EXPECT_TRUE(covers(result.ranges, mark));
}

TEST(GenericExclusionSearch, LargestRangeSplitFirst)
{
    /// A large ambiguous range and a small fully matching one. The budget is spent on splitting the
    /// large range, because that is where a successful exclusion saves the most marks.
    std::vector<bool> matching(104);
    matching[0] = true;
    for (size_t mark = 100; mark != 104; ++mark)
        matching[mark] = true;

    auto oracle = oracleFromFlags(matching);
    auto initial = makeRanges({{0, 64}, {100, 104}});

    /// The budget does not even cover one split, so the large range is accepted as a whole.
    GenericExclusionSearchSettings tiny{.coarse_index_granularity = 8, .max_steps = 2, .min_marks_for_seek = 0};
    auto tiny_result = genericExclusionSearch(initial, oracle, tiny, true);
    EXPECT_EQ(tiny_result.ranges, makeRanges({{0, 64}, {100, 104}}));
    EXPECT_TRUE(tiny_result.reached_step_limit);

    /// The budget covers one split of the large range: its non-matching subranges are excluded and
    /// only the first subrange (still ambiguous) is accepted without further refinement.
    GenericExclusionSearchSettings one_split{.coarse_index_granularity = 8, .max_steps = 10, .min_marks_for_seek = 0};
    auto one_split_result = genericExclusionSearch(initial, oracle, one_split, true);
    EXPECT_EQ(one_split_result.ranges, makeRanges({{0, 8}, {100, 104}}));
    EXPECT_EQ(one_split_result.exact_ranges, makeRanges({{100, 104}}));
    EXPECT_EQ(one_split_result.num_steps, 10u);
    EXPECT_TRUE(one_split_result.reached_step_limit);

    /// A budget that covers the whole search reproduces the unlimited result.
    GenericExclusionSearchSettings ample{.coarse_index_granularity = 8, .max_steps = 40, .min_marks_for_seek = 0};
    auto ample_result = genericExclusionSearch(initial, oracle, ample, true);
    EXPECT_EQ(ample_result.ranges, makeRanges({{0, 1}, {100, 104}}));
    EXPECT_EQ(ample_result.exact_ranges, makeRanges({{0, 1}, {100, 104}}));
    EXPECT_FALSE(ample_result.reached_step_limit);
}

TEST(GenericExclusionSearch, HugeLimitEqualsUnlimited)
{
    std::mt19937 rng(2026);

    for (double density : {0.1, 0.6})
    {
        auto matching = randomFlags(500, density, rng);
        auto oracle = oracleFromFlags(matching);

        GenericExclusionSearchSettings unlimited{.coarse_index_granularity = 8, .max_steps = 0, .min_marks_for_seek = 2};
        auto unlimited_result = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracle, unlimited, true);

        /// The largest possible budget also checks that the budget gate arithmetic cannot overflow.
        for (size_t max_steps : {size_t(1000000), std::numeric_limits<size_t>::max()})
        {
            GenericExclusionSearchSettings huge{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 2};
            auto huge_result = genericExclusionSearch(makeRanges({{0, matching.size()}}), oracle, huge, true);

            EXPECT_EQ(huge_result.ranges, unlimited_result.ranges);
            EXPECT_EQ(huge_result.exact_ranges, unlimited_result.exact_ranges);
            EXPECT_EQ(huge_result.num_steps, unlimited_result.num_steps);
            EXPECT_FALSE(huge_result.reached_step_limit);
        }
    }
}

TEST(GenericExclusionSearch, EdgeCases)
{
    std::vector<bool> matching(64, true);
    auto all_match = oracleFromFlags(matching);

    std::vector<bool> non_matching(64, false);
    auto none_match = oracleFromFlags(non_matching);

    for (size_t max_steps : {0, 10})
    {
        GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 0};

        auto empty = genericExclusionSearch(MarkRanges{}, all_match, settings, true);
        EXPECT_TRUE(empty.ranges.empty());
        EXPECT_TRUE(empty.exact_ranges.empty());
        EXPECT_EQ(empty.num_steps, 0u);

        /// A fully matching range is accepted in one step.
        auto everything = genericExclusionSearch(makeRanges({{0, 64}}), all_match, settings, true);
        EXPECT_EQ(everything.ranges, makeRanges({{0, 64}}));
        EXPECT_EQ(everything.exact_ranges, makeRanges({{0, 64}}));
        EXPECT_EQ(everything.num_steps, 1u);

        auto nothing = genericExclusionSearch(makeRanges({{0, 64}}), none_match, settings, true);
        EXPECT_TRUE(nothing.ranges.empty());
        EXPECT_TRUE(nothing.exact_ranges.empty());

        /// Exact ranges are collected only on request.
        auto no_exact = genericExclusionSearch(makeRanges({{0, 64}}), all_match, settings, false);
        EXPECT_EQ(no_exact.ranges, makeRanges({{0, 64}}));
        EXPECT_TRUE(no_exact.exact_ranges.empty());

        /// Single-mark initial ranges are checked individually.
        std::vector<bool> one_of_two(10);
        one_of_two[5] = true;
        auto singles = genericExclusionSearch(makeRanges({{5, 6}, {7, 8}}), oracleFromFlags(one_of_two), settings, true);
        EXPECT_EQ(singles.ranges, makeRanges({{5, 6}}));
        EXPECT_EQ(singles.exact_ranges, makeRanges({{5, 6}}));
    }
}

TEST(GenericExclusionSearch, NeverExactOracle)
{
    /// An oracle that never proves a range fully matching, modelling a key condition that cannot
    /// establish exactness. The result then comes entirely from single-mark accepts: the ranges to
    /// read must still be found, and the exact ranges must stay empty.
    std::mt19937 rng(29);
    auto matching = randomFlags(150, 0.4, rng);
    auto precise = oracleFromFlags(matching);
    MarkRangeCheck never_exact = [&](const MarkRange & range)
    {
        return BoolMask(precise(range).can_be_true, true);
    };

    for (size_t min_marks_for_seek : {0, 2})
    {
        GenericExclusionSearchSettings unlimited{.coarse_index_granularity = 8, .max_steps = 0, .min_marks_for_seek = min_marks_for_seek};
        auto result = genericExclusionSearch(makeRanges({{0, matching.size()}}), never_exact, unlimited, true);

        EXPECT_EQ(result.ranges, runsOfMatchingMarks(matching, min_marks_for_seek));
        EXPECT_TRUE(result.exact_ranges.empty());

        /// Under a budget the same invariants hold: a superset of the matching marks, no exact ranges.
        GenericExclusionSearchSettings limited{.coarse_index_granularity = 8, .max_steps = 20, .min_marks_for_seek = min_marks_for_seek};
        auto truncated = genericExclusionSearch(makeRanges({{0, matching.size()}}), never_exact, limited, true);

        EXPECT_NO_THROW(assertSortedAndNonIntersecting(truncated.ranges));
        EXPECT_TRUE(truncated.exact_ranges.empty());
        for (size_t mark = 0; mark != matching.size(); ++mark)
            if (matching[mark])
                EXPECT_TRUE(covers(truncated.ranges, mark)) << "mark " << mark;
    }
}

TEST(GenericExclusionSearch, ConservativeOracle)
{
    /// An over-approximating oracle that never allows exclusion, modelling an imprecise key
    /// condition. The output invariants must hold regardless of the oracle's precision.
    std::mt19937 rng(11);
    auto matching = randomFlags(128, 0.4, rng);
    auto precise = oracleFromFlags(matching);
    MarkRangeCheck conservative = [&](const MarkRange & range)
    {
        return BoolMask(true, precise(range).can_be_false);
    };

    for (size_t max_steps : {0, 25})
    {
        GenericExclusionSearchSettings settings{.coarse_index_granularity = 8, .max_steps = max_steps, .min_marks_for_seek = 0};
        auto result = genericExclusionSearch(makeRanges({{0, matching.size()}}), conservative, settings, true);

        /// Nothing can be excluded, so the whole extent is accepted in one way or another.
        EXPECT_EQ(result.ranges, makeRanges({{0, matching.size()}}));

        EXPECT_NO_THROW(assertSortedAndNonIntersecting(result.exact_ranges));
        for (const auto & range : result.exact_ranges)
            for (size_t mark = range.begin; mark != range.end; ++mark)
                EXPECT_TRUE(matching[mark]);
    }
}
