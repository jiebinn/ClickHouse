#include <gtest/gtest.h>

#include <Common/QueryFuzzer.h>
#include <Parsers/ASTSetQuery.h>
#include <Parsers/ParserQuery.h>
#include <Parsers/parseQuery.h>

using namespace DB;

namespace
{

/// Return the current Field value of setting `name` (or fail the test if it disappeared).
Field settingValue(const ASTPtr & ast, const String & name)
{
    const auto * set = ast->as<ASTSetQuery>();
    EXPECT_NE(set, nullptr);
    for (const auto & c : set->changes)
        if (c.name == name)
            return c.value;
    ADD_FAILURE() << "setting " << name << " disappeared";
    return {};
}

ASTPtr parseSet(const String & sql)
{
    ParserQuery parser(sql.data() + sql.size());
    return parseQuery(parser, sql.data(), sql.data() + sql.size(), "", 0, 0, 0);
}

}

/// Regression for the chronic hung-check family: the fuzzer must not mutate the value of the
/// test-only fault-injection sleep settings. Blowing a test's small `sleep_in_send_data_ms` up to
/// a multi-minute value manufactures an uninterruptible sleep in TCPHandler and trips the CI
/// hung-check without exercising any real code path.
TEST(QueryFuzzer, DoesNotFuzzUninterruptibleSleepSettings)
{
    static const std::vector<String> guarded
        = {"sleep_in_send_data_ms", "sleep_in_send_tables_status_ms", "sleep_after_receiving_query_ms"};

    for (const auto & name : guarded)
    {
        const String sql = "SET " + name + " = 5, max_threads = 4";

        bool control_mutated_at_least_once = false;
        for (UInt64 seed = 0; seed < 500; ++seed)
        {
            ASTPtr ast = parseSet(sql);
            QueryFuzzer fuzzer{pcg64(seed)};
            fuzzer.fuzzMain(ast);

            /// The guarded sleep setting must always keep its original value (UInt64 5).
            ASSERT_EQ(settingValue(ast, name), Field(UInt64(5))) << "seed=" << seed << " setting=" << name;

            /// The control setting (max_threads) is allowed to be fuzzed — confirm fuzzing is
            /// actually active so the test above is meaningful.
            if (settingValue(ast, "max_threads") != Field(UInt64(4)))
                control_mutated_at_least_once = true;
        }

        EXPECT_TRUE(control_mutated_at_least_once)
            << "fuzzer never mutated the control setting for " << name << " — the guard test is vacuous";
    }
}
