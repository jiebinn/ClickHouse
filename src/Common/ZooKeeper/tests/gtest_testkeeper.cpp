#include <Common/ZooKeeper/IKeeper.h>
#include <Common/ZooKeeper/TestKeeper.h>
#include <Common/ZooKeeper/Types.h>
#include <IO/ReadBufferFromString.h>
#include <IO/WriteBufferFromString.h>

#include <gtest/gtest.h>

#include <future>

using namespace Coordination;
using namespace DB;

namespace
{

Coordination::TestKeeper makeKeeper(int32_t operation_timeout_ms = DEFAULT_OPERATION_TIMEOUT_MS, std::string chroot = "")
{
    zkutil::ZooKeeperArgs args;
    args.operation_timeout_ms = operation_timeout_ms;
    args.chroot = chroot;

    return Coordination::TestKeeper(args);
}

void create(TestKeeper & keeper, const String & path, const String & data, bool is_ephemeral)
{
    std::promise<CreateResponse> sink;
    std::future<CreateResponse> future = sink.get_future();
    keeper.create(path, data, is_ephemeral, /* is_sequential */ false, {},
        [&](const auto & response) { sink.set_value(std::move(response)); });

    CreateResponse response = future.get();
    ASSERT_EQ(response.error, Error::ZOK);
}

bool exists(TestKeeper & keeper, const String & path)
{
    std::promise<ExistsResponse> sink;
    std::future<ExistsResponse> future = sink.get_future();
    keeper.exists(path, [&](const auto & response) { sink.set_value(std::move(response)); }, WatchCallbackPtrOrEventPtr());

    return future.get().error == Coordination::Error::ZOK;
}

ListResponse list(TestKeeper & keeper, const String & path, ListRequestType list_request_type, bool with_stat, bool with_data)
{
    std::promise<ListResponse> sink;
    std::future<ListResponse> future = sink.get_future();
    keeper.list(path, list_request_type,
        [&](const auto & response) { sink.set_value(std::move(response)); },
        WatchCallbackPtrOrEventPtr(), with_stat, with_data);

    return future.get();
}

}

TEST(TestKeeperTest, JustWorks)
{
    TestKeeper keeper = makeKeeper();

    ASSERT_TRUE(exists(keeper, "/"));
    ASSERT_FALSE(exists(keeper, "/A"));

    create(keeper, "/A", "hello", /*is_ephemeral=*/false);
    ASSERT_TRUE(exists(keeper, "/A"));
}

TEST(TestKeeperTest, FilteredListWithStatsAndDataIsAligned)
{
    TestKeeper keeper = makeKeeper();

    create(keeper, "/parent", "", /* is_ephemeral */ false);
    create(keeper, "/parent/ephemeral", "ephemeral_data", /* is_ephemeral */ true);
    create(keeper, "/parent/persistent", "persistent_data", /* is_ephemeral */ false);

    {
        ListResponse response = list(keeper, "/parent", ListRequestType::PERSISTENT_ONLY, /* with_stat */ true, /* with_data */ true);

        ASSERT_EQ(response.error, Error::ZOK);
        ASSERT_EQ(response.names, std::vector<std::string>({"persistent"}));

        ASSERT_EQ(response.data.size(), 1u);
        EXPECT_EQ(response.data[0], "persistent_data");

        ASSERT_EQ(response.stats.size(), 1u);
        EXPECT_EQ(response.stats[0].ephemeralOwner, 0);
    }

    {
        ListResponse response = list(keeper, "/parent", ListRequestType::EPHEMERAL_ONLY, /* with_stat */ true, /* with_data */ true);

        ASSERT_EQ(response.error, Error::ZOK);
        EXPECT_EQ(response.names, std::vector<std::string>({"ephemeral"}));

        ASSERT_EQ(response.data.size(), 1u);
        EXPECT_EQ(response.data[0], "ephemeral_data");

        ASSERT_EQ(response.stats.size(), 1u);
        EXPECT_NE(response.stats[0].ephemeralOwner, 0);
    }

    {
        ListResponse response = list(keeper, "/parent", ListRequestType::ALL, /* with_stat */ true, /* with_data */ true);

        ASSERT_EQ(response.error, Error::ZOK);
        ASSERT_EQ(response.names.size(), 2u);
        ASSERT_EQ(response.data.size(), 2u);
        ASSERT_EQ(response.stats.size(), 2u);
    }
}

TEST(TestKeeperTest, Create2ResponseHasStatInMulti)
{
    TestKeeper keeper = makeKeeper();

    ASSERT_TRUE(keeper.isFeatureEnabled(KeeperFeatureFlag::CREATE_WITH_STATS));

    // Plain create (include_stats = false) must yield CreateResponse, not Create2Response,
    // matching the real Keeper wire behaviour.
    {
        auto req = std::make_shared<CreateRequest>();
        req->path = "/plain_node";
        req->data = "data";
        req->include_stats = false;

        std::promise<MultiResponse> sink;
        std::future<MultiResponse> future = sink.get_future();
        keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

        MultiResponse multi = future.get();
        ASSERT_EQ(multi.error, Error::ZOK);
        ASSERT_EQ(multi.responses.size(), 1u);
        EXPECT_EQ(dynamic_cast<const Create2Response *>(multi.responses[0].get()), nullptr);
    }

    // Create2 (include_stats = true) must yield Create2Response with stat populated.
    {
        auto req = std::make_shared<CreateRequest>();
        req->path = "/node_with_stat";
        req->data = "hello";
        req->include_stats = true;

        std::promise<MultiResponse> sink;
        std::future<MultiResponse> future = sink.get_future();
        keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

        MultiResponse multi = future.get();
        ASSERT_EQ(multi.error, Error::ZOK);
        ASSERT_EQ(multi.responses.size(), 1u);

        const auto * create2 = dynamic_cast<const Create2Response *>(multi.responses[0].get());
        ASSERT_NE(create2, nullptr);
        EXPECT_EQ(create2->path_created, "/node_with_stat");
        EXPECT_EQ(create2->stat.dataLength, static_cast<int32_t>(std::string("hello").size()));
        EXPECT_NE(create2->stat.czxid, 0);
    }
}

TEST(TestKeeperTest, Create2DuplicateReturnsZNodeExists)
{
    TestKeeper keeper = makeKeeper();

    create(keeper, "/node", "data", /* is_ephemeral */ false);

    // include_stats=true (Create2) on an existing node must return ZNODEEXISTS,
    // even when not_exists is set — Create2 takes precedence over CreateIfNotExists.
    for (bool not_exists : {false, true})
    {
        auto req = std::make_shared<CreateRequest>();
        req->path = "/node";
        req->data = "data2";
        req->not_exists = not_exists;
        req->include_stats = true;

        std::promise<MultiResponse> sink;
        std::future<MultiResponse> future = sink.get_future();
        keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

        MultiResponse multi = future.get();
        EXPECT_EQ(multi.responses[0]->error, Error::ZNODEEXISTS)
            << "not_exists=" << not_exists;
    }

    // include_ttl=true (CreateTTL) on an existing node must also return ZNODEEXISTS.
    {
        auto req = std::make_shared<CreateRequest>();
        req->path = "/node";
        req->data = "data3";
        req->not_exists = true;
        req->include_ttl = true;
        req->ttl = 10000;

        std::promise<MultiResponse> sink;
        std::future<MultiResponse> future = sink.get_future();
        keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

        MultiResponse multi = future.get();
        EXPECT_EQ(multi.responses[0]->error, Error::ZNODEEXISTS);
    }
}

TEST(TestKeeperTest, FilteredListWithoutStatsAndData)
{
    TestKeeper keeper = makeKeeper();

    create(keeper, "/parent", "", /* is_ephemeral */ false);
    create(keeper, "/parent/ephemeral", "ephemeral_data", /* is_ephemeral */ true);
    create(keeper, "/parent/persistent", "persistent_data", /* is_ephemeral */ false);

    {
        ListResponse response = list(keeper, "/parent", ListRequestType::PERSISTENT_ONLY, /* with_stat */ false, /* with_data */ false);

        ASSERT_EQ(response.error, Error::ZOK);
        ASSERT_EQ(response.names, std::vector<std::string>({"persistent"}));

        EXPECT_TRUE(response.data.empty());
        EXPECT_TRUE(response.stats.empty());
    }
}
