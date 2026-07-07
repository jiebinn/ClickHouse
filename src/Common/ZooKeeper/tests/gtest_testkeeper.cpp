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

TEST(TestKeeperTest, DuplicateCreateIfNotExistsReportsPathCreated)
{
    TestKeeper keeper = makeKeeper();

    create(keeper, "/node", "data", /* is_ephemeral */ false);

    // A plain CreateIfNotExists (not_exists=true, no stats/ttl) on an existing node
    // must succeed with ZOK and still report the requested path in path_created,
    // mirroring KeeperStorage::process.
    auto req = std::make_shared<CreateRequest>();
    req->path = "/node";
    req->data = "data2";
    req->not_exists = true;

    std::promise<MultiResponse> sink;
    std::future<MultiResponse> future = sink.get_future();
    keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

    MultiResponse multi = future.get();
    ASSERT_EQ(multi.error, Error::ZOK);
    ASSERT_EQ(multi.responses.size(), 1u);

    const auto * create = dynamic_cast<const CreateResponse *>(multi.responses[0].get());
    ASSERT_NE(create, nullptr);
    EXPECT_EQ(create->error, Error::ZOK);
    EXPECT_EQ(create->path_created, "/node");
}

TEST(TestKeeperTest, InvalidTTLCreateReturnsBadArguments)
{
    TestKeeper keeper = makeKeeper();

    // CreateTTL requests that real Keeper rejects with ZBADARGUMENTS must be rejected
    // by TestKeeper too, and must not create the node.
    auto try_ttl_create = [&](const String & path, int64_t ttl, bool is_ephemeral)
    {
        auto req = std::make_shared<CreateRequest>();
        req->path = path;
        req->data = "data";
        req->is_ephemeral = is_ephemeral;
        req->include_ttl = true;
        req->ttl = ttl;

        std::promise<MultiResponse> sink;
        std::future<MultiResponse> future = sink.get_future();
        keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

        return future.get().responses[0]->error;
    };

    // TTL is incompatible with ephemeral nodes.
    EXPECT_EQ(try_ttl_create("/ttl_ephemeral", 10000, /* is_ephemeral */ true), Error::ZBADARGUMENTS);
    // Non-positive TTL is rejected.
    EXPECT_EQ(try_ttl_create("/ttl_zero", 0, /* is_ephemeral */ false), Error::ZBADARGUMENTS);
    EXPECT_EQ(try_ttl_create("/ttl_negative", -1, /* is_ephemeral */ false), Error::ZBADARGUMENTS);
    // TTL beyond the maximum bound is rejected.
    EXPECT_EQ(try_ttl_create("/ttl_too_large", MAX_TESTKEEPER_TTL_MS + 1, /* is_ephemeral */ false), Error::ZBADARGUMENTS);

    // None of the rejected requests must have created a node.
    ASSERT_FALSE(exists(keeper, "/ttl_ephemeral"));
    ASSERT_FALSE(exists(keeper, "/ttl_zero"));
    ASSERT_FALSE(exists(keeper, "/ttl_negative"));
    ASSERT_FALSE(exists(keeper, "/ttl_too_large"));

    // A valid TTL create at the maximum bound must succeed.
    EXPECT_EQ(try_ttl_create("/ttl_ok", MAX_TESTKEEPER_TTL_MS, /* is_ephemeral */ false), Error::ZOK);
    ASSERT_TRUE(exists(keeper, "/ttl_ok"));
}

TEST(TestKeeperTest, SequentialCreateIfNotExistsReturnsBadArguments)
{
    TestKeeper keeper = makeKeeper();

    create(keeper, "/parent", "", /* is_ephemeral */ false);

    // ZooKeeperCreateRequest::getOpNum maps not_exists to CreateIfNotExists, and real Keeper
    // (KeeperStorage::preprocess) rejects a sequential CreateIfNotExists with ZBADARGUMENTS.
    // TestKeeper must fail closed the same way, whether or not the target path already exists,
    // and must not create a node — matching the behaviour of a request built via
    // zkutil::makeCreateRequest(path, data, CreateMode::PersistentSequential, /*ignore_if_exists=*/ true).
    auto try_seq_create_if_not_exists = [&](const String & path)
    {
        auto req = std::make_shared<CreateRequest>();
        req->path = path;
        req->data = "data";
        req->is_sequential = true;
        req->not_exists = true;

        std::promise<MultiResponse> sink;
        std::future<MultiResponse> future = sink.get_future();
        keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

        return future.get().responses[0]->error;
    };

    // Target path does not exist yet.
    EXPECT_EQ(try_seq_create_if_not_exists("/parent/seq"), Error::ZBADARGUMENTS);
    // The rejected request must not have created the bare (non-sequential) node ...
    ASSERT_FALSE(exists(keeper, "/parent/seq"));
    // ... nor a sequential node with the seq-num suffix.
    {
        ListResponse response = list(keeper, "/parent", ListRequestType::ALL, /* with_stat */ false, /* with_data */ false);
        ASSERT_EQ(response.error, Error::ZOK);
        EXPECT_TRUE(response.names.empty());
    }

    // Even when the bare path already exists, the illegal combination is still rejected
    // (the check precedes the node-existence lookup in real Keeper).
    create(keeper, "/parent/existing", "data", /* is_ephemeral */ false);
    EXPECT_EQ(try_seq_create_if_not_exists("/parent/existing"), Error::ZBADARGUMENTS);
}

TEST(TestKeeperTest, SequentialCreateSucceedsWhenUnsuffixedPrefixExists)
{
    TestKeeper keeper = makeKeeper();

    create(keeper, "/parent", "", /* is_ephemeral */ false);
    // A literal node at the bare prefix must not block a sequential create on that
    // same prefix: KeeperStorage::preprocess appends the sequence suffix to path_created
    // before checking for a duplicate, so "/parent/log-" existing does not collide with
    // the freshly-suffixed "/parent/log-0000000000".
    create(keeper, "/parent/log-", "existing", /* is_ephemeral */ false);

    auto req = std::make_shared<CreateRequest>();
    req->path = "/parent/log-";
    req->data = "data";
    req->is_sequential = true;

    std::promise<MultiResponse> sink;
    std::future<MultiResponse> future = sink.get_future();
    keeper.multi(Requests{req}, [&](const MultiResponse & r) { sink.set_value(r); });

    MultiResponse multi = future.get();
    ASSERT_EQ(multi.error, Error::ZOK);
    ASSERT_EQ(multi.responses.size(), 1u);

    const auto * create_response = dynamic_cast<const CreateResponse *>(multi.responses[0].get());
    ASSERT_NE(create_response, nullptr);
    EXPECT_EQ(create_response->error, Error::ZOK);
    EXPECT_EQ(create_response->path_created, "/parent/log-0000000000");
    ASSERT_TRUE(exists(keeper, "/parent/log-0000000000"));
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
