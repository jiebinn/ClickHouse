#include <gtest/gtest.h>

#include <Common/ZooKeeper/ZooKeeper.h>
#include <Common/ZooKeeper/ZooKeeperCommon.h>

TEST(ZooKeeperTest, TestMatchPath)
{
    using namespace Coordination;

    ASSERT_EQ(matchPath("/path/file", "/path"), PathMatchResult::IS_CHILD);
    ASSERT_EQ(matchPath("/path/file", "/path/"), PathMatchResult::IS_CHILD);
    ASSERT_EQ(matchPath("/path/file", "/"), PathMatchResult::IS_CHILD);
    ASSERT_EQ(matchPath("/", "/"), PathMatchResult::EXACT);
    ASSERT_EQ(matchPath("/path", "/path/"), PathMatchResult::EXACT);
    ASSERT_EQ(matchPath("/path/", "/path"), PathMatchResult::EXACT);
}

TEST(ZooKeeperTest, ExtractZooKeeperPathAndCollapseTrailingSlashes)
{
    using zkutil::extractZooKeeperPathAndCollapseTrailingSlashes;

    /// Any number of trailing slashes collapses to the same canonical path, so that different spellings
    /// of the same keeper path compare equal (used by the SYSTEM DROP REPLICA self-protection guards).
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("/a/b", false), "/a/b");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("/a/b/", false), "/a/b");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("/a/b//", false), "/a/b");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("/a/b///", false), "/a/b");

    /// Auxiliary keeper prefix is stripped and the remaining path is canonicalized the same way.
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("aux:/a/b", false), "/a/b");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("aux:/a/b///", false), "/a/b");

    /// Root-only inputs collapse to "" or "/" (both are rejected by the "empty or root-only" check in the
    /// parser and never equal a real database/table path in the guards): normalizeZooKeeperPath first strips
    /// a single trailing slash, so "/" and "aux:/" become "", while "//"/"///" become "/".
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("/", false), "");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("aux:/", false), "");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("//", false), "/");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("///", false), "/");
    ASSERT_EQ(extractZooKeeperPathAndCollapseTrailingSlashes("aux://", false), "/");
}
