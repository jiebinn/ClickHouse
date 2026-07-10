#include <Backups/BackupMetadataHandler.h>

#include <Poco/SAX/SAXParser.h>
#include <Poco/DOM/DOMParser.h>
#include <Poco/DOM/Document.h>
#include <Poco/DOM/Node.h>
#include <Poco/AutoPtr.h>

#include <Common/Jemalloc.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>


using namespace DB;

namespace
{
    struct ParseResult
    {
        BackupMetadataHandler::Fields header;
        std::vector<BackupMetadataHandler::Fields> files;
        bool header_seen = false;
        bool file_seen_before_header = false;
        std::exception_ptr saved_exception;
    };

    /// Drives the handler over `xml` exactly like `BackupImpl::readBackupMetadata` does (default `SAXParser`,
    /// `parseMemoryNP`), recording the header and per-file field maps.
    ParseResult parse(const std::string & xml)
    {
        ParseResult result;
        BackupMetadataHandler handler;
        handler.on_header = [&](const BackupMetadataHandler::Fields & h)
        {
            result.header = h;
            result.header_seen = true;
        };
        handler.on_file = [&](const BackupMetadataHandler::Fields & f)
        {
            if (!result.header_seen)
                result.file_seen_before_header = true;
            result.files.push_back(f);
        };

        Poco::XML::SAXParser parser;
        parser.setContentHandler(&handler);
        parser.parseMemoryNP(xml.data(), xml.size());
        result.saved_exception = handler.saved_exception;
        return result;
    }

    /// A manifest with two files, mirroring the whitespace-free output of `writeBackupMetadata`.
    const std::string two_files_xml =
        "<config>"
        "<version>2</version>"
        "<timestamp>2020-01-01 00:00:00</timestamp>"
        "<uuid>00000000-0000-0000-0000-000000000001</uuid>"
        "<base_backup>Disk('backups', 'base')</base_backup>"
        "<base_backup_uuid>00000000-0000-0000-0000-000000000002</base_backup_uuid>"
        "<contents>"
        "<file>"
        "<name>data/db/tbl/full.bin</name>"
        "<size>100</size>"
        "<checksum>0123456789abcdef0123456789abcdef</checksum>"
        "</file>"
        "<file>"
        "<name>data/db/tbl/incremental.bin</name>"
        "<size>200</size>"
        "<checksum>fedcba9876543210fedcba9876543210</checksum>"
        "<use_base>true</use_base>"
        "<base_size>150</base_size>"
        "<base_checksum>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</base_checksum>"
        "<data_file>data/db/tbl/other.bin</data_file>"
        "<encrypted_by_disk>true</encrypted_by_disk>"
        "</file>"
        "</contents>"
        "</config>";
}


TEST(BackupMetadataHandler, ParsesHeaderFields)
{
    auto result = parse(two_files_xml);

    EXPECT_TRUE(result.header_seen);
    EXPECT_EQ(result.header.at("version"), "2");
    EXPECT_EQ(result.header.at("timestamp"), "2020-01-01 00:00:00");
    EXPECT_EQ(result.header.at("uuid"), "00000000-0000-0000-0000-000000000001");
    EXPECT_EQ(result.header.at("base_backup"), "Disk('backups', 'base')");
    EXPECT_EQ(result.header.at("base_backup_uuid"), "00000000-0000-0000-0000-000000000002");
    /// `<contents>` is not a header leaf and must not leak into the header map.
    EXPECT_EQ(result.header.count("contents"), 0u);
}

TEST(BackupMetadataHandler, ParsesFilesInOrderWithAllLeafFields)
{
    auto result = parse(two_files_xml);

    ASSERT_EQ(result.files.size(), 2u);

    const auto & f0 = result.files[0];
    EXPECT_EQ(f0.at("name"), "data/db/tbl/full.bin");
    EXPECT_EQ(f0.at("size"), "100");
    EXPECT_EQ(f0.at("checksum"), "0123456789abcdef0123456789abcdef");
    /// Optional fields that were not present must be absent (not empty strings).
    EXPECT_EQ(f0.count("use_base"), 0u);
    EXPECT_EQ(f0.count("base_size"), 0u);
    EXPECT_EQ(f0.count("data_file"), 0u);

    const auto & f1 = result.files[1];
    EXPECT_EQ(f1.at("name"), "data/db/tbl/incremental.bin");
    EXPECT_EQ(f1.at("size"), "200");
    EXPECT_EQ(f1.at("use_base"), "true");
    EXPECT_EQ(f1.at("base_size"), "150");
    EXPECT_EQ(f1.at("base_checksum"), "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    EXPECT_EQ(f1.at("data_file"), "data/db/tbl/other.bin");
    EXPECT_EQ(f1.at("encrypted_by_disk"), "true");
}

TEST(BackupMetadataHandler, HeaderIsAppliedBeforeAnyFile)
{
    auto result = parse(two_files_xml);
    EXPECT_FALSE(result.file_seen_before_header);
}

TEST(BackupMetadataHandler, EmptyContentsStillAppliesHeader)
{
    auto result = parse("<config><version>1</version><contents></contents></config>");

    EXPECT_TRUE(result.header_seen);
    EXPECT_EQ(result.header.at("version"), "1");
    EXPECT_TRUE(result.files.empty());
}

TEST(BackupMetadataHandler, FieldMapIsResetBetweenFiles)
{
    /// The second file omits `checksum`; it must not inherit the first file's value.
    auto result = parse(
        "<config><version>2</version><contents>"
        "<file><name>a</name><size>1</size><checksum>c1</checksum></file>"
        "<file><name>b</name><size>0</size></file>"
        "</contents></config>");

    ASSERT_EQ(result.files.size(), 2u);
    EXPECT_EQ(result.files[1].at("name"), "b");
    EXPECT_EQ(result.files[1].count("checksum"), 0u);
}

TEST(BackupMetadataHandler, CapturesFileCallbackExceptionAndShortCircuits)
{
    BackupMetadataHandler handler;
    int file_calls = 0;
    handler.on_file = [&](const BackupMetadataHandler::Fields &)
    {
        ++file_calls;
        throw std::runtime_error("boom");
    };

    Poco::XML::SAXParser parser;
    parser.setContentHandler(&handler);
    /// The exception must NOT propagate through the expat-based parser.
    EXPECT_NO_THROW(parser.parseMemoryNP(two_files_xml.data(), two_files_xml.size()));

    /// The first file threw; the second must be short-circuited.
    EXPECT_EQ(file_calls, 1);
    ASSERT_TRUE(handler.saved_exception);
    EXPECT_THROW(std::rethrow_exception(handler.saved_exception), std::runtime_error);
}

TEST(BackupMetadataHandler, HeaderCallbackExceptionShortCircuitsFiles)
{
    BackupMetadataHandler handler;
    int file_calls = 0;
    handler.on_header = [&](const BackupMetadataHandler::Fields &) { throw std::runtime_error("bad header"); };
    handler.on_file = [&](const BackupMetadataHandler::Fields &) { ++file_calls; };

    Poco::XML::SAXParser parser;
    parser.setContentHandler(&handler);
    EXPECT_NO_THROW(parser.parseMemoryNP(two_files_xml.data(), two_files_xml.size()));

    EXPECT_EQ(file_calls, 0);
    ASSERT_TRUE(handler.saved_exception);
    EXPECT_THROW(std::rethrow_exception(handler.saved_exception), std::runtime_error);
}

TEST(BackupMetadataHandler, MalformedXmlThrowsFromParser)
{
    BackupMetadataHandler handler;
    Poco::XML::SAXParser parser;
    parser.setContentHandler(&handler);

    /// A parse error (mismatched tags) is reported by the parser itself, not captured in saved_exception.
    const std::string bad = "<config><version>2</version></contents>";
    EXPECT_ANY_THROW(parser.parseMemoryNP(bad.data(), bad.size()));
}


/// Memory comparison: streaming SAX vs the old DOM parse.
///
/// `readBackupMetadata` used to build a full Poco::XML DOM tree for the whole `.backup` manifest before
/// walking it; for a large (especially incremental) backup that tree is the dominant, front-loaded cost
/// of opening the base backup. The SAX handler folds each `<file>` as it closes, so nothing beyond one
/// entry is retained. This test parses the same synthetic manifest both ways and compares the peak *live*
/// allocated bytes (jemalloc `stats.allocated`, which - unlike RSS - is not distorted by the allocator
/// holding on to freed pages).
///
/// Both walks here only count files; the real code builds the same per-file maps in the old and new
/// paths, so the measured difference is exactly the transient DOM tree the PR removes.
///
/// The number of files defaults to a value light enough for the unit-test run; set
/// BACKUP_METADATA_BENCH_FILES to a large value (e.g. 5000000) to reproduce the multi-GB figures for a
/// worst-case incremental backup. Sanitizer builds disable jemalloc, so the test skips itself there.
#if USE_JEMALLOC
namespace
{
    /// A manifest of `num_files` deduplicated ("use_base") entries, in the whitespace-free layout written
    /// by `writeBackupMetadata`. Deduplicated entries carry the most leaf elements - the most DOM nodes
    /// per file, i.e. the worst case for the DOM parser.
    std::string makeManifest(size_t num_files)
    {
        std::string xml;
        xml.reserve(num_files * 240 + 256);
        xml += "<config><version>2</version>"
               "<timestamp>2020-01-01 00:00:00</timestamp>"
               "<uuid>00000000-0000-0000-0000-000000000001</uuid>"
               "<contents>";
        for (size_t i = 0; i < num_files; ++i)
        {
            xml += "<file><name>data/default/table/all_1_1_0/column_";
            xml += std::to_string(i);
            xml += ".bin</name><size>1048576</size>"
                   "<checksum>0123456789abcdef0123456789abcdef</checksum>"
                   "<use_base>true</use_base><base_size>1048576</base_size>"
                   "<base_checksum>0123456789abcdef0123456789abcdef</base_checksum></file>";
        }
        xml += "</contents></config>";
        return xml;
    }

    /// Live bytes currently allocated by the process. `stats.allocated` is cached behind `epoch`, so the
    /// epoch is advanced first to force a refresh. Returns 0 if the mallctl is unavailable.
    size_t liveAllocatedBytes()
    {
        Jemalloc::setValue<UInt64>("epoch", 1);
        size_t allocated = 0;
        Jemalloc::tryGetValue("stats.allocated", allocated);
        return allocated;
    }
}
#endif

TEST(BackupMetadataHandler, PeakMemoryStreamingVsDom)
{
#if !USE_JEMALLOC
    GTEST_SKIP() << "Peak-memory comparison needs jemalloc stats (disabled in sanitizer builds)";
#else
    if (liveAllocatedBytes() == 0)
        GTEST_SKIP() << "jemalloc stats.allocated is unavailable";

    size_t num_files = 50000;
    if (const char * env = std::getenv("BACKUP_METADATA_BENCH_FILES"))
        num_files = std::strtoull(env, nullptr, 10);

    const std::string xml = makeManifest(num_files);

    /// Old behavior: build the whole DOM tree and keep it alive while walking <contents>.
    size_t dom_files = 0;
    Int64 dom_bytes = 0;
    {
        const size_t before = liveAllocatedBytes();
        Poco::XML::DOMParser dom_parser;
        Poco::AutoPtr<Poco::XML::Document> doc = dom_parser.parseMemory(xml.data(), xml.size());
        const Poco::XML::Node * contents = doc->documentElement()->getNodeByPath("contents");
        for (const Poco::XML::Node * child = contents->firstChild(); child; child = child->nextSibling())
            if (child->nodeName() == "file")
                ++dom_files;
        dom_bytes = static_cast<Int64>(liveAllocatedBytes()) - static_cast<Int64>(before);  /// `doc` still alive
    }

    /// New behavior: stream with the SAX handler; nothing beyond one <file> is retained.
    size_t sax_files = 0;
    Int64 sax_bytes = 0;
    {
        const size_t before = liveAllocatedBytes();
        BackupMetadataHandler handler;
        handler.on_file = [&](const BackupMetadataHandler::Fields &) { ++sax_files; };
        Poco::XML::SAXParser parser;
        parser.setContentHandler(&handler);
        parser.parseMemoryNP(xml.data(), xml.size());
        sax_bytes = static_cast<Int64>(liveAllocatedBytes()) - static_cast<Int64>(before);
    }

    ASSERT_EQ(dom_files, num_files);
    ASSERT_EQ(sax_files, num_files);

    const double mib = 1024.0 * 1024.0;
    std::cerr << "[ MEMORY   ] files=" << num_files
              << "  DOM live-peak=" << static_cast<double>(dom_bytes) / mib << " MiB (" << dom_bytes / static_cast<Int64>(num_files) << " B/file)"
              << "  SAX live-peak=" << static_cast<double>(sax_bytes) / mib << " MiB"
              << "  DOM/SAX=" << (sax_bytes > 0 ? static_cast<double>(dom_bytes) / static_cast<double>(sax_bytes) : 0.0) << "x\n";

    /// The streaming handler must use a small fraction of the DOM tree's memory.
    EXPECT_GT(dom_bytes, sax_bytes * 4);
#endif
}
