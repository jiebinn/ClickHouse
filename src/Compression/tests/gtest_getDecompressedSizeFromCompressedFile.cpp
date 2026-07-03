#include <string>

#include <Compression/CompressedReadBufferBase.h>
#include <Compression/CompressedWriteBuffer.h>
#include <Compression/CompressionFactory.h>
#include <IO/ReadBufferFromString.h>
#include <IO/WriteBufferFromString.h>

#include <gtest/gtest.h>

using namespace DB;

namespace
{
/// A small block_size forces multiple blocks so the header walk spans block boundaries.
std::string compress(const std::string & data, size_t block_size)
{
    std::string compressed;
    {
        WriteBufferFromString out(compressed);
        CompressedWriteBuffer compressed_out(out, CompressionCodecFactory::instance().getDefaultCodec(), block_size);
        compressed_out.write(data.data(), data.size());
        compressed_out.finalize();
    }
    return compressed;
}
}

TEST(GetDecompressedSizeFromCompressedFile, SingleBlock)
{
    const std::string data(1000, 'x');
    auto compressed = compress(data, 1 << 20);

    ReadBufferFromString in(compressed);
    EXPECT_EQ(getDecompressedSizeFromCompressedFile(in), data.size());
}

TEST(GetDecompressedSizeFromCompressedFile, MultipleBlocks)
{
    std::string data;
    for (size_t i = 0; i < 100 * 1024; ++i)
        data.push_back(static_cast<char>(i * 2654435761u >> 24));
    auto compressed = compress(data, 4096);

    ReadBufferFromString in(compressed);
    EXPECT_EQ(getDecompressedSizeFromCompressedFile(in), data.size());
}

TEST(GetDecompressedSizeFromCompressedFile, Empty)
{
    auto compressed = compress("", 1 << 20);

    ReadBufferFromString in(compressed);
    EXPECT_EQ(getDecompressedSizeFromCompressedFile(in), 0u);
}
