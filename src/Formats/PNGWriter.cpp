#include <Formats/PNGWriter.h>

#include <array>
#include <limits>

#include <zlib.h>

#include <base/types.h>
#include <Common/Exception.h>
#include <Common/StringWithMemoryTracking.h>
#include <IO/CompressionMethod.h>
#include <IO/WriteBuffer.h>
#include <IO/WriteBufferFromStringWithMemoryTracking.h>
#include <IO/WriteHelpers.h>
#include <IO/ZlibDeflatingWriteBuffer.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{
    /// Deflate level for the image data. 6 is zlib's default: a good size/speed balance.
    constexpr int COMPRESSION_LEVEL = 6;

    /// Write one PNG chunk: 4-byte big-endian length, 4-byte type, the data,
    /// and the 4-byte big-endian CRC-32 of the type followed by the data.
    void writeChunk(WriteBuffer & out, const char (&type)[5], const char * data, size_t size)
    {
        if (size > std::numeric_limits<UInt32>::max())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "PNG chunk is too large ({} bytes)", size);

        writeBinaryBigEndian(static_cast<UInt32>(size), out);
        out.write(type, 4);
        if (size)
            out.write(data, size);

        uLong crc = crc32_z(0, reinterpret_cast<const Bytef *>(type), 4);
        if (size)
            crc = crc32_z(crc, reinterpret_cast<const Bytef *>(data), size);
        writeBinaryBigEndian(static_cast<UInt32>(crc), out);
    }
}

PNGWriter::PNGWriter(WriteBuffer & out_, size_t width_, size_t height_, size_t channels_)
    : out(out_)
    , width(width_)
    , height(height_)
    , channels(channels_)
{
    if (channels != 1 && channels != 3 && channels != 4)
        throw Exception(ErrorCodes::LOGICAL_ERROR,
            "PNG writer supports only 1, 3, or 4 channels per pixel, got {}", channels);

    /// PNG stores width and height as 4-byte unsigned integers and disallows zero.
    static constexpr size_t MAX_DIMENSION = 0x7fffffff;
    if (width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION)
        throw Exception(ErrorCodes::LOGICAL_ERROR,
            "PNG image dimensions must be between 1 and {} (got {}x{})", MAX_DIMENSION, width, height);
}

void PNGWriter::writeImage(const unsigned char * pixels)
{
    if (written)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "PNG writer can encode only one image");
    written = true;

    /// The 8-byte PNG file signature.
    static constexpr std::array<char, 8> signature
        = {static_cast<char>(0x89), 'P', 'N', 'G', '\r', '\n', static_cast<char>(0x1A), '\n'};
    out.write(signature.data(), signature.size());

    /// IHDR: image header (13 bytes).
    UInt8 color_type = 0;
    switch (channels)
    {
        case 1: color_type = 0; break; /// grayscale
        case 3: color_type = 2; break; /// RGB
        case 4: color_type = 6; break; /// RGBA
        default: break; /// unreachable: the number of channels is validated in the constructor
    }

    auto put_be32 = [](char * p, UInt32 value)
    {
        p[0] = static_cast<char>(value >> 24);
        p[1] = static_cast<char>(value >> 16);
        p[2] = static_cast<char>(value >> 8);
        p[3] = static_cast<char>(value);
    };

    char ihdr[13];
    put_be32(ihdr, static_cast<UInt32>(width));
    put_be32(ihdr + 4, static_cast<UInt32>(height));
    ihdr[8] = 8;                              /// bit depth
    ihdr[9] = static_cast<char>(color_type);  /// color type
    ihdr[10] = 0;                             /// compression method: Deflate
    ihdr[11] = 0;                             /// filter method: the standard set (we only ever emit the "None" filter)
    ihdr[12] = 0;                             /// interlace method: none
    writeChunk(out, "IHDR", ihdr, sizeof(ihdr));

    /// IDAT: the pixel data, Deflate-compressed. Each scanline is prefixed with a single filter-type byte;
    /// we always use "None" (0), i.e. the raw bytes, and let Deflate do the compression.
    StringWithMemoryTracking compressed;
    {
        WriteBufferFromStringWithMemoryTracking compressed_buf(compressed);
        ZlibDeflatingWriteBuffer deflate(&compressed_buf, CompressionMethod::Zlib, COMPRESSION_LEVEL);

        const size_t row_bytes = width * channels;
        const char filter_none = 0;
        for (size_t y = 0; y < height; ++y)
        {
            deflate.write(&filter_none, 1);
            deflate.write(reinterpret_cast<const char *>(pixels) + y * row_bytes, row_bytes);
        }
        deflate.finalize(); /// Flushes the Deflate stream and finalizes `compressed_buf`.
    }
    writeChunk(out, "IDAT", compressed.data(), compressed.size());

    /// IEND: end of the image.
    writeChunk(out, "IEND", nullptr, 0);
}

void PNGWriter::finalize()
{
    out.next();
}

}
