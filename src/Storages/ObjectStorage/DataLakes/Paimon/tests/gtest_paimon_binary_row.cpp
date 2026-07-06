#include <gtest/gtest.h>

#include <limits>

#include <Storages/ObjectStorage/DataLakes/Paimon/BinaryRow.h>
#include <base/types.h>

using namespace Paimon;

namespace
{

/// Builds a little-endian Paimon BinaryRow blob holding a single non-null fixed-size field.
/// Layout: 4-byte big-endian arity, then the null bitset, then one 8-byte value slot.
String makeSingleFieldRow(UInt64 raw_value)
{
    constexpr Int32 arity = 1;
    /// ((arity + 63 + HEADER_SIZE_IN_BITS) / 64) * 8 == 8 for arity == 1.
    constexpr Int32 bit_set_width = 8;

    String bytes;
    /// arity is stored big-endian.
    bytes.push_back(static_cast<char>((arity >> 24) & 0xFF));
    bytes.push_back(static_cast<char>((arity >> 16) & 0xFF));
    bytes.push_back(static_cast<char>((arity >> 8) & 0xFF));
    bytes.push_back(static_cast<char>(arity & 0xFF));

    /// Null bitset: all zeros -> field 0 is not null.
    bytes.append(bit_set_width, '\0');

    /// 8-byte value slot, little-endian.
    for (int i = 0; i < 8; ++i)
        bytes.push_back(static_cast<char>((raw_value >> (i * 8)) & 0xFF));

    return bytes;
}

}

/// Paimon stores every fixed-size field in an 8-byte slot, so getLong must read all 64 bits.
/// Reading only the low 32 bits truncated large BIGINT partition values (issue #109477):
/// e.g. 9223372036854775807 became -1.
TEST(PaimonBinaryRow, GetLongReadsFull64Bits)
{
    {
        BinaryRow row(makeSingleFieldRow(1000));
        EXPECT_EQ(row.getLong(0), 1000);
    }
    {
        /// Value beyond the Int32 range must not truncate.
        BinaryRow row(makeSingleFieldRow(5000000000ULL));
        EXPECT_EQ(row.getLong(0), 5000000000LL);
    }
    {
        BinaryRow row(makeSingleFieldRow(static_cast<UInt64>(std::numeric_limits<Int64>::max())));
        EXPECT_EQ(row.getLong(0), std::numeric_limits<Int64>::max());
    }
    {
        BinaryRow row(makeSingleFieldRow(static_cast<UInt64>(std::numeric_limits<Int64>::min())));
        EXPECT_EQ(row.getLong(0), std::numeric_limits<Int64>::min());
    }
    {
        BinaryRow row(makeSingleFieldRow(static_cast<UInt64>(-1LL)));
        EXPECT_EQ(row.getLong(0), -1);
    }
}

/// getInt must keep reading exactly 32 bits (regression guard for the sibling getter).
TEST(PaimonBinaryRow, GetIntReads32Bits)
{
    BinaryRow row(makeSingleFieldRow(static_cast<UInt64>(-1LL)));
    EXPECT_EQ(row.getInt(0), -1);
}
