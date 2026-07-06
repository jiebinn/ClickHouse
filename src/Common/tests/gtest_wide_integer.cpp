#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <initializer_list>

#include <Core/Types.h>
#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>
#include <base/demangle.h>


static_assert(is_signed_v<Int128>);
static_assert(!is_unsigned_v<Int128>);
static_assert(is_integer<Int128>);
static_assert(sizeof(Int128) == 16);

static_assert(is_signed_v<Int256>);
static_assert(!is_unsigned_v<Int256>);
static_assert(is_integer<Int256>);
static_assert(sizeof(Int256) == 32);

static_assert(!is_signed_v<UInt128>);
static_assert(is_unsigned_v<UInt128>);
static_assert(is_integer<UInt128>);
static_assert(sizeof(UInt128) == 16);

static_assert(!is_signed_v<UInt256>);
static_assert(is_unsigned_v<UInt256>);
static_assert(is_integer<UInt256>);
static_assert(sizeof(UInt256) == 32);


using namespace DB;


GTEST_TEST(WideInteger, Conversions)
{
    ASSERT_EQ(toString(UInt128(12345678901234567890ULL)), "12345678901234567890");
    ASSERT_EQ(toString(UInt256(12345678901234567890ULL)), "12345678901234567890");

    Int128 minus_one = -1;
    ASSERT_EQ(minus_one.items[0], -1);
    ASSERT_EQ(minus_one.items[1], -1);

    ASSERT_EQ(0, memcmp(&minus_one, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(minus_one)));

    ASSERT_EQ(minus_one, -1);
    ASSERT_EQ(minus_one, -1LL);
    ASSERT_EQ(minus_one, Int8(-1));
    ASSERT_EQ(minus_one, Int16(-1));
    ASSERT_EQ(minus_one, Int32(-1));
    ASSERT_EQ(minus_one, Int64(-1));

    ASSERT_LT(minus_one, 0);

    Int128 zero = 0;
    zero += -1;
    ASSERT_EQ(zero, -1);
    ASSERT_EQ(zero, minus_one);

    zero += minus_one;
    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&zero, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE", sizeof(zero)));
    else
        ASSERT_EQ(0, memcmp(&zero, "\xFE\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(zero)));
    zero += 2;
    ASSERT_EQ(zero, 0);

    ASSERT_EQ(toString(Int128(-1)), "-1");
    ASSERT_EQ(toString(Int256(-1)), "-1");

    ASSERT_EQ(toString(Int128(-1LL)), "-1");
    ASSERT_EQ(toString(Int256(-1LL)), "-1");

    ASSERT_EQ(toString(Int128(-1234567890123456789LL)), "-1234567890123456789");
    ASSERT_EQ(toString(Int256(-1234567890123456789LL)), "-1234567890123456789");

    ASSERT_EQ(UInt64(UInt128(12345678901234567890ULL)), 12345678901234567890ULL);
    ASSERT_EQ(UInt64(UInt256(12345678901234567890ULL)), 12345678901234567890ULL);

    ASSERT_EQ(__uint128_t(UInt128(12345678901234567890ULL)), 12345678901234567890ULL);
    ASSERT_EQ(__uint128_t(UInt256(12345678901234567890ULL)), 12345678901234567890ULL);

    ASSERT_EQ(__int128_t(Int128(-1234567890123456789LL)), -1234567890123456789LL);
    ASSERT_EQ(__int128_t(Int256(-1234567890123456789LL)), -1234567890123456789LL);

    ASSERT_EQ(toString(Int128(-1)), "-1");
    ASSERT_EQ(toString(Int256(-1)), "-1");

    ASSERT_EQ(toString(UInt128(123.456)), "123");
    ASSERT_EQ(toString(UInt256(123.456)), "123");
    ASSERT_EQ(toString(Int128(-123.456)), "-123");
    ASSERT_EQ(toString(Int256(-123.456)), "-123");

    ASSERT_EQ(toString(UInt128(123.456f)), "123");
    ASSERT_EQ(toString(UInt256(123.456f)), "123");
    ASSERT_EQ(toString(Int128(-123.456f)), "-123");
    ASSERT_EQ(toString(Int256(-123.456f)), "-123");

    ASSERT_EQ(toString(UInt128(1) * 1000000000 * 1000000000 * 1000000000 * 1000000000), "1000000000000000000000000000000000000");
    ASSERT_EQ(Float64(UInt128(1) * 1000000000 * 1000000000 * 1000000000 * 1000000000), 1e36);

    ASSERT_EQ(toString(UInt256(1) * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000),
        "1000000000000000000000000000000000000000000000000000000000000000000000000");
    ASSERT_EQ(Float64(UInt256(1) * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000 * 1000000000), 1e72);

    EXPECT_EQ(toString(parse<Int128>("148873535527910577765226390751398592640")), "148873535527910577765226390751398592640");
    EXPECT_EQ(toString(parse<UInt128>("148873535527910577765226390751398592640")), "148873535527910577765226390751398592640");
}


template <typename T>
static T divide(T & numerator, T && denominator)
{
    if (!denominator)
        throwError("Division by zero");

    T & n = numerator;
    T & d = denominator;
    T x = 1;
    T quotient = 0;

    /// Multiply d to the power of two until it will be greater than n.
    /// The factor will be collected in x.
    while (d <= n && ((d >> (sizeof(T) * 8 - 1)) & 1) == 0)
    {
        x <<= 1;
        d <<= 1;
    }

    std::cerr << toString(x) << ", " << toString(d) << "\n";

    while (x)
    {
        if (d <= n)
        {
            n -= d;
            quotient |= x;
        }

        x >>= 1;
        d >>= 1;
    }

    return quotient;
}


GTEST_TEST(WideInteger, Arithmetic)
{
    Int128 minus_one = -1;
    Int128 zero = 0;

    zero += -1;
    ASSERT_EQ(zero, -1);
    ASSERT_EQ(zero, minus_one);

    zero += minus_one;
    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&zero, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE", sizeof(zero)));
    else
        ASSERT_EQ(0, memcmp(&zero, "\xFE\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(zero)));
    zero += 2;
    ASSERT_EQ(zero, 0);

    ASSERT_EQ(UInt256(12345678901234567890ULL) * 12345678901234567890ULL / 12345678901234567890ULL, 12345678901234567890ULL);
    ASSERT_EQ(UInt256(12345678901234567890ULL) * UInt256(12345678901234567890ULL) / 12345678901234567890ULL, 12345678901234567890ULL);
    ASSERT_EQ(UInt256(12345678901234567890ULL) * 12345678901234567890ULL / UInt256(12345678901234567890ULL), 12345678901234567890ULL);
    ASSERT_EQ(UInt256(12345678901234567890ULL) * 12345678901234567890ULL / 12345678901234567890ULL, UInt256(12345678901234567890ULL));
    ASSERT_EQ(UInt128(12345678901234567890ULL) * 12345678901234567890ULL / UInt128(12345678901234567890ULL), 12345678901234567890ULL);
    ASSERT_EQ(UInt256(12345678901234567890ULL) * UInt128(12345678901234567890ULL) / 12345678901234567890ULL, 12345678901234567890ULL);

    ASSERT_EQ(Int128(0) + Int32(-1), Int128(-1));

    Int128 x(parse<Int128>("148873535527910577765226390751398592640"));
    Int128 dividend = x / 10;
    ASSERT_EQ(toString(dividend), "14887353552791057776522639075139859264");
}


/// The 128-bit same-type ordering operators take a fast path via native __int128
/// (see operator_less / operator_greater in wide_integer_impl.h), while 256-bit
/// comparisons use the generic limb-wise loop. Comparing each pair in both widths
/// checks the fast path against an independent implementation.
template <typename T128, typename T256>
static void checkComparisonAgainstWiderOracle(const std::vector<T128> & values)
{
    for (const T128 & lhs : values)
    {
        for (const T128 & rhs : values)
        {
            const T256 wide_lhs = lhs;
            const T256 wide_rhs = rhs;

            EXPECT_EQ(lhs < rhs, wide_lhs < wide_rhs) << toString(lhs) << " < " << toString(rhs);
            EXPECT_EQ(lhs > rhs, wide_lhs > wide_rhs) << toString(lhs) << " > " << toString(rhs);
            EXPECT_EQ(lhs <= rhs, wide_lhs <= wide_rhs) << toString(lhs) << " <= " << toString(rhs);
            EXPECT_EQ(lhs >= rhs, wide_lhs >= wide_rhs) << toString(lhs) << " >= " << toString(rhs);
            EXPECT_EQ(lhs == rhs, wide_lhs == wide_rhs) << toString(lhs) << " == " << toString(rhs);
            EXPECT_EQ(lhs != rhs, wide_lhs != wide_rhs) << toString(lhs) << " != " << toString(rhs);
        }
    }
}


GTEST_TEST(WideInteger, Comparison128Boundaries)
{
    /// Constexpr evaluation must take the same fast path.
    static_assert(std::numeric_limits<Int128>::min() < Int128(-1));
    static_assert(Int128(-1) < Int128(0));
    static_assert(Int128(0) < std::numeric_limits<Int128>::max());
    static_assert(!(Int128(-1) < Int128(-1)));
    static_assert(UInt128(0) < std::numeric_limits<UInt128>::max());
    static_assert((UInt128(1) << 127) > ((UInt128(1) << 127) - 1));

    {
        const Int128 min = std::numeric_limits<Int128>::min();
        const Int128 max = std::numeric_limits<Int128>::max();
        const Int128 two_pow_64 = Int128(1) << 64;
        const Int128 high_limb = Int128(5) << 64;

        ASSERT_LT(min, Int128(-1));
        ASSERT_LT(Int128(-1), Int128(0));
        ASSERT_LT(Int128(0), Int128(1));
        ASSERT_LT(Int128(1), max);
        ASSERT_LT(min, max);
        ASSERT_LT(-two_pow_64, Int128(-1));
        ASSERT_LT(high_limb, high_limb + 1);

        checkComparisonAgainstWiderOracle<Int128, Int256>({
            0, 1, -1, 2, -2,
            min, min + 1, max, max - 1,
            two_pow_64 - 1, two_pow_64, two_pow_64 + 1,
            -(two_pow_64 - 1), -two_pow_64, -(two_pow_64 + 1),
            high_limb - 1, high_limb, high_limb + 1,
            -(high_limb - 1), -high_limb, -(high_limb + 1),
        });
    }

    {
        const UInt128 max = std::numeric_limits<UInt128>::max();
        const UInt128 sign_bit = UInt128(1) << 127;
        const UInt128 two_pow_64 = UInt128(1) << 64;
        const UInt128 high_limb = UInt128(5) << 64;

        ASSERT_LT(UInt128(0), UInt128(1));
        ASSERT_LT(sign_bit - 1, sign_bit);
        ASSERT_LT(sign_bit, max);
        ASSERT_LT(high_limb, high_limb + 1);

        checkComparisonAgainstWiderOracle<UInt128, UInt256>({
            0, 1, 2,
            max, max - 1,
            two_pow_64 - 1, two_pow_64, two_pow_64 + 1,
            sign_bit - 1, sign_bit, sign_bit + 1,
            high_limb - 1, high_limb, high_limb + 1,
        });
    }
}


GTEST_TEST(WideInteger, DecimalArithmetic)
{
    Decimal128 zero{};
    Decimal32 addend = -1000;

    zero += Decimal128(addend);
    ASSERT_EQ(zero.value, -1000);

    zero += addend;
    ASSERT_EQ(zero.value, -2000);
}


GTEST_TEST(WideInteger, FromDouble)
{
    /// Check that we are being able to convert double to big integer without the help of floating point instructions.
    /// (a prototype of a function that we may need)

    double f = -123.456;
    UInt64 u = {};
    memcpy(&u, &f, sizeof(f));

    bool is_negative = u >> 63;
    uint16_t exponent = (u >> 52) & (((1ull << 12) - 1) >> 1);
    int16_t normalized_exponent = exponent - 1023;
    UInt64 mantissa = u & ((1ull << 52) - 1);

    // std::cerr << is_negative << ", " << normalized_exponent << ", " << mantissa << "\n";

    /// x = sign * (2 ^ normalized_exponent + mantissa * 2 ^ (normalized_exponent - mantissa_bits))

    Int128 res = 0;

    if (normalized_exponent >= 128)
    {
    }
    else
    {
        res = mantissa;
        if (normalized_exponent > 52)
            res <<= (normalized_exponent - 52);
        else
            res >>= (52 - normalized_exponent);

        if (normalized_exponent > 0)
            res += Int128(1) << normalized_exponent;
    }

    if (is_negative)
        res = -res;

    ASSERT_EQ(toString(res), "-123");
}


GTEST_TEST(WideInteger, Shift)
{
    Int128 x = 1;

    auto y = x << 64;

    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&y, "\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00", sizeof(Int128)));
    else
        ASSERT_EQ(0, memcmp(&y, "\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00", sizeof(Int128)));
    auto z = y << 11;
    ASSERT_EQ(toString(z), "37778931862957161709568");

    auto a = x << 11;
    ASSERT_EQ(a, 2048);

    z >>= 64;
    ASSERT_EQ(z, a);

    x = -1;
    y = x << 16;

    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&y, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00", sizeof(Int128)));
    else
        ASSERT_EQ(0, memcmp(&y, "\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(Int128)));
    y >>= 16;
    ASSERT_EQ(0, memcmp(&y, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(Int128)));

    y <<= 64;
    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&y, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\x00\x00\x00\x00", sizeof(Int128)));
    else
        ASSERT_EQ(0, memcmp(&y, "\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(Int128)));
    y >>= 32;
    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&y, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00", sizeof(Int128)));
    else
        ASSERT_EQ(0, memcmp(&y, "\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", sizeof(Int128)));

    y <<= 64;
    if constexpr (std::endian::native == std::endian::big)
        ASSERT_EQ(0, memcmp(&y, "\xFF\xFF\xFF\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", sizeof(Int128)));
    else
        ASSERT_EQ(0, memcmp(&y, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF\xFF", sizeof(Int128)));
}


GTEST_TEST(WideInteger, DecimalFormatting)
{
    Decimal128 x(parse<Int128>("148873535527910577765226390751398592640"));

    EXPECT_EQ(toString(x.value), "148873535527910577765226390751398592640");
    EXPECT_EQ(toString(x.value / 10), "14887353552791057776522639075139859264");
    EXPECT_EQ(toString(x.value % 10), "0");

    Int128 fractional = DecimalUtils::getFractionalPart(x, 2);

    EXPECT_EQ(fractional, 40);
}
