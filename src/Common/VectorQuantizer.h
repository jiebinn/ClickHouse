#pragma once

#include <base/types.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace DB
{

/// Stateless, data-independent quantization of dense floating-point vectors.
///
/// Methods:
/// - turboquant
/// - rabitq
/// - int8
/// - mrl_int8 (Matryoshka)
/// - mrl_bf16 (Matryoshka)
struct VectorQuantizer
{
    /// Whether the method's approximate distance can rank by L2, i.e. if it retains/estimates the vector norm.
    static bool supportsL2(std::string_view method);

    /// Returns a human-readable non-empty error if an invalid configuration is given, or an empty string if valid.
    static std::string validateParams(std::string_view method, size_t dimensions, size_t bits);

    /// Size of one encoded vector in bytes for the given method/dimensions/bits.
    static size_t bytesPerVector(std::string_view method, size_t dimensions, size_t bits);

    struct Encoder;

    /// Prepare an encoder for a method/dimensions/bits
    static std::shared_ptr<Encoder> createEncoder(std::string_view method, size_t dimensions, size_t bits);

    /// Encode one `dimensions`-element vector into `dst` (exactly `bytesPerVector` bytes) with a prepared encoder.
    static void encode(Encoder & encoder, const float * vec, char * dst);

    /// Encode one `dimensions`-element vector into `dst` (exactly `bytesPerVector` bytes).
    /// Expensive as it builds an `Encoder` internally, prefer `createEncoder` + `encode(encoder, ...)` for many vectors.
    static void encode(std::string_view method, const float * vec, size_t dimensions, size_t bits, char * dst);

    struct Query;

    /// Prepare the query state once for a reference vector; `is_l2` selects L2Distance vs cosineDistance for 'e8'.
    static std::shared_ptr<const Query> prepareQuery(std::string_view method, const float * ref, size_t dimensions, size_t bits, bool is_l2);

    /// Calculates the approximate distance between the prepared query and one encoded vector (`code` is `bytesPerVector` bytes).
    static float distance(const Query & query, const char * code);
};

}
