#pragma once

#include <Common/CacheBase.h>
#include <Common/HashTable/Hash.h>
#include <Common/Logger.h>
#include <base/types.h>

#include <optional>

namespace DB
{

/// Process-global cache of the raw encryption-header bytes stored at the front of an encrypted
/// file, keyed by the file's stable storage path. It lets the experimental `ReaderExecutor` skip
/// the source read of those header bytes on repeated opens of the same file. Only disk-managed
/// files (whose paths are immutable) are cached; url / external-storage reads never use it, so a
/// reused external path can never return stale headers.
///
/// The cache stores opaque bytes, not parsed `FileEncryption::Header`s, so it carries no SSL
/// dependency and compiles in every build; the consumer re-parses the tiny header on a hit.
class EncryptionHeaderCache
{
public:
    /// The raw serialized header bytes (`N * FileEncryption::Header::kSize`).
    using HeaderBytes = String;

private:
    /// A hash of the storage path.
    using Key = UInt128;

    struct Entry
    {
        HeaderBytes bytes;
        explicit Entry(HeaderBytes bytes_) : bytes(std::move(bytes_)) {}
    };

    struct EntryWeight
    {
        size_t operator()(const Entry & entry) const
        {
            /// Charge the payload plus the fixed per-entry overhead (the path-hash key, the map
            /// node, the shared_ptr control block), so the byte budget bounds real memory rather
            /// than just the header payload.
            static constexpr size_t APPROXIMATE_ENTRY_OVERHEAD = 256;
            return sizeof(Key) + sizeof(Entry) + entry.bytes.capacity() + APPROXIMATE_ENTRY_OVERHEAD;
        }
    };

public:
    using Cache = CacheBase<Key, Entry, UInt128TrivialHash, EntryWeight>;

    static Key makeKey(const String & storage_path);

    EncryptionHeaderCache(const String & cache_policy, size_t max_size_in_bytes, double size_ratio);

    /// Cache the header bytes for `storage_path` (no-op if already present).
    void write(const String & storage_path, HeaderBytes bytes);

    /// Return the cached header bytes for `storage_path`, or nullopt on a miss.
    std::optional<HeaderBytes> read(const String & storage_path);

    void clear();

    void setMaxSizeInBytes(size_t max_size_in_bytes);
    size_t maxSizeInBytes() const;

private:
    Cache cache;
    LoggerPtr logger = getLogger("EncryptionHeaderCache");
};

using EncryptionHeaderCachePtr = std::shared_ptr<EncryptionHeaderCache>;

}
