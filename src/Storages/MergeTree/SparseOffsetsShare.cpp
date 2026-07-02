#include <Storages/MergeTree/SparseOffsetsShare.h>

#include <Columns/ColumnsNumber.h>
#include <DataTypes/Serializations/SerializationSparse.h>

#include <algorithm>
#include <mutex>
#include <shared_mutex>


namespace DB
{

void SparseOffsetsShare::insert(
    const std::string & part_name,
    const std::string & column_name,
    MarkRange range,
    size_t start_row_in_part,
    size_t total_rows,
    ColumnPtr offsets)
{
    std::unique_lock lock(mutex);
    auto & bucket = store[part_name][column_name];
    bucket.ranges.push_back(SparseOffsetsRange{range, start_row_in_part, total_rows, std::move(offsets)});
}

const SparseOffsetsShare::Bucket *
SparseOffsetsShare::findBucket(const std::string & part_name, const std::string & column_name) const
{
    std::shared_lock lock(mutex);

    auto part_it = store.find(part_name);
    if (part_it == store.end())
        return nullptr;

    auto col_it = part_it->second.find(column_name);
    if (col_it == part_it->second.end())
        return nullptr;

    return &col_it->second;
}

bool SparseOffsetsShare::empty() const
{
    std::shared_lock lock(mutex);
    return store.empty();
}

namespace
{

/// Drop stored ranges in `columns` that no surviving `MarkRange` overlaps; collapse empty
/// column entries. Returns true when the whole part bucket becomes empty.
bool pruneColumns(
    std::unordered_map<std::string, SparseOffsetsShare::Bucket> & columns,
    const MarkRanges & surviving_ranges)
{
    for (auto & column_entry : columns)
    {
        std::erase_if(column_entry.second.ranges, [&](const SparseOffsetsRange & stored)
        {
            for (const auto & sr : surviving_ranges)
            {
                if (stored.range.begin < sr.end && sr.begin < stored.range.end)
                    return false;
            }
            return true;
        });
    }
    std::erase_if(columns, [](const auto & col) { return col.second.ranges.empty(); });
    return columns.empty();
}

}

void SparseOffsetsShare::retainRangesForPart(const std::string & part_name, const MarkRanges & surviving_ranges)
{
    std::unique_lock lock(mutex);
    auto it = store.find(part_name);
    if (it == store.end())
        return;
    if (pruneColumns(it->second, surviving_ranges))
        store.erase(it);
}

void SparseOffsetsShare::retainSurvivingRanges(
    const std::unordered_map<std::string, MarkRanges> & per_part_surviving_ranges)
{
    std::unique_lock lock(mutex);
    std::erase_if(store, [&](auto & part_entry)
    {
        const auto it = per_part_surviving_ranges.find(part_entry.first);
        if (it == per_part_surviving_ranges.end())
            return true;
        return pruneColumns(part_entry.second, it->second);
    });
}

std::unique_ptr<SubstreamsCacheSparseOffsetsElement>
SparseOffsetsShare::sliceFromBucket(
    const Bucket & bucket,
    size_t abs_row_start,
    size_t rows_offset,
    size_t limit,
    size_t frame_prev_size)
{
    const auto & ranges = bucket.ranges;
    const size_t abs_row_end = abs_row_start + rows_offset + limit;

    /// Find the entry that covers `abs_row_start`. Entries are inserted in order of
    /// ascending `start_row_in_part`, so a linear scan picks the right starting chunk
    /// in `O(num stored ranges)`. The number of ranges depends on the selected mark
    /// ranges and on how the analyzer split the work. The per-reader bucket cache in
    /// `IMergeTreeReader` avoids repeating the `(part, column)` map lookup before this
    /// scan.
    const SparseOffsetsRange * start = nullptr;
    size_t start_idx = 0;
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        const auto & entry = ranges[i];
        const size_t entry_end_row = entry.start_row_in_part + entry.total_rows;
        if (entry.start_row_in_part <= abs_row_start && abs_row_start < entry_end_row)
        {
            start = &entry;
            start_idx = i;
            break;
        }
    }
    if (!start)
        return nullptr;

    const size_t start_end_row = start->start_row_in_part + start->total_rows;
    const bool single_chunk = abs_row_end <= start_end_row;

    /// `[skip_start_rel, skip_end_rel)` is the rows_offset zone (non-defaults here count
    /// as `skipped_values_rows`). `[skip_end_rel, produce_end_rel)` is the produce zone
    /// (non-defaults here are emitted to the offsets column). Both are relative to the
    /// first chunk's `start_row_in_part`.
    const size_t skip_start_rel = abs_row_start - start->start_row_in_part;
    const size_t skip_end_rel = skip_start_rel + rows_offset;

    if (single_chunk)
    {
        /// Fast path: scan window lies inside one stored chunk. Return a deferred-slice
        /// descriptor that points into the chunk's offsets; the consumer appends
        /// `src[i] + shift` directly into its persistent offsets column, avoiding both
        /// an intermediate allocation and a later `insertRangeFrom` copy.
        const size_t produce_end_rel = skip_end_rel + limit;
        const auto & src_offsets = assert_cast<const ColumnUInt64 &>(*start->offsets).getData();
        const auto * begin = src_offsets.data();
        const auto * end = begin + src_offsets.size();

        const auto * produce_zone_begin = std::lower_bound(begin, end, skip_end_rel);
        size_t skipped = 0;
        if (rows_offset != 0)
        {
            const auto * skip_zone_begin = std::lower_bound(begin, produce_zone_begin, skip_start_rel);
            skipped = produce_zone_begin - skip_zone_begin;
        }
        const auto * produce_zone_end = std::lower_bound(produce_zone_begin, end, produce_end_rel);

        const UInt64 shift = static_cast<UInt64>(frame_prev_size) - static_cast<UInt64>(skip_end_rel);
        return std::make_unique<SubstreamsCacheSparseOffsetsElement>(
            produce_zone_begin,
            static_cast<size_t>(produce_zone_end - produce_zone_begin),
            shift,
            /*read_rows_=*/limit,
            /*skipped_values_rows_=*/skipped);
    }

    /// The scan window crosses one or more stored-range boundaries. Walk the ranges in
    /// order, slice each one, and stitch the pieces into a fresh column. The disk
    /// fallback would be incorrect here because the scan's `DeserializeStateSparse` was
    /// never advanced through the SparseOffsets stream (previous calls were cache hits),
    /// so we must always produce a result. This path is rare because analyzer ranges are
    /// usually much larger than scan blocks, so the extra allocation is acceptable.
    auto stitched = ColumnUInt64::create();
    auto & stitched_data = stitched->getData();

    /// Compute size first to avoid `push_back` growth and to enable a single allocation.
    size_t skipped_total = 0;
    size_t produce_total = 0;
    for (size_t i = start_idx; i < ranges.size(); ++i)
    {
        const auto & entry = ranges[i];
        const size_t entry_start_row = entry.start_row_in_part;
        const size_t entry_end_row = entry_start_row + entry.total_rows;
        if (entry_start_row >= abs_row_end)
            break;
        if (entry_end_row <= abs_row_start)
            continue;

        const auto & entry_offsets = assert_cast<const ColumnUInt64 &>(*entry.offsets).getData();
        const auto * offsets_begin = entry_offsets.data();
        const auto * offsets_end = offsets_begin + entry_offsets.size();

        /// Rows of the scan window that fall inside this entry, split into skip
        /// (`rows_offset` prefix, counted but not emitted) and produce (emitted).
        const size_t entry_skip_start_row = std::max(abs_row_start, entry_start_row);
        const size_t entry_skip_end_row = std::min(abs_row_start + rows_offset, entry_end_row);
        const size_t entry_produce_start_row = std::max(abs_row_start + rows_offset, entry_start_row);
        const size_t entry_produce_end_row = std::min(abs_row_end, entry_end_row);

        if (entry_skip_end_row > entry_skip_start_row)
        {
            const size_t skip_start_within_entry = entry_skip_start_row - entry_start_row;
            const size_t skip_end_within_entry = entry_skip_end_row - entry_start_row;
            const auto * slice_begin = std::lower_bound(offsets_begin, offsets_end, skip_start_within_entry);
            const auto * slice_end = std::lower_bound(slice_begin, offsets_end, skip_end_within_entry);
            skipped_total += slice_end - slice_begin;
        }
        if (entry_produce_end_row > entry_produce_start_row)
        {
            const size_t produce_start_within_entry = entry_produce_start_row - entry_start_row;
            const size_t produce_end_within_entry = entry_produce_end_row - entry_start_row;
            const auto * slice_begin = std::lower_bound(offsets_begin, offsets_end, produce_start_within_entry);
            const auto * slice_end = std::lower_bound(slice_begin, offsets_end, produce_end_within_entry);
            produce_total += slice_end - slice_begin;
        }
    }
    stitched_data.resize(produce_total);

    size_t out_pos = 0;
    for (size_t i = start_idx; i < ranges.size(); ++i)
    {
        const auto & entry = ranges[i];
        const size_t entry_start_row = entry.start_row_in_part;
        const size_t entry_end_row = entry_start_row + entry.total_rows;
        if (entry_start_row >= abs_row_end)
            break;
        if (entry_end_row <= abs_row_start)
            continue;

        const size_t entry_produce_start_row = std::max(abs_row_start + rows_offset, entry_start_row);
        const size_t entry_produce_end_row = std::min(abs_row_end, entry_end_row);
        if (entry_produce_end_row <= entry_produce_start_row)
            continue;

        const auto & entry_offsets = assert_cast<const ColumnUInt64 &>(*entry.offsets).getData();
        const auto * offsets_begin = entry_offsets.data();
        const auto * offsets_end = offsets_begin + entry_offsets.size();
        const size_t produce_start_within_entry = entry_produce_start_row - entry_start_row;
        const size_t produce_end_within_entry = entry_produce_end_row - entry_start_row;
        const auto * slice_begin = std::lower_bound(offsets_begin, offsets_end, produce_start_within_entry);
        const auto * slice_end = std::lower_bound(slice_begin, offsets_end, produce_end_within_entry);

        /// Each source position `p` is relative to `entry_start_row`. To put it in the
        /// consumer's frame, shift to
        /// `(p + entry_start_row) - (abs_row_start + rows_offset) + frame_prev_size`.
        const UInt64 shift = static_cast<UInt64>(entry_start_row)
            + static_cast<UInt64>(frame_prev_size)
            - static_cast<UInt64>(abs_row_start + rows_offset);
        const size_t slice_count = slice_end - slice_begin;
        UInt64 * __restrict__ dst = stitched_data.data() + out_pos;
        const UInt64 * __restrict__ src = slice_begin;
        for (size_t j = 0; j < slice_count; ++j)
            dst[j] = src[j] + shift;
        out_pos += slice_count;
    }

    return std::make_unique<SubstreamsCacheSparseOffsetsElement>(
        ColumnPtr(std::move(stitched)),
        /*old_size_=*/0,
        /*read_rows_=*/limit,
        /*skipped_values_rows_=*/skipped_total);
}

}
