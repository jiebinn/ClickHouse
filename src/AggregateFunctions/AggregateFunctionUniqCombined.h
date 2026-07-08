#pragma once

#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/Helpers.h>

#include <Common/FieldVisitorConvertToNumber.h>

#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDate32.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeIPv4andIPv6.h>

#include <base/bit_cast.h>

#include <Common/CombinedCardinalityEstimator.h>
#include <Common/SipHash.h>
#include <Common/typeid_cast.h>
#include <Common/assert_cast.h>

#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeUUID.h>
#include <DataTypes/DataTypesNumber.h>

#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/UniqCombinedBiasData.h>
#include <AggregateFunctions/UniqVariadicHash.h>

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>

#include <functional>


namespace DB
{

struct Settings;


// Unlike HashTableGrower always grows to power of 2.
struct UniqCombinedHashTableGrower : public HashTableGrowerWithPrecalculation<>
{
    void increaseSize() { increaseSizeDegree(1); }
};

template <typename T, UInt8 K, typename HashValueType>
struct AggregateFunctionUniqCombinedData
{
    using Key = std::conditional_t<
        std::is_same_v<T, String> || std::is_same_v<T, IPv6>,
        UInt64,
        HashValueType>;

    // TODO(ilezhankin): pre-generate values for |UniqCombinedBiasData|,
    //                   at the moment gen-bias-data.py script doesn't work.

    // We want to migrate from |HashSet| to |HyperLogLogCounter| when the sizes in memory become almost equal.
    // The size per element in |HashSet| is sizeof(Key)*2 bytes, and the overall size of |HyperLogLogCounter| is 2^K * 6 bits.
    // For Key=UInt32 we can calculate: 2^X * 4 * 2 ≤ 2^(K-3) * 6 ⇒ X ≤ K-4.

    /// Note: I don't recall what is special with '17' - probably it is one of the original functions that has to be compatible.
    using Set = CombinedCardinalityEstimator<
        Key,
        HashSet<Key, TrivialHash, UniqCombinedHashTableGrower>,
        16,
        K - 5 + (sizeof(Key) == sizeof(UInt32)),
        K,
        TrivialHash,
        Key,
        std::conditional_t<K == 17, HyperLogLogBiasEstimator<UniqCombinedBiasData>, TrivialBiasEstimator>,
        HyperLogLogMode::FullFeatured>;

    Set set;
};


template <typename T, UInt8 K, typename HashValueType>
class AggregateFunctionUniqCombined final
    : public IAggregateFunctionDataHelper<AggregateFunctionUniqCombinedData<T, K, HashValueType>, AggregateFunctionUniqCombined<T, K, HashValueType>>
{
public:
    AggregateFunctionUniqCombined(const DataTypes & argument_types_, const Array & params_)
        : IAggregateFunctionDataHelper<AggregateFunctionUniqCombinedData<T, K, HashValueType>, AggregateFunctionUniqCombined<T, K, HashValueType>>(argument_types_, params_, std::make_shared<DataTypeUInt64>())
    {}

    String getName() const override
    {
        if constexpr (std::is_same_v<HashValueType, UInt64>)
            return "uniqCombined64";
        else
            return "uniqCombined";
    }

    bool allocatesMemoryInArena() const override { return false; }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        if constexpr (std::is_same_v<T, String> || std::is_same_v<T, IPv6>)
        {
            auto value = columns[0]->getDataAt(row_num);
            this->data(place).set.insert(CityHash_v1_0_2::CityHash64(value.data(), value.size()));
        }
        else
        {
            const auto & value = assert_cast<const ColumnVector<T> &>(*columns[0]).getElement(row_num);
            this->data(place).set.insert(hashOne(value));
        }
    }

    void addBatchSinglePlace( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        Arena *,
        ssize_t if_argument_pos = -1) const override
    {
        const UInt8 * flags = nullptr;
        if (if_argument_pos >= 0)
            flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData().data();

        addBatchImpl(row_begin, row_end, place, columns, flags, nullptr);
    }

    void addBatchSinglePlaceNotNull( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        const UInt8 * null_map,
        Arena *,
        ssize_t if_argument_pos = -1) const override
    {
        const UInt8 * flags = nullptr;
        if (if_argument_pos >= 0)
            flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData().data();

        addBatchImpl(row_begin, row_end, place, columns, flags, null_map);
    }

    void addManyDefaults(AggregateDataPtr __restrict place, const IColumn ** columns, size_t /*length*/, Arena * arena) const override
    {
        this->add(place, columns, 0, arena);
    }

    void mergeImpl(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).set.merge(this->data(rhs).set);
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> /* version */) const override
    {
        this->data(place).set.write(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> /* version */, Arena *) const override
    {
        this->data(place).set.read(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        assert_cast<ColumnUInt64 &>(to).getData().push_back(this->data(place).set.size());
    }

private:
    using Data = AggregateFunctionUniqCombinedData<T, K, HashValueType>;
    using Key = typename Data::Key;

    static constexpr size_t hash_chunk_size = 256;

    static ALWAYS_INLINE HashValueType hashOne(T value)
    {
        if constexpr (std::is_same_v<T, UInt128>)
        {
            /// This specialization exists due to historical circumstances.
            /// Initially UInt128 was introduced only for UUID, and then the other big-integer types were added.
            return static_cast<HashValueType>(sipHash64(value));
        }
        else if constexpr (is_floating_point<T>)
        {
            return static_cast<HashValueType>(intHash64(bit_cast<UInt64>(value)));
        }
        else if constexpr (sizeof(T) > sizeof(UInt64))
        {
            return static_cast<HashValueType>(DefaultHash64<T>(value));
        }
        else
        {
            /// This specialization exists also for compatibility with the initial implementation.
            return static_cast<HashValueType>(intHash64(value));
        }
    }

    void addBatchImpl(
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        const UInt8 * flags,
        const UInt8 * null_map) const
    {
        auto & set = this->data(place).set;

        if constexpr (std::is_same_v<T, String>)
        {
            if (const auto * column_string = typeid_cast<const ColumnString *>(columns[0]))
            {
                const auto & chars = column_string->getChars();
                const auto & offsets = column_string->getOffsets();

                insertChunked(row_begin, row_end, set, flags, null_map, [&](size_t row)
                {
                    return CityHash_v1_0_2::CityHash64(reinterpret_cast<const char *>(chars.data()) + offsets[row - 1], offsets[row] - offsets[row - 1]);
                });
            }
            else
            {
                const auto & column_fixed = assert_cast<const ColumnFixedString &>(*columns[0]);
                const auto & chars = column_fixed.getChars();
                const size_t n = column_fixed.getN();

                insertChunked(row_begin, row_end, set, flags, null_map, [&](size_t row)
                {
                    return CityHash_v1_0_2::CityHash64(reinterpret_cast<const char *>(chars.data()) + row * n, n);
                });
            }
        }
        else if constexpr (std::is_same_v<T, IPv6>)
        {
            const auto & data = assert_cast<const ColumnVector<IPv6> &>(*columns[0]).getData();

            insertChunked(row_begin, row_end, set, flags, null_map, [&](size_t row)
            {
                return CityHash_v1_0_2::CityHash64(reinterpret_cast<const char *>(&data[row]), sizeof(IPv6));
            });
        }
        else
        {
            const auto & data = assert_cast<const ColumnVector<T> &>(*columns[0]).getData();
            insertChunked(row_begin, row_end, set, flags, null_map, [&](size_t row) { return hashOne(data[row]); });
        }
    }

    /// Hashes rows a chunk ahead into a stack buffer, so that hash computation pipelines
    /// independently of the set inserts, and inserts whole chunks via insertMany.
    template <typename Hasher>
    static void insertChunked(size_t row_begin, size_t row_end, typename Data::Set & set, const UInt8 * flags, const UInt8 * null_map, Hasher hasher)
    {
        Key hashes[hash_chunk_size];
        size_t row = row_begin;

        while (row < row_end)
        {
            const size_t chunk_size = std::min(hash_chunk_size, row_end - row);
            size_t num_hashes = 0;

            if (!flags && !null_map)
            {
                for (size_t i = 0; i < chunk_size; ++i)
                    hashes[i] = hasher(row + i);

                num_hashes = chunk_size;
            }
            else
            {
                for (size_t i = 0; i < chunk_size; ++i)
                {
                    if ((flags && !flags[row + i]) || (null_map && null_map[row + i]))
                        continue;

                    hashes[num_hashes] = hasher(row + i);
                    ++num_hashes;
                }
            }

            set.insertMany(hashes, num_hashes);
            row += chunk_size;
        }
    }
};

/** For multiple arguments. To compute, hashes them.
  * You can pass multiple arguments as is; You can also pass one argument - a tuple.
  * But (for the possibility of efficient implementation), you can not pass several arguments, among which there are tuples.
  */
template <bool is_exact, bool argument_is_tuple, UInt8 K, typename HashValueType>
class AggregateFunctionUniqCombinedVariadic final : public IAggregateFunctionDataHelper<AggregateFunctionUniqCombinedData<UInt64, K, HashValueType>,
                                                           AggregateFunctionUniqCombinedVariadic<is_exact, argument_is_tuple, K, HashValueType>>
{
private:
    size_t num_args = 0;

public:
    explicit AggregateFunctionUniqCombinedVariadic(const DataTypes & arguments, const Array & params)
        : IAggregateFunctionDataHelper<AggregateFunctionUniqCombinedData<UInt64, K, HashValueType>,
            AggregateFunctionUniqCombinedVariadic<is_exact, argument_is_tuple, K, HashValueType>>(arguments, params, std::make_shared<DataTypeUInt64>())
    {
        if (argument_is_tuple)
            num_args = typeid_cast<const DataTypeTuple &>(*arguments[0]).getElements().size();
        else
            num_args = arguments.size();
    }

    String getName() const override
    {
        if constexpr (std::is_same_v<HashValueType, UInt64>)
            return "uniqCombined64";
        else
            return "uniqCombined";
    }

    bool allocatesMemoryInArena() const override { return false; }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        this->data(place).set.insert(typename AggregateFunctionUniqCombinedData<UInt64, K, HashValueType>::Set::value_type(
            UniqVariadicHash<is_exact, argument_is_tuple>::apply(num_args, columns, row_num)));
    }

    void mergeImpl(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).set.merge(this->data(rhs).set);
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> /* version */) const override
    {
        this->data(place).set.write(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> /* version  */, Arena *) const override
    {
        this->data(place).set.read(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        assert_cast<ColumnUInt64 &>(to).getData().push_back(this->data(place).set.size());
    }
};


template <UInt8 K, typename HashValueType>
struct WithK
{
    template <typename T>
    using AggregateFunction = AggregateFunctionUniqCombined<T, K, HashValueType>;

    template <bool is_exact, bool argument_is_tuple>
    using AggregateFunctionVariadic = AggregateFunctionUniqCombinedVariadic<is_exact, argument_is_tuple, K, HashValueType>;
};

template <UInt8 K, typename HashValueType>
AggregateFunctionPtr createAggregateFunctionWithK(const DataTypes & argument_types, const Array & params)
{
    /// We use exact hash function if the arguments are not contiguous in memory, because only exact hash function has support for this case.
    bool use_exact_hash_function = !isAllArgumentsContiguousInMemory(argument_types);

    if (argument_types.size() == 1)
    {
        const IDataType & argument_type = *argument_types[0];

        AggregateFunctionPtr res(createWithNumericType<WithK<K, HashValueType>::template AggregateFunction>(*argument_types[0], argument_types, params));

        WhichDataType which(argument_type);
        if (res)
            return res;
        if (which.isDate())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<DataTypeDate::FieldType>>(
                argument_types, params);
        if (which.isDate32())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<DataTypeDate32::FieldType>>(
                argument_types, params);
        if (which.isDateTime())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<DataTypeDateTime::FieldType>>(
                argument_types, params);
        if (which.isStringOrFixedString())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<String>>(argument_types, params);
        if (which.isUUID())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<DataTypeUUID::FieldType>>(
                argument_types, params);
        if (which.isIPv4())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<DataTypeIPv4::FieldType>>(
                argument_types, params);
        if (which.isIPv6())
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunction<DataTypeIPv6::FieldType>>(
                argument_types, params);
        if (which.isTuple())
        {
            if (use_exact_hash_function)
                return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunctionVariadic<true, true>>(
                    argument_types, params);
            return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunctionVariadic<false, true>>(
                argument_types, params);
        }
    }

    /// "Variadic" method also works as a fallback generic case for a single argument.
    if (use_exact_hash_function)
        return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunctionVariadic<true, false>>(argument_types, params);
    return std::make_shared<typename WithK<K, HashValueType>::template AggregateFunctionVariadic<false, false>>(argument_types, params);
}

template <UInt8 K>
AggregateFunctionPtr createAggregateFunctionWithHashType(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params)
{
    if (use_64_bit_hash)
        return createAggregateFunctionWithK<K, UInt64>(argument_types, params);
    return createAggregateFunctionWithK<K, UInt32>(argument_types, params);
}

/// Let's instantiate these templates in separate translation units,
/// otherwise this translation unit becomes too large.
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<12>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<13>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<14>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<15>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<16>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<17>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<18>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<19>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);
extern template AggregateFunctionPtr createAggregateFunctionWithHashType<20>(bool use_64_bit_hash, const DataTypes & argument_types, const Array & params);

}
