
#include <AggregateFunctions/Combinators/AggregateFunctionCombinatorFactory.h>
#include <AggregateFunctions/Combinators/AggregateFunctionTuple.h>

#include <Columns/ColumnSparse.h>
#include <Columns/ColumnsNumber.h>
#include <Common/Arena.h>
#include <Common/memory.h>
#include <Common/typeid_cast.h>
#include <DataTypes/DataTypeAggregateFunction.h>
#include <IO/ReadBuffer.h>
#include <IO/WriteBuffer.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

DataTypePtr AggregateFunctionTuple::deriveResultType(
    const VectorWithMemoryTracking<AggregateFunctionPtr> & nested_functions,
    const DataTypes & arguments)
{
    /// Every construction path goes through the combinator's `transformArgumentsForMultipleNestedFunctions`, which has
    /// already validated that there is exactly one non-empty `Tuple` argument.
    const auto & tuple_type = assert_cast<const DataTypeTuple &>(*arguments[0]);

    DataTypes result_types;
    result_types.reserve(nested_functions.size());
    for (const auto & nested_function : nested_functions)
        result_types.push_back(nested_function->getResultType());

    if (tuple_type.hasExplicitNames())
        return std::make_shared<DataTypeTuple>(result_types, tuple_type.getElementNames());
    return std::make_shared<DataTypeTuple>(result_types);
}

AggregateFunctionTuple::AggregateFunctionTuple(
    const String & nested_name,
    VectorWithMemoryTracking<AggregateFunctionPtr> nested_functions_,
    const DataTypes & arguments,
    const Array & params)
    : IAggregateFunctionHelper<AggregateFunctionTuple>(arguments, params, deriveResultType(nested_functions_, arguments))
    , nested_functions(std::move(nested_functions_))
    , nested_func_name(nested_name)
{
    state_offsets.resize(nested_functions.size());

    size_t offset = 0;
    for (size_t i = 0; i < nested_functions.size(); ++i)
    {
        size_t align = nested_functions[i]->alignOfData();
        max_state_align = std::max(max_state_align, align);
        offset = ::Memory::alignUp(offset, align);
        state_offsets[i] = offset;
        offset += nested_functions[i]->sizeOfData();
    }
    total_state_size = ::Memory::alignUp(offset, max_state_align);
}

bool AggregateFunctionTuple::isVersioned() const
{
    for (const auto & func : nested_functions)
        if (func->isVersioned())
            return true;
    return false;
}

/// All nested functions share the same base aggregate function, so they agree on one versioning
/// scheme and a single version number serves the whole tuple state. Placeholder functions for
/// only-null elements ignore the version entirely.
size_t AggregateFunctionTuple::getDefaultVersion() const
{
    size_t version = 0;
    for (const auto & func : nested_functions)
        version = std::max(version, func->getDefaultVersion());
    return version;
}

size_t AggregateFunctionTuple::getVersionFromRevision(size_t revision) const
{
    size_t version = 0;
    for (const auto & func : nested_functions)
        version = std::max(version, func->getVersionFromRevision(revision));
    return version;
}

void AggregateFunctionTuple::create(AggregateDataPtr __restrict place) const
{
    size_t i = 0;
    try
    {
        for (; i < nested_functions.size(); ++i)
            nested_functions[i]->create(place + state_offsets[i]);
    }
    catch (...)
    {
        for (size_t j = 0; j < i; ++j)
            nested_functions[j]->destroy(place + state_offsets[j]);
        throw;
    }
}

void AggregateFunctionTuple::destroy(AggregateDataPtr __restrict place) const noexcept
{
    for (size_t i = 0; i < nested_functions.size(); ++i)
        nested_functions[i]->destroy(place + state_offsets[i]);
}

void AggregateFunctionTuple::destroyUpToState(AggregateDataPtr __restrict place) const noexcept
{
    for (size_t i = 0; i < nested_functions.size(); ++i)
        nested_functions[i]->destroyUpToState(place + state_offsets[i]);
}

bool AggregateFunctionTuple::hasTrivialDestructor() const
{
    for (const auto & func : nested_functions)
        if (!func->hasTrivialDestructor())
            return false;
    return true;
}

void AggregateFunctionTuple::add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena * arena) const
{
    /// Per-row fallback path: materialize sparse children defensively so callers that bypass
    /// our batch overrides (e.g. direct `add` calls in tests or future code paths) stay correct.
    /// The hot paths (addBatch/addBatchSinglePlace) materialize once up front and use
    /// addRowFromMaterialized to avoid paying this cost per row.
    ColumnPtr materialized = recursiveRemoveSparse(columns[0]->getPtr());
    const auto & tuple_column = assert_cast<const ColumnTuple &>(*materialized);
    addRowFromMaterialized(place, tuple_column, row_num, arena);
}

template <bool has_null_map, typename GetPlace>
void AggregateFunctionTuple::addBatchImpl(
    size_t row_begin,
    size_t row_end,
    const IColumn ** columns,
    const UInt8 * null_map,
    ssize_t if_argument_pos,
    Arena * arena,
    GetPlace && get_place) const
{
    /// `MergeTree` may store individual `Tuple` elements as `ColumnSparse` while the outer `ColumnTuple`
    /// is dense. Nested aggregate functions cast their column to its concrete type, so the tuple is
    /// materialized once per batch; delegating to the `IAggregateFunctionHelper` batch loops would
    /// instead route every row through the `add` override and rerun `recursiveRemoveSparse` per row.
    ColumnPtr materialized = recursiveRemoveSparse(columns[0]->getPtr());
    const auto & tuple_column = assert_cast<const ColumnTuple &>(*materialized);

    ColumnRawPtrs element_columns(nested_functions.size());
    for (size_t i = 0; i < nested_functions.size(); ++i)
        element_columns[i] = &tuple_column.getColumn(i);

    auto add_row = [&](AggregateDataPtr place, size_t row)
    {
        for (size_t i = 0; i < nested_functions.size(); ++i)
            nested_functions[i]->add(place + state_offsets[i], &element_columns[i], row, arena);
    };

    if (if_argument_pos >= 0)
    {
        const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if (!flags[i])
                continue;
            if constexpr (has_null_map)
            {
                if (null_map[i])
                    continue;
            }
            if (AggregateDataPtr place = get_place(i))
                add_row(place, i);
        }
    }
    else
    {
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if constexpr (has_null_map)
            {
                if (null_map[i])
                    continue;
            }
            if (AggregateDataPtr place = get_place(i))
                add_row(place, i);
        }
    }
}

void AggregateFunctionTuple::addBatch( /// NOLINT
    size_t row_begin,
    size_t row_end,
    AggregateDataPtr * places,
    size_t place_offset,
    const IColumn ** columns,
    Arena * arena,
    ssize_t if_argument_pos) const
{
    addBatchImpl<false>(row_begin, row_end, columns, nullptr, if_argument_pos, arena,
        [&](size_t i) { return places[i] ? places[i] + place_offset : nullptr; });
}

void AggregateFunctionTuple::addBatchSinglePlace( /// NOLINT
    size_t row_begin,
    size_t row_end,
    AggregateDataPtr __restrict place,
    const IColumn ** columns,
    Arena * arena,
    ssize_t if_argument_pos) const
{
    addBatchImpl<false>(row_begin, row_end, columns, nullptr, if_argument_pos, arena,
        [&](size_t) { return place; });
}

void AggregateFunctionTuple::addBatchSinglePlaceNotNull( /// NOLINT
    size_t row_begin,
    size_t row_end,
    AggregateDataPtr __restrict place,
    const IColumn ** columns,
    const UInt8 * null_map,
    Arena * arena,
    ssize_t if_argument_pos) const
{
    /// Reached from `AggregateFunctionNullUnary` for `Nullable(Tuple(...))` inputs: rows whose tuple
    /// is NULL are skipped via the null map.
    addBatchImpl<true>(row_begin, row_end, columns, null_map, if_argument_pos, arena,
        [&](size_t) { return place; });
}

void AggregateFunctionTuple::mergeImpl(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena * arena) const
{
    for (size_t i = 0; i < nested_functions.size(); ++i)
        nested_functions[i]->merge(place + state_offsets[i], rhs + state_offsets[i], arena);
}

void AggregateFunctionTuple::serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> version) const
{
    for (size_t i = 0; i < nested_functions.size(); ++i)
        nested_functions[i]->serialize(place + state_offsets[i], buf, version);
}

void AggregateFunctionTuple::deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> version, Arena * arena) const
{
    for (size_t i = 0; i < nested_functions.size(); ++i)
        nested_functions[i]->deserialize(place + state_offsets[i], buf, version, arena);
}

void AggregateFunctionTuple::insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena * arena) const
{
    insertResultIntoImpl<false>(place, to, arena);
}

void AggregateFunctionTuple::insertMergeResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena * arena) const
{
    insertResultIntoImpl<true>(place, to, arena);
}

bool AggregateFunctionTuple::allocatesMemoryInArena() const
{
    for (const auto & func : nested_functions)
        if (func->allocatesMemoryInArena())
            return true;
    return false;
}

/// The tuple result contains nested aggregation states if any element produces one, so the whole
/// result requires state lifetime handling as soon as a single nested function is state-producing.
bool AggregateFunctionTuple::isState() const
{
    for (const auto & func : nested_functions)
        if (func->isState())
            return true;
    return false;
}

/// Both `haveSameStateRepresentationImpl` and `getNormalizedStateType` define state compatibility as
/// the element-wise composition of the same-named concept of the nested functions, plus equal arity.
/// They must agree so that equality of normalized types implies `haveSameStateRepresentation`; every
/// nested function guarantees that implication for itself, and the composition preserves it.
bool AggregateFunctionTuple::haveSameStateRepresentationImpl(const IAggregateFunction & rhs) const
{
    const auto * rhs_tuple = typeid_cast<const AggregateFunctionTuple *>(&rhs);
    if (!rhs_tuple)
        return false;
    if (nested_functions.size() != rhs_tuple->nested_functions.size())
        return false;
    for (size_t i = 0; i < nested_functions.size(); ++i)
        if (!nested_functions[i]->haveSameStateRepresentation(*rhs_tuple->nested_functions[i]))
            return false;
    return true;
}

DataTypePtr AggregateFunctionTuple::getNormalizedStateType() const
{
    /// Unlike `-Array` / `-If`, the `-Tuple` state is a concatenation of per-element nested states,
    /// so we cannot delegate to a single nested function. Two tuple states are interchangeable
    /// exactly when their nested states are interchangeable pairwise, and the normalized type
    /// encodes that: its argument types are the nested normalized state types (compared recursively
    /// by `DataTypeAggregateFunction::equals`), and its function is a `-Tuple` wrapper rebuilt
    /// around the normalized nested functions, so that its name uses the canonical nested spelling.
    /// This unifies spellings with identical states (`quantileExactTuple` and `quantilesExactTuple`
    /// both normalize to a `quantilesExact`-based name), states that do not depend on the argument
    /// types (`countTuple` normalizes to `count` elements regardless of the tuple element types),
    /// and placeholder elements for only-null types. The parameters are dropped: every parameter
    /// that affects a state representation is already part of the corresponding nested normalized
    /// state type.
    DataTypes nested_normalized_state_types;
    nested_normalized_state_types.reserve(nested_functions.size());
    VectorWithMemoryTracking<AggregateFunctionPtr> normalized_nested_functions;
    normalized_nested_functions.reserve(nested_functions.size());
    for (const auto & nested_function : nested_functions)
    {
        auto normalized_state_type = nested_function->getNormalizedStateType();
        normalized_nested_functions.push_back(assert_cast<const DataTypeAggregateFunction &>(*normalized_state_type).getFunction());
        nested_normalized_state_types.push_back(std::move(normalized_state_type));
    }

    String normalized_nested_name = normalized_nested_functions.front()->getName();
    auto normalized_function = std::make_shared<AggregateFunctionTuple>(
        normalized_nested_name, std::move(normalized_nested_functions), argument_types, Array{});
    return std::make_shared<DataTypeAggregateFunction>(std::move(normalized_function), nested_normalized_state_types, Array{});
}

namespace
{

class AggregateFunctionCombinatorTuple final : public IAggregateFunctionCombinator
{
public:
    String getName() const override { return "Tuple"; }

    bool transformsArgumentTypes() const override { return true; }

    bool transformsMultipleNestedFunctions() const override { return true; }

    VectorWithMemoryTracking<DataTypes> transformArgumentsForMultipleNestedFunctions(const DataTypes & arguments) const override
    {
        if (arguments.size() != 1)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                "Aggregate function with {} suffix requires exactly one Tuple argument", getName());

        const auto * tuple_type = typeid_cast<const DataTypeTuple *>(arguments[0].get());
        if (!tuple_type)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument for aggregate function with {} suffix. Must be Tuple.",
                arguments[0]->getName(), getName());

        const auto & elem_types = tuple_type->getElements();
        if (elem_types.empty())
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Tuple must not be empty for aggregate function with {} suffix", getName());

        VectorWithMemoryTracking<DataTypes> nested_arguments_list;
        nested_arguments_list.reserve(elem_types.size());
        for (const auto & elem_type : elem_types)
            nested_arguments_list.push_back(DataTypes{elem_type});
        return nested_arguments_list;
    }

    AggregateFunctionPtr transformAggregateFunctionFromMultipleNestedFunctions(
        const String & nested_name,
        VectorWithMemoryTracking<AggregateFunctionPtr> nested_functions,
        const AggregateFunctionProperties &,
        const DataTypes & arguments,
        const Array & params) const override
    {
        return std::make_shared<AggregateFunctionTuple>(nested_name, std::move(nested_functions), arguments, params);
    }
};

}

void registerAggregateFunctionCombinatorTuple(AggregateFunctionCombinatorFactory & factory);
void registerAggregateFunctionCombinatorTuple(AggregateFunctionCombinatorFactory & factory)
{
    factory.registerCombinator(std::make_shared<AggregateFunctionCombinatorTuple>(), Documentation{
        .description = "Applied as a suffix to an aggregate function name (e.g. `sumTuple`), it makes the function take a single `Tuple` argument and aggregate each tuple element independently, producing a tuple of per-element results.",
        .syntax = "<aggregate_function>Tuple",
        .related = {"Array", "ForEach", "Map"}});
}

}
