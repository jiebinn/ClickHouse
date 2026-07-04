
#include <AggregateFunctions/Combinators/AggregateFunctionCombinatorFactory.h>
#include <AggregateFunctions/Combinators/AggregateFunctionTuple.h>

#include <AggregateFunctions/AggregateFunctionFactory.h>
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

AggregateFunctionTuple::NestedFunctionsAndResultType AggregateFunctionTuple::initNested(
    const AggregateFunctionPtr & representative_nested_func,
    const DataTypes & arguments,
    const Array & params)
{
    const String & base_name = representative_nested_func->getName();

    /// Every construction path goes through the combinator's `transformArguments`, which has
    /// already validated that there is exactly one non-empty `Tuple` argument.
    const auto & tuple_type = assert_cast<const DataTypeTuple &>(*arguments[0]);
    const auto & elem_types = tuple_type.getElements();

    auto & factory = AggregateFunctionFactory::instance();
    VectorWithMemoryTracking<AggregateFunctionPtr> functions;
    functions.resize(elem_types.size());
    DataTypes result_types;
    result_types.reserve(elem_types.size());

    /// When every tuple element is only-null (e.g. `Tuple(Nullable(Nothing))`), the factory has
    /// collapsed the representative function to `AggregateFunctionNothing`, whose name is a
    /// placeholder such as `nothingNull` rather than the original aggregate name. Re-resolving
    /// each element by that placeholder name would feed the original parameters to the `nothing*`
    /// creators, which reject parameters, turning otherwise valid parametric aggregates over
    /// `NULL` into an exception (e.g. `groupArrayMovingAvgTuple(2)(tuple(NULL))`). Since all
    /// elements are only-null and therefore resolve identically, reuse the already-created
    /// representative function for every element instead of re-resolving by name.
    bool all_only_null = true;
    for (const auto & type : elem_types)
    {
        if (!type->onlyNull())
        {
            all_only_null = false;
            break;
        }
    }

    for (size_t i = 0; i < elem_types.size(); ++i)
    {
        if (all_only_null)
        {
            functions[i] = representative_nested_func;
        }
        else
        {
            AggregateFunctionProperties props;
            DataTypes nested_arg_types = {elem_types[i]};
            auto action = NullsAction::EMPTY;
            functions[i] = factory.get(base_name, action, nested_arg_types, params, props);
        }
        result_types.push_back(functions[i]->getResultType());
    }

    DataTypePtr result_type;
    if (tuple_type.hasExplicitNames())
        result_type = std::make_shared<DataTypeTuple>(result_types, tuple_type.getElementNames());
    else
        result_type = std::make_shared<DataTypeTuple>(result_types);

    return {std::move(functions), std::move(result_type)};
}

AggregateFunctionTuple::AggregateFunctionTuple(
    const AggregateFunctionPtr & representative_nested_func,
    const DataTypes & arguments,
    const Array & params)
    : AggregateFunctionTuple(representative_nested_func->getName(), arguments, params,
        initNested(representative_nested_func, arguments, params))
{
}

AggregateFunctionTuple::AggregateFunctionTuple(
    const String & func_name,
    const DataTypes & arguments,
    const Array & params,
    NestedFunctionsAndResultType && nested_and_type)
    : IAggregateFunctionHelper<AggregateFunctionTuple>(arguments, params, nested_and_type.result_type)
    , nested_functions(std::move(nested_and_type.functions))
    , nested_func_name(func_name)
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

void AggregateFunctionTuple::addManyDefaults(AggregateDataPtr __restrict place, const IColumn ** columns, size_t length, Arena * arena) const
{
    /// Reached from the sparse aggregation path for long runs of default values. Materialize the
    /// outer tuple once and add its default row directly, instead of delegating to
    /// `IAggregateFunctionHelper::addManyDefaults`, which would route each of the `length` iterations
    /// back through `add` and rerun `recursiveRemoveSparse` (allocating a new `ColumnTuple`) per row.
    ColumnPtr materialized = recursiveRemoveSparse(columns[0]->getPtr());
    const auto & tuple_column = assert_cast<const ColumnTuple &>(*materialized);
    for (size_t i = 0; i < length; ++i)
        addRowFromMaterialized(place, tuple_column, 0, arena);
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
    /// `MergeTree` may store individual `Tuple` elements as `ColumnSparse` while the outer `ColumnTuple` is dense.
    /// Nested aggregate functions cast their column to its concrete type, so we materialize once per batch.
    /// Note: we call `addRowFromMaterialized` directly in the row loop instead of delegating to
    /// `IAggregateFunctionHelper::addBatch`, because that would route back through our `add` override
    /// and rerun `recursiveRemoveSparse` on every row.
    ColumnPtr materialized = recursiveRemoveSparse(columns[0]->getPtr());
    const auto & tuple_column = assert_cast<const ColumnTuple &>(*materialized);

    if (if_argument_pos >= 0)
    {
        const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if (flags[i] && places[i])
                addRowFromMaterialized(places[i] + place_offset, tuple_column, i, arena);
        }
    }
    else
    {
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if (places[i])
                addRowFromMaterialized(places[i] + place_offset, tuple_column, i, arena);
        }
    }
}

void AggregateFunctionTuple::addBatchSinglePlace( /// NOLINT
    size_t row_begin,
    size_t row_end,
    AggregateDataPtr __restrict place,
    const IColumn ** columns,
    Arena * arena,
    ssize_t if_argument_pos) const
{
    ColumnPtr materialized = recursiveRemoveSparse(columns[0]->getPtr());
    const auto & tuple_column = assert_cast<const ColumnTuple &>(*materialized);

    if (if_argument_pos >= 0)
    {
        const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if (flags[i])
                addRowFromMaterialized(place, tuple_column, i, arena);
        }
    }
    else
    {
        for (size_t i = row_begin; i < row_end; ++i)
            addRowFromMaterialized(place, tuple_column, i, arena);
    }
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
    /// Reached from `AggregateFunctionNullUnary::addBatchSinglePlace` for `Nullable(Tuple(...))` inputs.
    /// Materialize the outer tuple once per batch instead of letting the base implementation route
    /// each surviving row back through `add`, which would rerun `recursiveRemoveSparse` per row.
    ColumnPtr materialized = recursiveRemoveSparse(columns[0]->getPtr());
    const auto & tuple_column = assert_cast<const ColumnTuple &>(*materialized);

    if (if_argument_pos >= 0)
    {
        const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if (!null_map[i] && flags[i])
                addRowFromMaterialized(place, tuple_column, i, arena);
        }
    }
    else
    {
        for (size_t i = row_begin; i < row_end; ++i)
        {
            if (!null_map[i])
                addRowFromMaterialized(place, tuple_column, i, arena);
        }
    }
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

bool AggregateFunctionTuple::isState() const
{
    for (const auto & func : nested_functions)
        if (func->isState())
            return true;
    return false;
}

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
    AggregateFunctionPtr normalized_function(new AggregateFunctionTuple(
        normalized_nested_name, argument_types, Array{}, {std::move(normalized_nested_functions), getResultType()}));
    return std::make_shared<DataTypeAggregateFunction>(std::move(normalized_function), nested_normalized_state_types, Array{});
}

namespace
{

class AggregateFunctionCombinatorTuple final : public IAggregateFunctionCombinator
{
public:
    String getName() const override { return "Tuple"; }

    bool transformsArgumentTypes() const override { return true; }

    DataTypes transformArguments(const DataTypes & arguments) const override
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

        /// Return a representative element type as a placeholder so that the factory can resolve the nested function name.
        /// Prefer the first non-only-null element: if the first element happens to be `Nullable(Nothing)`, the recursive
        /// `get` would wrap with the `Null` combinator and collapse to `AggregateFunctionNothing*`, which would then
        /// be used as the base function name and force every per-element nested function to also collapse to Nothing.
        /// The actual per-element functions are still created inside AggregateFunctionTuple based on real elem_types.
        for (const auto & type : elem_types)
            if (!type->onlyNull())
                return DataTypes({type});

        return DataTypes({elem_types[0]});
    }

    AggregateFunctionPtr transformAggregateFunction(
        const AggregateFunctionPtr & nested_function,
        const AggregateFunctionProperties &,
        const DataTypes & arguments,
        const Array & params) const override
    {
        return std::make_shared<AggregateFunctionTuple>(nested_function, arguments, params);
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
