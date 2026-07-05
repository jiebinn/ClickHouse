#include <gtest/gtest.h>

#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/IAggregateFunction.h>
#include <Columns/ColumnSparse.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnsNumber.h>
#include <Common/AlignedBuffer.h>
#include <Common/assert_cast.h>
#include <Common/tests/gtest_global_register.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>

using namespace DB;

/// The per-row `add` of the -Tuple combinator accepts element columns in sparse representation and
/// reads them through the dense values column at the translated index.
TEST(AggregateFunctionTupleCombinator, AddOverSparseElements)
{
    tryRegisterAggregateFunctions();

    constexpr size_t rows = 8;

    /// Element 0, sparse representation of [0, 0, 5, 0, 0, 0, 7, 9].
    auto values = ColumnInt64::create();
    values->getData().push_back(0); /// The shared default value at position 0.
    values->getData().push_back(5);
    values->getData().push_back(7);
    values->getData().push_back(9);
    auto offsets = ColumnUInt64::create();
    offsets->getData().push_back(2);
    offsets->getData().push_back(6);
    offsets->getData().push_back(7);
    ColumnPtr sparse_element = ColumnSparse::create(std::move(values), std::move(offsets), rows);

    /// Element 1, dense: [1.0] * 8.
    auto dense_element = ColumnFloat64::create();
    dense_element->getData().resize_fill(rows, 1.0);

    ColumnPtr tuple = ColumnTuple::create(Columns{sparse_element, std::move(dense_element)});

    DataTypes arguments{std::make_shared<DataTypeTuple>(
        DataTypes{std::make_shared<DataTypeInt64>(), std::make_shared<DataTypeFloat64>()})};
    AggregateFunctionProperties properties;
    AggregateFunctionPtr function
        = AggregateFunctionFactory::instance().get("sumTuple", NullsAction::EMPTY, arguments, {}, properties);

    AlignedBuffer place(function->sizeOfData(), function->alignOfData());
    function->create(place.data());

    const IColumn * column = tuple.get();
    for (size_t row = 0; row < rows; ++row)
        function->add(place.data(), &column, row, nullptr);

    auto result = function->getResultType()->createColumn();
    function->insertResultInto(place.data(), *result, nullptr);
    function->destroy(place.data());

    const auto & result_tuple = assert_cast<const ColumnTuple &>(*result);
    EXPECT_EQ(assert_cast<const ColumnInt64 &>(result_tuple.getColumn(0)).getElement(0), 21);
    EXPECT_EQ(assert_cast<const ColumnFloat64 &>(result_tuple.getColumn(1)).getElement(0), 8.0);
}
