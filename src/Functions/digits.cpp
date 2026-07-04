#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <Core/TypeId.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/IDataType.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/IFunction.h>
#include <Functions/castTypeToEither.h>
#include <Common/Exception.h>
#include <Common/digits10.h>
#include <Common/intExp10.h>

#include <limits>

namespace DB
{

namespace ErrorCodes
{
extern const int ZERO_ARRAY_OR_TUPLE_INDEX;
extern const int ILLEGAL_COLUMN;
}

namespace
{

UInt64 extractDigits(UInt64 num, Int64 offset, Int64 length, bool has_length)
{
    if (offset == 0)
        throw Exception(ErrorCodes::ZERO_ARRAY_OR_TUPLE_INDEX, "Indices in number are 1-based");
    if (has_length && length == 0) // No digits to return
        return 0ULL;

    const Int64 total_digits = common::digits10(num);

    if (offset < 0)
        offset = total_digits + offset + 1; // Index from the left

    if (offset > total_digits) // Index is greater than the right boundary
        return 0ULL;

    Int64 count = 0; // Number of digits to take from offset inclusive
    if (!has_length)
    {
        offset = std::max<Int64>(offset, 1);
        count = total_digits - offset + 1;
    }
    else
    {
        if (length < 0)
        {
            const Int64 end = total_digits + length; // negative length: absolute end position
            offset = std::max<Int64>(offset, 1);
            count = end - offset + 1;
        }
        else
        {
            // Length consumed by the off-edge positions left of index 1 that the window covers.
            // `1 - offset` cannot overflow: common::digits10 >= 1 guarantees offset >= INT64_MIN + 2 at this point.
            Int64 required = (offset <= 0 ? 1 - offset : 0);
            length = length - required;
            offset = std::max<Int64>(offset, 1);
            count = std::min<Int64>(length, total_digits - offset + 1);
        }
    }
    if (count <= 0)
        return 0ULL;
    const Int64 suffix = total_digits - (offset + count - 1); // Suffix to remove
    const UInt64 shifted = num / intExp10(static_cast<int>(suffix)); // Okay to cast suffix to int because suffix range is [0, 20)
    return count >= 20 ? shifted : shifted % intExp10(static_cast<int>(count)); // Okay to cast count to int because count range is [1, 20]
}

class FunctionDigits final : public IFunction
{
public:
    static constexpr auto name = "digits";

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionDigits>(); }

    String getName() const override { return name; }
    bool isVariadic() const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }
    bool useDefaultImplementationForConstants() const override { return true; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }
    bool isDeterministic() const override { return true; }
    bool canBeExecutedOnDefaultArguments() const override { return false; }
    bool useDefaultImplementationForNulls() const override { return false; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        const bool has_nullable = anyArgumentNullable(arguments);

        ColumnsWithTypeAndName args_without_nullable = arguments;
        for (auto & arg : args_without_nullable)
            arg.type = removeNullable(arg.type);

        FunctionArgumentDescriptors mandatory_args{
            {"number",
             static_cast<FunctionArgumentDescriptor::TypeValidator>(&isNativeInteger),
             nullptr,
             "Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64"},
            {"offset",
             static_cast<FunctionArgumentDescriptor::TypeValidator>(&isNativeInteger),
             nullptr,
             "Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64"}};
        FunctionArgumentDescriptors optional_args{
            {"length",
             static_cast<FunctionArgumentDescriptor::TypeValidator>(&isNativeInteger),
             nullptr,
             "Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64"}};
        validateFunctionArguments(*this, args_without_nullable, mandatory_args, optional_args);

        DataTypePtr result = std::make_shared<DataTypeUInt64>();
        if (has_nullable)
            return makeNullable(result);
        return result;
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {
        auto combined_null_map = ColumnUInt8::create(input_rows_count, static_cast<UInt8>(0));
        auto & null_map_data = combined_null_map->getData();
        bool any_nullable = false;

        auto combineNullMap = [&](ColumnPtr & col)
        {
            if (const auto * n = checkAndGetColumn<ColumnNullable>(col.get()))
            {
                any_nullable = true;
                const auto & src = n->getNullMapData();
                for (size_t i = 0; i < input_rows_count; ++i)
                    null_map_data[i] |= src[i];
                col = n->getNestedColumnPtr();
            }
        };
        ColumnPtr number_column = arguments[0].column->convertToFullColumnIfConst();
        combineNullMap(number_column);
        ColumnPtr offset_column = arguments[1].column->convertToFullColumnIfConst();
        combineNullMap(offset_column);
        ColumnPtr length_column = nullptr;
        bool has_length = false;
        if (arguments.size() == 3)
        {
            has_length = true;
            length_column = arguments[2].column->convertToFullColumnIfConst();
            combineNullMap(length_column);
        }
        auto result = ColumnUInt64::create(input_rows_count);
        auto & result_data = result->getData();

        const bool is_offset_uint64 = (offset_column->getDataType() == TypeIndex::UInt64);
        const bool is_length_uint64 = (has_length && length_column->getDataType() == TypeIndex::UInt64);

        if (!castTypeToEither<ColumnInt8, ColumnInt16, ColumnInt32, ColumnInt64, ColumnUInt8, ColumnUInt16, ColumnUInt32, ColumnUInt64>(
                number_column.get(),
                [&](const auto & col)
                {
                    const auto & data = col.getData();
                    using T = typename std::decay_t<decltype(col)>::ValueType;

                    for (size_t i = 0; i < input_rows_count; ++i)
                    {
                        if (any_nullable && null_map_data[i])
                        {
                            result_data[i] = 0;
                            continue;
                        }
                        T num = data[i];
                        UInt64 magnitude = 0;
                        if constexpr (std::is_signed_v<T>)
                            magnitude = num < 0 ? -static_cast<UInt64>(num) : static_cast<UInt64>(num);
                        else
                            magnitude = static_cast<UInt64>(num);
                        // If type is UInt64 and it exceeds max Int64 value, set it to max Int64 val
                        // as maximum number of digits can only be 20 and it would not affect result
                        Int64 offset = (is_offset_uint64 ? getMaxSignedValFromUnsigned(offset_column, i) : offset_column->getInt(i));
                        Int64 length = 0;
                        if (has_length)
                            length = (is_length_uint64 ? getMaxSignedValFromUnsigned(length_column, i) : length_column->getInt(i));
                        result_data[i] = extractDigits(magnitude, offset, length, has_length);
                    }
                    return true;
                }))
            throw Exception(
                ErrorCodes::ILLEGAL_COLUMN, "Illegal column {} of argument of function {}", number_column->getName(), getName());

        if (any_nullable)
            return ColumnNullable::create(std::move(result), std::move(combined_null_map));
        return result;
    }

    static bool anyArgumentNullable(const ColumnsWithTypeAndName & arguments)
    {
        bool has_nullable = false;
        for (const auto & arg : arguments)
            has_nullable |= arg.type->isNullable();
        return has_nullable;
    }

    static Int64 getMaxSignedValFromUnsigned(const ColumnPtr & col, size_t index)
    {
        if (col->getUInt(index) > std::numeric_limits<Int64>::max())
            return std::numeric_limits<Int64>::max();
        return col->getInt(index);
    }
};

}
REGISTER_FUNCTION(Digits)
{
    FunctionDocumentation::Description description = R"(
Returns the digits of a number `n` which starts at the specified index `offset`.
Counting starts from 1 with the following logic:
- If `offset` is `0`, an exception is thrown, as `offset` is 1-based.
- If `offset` is negative, counting starts `offset` digits from the end of the number, rather than from the beginning.
- If `offset` is greater than the number of digits in `n`, `0` is returned.

An optional argument `length` uses the following logic:
- If `length` is positive, it means number of digits to take from offset
- If `length` is negative, it means number of digits from the right of the number to exclude
    )";
    FunctionDocumentation::Syntax syntax = "digits(n, offset[, length])";
    FunctionDocumentation::Arguments arguments
        = {{"n", "The number to calculate digits from.", {"(U)Int8", "(U)Int16", "(U)Int32", "(U)Int64"}},
           {"offset", "The starting position of the digit in `n`.", {"(U)Int8", "(U)Int16", "(U)Int32", "(U)Int64"}},
           {"length", "Optional. The maximum length of the digits.", {"(U)Int8", "(U)Int16", "(U)Int32", "(U)Int64"}}};
    FunctionDocumentation::ReturnedValue returned_value
        = {"The selected digits of `n`, interpreted as a `UInt64`. Returns `0` if the selected range is empty. Leading zeros are not "
           "preserved.",
           {"UInt64"}};
    FunctionDocumentation::Examples examples
        = {{"Positive offset", "SELECT digits(1234567890, 7)", "7890"},
           {"Positive offset and length", "SELECT digits(1234567890, 7, 2)", "78"},
           {"Negative offset counts from the right", "SELECT digits(1234567890, -3)", "890"},
           {"Negative length excludes digits from the right", "SELECT digits(1234567890, 3, -2)", "345678"},
           {"Offset past the end returns 0", "SELECT digits(1234567890, 11)", "0"}};
    FunctionDocumentation::IntroducedIn introduced_in = {26, 7};
    FunctionDocumentation::Category category = FunctionDocumentation::Category::Other;
    FunctionDocumentation documentation = {description, syntax, arguments, {}, returned_value, examples, introduced_in, category};

    factory.registerFunction<FunctionDigits>(documentation);
}

}
