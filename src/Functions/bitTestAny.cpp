#include <Functions/FunctionFactory.h>
#include <Functions/FunctionBitTestMany.h>

namespace DB
{
namespace
{

struct BitTestAnyImpl
{
    template <typename A, typename B>
    static UInt8 apply(A a, B b) { return (a & b) != 0; }
};

struct NameBitTestAny { static constexpr auto name = "bitTestAny"; };
using FunctionBitTestAny = FunctionBitTestMany<BitTestAnyImpl, NameBitTestAny>;

}

REGISTER_FUNCTION(BitTestAny)
{
    factory.registerFunction<FunctionBitTestAny>();
}

}
