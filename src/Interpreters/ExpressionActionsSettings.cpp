#include <Interpreters/ExpressionActionsSettings.h>

#include <Core/Settings.h>
#include <Interpreters/Context.h>

namespace DB
{
namespace Setting
{
    extern const SettingsBool compile_expressions;
    extern const SettingsUInt64 max_temporary_columns;
    extern const SettingsUInt64 max_temporary_non_const_columns;
    extern const SettingsUInt64 min_count_to_compile_expression;
    extern const SettingsShortCircuitFunctionEvaluation short_circuit_function_evaluation;
}

ExpressionActionsSettings ExpressionActionsSettings::fromSettings(const Settings & from, CompileExpressions compile_expressions)
{
    ExpressionActionsSettings settings;
    settings.can_compile_expressions = from[Setting::compile_expressions];
    settings.min_count_to_compile_expression = from[Setting::min_count_to_compile_expression];
    settings.max_temporary_columns = from[Setting::max_temporary_columns];
    settings.max_temporary_non_const_columns = from[Setting::max_temporary_non_const_columns];
    settings.compile_expressions = compile_expressions;
    settings.short_circuit_function_evaluation = from[Setting::short_circuit_function_evaluation];

    return settings;
}

ExpressionActionsSettings ExpressionActionsSettings::fromContext(ContextPtr from, CompileExpressions compile_expressions)
{
    return fromSettings(from->getSettingsRef(), compile_expressions);
}

}
