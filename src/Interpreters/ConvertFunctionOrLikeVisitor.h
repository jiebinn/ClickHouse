#pragma once

#include <Interpreters/Context_fwd.h>
#include <Interpreters/InDepthNodeVisitor.h>
#include <Parsers/IAST_fwd.h>
#include <DataTypes/IDataType_fwd.h>

#include <string>
#include <unordered_map>

namespace DB
{

class ASTFunction;

/// Replaces chains of OR with `{i}like`/`match` by `multiSearchAny`/`multiMatchAny`.
///
/// For example:
///   x LIKE '%foo%' OR x LIKE '%bar%' --> multiSearchAny(x, ['foo', 'bar'])
///   x LIKE 'foo%' OR x LIKE '%bar' --> multiMatchAny(x, ['^foo', 'bar$'])  (with Vectorscan)
///   x LIKE '%foo%' OR match(x, 'bar.*') --> multiMatchAny(x, ['foo', 'bar.*'])
///
/// If all patterns are simple substring searches (`%substring%`) with the same case sensitivity,
/// the rewrite uses the faster `multiSearchAny`/`multiSearchAnyCaseInsensitiveUTF8`. Otherwise it
/// uses `multiMatchAny` (Vectorscan/Hyperscan) when ClickHouse is built with Vectorscan,
/// `allow_hyperscan` is on, and the patterns are eligible. When neither fast path applies, the
/// original `OR` chain is kept unchanged: a combined `match('(p1)|(p2)|...')` alternation over RE2
/// is consistently slower than the original short-circuit `OR`, so it is never emitted.
///
/// The two rewrite targets have per-target minimum branch counts, calibrated on `hits`:
/// `multiSearchAny` from `optimize_or_like_chain_min_substrings` branches, `multiMatchAny` from
/// `optimize_or_like_chain_min_patterns` branches. Shorter chains are left as-is to avoid
/// regressing queries where the rewrite overhead exceeds the original short-circuit OR-chain.
class ConvertFunctionOrLikeData
{
public:
    using TypeToVisit = ASTFunction;

    bool allow_hyperscan = true;
    size_t max_hyperscan_regexp_length = 0;
    size_t max_hyperscan_regexp_total_length = 0;
    bool reject_expensive_hyperscan_regexps = true;
    size_t min_patterns_for_rewrite = 0;
    size_t min_substrings_for_rewrite = 0;
    ContextPtr context;

    /// Map from source-column name to its type, used to gate the `multiSearchAny*` / `multiMatchAny`
    /// rewrite targets to a `String` haystack (they reject `FixedString`/`Enum`, which the original
    /// `like`/`ilike`/`match` predicates accept). A column whose type is unknown here (e.g. the LHS
    /// is an expression rather than a plain column) is treated as non-`String`, so the rewrite
    /// conservatively keeps the original branches (which accept those haystacks) instead.
    std::unordered_map<std::string, DataTypePtr> source_column_types;

    void visit(ASTFunction & function, ASTPtr & ast) const;
};

using ConvertFunctionOrLikeVisitor = InDepthNodeVisitor<OneTypeMatcher<ConvertFunctionOrLikeData>, true>;

}
