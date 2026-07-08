#pragma once

#include <algorithm>
#include <string_view>

namespace DB
{

/// Comparator for HTTP field (header) names.
/// Per RFC 9110 §5.1, field names are case-insensitive tokens composed of ASCII characters.
/// Only letters A-Z are folded; punctuation and other valid token characters are left intact,
/// so distinct names such as `Foo^Bar` and `Foo~Bar` remain distinct.
/// `is_transparent` enables heterogeneous lookup (find/contains with std::string_view).
struct HTTPFieldLess
{
    using is_transparent = void;

    bool operator()(std::string_view a, std::string_view b) const
    {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](unsigned char x, unsigned char y)
            {
                auto lower = [](unsigned char c) -> unsigned char { return (c >= 'A' && c <= 'Z') ? c | 0x20 : c; };
                return lower(x) < lower(y);
            });
    }
};

}
