#!/usr/bin/env bash
# Tags: no-fasttest
# no-fasttest: Parquet format is not available in fasttest builds

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

# External-writer Parquet files (pyarrow) whose LIST/MAP wrapper is OPTIONAL but the inner
# element/value group is REQUIRED. ClickHouse's own writer only emits REQUIRED list wrappers,
# so these schemas can only come from external files; embedded here as gzip+base64.
#
# The optional wrapper's nulls are normalized to empty collections by the reader and never reach
# the inner tuple null-map, so an always-defined REQUIRED element/value read as Nullable(Tuple)
# is lossless and must be accepted (issue #109605 follow-up). A genuinely OPTIONAL inner group
# carries real struct-level nulls and must still be rejected.

f1="$CLICKHOUSE_TMP/04065_optlist_reqelem.parquet"
f2="$CLICKHOUSE_TMP/04065_optstruct_under_list.parquet"
f3="$CLICKHOUSE_TMP/04065_map_reqval.parquet"

echo "H4sICNHzTGoAA2JvdF9vcHRsaXN0X3JlcWVsZW0ucGFycXVldAB9UslKA0EQrWmb0YOHROihG+aQgw5K4hIxHiQBa7KY4JqgBsVLDMMoZDNx+y5P4tdZ1Zm4HSyo6q7lVb1q+gxbeSWVVv6RchUsASzmHAAQpHOkClRGLecolyJ1fS05qiXXeLD6w0unyaSKFijYukLMLYOS5qYA2p1076J+RwkoCO3QqdyjIhUVpJa9+8kjJ0DPR72oHw3YU2IFtPMKnjS+8QPwlTAFcFOm7nQsYlbrvCrhud6b4304wX6w8D9Fk6NtgfsDbSSguBZIYXaJLVdTExksUCMwvl7EVuu0vTclrt/FJksWWZo4lUPEFwxjrCHGWI5tLAwRj/lSRmzYwPdJUmVzkdwuGEf4OvexfcOkM1aPaxYUf4Hs1Gne2iv2T86ZVZMNxyotSy3+M29ofR5xcDm5bud7twM7qpqsQk6T+ZJ0sZKAqzMq2LBF9V+rJ9e6HdVg/mVbEs+yoV0aSyXQmVFn/PAUPa53R6P1zng8fMk8R+PJ/XCQ2d7Z2NrYMr5Pz5+lX3NG//ETimdvwJQCAAA=" | base64 --decode | gunzip > "$f1"
echo "H4sICNHzTGoAA29wdHN0cnVjdF91bmRlcl9saXN0LnBhcnF1ZXQAfVJNT9tAEB1vXJNDD6HqRruSDzm0VisBDQhLCMJhNnxFgBpSRS29pZFlIiXGOLTOb+LI7+JHdGYX7KoHRvLuzLw383a8O8TRtvRlS76/oA3eATRDDwAa9EmQHflhQwYEBzIIlc9Z5TPeFp/+idbXaWn1BBcGnG4cgNcQ0tfzGFSwnN4ki4kUEAvl0S6Dix6RYl/589ny3gFryTxZJJmNQL2ZZVlSkC/FR1DeigR1qMMIQil0DEFL//AmtvqlzlV4KynaQfvBaz960V7UfP3QeoOHZhWgGQX0PkfC1ycAwmc6dRFRkzqBDtVbHI2+ft93s6gn8YWtTA0iXqGzc8QSTYoniCn2U5szRLhkp484sIl6Jzuulp0h9+Nu5ZjjU4tfVdRBxTc37GXMXyGLEXSNpnQ6x/V5zNHIutf/6TmctU638594NOZRui/62B3XerX+uY0ZMvmvRTyfpg439RBjnpvszDV34LP1LcmU1SGwnmf3+f9ccj1RBhX8za6HoDr5pLj7ndxvTvN8c1IUt2XnT1IsZ7dZZ2d3q7vV1WFIV1bQExzSk/4Lvx/0DdcCAAA=" | base64 --decode | gunzip > "$f2"
echo "H4sICCD0TGoAA21hcF9yZXF2YWwucGFycXVldAB9U91r01AUP/cklgoFV/GGGwhShIWJ3VzLOjpR8KbdWrHDNtIN9iJpCZk03bruo1v/Ch999NEHH3z0T/DRP8l7brKmOvQ83OSe38f5IOlKv8JNLrjTUQ94CFAoIwCMKvqscuCP+ZOywlZ4juecbVjDUVXgqFIsKsKDOtEMpNNEZqAmPrr1yjtMAQT+7SNMygqTcAvWlm7/8rXjGojc+fA4HAccoYaCjTly7JQVp2aK+6Pw5sNVEF+G3AReWAVhqMwqdBzCQdxLMASOCmPXYKHt2GUXHF6wa5BbsetsnJmQmKNlWp+Y9Zm5r938n7PbNAuQHaiREF5aedtFtLdVw6og2WJi2162Tbpg19r6K7O+M/cXc7+w/y/kTrGnS6WsH2ihm7d+ItiOKEjff3f4ItmT+GY8p5hHnpSyJ5N4K+VMepHckzKSjUjnPEXYp5eGlG90Inuq2KWjn771Saf0bfLRvl7qLHf9PS2KFiJdNcH1OaB7c66a2plRa5Rr+pofZfWodEPX135ySEg8OPFvgsODuWyR/lmU6Xu3pDv9egS1j44H7YOY7uOuEtVJX9fjaqY3y/pd6L2tdB/7et5Zsg8d79PJlHkj7a+ZFs9M+rqp1mIJaVCp1kUc9oi6tDopX4EoTYLp2WV4sT6cTNaD6fR0VroKp+cfT09K1a2NzY1Nu6w+Z/qiiwZAV/28vwFgd6u1wQMAAA==" | base64 --decode | gunzip > "$f3"

opts="--enable_nullable_tuple_type=1 --allow_experimental_nullable_tuple_type=1"

echo "-- optional LIST wrapper, REQUIRED element group: Array(Nullable(Tuple)) accepted (always-defined)"
$CLICKHOUSE_LOCAL $opts -q "SELECT a, toTypeName(a) FROM file('$f1', 'Parquet', 'a Array(Nullable(Tuple(x UInt32)))')"

echo "-- optional MAP wrapper, REQUIRED value group: Map(String, Nullable(Tuple)) accepted"
$CLICKHOUSE_LOCAL $opts -q "SELECT m, toTypeName(m) FROM file('$f3', 'Parquet', 'm Map(String, Nullable(Tuple(x UInt32)))')"

echo "-- optional element group under a list: genuine struct-level nulls, still rejected"
$CLICKHOUSE_LOCAL $opts -q "SELECT a FROM file('$f2', 'Parquet', 'a Array(Nullable(Tuple(inner Tuple(x UInt32))))')" 2>&1 | grep -o "TYPE_MISMATCH" | head -1

rm -f "$f1" "$f2" "$f3"
