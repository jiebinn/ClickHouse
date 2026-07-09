import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import ci.jobs.performance_tests as performance_tests
from ci.jobs.performance_tests import (
    INSERT_HISTORICAL_DATA,
    SLOWER_QUERIES_FAIL_THRESHOLD,
    sort_perf_tests_attention_first,
    too_many_slow,
)


class _StubResult:
    """Minimal Result-like stub - only .status/.name are read by the sort."""

    def __init__(self, name, status):
        self.name = name
        self.status = status

    def __repr__(self):
        return f"({self.status}, {self.name})"


def test_historical_insert_parses_threshold_columns():
    # compare.sh appends changed_threshold/unstable_threshold to
    # all-query-metrics.tsv. The historical-data CIDB insert reads that file
    # with a fixed input() TSV schema; if the schema does not declare these two
    # trailing columns, strict TSV parsing drops every row silently (0 rows
    # inserted, no error). This guards that coupling.
    assert "changed_threshold Float64" in INSERT_HISTORICAL_DATA
    assert "unstable_threshold Float64" in INSERT_HISTORICAL_DATA


def test_gate_threshold_value():
    # The Praktika gate must stay aligned with report.py's "> 10" slower-query
    # threshold. If this constant changes, report.py must change in lockstep.
    assert SLOWER_QUERIES_FAIL_THRESHOLD == 10


def test_no_slower_queries_does_not_fail():
    assert too_many_slow("ok") is False
    assert too_many_slow("3 faster") is False


def test_slower_count_at_or_below_threshold_does_not_fail():
    # 6-10 slower queries used to fail the check with the old threshold of 5.
    # They must now pass, which is the whole point of the change.
    for n in (1, 5, 6, 9, 10):
        assert too_many_slow(f"{n} slower") is False, n
        assert too_many_slow(f"2 faster, {n} slower") is False, n


def test_slower_count_above_threshold_fails():
    for n in (11, 12, 50):
        assert too_many_slow(f"{n} slower") is True, n
        assert too_many_slow(f"1 faster, {n} slower") is True, n


def test_cidb_inserts_are_best_effort_not_asserted():
    # CIDB metrics/history inserts are a reporting side-effect, NOT the perf
    # verdict. A bare `assert cidb.is_ready()` turns a transient LogCluster
    # (play.clickhouse.com) timeout into a whole-job exit-1, which is exactly
    # what failed arm_release/master_head shards (PR #107236). Every is_ready()
    # call site in the perf job must use the graceful skip-and-warn guard
    # (`if not cidb.is_ready(): ... return True`) instead of asserting.
    source = inspect.getsource(performance_tests.main)
    assert "assert cidb.is_ready()" not in source, (
        "A bare `assert cidb.is_ready()` re-introduces the whole-job failure on "
        "transient CIDB timeouts. Use `if not cidb.is_ready(): ... return True`."
    )
    # All four is_ready() call sites must be guarded; none asserted.
    assert source.count("cidb.is_ready()") == 4
    assert source.count("if not cidb.is_ready():") == 4


def _make_perf_rows():
    """Reproduce the shape produced by ci-checks.tsv: alphabetical by test name,
    with ``::old`` and ``::new`` sides adjacent and sharing a status. The four
    non-``success`` rows are buried among successes so we can see them move."""
    return [
        _StubResult("aggregation_in_order_2 #0::old", "success"),
        _StubResult("aggregation_in_order_2 #0::new", "success"),
        _StubResult("aggregation_in_order_2 #1::old", "success"),
        _StubResult("aggregation_in_order_2 #1::new", "success"),
        _StubResult("fixed_hash_table_parallel_merge #0::old", "slower"),
        _StubResult("fixed_hash_table_parallel_merge #0::new", "slower"),
        _StubResult("group_by_high_card #0::old", "success"),
        _StubResult("group_by_high_card #0::new", "success"),
        _StubResult("hash_table_sizes_stats_small #25::old", "unstable"),
        _StubResult("hash_table_sizes_stats_small #25::new", "unstable"),
        _StubResult("window_functions #5::old", "success"),
        _StubResult("window_functions #5::new", "success"),
    ]


def test_sort_puts_slower_and_unstable_at_the_top():
    rows = _make_perf_rows()
    sort_perf_tests_attention_first(rows)
    # The four attention-worthy rows must appear first, `slower` before
    # `unstable`, and each `::old`/`::new` pair kept adjacent.
    assert [r.name for r in rows[:4]] == [
        "fixed_hash_table_parallel_merge #0::old",
        "fixed_hash_table_parallel_merge #0::new",
        "hash_table_sizes_stats_small #25::old",
        "hash_table_sizes_stats_small #25::new",
    ], rows
    assert [r.status for r in rows[:4]] == ["slower", "slower", "unstable", "unstable"]
    # Everything after is `success`, in the original ci-checks.tsv order
    # (stable sort preserves within-bucket ordering).
    assert [r.name for r in rows[4:]] == [
        "aggregation_in_order_2 #0::old",
        "aggregation_in_order_2 #0::new",
        "aggregation_in_order_2 #1::old",
        "aggregation_in_order_2 #1::new",
        "group_by_high_card #0::old",
        "group_by_high_card #0::new",
        "window_functions #5::old",
        "window_functions #5::new",
    ]


def test_sort_surfaces_unknown_status_at_the_top():
    # A new status added to `compare.sh` without also updating the sort key
    # must NOT sink below hundreds of successful rows - we would rather be
    # noisy than silently hide a regression signal.
    rows = [
        _StubResult("something_ok::old", "success"),
        _StubResult("something_ok::new", "success"),
        _StubResult("weird_new_status::old", "error"),
        _StubResult("weird_new_status::new", "error"),
    ]
    sort_perf_tests_attention_first(rows)
    assert [r.status for r in rows] == ["error", "error", "success", "success"]


def test_sort_is_stable_when_all_success():
    # No-op behaviour: an all-`success` list must come out in the same order.
    rows = [_StubResult(f"t #{i}::old", "success") for i in range(6)]
    original = [r.name for r in rows]
    sort_perf_tests_attention_first(rows)
    assert [r.name for r in rows] == original


if __name__ == "__main__":
    test_historical_insert_parses_threshold_columns()
    test_gate_threshold_value()
    test_no_slower_queries_does_not_fail()
    test_slower_count_at_or_below_threshold_does_not_fail()
    test_slower_count_above_threshold_fails()
    test_cidb_inserts_are_best_effort_not_asserted()
    test_sort_puts_slower_and_unstable_at_the_top()
    test_sort_surfaces_unknown_status_at_the_top()
    test_sort_is_stable_when_all_success()
    print("All perf slow-gate tests passed.")
