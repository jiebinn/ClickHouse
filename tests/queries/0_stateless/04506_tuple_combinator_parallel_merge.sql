-- `uniqExact` merges its states through a thread pool, and the `-Tuple` combinator forwards the
-- capability element-wise. `max_threads` is pinned above 1 so that several partial aggregation
-- states exist and the final merge actually takes the parallel path; the results must match the
-- plain functions. An only-null element resolves to a placeholder without parallel merge support
-- and merges plainly alongside the parallelized element.
SET max_threads = 4;

SELECT 'parallel merge equals plain';
SELECT (SELECT uniqExactTuple((number, intDiv(number, 2))) FROM numbers_mt(300000)) = (SELECT (uniqExact(number), uniqExact(intDiv(number, 2))) FROM numbers_mt(300000));

SELECT 'placeholder element alongside a parallelized element';
SELECT uniqExactTuple((NULL, number)) FROM numbers_mt(300000);
