-- Regression test for a signed integer overflow in `DateLUTImpl::toStartOfMonthInterval` with an extreme
-- interval count. This is the `MONTH` sibling (plus `WEEK` and `DAY`) of the `toStartOfHourInterval` /
-- `toStartOfMinuteInterval` overflows covered by 04498 and 04499. `toStartOfInterval` only validates that
-- the interval value is positive, so a huge but positive `toIntervalMonth` reaches the out-of-LUT-range
-- branch of `toStartOfMonthInterval`, where the hand-rolled floored-division idiom `-((-rel + div - 1) / div)`
-- overflows on `-rel + div` (`22800 + 9223372036854775807 cannot be represented in type 'Int64'`). The
-- rounding now goes through the overflow-safe `roundDownToMultiple` helper; the `WEEK` and `DAY` out-of-range
-- paths shared the same `x + 1 - divisor` idiom and are fixed the same way. `QUARTER` delegates to the month
-- path after `quarters * 3`, which wrapped in UInt64 (a huge quarter count became a tiny month count that
-- rounded to the wrong boundary); the product is now saturated before delegating.

-- Normal values must still be rounded down to the start of the interval correctly.
SELECT toStartOfInterval(toDate('2021-06-22'), INTERVAL 5 MONTH);
SELECT toStartOfInterval(toDate('2021-06-22'), INTERVAL 3 MONTH);
SELECT toStartOfInterval(toDate32('2021-06-22'), INTERVAL 5 MONTH);
SELECT toStartOfInterval(toDate('2021-06-22'), INTERVAL 2 WEEK);
SELECT toStartOfInterval(toDate32('2021-06-22'), INTERVAL 2 WEEK);
SELECT toStartOfInterval(toDate('2021-06-22'), INTERVAL 10 DAY);
SELECT toStartOfInterval(toDate32('2021-06-22'), INTERVAL 10 DAY);
SELECT toStartOfInterval(toDate('2021-06-22'), INTERVAL 2 QUARTER);
SELECT toStartOfInterval(toDate32('2021-06-22'), INTERVAL 2 QUARTER);

-- An extreme but positive `QUARTER` count must not wrap in `quarters * 3` down to a small month count and
-- silently round to that wrong boundary. `6148914691236517206 * 3` wraps to `2` in UInt64, so before the fix
-- this equalled `INTERVAL 2 MONTH`; with the saturated product it does not.
SELECT toStartOfInterval(toDate('2021-06-22'), INTERVAL 6148914691236517206 QUARTER)
     = toStartOfInterval(toDate('2021-06-22'), INTERVAL 2 MONTH);

-- Extreme `DateTime64` values combined with extreme interval counts reach the out-of-LUT-range rounding
-- branch, where the addition used to overflow. The result for such out-of-range values is
-- implementation-defined, so it is discarded; the test only requires that the evaluation does not overflow.
SELECT toStartOfInterval(toDateTime64(-9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 MONTH) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 MONTH) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(-9223372036854775807, 0, 'UTC'), INTERVAL 4611686018427387904 WEEK) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 WEEK) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(-9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 DAY) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 DAY) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(-9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 QUARTER) FORMAT Null;
SELECT toStartOfInterval(toDateTime64(9223372036854775807, 0, 'UTC'), INTERVAL 9223372036854775807 QUARTER) FORMAT Null;

SELECT 'ok';
