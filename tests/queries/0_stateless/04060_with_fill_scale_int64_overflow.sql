-- Regression test: FillingRow::doLongJump doubles its internal jump length up to 2^62 to skip
-- large gaps, scaling the plain-numeric WITH FILL step via FieldVisitorScale (step * jump_len).
-- With STEP 2 and a jump length of 2^62 this computed 2 * 2^62 = 2^63, a signed integer overflow
-- (UBSan abort; silent wrap in release). doLongJump relies on the wrap to detect the overflow,
-- so the multiplication must use well-defined unsigned wraparound.
-- { echo }
SELECT arrayJoin([-4611686018427387904::Int64, 9223372036854775806::Int64]) a ORDER BY a ASC WITH FILL STEP 2 STALENESS 3;
SELECT arrayJoin([0::UInt64, 18446744073709551614::UInt64]) a ORDER BY a ASC WITH FILL STEP 2 STALENESS 3;
