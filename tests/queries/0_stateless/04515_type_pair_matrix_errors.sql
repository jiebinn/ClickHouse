-- Exceptional cases of comparison and arithmetic over mixed type pairs involving
-- Decimal and 128/256-bit integers. Companion of 04513_type_pair_matrix_comparison
-- and 04514_type_pair_matrix_arithmetic, which cover the non-throwing matrix.

SELECT '-- comparison overflow during scale conversion';
SELECT materialize(CAST('18446744073709551615', 'UInt64')) > materialize(toDecimal32('1.5', 2)); -- { serverError DECIMAL_OVERFLOW }
SELECT materialize(toUInt64(9223372036854775807)) > materialize(toDecimal64('1.5', 4)); -- { serverError DECIMAL_OVERFLOW }
SELECT materialize(toDecimal32('1.5', 2)) = materialize(CAST('115792089237316195423570985008687907853269984665640564039457584007913129639935', 'UInt256')); -- { serverError DECIMAL_OVERFLOW }

SELECT '-- the same comparisons with the overflow check disabled';
SELECT materialize(CAST('18446744073709551615', 'UInt64')) > materialize(toDecimal32('1.5', 2)) SETTINGS decimal_check_overflow = 0;
SELECT materialize(toDecimal32('1.5', 2)) = materialize(CAST('115792089237316195423570985008687907853269984665640564039457584007913129639935', 'UInt256')) SETTINGS decimal_check_overflow = 0;

SELECT '-- extreme values that must compare without overflow';
SELECT materialize(CAST('-170141183460469231731687303715884105728', 'Int128')) < materialize(toDecimal32('1.5', 2));
SELECT materialize(CAST('-57896044618658097711785492504343953926634992332820282019728792003956564819968', 'Int256')) != materialize(toDecimal64('1.5', 4));

SELECT '-- division by zero in every width direction';
SELECT intDiv(materialize(toUInt256(1)), materialize(toUInt8(0))); -- { serverError ILLEGAL_DIVISION }
SELECT intDiv(materialize(toUInt8(1)), materialize(toUInt256(0))); -- { serverError ILLEGAL_DIVISION }
SELECT intDiv(materialize(toInt256(1)), materialize(toInt256(0))); -- { serverError ILLEGAL_DIVISION }
SELECT modulo(materialize(toInt256(-1)), materialize(toUInt64(0))); -- { serverError ILLEGAL_DIVISION }
SELECT modulo(materialize(toUInt64(1)), materialize(toUInt256(0))); -- { serverError ILLEGAL_DIVISION }
SELECT positiveModulo(materialize(toInt128(-1)), materialize(toUInt32(0))); -- { serverError ILLEGAL_DIVISION }
SELECT moduloLegacy(materialize(toUInt256(1)), materialize(toUInt8(0))); -- { serverError ILLEGAL_DIVISION }

SELECT '-- division of the minimal signed number by minus one, wide and narrow dividends';
SELECT intDiv(materialize(CAST('-57896044618658097711785492504343953926634992332820282019728792003956564819968', 'Int256')), materialize(toInt8(-1))); -- { serverError ILLEGAL_DIVISION }
SELECT intDiv(materialize(CAST('-170141183460469231731687303715884105728', 'Int128')), materialize(toInt64(-1))); -- { serverError ILLEGAL_DIVISION }
SELECT intDiv(materialize(toInt8(-128)), materialize(toInt256(-1))); -- { serverError ILLEGAL_DIVISION }
SELECT modulo(materialize(CAST('-170141183460469231731687303715884105728', 'Int128')), materialize(toInt64(-1))); -- { serverError ILLEGAL_DIVISION }
SELECT modulo(materialize(toInt8(-128)), materialize(toInt128(-1))); -- { serverError ILLEGAL_DIVISION }

SELECT '-- positiveModulo by the most negative divisor value';
SELECT positiveModulo(materialize(toInt256(-100)), materialize(toInt8(-128))); -- { serverError ILLEGAL_DIVISION }

SELECT '-- OrZero and OrNull variants of the same exceptional cases do not throw';
SELECT intDivOrZero(materialize(CAST('-57896044618658097711785492504343953926634992332820282019728792003956564819968', 'Int256')), materialize(toInt8(-1)));
SELECT intDivOrNull(materialize(CAST('-170141183460469231731687303715884105728', 'Int128')), materialize(toInt64(-1)));
SELECT moduloOrNull(materialize(toInt8(-128)), materialize(toInt128(-1)));
SELECT intDivOrZero(materialize(toUInt8(1)), materialize(toUInt256(0)));
SELECT moduloOrNull(materialize(toInt256(-1)), materialize(toUInt64(0)));

SELECT '-- combinations of a floating point number and a wide integer';
SELECT materialize(toFloat64(1)) + materialize(toUInt256(1)); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT materialize(toUInt256(1)) - materialize(toFloat64(1)); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT materialize(toFloat32(1)) * materialize(toInt128(1)); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT intDiv(materialize(toUInt256(1)), materialize(toFloat64(1)));
SELECT modulo(materialize(toInt128(1)), materialize(toFloat32(1)));

SELECT '-- isDistinctFrom does not support Decimal vs wide integer';
SELECT isDistinctFrom(materialize(toDecimal32('1.00', 2)), materialize(toUInt128(1))); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
