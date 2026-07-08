-- Fast path for `arraySort`/`arrayReverseSort` without a lambda over numeric arrays: results must be
-- identical to the generic comparator path (which sorting through 1-tuples always takes).

SELECT 'special float values';
SELECT arraySort([1., nan, 2., 3., nan, -4., inf, -inf]);
SELECT arraySort(materialize([1., nan, 2., 3., nan, -4., inf, -inf]));
SELECT arrayReverseSort(materialize([1., nan, 2., 3., nan, -4., inf, -inf]));
SELECT arraySort(materialize(CAST([1., nan, 2., 3., nan, -4., inf, -inf], 'Array(Float32)')));
SELECT arrayReverseSort(materialize(CAST([1., nan, 2., 3., nan, -4., inf, -inf], 'Array(Float32)')));

SELECT 'nullable';
SELECT arraySort(materialize([3, NULL, 1, NULL, 2]));
SELECT arrayReverseSort(materialize([3, NULL, 1, NULL, 2]));
SELECT arraySort(materialize([1., nan, NULL, -inf, inf]));
SELECT arrayReverseSort(materialize([1., nan, NULL, -inf, inf]));

SELECT 'empty and single element';
SELECT arraySort(materialize(CAST([], 'Array(UInt8)')));
SELECT arrayReverseSort(materialize(CAST([], 'Array(Float64)')));
SELECT arraySort(materialize([toInt16(-5)]));
SELECT arrayReverseSort(materialize([toFloat64(nan)]));

SELECT 'integer types';
SELECT arraySort(materialize([toUInt8(3), 255, 0, 128, 1])), arrayReverseSort(materialize([toUInt8(3), 255, 0, 128, 1]));
SELECT arraySort(materialize([toUInt16(3), 65535, 0, 256, 1])), arrayReverseSort(materialize([toUInt16(3), 65535, 0, 256, 1]));
SELECT arraySort(materialize([toUInt32(3), 4294967295, 0, 65536, 1])), arrayReverseSort(materialize([toUInt32(3), 4294967295, 0, 65536, 1]));
SELECT arraySort(materialize([toUInt64(3), 18446744073709551615, 0, 4294967296, 1])), arrayReverseSort(materialize([toUInt64(3), 18446744073709551615, 0, 4294967296, 1]));
SELECT arraySort(materialize([toInt8(3), 127, -128, 0, -1])), arrayReverseSort(materialize([toInt8(3), 127, -128, 0, -1]));
SELECT arraySort(materialize([toInt16(3), 32767, -32768, 0, -1])), arrayReverseSort(materialize([toInt16(3), 32767, -32768, 0, -1]));
SELECT arraySort(materialize([toInt32(3), 2147483647, -2147483648, 0, -1])), arrayReverseSort(materialize([toInt32(3), 2147483647, -2147483648, 0, -1]));
SELECT arraySort(materialize([toInt64(3), 9223372036854775807, -9223372036854775808, 0, -1])), arrayReverseSort(materialize([toInt64(3), 9223372036854775807, -9223372036854775808, 0, -1]));

SELECT 'date and datetime';
SELECT arraySort(materialize([toDate('2020-01-02'), toDate('2019-12-31'), toDate('2020-01-01')]));
SELECT arrayReverseSort(materialize([toDateTime('2020-01-02 00:00:02', 'UTC'), toDateTime('2020-01-02 00:00:01', 'UTC')]));

SELECT 'counting sort for long one-byte arrays';
SELECT arraySort(arrayMap(x -> toUInt8(255 - x), range(256))) = arrayMap(x -> toUInt8(x), range(256)),
       arrayReverseSort(arrayMap(x -> toUInt8(x), range(256))) = arrayMap(x -> toUInt8(255 - x), range(256));
SELECT arraySort(arrayMap(x -> toInt8(255 - x), range(256))) = arrayMap(x -> toInt8(x - 128), range(256)),
       arrayReverseSort(arrayMap(x -> toInt8(255 - x), range(256))) = arrayMap(x -> toInt8(127 - x), range(256));

SELECT 'consistency with the generic comparator path';
WITH arrayMap(x -> toUInt8(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'UInt8', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toUInt16(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'UInt16', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toUInt32(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'UInt32', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> cityHash64(number, x), range(number % 33)) AS a
SELECT 'UInt64', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toInt8(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'Int8', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toInt16(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'Int16', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toInt32(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'Int32', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toInt64(cityHash64(number, x)), range(number % 33)) AS a
SELECT 'Int64', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toFloat32(toInt64(cityHash64(number, x) % 1000) - 500) / 8, range(number % 33)) AS a
SELECT 'Float32', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> toFloat64(toInt64(cityHash64(number, x) % 1000) - 500) / 8, range(number % 33)) AS a
SELECT 'Float64', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
WITH arrayMap(x -> if(cityHash64(number, x) % 5 = 0, NULL, toInt32(cityHash64(number, x))), range(number % 33)) AS a
SELECT 'Nullable(Int32)', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(300);
-- Sizes 200..349 straddle the counting sort threshold of 256 for one-byte types.
WITH arrayMap(x -> toUInt8(cityHash64(number, x)), range(200 + number % 150)) AS a
SELECT 'UInt8 long', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(100);
WITH arrayMap(x -> toInt8(cityHash64(number, x)), range(200 + number % 150)) AS a
SELECT 'Int8 long', sum(toString(arraySort(a)) != toString(arrayMap(t -> t.1, arraySort(arrayMap(x -> tuple(x), a)))))
     + sum(toString(arrayReverseSort(a)) != toString(arrayMap(t -> t.1, arrayReverseSort(arrayMap(x -> tuple(x), a)))))
FROM numbers(100);

SELECT 'consistency with the identity lambda';
WITH arrayMap(x -> toUInt32(cityHash64(number, x)), range(number % 33)) AS a
SELECT sum(toString(arraySort(a)) != toString(arraySort(x -> x, a)))
     + sum(toString(arrayReverseSort(a)) != toString(arrayReverseSort(x -> x, a)))
FROM numbers(300);
