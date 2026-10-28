SET max_threads = 16;
SET max_block_size = 17;

SELECT count() = 100003, uniqExact(number) = 100003,
    min(number) = 0, max(number) = 100002,
    sum(number) = toUInt64(100002) * 100003 / 2,
    max(blockSize()) <= 17
FROM numbers_mt(100003);

SELECT parallel = serial
FROM (SELECT arraySort(groupArray(number)) AS parallel
      FROM numbers_mt(200003)
      WHERE number BETWEEN 10 AND 50000 OR number BETWEEN 70000 AND 120000) AS p
CROSS JOIN (SELECT arraySort(groupArray(number)) AS serial
            FROM numbers(200003)
            WHERE number BETWEEN 10 AND 50000 OR number BETWEEN 70000 AND 120000) AS s;

SELECT parallel = serial
FROM (SELECT arraySort(groupArray(number)) AS parallel
      FROM numbers_mt(18446744073709501615, 100003, 3)) AS p
CROSS JOIN (SELECT arraySort(groupArray(number)) AS serial
            FROM numbers(18446744073709501615, 100003, 3)) AS s;

SELECT parallel = serial
FROM (SELECT arraySort(groupArray(number)) AS parallel
      FROM numbers_mt(18446744073709501615, 100003, 3)
      WHERE number < 40000 OR number >= 18446744073709521615) AS p
CROSS JOIN (SELECT arraySort(groupArray(number)) AS serial
            FROM numbers(18446744073709501615, 100003, 3)
            WHERE number < 40000 OR number >= 18446744073709521615) AS s;

SELECT groupArray(number) = range(73, 273)
FROM (SELECT number FROM numbers_mt(100003) ORDER BY number LIMIT 200 OFFSET 73);

SELECT count() = 200, uniqExact(number) = 200
FROM (SELECT number FROM numbers_mt(100003) LIMIT 200 OFFSET 73);
