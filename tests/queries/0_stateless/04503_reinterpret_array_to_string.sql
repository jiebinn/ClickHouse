-- https://github.com/ClickHouse/ClickHouse/issues/109379
-- reinterpret(<array of fixed size elements>, 'String') is the inverse of
-- reinterpret(<string>, 'Array(...)') and must be supported symmetrically.

SELECT 'Verify destination type is String';
SELECT toTypeName(reinterpret([toInt32(1)]::Array(Int32), 'String'));
SELECT toTypeName(reinterpretAsString([toInt32(1)]::Array(Int32)));

SELECT 'Verify output bytes are correct';
SELECT hex(reinterpret([toUInt8(1), toUInt8(2), toUInt8(255)]::Array(UInt8), 'String'));
SELECT hex(reinterpret([toUInt16(0x0102), toUInt16(0x0304)]::Array(UInt16), 'String'));
SELECT hex(reinterpret([toInt32(1), toInt32(2), toInt32(3)]::Array(Int32), 'String'));
SELECT hex(reinterpretAsString([toUInt16(0x0102), toUInt16(0x0304)]::Array(UInt16)));
SELECT hex(reinterpret(['ab', 'cd']::Array(FixedString(2)), 'String'));

SELECT 'The exact scenario from the issue (round-trips through String)';
WITH [toBFloat16(1.5), toBFloat16(-2.25), toBFloat16(3.0)]::Array(BFloat16) AS target
SELECT reinterpret(reinterpret(target, 'String'), 'Array(BFloat16)');

SELECT 'Empty array reinterprets to an empty string';
SELECT length(reinterpret([]::Array(Int32), 'String'));

SELECT 'Array element type must be fixed length and contiguous';
SELECT reinterpret(['a', 'b']::Array(String), 'String'); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT reinterpret([toInt32(1)]::Array(Nullable(Int32)), 'String'); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }
SELECT reinterpret([[toInt32(1)]]::Array(Array(Int32)), 'String'); -- { serverError ILLEGAL_TYPE_OF_ARGUMENT }

SELECT 'LowCardinality nested element is stripped and works';
SELECT hex(reinterpret([toInt32(1), toInt32(2)]::Array(LowCardinality(Int32)), 'String')) SETTINGS allow_suspicious_low_cardinality_types = 1;

SELECT 'A few rows read from a table';
DROP TABLE IF EXISTS tab_04503;
CREATE TABLE tab_04503 (id Int32, arr Array(UInt8)) ENGINE = Memory;
INSERT INTO tab_04503 VALUES (1, [1, 2]), (2, [10, 20, 30]), (3, []);
SELECT id, hex(reinterpret(arr, 'String')) FROM tab_04503 ORDER BY id;
DROP TABLE tab_04503;
