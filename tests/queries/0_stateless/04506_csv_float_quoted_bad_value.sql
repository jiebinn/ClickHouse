-- Test quoted-float parsing in CSV via the SerializationNumber CSV helpers,
-- including rejection of an unterminated quoted value.

-- A double-quoted float is accepted (2.5 is exact, so precise/fast agree).
SELECT x FROM format(CSV, 'x Float64', '"2.5"') ORDER BY x;

-- An unterminated quoted float is malformed: with input_format_csv_use_default_on_bad_values
-- the non-throwing CSV parser rejects it and the column default is inserted instead.
SELECT x FROM format(CSV, 'x Float64', '"1.5') ORDER BY x SETTINGS input_format_csv_use_default_on_bad_values = 1;
