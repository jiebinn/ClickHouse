-- Every non-alias function must carry a non-empty description in its structured documentation.
-- Aliases inherit their target's documentation, so they are exempt; internal functions carry the
-- shared internal-function documentation (FunctionDocumentation::INTERNAL_FUNCTION_DOCS).
SELECT name FROM system.functions WHERE length(description) < 10 AND alias_to = '';
