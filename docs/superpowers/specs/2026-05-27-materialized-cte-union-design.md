# Design: Deduplicate materialized CTEs across UNION branches

Status: Implemented (with amendments); see Amendments section below.
Owner: Dmitry Novik (branch `fix-mat-cte`).
Date: 2026-05-27. Amended: 2026-07-07.

## Amendments (2026-07-07)

The implementation (commits `d48b43495a0`, `a7b517a25c1`, `250a40b6444`,
`24e8fbc743d`) diverges from the original plan below in several ways. Rather
than silently rewriting the plan, the affected passages are kept and marked
`~~struck through~~` with an inline `*Amended: ...*` note explaining what
was actually built instead. Summary of the divergences:

1. **Merge key is `(cte_name, subquery hash)`, not body-only.** The
   "Note on scope" decision to deliberately merge identical bodies under
   different `cte_name`s was reversed. `.ignore_cte = true` is still used
   for the subquery hash/`isEqual`, but `cte_name` is a separate, required
   key component - different names never merge, even with identical bodies.
2. **Steps A and B are one fused, single-pass post-order traversal**,
   not a collect-then-sort-then-group pass followed by a separate
   counting pass. Canonical selection is "first encountered in traversal
   order" rather than "lexicographically smallest `temporary_table_name`
   after a descending-depth sort".
3. **Only definition-bearing `TableNode`s participate.** `TableNode`s that
   merely inherited a `MaterializedCTE` pointer from an already-materialized
   `StorageMemory` (via `TableNode::extractCTE`, e.g. on shard-side
   re-analysis of a `Distributed` query) are skipped entirely - they are
   neither merge candidates/canonicals nor counted towards reuse. Counting
   them caused `TABLE_ALREADY_EXISTS` on shards
   (`04043_materialized_cte_serialize_query_plan`).
4. **The schema gate moved and became non-throwing before adoption.**
   `verifyMaterializedCTESubqueryMatchesStorage` moved to
   `Analyzer/Utils.{h,cpp}`, returns `bool`, and takes a
   `throw_on_mismatch` parameter; the merge path calls it with `false` and
   simply skips merging on mismatch instead of throwing. Adoption itself
   is `TableNode::adoptMaterializedCTE`, a new method.
5. **The registration guard is now real.** Reused CTEs are registered via
   `addExternalTable` only when `table_holder.has_value()`; a bug in
   `MaterializedCTE::extractTableHolder` (it moved out of `table_holder`
   without resetting the `std::optional`) was fixed so `has_value()`
   reliably reports whether the holder was already extracted.
6. **`IQueryTreeNode::CompareOptions` has no `compare_types` field.** Places
   below that say `compare_types = true` are wrong; column types are
   compared unconditionally via `verifyMaterializedCTESubqueryMatchesStorage`,
   which checks the subquery's projection column types against the
   canonical's storage columns - not via `isEqual`/`CompareOptions`.
7. **Tests** landed as `tests/queries/0_stateless/04507_materialized_cte_union_merge.sql`
   (the planned `04045_materialized_cte_union_dedup.sql` name was not used).
   Its case for "different `cte_name`, identical body" now asserts the
   opposite of the original plan: it must **not** merge (two
   `MaterializingCTE` steps), matching amendment 1.

## Problem

A `MATERIALIZED` CTE referenced from multiple branches of a `UNION` is silently
duplicated: each branch ends up with its own `MaterializedCTE` object, its own
`StorageMemory`, its own `temporary_table_name`, and (worse) gets inlined
as a plain subquery instead of being materialized once and shared.

Example:

```sql
WITH x AS MATERIALIZED (SELECT * FROM big_table WHERE expensive_filter)
SELECT count() FROM x
UNION ALL
SELECT max(value) FROM x;
```

Expected: `x` is materialized once, both branches read the temporary table.
Observed: `big_table` is scanned twice with the expensive filter applied each
time; the CTE never materializes.

### Root cause

The duplication originates in `ApplyWithGlobalVisitor`
(`src/Interpreters/ApplyWithGlobalVisitor.cpp:26`). At AST level, it
propagates the `WITH` definitions from the first `ASTSelectQuery` of a
`UNION` into every other branch by `child->clone()`. This is correct and
necessary for non-materialized CTEs - each branch's scope needs its own
copy of the binding to resolve - but for materialized CTEs it produces
two AST nodes that look like independent CTE definitions to the
analyzer.

Then in the analyzer, `QueryAnalyzer::tryResolveIdentifierFromCTE`
(`src/Analyzer/Resolve/QueryAnalyzer.cpp:1293`) hits the cloned CTE node
independently in each branch's scope and runs:

```cpp
auto table_node = std::make_shared<TableNode>(full_name, cte_node, scope.context);
```

The `TableNode` constructor at `src/Analyzer/TableNode.cpp:80-93`
allocates a fresh `MaterializedCTE` shared_ptr per call. One logical CTE
becomes N `MaterializedCTE` objects (one per UNION branch), each with
its own random `temporary_table_name`, its own dummy storage, and -
after `finalizeMaterializedCTE` runs in each branch's resolution - its
own real `StorageMemory` and `TemporaryTableHolder`.

Two downstream consumers key off `MaterializedCTEPtr` identity:

1. `collectMaterializedCTEs` (`src/Planner/CollectMaterializedCTE.cpp:19`)
   keys its map by pointer, so distinct pointers stay distinct and each
   gets its own materialization plan + storage.
2. `inlineMaterializedCTEIfNeeded`
   (`src/Analyzer/inlineMaterializedCTEIfNeeded.cpp`) uses
   `reused_materialized_cte` - a set populated in
   `QueryAnalyzer::tryResolveIdentifierFromCTE` only on the
   *second-and-later* resolution that hits the same pointer
   (`src/Analyzer/Resolve/QueryAnalyzer.cpp:1302`). A UNION-clone CTE
   referenced once per branch is resolved once per pointer, never lands
   in the set, and so gets *inlined* by the visitor instead of staying
   materialized. This is the surface bug.

### Precedent

`CollectSetsVisitor` (`src/Planner/CollectSets.cpp:50`) is the
analogous deduplication pattern in the planner: it keys `PreparedSets`
entries by `getTreeHash({.ignore_cte = true})`. We follow the same
approach at the analyzer level for materialized CTEs.

## Goals

1. Materialized CTEs referenced from multiple UNION branches materialize
   exactly once and both branches read the shared temporary table.
2. No regression for the existing single-branch and in-branch
   repeat-reference cases (already handled by
   `tryResolveIdentifierFromCTE`'s pointer-identity reuse path).
3. No new public API; no new setting; no compat flag - the prior
   behavior was strictly buggy.
4. Distributed and parallel-replicas execution inherits the fix without
   special handling.

## Non-goals

- A smarter `ApplyWithGlobalVisitor` that does not clone for
  materialized CTEs. Considered and rejected: many code paths assume
  WITH definitions are physically present in each branch's AST; a
  scope-climb resolution would touch much more surface.

~~Note on scope: the dedup uses `.ignore_cte = true` on the subquery
hash, so it will *also* collapse two materialized CTEs that happen to
have identical bodies under different `cte_name`s (e.g.
`WITH x AS MATERIALIZED (SELECT 1), y AS MATERIALIZED (SELECT 1)
SELECT * FROM x, y`). This is a deliberate side-effect, parallel to
how `CollectSets` already deduplicates structurally-equal sets
regardless of source naming. Duplicate work is duplicate work; merging
them is a net win.~~ *Amended: this decision was reversed by the owner
before implementation. The merge key is `(cte_name, subquery hash)`,
so `x` and `y` above do **not** merge even though their bodies are
identical - only same-named CTEs with equal bodies merge.
`.ignore_cte = true` is still used, but only to strip the `is_cte`/
`cte_name`/`is_materialized` binding metadata out of the subquery hash
itself; the name is compared separately as its own key component. See
"Amendments" above.* Column aliases inside the body still keep CTEs
separate via `compare_aliases = true`, so `SELECT 1 AS a` vs
`SELECT 1 AS b` does not merge.

## Design

### Locus

Primary changes inside `src/Analyzer/inlineMaterializedCTEIfNeeded.cpp`.
Companion cleanup in `src/Analyzer/Resolve/QueryAnalyzer.{h,cpp}`.

`MaterializedCTE` is not changed (we considered storing nesting depth
on the struct but ruled it out: depth becomes stale once inlining
removes references).

#### Signature change

The public function loses the `reused_materialized_cte` out-parameter.
The set is now derivable from the deduped tree and is built locally:

```cpp
// before
void inlineMaterializedCTEIfNeeded(
    QueryTreeNodePtr & node,
    ReusedMaterializedCTEs & reused_materialized_cte,
    ContextPtr context);

// after
void inlineMaterializedCTEIfNeeded(
    QueryTreeNodePtr & node,
    ContextPtr context);
```

`ReusedMaterializedCTEs` (currently `std::unordered_set<MaterializedCTEPtr>`
typedef in the header) becomes a file-local detail and moves into
`inlineMaterializedCTEIfNeeded.cpp` along with
`InlineMaterializedCTEsVisitor`. Nothing outside the file uses the
type after the signature change.

#### Companion cleanup in `QueryAnalyzer`

The `std::unordered_set<MaterializedCTEPtr> reused_materialized_cte;`
member at `src/Analyzer/Resolve/QueryAnalyzer.h:306` is deleted. The
incremental `reused_materialized_cte.insert(table_node->getMaterializedCTE());`
at `src/Analyzer/Resolve/QueryAnalyzer.cpp:1302` is deleted. The call
site at line 274 becomes `inlineMaterializedCTEIfNeeded(node, context);`.

Rationale: before this change, the set was incrementally accumulated
during identifier resolution as a side-effect of `tryResolveIdentifierFromCTE`
encountering a CTE `TableNode` for the second time. That accumulation
was always tied to pointer identity; with dedup, identity is now
established post-resolution, so the only correct moment to count
references is *after* the dedup pass has finished. There is no longer
any value computed during resolution that the inline pass needs;
keeping the member and the line-1302 insert as a redundant
pre-population would just be dead work that risks future readers
treating it as load-bearing.

### ~~Step A - `deduplicateMaterializedCTEs(node, context)` (file-local static)~~ *(superseded)*

*Amended: Steps A and B below were fused into a single post-order pass in
the implementation; see "Step A+B (as implemented)" further down for what
actually shipped. The plan in this subsection - collect entries with
depth, stable-sort descending, group by hash into buckets, pick the
lexicographically-smallest `temporary_table_name` as canonical - is kept
verbatim for historical context but does not describe the merged code.*

1. Walk the tree with `traverseQueryTree(node, Everything{}, enter, leave)`
   mirroring the depth-tracking shape of `collectMaterializedCTEs`
   (`src/Planner/CollectMaterializedCTE.cpp:33-56`). Maintain a local
   `size_t depth = 0`; on entering a materialized-CTE `TableNode`,
   append `{table_node, depth}` to a `std::vector<Entry>` and `++depth`;
   on leaving, `--depth`. Depth is pass-local; not persisted anywhere.

2. Stable-sort entries by descending depth. This is the load-bearing
   ordering property: any subquery body that references an inner
   materialized CTE must have that inner CTE canonicalized first, so
   the inner's `TableNode::temporary_table_name` (which participates in
   `TableNode::updateTreeHashImpl`) is stable across cloned outer
   bodies.

3. Group by `entry.table_node->getMaterializedCTESubquery()->getTreeHash({.ignore_cte = true})`
   into `std::unordered_map<IQueryTreeNode::Hash, std::vector<size_t>>`.
   Defaults on the other two `CompareOptions` fields, so
   `compare_aliases = true` and `compare_types = true`. Within a hash
   bucket, verify with
   `subquery->isEqual(bucket_head_subquery, {.ignore_cte = true})`;
   inequality starts a new singleton bucket. Hash collisions never
   collapse semantically distinct CTEs.

   Why `ignore_cte = true`: the subquery body's outer `QueryNode` /
   `UnionNode` carries binding-context metadata (`is_cte`, `cte_name`,
   `is_materialized`). For the UNION-clone case both clones share the
   same `cte_name`, so the flag's value doesn't matter for this case.
   We use `true` to match the `CollectSets` precedent and to keep the
   hash a pure function of the body's content.

4. For every bucket with size >= 2, pick canonical = entry with
   lexicographically smallest `materialized_cte->temporary_table_name`
   (deterministic, traversal-order-independent - keeps EXPLAIN stable
   across multiple runs of the same query). For every non-canonical
   entry, mirror the handoff the analyzer already uses for the
   in-branch repeat-reference case
   (`src/Analyzer/Resolve/QueryAnalyzer.cpp:3135`):

   ```cpp
   table_node->materialized_cte = canonical.materialized_cte;
   table_node->setTemporaryTableName(canonical.materialized_cte->temporary_table_name);
   table_node->updateStorage(canonical.materialized_cte->storage, context);
   ```

   `updateStorage` (`src/Analyzer/TableNode.cpp:104`) resets `storage`,
   `storage_id`, `storage_lock`, and `storage_snapshot` consistently.
   `children[materialized_cte_subquery_index]` is left in place: it is
   structurally equal to the canonical's subquery by step 3, only the
   canonical's gets planned downstream, and keeping the local copy
   leaves EXPLAIN QUERY TREE output for that branch self-contained.

   Orphan lifecycle: the non-canonical `TableNode` releases its old
   `MaterializedCTE` shared_ptr (last strong ref); orphan destructs,
   orphan's `TemporaryTableHolder` destructs (unregistering its
   external-table row), orphan `StorageMemory` destructs. The back-ref
   from `StorageMemory` to `MaterializedCTE` is
   `MaterializedCTEWeakPtr` (`src/Storages/StorageMemory.h:129/149`)
   so no cycle. Clean handoff.

### ~~Step B - `collectReusedMaterializedCTEs(node) -> ReusedMaterializedCTEs` (file-local static)~~ *(superseded)*

*Amended: this separate walk was never built as its own pass. The
use-count map it describes is instead produced as a side output of the
fused Step A+B pass below. Kept verbatim for historical context.*

The set is built fresh from the deduped tree:

1. Walk the tree once; for each `TableNode` whose
   `getMaterializedCTE()` is non-null, increment a counter in a local
   `std::unordered_map<MaterializedCTEPtr, size_t>`.
2. Return a `ReusedMaterializedCTEs` containing every pointer with
   count >= 2.

After Step B, the set contains exactly the canonical pointers that are
referenced from >= 2 places in the deduped tree - including
cross-branch references that previously couldn't accumulate because
they hit distinct pointers. The set lives only for the remaining
duration of `inlineMaterializedCTEIfNeeded`; nothing outside the
function depends on it.

### Step A+B (as implemented) - `mergeDuplicateMaterializedCTEs(node, context)` (file-local static)

What actually shipped, in `src/Analyzer/inlineMaterializedCTEIfNeeded.cpp`:
a single post-order traversal via `traverseQueryTree(node, Everything{}, NoOp{}, leave)`
that merges duplicates *and* builds the use-count map in the same pass, keyed
by a `MergeKey{cte_name, subquery_hash}` struct (not just the subquery hash).

1. On the leave callback for each `TableNode`: skip immediately unless
   `getMaterializedCTE()` is non-null *and* `isMaterializedCTE()` is true.
   The latter check excludes non-definition-bearing occurrences - a
   `TableNode` can carry a non-null `MaterializedCTE` purely because
   `TableNode::extractCTE` read it off an already-materialized
   `StorageMemory` resolved as an ordinary table (e.g. a nested
   re-resolve of a `Distributed` table's shard-local plan, where the
   CTE's temporary table name was substituted for the original CTE
   reference). Counting or merging those threw `TABLE_ALREADY_EXISTS`
   on shards, reproduced by
   `04043_materialized_cte_serialize_query_plan`.

2. Because the traversal acts on *leave*, any inner materialized CTE
   nested inside the current node's subquery has already been merged by
   the time the current node is processed - so its `TableNode`s carry
   canonical, stable `temporary_table_name`s, and the current subquery's
   `getTreeHash` is stable across UNION-branch clones. This replaces the
   depth-tracking-plus-descending-sort of the original plan with a
   simpler invariant: post-order traversal order already guarantees
   inner-before-outer.

3. Compute `MergeKey{table_node->getMaterializedCTE()->cte_name, subquery->getTreeHash({.compare_aliases = true, .ignore_cte = true})}`
   and look it up in a `std::unordered_map<MergeKey, std::vector<TableNode *>, MergeKeyHash>`.
   `IQueryTreeNode::CompareOptions` has only two fields, `compare_aliases`
   and `ignore_cte` - there is no `compare_types`; see the third bullet
   below for how column types are actually checked. Within the bucket,
   each existing candidate is checked with
   `subquery->isEqual(*candidate->getMaterializedCTESubquery(), compare_options)`
   *and* a schema gate (next bullet); the first candidate that passes
   both is the canonical for this node. Hash collisions never collapse
   semantically distinct CTEs because of the `isEqual` check.

4. Schema gate: `verifyMaterializedCTESubqueryMatchesStorage` (moved to
   `Analyzer/Utils.{h,cpp}`, returning `bool` and taking a
   `throw_on_mismatch` parameter) is called with `throw_on_mismatch =
   false`. It compares the candidate node's projection columns (name
   count and types) against the canonical's storage columns; this is
   where column types are actually compared - unconditionally, not
   gated by any `CompareOptions` flag. On mismatch (should be impossible
   past the `isEqual` gate, but the check is defensive) the node simply
   does not merge with that candidate and stays a separate
   materialization; it does not throw.

5. If no candidate qualifies, the current node becomes the canonical for
   its `MergeKey` (first encountered in traversal order = canonical;
   deterministic per tree, unlike the original plan's
   lexicographically-smallest-name rule). Otherwise, if the canonical's
   `MaterializedCTE` pointer differs from this node's, the node calls
   `TableNode::adoptMaterializedCTE(canonical_cte, context)` - a new
   method that sets `materialized_cte`, `temporary_table_name`, and
   calls `updateStorage` in one step (functionally the same handoff the
   original plan described inline). If the pointers already match (the
   in-branch repeat-reference case), adoption is a no-op.

6. The per-`MaterializedCTEPtr` use count is incremented *after*
   adoption, on `table_node->getMaterializedCTE()` - so both in-branch
   shared-pointer repeats and cross-branch merged repeats accumulate on
   the same canonical pointer. The function returns the finished
   `std::unordered_map<MaterializedCTEPtr, size_t>` directly; the driver
   (`inlineMaterializedCTEIfNeeded`) fast-exits if the map is empty
   (no materialized CTEs at all - skips the second traversal and
   `cloneAndReplace` entirely), and otherwise derives the reused set as
   every pointer with count >= 2.

Orphan lifecycle is unchanged from the original plan: the non-canonical
`TableNode` releases its old `MaterializedCTE` shared_ptr (last strong
ref); orphan destructs, orphan's `TemporaryTableHolder` destructs
(unregistering its external-table row), orphan `StorageMemory` destructs.
The back-ref from `StorageMemory` to `MaterializedCTE` is
`MaterializedCTEWeakPtr` (`src/Storages/StorageMemory.h:129/149`) so no
cycle.

### Step C - existing `InlineMaterializedCTEsVisitor` (unchanged behavior)

Driven by the use-count-derived reused set from Step A+B. CTEs that
survived merging with multiple references stay materialized; CTEs that
genuinely have a single use get inlined as today.

The existing `addExternalTable` loop in `inlineMaterializedCTEIfNeeded`
runs over the locally-built set; each `temporary_table_name` is
registered exactly once because the set contains only canonical
pointers. *Amended: the loop additionally guards each registration with
`materialized_cte->table_holder.has_value()`, skipping a CTE whose
holder was already extracted by an earlier pass - `inlineViewSubqueryIfNeeded`
runs a nested `QueryAnalyzer::resolve` on a view subtree, which can
register a reused CTE before the outer resolve reaches this loop again.
This guard only became reliable after fixing
`MaterializedCTE::extractTableHolder`, which moved out of `table_holder`
without calling `reset()` on the `std::optional` - a moved-from
`std::optional` still reports `has_value() == true`, so the stale value
let a second registration attempt through and threw `TABLE_ALREADY_EXISTS`.*

## Edge cases

| Case | Behavior |
|------|----------|
| Recursive WITH containing MATERIALIZED | Already throws `UNSUPPORTED_METHOD` at `src/Analyzer/QueryTreeBuilder.cpp:354-356`. Dedup never runs. |
| Self-cycle (`WITH a AS MATERIALIZED (SELECT FROM b), b AS MATERIALIZED (SELECT FROM a)`) | Already throws `UNKNOWN_TABLE` during resolution (`tests/queries/0_stateless/04044_materialized_cte_cycle.sql`). Dedup never runs. |
| Subquery (`SelectQueryOptions::is_subquery == true`) | Dedup runs (analyzer always runs); Planner-side `collectMaterializedCTEs` early-returns unless `force_materialize_cte` is set (`src/Planner/CollectMaterializedCTE.cpp:25`). Dedup is a no-op cost here, no behavior change. |
| Distributed query | The initiator's analyzed tree is already deduped. The receiving node re-parses + re-analyzes from the AST/serialized form, runs `inlineMaterializedCTEIfNeeded` again, applies the same dedup. Outcome by construction is identical. *Amended: a shard-side re-analysis (`TableNode::extractCTE` re-reading a `MaterializedCTE` off an already-materialized `StorageMemory`, seen when a `Distributed` query has a materialized CTE referenced twice inside a `WHERE ... IN (...)` shape) is a non-definition-bearing occurrence and must be excluded from merge candidacy and reuse counting; otherwise it throws `TABLE_ALREADY_EXISTS`. Reproduced by `04043_materialized_cte_serialize_query_plan` and fixed as described in "Amendments" point 3.* |
| Parallel replicas | Same as distributed. Replica-side rebuild from query tree (e.g. `src/Storages/buildQueryTreeForShard.cpp`) operates on already-deduped state from the initiator and re-analyzes deduped state on each replica. |
| Two materialized CTEs with identical bodies, different `cte_name` | ~~*Do* dedup. `ignore_cte = true` hides the outer-node binding metadata so the bucket sees one logical body. See "Non-goals" note. Test #5 pins this behavior.~~ *Amended: does **not** merge. `cte_name` is part of the merge key, so distinctly-named CTEs stay separate regardless of body equality. Pinned by case 3 of `04507_materialized_cte_union_merge.sql`.* |
| Two materialized CTEs with same body but different column aliases (`SELECT 1 AS a` vs `SELECT 1 AS b`) | Stay separate. Inner aliases participate in the hash because `compare_aliases = true`. |
| In-branch repeat reference (`SELECT FROM x JOIN x`) within a single SELECT | Already deduped by `tryResolveIdentifierFromCTE`'s pointer-reuse path (line 1302). Dedup is a no-op; the bucket sees only one entry per branch. |
| Two genuinely independent `MATERIALIZED` CTE definitions in sibling scopes, same `cte_name` AND structurally identical body | *DO* merge. The merge key (`cte_name` + subquery hash) has no scope component, so this is indistinguishable from the bug-reproducing UNION-branch-clone shape the pass exists to fix, and is unavoidable without adding scope to the key. Only observable for a nondeterministic body (e.g. `rand()`), where it collapses two independent random draws into one shared value. Pinned by case 8 of `04507_materialized_cte_union_merge.sql`. |

## Testing

### ~~Original test plan~~ *(superseded)*

*Amended: none of the file name, prefix, or case list below match what was
actually implemented. Kept verbatim for historical context; see
"Testing (as implemented)" further down for the real test file and cases.*

New test file: `tests/queries/0_stateless/04045_materialized_cte_union_dedup.sql`
(use `./tests/queries/0_stateless/add-test 04045_materialized_cte_union_dedup`
to allocate the prefix).

1. **Bug-reproducing UNION** - `WITH x AS MATERIALIZED (SELECT FROM big_table) SELECT count() FROM x UNION ALL SELECT max() FROM x`. Assert correctness of the result, and assert single materialization via `EXPLAIN PIPELINE` containing exactly one `MaterializingCTE` step. Optionally cross-check via `system.query_log` that `ProfileEvents['SelectedRows']` is consistent with reading the source once.

2. **Body equal, column names differ** - `WITH x AS MATERIALIZED (SELECT 1 AS a), y AS MATERIALIZED (SELECT 1 AS b) SELECT * FROM x UNION ALL SELECT * FROM y`. Different column aliases produce different `getTreeHash` (because `compare_aliases = true`). Assert `EXPLAIN PIPELINE` contains two `MaterializingCTE` steps.

3. **Body equal, types differ** - `WITH x AS MATERIALIZED (SELECT toUInt8(1) AS a), y AS MATERIALIZED (SELECT toUInt64(1) AS a) ...`. `compare_types = true` keeps them separate. Two `MaterializingCTE` steps.

4. **Nested materialized CTE under UNION** - `WITH inner AS MATERIALIZED (SELECT FROM t), outer AS MATERIALIZED (SELECT FROM inner) SELECT FROM outer UNION ALL SELECT FROM outer`. Exercises the deepest-first ordering. Assert one materialization per logical CTE (two total: one for `inner`, one for `outer`).

5. **Different `cte_name`, identical body** - `WITH x AS MATERIALIZED (SELECT FROM big_table), y AS MATERIALIZED (SELECT FROM big_table) SELECT FROM x UNION ALL SELECT FROM y`. Confirm the `.ignore_cte = true` choice: pins that the two CTEs *are* deduped (one `MaterializingCTE` step in `EXPLAIN PIPELINE`). This is the deliberate side-effect documented in non-goals.

6. **UNION + materialized CTE inside `WHERE x IN (...)`** - the shape from the existing `04042_materialized_cte_union.sql`, but with an actual `UNION ALL`. Exercises the `PreparedSets` path that was the original symptom on this branch.

7. **Three-branch UNION ALL, same CTE referenced from all three** - confirm bucket size > 2 works.

8. **Distributed read** - run the same UNION shape against a `Distributed` table over two shards; assert correctness and assert (via `system.query_log` on the remote shards) that each shard also materializes the CTE only once.

9. **Negative: CTE referenced once total (no UNION)** - assert it still inlines (the analyzer's existing behavior).

### Testing (as implemented)

What actually landed is `tests/queries/0_stateless/04507_materialized_cte_union_merge.sql`
(the `04045` prefix above was not used - by the time this test was added,
`04045` had already been allocated elsewhere). Since `EXPLAIN` output
includes the random `temporary_table_name`, none of the merged cases pin
raw `EXPLAIN` text directly; instead every case counts `MaterializingCTE
(Materializing CTE:` lines in the plan. Cases, in file order:

1. Two-branch `UNION ALL`, same CTE, referenced once per branch: one
   `MaterializingCTE` step. A companion query uses `rand()` plus
   `uniqExact` to pin functionally that both branches observe the same
   materialization (not just that the plan shape looks right).
2. Three-branch `UNION ALL`, same CTE: still one `MaterializingCTE`
   step.
3. Two differently-named CTEs with an identical body, each used twice
   across a four-way `UNION ALL`: **two** `MaterializingCTE` steps - this
   is the amended replacement for the old test #5, now pinning that
   different names do **not** merge (see "Amendments" point 1 and the
   "Non-goals" note above).
4. Same CTE name, different body, in two sibling scopes (each
   parenthesized subquery defines and doubles its own `t`): two
   `MaterializingCTE` steps, plus a data assertion that each sibling's
   values stay independent.
5. Nested materialized CTE under `UNION ALL` (`outer_cte` references
   `inner_cte`, and `outer_cte` is referenced from both branches): two
   `MaterializingCTE` steps (one per logical CTE), pinning the
   inner-before-outer post-order invariant, plus a data assertion.
6. Single-use materialized CTE, no `UNION`: zero `MaterializingCTE`
   steps (still inlined).
7. One `UNION ALL` branch plus an `IN`-subquery branch referencing the
   same CTE: one `MaterializingCTE` step across all three usages, plus a
   data assertion.

The `04507_...` commit (`24e8fbc743d`) also regenerated
`04077_materialized_cte_union.reference`, whose old content had gone stale
for reasons unrelated to this change (master's switch of
`explain_query_plan_default` from `legacy` to `pretty`, plus a pre-existing
per-branch safety-net `MaterializingCTEs` planner step that now renders as
an empty wrapper once the union-level step claims the CTE).

## Risks

- **Aliasing or storage-locking surprises in `updateStorage`
  (via `TableNode::adoptMaterializedCTE`).** The primitive is exercised
  by the analyzer's existing in-branch repeat-reference path, so we know
  it works under at least that shape. The merge pass uses it more
  broadly. *Amended: cases 1, 2, and 7 of `04507_materialized_cte_union_merge.sql`
  exercise the new shape (two-branch UNION, three-branch UNION, and a
  UNION branch plus an `IN`-subquery branch); the distributed shape is
  covered separately by `04043_materialized_cte_serialize_query_plan`,
  not by a case in this file.*
- **Hash collisions.** Defended against by structural `isEqual` check
  inside each bucket.
- **`reused_materialized_cte` rebuild misses a case.** Pinned by the
  negative cases in `04507_materialized_cte_union_merge.sql` (case 4:
  same name, different body, in sibling scopes must not merge; case 6:
  a genuinely single-use CTE must still inline).
- ~~**Surprise dedup of same-body different-name CTEs.** Pinned by test
  #5. If this side-effect is judged unacceptable later, flip
  `.ignore_cte` to `false` in both the hash call and the `isEqual`
  call (one-line change each); the rest of the design is unaffected.~~
  *Amended: moot. The owner decision was reversed before implementation
  - `cte_name` is a required merge-key component, so same-body,
  different-name CTEs never merge in the first place. Pinned by case 3
  of `04507_materialized_cte_union_merge.sql`. There is no `.ignore_cte`
  escape hatch to flip because there is nothing to escape from.*
- **Non-definition-bearing `TableNode`s polluting merge/reuse
  counting.** A risk not anticipated by the original plan: shard-side
  re-analysis of a `Distributed` query can produce a `TableNode` that
  reports a non-null `getMaterializedCTE()` without being a CTE
  reference site (see the amended "Distributed query" edge case).
  Guarded by the `isMaterializedCTE()` check in `mergeDuplicateMaterializedCTEs`;
  regression-pinned by `04043_materialized_cte_serialize_query_plan`.

## Out of scope (future work)

- Stronger sharing for the cross-shard / distributed case (the current
  design re-runs dedup independently on each node; if a shape ever
  needs *shared* materialization across nodes, it would be a separate
  effort layered on top).
- A general analyzer pass that deduplicates structurally-equal
  subqueries beyond materialized CTEs - e.g., the same scalar
  subquery written twice. Out of scope; would need its own design.
