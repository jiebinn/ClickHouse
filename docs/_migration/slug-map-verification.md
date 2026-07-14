# Slug map verification

Verified on 2026-07-14 against:

- Current docs: `https://clickhouse.com/docs`
- Mintlify preview: `https://private-7c7dfe99.mintlify.app`
- Mapping file: `_migration/slug-map.csv`
- Redirects: `_site/redirects.json`

## Result

The current slug map is **not yet a complete 1:1 mapping**.

| Check | Result |
| --- | ---: |
| CSV rows | 2,000 |
| Current sitemap URLs | 2,057 |
| Current sitemap URLs represented in the CSV | 1,997 |
| Current sitemap URLs absent from the CSV | 60 |
| Content pages absent from the CSV | 14 |
| Generated index/search/tag pages absent from the CSV | 46 |
| CSV destination URLs tested | 1,943 |
| CSV destinations returning 200 | 1,942 |
| CSV destinations returning 404 | 1 |
| Legacy CSV slugs tested on Mintlify | 2,000 |
| Legacy slugs resolving successfully | 1,972 |
| Legacy slugs returning 404 | 28 |
| Ambiguous CSV rows | 2 |
| Duplicate Mintlify destinations | 11 |
| Source/migrated hashes equal | 1,815 |
| Source/migrated hashes differ | 59 |
| Rows without a migrated hash | 126 |

The initial legacy-route pass observed one transient `500` for
`/sql-reference/functions`. An immediate retry resolved to
`/reference/functions` with a `200`, so it is not counted as a persistent
failure.

## Triage plan

Work through these themes in order. Each theme is independently reviewable and
has a concrete completion condition.

| Order | Priority | Theme | Scope | Definition of done |
| ---: | --- | --- | ---: | --- |
| 1 | P0 | Missing redirects for known destinations | 26 routes | Every legacy route redirects to its recorded destination and returns `200` on the preview. |
| 2 | P0 | Current content absent from the CSV | 14 pages | Every current sitemap content page has one reviewed Mintlify destination, CSV row, and legacy redirect. Missing content is migrated where no equivalent exists. |
| 3 | P0 | Unresolved or stale mappings | 5 rows | The two targetless AI/ML rows, stale `/about-us/cloud` destination, and two ambiguous rows each have one authoritative destination recorded in the CSV. |
| 4 | P1 | Generated route preservation | 46 routes | Archive, pagination, tag, category, and search routes have a documented keep/redirect/drop policy; every kept or redirected route resolves successfully. |
| 5 | P1 | Many-to-one destination review | 11 pairs | Each duplicate destination is confirmed as an intentional alias or split into distinct destinations after content comparison. |
| 6 | P2 | Content-hash and audit metadata | 185 rows | All 59 hash mismatches are reviewed and all 126 missing migrated hashes are either populated or explicitly exempted. |

Theme 1 is deliberately first because all destinations already exist and
return `200`; the required changes are mechanical and carry little editorial
risk. Themes 2 and 3 require page-by-page content or product decisions. Theme 4
is mostly an SEO policy decision. Themes 5 and 6 validate semantic parity after
route coverage is complete.

### Theme 1 implementation status

Implemented locally on `codex/verify-slug-map`:

- Added all 26 missing legacy redirects to `_site/redirects.json`.
- Confirmed every source is unique in the redirect registry.
- Confirmed all 26 destinations resolve to files in the Mintlify source tree.
- Ran the full offline redirect validator: 12,878 links checked with zero
  errors.

The historical preview results below continue to describe the deployed
preview at verification time. The 26 legacy routes should be retested against
a preview built from this branch before Theme 1 is marked complete.

### Theme 3 implementation status

Four of the five unresolved or stale routes now have approved destinations:

- Both uppercase `/cloud/get-started/cloud/use-cases/AI_ML` routes redirect to
  `/get-started/use-cases/agentic-analytics`. The older machine-learning data
  layer article will not be migrated.
- `/about-us/cloud` already redirects to
  `/products/cloud/getting-started/intro`. Its CSV destination has been
  corrected, and the stale `/resources/about/cloud` path now redirects there
  as well.
- `/interfaces/cpp` already redirects to
  `/integrations/language-clients/cpp`.

`/operations/overview` remains the only unresolved row in this theme. Its
runtime redirect currently lands on
`/guides/clickhouse/performance-and-monitoring/query-optimization`.

## Current content pages absent from the CSV

These are current, sitemap-listed content pages rather than generated index,
pagination, tag, or search pages:

1. `/chdb/guides/python-udf`
2. `/cloud/guides/security/migrating-rbac-custom-roles`
3. `/cloud/manage/clickstack`
4. `/cloud/reference/byoc/reference/security-shared-responsibility`
5. `/cloud/security/scim-setup-entra`
6. `/integrations/data-catalogs`
7. `/integrations/estuary/clickpipes`
8. `/integrations/estuary/native`
9. `/integrations/tigris`
10. `/knowledgebase/aws-privatelink-vpc-endpoint-service-for-msk-cluster`
11. `/knowledgebase/confluent-cloud-private-connectivity-for-clickpipes`
12. `/sql-reference/table-functions/eval`
13. `/use-cases/observability/clickstack/demo-days/2026/2026-06-18`
14. `/use-cases/observability/clickstack/demo-days/2026/2026-06-26`

The two knowledge-base destination files already exist in the Mintlify tree,
but neither current slug has a CSV row or legacy redirect. Some of the other
pages may also correspond to newer or consolidated Mintlify content; they need
explicit mapping decisions rather than being silently omitted.

## Generated current routes absent from the CSV

The other 46 sitemap URLs are generated navigation surfaces:

- `/category/changelog`
- `/search`
- `/knowledgebase`
- `/knowledgebase/archive`
- 12 `/knowledgebase/page/*` pagination routes
- `/knowledgebase/tags`
- 26 `/knowledgebase/tags/*` tag routes
- 3 `/knowledgebase/tags/*/page/*` pagination routes

Resolved locally: `/search` is now a standalone page with an inline Inkeep
search that uses the same result corpus as the global site search. The other
45 generated routes redirect to the documentation home page.

## Broken CSV destination at verification time

The only recorded Mintlify destination returning `404` is:

| Current slug | CSV destination |
| --- | --- |
| `/about-us/cloud` | `/resources/about/cloud` |

The mapped file `resources/about/cloud.mdx` no longer exists. The legacy slug
currently redirects successfully to `/products/cloud/getting-started/intro`,
so the CSV is stale even though the old route remains usable.

Resolved locally: the CSV now records
`products/cloud/getting-started/intro.mdx`, and `/resources/about/cloud`
redirects to `/products/cloud/getting-started/intro`.

## Legacy slugs returning 404 on Mintlify

Two rows have no destination and no working redirect:

1. `/cloud/get-started/cloud/use-cases/AI_ML`
2. `/cloud/get-started/cloud/use-cases/AI_ML/agent_facing_analytics`

The following 26 rows have working recorded destinations, but the legacy slug
itself returns `404`, indicating a missing redirect:

1. `/engines/table-engines/special/query-runner`
2. `/getting-started/example-datasets/job`
3. `/integrations/airflow`
4. `/interfaces/documentation-search`
5. `/interfaces/web-sql`
6. `/interfaces/web-ui-color-coding`
7. `/operations/system-tables/hypothetical_indexes`
8. `/operations/system-tables/stemmers`
9. `/sql-reference/aggregate-functions/reference/groupformat`
10. `/sql-reference/aggregate-functions/reference/quantilesBFloat16`
11. `/sql-reference/aggregate-functions/reference/quantilesBFloat16Weighted`
12. `/sql-reference/aggregate-functions/reference/quantilesDD`
13. `/sql-reference/aggregate-functions/reference/quantilesDeterministic`
14. `/sql-reference/aggregate-functions/reference/quantilesExact`
15. `/sql-reference/aggregate-functions/reference/quantilesExactHigh`
16. `/sql-reference/aggregate-functions/reference/quantilesExactLow`
17. `/sql-reference/aggregate-functions/reference/quantilesExactWeighted`
18. `/sql-reference/aggregate-functions/reference/quantilesExactWeightedInterpolated`
19. `/sql-reference/aggregate-functions/reference/quantilesInterpolatedWeighted`
20. `/sql-reference/aggregate-functions/reference/quantilesPrometheusHistogram`
21. `/sql-reference/aggregate-functions/reference/quantilesTDigest`
22. `/sql-reference/aggregate-functions/reference/quantilesTDigestWeighted`
23. `/sql-reference/aggregate-functions/reference/quantilesTiming`
24. `/sql-reference/statements/hypothetical-index`
25. `/sql-reference/window-functions/ntile`
26. `/use-cases/observability/clickstack/demo-days/2026/2026-06-12`

A browser check confirmed the behavior for
`/interfaces/documentation-search`: the current site renders the expected
article, the legacy path on Mintlify renders `Page Not Found`, and the recorded
destination `/concepts/features/interfaces/documentation-search` renders the
correct article.

## Ambiguous rows

The CSV still contains two rows with multiple candidate files and no
`new_url`:

| Current slug | Candidate files | Runtime destination |
| --- | --- | --- |
| `/interfaces/cpp` | `concepts/features/interfaces/cpp.mdx`; `integrations/language-clients/cpp.mdx` | `/integrations/language-clients/cpp` |
| `/operations/overview` | `concepts/features/performance/index.mdx`; `products/cloud/guides/best-practices/index.mdx` | `/guides/clickhouse/performance-and-monitoring/query-optimization` |

Both legacy routes currently resolve successfully, but the CSV itself does not
record the selected destination and therefore cannot serve as the authoritative
1:1 migration record.

## Duplicate destinations

Eleven pairs of distinct current slugs map to one Mintlify destination. These
look like FAQ/knowledge-base aliases, but their source hashes differ, so they
need an explicit content-equivalence decision before they can be accepted as
intentional many-to-one mappings.

| Mintlify destination | Current slugs |
| --- | --- |
| `/resources/support-center/knowledge-base/cloud-services/multi-region-replication` | `/faq/operations/multi-region-replication`; `/knowledgebase/multi-region-replication` |
| `/resources/support-center/knowledge-base/data-import-export/json-import` | `/faq/integration/json-import`; `/knowledgebase/json-import` |
| `/resources/support-center/knowledge-base/general-faqs/dbms-naming` | `/faq/general/dbms-naming`; `/knowledgebase/dbms-naming` |
| `/resources/support-center/knowledge-base/general-faqs/key-value` | `/faq/use-cases/key-value`; `/knowledgebase/key-value` |
| `/resources/support-center/knowledge-base/general-faqs/mapreduce` | `/faq/general/mapreduce`; `/knowledgebase/mapreduce` |
| `/resources/support-center/knowledge-base/general-faqs/olap` | `/faq/general/olap`; `/knowledgebase/olap` |
| `/resources/support-center/knowledge-base/general-faqs/time-series` | `/faq/use-cases/time-series`; `/knowledgebase/time-series` |
| `/resources/support-center/knowledge-base/general-faqs/who-is-using-clickhouse` | `/faq/general/who-is-using-clickhouse`; `/knowledgebase/who-is-using-clickhouse` |
| `/resources/support-center/knowledge-base/integrations/oracle-odbc` | `/faq/integration/oracle-odbc`; `/knowledgebase/oracle-odbc` |
| `/resources/support-center/knowledge-base/setup-installation/production` | `/faq/operations/production`; `/knowledgebase/production` |
| `/resources/support-center/knowledge-base/tables-schema/delete-old-data` | `/faq/operations/delete-old-data`; `/knowledgebase/delete-old-data` |

## Verification method

1. Compared all normalized `<loc>` entries in the current docs sitemap with
   every `old_url` in `slug-map.csv`.
2. Checked CSV uniqueness for old slugs, old URLs, Mintlify files, and new URLs.
3. Verified every mapped local destination file exists.
4. Requested all 1,943 recorded `new_url` values and followed redirects to the
   final HTTP response.
5. Requested all 2,000 legacy slugs against the Mintlify preview and followed
   redirects to the final HTTP response.
6. Compared source and migrated hash coverage from the CSV.
7. Opened the current site and Mintlify preview side by side and visually
   inspected a representative broken route and its working destination.

The HTTP tests used GET requests, a 30-second per-request timeout, and up to 40
concurrent workers. A non-200 result was retried individually when it appeared
transient.
