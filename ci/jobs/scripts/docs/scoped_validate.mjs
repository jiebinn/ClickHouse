#!/usr/bin/env node
// Scoped replacement for `mint validate`, for client repos that own one slice
// of the aggregated docs site (e.g. clickhouse-connect owns
// integrations/language-clients/python).
//
// `mint validate` runs Mintlify's prebuild over the WHOLE site: it MDX-parses
// every page, which takes ~13 minutes for a site this size, single-threaded.
// A client PR can only change the files inside its slice, and the aggregator
// repo's own CI still runs the full validate before anything deploys, so a
// client check only needs to prove:
//
//   1. docs.json itself is valid (schema check -- cheap, no page parsing).
//      The client cannot edit docs.json, but this catches a broken aggregator
//      snapshot early instead of producing confusing downstream errors.
//   2. Every docs.json navigation entry pointing INTO the slice resolves to a
//      file -- catches the client deleting or renaming a page that the
//      aggregator's navigation still references.
//   3. Every page inside the slice parses -- the same per-page processing
//      (frontmatter + MDX parse) that full prebuild runs, via Mintlify's own
//      `createPage`, so the errors are identical to what `mint validate`
//      would print for those files.
//   4. Every `import ... from '/snippets/...'` inside the slice resolves to a
//      file -- import resolution is otherwise a whole-site prebuild step.
//
// Site-wide link/anchor integrity is NOT checked here; the lychee check that
// runs alongside this script already covers it for the whole site.
//
// The Mintlify internals come from the `mint` CLI installed globally in the
// docs image (clickhouse/docs-builder) -- nothing is downloaded. The packages
// are pinned by whatever `mint` version the image ships, which is the same
// version the full check uses, so the two cannot drift apart.
//
// Usage: node scoped_validate.mjs --scope <dir-relative-to-docs-root> [--scope ...] [docs-root]
// Override package resolution for local testing: MINT_PACKAGES_DIR=<node_modules dir>

import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

// --- Locate Mintlify's packages inside the installed `mint` CLI ---------------
//
// `mint` is installed with `npm install -g mint` in the docs image, which
// flattens its dependencies under <npm root -g>/mint/node_modules. There is no
// `exports` map in these packages (plain `main` entries), so createRequire
// resolution works.
function importMintlifyPackage(name)
{
    const candidates = [];
    if (process.env.MINT_PACKAGES_DIR)
        candidates.push(process.env.MINT_PACKAGES_DIR);
    try
    {
        const npmRoot = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
        candidates.push(join(npmRoot, 'mint', 'node_modules'));
        candidates.push(npmRoot);
    }
    catch
    {
        // npm missing entirely -- fall through to the error below.
    }
    for (const dir of candidates)
    {
        try
        {
            const resolved = createRequire(join(dir, 'noop.js')).resolve(name);
            return import(pathToFileURL(resolved).href);
        }
        catch
        {
            // Not in this candidate dir -- try the next one.
        }
    }
    throw new Error(
        `Cannot resolve ${name}. Expected a global \`mint\` install ` +
        `(npm install -g mint) or MINT_PACKAGES_DIR pointing at a node_modules ` +
        `directory that contains it. Tried: ${candidates.join(', ') || '(nothing)'}`);
}

// --- Argument parsing ---------------------------------------------------------
function parseArgs(argv)
{
    const scopes = [];
    let docsRoot = '.';
    for (let i = 0; i < argv.length; i++)
    {
        if (argv[i] === '--scope')
        {
            const value = argv[++i];
            if (!value)
                throw new Error('--scope requires a value');
            scopes.push(value.replace(/^\/+|\/+$/g, ''));
        }
        else
        {
            docsRoot = argv[i];
        }
    }
    if (scopes.length === 0)
        throw new Error('At least one --scope is required');
    return { scopes, docsRoot: resolve(docsRoot) };
}

// --- Navigation walk ----------------------------------------------------------
//
// Collect every page reference (plain strings in `pages` arrays and `root`
// keys) from docs.json navigation. Subtrees carrying an `openapi` key are
// skipped: their page strings can refer to pages generated from the OpenAPI
// spec at build time, which have no source file.
function collectNavPages(node, acc)
{
    if (typeof node === 'string')
    {
        acc.push(node);
        return;
    }
    if (Array.isArray(node))
    {
        for (const item of node)
            collectNavPages(item, acc);
        return;
    }
    if (node && typeof node === 'object')
    {
        if ('openapi' in node)
            return;
        for (const [key, value] of Object.entries(node))
        {
            if (key === 'pages' || key === 'root')
                collectNavPages(value, acc);
            else if (typeof value === 'object' && value !== null)
                collectNavPages(value, acc);
        }
    }
}

function isInScope(pagePath, scopes)
{
    const cleaned = pagePath.replace(/^\/+/, '');
    return scopes.some(scope => cleaned === scope || cleaned.startsWith(scope + '/'));
}

// A navigation entry `foo/bar` resolves to foo/bar.mdx or foo/bar.md.
function navEntryExists(docsRoot, page)
{
    const cleaned = page.replace(/^\/+/, '');
    return ['.mdx', '.md'].some(ext => existsSync(join(docsRoot, cleaned + ext)));
}

// --- Page discovery -----------------------------------------------------------
async function listPages(dir)
{
    const out = [];
    for (const entry of await readdir(dir, { withFileTypes: true, recursive: true }))
    {
        if (entry.isFile() && /\.mdx?$/.test(entry.name))
            out.push(join(entry.parentPath, entry.name));
    }
    return out;
}

// Match top-of-file MDX ESM imports of shared snippets, e.g.:
//   import Thing from '/snippets/foo.mdx';
// Only absolute /snippets paths are checked: relative imports inside the scope
// are covered by the per-page parse, and anything else is not a docs snippet.
const SNIPPET_IMPORT_RE = /^import\s[^;'"]*['"](\/snippets\/[^'"]+)['"]/gm;

async function main()
{
    const { scopes, docsRoot } = parseArgs(process.argv.slice(2));
    const errors = [];

    const docsJsonPath = join(docsRoot, 'docs.json');
    if (!existsSync(docsJsonPath))
        throw new Error(`No docs.json in docs root: ${docsRoot}`);

    const [{ validateDocsConfig }, { createPage, getConfigObj }] = await Promise.all([
        importMintlifyPackage('@mintlify/validation'),
        importMintlifyPackage('@mintlify/prebuild'),
    ]);

    // 1. docs.json schema validation. getConfigObj is what full prebuild uses
    // to load the config: it resolves `$ref` includes (this docs.json pulls
    // its redirects and parts of its navigation from separate files) before
    // the schema sees it.
    const docsConfig = await getConfigObj(docsRoot, 'docs');
    const configResult = validateDocsConfig(docsConfig);
    if (!configResult.success)
    {
        for (const issue of configResult.error.issues)
            errors.push(`docs.json: ${issue.path.join('.')}: ${issue.message}`);
    }

    // 2. Navigation entries inside the scopes must resolve to files.
    const navPages = [];
    collectNavPages(docsConfig.navigation, navPages);
    const scopedNavPages = navPages.filter(page => isInScope(page, scopes));
    for (const page of scopedNavPages)
    {
        if (!navEntryExists(docsRoot, page))
            errors.push(
                `docs.json navigation references "${page}" but no such page exists ` +
                `under the scope (looked for ${page}.mdx and ${page}.md)`);
    }

    // 3 + 4. Parse every page in the scopes and check its snippet imports.
    let pageCount = 0;
    for (const scope of scopes)
    {
        const scopeDir = join(docsRoot, scope);
        if (isAbsolute(scope) || relative(docsRoot, scopeDir).startsWith('..' + sep))
            throw new Error(`--scope must be a relative path inside the docs root: ${scope}`);
        if (!existsSync(scopeDir))
        {
            errors.push(`Scope directory does not exist in the docs root: ${scope}`);
            continue;
        }
        const pages = await listPages(scopeDir);
        pageCount += pages.length;
        await Promise.all(pages.map(async (absPath) =>
        {
            const relPath = relative(docsRoot, absPath);
            const content = await readFile(absPath, 'utf8');
            // The same per-page processing full prebuild runs: frontmatter
            // extraction plus the MDX parse. Parse failures arrive via onError
            // with the same formatted message `mint validate` prints.
            try
            {
                await createPage(relPath, content, docsRoot, [], [], message => errors.push(message.trim()));
            }
            catch (error)
            {
                errors.push(`${relPath}: ${error.message}`);
            }
            for (const match of content.matchAll(SNIPPET_IMPORT_RE))
            {
                const snippetPath = match[1];
                if (!existsSync(join(docsRoot, snippetPath.replace(/^\/+/, ''))))
                    errors.push(`${relPath}: imports "${snippetPath}" but the file does not exist`);
            }
        }));
    }

    console.log(
        `Scoped validate: checked docs.json, ${scopedNavPages.length} in-scope navigation ` +
        `entries, and ${pageCount} pages under: ${scopes.join(', ')}`);
    if (errors.length > 0)
    {
        console.error(`\n${errors.length} error(s):`);
        for (const error of errors)
            console.error(`  - ${error}`);
        process.exit(1);
    }
    console.log('scoped validation passed');
}

main().catch(error =>
{
    console.error(error.message);
    process.exit(1);
});