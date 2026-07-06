#!/usr/bin/env node
// Scoped replacement for `mint validate`, for client repos that own one slice
// of the aggregated docs site (e.g. clickhouse-connect owns
// integrations/language-clients/python). Full validate MDX-parses every page
// of the site (~13 minutes, single-threaded); this checks only what a client
// PR can break, in seconds:
//
//   1. docs.json passes Mintlify's schema validation.
//   2. Navigation entries pointing into the slice resolve to files.
//   3. Every page inside the slice parses (via Mintlify's own `createPage`,
//      so the errors match what `mint validate` would print).
//   4. Imports in the slice resolve, via Mintlify's own import pipeline
//      (`findAndRemoveImports` + `resolveAllImports`): path validity, file
//      existence, and named-export resolution, followed transitively through
//      imported snippets.
//
// Site-wide link/anchor integrity is left to the lychee check that runs
// alongside, and the aggregator's own CI still runs the full validate.
//
// Usage: node scoped_validate.mjs --scope <dir-relative-to-docs-root> [--scope ...] [docs-root]
// MINT_PACKAGES_DIR overrides package resolution (for testing outside the docs image).

import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

// The Mintlify packages come from the `mint` CLI installed globally in the
// docs image (`npm install -g mint` flattens them under
// <npm root -g>/mint/node_modules), so they cannot drift from the version the
// full check uses. They have plain `main` entries, no `exports` maps, so
// createRequire resolution works.
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
        }
    }
    throw new Error(
        `Cannot resolve ${name}. Expected a global \`mint\` install ` +
        `(npm install -g mint) or MINT_PACKAGES_DIR pointing at a node_modules ` +
        `directory that contains it. Tried: ${candidates.join(', ') || '(nothing)'}`);
}

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

// Collect page references (strings in `pages` arrays and `root` keys) from the
// navigation. Subtrees with an `openapi` key are skipped: their page strings
// can refer to pages generated from the spec, which have no source file.
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

function navEntryExists(docsRoot, page)
{
    const cleaned = page.replace(/^\/+/, '');
    return ['.mdx', '.md'].some(ext => existsSync(join(docsRoot, cleaned + ext)));
}

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

async function main()
{
    const { scopes, docsRoot } = parseArgs(process.argv.slice(2));
    const errors = [];

    const docsJsonPath = join(docsRoot, 'docs.json');
    if (!existsSync(docsJsonPath))
        throw new Error(`No docs.json in docs root: ${docsRoot}`);

    const [{ validateDocsConfig }, { createPage, getConfigObj, preparseMdxTree }, common] = await Promise.all([
        importMintlifyPackage('@mintlify/validation'),
        importMintlifyPackage('@mintlify/prebuild'),
        importMintlifyPackage('@mintlify/common'),
    ]);
    const { findAndRemoveImports, hasImports, resolveAllImports, resolveImportPath,
            isSnippetExtension, topologicalSort } = common;

    // getConfigObj resolves `$ref` includes (the redirects map, per-slice
    // navigation fragments) before the schema sees the config.
    const docsConfig = await getConfigObj(docsRoot, 'docs');
    const configResult = validateDocsConfig(docsConfig);
    if (!configResult.success)
    {
        for (const issue of configResult.error.issues)
            errors.push(`docs.json: ${issue.path.join('.')}: ${issue.message}`);
    }

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

    // The import graph reachable from the scope pages. Filenames use the same
    // leading-slash docs-root-relative form full prebuild uses, which is what
    // resolveAllImports matches import paths against. A missing entry is left
    // out so resolveAllImports reports it as a missing file, like full validate.
    const importedEntries = new Map();
    async function loadImportedFile(filename)
    {
        if (importedEntries.has(filename))
            return;
        importedEntries.set(filename, null);
        const absPath = join(docsRoot, filename);
        if (!isSnippetExtension(extname(filename).toLowerCase()) || !existsSync(absPath))
            return;
        const content = await readFile(absPath, 'utf8');
        const tree = await preparseMdxTree(content, docsRoot, absPath, message => errors.push(message.trim()));
        if (!tree)
            return;
        const entry = { ...(await findAndRemoveImports(tree)), filename };
        importedEntries.set(filename, entry);
        for (const source of Object.keys(entry.importMap))
        {
            const resolved = resolveImportPath(source, filename);
            if (resolved)
                await loadImportedFile(resolved);
        }
    }

    const pagesWithImports = [];
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
            try
            {
                await createPage(relPath, content, docsRoot, [], [], message => errors.push(message.trim()));
            }
            catch (error)
            {
                errors.push(`${relPath}: ${error.message}`);
            }
            // Parse errors were already reported by createPage above.
            const tree = await preparseMdxTree(content, docsRoot, absPath, () => {});
            if (!tree)
                return;
            const fileWithImports = { ...(await findAndRemoveImports(tree)), filename: '/' + relPath };
            if (hasImports(fileWithImports))
                pagesWithImports.push(fileWithImports);
        }));
    }

    for (const page of pagesWithImports)
        for (const source of Object.keys(page.importMap))
        {
            const resolved = resolveImportPath(source, page.filename);
            if (resolved)
                await loadImportedFile(resolved);
        }

    // Resolve imports the way full prebuild does (resolveImportsAndWriteFiles):
    // snippets first, in reverse topological order so nested imports are
    // already inlined, then the pages. Every problem -- invalid path, missing
    // file, unresolvable named export -- arrives via onWarning, which strict
    // mode turns into failures.
    const snippetEntries = [...importedEntries.values()].filter(entry => entry !== null);
    const graph = {};
    for (const entry of snippetEntries)
        graph[entry.filename] = Object.keys(entry.importMap)
            .map(source => resolveImportPath(source, entry.filename))
            .filter(resolved => resolved !== null);
    const onWarning = warning => errors.push(warning.message);
    for (const filename of topologicalSort(graph).reverse())
    {
        const entry = importedEntries.get(filename);
        if (entry && hasImports(entry))
            entry.tree = await resolveAllImports({ snippets: snippetEntries, fileWithImports: entry, onWarning });
    }
    for (const page of pagesWithImports)
        await resolveAllImports({ snippets: snippetEntries, fileWithImports: page, onWarning });

    console.log(
        `Scoped validate: checked docs.json, ${scopedNavPages.length} in-scope navigation ` +
        `entries, ${pageCount} pages, and ${snippetEntries.length} imported files ` +
        `under: ${scopes.join(', ')}`);
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