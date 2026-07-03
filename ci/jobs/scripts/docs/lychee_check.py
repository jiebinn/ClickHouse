#!/usr/bin/env python3
"""
Link checks for the Mintlify docs, powered by lychee. Replaces ``mint
broken-links``.

Run from the docs root (the directory holding ``docs.json`` and ``lychee.toml``)
with one of three modes:

  --mode links      Internal links and heading anchors, offline. Blocking.
  --mode redirects  Every destination in ``_site/redirects.json`` resolves to a
                    real page (and anchor, if any), offline. Blocking.
  --mode external   External http(s) URLs, online. Non-blocking: reports broken
                    external links as warnings and always exits 0, because the
                    result depends on third-party sites being reachable.

lychee loads ``lychee.toml`` for shared configuration -- notably the
``exclude_path`` list that scopes the check to the Mintlify site and drops
legacy and generated content.

The ``links`` and ``redirects`` modes run against a throwaway copy of the docs
so the source tree is never modified, applying one Mintlify-specific transform:
heading anchors declared as ``## Title {#anchor}`` are rewritten to an explicit
``<a id="anchor"></a>``. lychee's markdown parser does not understand the
``{#anchor}`` syntax, and headings nested in JSX components (``<Steps>``,
``<Tabs>``, ...) are not parsed as headings at all, so without this their
anchors are invisible to fragment checking. The ``<a id>`` form is extracted by
lychee even inside JSX. The copy holds only the files lychee actually checks
(from ``lychee --dump-inputs``, which honours ``lychee.toml``), so the large
image and generated-translation trees under ``docs/`` are never copied.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# A heading line ending in a Mintlify `{#anchor}`. Captures the heading up to
# the anchor and the anchor id; the replacement appends an HTML anchor lychee
# can see regardless of surrounding JSX. Only horizontal whitespace ([ \t]) is
# matched around the anchor -- using `\s` would let the trailing quantifier eat
# the newline(s) after the heading, collapsing the blank line that separates it
# from following content and shifting markdown block boundaries (e.g. pushing a
# connection string out of its fenced code block).
HEADING_ANCHOR = re.compile(
    r"^([ \t]{0,3}#{1,6}[ \t].*?)[ \t]*\{#([A-Za-z0-9_-]+)\}[ \t]*$", re.MULTILINE
)


def transform_anchors(text):
    return HEADING_ANCHOR.sub(r'\1 <a id="\2"></a>', text)


# MDX comments `{/* ... */}` are not rendered, so any links inside them (e.g. a
# commented-out `[text](tbd) -- add link once published` placeholder) are not
# real links. Strip them before checking. Newlines in multi-line comments are
# preserved so error line numbers stay roughly accurate.
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)


def strip_mdx_comments(text):
    return MDX_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


# Code blocks -- fenced (``` ... ```) and inline (`code`). URLs inside code are
# examples (docker images, sample configs, connection strings), not real links.
FENCED_CODE_RE = re.compile(r"^[ \t]{0,3}```.*?^[ \t]{0,3}```[^\n]*$", re.DOTALL | re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def strip_code_blocks(text):
    text = FENCED_CODE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return INLINE_CODE_RE.sub("", text)


# `import Foo from '/snippets/x.mdx'` (or a relative path). Mintlify renders the
# snippet's body inline where the component is used, so a heading anchor defined
# in the snippet resolves on the page -- but lychee never expands imports.
IMPORT_RE = re.compile(r"^import\s+\w+\s+from\s+['\"]([^'\"]+\.mdx?)['\"]", re.MULTILINE)
# Any anchor id a snippet contributes: a Mintlify `{#id}` heading or an `<a id>`.
ANCHOR_ID_RE = re.compile(r'\{#([A-Za-z0-9_-]+)\}|<a\s+id=["\']([A-Za-z0-9_-]+)["\']')
# Any element id, e.g. `<div id="mcp-setup">`. Browsers and Mintlify treat these
# as valid fragment targets, but lychee only recognizes headings and `<a id>`.
ELEMENT_ID_RE = re.compile(r'\bid=["\']([A-Za-z0-9_-]+)["\']')


def collect_snippet_anchors(text, docs_root, page_dir, seen):
    # Anchor ids reachable through the page's snippet imports (recursively, so a
    # snippet that imports another snippet still contributes its anchors). Reads
    # the original snippet files; `seen` guards against import cycles.
    ids = set()
    for imp in IMPORT_RE.findall(text):
        sp = os.path.abspath(
            os.path.join(docs_root, imp.lstrip("/")) if imp.startswith("/")
            else os.path.join(page_dir, imp)
        )
        if sp in seen or not os.path.isfile(sp):
            continue
        seen.add(sp)
        with open(sp, "r", encoding="utf-8", errors="replace") as f:
            snip = f.read()
        ids.update(m.group(1) or m.group(2) for m in ANCHOR_ID_RE.finditer(snip))
        ids |= collect_snippet_anchors(snip, docs_root, os.path.dirname(sp), seen)
    return ids


def dump_inputs(docs_root):
    # Ask lychee itself which files it would check, so the copy honours the
    # exclude_path scoping in lychee.toml without duplicating it here.
    out = subprocess.run(
        ["lychee", "--dump-inputs", "."],
        cwd=docs_root, check=True, capture_output=True, text=True,
    ).stdout
    paths = []
    for line in out.splitlines():
        line = line.strip()
        # Keep on-disk files under the docs root; ignore any remote inputs.
        if line.startswith("./") and os.path.isfile(os.path.join(docs_root, line)):
            paths.append(line[2:])
    return paths


# Locale prefixes whose redirect sources are irrelevant to the English site.
LOCALE_PREFIXES = {"ar", "es", "fr", "ja", "jp", "ko", "pt-BR", "ru", "zh"}


def materialize_redirects(docs_root, dest):
    # Mintlify serves every `source` in redirects.json via a redirect, so a link
    # to a redirected path resolves on the site. lychee can't follow redirects
    # offline, so drop an empty placeholder at each (English) redirect source
    # that has no real page -- enough for lychee's file-existence check, matching
    # Mintlify's "file or redirect" resolution.
    redirects_json = os.path.join(docs_root, "_site", "redirects.json")
    if not os.path.isfile(redirects_json):
        return
    with open(redirects_json) as f:
        redirects = json.load(f)
    for r in redirects:
        src = (r.get("source") or "").strip().lstrip("/")
        if not src:
            continue
        if any(os.path.exists(os.path.join(dest, src + e)) for e in ("", ".mdx", ".md")):
            continue  # a real page already covers this path
        # For English sources, seed the placeholder with the destination's anchor
        # ids (only `<a id>` tags, not its content, so we don't re-check the
        # destination's own links) -- this makes a fragment link to the redirect
        # source (e.g. .../oss#install-clickhouse) resolve, since Mintlify applies
        # the fragment on the destination. Locale sources get an empty placeholder
        # (their destinations are checked for page-existence only; see below).
        anchors = set()
        if src.split("/")[0] not in LOCALE_PREFIXES:
            dest_url = (r.get("destination") or "").strip()
            if dest_url.startswith("/"):
                for e in (".mdx", ".md"):
                    cand = os.path.join(dest, dest_url.lstrip("/") + e)
                    if os.path.isfile(cand):
                        with open(cand, encoding="utf-8", errors="replace") as f:
                            anchors = {m.group(1) or m.group(2)
                                       for m in ANCHOR_ID_RE.finditer(f.read())}
                        break
        p = os.path.join(dest, src + ".mdx")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("".join(f'<a id="{a}"></a>\n' for a in sorted(anchors)))


def build_tree(docs_root, dest):
    # Build a throwaway mirror of the docs tree in `dest`, a fresh temp dir.
    #
    # Safety: the source tree must never be modified. We therefore only ever
    # create brand-new files inside `dest` and never create a symlink or
    # hardlink -- writing through such a link would corrupt the real docs. The
    # markdown files lychee checks are written as real, anchor-rewritten copies;
    # every other path (images, data files, non-checked docs) becomes an empty
    # placeholder, which is enough for lychee's local link check because it only
    # tests that the target file exists, not its contents. This also avoids
    # copying the hundreds of MB of images and generated translations.
    checked = set(dump_inputs(docs_root))
    for root, _dirs, files in os.walk(docs_root):
        rel_dir = os.path.relpath(root, docs_root)
        out_dir = dest if rel_dir == "." else os.path.join(dest, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            dst = os.path.join(out_dir, name)
            if rel in checked and name.endswith((".md", ".mdx")):
                with open(os.path.join(root, name), "r",
                          encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                text = strip_mdx_comments(transform_anchors(raw))
                # Append anchors the page inherits from imported snippets, which
                # Mintlify renders inline but lychee cannot see across the import.
                anchors = collect_snippet_anchors(raw, docs_root, root, set())
                # Non-<a> element ids (e.g. <div id="...">) are valid fragment
                # targets too, but lychee doesn't extract them -- add them here.
                anchors |= {m.group(1) for m in ELEMENT_ID_RE.finditer(raw)}
                if anchors:
                    text += "\n\n" + "".join(
                        f'<a id="{a}"></a>\n' for a in sorted(anchors)
                    )
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(text)
            elif name.endswith((".md", ".mdx")):
                # Not checked as an input (locale/legacy content), but it can be a
                # link/redirect *target* -- so expose its heading anchors with a
                # light transform (no snippet/comment handling) so fragment links
                # to it (e.g. locale redirect destinations) resolve.
                with open(os.path.join(root, name), "r",
                          encoding="utf-8", errors="replace") as f:
                    out = transform_anchors(f.read())
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(out)
            else:
                # Empty placeholder: only its existence matters to lychee.
                open(dst, "w").close()
    # Placeholders for redirect sources so links to redirected paths resolve.
    materialize_redirects(docs_root, dest)
    # lychee reads its configuration from the working directory.
    cfg = os.path.join(docs_root, "lychee.toml")
    if os.path.isfile(cfg):
        with open(cfg) as fin, open(os.path.join(dest, "lychee.toml"), "w") as fout:
            fout.write(fin.read())


def write_redirects(docs_root, dest):
    # _site/redirects.json is produced by the Mintlify build. Emit each
    # destination as a markdown link so lychee resolves it against the docs tree.
    redirects_json = os.path.join(docs_root, "_site", "redirects.json")
    if not os.path.isfile(redirects_json):
        raise FileNotFoundError(
            f"Expected redirects at {redirects_json}; build the docs first."
        )
    with open(redirects_json) as f:
        redirects = json.load(f)
    out = os.path.join(dest, "_lychee_redirects.md")
    with open(out, "w") as f:
        f.write("# Redirect destinations\n\n")
        for r in redirects:
            dest_url = (r.get("destination") or "").strip()
            if not dest_url:
                continue
            # Skip only what we genuinely can't verify offline: external URLs and
            # dynamic paths (`:path*` wildcards).
            if dest_url.startswith(("http://", "https://")) or ":" in dest_url:
                continue
            # Locale destinations are verified for page existence only -- drop the
            # fragment, since auto-generated translations legitimately lag the
            # English source on heading anchors.
            if dest_url.lstrip("/").split("/")[0] in LOCALE_PREFIXES:
                dest_url = dest_url.split("#")[0]
            f.write(f"- [{dest_url}]({dest_url})\n")
    return "_lychee_redirects.md"


# The throwaway tree's absolute path that lychee prints in front of every local
# target, e.g. "file:///private/var/.../lychee-links-ab12cd/". Strip it so a
# target reads as a repo-relative path like "reference/settings/x#anchor".
TREE_PREFIX = re.compile(r"file://[^\s)]*?/lychee-[a-z]+-[A-Za-z0-9_]+/")


def run_lychee(cmd, cwd):
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(TREE_PREFIX.sub("", proc.stdout), end="", flush=True)
    return proc.returncode


def check_links(docs_root):
    dest = tempfile.mkdtemp(prefix="lychee-links-")
    build_tree(docs_root, dest)
    return run_lychee(
        ["lychee", "--mode", "color", "--offline", "--include-fragments", "."], dest
    )


def check_redirects(docs_root):
    dest = tempfile.mkdtemp(prefix="lychee-redirects-")
    build_tree(docs_root, dest)
    redirects_md = write_redirects(docs_root, dest)
    return run_lychee(
        ["lychee", "--mode", "color", "--offline", "--include-fragments", redirects_md],
        dest,
    )


def check_external(docs_root):
    # Restrict to http(s) so only external URLs are checked; internal file links
    # are covered by --mode links. Non-blocking: network reachability is not a
    # property of the docs, so surface failures as warnings and exit 0.
    #
    # Run against a copy of the checked pages with code blocks stripped, so
    # example URLs inside fenced/inline code (docker images, sample configs, ...)
    # are not treated as real external links.
    dest = tempfile.mkdtemp(prefix="lychee-external-")
    for rel in dump_inputs(docs_root):
        if not rel.endswith((".md", ".mdx")):
            continue
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(os.path.join(docs_root, rel), encoding="utf-8", errors="replace") as f:
            content = strip_code_blocks(f.read())
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
    cfg = os.path.join(docs_root, "lychee.toml")
    if os.path.isfile(cfg):
        with open(cfg) as fin, open(os.path.join(dest, "lychee.toml"), "w") as fout:
            fout.write(fin.read())
    rc = run_lychee(
        ["lychee", "--scheme", "http", "--scheme", "https", "."], dest,
    )
    if rc != 0:
        print(
            f"\nlychee reported broken external links (exit {rc}); "
            f"treated as a warning, not failing the check.",
            flush=True,
        )
    return 0


MODES = {
    "links": check_links,
    "redirects": check_redirects,
    "external": check_external,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument("docs_root", nargs="?", default=".",
                   help="Docs root with docs.json and lychee.toml (default: .).")
    args = p.parse_args(argv)
    return MODES[args.mode](os.path.abspath(args.docs_root))


if __name__ == "__main__":
    sys.exit(main())