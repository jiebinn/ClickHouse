#!/usr/bin/env python3
"""Validate navigation links inside localized components and page data.

The published locale pages render JSX components (QuickStartsGrid,
SampleDatasetExplorer, KBExplorer, ...) and MDX `export const` data whose card
navigation lives in `href:`/`to:` string literals -- not markdown links. lychee
neither sees `snippets/<locale>/...` nor parses JSX/JS, so `--mode locale-links`
cannot catch when a localized component routes users to the wrong place.

For every localized file (under `<locale>/` and `snippets/<locale>/`), extract
internal `href`/`to` paths and check:

  * a path already under `/<locale>/` must resolve to a real page/redirect;
  * an unprefixed path (e.g. `/get-started/...`) whose localized counterpart
    `/<locale>/...` EXISTS is a regression -- the localized surface should link
    to the localized page, not send readers to English. (If no localized
    counterpart exists, the English path is an acceptable fallback.)

`--fix` rewrites those regressions to the localized path. Without it, the script
only reports and exits non-zero when violations remain.
"""
import argparse
import json
import os
import re
import sys

LOCALE_DIRS = ["ar", "es", "fr", "ja", "ko", "pt-BR", "ru", "zh"]
EXTS = (".mdx", ".md", ".jsx", ".tsx", ".js")
# Non-page asset/base paths that are legitimately unprefixed. `/docs/` is the
# production mount some shared components hardcode (identically in English), not
# a repo-relative doc path, so it is out of scope for the locale check.
SKIP_PREFIXES = ("/images/", "/assets/", "/_site/", "/.well-known/", "/docs/")
SKIP_EXACT = {"/docs", "/"}
# `href: "/x"`, `href="/x"`, `href={'/x'}`, `to: "/x"`, ...
HREF = re.compile(r"""\b(?:href|to)\s*[:=]\s*\{?\s*(['"`])(/[^'"`\s]+)\1""")


def build_targets(docs_root):
    pages = set()
    for root, dirs, files in os.walk(docs_root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
        for n in files:
            if n.endswith((".mdx", ".md")):
                rel = os.path.relpath(os.path.join(root, n), docs_root)
                pages.add(re.sub(r"\.mdx?$", "", rel))
    redirects = set()
    rj = os.path.join(docs_root, "_site", "redirects.json")
    if os.path.isfile(rj):
        for r in json.load(open(rj)):
            s = (r.get("source") or "").strip().strip("/")
            if s:
                redirects.add(s)
    return pages, redirects


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("docs_root", nargs="?", default=".")
    p.add_argument("--fix", action="store_true", help="Rewrite regressions in place.")
    args = p.parse_args(argv)
    docs_root = os.path.abspath(args.docs_root)
    pages, redirects = build_targets(docs_root)

    def resolves(bare):
        return bare in pages or (bare + "/index") in pages or bare in redirects

    violations = []   # (file, path, kind, suggestion)
    fixed = 0
    for loc in LOCALE_DIRS:
        roots = [os.path.join(docs_root, loc),
                 os.path.join(docs_root, "snippets", loc)]
        for base in roots:
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
                for n in files:
                    if not n.endswith(EXTS):
                        continue
                    fp = os.path.join(root, n)
                    rel = os.path.relpath(fp, docs_root)
                    s = open(fp, encoding="utf-8", errors="replace").read()

                    def check(m):
                        nonlocal fixed
                        q, path = m.group(1), m.group(2)
                        raw = path
                        path = path.split("#")[0].split("?")[0]
                        if (path in SKIP_EXACT or path.startswith(SKIP_PREFIXES)):
                            return m.group(0)
                        bare = path.lstrip("/")
                        seg = bare.split("/")
                        if seg and seg[0] == loc:
                            # already localized -- must resolve
                            if not resolves(bare):
                                violations.append((rel, raw, "broken-localized", None))
                            return m.group(0)
                        # unprefixed: localized counterpart exists => must localize
                        localized = f"{loc}/{bare}"
                        if resolves(localized):
                            suggestion = "/" + loc + raw  # keep fragment
                            violations.append((rel, raw, "should-localize", suggestion))
                            if args.fix:
                                fixed += 1
                                return m.group(0).replace(raw, "/" + loc + raw, 1)
                            return m.group(0)
                        if not resolves(bare):
                            violations.append((rel, raw, "broken", None))
                        return m.group(0)

                    ns = HREF.sub(check, s)
                    if args.fix and ns != s:
                        open(fp, "w", encoding="utf-8").write(ns)

    kinds = {}
    for _, _, k, _ in violations:
        kinds[k] = kinds.get(k, 0) + 1
    if args.fix:
        print(f"fixed (localized): {fixed}")
    remaining = [v for v in violations if not (args.fix and v[2] == "should-localize")]
    print(f"violations: {len(remaining)}  by kind: {kinds}")
    for rel, raw, k, sug in remaining[:40]:
        print(f"  [{k}] {raw}  in {rel}" + (f"  -> {sug}" if sug else ""))
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
