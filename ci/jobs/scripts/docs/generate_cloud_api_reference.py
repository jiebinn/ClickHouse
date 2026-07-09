#!/usr/bin/env python3
"""Generate the ClickHouse Cloud API reference documentation from the OpenAPI spec.

The Cloud API reference is backed by the live OpenAPI spec at
`https://api.clickhouse.cloud/v1`. Individual endpoints can be in different
maturity stages (GA, beta, private preview) or be deprecated, but Mintlify only
allows a `tag` badge on navigation *groups*, not on individual OpenAPI operation
entries. As a result a group badge could only be coarse, which is wrong whenever
a group mixes beta and non-beta endpoints.

To get a per-endpoint badge in the sidebar, each operation needs its own `.mdx`
page whose frontmatter carries the operation reference and (optionally) a `tag`
or `deprecated` flag. This script generates one such page per operation, writes
the whole navigation subtree as a single fragment
(`docs/products/cloud/api-reference/navigation.json`), and rewires the consuming
files (`docs.json` and `products/cloud/navigation.json`) to reference that one
definition via `$ref` so the navigation is never duplicated.

The navigation grouping mirrors the spec's `x-tagGroups`/`tags` (the same
hierarchy the hosted Swagger view shows): each tag group is a top-level group;
groups with more than one tag nest every tag as a subgroup, while single-tag
groups list their operations directly.

Maturity is not a structured field in the spec; it is signalled in the operation
`summary`/`description` free text ("This endpoint is in beta.", "private
preview", ...) plus the standard OpenAPI `deprecated` boolean.
`classify_operation` centralises that detection.

Usage:
    generate_cloud_api_reference.py [--spec URL_OR_PATH] [--docs-dir DIR]
                                    [--write | --check]

  --write   Write the generated pages, the navigation fragment and the $ref
            wiring (default).
  --check   Exit non-zero if anything would change (for CI drift detection).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_SPEC = "https://api.clickhouse.cloud/v1"
API_REF_GROUP = "API reference"

# Directory (relative to the docs root) that holds the generated pages and the
# single navigation fragment that defines the whole "API reference" group.
API_REF_DIR = "products/cloud/api-reference"
FRAGMENT_REL = f"{API_REF_DIR}/navigation.json"

# Files that consume the fragment. Each currently holds its own copy of the
# "API reference" group and is rewired to a `$ref`. The ref path is relative to
# the consuming file's own directory (Mintlify resolves `$ref` relative to the
# file it appears in).
CONSUMERS = {
    "docs.json": f"./{FRAGMENT_REL}",
    f"{Path(API_REF_DIR).parent}/navigation.json": "./api-reference/navigation.json",
}

# HTTP methods an OpenAPI path item may carry (other keys such as `parameters`
# are not operations). Operations are emitted in the spec's own path/method
# order so the sidebar mirrors the spec.
HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


def load_spec(source: str) -> dict:
    """Load the OpenAPI spec from a URL or a local file path."""
    if re.match(r"^https?://", source):
        with urllib.request.urlopen(source) as response:  # noqa: S310 (trusted URL)
            return json.loads(response.read())
    return json.loads(Path(source).read_text(encoding="utf-8"))


def dir_slug(tag: str) -> str:
    """Kebab-case a tag for use as a directory name (no camelCase splitting)."""
    return re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")


def file_slug(operation_id: str) -> str:
    """Kebab-case an operationId (camelCase aware) for use as a file name."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", operation_id)
    return re.sub(r"[^a-z0-9]+", "-", spaced.lower()).strip("-")


def classify_operation(operation: dict) -> tuple[str | None, bool]:
    """Return (tag, deprecated) for an operation.

    `tag` is the sidebar badge text ("Beta"/"Private preview") or None for GA.
    Maturity is detected from the free-text summary/description because the spec
    exposes no structured field for it.
    """
    text = f"{operation.get('summary', '') or ''} {operation.get('description', '') or ''}".lower()
    tag = None
    if "private preview" in text:
        tag = "Private preview"
    elif re.search(r"\bbeta\b", text):
        tag = "Beta"
    return tag, bool(operation.get("deprecated"))


def iter_operations(spec: dict):
    """Yield operations in spec traversal order.

    Yields dicts with: method, path, tag (OpenAPI tag), badge, deprecated,
    operation_id, summary.
    """
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            tags = operation.get("tags") or []
            badge, deprecated = classify_operation(operation)
            yield {
                "method": method.upper(),
                "path": path,
                "tag": tags[0] if tags else "Other",
                "badge": badge,
                "deprecated": deprecated,
                "operation_id": operation["operationId"],
                "summary": operation.get("summary") or operation["operationId"],
            }


def page_content(entry: dict) -> str:
    """Render the MDX frontmatter for a single operation page."""
    lines = [
        "---",
        f"title: {json.dumps(entry['summary'])}",
        f"openapi: {json.dumps(entry['method'] + ' ' + entry['path'])}",
    ]
    if entry["badge"]:
        lines.append(f"tag: {json.dumps(entry['badge'])}")
    if entry["deprecated"]:
        lines.append("deprecated: true")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_fragment(spec: dict, entries_by_tag: dict, openapi_source: str) -> dict:
    """Build the "API reference" group object stored as the navigation fragment.

    Each x-tagGroup becomes a top-level group. A group with more than one tag
    nests every tag as a subgroup (mirroring the hosted Swagger view, e.g.
    "Organization" > "Organization"); a single-tag group lists its operations
    directly.
    """
    def page_ref(entry: dict) -> str:
        return f"{API_REF_DIR}/{dir_slug(entry['tag'])}/{file_slug(entry['operation_id'])}"

    def pages_for(tag: str) -> list:
        return [page_ref(e) for e in entries_by_tag.get(tag, [])]

    groups = []
    covered = set()
    for tag_group in spec.get("x-tagGroups", []):
        tags = tag_group["tags"]
        covered.update(tags)
        if len(tags) == 1:
            pages = pages_for(tags[0])
        else:
            pages = [{"group": tag, "pages": pages_for(tag)} for tag in tags]
        groups.append({"group": tag_group["name"], "pages": pages})

    orphan_tags = sorted(set(entries_by_tag) - covered)
    if orphan_tags:
        raise SystemExit(
            f"OpenAPI tags not present in x-tagGroups (would be dropped from nav): {orphan_tags}"
        )

    return {"group": API_REF_GROUP, "openapi": openapi_source, "pages": groups}


def match_object_span(text: str, open_brace: int) -> int:
    """Return the index just past the `}` matching the `{` at `open_brace`."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_brace, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise SystemExit("Unbalanced braces while locating the API reference group")


def rewire_to_ref(text: str, ref_path: str) -> str:
    """Replace the inline "API reference" openapi group with a `$ref`, once.

    A textual splice (rather than re-serializing the whole file) keeps every
    unrelated byte unchanged. Idempotent: once the file already holds the
    `$ref`, it is returned untouched.
    """
    match = re.search(
        r'(?m)^(?P<indent> *)\{\n *"group": "' + re.escape(API_REF_GROUP) + r'",\n *"openapi":',
        text,
    )
    if match:
        indent = match.group("indent")
        start = match.start() + len(indent)  # position of the opening '{'
        end = match_object_span(text, start)
        return text[:start] + f'{{ "$ref": "{ref_path}" }}' + text[end:]
    if f'"$ref": "{ref_path}"' in text:
        return text  # already rewired
    raise SystemExit(
        f'Could not find the "{API_REF_GROUP}" openapi group nor an existing '
        f'$ref "{ref_path}" to rewire'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC, help="OpenAPI spec URL or file path")
    parser.add_argument("--docs-dir", default="docs", help="Path to the docs root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write changes (default)")
    mode.add_argument("--check", action="store_true", help="Fail if anything would change")
    args = parser.parse_args()
    check_only = args.check

    docs_dir = Path(args.docs_dir)
    api_ref_dir = docs_dir / API_REF_DIR
    fragment_path = docs_dir / FRAGMENT_REL

    spec = load_spec(args.spec)
    openapi_source = args.spec if re.match(r"^https?://", args.spec) else DEFAULT_SPEC

    # Group operations by OpenAPI tag, preserving spec traversal order.
    entries_by_tag: dict[str, list] = {}
    expected_files: dict[Path, str] = {}
    badge_counts = {"Beta": 0, "Private preview": 0}
    deprecated_count = 0
    for entry in iter_operations(spec):
        entries_by_tag.setdefault(entry["tag"], []).append(entry)
        rel = Path(API_REF_DIR) / dir_slug(entry["tag"]) / f"{file_slug(entry['operation_id'])}.mdx"
        if rel in expected_files:
            raise SystemExit(f"Duplicate generated page path: {rel}")
        expected_files[rel] = page_content(entry)
        if entry["badge"]:
            badge_counts[entry["badge"]] += 1
        if entry["deprecated"]:
            deprecated_count += 1

    # The navigation fragment is the single source of truth for the group.
    fragment = build_fragment(spec, entries_by_tag, openapi_source)
    expected_files[Path(FRAGMENT_REL)] = json.dumps(fragment, indent=2, ensure_ascii=False) + "\n"

    # Determine page/fragment changes (create/update/delete).
    changes = []
    for rel, content in sorted(expected_files.items()):
        path = docs_dir / rel
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != content:
            changes.append(("update" if current is not None else "create", rel))
    if api_ref_dir.exists():
        for path in api_ref_dir.rglob("*.mdx"):
            if path.relative_to(docs_dir) not in expected_files:
                changes.append(("delete", path.relative_to(docs_dir)))

    # Rewire the consuming files to the fragment via $ref (idempotent).
    consumer_updates = {}
    for rel, ref_path in CONSUMERS.items():
        path = docs_dir / rel
        text = path.read_text(encoding="utf-8")
        new_text = rewire_to_ref(text, ref_path)
        if new_text != text:
            json.loads(new_text)  # fail early if the splice produced invalid JSON
            consumer_updates[path] = new_text
            changes.append(("rewire", rel))

    total_pages = len(expected_files) - 1  # minus the fragment
    print(
        f"{total_pages} operations | Beta {badge_counts['Beta']} | "
        f"Private preview {badge_counts['Private preview']} | deprecated {deprecated_count}"
    )

    if check_only:
        if changes:
            for kind, rel in changes:
                print(f"  {kind}: {rel}")
            print(f"Drift detected: {len(changes)} change(s)")
            return 1
        print("No drift.")
        return 0

    # Write pages + fragment.
    for rel, content in expected_files.items():
        path = docs_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    # Remove stale generated pages and prune empty directories.
    if api_ref_dir.exists():
        for path in sorted(api_ref_dir.rglob("*.mdx")):
            if path.relative_to(docs_dir) not in expected_files:
                path.unlink()
        for path in sorted(api_ref_dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    # Rewire consumers.
    for path, new_text in consumer_updates.items():
        path.write_text(new_text, encoding="utf-8")

    print(f"Wrote {total_pages} pages + fragment; rewired {len(consumer_updates)} consumer(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
