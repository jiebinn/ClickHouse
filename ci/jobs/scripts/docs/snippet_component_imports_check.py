#!/usr/bin/env python3
"""Validate imports inside default-locale Mintlify snippets.

Every snippet resolves its own imports. A page-level import is not visible to
a snippet imported by that page, and the same rule applies recursively when a
snippet renders another snippet. This check requires every non-built-in MDX
tag to have a local import.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


IMPORT_RE = re.compile(
    r"^\s*import\s+(?P<spec>.+?)\s+from\s+['\"](?P<src>[^'\"]+)['\"]\s*;?\s*$",
    re.MULTILINE,
)
TAG_RE = re.compile(r"<\/?([A-Z][A-Za-z0-9_]*)\b")
DECLARED_EXPORT_RE = re.compile(
    r"\bexport\s+(?:default\s+)?(?:const|let|var|function|class)\s+"
    r"([A-Z][A-Za-z0-9_]*)"
)
TRANSLATION_DIRS = {"ar", "es", "fr", "ja", "ko", "pt-BR", "ru", "zh"}
MINTLIFY_BUILTINS = {
    "Accordion",
    "CodeBlock",
    "Info",
    "Note",
    "Step",
    "Steps",
    "Tab",
    "Tabs",
    "Tip",
    "Warning",
}


@dataclass(frozen=True)
class Binding:
    exported: str
    local: str
    source: str


def without_fenced_code(text: str) -> str:
    """Blank fenced code while retaining line boundaries for import parsing."""
    output: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker.group(1)[0]
            if fence is None:
                fence = char
            elif fence == char:
                fence = None
            output.append("\n" if line.endswith("\n") else "")
        elif fence is None:
            output.append(line)
        else:
            output.append("\n" if line.endswith("\n") else "")
    return "".join(output)


def parse_bindings(text: str) -> list[Binding]:
    bindings: list[Binding] = []
    for match in IMPORT_RE.finditer(without_fenced_code(text)):
        spec = match.group("spec").strip()
        source = match.group("src")
        if spec.startswith("{") and spec.endswith("}"):
            for piece in spec[1:-1].split(","):
                piece = piece.strip()
                if not piece:
                    continue
                names = re.split(r"\s+as\s+", piece, maxsplit=1)
                bindings.append(Binding(names[0], names[-1], source))
        elif spec.startswith("*"):
            local = re.split(r"\s+as\s+", spec, maxsplit=1)[-1]
            bindings.append(Binding(Path(source).stem, local, source))
        elif re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", spec):
            bindings.append(Binding(Path(source).stem, spec, source))
    return bindings


def is_nested_snippet_source(source: str) -> bool:
    return source.startswith("/snippets/") and source.endswith((".md", ".mdx"))


def is_translation_source(source: str) -> bool:
    parts = Path(source.lstrip("/")).parts
    return len(parts) > 1 and parts[0] == "snippets" and parts[1] in TRANSLATION_DIRS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs_root", nargs="?", default=".")
    args = parser.parse_args()

    docs_root = Path(args.docs_root).resolve()
    snippets_root = docs_root / "snippets"
    if not snippets_root.is_dir():
        parser.error(f"No snippets directory under docs root: {docs_root}")

    errors: list[str] = []
    checked = 0

    for path in sorted(snippets_root.rglob("*")):
        if path.suffix not in {".md", ".mdx"}:
            continue
        relative_path = path.relative_to(snippets_root)
        if (
            "components" in relative_path.parts
            or relative_path.parts[0] in TRANSLATION_DIRS
        ):
            continue

        checked += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        visible_text = without_fenced_code(text)
        bindings = parse_bindings(text)
        imported = {binding.local for binding in bindings}
        declared = set(DECLARED_EXPORT_RE.findall(visible_text))

        used_tags = set(TAG_RE.findall(visible_text))
        missing = sorted(used_tags - imported - declared - MINTLIFY_BUILTINS)
        if missing:
            errors.append(
                f"{relative_path.as_posix()}: missing local import(s) for "
                + ", ".join(missing)
            )

        for binding in bindings:
            if not is_nested_snippet_source(binding.source):
                continue
            target = docs_root / binding.source.lstrip("/")
            if not target.is_file():
                errors.append(
                    f"{relative_path.as_posix()}: nested snippet import does not exist: "
                    f"{binding.source}"
                )
            if is_translation_source(binding.source):
                errors.append(
                    f"{relative_path.as_posix()}: default-locale snippet imports translated "
                    f"snippet {binding.source}"
                )

    if errors:
        print(f"Found {len(errors)} snippet import error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Checked imports in {checked} snippet file(s): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
