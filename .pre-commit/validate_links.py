#!/usr/bin/env python3
"""Validate local file links and first-party Python imports across the repo.

Scans every git-tracked .ipynb, .md, and .py file and reports:
  - broken links: [text](target) where target does not exist on disk
  - stale labels: target exists but the label names a different file
  - broken imports: first-party `from pkg.x import y` or `import pkg.x` where
    pkg.x does not resolve under any configured package root

Read-only. Exit 0 on clean, exit 1 on any finding.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
LABEL_FILENAME_RE = re.compile(
    r"[\w./-]+\.(?:ipynb|md|py|png|jpg|jpeg|svg|json|sh)\b",
    re.IGNORECASE,
)
EXT_OK = {".ipynb", ".md", ".py", ".png", ".jpg", ".jpeg", ".svg", ".json", ".sh"}


def git_ls_files(repo: Path) -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()


def discover_package_roots(repo: Path) -> tuple[list[Path], set[str]]:
    roots: list[Path] = []
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        cfg = tomllib.loads(pyproject.read_text())
        find = cfg.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
        for w in find.get("where", []) or []:
            roots.append((repo / w).resolve())
    if not roots:
        for cand in (".", "src"):
            p = (repo / cand).resolve()
            if p.exists():
                roots.append(p)
    first_party: set[str] = set()
    for r in roots:
        if not r.exists():
            continue
        for child in r.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                first_party.add(child.name)
    return roots, first_party


def module_resolves(dotted: str, roots: list[Path]) -> bool:
    parts = dotted.split(".")
    for root in roots:
        cur = root
        ok = True
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            pkg_dir = cur / part
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                cur = pkg_dir
                continue
            if is_last and (cur / f"{part}.py").exists():
                return True
            ok = False
            break
        if ok:
            return True
    return False


def resolve_relative(
    file_path: Path, level: int, module: str | None, roots: list[Path]
) -> str | None:
    for root in roots:
        try:
            rel = file_path.resolve().relative_to(root)
        except ValueError:
            continue
        pkg_parts = list(rel.parts[:-1])
        if level - 1 > len(pkg_parts):
            return None
        base = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
        tail = module.split(".") if module else []
        return ".".join(base + tail)
    return None


def scan_text(
    text: str,
    source_path: Path,
    source_label: str,
    existing: set[Path],
    existing_dirs: set[Path],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str]]]:
    broken: list[tuple[str, str]] = []
    stale: list[tuple[str, str, str, str]] = []
    for m in LINK_RE.finditer(text):
        label_text = m.group(1)
        target = m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "attachment:")):
            continue
        bare = target.split("#")[0].split("?")[0]
        if not bare:
            continue
        ext = Path(bare).suffix.lower()
        if ext and ext not in EXT_OK:
            continue
        resolved = (source_path.parent / bare).resolve()
        target_basename = Path(bare).name
        target_ok = resolved in existing or (not ext and resolved in existing_dirs)
        if not target_ok:
            broken.append((source_label, target))
            continue
        for tok_match in LABEL_FILENAME_RE.finditer(label_text):
            label_token = tok_match.group(0).split("/")[-1]
            if label_token != target_basename:
                stale.append((source_label, label_text, target, target_basename))
                break
    return broken, stale


def scan_imports(
    source: str,
    source_label: str,
    file_path: Path,
    roots: list[Path],
    first_party: set[str],
) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in first_party and not module_resolves(alias.name, roots):
                    out.append((source_label, f"import {alias.name}", node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                dotted = resolve_relative(file_path, node.level, node.module, roots)
                if dotted is None:
                    continue
            else:
                dotted = node.module or ""
            if not dotted:
                continue
            top = dotted.split(".")[0]
            if top not in first_party:
                continue
            if len(node.names) == 1 and module_resolves(f"{dotted}.{node.names[0].name}", roots):
                continue
            if not module_resolves(dotted, roots):
                names = ", ".join(a.name for a in node.names)
                out.append((source_label, f"from {dotted} import {names}", node.lineno))
    return out


def main() -> int:
    repo = Path(".").resolve()
    tracked = git_ls_files(repo)
    existing = {(repo / p).resolve() for p in tracked}
    existing_dirs: set[Path] = set()
    for p in tracked:
        for parent in (repo / p).resolve().parents:
            existing_dirs.add(parent)

    roots, first_party = discover_package_roots(repo)

    broken_links: list[tuple[str, str]] = []
    stale_labels: list[tuple[str, str, str, str]] = []
    broken_imports: list[tuple[str, str, int]] = []

    for rel in tracked:
        p = repo / rel
        if not p.exists():
            continue
        suffix = p.suffix
        if suffix == ".md":
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            b, s = scan_text(text, p, rel, existing, existing_dirs)
            broken_links += b
            stale_labels += s
        elif suffix == ".py":
            try:
                src = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            broken_imports += scan_imports(src, rel, p, roots, first_party)
        elif suffix == ".ipynb":
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            for ci, cell in enumerate(data.get("cells", [])):
                ctype = cell.get("cell_type")
                src_lines = cell.get("source", [])
                if ctype == "markdown":
                    src = "".join(src_lines)
                    b, s = scan_text(src, p, f"{rel} (cell {ci})", existing, existing_dirs)
                    broken_links += b
                    stale_labels += s
                elif ctype == "code":
                    if not src_lines:
                        continue
                    first_nonblank = next((line for line in src_lines if line.strip()), "")
                    if first_nonblank.lstrip().startswith(("%", "!")):
                        continue
                    src = "".join(src_lines)
                    broken_imports += scan_imports(src, f"{rel} (cell {ci})", p, roots, first_party)

    findings = len(broken_links) + len(stale_labels) + len(broken_imports)

    if broken_links:
        print("BROKEN LINKS\n")
        for label, target in broken_links:
            print(f"  {label}\n    {target}")
        print(f"\n  {len(broken_links)} broken link(s)\n")

    if stale_labels:
        print("STALE LABELS (target works, but the label names the wrong file)\n")
        for label, ltext, target, expected in stale_labels:
            print(f"  {label}\n    [{ltext}]({target})  -> label should name {expected}")
        print(f"\n  {len(stale_labels)} stale label(s)\n")

    if broken_imports:
        print("BROKEN IMPORTS (first-party module path does not resolve on disk)\n")
        for label, stmt, lineno in broken_imports:
            print(f"  {label} (line {lineno})\n    {stmt}")
        print(f"\n  {len(broken_imports)} broken import(s)\n")
        roots_str = "  ".join(str(r.relative_to(repo)) or "." for r in roots)
        print(
            f"  (package roots used: {roots_str}  |  "
            f"first-party packages: {', '.join(sorted(first_party)) or '(none)'})\n"
        )

    if findings == 0:
        print("validate_links: clean (no broken links, stale labels, or broken imports)")
        return 0

    print(
        f"validate_links: {findings} finding(s). "
        "Fix the broken links, stale labels, and imports reported above."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
