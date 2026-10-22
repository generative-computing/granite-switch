#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diff two vLLM tags over only the API surface granite-switch actually depends on.

The whole-release changelog is thousands of commits and useless for this. What
matters is the few dozen symbols this repo imports from vLLM or overrides on a vLLM
base class. For each one this reports whether, between tag A and tag B, it was

  * DELETED      -- definition count went to zero
  * THINNED      -- fewer definitions (a backend-by-backend removal in progress)
  * SIGNATURE    -- still there, but the `def` line changed
  * gone-missing -- an imported name no longer resolvable anywhere in the tree

and then names the upstream commits that touched the files involved.

DELETED is the dangerous one, and not because it breaks loudly. When the deleted
thing is a HOOK this repo overrides, nothing breaks: the override simply stops
being called and the feature silently stops working. vLLM 0.29 did this to
`BaseMultiModalProcessor._call_hf_processor`, which is where ASR happened, and
every test stayed green while no transcript was ever produced again. Treat a
deleted hook as a correctness incident, not a refactor.

Usage:
    upstream_api_diff.py v0.30.0 v0.31.0
    upstream_api_diff.py v0.30.0 v0.31.0 --cache ~/.cache/gsw-vllm
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "granite-switch-upgrade" / "vllm"
UPSTREAM = "https://github.com/vllm-project/vllm.git"

# Where our vLLM coupling lives. tests/ is included deliberately: most of the
# 0.26->0.30 breakage was test scaffolding written against APIs that 0.29 deleted,
# and a red test leg is just as blocking as a broken backend.
OUR_PATHS = [
    "src/granite_switch/vllm",
    "src/granite_switch/kernels",
    "tests/shared",
    "tests/vllm",
    "tests/integration",
]

IMPORT_RE = re.compile(r"^\s*from\s+(vllm[\w.]*)\s+import\s+(.+)$", re.M)
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.M)

# Dunders and pytest plumbing are ours, never vLLM's. Including them produces pure
# noise (`__init__` has thousands of definitions upstream).
IGNORE = re.compile(r"^(__\w+__|test_\w+|main|setUp|tearDown)$")


def sh(args: list[str], cwd: Path | None = None, check: bool = True) -> str:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"command failed: {' '.join(args)}\n{p.stderr}")
    return p.stdout


def repo_root() -> Path:
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        p = d / "pyproject.toml"
        if p.is_file() and 'name = "granite-switch"' in p.read_text():
            return d
    sys.exit("could not locate the granite-switch repo root")


def ensure_clone(cache: Path, tags: list[str]) -> Path:
    """A bare-ish cached clone, fetched just deeply enough to hold both tags."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / ".git").is_dir():
        print(f"cloning vLLM into {cache} (once; a few minutes)...", file=sys.stderr)
        sh(["git", "clone", "--filter=blob:none", UPSTREAM, str(cache)])
    print("fetching tags...", file=sys.stderr)
    sh(["git", "fetch", "--tags", "--force", "origin"], cwd=cache)
    for t in tags:
        if sh(["git", "tag", "-l", t], cwd=cache).strip() != t:
            sys.exit(
                f"tag {t} not found upstream. Released tags look like v0.31.0 -- "
                "check the name with:\n  git -C "
                f"{cache} tag -l 'v0.3*'"
            )
    return cache


def our_surface(root: Path) -> tuple[set[str], dict[str, set[str]]]:
    """(method names we define, {vllm module: names we import from it}).

    Our own `def`s are the override candidates. We cannot know which are overrides
    without resolving vLLM's MRO, so we test all of them against the upstream tree:
    a name we define that ALSO exists upstream is an override or a shadow, and
    either way its disappearance matters.
    """
    defs: set[str] = set()
    imports: dict[str, set[str]] = defaultdict(set)

    for rel in OUR_PATHS:
        base = root / rel
        if not base.is_dir():
            continue
        for f in base.rglob("*.py"):
            text = f.read_text(errors="replace")
            for m in DEF_RE.finditer(text):
                if not IGNORE.match(m.group(1)):
                    defs.add(m.group(1))
            for m in IMPORT_RE.finditer(text):
                mod, names = m.group(1), m.group(2)
                names = names.split("#")[0].strip().strip("()")
                for n in names.split(","):
                    n = n.strip().split(" as ")[0].strip()
                    if n and n != "*":
                        imports[mod].add(n)
    return defs, imports


def defs_at(cache: Path, tag: str, name: str) -> list[tuple[str, str]]:
    """[(file, def-line)] for every definition of `name` in vllm/ at `tag`.

    NOTE: these patterns go to `git grep -E`, which is POSIX ERE and does NOT
    understand the `\\s` / `\\w` Perl shorthands -- it silently matches nothing
    rather than erroring. Using `\\s` here made EVERY symbol report zero
    definitions at every tag, so each one compared 0-to-0 and the whole report
    came back "nothing on our surface moved" no matter what upstream did. It hid
    the real 0.28->0.29 deletion of `get_kv_cache_shape` (30 defs -> 1) that
    REFERENCE.md documents. Keep these as POSIX classes.
    """
    out = sh(
        [
            "git",
            "grep",
            "-n",
            "-E",
            rf"^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+{re.escape(name)}[[:space:]]*\(",
            tag,
            "--",
            "vllm/",
        ],
        cwd=cache,
        check=False,
    )
    hits = []
    for line in out.splitlines():
        # format: <tag>:<path>:<lineno>:<text>
        parts = line.split(":", 3)
        if len(parts) == 4:
            hits.append((parts[1], parts[3].strip()))
    return hits


def symbol_exists(cache: Path, tag: str, name: str) -> bool:
    out = sh(
        [
            "git",
            "grep",
            "-l",
            "-E",
            rf"^[[:space:]]*(class|def|{re.escape(name)}[[:space:]]*=)[[:space:]]*"
            rf"{re.escape(name)}\b|^[[:space:]]*{re.escape(name)}[[:space:]]*[:=]",
            tag,
            "--",
            "vllm/",
        ],
        cwd=cache,
        check=False,
    )
    return bool(out.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_tag")
    ap.add_argument("new_tag")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    root = repo_root()
    cache = ensure_clone(args.cache, [args.old_tag, args.new_tag])
    defs, imports = our_surface(root)

    print(f"\n=== granite-switch vLLM surface: {args.old_tag} -> {args.new_tag} ===")
    print(f"{len(defs)} locally-defined names, ", end="")
    print(f"{sum(len(v) for v in imports.values())} imported names\n")

    deleted: list[tuple[str, list[tuple[str, str]]]] = []
    thinned: list[tuple[str, int, int]] = []
    changed: list[tuple[str, str, str]] = []
    touched_files: set[str] = set()

    print("-- overridable definitions --")
    for name in sorted(defs):
        old = defs_at(cache, args.old_tag, name)
        if not old:
            continue  # not a vLLM name at all; purely ours
        new = defs_at(cache, args.new_tag, name)
        touched_files.update(f for f, _ in old + new)

        if not new:
            deleted.append((name, old))
            print(f"  DELETED   {name}  ({len(old)} def(s) at {args.old_tag}, 0 now)")
            for f, _ in old[:4]:
                print(f"              was in {f}")
        elif len(new) < len(old):
            thinned.append((name, len(old), len(new)))
            print(f"  THINNED   {name}  {len(old)} -> {len(new)} def(s)")
        else:
            old_sigs = {s for _, s in old}
            new_sigs = {s for _, s in new}
            if old_sigs != new_sigs:
                for s in sorted(old_sigs - new_sigs):
                    changed.append((name, s, ""))
                    print(f"  SIGNATURE {name}")
                    print(f"              - {s}")
                for s in sorted(new_sigs - old_sigs):
                    print(f"              + {s}")

    print("\n-- imported names --")
    for mod in sorted(imports):
        for name in sorted(imports[mod]):
            if not symbol_exists(cache, args.old_tag, name):
                continue  # re-exported or dynamically built; this check can't see it
            if not symbol_exists(cache, args.new_tag, name):
                print(f"  MISSING   {mod}.{name}  -- no longer defined in vllm/")

    if touched_files:
        print(
            f"\n-- upstream commits touching those files ({args.old_tag}..{args.new_tag}) --"
        )
        files = sorted(touched_files)
        log = sh(
            [
                "git",
                "log",
                "--oneline",
                "--no-merges",
                f"{args.old_tag}..{args.new_tag}",
                "--",
                *files,
            ],
            cwd=cache,
            check=False,
        )
        lines = log.splitlines()
        for line in lines[:60]:
            print(f"  {line}")
        if len(lines) > 60:
            print(f"  ... and {len(lines) - 60} more")
            print(
                f"  full list: git -C {cache} log --oneline --no-merges "
                f"{args.old_tag}..{args.new_tag} -- <files>"
            )

    print("\n-- what to do with this --")
    if deleted:
        print(f"  {len(deleted)} DELETED name(s). For each, answer: do we OVERRIDE it?")
        print("  If yes, our override is now dead code and whatever it did stopped")
        print("  happening SILENTLY. Find the new hook, delegate to super() while the")
        print("  old one still exists (hasattr), and add a test asserting the EFFECT.")
    if thinned:
        print(
            f"  {len(thinned)} THINNED name(s): a removal mid-flight. Expect it gone next line."
        )
    if changed:
        print(
            f"  {len(changed)} SIGNATURE change(s). If both call sites must be served,"
        )
        print("  take **kwargs -- no single explicit signature matches two lines.")
    if not (deleted or thinned or changed):
        print("  Nothing on our surface moved. The port may be plumbing-only --")
        print("  but still run the GPU legs: behaviour changes do not show up here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
