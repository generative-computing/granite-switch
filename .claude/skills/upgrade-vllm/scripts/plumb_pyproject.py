#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add vLLM lines to pyproject.toml: per-line groups, dev groups, extra, conflicts.

Text-level editing on purpose. `tomllib` is read-only and `tomlkit` is not a
dependency of this repo, but more importantly pyproject.toml's comments here are
load-bearing documentation (why these two CI legs, why vllm27 is still declared)
and a round-trip through a generic TOML writer would reflow or drop them.

What it does:
  * adds `vllmNN = ["vllm>=0.NN.0,<0.NN+1.0"]` for each new line
  * adds `dev-vllmNN = [pytest, pytest-cov, include-group vllmNN, extras]`
  * widens the `vllm` extra's exclusive upper bound
  * REGENERATES `[tool.uv] conflicts` from the full set of declared lines
  * points `dev-latest` at the line whose GPU evidence is being collected next

It never removes a group. A group that main's GPU matrix still names must survive,
or that leg's `uv sync --group X` fails before pytest starts (see REFERENCE.md §4).

WHY `dev-latest` IS A PARAMETER AND NOT JUST THE CEILING. The GPU workflow has no
matrix to widen: `/opt/gsw/derive_legs.py` lives in the runner image, reads the
TESTED COMMIT's own pyproject, and emits at most two legs -- one from `dev` (the
floor) and one from `dev-latest`. So the only way to get independent green
evidence for each added line is to make each line be `dev-latest` at some commit
and dispatch at that commit's sha. Workflow concurrency is keyed on sha, so those
dispatches run in PARALLEL rather than serialising. `--add` therefore points
`dev-latest` at the LOWEST added line, and one `--point-latest` commit per
remaining line walks it up; the last one leaves it on the true ceiling, which is
also the correct steady state to merge.

Usage:
    plumb_pyproject.py --add 0.31              # support one more line
    plumb_pyproject.py --add 0.31 0.32         # and the one after it
    plumb_pyproject.py --add 0.31 --floor 0.28 # also RAISE the floor (drops 0.26/0.27)
    plumb_pyproject.py --point-latest 0.32     # next line to collect evidence for
    plumb_pyproject.py --check                 # verify conflicts are complete, edit nothing
    plumb_pyproject.py --check --final         # ALSO require dev-latest == the ceiling
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

GROUP_RE = re.compile(r"^(vllm(\d+))\s*=", re.M)
NAME_W = 10  # existing files pad group names to this width before `=`


def repo_root() -> Path:
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        p = d / "pyproject.toml"
        if p.is_file() and 'name = "granite-switch"' in p.read_text():
            return d
    sys.exit("could not locate the granite-switch repo root")


def line_key(line: str) -> str:
    """'0.31' -> '31'. The group suffix is the minor only; every line is 0.x today."""
    major, minor = line.split(".")
    if major != "0":
        # vLLM 1.x would make `vllmNN` ambiguous (1.1 and 0.1 both -> vllm1).
        sys.exit(
            f"line {line} is not 0.x -- the vllmNN group naming assumes a 0.x series.\n"
            "Pick a naming scheme (e.g. vllm1_1) and update this script and the "
            "GPU matrix together."
        )
    return minor


def parse_declared(text: str) -> list[str]:
    """Declared per-line groups, as '0.NN' strings, ascending."""
    return sorted(
        (f"0.{m.group(2)}" for m in GROUP_RE.finditer(text)),
        key=lambda s: int(s.split(".")[1]),
    )


def find_floor(root: Path) -> str:
    """The line that the bare `dev` group pins -- i.e. the floor's dev flavour."""
    data = tomllib.loads((root / "pyproject.toml").read_text())
    for item in data["dependency-groups"]["dev"]:
        if isinstance(item, dict) and "include-group" in item:
            g = item["include-group"]
            m = re.fullmatch(r"vllm(\d+)", g)
            if m:
                return f"0.{m.group(1)}"
    sys.exit("could not determine the floor: `dev` includes no vllmNN group")


DEV_INCLUDE_RE = re.compile(r'^(dev\s*=\s*\[.*?\{ include-group = ")vllm\d+(")', re.M)


DEFAULT_GROUPS_RE = re.compile(r'^(default-groups = \[")vllm\d+("\])', re.M)


def point_floor(text: str, line: str) -> str:
    """Repoint everything that names the floor by line: `dev` and `default-groups`.

    Both have to move or raising the floor is actively broken, in two distinct ways:

      * `dev` IS the lower GPU leg (derive_legs.py reads it), so leaving it behind
        means CI keeps testing the line that just left the supported range and
        never tests the new floor.
      * `default-groups` is what a bare `uv sync` and the CPU CI jobs resolve. Left
        pointing below the extra's new lower bound it is not merely stale, it is
        UNSATISFIABLE -- an extra pinned to one line while default-groups named
        another is the exact failure pyproject's own comment records.
    """
    nn = line_key(line)
    text, n = DEV_INCLUDE_RE.subn(rf"\g<1>vllm{nn}\g<2>", text, count=1)
    if n != 1:
        sys.exit(
            "could not rewrite the `dev` group's include-group -- expected\n"
            '  dev = [..., { include-group = "vllmNN" }, ...]\n'
            "Fix it by hand, or CI's lower leg will test the dropped floor."
        )
    text, n = DEFAULT_GROUPS_RE.subn(rf"\g<1>vllm{nn}\g<2>", text, count=1)
    if n != 1:
        sys.exit(
            "could not rewrite `default-groups` -- expected the single-group form\n"
            '  default-groups = ["vllmNN"]\n'
            "Fix it by hand, or `uv sync` resolves a line outside the extra's range."
        )
    return text


LATEST_RE = re.compile(
    r'^(dev-latest\s*=\s*\[\{ include-group = ")dev-vllm(\d+)("\s*\}\])', re.M
)


def find_latest(text: str) -> str | None:
    """The line `dev-latest` currently aliases, as '0.NN', or None if absent.

    This is the second GPU leg's identity. `derive_legs.py` reads it from the
    commit under test, so whatever it says here is what the upper leg installs.
    """
    m = LATEST_RE.search(text)
    return f"0.{m.group(2)}" if m else None


# The one sentence of prose that names the line the alias points at. Rewritten
# with the alias: this script owns `dev-latest`, so leaving its own comment saying
# "0.30" while the alias reads dev-vllm32 would be a review failure it caused
# itself. The surrounding paragraph is human-written and left alone -- step 5 of
# the NEXT list asks for that.
# Not anchored to the start of the line: the sentence is wrapped into a comment
# paragraph and in practice lands mid-line ("# every caller. Currently the 0.30
# line."). An `^#`-anchored version silently matched nothing.
LATEST_NOTE_RE = re.compile(r"(Currently the )0\.\d+( line\.)")


def point_latest(text: str, line: str, declared: list[str]) -> str:
    """Repoint `dev-latest` at `line`'s dev group."""
    if line not in declared:
        sys.exit(
            f"--point-latest {line} has no dev-vllm{line_key(line)} group to alias.\n"
            f"Declare the line first: --add {line}"
        )
    text, n = LATEST_RE.subn(rf"\g<1>dev-vllm{line_key(line)}\g<3>", text, count=1)
    if n != 1:
        sys.exit(
            "could not rewrite `dev-latest` -- expected the single-line form\n"
            '  dev-latest = [{ include-group = "dev-vllmNN" }]\n'
            "Fix it by hand, or the upper GPU leg will test the wrong line."
        )
    return LATEST_NOTE_RE.sub(rf"\g<1>{line}\g<2>", text, count=1)


def flavours(line: str, floor: str, latest: str | None = None) -> list[str]:
    """Every group name that pulls `line` into a venv.

    Two asymmetries defeat a naive name template, which is why this is a function:

      * the FLOOR's dev flavour is bare `dev`, not `dev-vllmNN`
      * the line `dev-latest` aliases has a THIRD name

    On the alias: uv does NOT actually need those entries -- it derives the
    conflict transitively through the include-group chain, verified by stripping
    the two `vllm31`/`dev-latest` pairs and watching
    `uv sync --no-default-groups --group vllm31 --group dev-latest` still refuse
    with "transitively inferred conflicts". They are emitted anyway because main's
    committed table lists them explicitly, and `--check` compares against that
    committed form; generating a shorter-but-equivalent table would report every
    hand-maintained file as incomplete. `test` is the one group left implicit,
    matching main.
    """
    nn = line_key(line)
    names = [f"vllm{nn}", "dev" if line == floor else f"dev-vllm{nn}"]
    if latest is not None and line == latest:
        names.append("dev-latest")
    return names


def build_conflicts(lines: list[str], floor: str, latest: str | None = None) -> str:
    """The full `conflicts = [...]` block: every pair that mixes two vLLM lines.

    Generated rather than hand-maintained because the entry count grows
    quadratically in the number of lines and a MISSING pair does not fail loudly --
    it lets `uv sync` resolve two vLLM wheels into one environment.
    """
    names = {ln: flavours(ln, floor, latest) for ln in lines}
    n = len(lines)
    pairs = n * (n - 1) // 2
    total = sum(
        len(names[a]) * len(names[b])
        for i, a in enumerate(lines)
        for b in lines[i + 1 :]
    )
    top = max(int(line_key(ln)) for ln in lines)
    out = [
        "# GENERATED by .claude/skills/upgrade-vllm/scripts/plumb_pyproject.py.",
        "# Only the per-line GROUPS are mutually exclusive. The `vllm` extra spans",
        f"# {floor}-0.{top}, so it overlaps every group and conflicts with none of them --",
        "# an extra pinned to one line while default-groups named another is what made",
        "# `uv sync` unsatisfiable once before.",
        "# Below: every pair of groups that would pull two different vLLM lines into one",
        f"# venv. {n} lines are declared ({' / '.join(lines)}), each with a bare and a",
        f"# dev flavour, so the {pairs} cross-line pairs give {total} entries. The line",
        "# `dev-latest` aliases carries a third name and so appears in more pairs --",
        "# uv matches group names literally, so the alias needs its own entries.",
        "# `test` is absent deliberately: it reaches the floor through",
        '# `include-group = "dev"` and uv derives that transitively.',
        "conflicts = [",
    ]
    width = max(len(f) for v in names.values() for f in v) + 2
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            out.append(f"    # {a} vs {b}")
            for fa in names[a]:
                for fb in names[b]:
                    la = f'{{ group = "{fa}" }},'.ljust(width + 14)
                    out.append(f'    [{la}{{ group = "{fb}" }}],')
    out.append("]")
    return "\n".join(out)


def replace_conflicts(text: str, block: str) -> str:
    """Swap the existing conflicts block (and the comments above it) for `block`."""
    start = text.find("\nconflicts = [")
    if start < 0:
        sys.exit("no `conflicts = [` found in pyproject.toml")
    end = text.find("\n]\n", start)
    if end < 0:
        sys.exit("unterminated `conflicts = [` block")

    # Swallow the comment run immediately above, which the old block's generator
    # (or a human) wrote to describe the old shape. Leaving it would contradict the
    # new table.
    head = text[:start].rstrip("\n").split("\n")
    while head and head[-1].lstrip().startswith("#"):
        head.pop()
    # end+2 consumes "\n]" only, leaving the newline that terminated the `]` line in
    # the tail. That preserves whatever blank-line spacing followed the old block
    # instead of eating the separator before the next table.
    return "\n".join(head) + "\n" + block + text[end + 2 :]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--add", nargs="*", default=[], metavar="LINE", help="e.g. 0.31 0.32"
    )
    ap.add_argument("--floor", help="raise the floor, DROPPING support below it")
    ap.add_argument(
        "--point-latest",
        metavar="LINE",
        help="point dev-latest at LINE (the next line to collect GPU evidence for)",
    )
    ap.add_argument("--check", action="store_true", help="verify only; make no edits")
    ap.add_argument(
        "--final",
        action="store_true",
        help="with --check: also require dev-latest to be on the ceiling",
    )
    args = ap.parse_args()

    root = repo_root()
    path = root / "pyproject.toml"
    text = path.read_text()

    floor = args.floor or find_floor(root)
    declared = parse_declared(text)
    new = [ln for ln in args.add if ln not in declared]

    if args.floor:
        kept = [ln for ln in declared if int(line_key(ln)) >= int(line_key(args.floor))]
        dropped = [ln for ln in declared if ln not in kept]
        if dropped:
            print(f"floor raised to {args.floor}: lines {', '.join(dropped)} leave the")
            print("supported range. Their GROUPS are kept (REFERENCE.md §4); what")
            print("moves is the `vllm` extra's lower bound plus the floor's two other")
            print("names: `dev` (the LOWER GPU leg) and `default-groups` (bare")
            print("`uv sync` and CPU CI). Record the drop in the PR body.")
        if args.floor not in declared + new:
            sys.exit(
                f"--floor {args.floor} has no vllm{line_key(args.floor)} group to pin it"
            )

    lines = sorted(set(declared + new), key=lambda s: int(line_key(s)))

    ceiling = max(lines, key=lambda s: int(line_key(s)))

    if args.check:
        latest = find_latest(text)
        want = build_conflicts(lines, floor, latest)
        have_start = text.find("\nconflicts = [")
        have_end = text.find("\n]\n", have_start)
        have = text[have_start + 1 : have_end + 2]
        wanted_entries = set(re.findall(r"\[\{[^\]]+\}\]", want))
        have_entries = set(re.findall(r"\[\{[^\]]+\}\]", have))
        missing = wanted_entries - have_entries
        print(f"declared lines : {', '.join(lines)} (floor {floor})")
        print(f"GPU legs       : dev -> {floor}, dev-latest -> {latest or '(unset!)'}")
        print(
            f"conflicts      : {len(have_entries)} entries, {len(wanted_entries)} required"
        )
        rc = 0
        if missing:
            print(f"MISSING {len(missing)}:")
            for m in sorted(missing):
                print(f"  {m}")
            rc = 1
        else:
            print("conflicts table is complete.")
        # The steady state to MERGE is dev-latest on the ceiling. Mid-upgrade it is
        # parked on whichever line is collecting evidence, which is correct then and
        # wrong to merge: it would leave the top line permanently untested by CI.
        if args.final and latest != ceiling:
            print(
                f"NOT FINAL: dev-latest is on {latest}, ceiling is {ceiling}.\n"
                f"  Run --point-latest {ceiling} before merging, or CI's upper leg"
                " never tests the top line."
            )
            rc = 1
        elif args.final:
            print(f"dev-latest is on the ceiling ({ceiling}).")
        return rc

    if args.point_latest and not new:
        text = point_latest(text, args.point_latest, declared)
        # The conflicts table must move WITH the alias. `dev-latest` is a third
        # name for whichever line it points at, so repointing it both removes
        # pairs from the old line and adds them to the new one. uv would infer
        # those pairs either way (see flavours()), so this is about keeping the
        # written table honest rather than about correctness of resolution --
        # a table naming `vllm31`/`dev-latest` after the alias moved to 0.32 is
        # simply false. --check compares membership, not just the entry count,
        # which is what catches it.
        text = replace_conflicts(text, build_conflicts(lines, floor, args.point_latest))
        path.write_text(text)
        print(f"dev-latest     : -> {args.point_latest}")
        print("conflicts      : regenerated to follow the alias")
        print()
        print("NEXT: commit this alone, push, and dispatch the GPU suites at its sha.")
        print("  The upper leg will install " + args.point_latest + "; the lower leg")
        print(f"  stays on the floor ({floor}). Concurrency is keyed on sha, so this")
        print("  run does not cancel the lines already dispatched.")
        return 0

    if not new and not args.floor:
        print("nothing to add.")
        return 0

    # --- new per-line groups, inserted after the last existing vllmNN line ---
    for ln in new:
        nn = line_key(ln)
        nxt = int(nn) + 1
        entry = f'{f"vllm{nn}".ljust(NAME_W)} = ["vllm>=0.{nn}.0,<0.{nxt}.0"]'
        last = list(GROUP_RE.finditer(text))[-1]
        eol = text.find("\n", last.start())
        text = text[: eol + 1] + entry + "\n" + text[eol + 1 :]

    # --- new dev groups, after the last dev-vllmNN (or after bare `dev`) ---
    for ln in new:
        nn = line_key(ln)
        entry = (
            f'{f"dev-vllm{nn}".ljust(NAME_W)} = ["pytest", "pytest-cov", '
            f'{{ include-group = "vllm{nn}" }}, "granite-switch[hf,compose,audio]"]'
        )
        anchors = list(re.finditer(r"^dev(-vllm\d+)?\s*=", text, re.M))
        eol = text.find("\n", anchors[-1].start())
        text = text[: eol + 1] + entry + "\n" + text[eol + 1 :]

    # --- the `vllm` extra: this is the supported-range CLAIM ---
    top = max(int(line_key(ln)) for ln in lines)
    lo = line_key(floor)
    extra = f'vllm = ["vllm>=0.{lo}.0,<0.{top + 1}.0"]'
    text, n_sub = re.subn(r'^vllm = \["vllm[^"]*"\]', extra, text, count=1, flags=re.M)
    if n_sub != 1:
        sys.exit("could not rewrite the `vllm` extra -- check its formatting by hand")

    # --- dev-latest: the upper GPU leg's identity ---
    # Park it on the LOWEST added line, not the ceiling. The added lines need
    # evidence in ascending order and only one can be `dev-latest` per commit, so
    # this commit buys the first of them; `--point-latest` walks the rest up.
    #
    # With `--floor` and no `--add` there is no new line to buy evidence for, so
    # leave `dev-latest` where it is rather than moving it arbitrarily.
    latest_target = (
        args.point_latest
        or (min(new, key=lambda s: int(line_key(s))) if new else None)
        or find_latest(text)
        or ceiling
    )
    text = point_latest(text, latest_target, lines)

    # --- the floor's two other names: `dev` (lower GPU leg) and `default-groups`
    # (bare `uv sync` / CPU CI). Only --floor moves them. ---
    if args.floor:
        text = point_floor(text, floor)

    text = replace_conflicts(text, build_conflicts(lines, floor, latest_target))
    path.write_text(text)

    remaining = [ln for ln in lines if int(line_key(ln)) > int(line_key(latest_target))]

    print(f"added lines    : {', '.join(new) or '(none)'}")
    print(f"declared lines : {', '.join(lines)} (floor {floor})")
    print(f"vllm extra     : >=0.{lo}.0,<0.{top + 1}.0")
    print(f"GPU legs       : dev -> {floor}, dev-latest -> {latest_target}")
    print("conflicts      : regenerated")
    print()
    print("NEXT, and none of it is optional:")

    # Numbered from a counter rather than literals: the branches below used to
    # print 1,2,3,5,6 when one of them was skipped.
    step = 0

    def nxt(first: str, *rest: str) -> None:
        nonlocal step
        step += 1
        print(f"  {step}. {first}")
        for line in rest:
            print(f"     {line}")

    nxt("relock:   uvx uv@0.8.4 lock")
    if new:
        nxt(
            "commit + push, then dispatch the GPU suites at THIS sha. The two",
            f"legs cover {floor} and {latest_target}. There is no matrix to widen:",
            "derive_legs.py is in the runner image and emits at most two legs,",
            "from `dev` and `dev-latest`, read out of the TESTED COMMIT.",
        )
        if remaining:
            steps = ", ".join(f"--point-latest {ln}" for ln in remaining)
            nxt(
                f"then one commit per remaining line ({steps}), each pushed",
                "and dispatched at its own sha. Concurrency is keyed on sha, so",
                "these run in parallel instead of cancelling one another -- that",
                "is what makes per-line evidence affordable.",
            )
            nxt(
                f"finish with --point-latest 0.{top} (the ceiling) and verify:",
                "plumb_pyproject.py --check --final",
            )
        else:
            nxt("verify the merge state: plumb_pyproject.py --check --final")
    else:
        nxt("dispatch the GPU suites; the legs are the ends of the new range")
    nxt(
        "update the [dependency-groups] and [vllm]-extra comments to name",
        "the new range -- a stale comment is a review failure here",
    )
    nxt(f"grep -rn '0\\.{top - 1}' README.md tutorials/ docs/   # prose ranges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
