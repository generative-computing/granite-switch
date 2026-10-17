#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report which vLLM minor lines are released upstream but not yet supported here.

The supported range is read from the `vllm` extra in pyproject.toml, which is this
repo's actual claim -- not from the dependency groups, which only say which lines
CI pins. The released set comes from PyPI, filtered to final releases of
non-yanked files.

Granularity is the MINOR line (0.31, 0.32), not the patch: the repo supports lines,
pins lines, and runs one CI leg per line. A new patch inside an already-supported
line needs no work.

Usage:
    vllm_versions.py              # human-readable report
    vllm_versions.py --json       # machine-readable, for the skill's state file
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

PYPI = "https://pypi.org/pypi/vllm/json"
# Final releases only. A pre-release is not something to claim support for, and
# ".post"/".dev" suffixes are not new lines.
FINAL = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def repo_root() -> Path:
    """The repo root, found by walking up to the pyproject.toml that has our name."""
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (d / "pyproject.toml").is_file():
            txt = (d / "pyproject.toml").read_text()
            if 'name = "granite-switch"' in txt:
                return d
    sys.exit("could not locate the granite-switch repo root")


def supported_range(root: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """(floor, exclusive_ceiling) as (major, minor) pairs, from the `vllm` extra.

    Parses `vllm = ["vllm>=0.26.0,<0.31.0"]`. The upper bound is exclusive, so
    `<0.31.0` means the highest SUPPORTED line is 0.30.
    """
    data = tomllib.loads((root / "pyproject.toml").read_text())
    specs = data["project"]["optional-dependencies"]["vllm"]
    spec = next((s for s in specs if s.startswith("vllm")), None)
    if spec is None:
        sys.exit("no vllm requirement found in the [vllm] extra")

    lo = re.search(r">=\s*(\d+)\.(\d+)", spec)
    hi = re.search(r"<\s*(\d+)\.(\d+)", spec)
    if not lo or not hi:
        sys.exit(
            f"the [vllm] extra must carry both a >= and a < bound; got {spec!r}.\n"
            "An open-ended range cannot be checked for gaps -- fix the pin first."
        )
    return (int(lo.group(1)), int(lo.group(2))), (int(hi.group(1)), int(hi.group(2)))


def _fetch(url: str) -> bytes:
    """GET `url`, preferring curl.

    curl goes through the system trust store. A bare framework Python on macOS
    often has no CA bundle at all (no certifi), so urllib raises
    CERTIFICATE_VERIFY_FAILED on a perfectly good URL -- which looks like "PyPI is
    down" and is not. curl first, urllib only as a fallback for a curl-less box.
    """
    if shutil.which("curl"):
        p = subprocess.run(
            ["curl", "-sSfL", "--max-time", "30", url],
            capture_output=True,
        )
        if p.returncode == 0:
            return p.stdout
        curl_err = p.stderr.decode(errors="replace").strip()
    else:
        curl_err = "curl not installed"

    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except urllib.error.URLError as e:
        sys.exit(f"could not fetch {url}\n  curl: {curl_err}\n  urllib: {e}")


def released_lines() -> dict[tuple[int, int], str]:
    """{(major, minor): newest final patch} for every line with a usable release."""
    releases = json.loads(_fetch(PYPI))["releases"]

    newest: dict[tuple[int, int], tuple[int, int, int]] = {}
    for ver, files in releases.items():
        m = FINAL.match(ver)
        if not m:
            continue
        # A release with no files, or whose every file is yanked, is not installable.
        if not files or all(f.get("yanked") for f in files):
            continue
        major, minor, patch = (int(g) for g in m.groups())
        key = (major, minor)
        if key not in newest or (major, minor, patch) > newest[key]:
            newest[key] = (major, minor, patch)
    return {k: f"{v[0]}.{v[1]}.{v[2]}" for k, v in newest.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    root = repo_root()
    floor, ceil_excl = supported_range(root)
    # `<0.31.0` -> the top supported line is the largest released line below it.
    released = released_lines()

    supported = sorted(k for k in released if floor <= k < ceil_excl)
    gap = sorted(k for k in released if k >= ceil_excl)
    newest = max(released) if released else None

    def s(k: tuple[int, int]) -> str:
        return f"{k[0]}.{k[1]}"

    payload = {
        "supported_floor": s(floor),
        "supported_ceiling": s(max(supported)) if supported else None,
        "declared_exclusive_ceiling": s(ceil_excl),
        "newest_released": s(newest) if newest else None,
        "supported_lines": [s(k) for k in supported],
        "gap": [s(k) for k in gap],
        "gap_pins": {s(k): released[k] for k in gap},
        "new_exclusive_ceiling": s((newest[0], newest[1] + 1)) if newest else None,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"supported range   : {payload['supported_floor']} .. ", end="")
    print(
        f"{payload['supported_ceiling']} (extra says <{payload['declared_exclusive_ceiling']})"
    )
    print(f"newest on PyPI    : {payload['newest_released']}")
    if not gap:
        print("\nNo gap: the repo already claims every released vLLM line.")
        return 0

    print(f"\nUNSUPPORTED lines : {', '.join(payload['gap'])}")
    for k in gap:
        print(f"  {s(k):<6} newest patch {released[k]}")
    print(
        f"\nTarget: support {payload['supported_floor']} .. {payload['gap'][-1]}, "
        f"i.e. widen the [vllm] extra to <{payload['new_exclusive_ceiling']}.0"
    )
    print("Target EVERY line in the gap -- the range claim is contiguous.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
