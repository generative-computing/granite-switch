# SPDX-License-Identifier: Apache-2.0
"""Put a recorded agent race into its page, ``docs/agent_race.html``.

Runs locally, on the log of the pod that ran ``run_race.py``::

    python -m benchmarks.adapter_eval.race.build_page <pod.log> [--page <html>]

The race is read from between the log's markers and written into the page's
``const RACE_EVENTS_EMBEDDED = ...; // <<RACE_EVENTS>>`` line, so the page opens
straight from disk, as the ALORA-vs-LoRA race page does. A race that carries an
absolute path anywhere is refused: the page is public, the run's storage is not.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .run_race import decode_block

REPO_ROOT = Path(__file__).resolve().parents[3]
PAGE = REPO_ROOT / "docs" / "agent_race.html"
EMBED = re.compile(r"^const RACE_EVENTS_EMBEDDED = .*; // <<RACE_EVENTS>>$", re.M)


def private_strings(value, where: str = "race") -> list[str]:
    """Where ``value`` holds a string that looks like a path on the run's machine."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in private_strings(v, f"{where}.{k}")]
    if isinstance(value, list):
        return [
            p for i, v in enumerate(value) for p in private_strings(v, f"{where}[{i}]")
        ]
    if isinstance(value, str) and value.startswith("/"):
        return [where]
    return []


def embed(page: str, race: dict | None) -> str:
    """``page`` with ``race`` as its data (``None`` empties it)."""
    line = (
        f"const RACE_EVENTS_EMBEDDED = {json.dumps(race, separators=(',', ':'))};"
        " // <<RACE_EVENTS>>"
    )
    if not EMBED.search(page):
        raise ValueError("the page has no RACE_EVENTS_EMBEDDED line")
    return EMBED.sub(lambda _: line, page, count=1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", type=Path, help="the race pod's saved log")
    p.add_argument("--page", type=Path, default=PAGE)
    args = p.parse_args(argv)
    race = decode_block(args.log.read_text(errors="replace"))
    leaks = private_strings(race)
    if leaks:
        raise SystemExit(f"refusing to publish: absolute paths at {leaks[:5]}")
    args.page.write_text(embed(args.page.read_text(), race))
    meta = race["metadata"]
    walls = ", ".join(f"{k} {v['wall_s']:.1f}s" for k, v in meta["lanes"].items())
    print(f"wrote {args.page}: {len(race['events'])} events; {walls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
