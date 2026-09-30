# SPDX-License-Identifier: Apache-2.0
"""Benchmark definition, cell helpers and the results-block format.

Standard library plus ``yaml`` only: this module is imported both inside the
benchmarked commit's virtualenv (on the pod) and by ``publish.py`` locally.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
SPEC_PATH = HARNESS_DIR / "adapters.yaml"

BEGIN_MARKER = "=== ADAPTER_BENCH_RESULTS_BEGIN ==="
END_MARKER = "=== ADAPTER_BENCH_RESULTS_END ==="

# Technologies composed into the same checkpoint. SR is a whole-checkpoint
# dual-stream mode and the composer refuses to mix it with LoRA / aLoRA.
COMPOSE_GROUPS = {"single_stream": ("lora", "alora"), "dual_stream": ("sr",)}


@dataclass(frozen=True)
class Technology:
    id: str
    label: str
    source: str


@dataclass(frozen=True)
class Intrinsic:
    id: str
    name: str
    scorer: str
    headline: str
    headline_label: str
    max_new_tokens: int


@dataclass(frozen=True)
class Spec:
    bench_version: int
    base_model: str
    technologies: tuple[Technology, ...]
    intrinsics: tuple[Intrinsic, ...]

    def intrinsic(self, intrinsic_id: str) -> Intrinsic:
        for i in self.intrinsics:
            if i.id == intrinsic_id:
                return i
        raise KeyError(f"unknown intrinsic {intrinsic_id!r}")

    def select(self, only: list[str] | None) -> tuple[Intrinsic, ...]:
        if not only:
            return self.intrinsics
        known = {i.id for i in self.intrinsics}
        unknown = sorted(set(only) - known)
        if unknown:
            raise ValueError(f"unknown intrinsics {unknown}; known: {sorted(known)}")
        return tuple(i for i in self.intrinsics if i.id in only)

    def public(self) -> dict:
        """The part of the definition the results page needs."""
        return {
            "bench_version": self.bench_version,
            "base_model": self.base_model,
            "technologies": [vars(t) for t in self.technologies],
            "intrinsics": [
                {
                    "id": i.id,
                    "name": i.name,
                    "headline": i.headline,
                    "headline_label": i.headline_label,
                }
                for i in self.intrinsics
            ],
        }


def load_spec(path: Path = SPEC_PATH) -> Spec:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    spec = Spec(
        bench_version=int(raw["bench_version"]),
        base_model=raw["base_model"],
        technologies=tuple(Technology(**t) for t in raw["technologies"]),
        intrinsics=tuple(Intrinsic(**i) for i in raw["intrinsics"]),
    )
    grouped = {t for techs in COMPOSE_GROUPS.values() for t in techs}
    if {t.id for t in spec.technologies} != grouped:
        raise ValueError(f"technologies must be exactly {sorted(grouped)}")
    for i in spec.intrinsics:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", i.id):
            raise ValueError(f"intrinsic id {i.id!r} must be snake_case")
    return spec


def adapter_name(intrinsic_id: str, tech_id: str) -> str:
    """Name of the adapter inside the composed checkpoint."""
    return f"{intrinsic_id}_{tech_id}"


def compose_group(tech_id: str) -> str:
    for group, techs in COMPOSE_GROUPS.items():
        if tech_id in techs:
            return group
    raise KeyError(tech_id)


# --- cells -----------------------------------------------------------------
# A cell is a dict: ``{metric: value, ..., "n": int}`` when scored,
# ``{"skipped": reason}`` when there is nothing to run, ``{"error": reason}``
# when the run failed. Reasons are published, so they must stay generic.


def skipped(reason: str) -> dict:
    return {"skipped": reason}


def error(reason: str) -> dict:
    return {"error": reason}


def is_error(cell: dict) -> bool:
    return "error" in cell


def is_scored(cell: dict) -> bool:
    return "skipped" not in cell and "error" not in cell


def iter_cells(cells: dict):
    for intrinsic_id, by_tech in cells.items():
        for tech_id, cell in by_tech.items():
            yield intrinsic_id, tech_id, cell


# --- results block -----------------------------------------------------------


def format_results_block(results: dict) -> str:
    return f"{BEGIN_MARKER}\n{json.dumps(results, sort_keys=True)}\n{END_MARKER}"


def extract_block(text: str, begin_marker: str, end_marker: str):
    """The JSON between the last pair of markers in a log.

    Pod logs can wrap or prefix lines, so the JSON is taken as everything
    between the markers with surrounding whitespace removed.
    """
    begin = text.rfind(begin_marker)
    if begin == -1:
        raise ValueError(f"no {begin_marker} in log")
    end = text.find(end_marker, begin)
    if end == -1:
        raise ValueError(f"{begin_marker} is not terminated")
    return json.loads(text[begin + len(begin_marker) : end].strip())


def extract_results_block(text: str) -> dict:
    return extract_block(text, BEGIN_MARKER, END_MARKER)
