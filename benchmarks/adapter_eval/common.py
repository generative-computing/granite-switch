# SPDX-License-Identifier: Apache-2.0
"""Benchmark definition, cell helpers and the results-block format.

Standard library plus ``yaml`` only: this module is imported both inside the
benchmarked commit's virtualenv (on the pod) and by ``publish.py`` locally.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .prompts import DOCUMENT_STYLES
from .scorers import JUDGE_SCORERS

HARNESS_DIR = Path(__file__).resolve().parent
SPEC_PATH = HARNESS_DIR / "adapters.yaml"

BEGIN_MARKER = "=== ADAPTER_BENCH_RESULTS_BEGIN ==="
END_MARKER = "=== ADAPTER_BENCH_RESULTS_END ==="
REFERENCE_BEGIN = "=== ADAPTER_BENCH_REFERENCE_BEGIN ==="
REFERENCE_END = "=== ADAPTER_BENCH_REFERENCE_END ==="
RESCORE_BEGIN = "=== ADAPTER_BENCH_RESCORE_BEGIN ==="
RESCORE_END = "=== ADAPTER_BENCH_RESCORE_END ==="

# Technologies composed into the same checkpoint. SR is a whole-checkpoint
# dual-stream mode and the composer refuses to mix it with LoRA / aLoRA.
COMPOSE_GROUPS = {"single_stream": ("lora", "alora"), "dual_stream": ("sr",)}

# Reference columns (``reference.py``): each technology's staged checkpoint
# with HF + PEFT and no granite-switch, and the base model with no adapter.
BASE_COLUMN = "base"
REFERENCE_COLUMNS = ("lora", "alora", "sr", BASE_COLUMN)
# Libraries whose versions the reference run records.
REFERENCE_LIBRARIES = ("torch", "transformers", "peft")


@dataclass(frozen=True)
class Model:
    """A base model: its own staged adapters, bench_version and page tab."""

    id: str
    name: str  # on the Hugging Face Hub
    label: str
    bench_version: int
    # How its prompts carry documents, and its chat-template options
    # (prompts.py): documents, documents_by_technology, chat_template_kwargs.
    prompt: dict = field(default_factory=dict)

    def prompt_for(self, tech_id: str | None) -> dict:
        """The prompt settings of a technology's cells; None for the base model."""
        by_tech = self.prompt.get("documents_by_technology", {})
        return {
            "documents": by_tech.get(tech_id) or self.prompt.get("documents", "native"),
            "chat_template_kwargs": dict(self.prompt.get("chat_template_kwargs", {})),
        }


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
    # Bumped when only this intrinsic's scoring changes: its saved answers are
    # scored again (rescore.py) instead of generated again.
    score_version: int = 1


@dataclass(frozen=True)
class Throughput:
    """How decode throughput is measured (``throughput`` in adapters.yaml)."""

    batch: int = 32
    generated_tokens: int = 128
    warmup_runs: int = 2
    timed_runs: int = 5

    def settings(self) -> dict:
        return vars(self).copy()

    def matches(self, measured: dict) -> bool:
        """Whether a cell's throughput was measured with these settings."""
        return (
            measured.get("batch") == self.batch
            and measured.get("generated_tokens") == self.generated_tokens
        )


@dataclass(frozen=True)
class Spec:
    """The benchmark definition, for one of its models (``model_id``)."""

    reference_version: int
    models: tuple[Model, ...]
    technologies: tuple[Technology, ...]
    intrinsics: tuple[Intrinsic, ...]
    model_id: str
    throughput: Throughput = field(default_factory=Throughput)

    @property
    def model(self) -> Model:
        return self.get_model(self.model_id)

    @property
    def bench_version(self) -> int:
        return self.model.bench_version

    @property
    def base_model(self) -> str:
        return self.model.name

    @property
    def needs_judge(self) -> bool:
        """Whether an intrinsic is scored by the LLM judge, so jobs need its key."""
        return any(i.scorer in JUDGE_SCORERS for i in self.intrinsics)

    def get_model(self, model_id: str) -> Model:
        for m in self.models:
            if m.id == model_id:
                return m
        known = [m.id for m in self.models]
        raise ValueError(f"unknown model {model_id!r}; known: {known}")

    def for_model(self, model_id: str | None) -> Spec:
        """This definition for another model; None keeps the current one."""
        if model_id is None:
            return self
        return replace(self, model_id=self.get_model(model_id).id)

    def model_of(self, block: dict) -> str:
        """The model id of a results or reference block.

        Blocks from before there were several models name only the base model.
        """
        if block.get("model"):
            return self.get_model(block["model"]).id
        for m in self.models:
            if m.name == block.get("base_model"):
                return m.id
        raise ValueError(f"no model has base model {block.get('base_model')!r}")

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
            "reference_version": self.reference_version,
            "models": [vars(m) for m in self.models],
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
            "throughput": self.throughput.settings(),
        }


def load_spec(path: Path = SPEC_PATH, model: str | None = None) -> Spec:
    """The benchmark definition for ``model``; None is the first model."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    models = tuple(
        Model(**{**m, "bench_version": int(m["bench_version"])}) for m in raw["models"]
    )
    if not models:
        raise ValueError("adapters.yaml lists no models")
    ids = [m.id for m in models]
    if len(set(ids)) != len(ids):
        raise ValueError(f"model ids must be unique: {ids}")
    for model_id in ids:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", model_id):
            raise ValueError(f"model id {model_id!r}: use a-z, 0-9, '.' and '-'")
    spec = Spec(
        reference_version=int(raw["reference_version"]),
        models=models,
        technologies=tuple(Technology(**t) for t in raw["technologies"]),
        intrinsics=tuple(Intrinsic(**i) for i in raw["intrinsics"]),
        model_id=ids[0],
        throughput=Throughput(**raw.get("throughput", {})),
    )
    if any(
        not isinstance(v, int) or v < (0 if k == "warmup_runs" else 1)
        for k, v in spec.throughput.settings().items()
    ):
        raise ValueError(f"bad throughput settings {spec.throughput.settings()}")
    grouped = {t for techs in COMPOSE_GROUPS.values() for t in techs}
    if {t.id for t in spec.technologies} != grouped:
        raise ValueError(f"technologies must be exactly {sorted(grouped)}")
    for m in models:
        unknown = set(m.prompt) - {
            "documents",
            "documents_by_technology",
            "chat_template_kwargs",
        }
        by_tech = m.prompt.get("documents_by_technology", {})
        styles = {m.prompt.get("documents", "native"), *by_tech.values()}
        if unknown or set(by_tech) - grouped or styles - set(DOCUMENT_STYLES):
            raise ValueError(f"model {m.id}: bad prompt settings {m.prompt}")
    if set(REFERENCE_COLUMNS) != grouped | {BASE_COLUMN}:
        raise ValueError("REFERENCE_COLUMNS must be every technology plus the base")
    for i in spec.intrinsics:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", i.id):
            raise ValueError(f"intrinsic id {i.id!r} must be snake_case")
    return spec.for_model(model)


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


def format_block(results: dict, begin_marker: str, end_marker: str) -> str:
    return f"{begin_marker}\n{json.dumps(results, sort_keys=True)}\n{end_marker}"


def format_results_block(results: dict) -> str:
    return format_block(results, BEGIN_MARKER, END_MARKER)


def format_reference_block(reference: dict) -> str:
    return format_block(reference, REFERENCE_BEGIN, REFERENCE_END)


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
