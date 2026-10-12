# SPDX-License-Identifier: Apache-2.0
"""Benchmark definition, cell helpers and the results-block format.

Standard library plus ``yaml`` only: this module is imported both inside the
benchmarked commit's virtualenv (on the pod) and by ``publish.py`` locally.
"""

from __future__ import annotations

import json
import os
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

# What a model may set for itself (``models[]`` in adapters.yaml). Each
# setting defaults to the benchmark-wide behaviour.
SHAPE_KEYS = {
    "rank",
    "cross_rank",
    "modules",
    "shared_kv",
    "share_moe_routing",
    "names",
}
RESOURCE_MODES = {"discover", "stage", "bench", "reference", "script", "rescore"}
RESOURCE_KEYS = {"gpus", "cpus", "memory"}
SERVING_BLOCKS = ("throughput", "switching")
REFERENCE_KEYS = {"token_budget", "max_batch", "serial_loads"}
# Models kept off the results page are defined in a file of their own, next to
# local.env, which submit.sh points this variable at and ships with each job.
PRIVATE_MODELS_ENV = "ADAPTER_BENCH_PRIVATE_MODELS"


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
    # Per technology, what differs from the benchmark-wide definition: how the
    # adapters were trained, as the page says it (``source``), and the shape
    # their weights must have (``shape``, over stage.EXPECTED). For a model
    # whose adapters can target attention only, say.
    technologies: dict = field(default_factory=dict)
    # Pod sizes per launcher mode, over vela/render_job.py's RESOURCES.
    resources: dict = field(default_factory=dict)
    # Which serving blocks are measured (SERVING_BLOCKS); all when not said.
    serving: dict = field(default_factory=dict)
    # Settings of the HF + PEFT columns (reference.py): REFERENCE_KEYS.
    reference: dict = field(default_factory=dict)
    # Never published: defined in the private models file (PRIVATE_MODELS_ENV).
    private: bool = False

    def prompt_for(self, tech_id: str | None) -> dict:
        """The prompt settings of a technology's cells; None for the base model."""
        by_tech = self.prompt.get("documents_by_technology", {})
        return {
            "documents": by_tech.get(tech_id) or self.prompt.get("documents", "native"),
            "chat_template_kwargs": dict(self.prompt.get("chat_template_kwargs", {})),
        }

    def shape(self, tech_id: str) -> dict:
        """The model's own shape for a technology's adapters, over stage.EXPECTED."""
        return dict(self.technologies.get(tech_id, {}).get("shape", {}))

    def measures(self, block: str) -> bool:
        """Whether a serving block (SERVING_BLOCKS) is measured for this model."""
        return bool(self.serving.get(block, True))

    @property
    def shared_routing(self) -> bool:
        """Whether its SR adapters must take the expert routing from the base
        stream, as granite-switch serves a mixture-of-experts model."""
        return self.shape("sr").get("share_moe_routing") is True

    def resources_for(self, mode: str) -> dict:
        """Pod size settings for a launcher mode, over the mode's defaults."""
        return dict(self.resources.get(mode, {}))

    def public(self) -> dict:
        """What the results page records of the model."""
        out = {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "bench_version": self.bench_version,
            "prompt": self.prompt,
        }
        if self.technologies:
            out["technologies"] = self.technologies
        if self.serving:
            out["serving"] = self.serving
        return out


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
class Switching:
    """How agents switching adapters are measured (``switching`` in adapters.yaml)."""

    concurrency: int = 8
    decode_tokens: int = 512
    span: int = 32
    adapters: int = 64
    rank: int = 32
    min_agents: int = 16
    warmup_runs: int = 1

    def settings(self) -> dict:
        return vars(self).copy()

    def matches(self, measured: dict) -> bool:
        """Whether a cell was measured with these settings."""
        return all(measured.get(k) == v for k, v in self.settings().items())


@dataclass(frozen=True)
class Spec:
    """The benchmark definition, for one of its models (``model_id``)."""

    reference_version: int
    models: tuple[Model, ...]
    technologies: tuple[Technology, ...]
    intrinsics: tuple[Intrinsic, ...]
    model_id: str
    throughput: Throughput = field(default_factory=Throughput)
    switching: Switching = field(default_factory=Switching)

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

    @property
    def public_models(self) -> tuple[Model, ...]:
        """The models the results page shows."""
        return tuple(m for m in self.models if not m.private)

    def source(self, tech_id: str) -> str:
        """How the model's adapters of a technology were trained, for the page."""
        own = self.model.technologies.get(tech_id, {}).get("source")
        return own or next(t.source for t in self.technologies if t.id == tech_id)

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
            "models": [m.public() for m in self.public_models],
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
            "switching": self.switching.settings(),
        }


def load_spec(path: Path = SPEC_PATH, model: str | None = None) -> Spec:
    """The benchmark definition for ``model``; None is the first model.

    With PRIVATE_MODELS_ENV set to a file, that file's ``models`` are added,
    marked private: they run like any other, but are never published.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    entries = list(raw["models"])
    private_file = os.environ.get(PRIVATE_MODELS_ENV)
    if private_file and Path(private_file).is_file():
        private = yaml.safe_load(Path(private_file).read_text()) or {}
        entries += [{**m, "private": True} for m in private.get("models", [])]
    models = tuple(
        Model(**{**m, "bench_version": int(m["bench_version"])}) for m in entries
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
        switching=Switching(**raw.get("switching", {})),
    )

    def bad(settings: dict) -> bool:
        return any(
            not isinstance(v, int) or v < (0 if k == "warmup_runs" else 1)
            for k, v in settings.items()
        )

    if bad(spec.throughput.settings()):
        raise ValueError(f"bad throughput settings {spec.throughput.settings()}")
    sw = spec.switching
    if bad(sw.settings()) or sw.decode_tokens % sw.span:
        raise ValueError(f"bad switching settings {sw.settings()}")
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
        for tech_id, own in m.technologies.items():
            if (
                tech_id not in grouped
                or not isinstance(own, dict)
                or set(own) - {"source", "shape"}
                or not isinstance(own.get("source", ""), str)
                or not isinstance(own.get("shape", {}), dict)
                or set(own.get("shape", {})) - SHAPE_KEYS
            ):
                raise ValueError(
                    f"model {m.id}: bad technologies settings {m.technologies}"
                )
        if set(m.resources) - RESOURCE_MODES or any(
            not isinstance(v, dict) or set(v) - RESOURCE_KEYS
            for v in m.resources.values()
        ):
            raise ValueError(f"model {m.id}: bad resources {m.resources}")
        if set(m.serving) - set(SERVING_BLOCKS) or any(
            not isinstance(v, bool) for v in m.serving.values()
        ):
            raise ValueError(f"model {m.id}: bad serving settings {m.serving}")
        if set(m.reference) - REFERENCE_KEYS:
            raise ValueError(f"model {m.id}: bad reference settings {m.reference}")
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
