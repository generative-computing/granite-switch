# SPDX-License-Identifier: Apache-2.0
"""Find and check the staged adapters and eval sets under the bench root.

Layout (see ``adapters.yaml``)::

    <bench_root>/adapters/<intrinsic>/<technology>/
        adapter_config.json, adapter_model.safetensors, provenance.json
    <bench_root>/eval/<intrinsic>/evaluation.jsonl

The technology of a checkpoint is checked from its own files, the same way
the composer decides it, so a checkpoint staged in the wrong folder is
rejected instead of benchmarked under the wrong label:

* SR:    the weights carry a ``cross_stream`` LoRA, and the config records
  where the adapter turns on: ``last_context_token`` (shadow-residual
  trainer) or ``alora_invocation_tokens`` (internal trainer).
* aLoRA: non-empty ``alora_invocation_tokens``, no ``cross_stream``.
* LoRA:  neither.

The composer accepts only the ``last_context_token`` form for SR, so an SR
checkpoint with invocation tokens is converted before composing
(``sr_anchor_copy``; ``peft_sr_copy`` for the PEFT reference instead). So is a
checkpoint whose MLP weights lack the base model's ``mlp.`` level
(``mlp_key_copy``). After both, every LoRA weight must name a base-model
weight (``modules_missing_from_base``).

An SR checkpoint must also have been trained with shared K/V
(``shared_kv_reason``).

Standard library only.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from .common import Spec

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"
EVAL_FILE = "evaluation.jsonl"
PROVENANCE_FILE = "provenance.json"
# A saved SR checkpoint does not record whether its adapter stream had its own
# K/V; only the run's config name says so ("..._sharedkv"). This backend always
# takes K/V from the base stream, so a run with its own K/V would compose
# cleanly and give wrong outputs. So SR runs only when the name says shared K/V.
SHARED_KV = re.compile(r"shared_?kv")
# The internal trainer's decoder layer holds gate/up/down itself, with no
# ``mlp`` block, so its checkpoints name them ``layers.N.gate_proj``.
FLAT_MLP = re.compile(r"(\.layers\.\d+\.)(gate_proj|up_proj|down_proj)\.")
PEFT_PREFIX = "base_model.model."


@dataclass(frozen=True)
class StagedCell:
    intrinsic: str
    tech: str
    adapter_dir: Path | None
    eval_path: Path | None
    # None when the cell can run; otherwise a publishable skip reason.
    skip_reason: str | None


def safetensors_header(path: Path) -> tuple[dict, int]:
    """A safetensors file's JSON header and the offset where its tensors start."""
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size)), 8 + size


def safetensors_keys(path: Path) -> list[str]:
    """Tensor names from a safetensors header, without loading the tensors."""
    return [k for k in safetensors_header(path)[0] if k != "__metadata__"]


def detect_technology(adapter_dir: Path) -> tuple[str | None, str | None]:
    """``(technology, None)`` for a usable checkpoint, else ``(None, reason)``."""
    config_path = adapter_dir / CONFIG_FILE
    weights_path = adapter_dir / WEIGHTS_FILE
    if not config_path.is_file():
        return None, f"no {CONFIG_FILE}"
    if not weights_path.is_file():
        return None, f"no {WEIGHTS_FILE}"
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return None, f"unreadable {CONFIG_FILE}"
    try:
        keys = safetensors_keys(weights_path)
    except (OSError, ValueError, struct.error):
        return None, f"unreadable {WEIGHTS_FILE}"
    if not any("lora_" in k for k in keys):
        return None, "no LoRA weights"

    has_cross_stream = any(".cross_stream." in k for k in keys)
    invocation = config.get("alora_invocation_tokens")
    if has_cross_stream:
        if not invocation and not config.get("last_context_token"):
            return None, "SR checkpoint without an activation anchor"
        return "sr", None
    if invocation:
        return "alora", None
    return "lora", None


def sr_anchor_copy(src: Path, dest: Path, anchor: tuple[str, int]) -> dict | None:
    """Give an invocation-token SR checkpoint the anchor the composer needs.

    The internal trainer turns SR on at the aLoRA invocation sequence, e.g.
    ``<|start_of_role|>assistant<|end_of_role|>``, or ``<guardian>`` inside
    the last user message. The composer's SR control token instead replaces
    the last token of the generation prompt. ``anchor`` is that token, as
    ``(text, id)``, read from the base model's chat template.

    The two agree on every generated token, as long as SR turns on anywhere
    in the prompt. SR takes K/V from the base stream only, so the adapter
    stream at a position never reads the adapter stream at earlier ones.
    Turning it on later in the prompt changes only those earlier positions'
    own (unused) outputs.

    Writes ``dest`` with the converted config and links to the other files;
    ``src`` is left as staged. Returns what changed, or None when ``src``
    already has an anchor (nothing written).
    """
    config = json.loads((src / CONFIG_FILE).read_text())
    if config.get("last_context_token"):
        return None
    invocation = config.pop("alora_invocation_tokens")
    config["last_context_token"], config["last_context_token_id"] = anchor
    link_others(src, dest, CONFIG_FILE)
    (dest / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    return {
        "invocation_tokens": invocation,
        "anchor": config["last_context_token"],
        "anchor_id": config["last_context_token_id"],
    }


def peft_sr_copy(src: Path, dest: Path) -> dict | None:
    """Make an invocation-token SR checkpoint load as plain LoRA with PEFT.

    For the reference columns (``reference.py``): PEFT reads
    ``alora_invocation_tokens`` as aLoRA and would turn the LoRA weights off
    before the invocation sequence. The shadow-residual model needs no
    activation point: its adapter stream runs at every position, and, as in
    ``sr_anchor_copy``, generated tokens do not depend on where it turns on.

    Writes ``dest`` with the config without the invocation tokens and links
    to the other files; ``src`` is left as staged. Returns what changed, or
    None when the config has no invocation tokens (nothing written).
    """
    config = json.loads((src / CONFIG_FILE).read_text())
    invocation = config.pop("alora_invocation_tokens", None)
    if not invocation:
        return None
    link_others(src, dest, CONFIG_FILE)
    (dest / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    return {"dropped_invocation_tokens": invocation}


def mlp_key_copy(src: Path, dest: Path) -> dict | None:
    """Give MLP LoRA weights saved without their ``mlp.`` level the base names.

    The internal trainer names its MLP weights ``layers.N.gate_proj``; the
    base model, and so the composer, has ``layers.N.mlp.gate_proj``. The
    composer maps only the names it knows and leaves the others out without
    an error, so such a checkpoint would compose with its attention LoRA only.

    Only the safetensors header changes; the tensor bytes are copied as they
    are. Writes ``dest`` with the renamed weights and links to the other
    files. ``dest`` may be ``src`` itself when it is already a converted copy
    (``sr_anchor_copy``, ``peft_sr_copy``). Returns what changed, or None when
    every name is already standard (nothing written).
    """
    weights = src / WEIGHTS_FILE
    header, data_start = safetensors_header(weights)
    renamed = {FLAT_MLP.sub(r"\1mlp.\2.", k): v for k, v in header.items()}
    count = sum(bool(FLAT_MLP.search(k)) for k in header)
    if not count:
        return None
    if len(renamed) != len(header):
        raise ValueError("MLP weights saved under both names")
    raw = json.dumps(renamed, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # tensors start 8-byte aligned, as when saved
    link_others(src, dest, WEIGHTS_FILE)
    tmp = dest / f"{WEIGHTS_FILE}.tmp"
    with weights.open("rb") as f, tmp.open("wb") as out:
        f.seek(data_start)
        out.write(struct.pack("<Q", len(raw)) + raw)
        shutil.copyfileobj(f, out, 16 << 20)
    tmp.replace(dest / WEIGHTS_FILE)  # also replaces a link to the staged file
    return {"renamed_mlp_weights": count}


def link_others(src: Path, dest: Path, own: str) -> None:
    """Link every file of ``src`` except ``own`` into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == src.resolve():
        return
    for f in src.iterdir():
        link = dest / f.name
        if f.name == own or not f.is_file():
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f.resolve())


def lora_modules(adapter_dir: Path) -> set[str]:
    """Model module names the LoRA weights attach to, e.g. ``model.layers.0.mlp.up_proj``."""
    modules = set()
    for key in safetensors_keys(adapter_dir / WEIGHTS_FILE):
        for tag in (".lora_A.", ".lora_B."):
            if tag in key:
                modules.add(key.split(tag)[0].removeprefix(PEFT_PREFIX))
    return modules


def model_modules(model_dir: Path) -> set[str]:
    """Module names of a model checkpoint's weights."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        names = list(json.loads(index.read_text())["weight_map"])
    else:
        names = [
            k
            for f in sorted(model_dir.glob("*.safetensors"))
            for k in safetensors_keys(f)
        ]
    return {n.rsplit(".", 1)[0] for n in names}


def modules_missing_from_base(adapter_dir: Path, base_modules: set[str]) -> list[str]:
    """LoRA modules of a checkpoint that name no base-model weight.

    The composer would leave these out without an error. SR's
    ``cross_stream`` has no base weight by design and is not counted.
    """
    return sorted(
        m
        for m in lora_modules(adapter_dir) - base_modules
        if not m.endswith(".cross_stream")
    )


def says_shared_kv(path: str) -> bool:
    """Whether a run's path names a shared-K/V configuration."""
    return bool(SHARED_KV.search(path.lower().replace("-", "_")))


def shared_kv_reason(adapter_dir: Path) -> str | None:
    """None if a staged SR checkpoint was trained with shared K/V.

    The evidence is the source run's name, kept in the staged provenance.
    """
    try:
        provenance = json.loads((adapter_dir / PROVENANCE_FILE).read_text())
    except (OSError, ValueError):
        return "SR adapter: K/V mode unknown"
    shared = provenance.get("shared_kv")
    if shared is None:  # staged before the flag was recorded
        shared = says_shared_kv(provenance.get("source", ""))
    return None if shared else "SR adapter not trained with shared K/V"


def check_adapter(adapter_dir: Path, expected_tech: str) -> str | None:
    """None if the checkpoint is a usable ``expected_tech`` adapter."""
    tech, reason = detect_technology(adapter_dir)
    if reason:
        return f"invalid checkpoint: {reason}"
    if tech != expected_tech:
        return f"invalid checkpoint: staged as {expected_tech}, looks like {tech}"
    return None


def adapter_dir(bench_root: Path, intrinsic: str, tech: str) -> Path:
    return bench_root / "adapters" / intrinsic / tech


def eval_path(bench_root: Path, intrinsic: str) -> Path:
    return bench_root / "eval" / intrinsic / EVAL_FILE


def discover(
    bench_root: Path, spec: Spec, only: list[str] | None = None
) -> list[StagedCell]:
    cells = []
    for intrinsic in spec.select(only):
        ev = eval_path(bench_root, intrinsic.id)
        for tech in spec.technologies:
            ad = adapter_dir(bench_root, intrinsic.id, tech.id)
            if not ev.is_file():
                reason = "eval set not staged"
            elif not ad.is_dir():
                reason = "adapter not staged"
            else:
                reason = check_adapter(ad, tech.id)
                if reason is None and tech.id == "sr":
                    reason = shared_kv_reason(ad)
            cells.append(
                StagedCell(
                    intrinsic=intrinsic.id,
                    tech=tech.id,
                    adapter_dir=ad if reason is None else None,
                    eval_path=ev if reason is None else None,
                    skip_reason=reason,
                )
            )
    return cells


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
