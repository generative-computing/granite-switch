# SPDX-License-Identifier: Apache-2.0
"""Declare decision adapters: io.yaml ``decision:`` block -> ``config.json``.

Two entry points, one result (``decision_adapter`` + ``decision_specs`` in
``config.json``, read by the vLLM ``/v1/systemone`` strategy):

* compose: ``compose_granite_switch --decision-adapter NAME`` resolves each named
  adapter's block from its source io.yaml. Where the control token lands
  (``activation``) and, for aLoRA, the marker come from the adapter itself; the spec
  is then probed on the composed chat template.
* stamp: ``python -m granite_switch.composer.decision_config --model DIR
  --decision-adapter NAME`` does the same for an already composed checkpoint from
  its ``io_configs/NAME/io.yaml`` and ``adapter_index.json``. The source adapter is
  not needed, so an aLoRA block must name its ``invocation_marker`` (or declare
  ``activation: generation``); the probe still checks it on the checkpoint's template.

The first ``--decision-adapter`` is the default one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from granite_switch.decision_spec import (
    DecisionSpecError,
    check_activation,
    normalize_decision_block,
)

from .tokenizer_setup import (
    ANCHOR_MODE_ALORA,
    ANCHOR_MODE_SR,
    resolve_activation_anchor,
)

_TECHNOLOGIES = ("lora", ANCHOR_MODE_ALORA, ANCHOR_MODE_SR)


def load_decision_block(io_yaml: Path, adapter_name: str) -> dict:
    """The ``decision:`` block of an io.yaml, or a DecisionSpecError naming the file."""
    import yaml

    if not io_yaml.is_file():
        raise DecisionSpecError(
            f"decision adapter {adapter_name!r} has no io.yaml at {io_yaml}"
        )
    data = yaml.safe_load(io_yaml.read_text()) or {}
    block = data.get("decision") if isinstance(data, dict) else None
    if block is None:
        raise DecisionSpecError(
            f"{io_yaml} has no 'decision:' block; a decision adapter declares its "
            f"prompt layout there (see docs/DECISION_ADAPTERS.md)"
        )
    return block


def _check_names(names: list[str], known: list[str]) -> None:
    if len(set(names)) != len(names):
        raise DecisionSpecError(f"--decision-adapter repeats a name: {names}")
    missing = [n for n in names if n not in known]
    if missing:
        raise DecisionSpecError(
            f"--decision-adapter {missing} not among the adapters {known}"
        )


def _check_technology(name: str, technology: str | None) -> None:
    if technology not in _TECHNOLOGIES:
        raise DecisionSpecError(
            f"decision adapter {name!r} is a {technology} slot; only LoRA, aLoRA "
            f"and Shadow Residual adapters answer decisions"
        )


def _alora_activation(anchor_text: str, tokenizer) -> str:
    """``generation`` when the invocation is the assistant header itself (it starts
    with an added token such as ``<|im_start|>``): the template then places the
    control token before the generation prompt, and the prompt carries no marker."""
    added = getattr(tokenizer, "get_added_vocab", dict)()
    return "generation" if any(anchor_text.startswith(t) for t in added) else "marker"


def resolve_decision_specs(all_discovered, names: list[str], tokenizer) -> dict:
    """Compose time: ``{"decision_adapter", "decision_specs"}`` for the named adapters.

    ``all_discovered`` is the composer's ``(path, name, technology, source)`` list and
    ``tokenizer`` carries the composed chat template, which every spec is probed on.
    """
    by_name = {t[1]: t for t in all_discovered}
    _check_names(names, list(by_name))
    specs = {}
    for name in names:
        path, _, technology, _ = by_name[name]
        _check_technology(name, technology if path else None)
        anchor_text = None
        if technology == ANCHOR_MODE_ALORA:
            anchor_text, _ = resolve_activation_anchor(path, tokenizer)
            activation = _alora_activation(anchor_text, tokenizer)
            if activation != "marker":
                anchor_text = None
        else:
            activation = "start" if technology == "lora" else "generation"
        block = load_decision_block(Path(path) / "io.yaml", name)
        spec = normalize_decision_block(
            block, adapter_name=name, activation=activation, anchor_text=anchor_text
        )
        check_activation(tokenizer, name, spec)
        specs[name] = spec
        print(
            f"  Decision adapter {name!r}: activation {spec['activation']}, marker "
            f"{spec['invocation_marker']!r}, answers {sorted(spec['question_types'])}"
        )
    return {"decision_adapter": names[0], "decision_specs": specs}


def _stamp_activation(technology: str, block: dict) -> str:
    if technology == "lora":
        return "start"
    if technology == ANCHOR_MODE_SR:
        return "generation"
    declared = block.get("activation")
    return declared if declared in ("marker", "generation") else "marker"


def stamp_checkpoint(model_dir: str | os.PathLike, names: list[str]) -> dict:
    """Write ``decision_adapter`` / ``decision_specs`` into a composed checkpoint.

    Reads ``io_configs/<name>/io.yaml`` and each adapter's technology from
    ``adapter_index.json``, probes each spec on the checkpoint's own tokenizer,
    validates the result as a GraniteSwitchConfig, and only then replaces
    ``config.json``. Returns the new decision fields.
    """
    from transformers import AutoTokenizer

    from granite_switch.config import GraniteSwitchConfig

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    _check_names(names, config.get("adapter_names") or [])
    index = json.loads((model_dir / "adapter_index.json").read_text())
    technologies = {a["adapter_name"]: a.get("technology") for a in index["adapters"]}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    specs = {}
    for name in names:
        _check_technology(name, technologies.get(name))
        block = load_decision_block(model_dir / "io_configs" / name / "io.yaml", name)
        spec = normalize_decision_block(
            block,
            adapter_name=name,
            activation=_stamp_activation(technologies[name], block),
        )
        check_activation(tokenizer, name, spec)
        specs[name] = spec

    fields = {"decision_adapter": names[0], "decision_specs": specs}
    tmp = config_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**config, **fields}, indent=2, sort_keys=True) + "\n")
    try:
        # from_json_file, not from_dict: it decodes the {"__float__": ...} encoding.
        GraniteSwitchConfig.from_json_file(str(tmp))
    except BaseException:
        tmp.unlink()
        raise
    tmp.replace(config_path)
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record decision adapters in a composed checkpoint's config.json "
        "from their io_configs/<name>/io.yaml decision blocks."
    )
    parser.add_argument("--model", required=True, help="Composed checkpoint directory")
    parser.add_argument(
        "--decision-adapter",
        dest="decision_adapters",
        action="append",
        required=True,
        metavar="NAME",
        help="Adapter to declare (repeatable; the first is the default)",
    )
    args = parser.parse_args(argv)
    try:
        fields = stamp_checkpoint(args.model, args.decision_adapters)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for name, spec in fields["decision_specs"].items():
        print(
            f"{name}: activation {spec['activation']}, marker "
            f"{spec['invocation_marker']!r}, answers {sorted(spec['question_types'])}"
        )
    print(f"default decision adapter: {fields['decision_adapter']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
