# SPDX-License-Identifier: Apache-2.0
"""Render a human-readable ``BUILD.md`` for a composed Granite Switch model.

``BUILD.md`` is a compose-specific companion to the base model's upstream
documentation: it summarises the base model plus the embedded adapters so the
build output is self-describing without having to read ``adapter_index.json``
or ``compose_report.json``.

Layout (top to bottom):
  - H1 title
  - Base Model section (identifier, arch fields, param count delta)
  - Embedded Adapters section (table)
  - Composition Details section (YAML-style text block with raw values)
"""

from pathlib import Path
from typing import Iterable, List, Mapping, Optional


_BASE_MODEL_FIELDS = [
    ("model_type", "Model type"),
    ("architectures", "Architectures"),
    ("hidden_size", "Hidden size"),
    ("num_hidden_layers", "Hidden layers"),
    ("num_attention_heads", "Attention heads"),
    ("vocab_size", "Vocab size"),
]


def _escape_pipes(text: str) -> str:
    """Escape ``|`` characters so they render as literal pipes inside a markdown
    table cell (otherwise they're treated as column separators)."""
    return text.replace("|", r"\|")


def _yaml_scalar(value) -> str:
    """Render a Python scalar as a YAML scalar.

    Uses double-quoted form for strings to sidestep most YAML escaping
    concerns (only ``"`` and ``\\`` need escaping inside double quotes).
    Numbers and booleans are rendered unquoted.
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_base_model_section(
    base_model_name: str,
    base_config,
) -> List[str]:
    lines = ["## Base Model", "", f"- Identifier: {base_model_name}"]
    for attr, label in _BASE_MODEL_FIELDS:
        value = getattr(base_config, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {label}: {value}")
    lines.append("")
    return lines


def _format_alpha(alpha) -> str:
    """Render a LoRA alpha value. Emits an integer when the float is whole
    (most alphas are integer-valued in practice)."""
    if alpha is None:
        return ""
    try:
        f = float(alpha)
    except (TypeError, ValueError):
        return str(alpha)
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def _format_target_modules(targets) -> str:
    """Render a set/list of target module names as a sorted comma-joined string."""
    if not targets:
        return ""
    try:
        items = sorted(targets)
    except TypeError:
        items = list(targets)
    return ", ".join(str(t) for t in items)


def _format_adapter_row(
    entry: dict,
    rank: Optional[int],
    alpha,
    targets,
    source: Optional[str],
) -> str:
    name = entry.get("adapter_name", "")
    technology = entry.get("technology") or ("built-in" if entry.get("built_in") else "")
    control_token = entry.get("control_token") or {}
    token_text = _escape_pipes(control_token.get("token", ""))
    token_id = control_token.get("id", "")
    rank_str = "" if rank is None else str(rank)
    alpha_str = _format_alpha(alpha)
    targets_str = _format_target_modules(targets)
    if source is None:
        source_str = "built-in" if entry.get("built_in") else ""
    else:
        source_str = source
    return (
        f"| {entry.get('adapter_index', '')} "
        f"| {name} "
        f"| {technology} "
        f"| `{token_text}` "
        f"| {token_id} "
        f"| {rank_str} "
        f"| {alpha_str} "
        f"| {targets_str} "
        f"| {source_str} |"
    )


def _pad(values: Optional[List], length: int) -> List:
    out = list(values) if values is not None else [None] * length
    if len(out) < length:
        out = out + [None] * (length - len(out))
    return out


def _format_adapters_section(
    adapters: Iterable[dict],
    adapter_ranks: Optional[List[int]],
    adapter_alphas: Optional[List],
    adapter_targets: Optional[List],
    adapter_sources: Optional[List[Optional[str]]],
) -> List[str]:
    adapters = list(adapters)
    lines = ["## Embedded Adapters", ""]
    if not adapters:
        lines.append("_No adapters embedded._")
        lines.append("")
        return lines

    lines.append(f"Total adapters: **{len(adapters)}**")
    lines.append("")
    lines.append("| # | Name | Technology | Control Token | Token ID | Rank | Alpha | Target Modules | Source |")
    lines.append("|---|------|------------|---------------|----------|------|-------|----------------|--------|")

    n = len(adapters)
    ranks = _pad(adapter_ranks, n)
    alphas = _pad(adapter_alphas, n)
    targets = _pad(adapter_targets, n)
    sources = _pad(adapter_sources, n)

    for entry, rank, alpha, tgt, source in zip(adapters, ranks, alphas, targets, sources):
        lines.append(_format_adapter_row(entry, rank, alpha, tgt, source))
    lines.append("")
    return lines


def _format_composition_details_section(
    compose_settings: Optional[Mapping[str, object]],
    adapter_commits_by_source: Optional[Mapping[str, str]],
    base_param_count: Optional[int],
    composed_param_count: Optional[int],
) -> List[str]:
    """Render the Composition Details section.

    Starts with a human-readable ``Params (base → composed)`` summary (the
    ``+X%`` is a property of the composition, not of the base model, so it
    lives here rather than in the Base Model section). Follows with a plain
    YAML-style block of raw values (``base_param_count``,
    ``composed_param_count``, ``compose_settings``, ``adapter_sources``) —
    indented key/value pairs under the section heading, not wrapped in a
    code fence.
    """
    visible_settings = {}
    if compose_settings:
        visible_settings = {
            k: v for k, v in compose_settings.items()
            if v is not None and v != "" and v != []
        }
    visible_sources = dict(adapter_commits_by_source or {})

    has_params = base_param_count is not None or composed_param_count is not None
    if not visible_settings and not visible_sources and not has_params:
        return []

    lines: List[str] = ["## Composition Details", ""]

    # Markdown-list block. Integer counts use thousands separators. Param
    # delta follows immediately. Nested mappings (compose_settings,
    # adapter_sources) render as indented sub-list items.
    if base_param_count is not None:
        lines.append(f"- base_param_count: {int(base_param_count):,}")
    if composed_param_count is not None:
        lines.append(f"- composed_param_count: {int(composed_param_count):,}")
    if (
        base_param_count is not None
        and composed_param_count is not None
        and base_param_count > 0
    ):
        pct = (composed_param_count - base_param_count) / base_param_count * 100
        lines.append(f"- Param delta: {pct:+.2f}%")
    if visible_settings:
        lines.append("- compose_settings:")
        for key, value in visible_settings.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"  - {key}:")
                for item in value:
                    lines.append(f"    - {_yaml_scalar(item)}")
            else:
                lines.append(f"  - {key}: {_yaml_scalar(value)}")
    if visible_sources:
        lines.append("- adapter_sources:")
        for source, commit in visible_sources.items():
            lines.append(f"  - {_yaml_scalar(source)}: {_yaml_scalar(commit)}")
    lines.append("")
    return lines


def render_model_card(
    base_model_name: str,
    base_config,
    adapter_index: dict,
    adapter_ranks: Optional[List[int]] = None,
    adapter_alphas: Optional[List] = None,
    adapter_targets: Optional[List] = None,
    adapter_sources: Optional[List[Optional[str]]] = None,
    adapter_commits_by_source: Optional[Mapping[str, str]] = None,
    compose_settings: Optional[Mapping[str, object]] = None,
    base_param_count: Optional[int] = None,
    composed_param_count: Optional[int] = None,
) -> str:
    """Render a Markdown model card describing the composed model.

    Args:
        base_model_name: Human-readable identifier of the base model.
        base_config: HuggingFace ``PretrainedConfig`` of the base model.
        adapter_index: The ``adapter_index.json`` dict produced by the
            composer.
        adapter_ranks: Optional per-adapter LoRA rank list, aligned with
            ``adapter_index["adapters"]``.
        adapter_sources: Optional per-adapter source (HF repo ID or local
            path), aligned with ``adapter_index["adapters"]``.
        adapter_commits_by_source: Optional deduped ``{source: commit_sha}``
            mapping (full 40-char SHA). Rendered into Composition Details
            under ``adapter_sources``.
        compose_settings: Optional mapping of compose flag names to values.
            Rendered into Composition Details under ``compose_settings``.
        base_param_count: Optional total base-model parameter count.
        composed_param_count: Optional total composed-model parameter count.

    Returns:
        Markdown string suitable for writing to ``BUILD.md``.
    """
    adapters = adapter_index.get("adapters", []) if adapter_index else []

    lines: List[str] = []
    lines.append("# Granite Switch Composed Model")
    lines.append("")
    lines.extend(_format_base_model_section(base_model_name, base_config))
    lines.extend(_format_adapters_section(
        adapters, adapter_ranks, adapter_alphas,
        adapter_targets, adapter_sources,
    ))
    lines.extend(_format_composition_details_section(
        compose_settings, adapter_commits_by_source,
        base_param_count, composed_param_count,
    ))
    return "\n".join(lines).rstrip() + "\n"


def write_model_card(
    output_path: str,
    base_model_name: str,
    base_config,
    adapter_index: dict,
    adapter_ranks: Optional[List[int]] = None,
    adapter_alphas: Optional[List] = None,
    adapter_targets: Optional[List] = None,
    adapter_sources: Optional[List[Optional[str]]] = None,
    adapter_commits_by_source: Optional[Mapping[str, str]] = None,
    compose_settings: Optional[Mapping[str, object]] = None,
    base_param_count: Optional[int] = None,
    composed_param_count: Optional[int] = None,
) -> Path:
    """Render and write ``BUILD.md`` into ``output_path``.

    Returns the path to the written file.
    """
    markdown = render_model_card(
        base_model_name=base_model_name,
        base_config=base_config,
        adapter_index=adapter_index,
        adapter_ranks=adapter_ranks,
        adapter_alphas=adapter_alphas,
        adapter_targets=adapter_targets,
        adapter_sources=adapter_sources,
        adapter_commits_by_source=adapter_commits_by_source,
        compose_settings=compose_settings,
        base_param_count=base_param_count,
        composed_param_count=composed_param_count,
    )
    dst = Path(output_path) / "BUILD.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(markdown)
    return dst


def write_build_doc(
    model,
    args,
    all_discovered: List,
    output_path: str,
    base_model_local_path: str,
    adapter_index: dict,
    extract_hf_snapshot_commit,
) -> Path:
    """Gather metadata from the composed model and write BUILD.md.

    This is the high-level orchestrator that collects adapter alphas, target
    modules, parameter counts, commit SHAs, and compose settings from the
    model and CLI args, then delegates to :func:`write_model_card` for
    rendering.

    Args:
        model: The composed ``GraniteSwitchForCausalLM`` model (must have
            ``_build_mappings`` set by the composer).
        args: Parsed CLI arguments namespace (needs ``base_model``,
            ``built_in_adapters``, ``lora_rank``, ``lora_alpha``, etc.).
        all_discovered: List of (path, name, technology, source) tuples.
        output_path: Output directory path.
        base_model_local_path: Resolved local path to the base model.
        adapter_index: The ``adapter_index.json`` dict produced by the
            composer.
        extract_hf_snapshot_commit: Callable ``(adapter_path) -> sha | None``
            that extracts a HF snapshot commit SHA from an adapter path.

    Returns:
        Path to the written BUILD.md file.
    """
    from ..arch import load_base_config

    base_config = load_base_config(base_model_local_path)
    adapter_ranks = getattr(model.config, "adapter_ranks", None)
    # all_discovered tuples are (path, name, technology, source).
    # For BUILD.md readability, shorten local paths to the basename
    # (filename for YAML manifests, last directory for folders). HF repo
    # IDs (org/repo) are kept as-is.
    def _short_source(source):
        if source is None:
            return None
        p = Path(source)
        # YAML manifest: show filename only
        if p.suffix in (".yaml", ".yml"):
            return p.name
        # Local path (absolute or relative with many components): last part
        if p.is_absolute() or len(p.parts) > 2:
            return p.name
        # HF repo ID (org/repo) or short relative — keep as-is
        return source

    adapter_sources = [
        _short_source(t[3] if len(t) > 3 else None) for t in all_discovered
    ]
    # Deduped source → HF snapshot commit SHA. Adapters not resolved from
    # the HF Hub (local paths, YAML-declared, built-ins) are omitted.
    adapter_commits_by_source: dict = {}
    for t in all_discovered:
        source = t[3] if len(t) > 3 else None
        if not source or source in adapter_commits_by_source:
            continue
        commit = extract_hf_snapshot_commit(t[0])
        if commit:
            adapter_commits_by_source[source] = commit
    # CLI flags worth surfacing in the build doc. Flags with None / [] / ""
    # values are omitted by the renderer.
    built_in = getattr(args, "built_in_adapters", None) or []
    compose_settings = {
        "technology": getattr(args, "technology", None),
        "technology_filter": getattr(args, "technology_filter", None),
        "include_adapters": getattr(args, "include_adapters", None),
        "exclude_adapters": getattr(args, "exclude_adapters", None),
        "built_in_adapters": built_in or None,
        # lora_rank / lora_alpha are only meaningful when built-in adapters
        # are present; otherwise they're defaults that weren't consulted.
        "lora_rank": getattr(args, "lora_rank", None) if built_in else None,
        "lora_alpha": getattr(args, "lora_alpha", None) if built_in else None,
        "switch_head_dim": getattr(args, "switch_head_dim", None),
        "adapter_substitute_token_ids": getattr(
            model.config, "adapter_substitute_token_ids", None
        ),
        "target_model": getattr(args, "target_model", None),
    }
    # Parameter counts: base is captured during transfer (see
    # weight_transfer.py); composed is summed from the in-memory model.
    base_param_count = model._build_mappings["base"].get("base_param_count")
    composed_param_count = sum(p.numel() for p in model.parameters())
    # Per-adapter alpha. External adapters come from _build_mappings;
    # built-in slots fall back to the resolved --lora-alpha (which defaults
    # to --lora-rank when not passed). Order is external first, built-ins
    # last (see build() tuple assembly).
    adapter_alphas_src = model._build_mappings.get("adapter_alphas") or []
    adapter_alphas: list = list(adapter_alphas_src)
    built_in_alpha: Optional[float] = None
    if getattr(args, "built_in_adapters", None):
        built_in_alpha = float(
            args.lora_alpha if args.lora_alpha is not None else args.lora_rank
        )
    while len(adapter_alphas) < len(all_discovered):
        adapter_alphas.append(built_in_alpha)
    # Per-adapter target module sets (external adapters only). Built-in
    # slots have no source files to analyse, so we pad None.
    source_analysis = model._build_mappings.get("source_analysis") or {}
    adapter_targets_src = source_analysis.get("adapter_targets") or []
    adapter_targets: list = list(adapter_targets_src)
    while len(adapter_targets) < len(all_discovered):
        adapter_targets.append(None)
    build_doc_path = write_model_card(
        output_path=output_path,
        base_model_name=args.base_model,
        base_config=base_config,
        adapter_index=adapter_index,
        adapter_ranks=adapter_ranks,
        adapter_alphas=adapter_alphas,
        adapter_targets=adapter_targets,
        adapter_sources=adapter_sources,
        adapter_commits_by_source=adapter_commits_by_source,
        compose_settings=compose_settings,
        base_param_count=base_param_count,
        composed_param_count=composed_param_count,
    )
    print(f"Composed build doc written to: {build_doc_path.name}")
    return build_doc_path
