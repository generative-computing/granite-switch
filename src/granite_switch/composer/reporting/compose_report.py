# SPDX-License-Identifier: Apache-2.0
"""Compose report generation and printing."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from ..arch import ArchDescriptor


# ---------------------------------------------------------------------------
# Arch-driven parameter categorisation
# ---------------------------------------------------------------------------

def _build_categorizer(arch: ArchDescriptor):
    """Build ``(categorize_fn, category_order, category_names)`` from *arch*.

    Returns a classifier that maps a parameter name to a category key, plus
    display metadata (ordered list and pretty-names dict).
    """
    category_order: List[str] = ["embedding"]
    category_names: Dict[str, str] = {"embedding": "embedding"}

    # Build group match patterns (most-specific first)
    group_match: List[Tuple[str, str]] = []
    seen_parents: List[str] = []

    for g in arch.groups:
        attr = g.effective_attr_name
        key = f"{g.parent}_{g.name}"
        # Match ".parent.attr." in parameter path
        group_match.append((f".{g.parent}.{attr}.", key))
        category_order.append(key)
        category_names[key] = f"{g.parent} ({g.name})"
        if g.parent not in seen_parents:
            seen_parents.append(g.parent)

    # Parent-level "other" fallbacks
    for parent in seen_parents:
        key = f"{parent}_other"
        group_match.append((f".{parent}.", key))
        category_order.append(key)
        category_names[key] = f"{parent} (other)"

    category_order.extend(["normalization", "lm_head", "switch", "other"])
    category_names.update({
        "normalization": "normalization",
        "lm_head": "lm_head",
        "switch": "switch",
        "other": "other",
    })

    def categorize(param_name: str) -> str:
        if "embed_tokens" in param_name:
            return "embedding"
        if "lm_head" in param_name:
            return "lm_head"
        if "adapter_switch" in param_name:
            return "switch"
        for fragment, key in group_match:
            if fragment in param_name:
                return key
        if "norm" in param_name:
            return "normalization"
        return "other"

    return categorize, category_order, category_names


def _classify_adapter_target(target_name: str, arch: ArchDescriptor) -> str:
    """Classify an adapter target parameter name to ``parent.group``.

    E.g., ``"model.layers.0.self_attn.qkv_proj.lora_A_slices.0"``
    → ``"self_attn.qkv_proj"``.
    """
    key = arch.extract_module_key(target_name)
    return key if key else "other"


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_compose_report(
    base_mapping: Dict,
    adapter_mapping: Dict,
    output_path: str,
    model=None,
    adapter_paths: List[str] = None,
    adapter_names: List[str] = None,
    arch: ArchDescriptor = None,
    source_analysis: Dict = None,
):
    """Generate detailed compose report showing parameter mappings and statistics.

    Args:
        base_mapping: Mapping record from base weight transfer.
        adapter_mapping: Mapping record from adapter weight transfer.
        output_path: Path where report should be saved.
        model: Optional model to check dtype and size information.
        adapter_paths: Optional list of adapter directory paths.
        adapter_names: Display names for each adapter.
        arch: Architecture descriptor for categorisation.
        source_analysis: Pre-computed source adapter analysis (avoids
            re-loading adapter weight files if already computed).
    """
    from .population_table import generate_adapter_population_table
    from .adapter_analysis import print_source_adapter_analysis

    print("\n" + "="*80)
    print("GENERATING COMPOSE REPORT")
    print("="*80)

    # Analyze source adapters FIRST - before any loading/transformation
    if adapter_paths:
        if source_analysis is None:
            from ..adapter_loader import analyze_source_adapters

            source_analysis = analyze_source_adapters(
                adapter_paths,
                peft_modules=arch.all_peft_modules,
                adapter_names=adapter_names,
            )
        print_source_adapter_analysis(source_analysis)

    # ---- Build JSON report ----
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
        },
        "base_model_mapping": base_mapping["mappings"],
        "adapter_mapping": adapter_mapping["mappings"] if adapter_mapping else [],
    }

    # Save report
    report_path = Path(output_path) / "compose_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving detailed build report to: {report_path}")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Generate adapter population table
    adapter_population_table = None
    if model is not None and adapter_paths and getattr(model.config, "adapter_ranks", None):
        adapter_population_table = generate_adapter_population_table(
            model, adapter_paths, adapter_names=adapter_names, arch=arch,
            target_module_sets=source_analysis["adapter_targets"] if source_analysis else None,
        )

    # ---- Print summary ----
    _print_summary(
        report, model, base_mapping, adapter_mapping,
        adapter_population_table, report_path, arch,
    )


def _print_summary(
    report, model, base_mapping, adapter_mapping,
    adapter_population_table, report_path, arch,
):
    """Print the human-readable build report summary."""
    from .population_table import print_adapter_population_table

    # ---- Derive computed values ----
    categorize, category_order, category_names = _build_categorizer(arch)

    source_connected = set()
    target_connected = set()
    for mapping in report["base_model_mapping"]:
        for src in mapping["source"]:
            source_connected.add(src)
        target_connected.add(mapping["target"])
    if adapter_mapping:
        for mapping in report["adapter_mapping"]:
            for src in mapping["source"]:
                source_connected.add(src)
            target_connected.add(mapping["target"])

    mapping_types = {}
    total_source_in_mappings = 0
    fusion_stats = {}
    for mapping in report["base_model_mapping"]:
        mtype = mapping["type"]
        mapping_types[mtype] = mapping_types.get(mtype, 0) + 1
        num_sources = len(mapping["source"])
        total_source_in_mappings += num_sources
        if mtype not in fusion_stats:
            fusion_stats[mtype] = {"targets": 0, "sources": 0}
        fusion_stats[mtype]["targets"] += 1
        fusion_stats[mtype]["sources"] += num_sources

    layer_type_mappings: Dict[str, list] = {k: [] for k in category_order}
    for mapping in report["base_model_mapping"]:
        cat = categorize(mapping["target"])
        layer_type_mappings.setdefault(cat, []).append(mapping)

    adapter_layer_type_mappings: Dict[str, list] = {k: [] for k in category_order}
    if adapter_mapping:
        for mapping in report["adapter_mapping"]:
            cat = categorize(mapping["target"])
            adapter_layer_type_mappings.setdefault(cat, []).append(mapping)

    zero_initialized_adapter_targets = []
    if adapter_mapping and "target_params" in adapter_mapping:
        all_adapter_targets = set(adapter_mapping.get("target_params", []))
        zero_initialized_adapter_targets = sorted(
            all_adapter_targets - target_connected
        )

    # ---- Print ----
    print("\n" + "="*80)
    print("COMPOSE REPORT SUMMARY")
    print("="*80)

    # Model info
    if model is not None:
        sample_param = next(iter(model.parameters()))
        print(f"Model dtype: {sample_param.dtype}")
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"Model size: {total_bytes / (1024**3):.2f} GB")
        print()

    # Source/target module counts
    base_source_modules = base_mapping["source_params"]
    adapter_source_modules = adapter_mapping.get("source_params", []) if adapter_mapping else []
    base_target_modules = base_mapping["target_params"]
    adapter_target_modules = adapter_mapping.get("target_params", []) if adapter_mapping else []

    base_source_connected = set(base_source_modules) & source_connected
    adapter_source_connected = set(adapter_source_modules) & source_connected
    base_target_connected = set(base_target_modules) & target_connected
    adapter_target_connected = set(adapter_target_modules) & target_connected

    print(f"Source Modules: {len(base_source_modules) + len(adapter_source_modules):,}")
    print(f"  Base: {len(base_source_modules):,} (connected: {len(base_source_connected):,})")
    if adapter_source_modules:
        print(f"  Adapters: {len(adapter_source_modules):,} (connected: {len(adapter_source_connected):,})")

    print(f"\nTarget Modules: {len(base_target_modules) + len(adapter_target_modules):,}")
    print(f"  Base: {len(base_target_modules):,} (connected: {len(base_target_connected):,})")
    if adapter_target_modules:
        adapter_not_connected = len(adapter_target_modules) - len(adapter_target_connected)
        print(f"  Adapters: {len(adapter_target_modules):,} (connected: {len(adapter_target_connected):,}, not connected: {adapter_not_connected:,})")

    # Adapter mapping type counts
    adapter_mapping_types = {}
    total_adapter_sources = 0
    if adapter_mapping:
        for mapping in report['adapter_mapping']:
            mtype = mapping['type']
            adapter_mapping_types[mtype] = adapter_mapping_types.get(mtype, 0) + 1
            total_adapter_sources += len(mapping['source'])

    # Fusion summary
    print(f"\nModule Fusion Summary:")
    if total_source_in_mappings != len(report['base_model_mapping']):
        print(f"  Base: {total_source_in_mappings} source modules -> {len(report['base_model_mapping'])} target modules")
        print(f"        (reduction: {total_source_in_mappings - len(report['base_model_mapping'])} modules due to fusion)")
    else:
        print(f"  Base: {total_source_in_mappings} source modules -> {len(report['base_model_mapping'])} target modules (1->1)")

    if adapter_mapping:
        print(f"  Adapters: {total_adapter_sources} source modules -> {len(report['adapter_mapping'])} target modules")
        if total_adapter_sources != len(report['adapter_mapping']):
            print(f"            (reduction: {total_adapter_sources - len(report['adapter_mapping'])} modules due to stacking)")

    # Mapping details
    print(f"\nMapping Details:")
    total_mappings = len(report['base_model_mapping']) + len(report['adapter_mapping'])
    print(f"  Total mappings: {total_mappings}")
    print(f"    Base model: {len(report['base_model_mapping'])}")
    for mtype in sorted(mapping_types.keys()):
        count = mapping_types[mtype]
        stats = fusion_stats[mtype]
        if stats['sources'] == stats['targets']:
            print(f"      - {mtype}: {count} (1->1)")
        else:
            ratio = stats['sources'] // stats['targets'] if stats['targets'] > 0 else 0
            print(f"      - {mtype}: {count} ({ratio}->1, {stats['sources']} sources -> {stats['targets']} targets)")

    if adapter_mapping and adapter_mapping_types:
        print(f"    Adapters: {len(report['adapter_mapping'])}")
        sources_per_target = total_adapter_sources // len(report['adapter_mapping']) if report['adapter_mapping'] else 0
        print(f"      (Each target module stacks {sources_per_target} adapters in dimension 0)")

        _print_adapter_projection_breakdown(
            adapter_mapping_types, report['adapter_mapping'], arch,
        )

    # Layer type breakdown (using shared category_order / category_names)
    _print_layer_type_breakdown(
        layer_type_mappings, adapter_layer_type_mappings,
        category_order, category_names, adapter_mapping,
    )

    # Population table
    if adapter_population_table:
        print("\n" + "="*80)
        print("ADAPTER MODULE POPULATION TABLE")
        print("="*80)
        print_adapter_population_table(adapter_population_table)

    # Zero-initialized adapter targets
    if zero_initialized_adapter_targets:
        print(f"\nAdapter Targets Not Loaded from Source: {len(zero_initialized_adapter_targets)}")
        print(f"  (Expected: These are adapter modules missing from source adapters or zero-padded)")
        print(f"  (See detailed validation output above for breakdown by reason)")

    # Unmapped base sources
    source_not_connected = sorted(
        set(base_source_modules + adapter_source_modules) - source_connected
    )
    if source_not_connected:
        base_source_not_connected = [
            name for name in source_not_connected
            if not name.startswith("adapter_")
        ]
        if base_source_not_connected:
            print(f"\n  Base source modules not connected: {len(base_source_not_connected)}")
            print(f"  (First 10):")
            for name in base_source_not_connected[:10]:
                print(f"    - {name}")
            if len(base_source_not_connected) > 10:
                print(f"  ... and {len(base_source_not_connected) - 10} more")

    print(f"\nDetailed report saved to: {report_path}")
    print("="*80)


def _print_adapter_projection_breakdown(
    adapter_mapping_types: Dict[str, int],
    adapter_mappings: list,
    arch: ArchDescriptor,
):
    """Print adapter mapping breakdown grouped by parent module.

    Classifies each mapping's target name via the arch descriptor.
    """
    # Count targets and sources per module key
    module_targets: Dict[str, int] = {}
    module_sources: Dict[str, int] = {}
    for mapping in adapter_mappings:
        key = _classify_adapter_target(mapping['target'], arch)
        module_targets[key] = module_targets.get(key, 0) + 1
        module_sources[key] = module_sources.get(key, 0) + len(mapping['source'])

    # Group by parent for display
    by_parent: Dict[str, List[str]] = defaultdict(list)
    for key in sorted(module_targets.keys()):
        parent = key.split(".")[0] if "." in key else "other"
        by_parent[parent].append(key)

    for parent in sorted(by_parent.keys()):
        print(f"      {parent}:")
        for key in by_parent[parent]:
            target_count = module_targets[key]
            source_count = module_sources.get(key, 0)
            ratio = source_count // target_count if target_count > 0 else 0
            print(f"        - {key}: {target_count} targets ({source_count} sources, {ratio}->1 stacking)")


def _print_layer_type_breakdown(
    layer_type_mappings, adapter_layer_type_mappings,
    category_order, category_names, adapter_mapping,
):
    """Print layer type breakdown using shared category metadata."""
    print(f"\nLayer Type Breakdown:")
    print(f"  Base Model:")
    has_any = False
    for cat in category_order:
        layer_mappings = layer_type_mappings.get(cat, [])
        if not layer_mappings:
            continue
        has_any = True
        count = len(layer_mappings)
        display = category_names.get(cat, cat)
        total_sources = sum(len(m["source"]) for m in layer_mappings)
        if total_sources == count:
            print(f"    - {display}: {count} modules (1->1)")
        else:
            print(f"    - {display}: {count} modules ({total_sources} sources -> {count} targets)")
    if not has_any:
        print(f"    (none)")

    if adapter_mapping:
        adapter_has_any = any(adapter_layer_type_mappings.get(cat) for cat in category_order)
        if adapter_has_any:
            print(f"  Adapters:")
            for cat in category_order:
                adapter_mappings = adapter_layer_type_mappings.get(cat, [])
                if not adapter_mappings:
                    continue
                count = len(adapter_mappings)
                display = category_names.get(cat, cat)
                print(f"    - {display}: {count} modules")
