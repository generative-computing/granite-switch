# SPDX-License-Identifier: Apache-2.0
"""Adapter population table generation and printing."""

import torch

from ..adapter_loader import load_adapter_target_modules
from ..arch import ArchDescriptor


def generate_adapter_population_table(
    model,
    adapter_paths: list[str],
    adapter_names: list[str] | None = None,
    arch: ArchDescriptor = None,
    target_module_sets: list[set] | None = None,
):
    """Generate table showing how each module type was populated from each adapter.

    Args:
        model: The Granite Switch model.
        adapter_paths: List of adapter directory paths.
        adapter_names: Display names for each adapter.  When ``None``,
            derived from the directory structure.
        arch: Architecture descriptor (used to derive switch_to_peft and
            sliced-module detection).
        target_module_sets: Pre-computed target module sets (one per adapter).
            When ``None``, falls back to loading from disk via
            :func:`load_adapter_target_modules`.

    Returns:
        dict with keys ``module_types``, ``adapter_names``, ``entries``,
        ``adapter_ranks``, ``max_rank``.
    """
    # Get adapter information from model config
    num_adapters = model.config.num_adapters
    adapter_ranks = model.config.adapter_ranks
    max_rank = model.config.max_lora_rank

    # Build lookup structures from arch descriptor
    groups_by_name = {g.name: g for g in arch.groups}
    switch_to_peft = {g.name: list(g.peft_modules) for g in arch.groups}
    sliced_modules = {g.name for g in arch.groups if g.is_sliced}

    # Resolve adapter names
    if adapter_names is None:
        from pathlib import Path

        adapter_names = [Path(p).parent.parent.name for p in adapter_paths]

    if target_module_sets is None:
        target_module_sets = load_adapter_target_modules(adapter_paths)
    adapter_configs = [
        {
            "rank": adapter_ranks[i] if adapter_ranks else max_rank,
            "target_modules": target_module_sets[i],
        }
        for i in range(len(adapter_paths))
    ]

    # Define module types (rows) — only modules in lora_target_modules
    lora_target_modules = model.config.lora_target_modules
    module_types = []
    for module_name in lora_target_modules:
        module_types.append(f"{module_name}.lora_A")
        module_types.append(f"{module_name}.lora_B")

    state_dict = model.state_dict()

    entries = {}
    for module_type in module_types:
        entries[module_type] = {}

        base_module = module_type.rsplit(".lora_", 1)[0]
        is_lora_A = "lora_A" in module_type
        is_sliced = base_module in sliced_modules
        needs_padding_by = {
            i: (adapter_configs[i]["rank"] < max_rank) for i in range(num_adapters)
        }

        # Build the pattern fragment to match
        if is_sliced:
            lora_type = "lora_A_slices" if is_lora_A else "lora_B_slices"
        else:
            lora_type = "lora_A" if is_lora_A else "lora_B"

        if base_module in groups_by_name:
            attr = groups_by_name[base_module].effective_attr_name
        else:
            attr = base_module
        param_pattern = f"{attr}.{lora_type}"

        # Find matching params once per module type (not per adapter)
        matching_params = [
            (name, tensor)
            for name, tensor in state_dict.items()
            if param_pattern in name
        ]

        for adapter_idx in range(num_adapters):
            adapter_name = adapter_names[adapter_idx]
            target_modules = adapter_configs[adapter_idx]["target_modules"]
            needs_padding = needs_padding_by[adapter_idx]

            peft_modules = switch_to_peft.get(base_module, [])
            adapter_has_module = any(pm in target_modules for pm in peft_modules)

            has_nonzero = False
            for _param_name, param_tensor in matching_params:
                if param_tensor.dim() >= 1 and param_tensor.shape[0] > adapter_idx:
                    if is_sliced:
                        adapter_slice = param_tensor[adapter_idx]
                        for slice_idx in range(adapter_slice.shape[0]):
                            if torch.any(adapter_slice[slice_idx] != 0):
                                has_nonzero = True
                                break
                    else:
                        adapter_data = param_tensor[adapter_idx, 0]
                        if torch.any(adapter_data != 0):
                            has_nonzero = True

                    if has_nonzero:
                        break

            if not has_nonzero:
                if not adapter_has_module:
                    entries[module_type][adapter_name] = "zero-init"
                else:
                    entries[module_type][adapter_name] = "zero-init*"
            else:
                if not adapter_has_module:
                    entries[module_type][adapter_name] = "unexpected*"
                elif is_sliced and needs_padding:
                    entries[module_type][adapter_name] = "sliced+padded"
                elif is_sliced:
                    entries[module_type][adapter_name] = "sliced"
                elif needs_padding:
                    entries[module_type][adapter_name] = "copied+padded"
                else:
                    entries[module_type][adapter_name] = "copied"

    return {
        "module_types": module_types,
        "adapter_names": adapter_names,
        "entries": entries,
        "adapter_ranks": adapter_ranks,
        "max_rank": max_rank,
    }


def print_adapter_population_table(table_data):
    """Print the adapter population table in a readable format."""
    module_types = table_data["module_types"]
    adapter_names = table_data["adapter_names"]
    entries = table_data["entries"]
    adapter_ranks = table_data["adapter_ranks"]
    max_rank = table_data["max_rank"]

    # Print header with adapter info
    print("\nHow each module type was populated from each adapter:")
    print(f"  (max_rank={max_rank}, per-adapter ranks: {adapter_ranks})")
    print()

    # Calculate column widths
    module_col_width = max(len(mt) for mt in module_types) + 2
    adapter_col_width = max(15, max(len(name) for name in adapter_names) + 2)

    # Print header row
    header = f"{'Module Type':<{module_col_width}}"
    for adapter_name in adapter_names:
        header += f" | {adapter_name:<{adapter_col_width}}"
    print(header)
    print("-" * len(header))

    # Print data rows
    for module_type in module_types:
        row = f"{module_type:<{module_col_width}}"
        for adapter_name in adapter_names:
            status = entries[module_type].get(adapter_name, "unknown")
            row += f" | {status:<{adapter_col_width}}"
        print(row)

    # Print legend
    print("\nLegend:")
    print("  copied         : Straight copy from adapter (no padding)")
    print("  copied+padded  : Copied and zero-padded (adapter rank < max rank)")
    print("  sliced         : Sliced module (multi-PEFT or split, e.g., q/k/v -> qkv)")
    print("  sliced+padded  : Sliced and zero-padded")
    print("  zero-init      : Zero-initialized (adapter doesn't target this module)")
    print(
        "  zero-init*     : Config declares module but tensor is zero (loading failed)"
    )
    print("  unexpected*    : Config doesn't declare module but tensor has data")
