# SPDX-License-Identifier: Apache-2.0
"""Weight transfer logic for base model and adapter weights.

Driven by :class:`ArchDescriptor` fusion rules instead of inline if/elif chains.
"""

import gc
import re

import torch
from tqdm import tqdm

from .arch import ArchDescriptor

# ---------------------------------------------------------------------------
# Base weight loading
# ---------------------------------------------------------------------------


def _load_base_state_dict(
    model_name_or_path: str,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], int]:
    """Load base model state dict via ``AutoModelForCausalLM``.

    Returns the state dict and the unique-parameter count (from
    ``model.parameters()``) so the caller can report the same figure the
    composed model will report via ``sum(p.numel() for p in model.parameters())``.
    """
    from transformers import AutoModelForCausalLM

    temp_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    state_dict = temp_model.state_dict()
    param_count = sum(p.numel() for p in temp_model.parameters())
    del temp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state_dict, param_count


# ---------------------------------------------------------------------------
# Base weight transfer
# ---------------------------------------------------------------------------


def transfer_base_weights(
    base_model_name_or_path: str,
    model,
    switch_config,
    arch: ArchDescriptor,
    return_mapping: bool = True,
) -> dict | None:
    """Load base model weights and transfer to switch model.

    Uses ``arch.groups`` to drive QKV and gate/up fusion instead of
    hardcoded if/elif chains.

    Args:
        base_model_name_or_path: Path or HF model ID for base model.
        model: GraniteSwitch model to load weights into.
        switch_config: Switch configuration.
        arch: Architecture descriptor.
        return_mapping: If True, return detailed mapping information.

    Returns:
        Mapping record dict, or None if *return_mapping* is False.
    """
    mapping_record = {
        "source_params": [],
        "target_params": [],
        "mappings": [],
        # None until the base state dict has been loaded successfully; the
        # renderer treats None as "unknown" and omits the param-count section
        # rather than printing a misleading zero.
        "base_param_count": None,
    }

    print("Loading base model weights from checkpoint...")
    original_dtype = getattr(switch_config, "torch_dtype", None) or torch.float32
    print(f"  Base model will be loaded in: {original_dtype}")

    base_state_dict, base_param_count = _load_base_state_dict(
        base_model_name_or_path, original_dtype
    )

    print("Transferring base model weights to GraniteSwitch...")

    mapping_record["source_params"] = list(base_state_dict.keys())
    mapping_record["base_param_count"] = base_param_count

    switch_state_dict = model.state_dict()

    # Record target params (excluding LoRA and buffers)
    exclude_keywords = arch.lora_keywords + arch.buffer_keywords
    mapping_record["target_params"] = [
        name
        for name in switch_state_dict.keys()
        if not any(kw in name for kw in exclude_keywords)
    ]

    lora_target_modules = switch_config.lora_target_modules

    # ---- Classify every base weight ----
    fused_collections, base_to_switch = _classify_base_weights(
        base_state_dict,
        arch,
        lora_target_modules,
    )

    # ---- Transfer weights ----
    groups_by_name = {g.name: g for g in arch.groups}
    transferred_count = 0
    with torch.no_grad():
        # Direct mappings
        for base_name, base_param in tqdm(
            base_state_dict.items(), desc="Transferring weights", unit="tensor"
        ):
            if base_name in base_to_switch:
                switch_name = base_to_switch[base_name]
                if switch_name in switch_state_dict:
                    switch_state_dict[switch_name].copy_(base_param)
                    transferred_count += 1
                    mapping_record["mappings"].append(
                        {
                            "source": [base_name],
                            "target": switch_name,
                            "type": "direct",
                        }
                    )

        # Fused mappings
        for (layer_idx, target_name), collection in fused_collections.items():
            g = groups_by_name[target_name]
            attr = g.effective_attr_name
            if target_name in lora_target_modules:
                switch_name = (
                    f"model.layers.{layer_idx}.{g.parent}.{attr}.base_layer.weight"
                )
            else:
                switch_name = f"model.layers.{layer_idx}.{g.parent}.{attr}.weight"

            if "fused" in collection:
                # Already fused (from remapper, e.g., Granite 4-micro shared_mlp)
                if switch_name in switch_state_dict:
                    switch_state_dict[switch_name].copy_(collection["fused"])
                    transferred_count += 1
                    mapping_record["mappings"].append(
                        {
                            "source": collection["source_names"],
                            "target": switch_name,
                            "type": "direct_fused",
                        }
                    )
            else:
                # Fuse from separate sources
                if all(src in collection for src in g.peft_modules):
                    fused_tensor = torch.cat(
                        [collection[src] for src in g.peft_modules], dim=0
                    )
                    if switch_name in switch_state_dict:
                        switch_state_dict[switch_name].copy_(fused_tensor)
                        transferred_count += len(g.peft_modules)
                        mapping_record["mappings"].append(
                            {
                                "source": collection["source_names"],
                                "target": switch_name,
                                "type": f"fused_{target_name}",
                            }
                        )

    print(f"\nTransferred {transferred_count} weight tensors from base model")

    # ---- Validate ----
    _validate_base_transfer(
        switch_state_dict,
        base_to_switch,
        fused_collections,
        lora_target_modules,
        arch,
    )

    # Free base state dict
    del base_state_dict
    gc.collect()
    print("Base weights loaded and temporary data freed")

    return mapping_record if return_mapping else None


def _classify_base_weights(
    base_state_dict: dict[str, torch.Tensor],
    arch: ArchDescriptor,
    lora_target_modules: list[str],
):
    """Classify every base model weight into fusion collections or direct mappings.

    Returns:
        ``(fused_collections, base_to_switch)`` — fused_collections maps
        ``(layer_idx, target_name)`` to collected tensors; base_to_switch
        maps base param names to their switch model counterparts.
    """
    # Build lookup structures from groups
    source_to_group: dict[str, object] = {}  # peft_module -> ModuleDescriptor
    fusion_names = set()
    for g in arch.groups:
        if g.is_base_fusion:
            fusion_names.add(g.name)
            for src in g.peft_modules:
                source_to_group[src] = g

    fused_collections: dict = {}
    base_to_switch: dict[str, str] = {}

    for base_name in base_state_dict.keys():
        matched = False

        # (A) Fusion source? (q_proj, k_proj, v_proj, gate_proj, up_proj)
        for src_module, g in source_to_group.items():
            if f".{src_module}.weight" in base_name:
                layer_match = re.search(arch.layer_pattern, base_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    key = (layer_idx, g.name)
                    if key not in fused_collections:
                        fused_collections[key] = {"source_names": []}
                    fused_collections[key][src_module] = base_state_dict[base_name]
                    fused_collections[key]["source_names"].append(base_name)
                    matched = True
                break  # src_module matched the name pattern; stop checking others

        if matched:
            continue

        # (B) Already-fused target? (e.g., gate_up_proj.weight)
        already_fused = False
        for target in fusion_names:
            if f".{target}.weight" in base_name:
                layer_match = re.search(arch.layer_pattern, base_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    key = (layer_idx, target)
                    if key not in fused_collections:
                        fused_collections[key] = {"source_names": []}
                    fused_collections[key]["fused"] = base_state_dict[base_name]
                    fused_collections[key]["source_names"].append(base_name)
                    already_fused = True
                break

        if already_fused:
            continue

        # (C) Standalone LoRA target module? (o_proj, down_proj)
        is_standalone = False
        for g in arch.groups:
            if g.is_base_fusion:
                continue
            src_name = g.peft_modules[0]
            if f".{src_name}.weight" in base_name:
                # Verify the weight belongs to the expected parent module.
                # Without this, MoE expert weights (e.g.,
                # block_sparse_moe.experts.N.input_linear) falsely match
                # shared_mlp groups that also use "input_linear".
                if f".{g.effective_source_parent}." not in base_name:
                    break  # Wrong parent — fall through to (D)
                layer_match = re.search(arch.layer_pattern, base_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    target_attr = g.effective_attr_name
                    inner = g.target_inner_path
                    if g.name in lora_target_modules:
                        switch_name = (
                            f"model.layers.{layer_idx}.{g.parent}.{target_attr}."
                            f"{inner}base_layer.weight"
                        )
                    else:
                        switch_name = (
                            f"model.layers.{layer_idx}.{g.parent}.{target_attr}.weight"
                        )
                    base_to_switch[base_name] = switch_name
                    is_standalone = True
                break

        if is_standalone:
            continue

        # (D) Everything else: direct mapping (embeddings, norms, lm_head)
        base_to_switch[base_name] = base_name

    return fused_collections, base_to_switch


def _validate_base_transfer(
    switch_state_dict,
    base_to_switch,
    fused_collections,
    lora_target_modules,
    arch: ArchDescriptor,
):
    """Validate that all expected base parameters were loaded."""
    exclude_keywords = arch.lora_keywords + arch.buffer_keywords
    expected_switch_params = {
        name
        for name in switch_state_dict.keys()
        if not any(kw in name for kw in exclude_keywords)
    }

    loaded_switch_params = set(base_to_switch.values())

    # Add fused params
    groups_by_name = {g.name: g for g in arch.groups}
    for (layer_idx, target_name), _collection in fused_collections.items():
        g = groups_by_name[target_name]
        parent = g.parent
        attr = g.effective_attr_name
        if target_name in lora_target_modules:
            loaded_switch_params.add(
                f"model.layers.{layer_idx}.{parent}.{attr}.base_layer.weight"
            )
        else:
            loaded_switch_params.add(f"model.layers.{layer_idx}.{parent}.{attr}.weight")

    missing_in_switch = loaded_switch_params - expected_switch_params
    if missing_in_switch:
        print(
            f"\nWARNING: {len(missing_in_switch)} base parameters "
            f"not found in switch model:"
        )
        for name in sorted(list(missing_in_switch)[:10]):
            print(f"  - {name}")
        if len(missing_in_switch) > 10:
            print(f"  ... and {len(missing_in_switch) - 10} more")
        raise ValueError(
            f"Base model has {len(missing_in_switch)} parameters "
            f"not found in switch model"
        )

    missing_in_base = expected_switch_params - loaded_switch_params
    if missing_in_base:
        print(
            f"\nWARNING: {len(missing_in_base)} switch parameters not loaded from base:"
        )
        for name in sorted(list(missing_in_base)[:10]):
            print(f"  - {name}")
        if len(missing_in_base) > 10:
            print(f"  ... and {len(missing_in_base) - 10} more")
        raise ValueError(
            f"Switch model has {len(missing_in_base)} parameters "
            f"not loaded from base model"
        )

    print(
        f"All {len(expected_switch_params)} base model parameters "
        f"successfully validated"
    )


# ---------------------------------------------------------------------------
# Adapter stacking (separated from WeightRemapper in Stage 3)
# ---------------------------------------------------------------------------


def stack_adapters(
    adapter_state_dicts: list[dict[str, torch.Tensor]],
    remapper,
    num_adapters: int,
    max_lora_rank: int,
    adapter_ranks: list[int],
    adapter_alphas: list[float],
    verbose: bool = True,
):
    """Stack adapter weights into ``[num_adapters, 1, ...]`` tensors.

    Uses ``remapper.remap_adapter_name()`` for name remapping, then handles
    stacking, zero-padding, lora_B pre-scaling, and fused-to-sliced splitting.

    Split handling (for natively fused modules like GraniteMoeHybrid's
    ``input_linear``):

    - ``split_type="duplicate"``: Copy the same tensor to each slice.
      Used for lora_A where both gate and up share the same input space.
    - ``split_type="chunk_dim0"``: Chunk tensor along dim=0 into N slices.
      Used for lora_B where output dim is ``[gate_size | up_size]``.

    Args:
        adapter_state_dicts: List of adapter state dicts (one per adapter).
        remapper: WeightRemapper instance (only ``remap_adapter_name`` is used).
        num_adapters: Total number of adapters.
        max_lora_rank: Maximum rank across all adapters.
        adapter_ranks: Per-adapter ranks.
        adapter_alphas: Per-adapter alpha values.
        verbose: Print remapping information.

    Returns:
        ``(stacked, mappings)`` — stacked state dict ready for Granite Switch,
        and a list of ``{"source": src, "target": target, "type": mtype}``
        dicts recorded during stacking.
    """
    stacked: dict[str, torch.Tensor] = {}
    # Track source→target mappings: target_name → set of source names
    source_map: dict[str, set] = {}
    remap_count = 0

    for adapter_idx, adapter_state_dict in enumerate(adapter_state_dicts):
        adapter_rank = adapter_ranks[adapter_idx]
        adapter_alpha = adapter_alphas[adapter_idx]

        for src_name, tensor in adapter_state_dict.items():
            result = remapper.remap_adapter_name(src_name)
            if result is None:
                continue

            tagged_src = f"adapter_{adapter_idx}::{src_name}"

            if result.split_slices:
                # Fused-to-sliced split: distribute tensor across slices
                _stack_split(
                    stacked,
                    result,
                    tensor,
                    adapter_idx,
                    adapter_rank,
                    adapter_alpha,
                    num_adapters,
                    max_lora_rank,
                    src_name,
                )
                # Record mapping for each produced slice
                for i in range(result.split_slices):
                    slice_name = f"{result.target_name}.{i}"
                    source_map.setdefault(slice_name, set()).add(tagged_src)
            else:
                # Standard (non-split) stacking
                target_name = result.target_name
                _stack_single(
                    stacked,
                    target_name,
                    tensor,
                    adapter_idx,
                    adapter_rank,
                    adapter_alpha,
                    num_adapters,
                    max_lora_rank,
                    src_name,
                )
                source_map.setdefault(target_name, set()).add(tagged_src)

            if verbose and remap_count < 5:
                print(
                    f"  Stack: {src_name} -> {result.target_name}[{adapter_idx}, 0]"
                    + (
                        f" (split: {result.split_type}x{result.split_slices})"
                        if result.split_slices
                        else ""
                    )
                )
            remap_count += 1

    if verbose and remap_count > 0:
        print(f"Stacked {remap_count} adapter weights across {num_adapters} adapters")
        if remap_count > 5:
            print("  (showing first 5 examples)")

    # Convert source_map to mapping records
    mappings = [
        {
            "source": sorted(sources),
            "target": target,
            "type": "adapter_stacked",
        }
        for target, sources in sorted(source_map.items())
    ]

    return stacked, mappings


def _ensure_stacked(stacked, name, is_lora_a, tensor, num_adapters, max_lora_rank):
    """Create zero-initialized stacked tensor if *name* is not already present."""
    if name not in stacked:
        if is_lora_a:
            shape = (num_adapters, 1, max_lora_rank, tensor.shape[1])
        else:
            shape = (num_adapters, 1, tensor.shape[0], max_lora_rank)
        stacked[name] = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)


def _stack_single(
    stacked,
    target_name,
    tensor,
    adapter_idx,
    adapter_rank,
    adapter_alpha,
    num_adapters,
    max_lora_rank,
    src_name,
):
    """Stack a single (non-split) adapter tensor into the stacked dict."""
    is_lora_a = "lora_A" in src_name
    _ensure_stacked(
        stacked, target_name, is_lora_a, tensor, num_adapters, max_lora_rank
    )

    if is_lora_a:
        stacked[target_name][adapter_idx, 0, :adapter_rank, :] = tensor
    else:
        scaling_factor = adapter_alpha / adapter_rank
        stacked[target_name][adapter_idx, 0, :, :adapter_rank] = tensor * scaling_factor


def _stack_split(
    stacked,
    result,
    tensor,
    adapter_idx,
    adapter_rank,
    adapter_alpha,
    num_adapters,
    max_lora_rank,
    src_name,
):
    """Handle fused-to-sliced split: distribute one tensor across N slices.

    For ``split_type="duplicate"`` (lora_A): both gate and up share the same
    input space, so copy the identical tensor to each slice.

    For ``split_type="chunk_dim0"`` (lora_B): the output dimension is
    ``[gate_size | up_size]``, so chunk along dim=0 and assign each chunk
    to its corresponding slice.
    """
    n = result.split_slices
    base_name = result.target_name

    if result.split_type == "duplicate":
        # lora_A: same input projection for all slices
        for i in range(n):
            slice_name = f"{base_name}.{i}"
            _ensure_stacked(
                stacked, slice_name, True, tensor, num_adapters, max_lora_rank
            )
            stacked[slice_name][adapter_idx, 0, :adapter_rank, :] = tensor

    elif result.split_type == "chunk_dim0":
        # lora_B: split output dimension across slices
        chunks = tensor.chunk(n, dim=0)
        scaling_factor = adapter_alpha / adapter_rank
        for i, chunk in enumerate(chunks):
            slice_name = f"{base_name}.{i}"
            _ensure_stacked(
                stacked, slice_name, False, chunk, num_adapters, max_lora_rank
            )
            stacked[slice_name][adapter_idx, 0, :, :adapter_rank] = (
                chunk * scaling_factor
            )

    else:
        raise ValueError(f"Unknown split_type: {result.split_type}")


# ---------------------------------------------------------------------------
# Adapter weight transfer
# ---------------------------------------------------------------------------


def transfer_adapter_weights(
    adapter_paths: list[str],
    model,
    adapter_alphas: list[float],
    arch: ArchDescriptor,
    return_mapping: bool = True,
) -> dict | None:
    """Load adapter weights, stack them, and transfer to switch model.

    Uses :func:`stack_adapters` for stacking and
    :meth:`AdapterRemapper.remap_adapter_name` for name remapping.

    Args:
        adapter_paths: List of paths to adapter directories.
        model: GraniteSwitch model to load adapters into.
        adapter_alphas: Per-adapter alpha values.
        arch: Architecture descriptor.
        return_mapping: If True, return detailed mapping information.

    Returns:
        Mapping record dict, or None if *return_mapping* is False.
    """
    from .adapter_loader import load_adapter_files

    mapping_record = {
        "source_params": [],
        "target_params": [],
        "mappings": [],
    }

    print(f"Loading {len(adapter_paths)} LoRA adapters...")
    adapter_state_dicts = load_adapter_files(adapter_paths)

    # Record source params
    source_adapter_params = []
    for adapter_idx, adapter_dict in enumerate(adapter_state_dicts):
        for param_name in adapter_dict.keys():
            source_adapter_params.append(f"adapter_{adapter_idx}::{param_name}")
    mapping_record["source_params"] = sorted(source_adapter_params)

    # Remap and stack — adapter remapping is derived from arch descriptor
    print("Remapping and stacking adapter weights...")
    adapter_remapper = arch.build_adapter_remapper()
    stacked_adapters, adapter_mappings = stack_adapters(
        adapter_state_dicts=adapter_state_dicts,
        remapper=adapter_remapper,
        num_adapters=model.config.num_adapters,
        max_lora_rank=model.config.max_lora_rank,
        adapter_ranks=model.config.adapter_ranks[: len(adapter_paths)],
        adapter_alphas=adapter_alphas,
        verbose=True,
    )

    # Record target params and mappings
    switch_state_dict = model.state_dict()
    mapping_record["target_params"] = [
        name
        for name in switch_state_dict.keys()
        if any(kw in name for kw in arch.lora_keywords)
    ]
    mapping_record["mappings"] = adapter_mappings

    # Copy stacked weights into model
    loaded_adapter_count = 0
    with torch.no_grad():
        for name, tensor in stacked_adapters.items():
            if name in switch_state_dict:
                switch_state_dict[name].copy_(tensor)
                loaded_adapter_count += 1

    print(f"\nLoaded {loaded_adapter_count} adapter parameter tensors")

    # Validate
    _validate_adapter_transfer(
        switch_state_dict,
        stacked_adapters,
        arch,
    )

    # Free memory
    del adapter_state_dicts
    gc.collect()
    print("Adapter weights loaded and temporary data freed")

    return mapping_record if return_mapping else None


def _validate_adapter_transfer(
    switch_state_dict,
    stacked_adapters,
    arch: ArchDescriptor,
):
    """Validate adapter parameter transfer.

    Checks that every stacked adapter param landed in the switch model.
    Zero-initialized LoRA params (adapters that don't target certain modules)
    are reported by :func:`validator.validate_all_parameters`, not here.
    """
    expected_lora_params = {
        name
        for name in switch_state_dict.keys()
        if any(kw in name for kw in arch.lora_keywords)
    }
    loaded_lora_params = set(stacked_adapters.keys())

    missing_in_switch = loaded_lora_params - expected_lora_params
    if missing_in_switch:
        print(
            f"\nWARNING: {len(missing_in_switch)} adapter parameters "
            f"not found in switch model:"
        )
        for name in sorted(list(missing_in_switch)[:10]):
            print(f"  - {name}")
        if len(missing_in_switch) > 10:
            print(f"  ... and {len(missing_in_switch) - 10} more")
        raise ValueError(
            f"Adapters have {len(missing_in_switch)} parameters "
            f"not found in switch model"
        )

    zero_init_count = len(expected_lora_params - loaded_lora_params)
    if zero_init_count:
        print(
            f"\n  {zero_init_count} switch LoRA parameters zero-initialized "
            f"(detailed breakdown in validation step)"
        )

    print(f"All {len(expected_lora_params)} LoRA parameters accounted for")
