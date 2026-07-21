# SPDX-License-Identifier: Apache-2.0
"""Adapter loading, detection, and analysis utilities.

Adapter configuration detection, module discovery, file loading, and
source adapter analysis.
"""

import json
from pathlib import Path

import torch

from .arch import ArchDescriptor

# ---------------------------------------------------------------------------
# Shared config loader
# ---------------------------------------------------------------------------


def load_adapter_config(adapter_path: str) -> dict:
    """Load and return ``adapter_config.json`` from an adapter directory.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        Parsed JSON config dict.

    Raises:
        FileNotFoundError: If ``adapter_config.json`` does not exist.
    """
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    with open(config_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LoRA configuration detection
# ---------------------------------------------------------------------------


def detect_lora_config(
    adapter_paths: list[str],
) -> tuple[int, float, list[int], list[float]]:
    """Detect LoRA rank and alpha from adapter configs.

    Supports variable rank/alpha adapters.  Returns maximum rank for tensor
    allocation and per-adapter configuration for proper scaling.

    Args:
        adapter_paths: List of paths to adapter directories.

    Returns:
        ``(max_lora_rank, default_lora_alpha, adapter_ranks, adapter_alphas)``
    """
    print("Detecting LoRA configuration from adapters...")
    adapter_info: list[tuple[int, float]] = []

    for adapter_path in adapter_paths:
        config = load_adapter_config(adapter_path)

        rank = config.get("r")
        alpha = config.get("lora_alpha")

        if rank is None:
            raise ValueError(
                f"Could not find 'r' (rank) in adapter config: {adapter_path}"
            )
        if alpha is None:
            raise ValueError(
                f"Could not find 'lora_alpha' in adapter config: {adapter_path}"
            )

        adapter_info.append((rank, alpha))

    adapter_ranks_list = [info[0] for info in adapter_info]
    adapter_alphas_list = [info[1] for info in adapter_info]
    max_rank = max(adapter_ranks_list)
    default_alpha = adapter_alphas_list[0]

    unique_configs = set(adapter_info)

    if len(unique_configs) == 1:
        print("  Uniform configuration across all adapters:")
        print(f"    - Rank: {max_rank}")
        print(f"    - Alpha: {default_alpha}")
        print(f"    - Effective scaling (alpha/rank): {default_alpha / max_rank:.6f}")
    else:
        print("  Variable rank/alpha configuration detected:")
        print("    Adapter configurations:")

        default_scaling = default_alpha / max_rank

        for idx, (rank, alpha) in enumerate(adapter_info):
            adapter_scaling = alpha / rank
            scaling_ratio = adapter_scaling / default_scaling

            status = (
                "ok" if (rank == max_rank and alpha == default_alpha) else "variable"
            )
            padding_info = "" if rank == max_rank else f", +{max_rank - rank} padding"
            print(
                f"      {status} Adapter {idx}: rank={rank}, alpha={alpha}, "
                + f"scaling={adapter_scaling:.6f}, ratio={scaling_ratio:.4f}x{padding_info}"
            )

        print("    Model configuration:")
        print(f"      - Max rank (for allocation): {max_rank}")
        print(f"      - Default alpha (for config): {default_alpha}")
        print(f"      - Default scaling: {default_scaling:.6f}")
        print("    Per-adapter ranks/alphas will be stored in config")
        print(f"    Adapters with rank < {max_rank} will be zero-padded")

    return max_rank, default_alpha, adapter_ranks_list, adapter_alphas_list


# ---------------------------------------------------------------------------
# Module detection
# ---------------------------------------------------------------------------


def _extract_modules_from_weights(adapter_path: str) -> set:
    """Extract unique PEFT module names from adapter weight keys.

    Parses keys like ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    and returns the leaf module name before the lora suffix (e.g., ``"q_proj"``).
    """
    from pathlib import Path as _Path

    adapter_path_obj = _Path(adapter_path)
    safetensors_file = adapter_path_obj / "adapter_model.safetensors"
    bin_file = adapter_path_obj / "adapter_model.bin"

    if safetensors_file.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_file))
    elif bin_file.exists():
        state_dict = torch.load(str(bin_file), map_location="cpu")
    else:
        return set()

    modules = set()
    for key in state_dict.keys():
        for lora_kw in (".lora_A.", ".lora_B."):
            if lora_kw in key:
                # e.g. "...self_attn.q_proj.lora_A.weight" → "q_proj"
                before = key.split(lora_kw)[0]
                module_name = before.split(".")[-1]
                modules.add(module_name)
                break

    return modules


def detect_present_modules(
    adapter_paths: list[str],
    arch: ArchDescriptor,
    adapter_names: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Detect which module groups have adapters present in at least one adapter.

    Analyzes actual adapter weight files (not just configs) to determine which
    modules are populated.

    Args:
        adapter_paths: List of paths to adapter directories.
        arch: Architecture descriptor providing module group definitions.
        adapter_names: Display names for each adapter.  When ``None``,
            derived from the directory structure.

    Returns:
        ``(present_groups, source_analysis)`` — sorted list of Switch module
        group names present in at least one adapter, plus the raw analysis dict
        (reusable by :func:`generate_compose_report` to avoid re-loading files).
    """
    print(
        f"Detecting present LoRA module groups across {len(adapter_paths)} adapters..."
    )

    # Validate adapter compatibility using actual weight keys (ground truth).
    # This catches mismatches regardless of how target_modules is specified
    # (list or regex pattern) — we check what modules are actually present.
    known_peft = set(arch.all_peft_modules)
    display_names = adapter_names or [Path(p).parent.parent.name for p in adapter_paths]
    for idx, adapter_path in enumerate(adapter_paths):
        actual_modules = _extract_modules_from_weights(adapter_path)
        unknown = actual_modules - known_peft
        if unknown:
            name = display_names[idx] if idx < len(display_names) else adapter_path
            raise ValueError(
                f"Adapter '{name}' contains weights for modules {sorted(unknown)} "
                f"which are not recognized by the current architecture.\n"
                f"  Adapter modules (from weights): {sorted(actual_modules)}\n"
                f"  Architecture peft_modules: {sorted(known_peft)}\n"
                f"  Was this adapter trained for a different model type?"
            )

    print("  Using empirical analysis (checking actual weight files)...")

    analysis = analyze_source_adapters(
        adapter_paths,
        peft_modules=arch.all_peft_modules,
        adapter_names=adapter_names,
    )

    # Extract which PEFT modules are actually populated in at least one adapter
    peft_modules_populated: set = set()
    problem_count = 0

    for module_type in analysis["module_types"]:
        for adapter_name in analysis["adapter_names"]:
            status = analysis["status"][module_type].get(adapter_name, "unknown")

            if status in ["missing*", "zero*", "unexpected", "no-file"]:
                problem_count += 1

            if status == "populated":
                base_module = module_type.split(".lora_")[0]
                peft_modules_populated.add(base_module)
                break

    # Map PEFT modules to Switch module groups using arch descriptor
    switch_groups_present: set = set()
    for g in arch.groups:
        if any(mod in peft_modules_populated for mod in g.peft_modules):
            switch_groups_present.add(g.name)

    present_groups = sorted(switch_groups_present)

    all_groups = sorted(g.name for g in arch.groups)
    absent_groups = sorted(set(all_groups) - switch_groups_present)

    print(f"\n  Present module groups: {present_groups}")
    if absent_groups:
        print(f"  Absent module groups: {absent_groups}")
        print(
            f"  Performance: {len(absent_groups)} module group(s) will not be instantiated"
        )
    else:
        print("  All standard module groups have data")

    if problem_count > 0:
        print(
            f"\n  WARNING: Found {problem_count} problematic module/adapter combinations"
        )
        print("     Only modules with 'populated' status will be included")

    return present_groups, analysis


# ---------------------------------------------------------------------------
# Target module loading
# ---------------------------------------------------------------------------


def load_adapter_target_modules(
    adapter_paths: list[str],
) -> list[set]:
    """Load target_modules from each adapter's config.

    Returns an explicit list when available. String patterns (regex) cannot
    be expanded without the original training model, so an empty set is
    returned for them — compatibility validation uses weight keys instead.

    Args:
        adapter_paths: List of paths to adapter directories.

    Returns:
        List of sets, one per adapter, containing PEFT module names.

    Raises:
        FileNotFoundError: If ``adapter_config.json`` is missing.
    """
    result = []
    for adapter_path in adapter_paths:
        config = load_adapter_config(adapter_path)
        target_modules = config.get("target_modules", [])
        if isinstance(target_modules, list):
            result.append(set(target_modules))
        else:
            result.append(set())
    return result


# ---------------------------------------------------------------------------
# Adapter file loading
# ---------------------------------------------------------------------------


def load_adapter_files(
    adapter_paths: list[str],
) -> list[dict[str, torch.Tensor]]:
    """Load adapter weight files from disk.

    Supports both safetensors and PyTorch bin formats.

    Args:
        adapter_paths: List of paths to adapter directories.

    Returns:
        List of state dicts, one per adapter.
    """
    adapter_state_dicts = []
    for adapter_idx, adapter_path in enumerate(adapter_paths, start=1):
        print(f"  Loading adapter {adapter_idx} from {adapter_path}...")

        adapter_file_st = Path(adapter_path) / "adapter_model.safetensors"
        adapter_file_bin = Path(adapter_path) / "adapter_model.bin"

        if adapter_file_st.exists():
            from safetensors.torch import load_file

            adapter_state_dict = load_file(str(adapter_file_st))
        elif adapter_file_bin.exists():
            adapter_state_dict = torch.load(str(adapter_file_bin), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Could not find adapter_model.bin or "
                f"adapter_model.safetensors in {adapter_path}"
            )

        adapter_state_dicts.append(adapter_state_dict)

    return adapter_state_dicts


# ---------------------------------------------------------------------------
# Source adapter analysis
# ---------------------------------------------------------------------------


def analyze_source_adapters(
    adapter_paths: list[str],
    peft_modules: list[str],
    adapter_names: list[str] | None = None,
) -> dict:
    """Analyze source adapter files to understand their actual content.

    Compares what is in the adapter files against what the config declares
    as target modules, to help diagnose loading issues.

    Args:
        adapter_paths: List of paths to adapter directories.
        peft_modules: PEFT module names to check
            (e.g., from ``arch.all_peft_modules``).
        adapter_names: Display names for each adapter.  When ``None``,
            derived from the directory structure (``parent.parent.name``).

    Returns:
        Analysis dict with keys ``adapter_names``, ``module_types``,
        ``status``, ``adapter_ranks``, ``adapter_targets``, ``file_info``.
    """
    from safetensors.torch import load_file

    print("\n" + "=" * 80)
    print("ANALYZING SOURCE ADAPTER FILES")
    print("=" * 80)
    print("Comparing actual file content vs. declared target_modules...")

    resolved_names = []
    adapter_ranks = []
    adapter_targets = []
    file_info = []

    module_types = []
    for mod in peft_modules:
        module_types.append(f"{mod}.lora_A")
        module_types.append(f"{mod}.lora_B")

    status = {mt: {} for mt in module_types}

    for idx, adapter_path in enumerate(adapter_paths):
        adapter_path_obj = Path(adapter_path)

        # Use provided name or fall back to directory structure
        if adapter_names is not None:
            adapter_name = adapter_names[idx]
        else:
            adapter_name = adapter_path_obj.parent.parent.name
        resolved_names.append(adapter_name)

        # Load config
        config_file = adapter_path_obj / "adapter_config.json"
        if config_file.exists():
            config = load_adapter_config(adapter_path)
            rank = config.get("r", "?")
            adapter_ranks.append(rank)
            target_modules = config.get("target_modules", [])
            if isinstance(target_modules, str):
                target_modules = list(peft_modules)
            adapter_targets.append(set(target_modules))
        else:
            adapter_ranks.append(None)
            adapter_targets.append(set())

        # Find adapter weight file
        safetensors_file = adapter_path_obj / "adapter_model.safetensors"
        bin_file = adapter_path_obj / "adapter_model.bin"

        if safetensors_file.exists():
            weight_file = safetensors_file
            file_format = "safetensors"
        elif bin_file.exists():
            weight_file = bin_file
            file_format = "pytorch"
        else:
            print(f"  WARNING: No weight file found for {adapter_name}")
            file_info.append(
                {
                    "adapter": adapter_name,
                    "file": None,
                    "format": None,
                    "size_mb": 0,
                }
            )
            for module_type in module_types:
                status[module_type][adapter_name] = "no-file"
            continue

        file_size_mb = weight_file.stat().st_size / (1024 * 1024)
        file_info.append(
            {
                "adapter": adapter_name,
                "file": str(weight_file.name),
                "format": file_format,
                "size_mb": file_size_mb,
            }
        )

        # Load weights
        if file_format == "safetensors":
            state_dict = load_file(str(weight_file))
        else:
            state_dict = torch.load(str(weight_file), map_location="cpu")

        targets = adapter_targets[idx]

        for module_type in module_types:
            base_module = module_type.split(".lora_")[0]
            is_targeted = base_module in targets

            found = False
            has_nonzero = False
            for param_name, param_tensor in state_dict.items():
                if module_type in param_name and ".weight" in param_name:
                    found = True
                    if torch.any(param_tensor != 0):
                        has_nonzero = True
                        break

            if not is_targeted:
                if found:
                    status[module_type][adapter_name] = "unexpected"
                else:
                    status[module_type][adapter_name] = "not-targeted"
            else:
                if not found:
                    status[module_type][adapter_name] = "missing*"
                elif not has_nonzero:
                    status[module_type][adapter_name] = "zero*"
                else:
                    status[module_type][adapter_name] = "populated"

    return {
        "adapter_names": resolved_names,
        "module_types": module_types,
        "status": status,
        "adapter_ranks": adapter_ranks,
        "adapter_targets": adapter_targets,
        "file_info": file_info,
    }
