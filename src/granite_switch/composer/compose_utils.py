# SPDX-License-Identifier: Apache-2.0
"""Granite Switch model composer — thin orchestrator.

Delegates to :mod:`arch`, :mod:`adapter_loader`, :mod:`weight_transfer`, and
:mod:`validator` for the heavy lifting.
"""

import torch

from .adapter_loader import detect_lora_config, detect_present_modules
from .arch import resolve_arch
from .validator import validate_all_parameters
from .weight_transfer import transfer_adapter_weights, transfer_base_weights


class GraniteSwitchComposer:
    """Composer for creating Granite Switch models from base + adapters."""

    @classmethod
    def from_base_and_adapters(
        cls,
        base_model_name_or_path: str,
        adapter_paths: list[str] | None = None,
        adapter_token_ids: list[int] | None = None,
        adapter_substitute_token_ids: list[int] | None = None,
        adapter_names: list[str] | None = None,
        built_in_adapter_names: list[str] | None = None,
        built_in_lora_rank: int = 8,
        built_in_lora_alpha: float = 8.0,
        **kwargs,
    ):
        """Create a GraniteSwitch model from base model and LoRA adapters.

        This method:
        1. Auto-detects the base model architecture
        2. Detects LoRA rank/alpha from adapter configs
        3. Detects which module groups are present
        4. Builds the Switch config from architecture descriptor fields
        5. Transfers base weights (with arch-driven fusion)
        6. Transfers adapter weights (stacking)
        7. Validates all parameters

        Args:
            base_model_name_or_path: Path or HF model ID for base model.
            adapter_paths: Paths to LoRA adapter checkpoints.  ``None`` or
                empty for zero-adapter skinning (base model only).
            adapter_token_ids: Token IDs for adapter control.  Required when
                ``adapter_paths`` is non-empty.
            adapter_substitute_token_ids: Token IDs whose embeddings replace
                control-token embeddings at the switch. Required when
                ``adapter_paths`` is non-empty; one per adapter.
            adapter_names: Display names for each adapter (external + built-in).
                When ``None``, derived from the directory structure.
            built_in_adapter_names: Names for built-in (empty LoRA) adapter slots.
            built_in_lora_rank: LoRA rank for built-in adapters.
            built_in_lora_alpha: LoRA alpha for built-in adapters.
            **kwargs: Additional arguments passed to ``GraniteSwitchConfig``.

        Returns:
            ``GraniteSwitchForCausalLM`` with adapters loaded and switch configured.
        """
        from granite_switch.config import GraniteSwitchConfig
        from granite_switch.hf.modeling_granite_switch import GraniteSwitchForCausalLM

        from .arch import load_base_config

        if adapter_paths is None:
            adapter_paths = []
        if built_in_adapter_names is None:
            built_in_adapter_names = []

        num_external = len(adapter_paths)
        num_built_in = len(built_in_adapter_names)
        num_total = num_external + num_built_in

        # --- Step 1: Resolve architecture ---
        print(f"Loading config from {base_model_name_or_path}...")
        base_config = load_base_config(base_model_name_or_path)
        arch = resolve_arch(base_model_name_or_path, base_config=base_config)

        # --- Step 2–3: Detect LoRA config and present modules ---
        if adapter_paths:
            lora_rank, lora_alpha, adapter_ranks, adapter_alphas = detect_lora_config(
                adapter_paths
            )
            lora_target_modules, source_analysis = detect_present_modules(
                adapter_paths,
                arch,
                adapter_names=adapter_names,
            )

            # Extend adapter_ranks with built-in entries
            if num_built_in > 0:
                if built_in_lora_rank != lora_rank:
                    raise ValueError(
                        f"Built-in LoRA rank ({built_in_lora_rank}) must match "
                        f"external adapter rank ({lora_rank}). "
                        f"All adapters must have the same rank."
                    )
                adapter_ranks = (
                    list(adapter_ranks) + [built_in_lora_rank] * num_built_in
                )
                lora_rank = max(lora_rank, built_in_lora_rank)
        else:
            # Built-in only or zero-adapter skinning
            if num_built_in > 0:
                lora_rank = built_in_lora_rank
                adapter_ranks = [built_in_lora_rank] * num_built_in
                adapter_alphas = {}
                # Auto-detect lora_target_modules from layer_types
                lora_target_modules = None
                source_analysis = {}
            else:
                lora_rank = 0
                adapter_ranks = None
                adapter_alphas = {}
                lora_target_modules = []
                source_analysis = {}

        # --- Step 4: Build switch config from arch descriptor ---
        # Copy config fields driven by architecture descriptor
        config_kwargs: dict = {}

        for field_name in arch.required_config_fields:
            config_kwargs[field_name] = getattr(base_config, field_name)

        for field_name, default in arch.optional_config_fields.items():
            config_kwargs[field_name] = getattr(base_config, field_name, default)

        # For Granite 3.x whose arch descriptor doesn't include
        # shared_intermediate_size, default it to intermediate_size.
        # GraniteMoeHybridConfig defaults it to 1024 (not None), so
        # GraniteSwitchConfig's fallback logic doesn't trigger.
        if "shared_intermediate_size" not in config_kwargs:
            config_kwargs["shared_intermediate_size"] = config_kwargs[
                "intermediate_size"
            ]

        # Normalize layer_types: map everything to "attention" (only attention
        # layers are supported).
        lt = config_kwargs.get("layer_types")
        if lt is not None:
            config_kwargs["layer_types"] = ["attention" for _ in lt]

        # When adapters are present, prepend a switch layer at index 0.
        if num_total > 0:
            config_kwargs["num_hidden_layers"] = config_kwargs["num_hidden_layers"] + 1
            if config_kwargs.get("layer_types") is not None:
                config_kwargs["layer_types"] = [
                    "attention",
                    *list(config_kwargs["layer_types"]),
                ]

        # Switch-specific parameters
        config_kwargs.update(
            {
                "num_adapters": num_total,
                "adapter_token_ids": adapter_token_ids,
                "adapter_substitute_token_ids": adapter_substitute_token_ids,
                "adapter_names": adapter_names,
                "max_lora_rank": lora_rank,
                "adapter_ranks": adapter_ranks,
                "lora_target_modules": lora_target_modules,
            }
        )

        # Merge caller-provided overrides (switch_head_dim, etc.)
        config_kwargs.update(kwargs)

        switch_config = GraniteSwitchConfig(**config_kwargs)

        # --- Step 5: Create model ---
        print(
            f"Creating GraniteSwitch model with {num_total} adapters "
            f"({num_external} external, {num_built_in} built-in)..."
        )
        model = GraniteSwitchForCausalLM(switch_config)

        if switch_config.torch_dtype is not None:
            print(f"Converting model to {switch_config.torch_dtype}...")
            model = model.to(dtype=switch_config.torch_dtype)

        # Set adapter_token_ids Parameter
        if num_total > 0:
            print(f"Setting adapter_token_ids Parameter: {adapter_token_ids}")
            with torch.no_grad():
                model.model.adapter_token_ids.copy_(
                    torch.tensor(adapter_token_ids, dtype=torch.long)
                )

        # --- Step 6: Transfer base weights ---
        base_mapping = transfer_base_weights(
            base_model_name_or_path, model, switch_config, arch
        )

        if adapter_paths:
            # --- Step 7: Transfer adapter weights ---
            adapter_mapping = transfer_adapter_weights(
                adapter_paths, model, adapter_alphas, arch
            )

            # --- Step 8: Validate ---
            # Reuse target_module_sets from source_analysis to avoid re-reading configs
            target_module_sets = source_analysis.get("adapter_targets")
            validate_all_parameters(
                model,
                arch,
                adapter_paths=adapter_paths,
                adapter_names=adapter_names[:num_external],
                target_module_sets=target_module_sets,
            )
        else:
            adapter_mapping = {}

        print("\nModel created successfully!")
        print(f"  Base model: {base_model_name_or_path}")
        print(
            f"  Total adapters: {num_total} ({num_external} external, {num_built_in} built-in)"
        )
        print(f"  Adapter token IDs: {adapter_token_ids}")
        print("\nSingleSwitch uses attention for adapter selection.")
        print("All parameters are frozen. Use the special tokens to trigger adapters.")

        # Store mappings for report generation
        model._build_mappings = {
            "base": base_mapping,
            "adapter": adapter_mapping,
            "source_analysis": source_analysis,
            # Per-external-adapter alpha, parallel to adapter_paths. When no
            # external adapters are provided, detect_lora_config isn't called
            # and adapter_alphas is a {} dict — surface an empty list in that
            # case so consumers don't need to special-case the shape.
            "adapter_alphas": list(adapter_alphas)
            if isinstance(adapter_alphas, (list, tuple))
            else [],
        }

        return model
