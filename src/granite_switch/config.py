# SPDX-License-Identifier: Apache-2.0
"""Configuration for Granite model with adapter switching."""

from typing import List, Optional, Dict

from transformers import GraniteMoeHybridConfig


class GraniteSwitchConfig(GraniteMoeHybridConfig):
    """Configuration class for GraniteSwitch model.

    Extends the Granite base config with parameters for adapter switching using
    the SingleSwitch mechanism.

    Inherits from GraniteMoeHybridConfig (the transformers base class for
    Granite 4 models) and adds adapter routing parameters.

    Args:
        num_adapters (int): Number of LoRA adapters available. Default: 0 (no adapters).
            This counts real LoRA adapters only (not base). Index 0 always means "base / no adapter".
        adapter_token_ids (List[int]): Token IDs for adapter control.
            Length: num_adapters (one token per real adapter). Must be unique.
            adapter_token_ids[i] activates adapter i+1 (1-indexed output).
            Output 0 = base (implicit default, no token needed to return to base).
            NOTE: SingleSwitch cannot transition back to base mid-sequence.
        adapter_substitute_token_ids (List[int]): Token IDs whose embeddings replace
            the control-token embeddings before the decoder runs (token-exchange mode).
            Length: num_adapters. When provided together with control_dims=0, the model
            uses token exchange instead of KV hiding.

        SingleSwitch parameters:
            control_token_gain (float): Attention gain for control/non-control separation. Default: 15.0.
            switch_head_dim (int): Dimension of Q/K/V vectors in switch attention. Default: 32.
            control_dims (int): Extra dimensions for K/V to mask control tokens. Must be >= 0.
                Default: 0 (token-exchange mode — adapter_substitute_token_ids must be provided).
                Set >= 1 to enable the legacy KV-hiding path.

        adapter_names (List[str]): Ordered adapter names for name-to-index mapping.
            Used by hiding_groups and hiding_policy to resolve names to indices.
        hiding_groups (Dict[str, List[str]]): Hiding group definitions.
            Maps group_name → list of adapter names whose control tokens belong to this group.
            Each group uses one control dimension. Requires control_dims >= len(hiding_groups).
        hiding_policy (Dict[str, List[str]]): Per-adapter hiding policy.
            Maps adapter_name → list of group names that adapter hides. Use "base" for the
            base adapter (adapter_index 0).
        adapter_third_party (List[str]): Adapter names that are third-party (externally trained).
            Third-party adapters were not trained with control tokens in their vocabulary,
            which affects KV hiding policy.

        max_lora_rank (int): Maximum rank across all LoRA adapters (for allocation). Default: 8.
        adapter_ranks (List[int]): Per-adapter ranks. Must have length equal to num_adapters.
        lora_target_modules (List[str]): List of module GROUP names to apply LoRA to.
            Module groups: "qkv_proj", "o_proj", "shared_input_linear", "shared_output_linear".
            Default: all four groups
        **kwargs: Additional arguments passed to GraniteConfig.
    """

    model_type = "granite_switch"

    def __init__(
        self,
        num_adapters: int = 0,
        adapter_token_ids: Optional[List[int]] = None,
        adapter_substitute_token_ids: Optional[List[int]] = None,
        # SingleSwitch parameters
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        control_dims: int = 0,
        # Hiding groups and policy
        adapter_names: Optional[List[str]] = None,
        hiding_groups: Optional[Dict[str, List[str]]] = None,
        hiding_policy: Optional[Dict[str, List[str]]] = None,
        adapter_third_party: Optional[List[str]] = None,
        # Adapter parameters
        max_lora_rank: int = 8,
        adapter_ranks: List[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        # vLLM residual-norm convention (for bit-exact skinning equivalence)
        fused_add_norm: bool = False,
        # Parent class defaults (Granite 4 dense configuration)
        num_local_experts: int = 0,
        position_embedding_type: str = "rope",
        layer_types: Optional[List[str]] = None,
        **kwargs,
    ):
        # Compute default layer_types before parent init.
        # layer_types must have length == num_hidden_layers (includes switch layer at
        # index 0 when adapters are present). This ensures DynamicCache pre-allocation
        # matches the global layer indices used by decoder layers.
        if layer_types is None:
            num_hidden_layers = kwargs.get("num_hidden_layers", 32)
            layer_types = ["attention"] * num_hidden_layers

        super().__init__(
            num_local_experts=num_local_experts,
            position_embedding_type=position_embedding_type,
            layer_types=layer_types,
            **kwargs,
        )

        # Default shared_intermediate_size from intermediate_size.
        # All Granite 4 models use shared_mlp naming; for dense models
        # shared_intermediate_size == intermediate_size.
        if self.shared_intermediate_size is None:
            self.shared_intermediate_size = self.intermediate_size

        # Validate num_adapters
        if num_adapters < 0:
            raise ValueError(f"num_adapters must be >= 0, got {num_adapters}")
        self.num_adapters = num_adapters

        # Validate adapter_token_ids if provided
        if num_adapters > 0 and adapter_token_ids is not None:
            if len(adapter_token_ids) != num_adapters:
                raise ValueError(
                    f"adapter_token_ids length ({len(adapter_token_ids)}) must equal "
                    f"num_adapters ({num_adapters})."
                )
            # Token-exchange builds the control→substitute LUT keyed by adapter token id;
            # duplicates would silently collapse to a single slot.
            if len(set(adapter_token_ids)) != len(adapter_token_ids):
                raise ValueError(
                    f"adapter_token_ids must be unique; got {adapter_token_ids}"
                )
        self.adapter_token_ids = adapter_token_ids

        # Validate adapter_substitute_token_ids if provided
        if num_adapters > 0 and adapter_substitute_token_ids is not None:
            if len(adapter_substitute_token_ids) != num_adapters:
                raise ValueError(
                    f"adapter_substitute_token_ids length ({len(adapter_substitute_token_ids)}) "
                    f"must equal num_adapters ({num_adapters})."
                )
            if any(sid < 0 for sid in adapter_substitute_token_ids):
                raise ValueError(
                    f"adapter_substitute_token_ids must all be >= 0 (real token ids); "
                    f"got {adapter_substitute_token_ids}"
                )
            if adapter_token_ids is None:
                raise ValueError(
                    "adapter_token_ids is required when adapter_substitute_token_ids "
                    "is provided (token-exchange mode maps control ids to substitute ids)."
                )
        self.adapter_substitute_token_ids = adapter_substitute_token_ids

        # SingleSwitch parameters
        self.control_token_gain = control_token_gain
        self.switch_head_dim = switch_head_dim
        if control_dims < 0:
            raise ValueError(
                f"control_dims must be >= 0 (got {control_dims}). "
                "Use control_dims=0 for native mode (no KV hiding). "
                "Use control_dims >= 1 for third-party mode (KV cache masking)."
            )
        self.control_dims = control_dims
        self.fused_add_norm = fused_add_norm

        # Control tokens need one of two handling paths when adapters are present:
        # legacy KV hiding (control_dims > 0) or token exchange (substitute ids present).
        # The combination of neither would leak raw control-token embeddings into attention.
        if (
            num_adapters > 0
            and control_dims == 0
            and adapter_substitute_token_ids is None
        ):
            raise ValueError(
                "When num_adapters > 0, either control_dims > 0 (legacy KV hiding) "
                "or adapter_substitute_token_ids (token exchange) must be provided. "
                "Neither is set, which would leave control tokens unhandled."
            )

        # Hiding groups and policy
        self.adapter_names = adapter_names
        self.hiding_groups = hiding_groups
        self.hiding_policy = hiding_policy
        self.adapter_third_party = adapter_third_party

        # Validate control_dims >= num_hiding_groups
        if hiding_groups is not None and len(hiding_groups) > control_dims:
            raise ValueError(
                f"control_dims ({control_dims}) must be >= number of hiding groups "
                f"({len(hiding_groups)}). Each hiding group uses one control dimension."
            )

        # KV cache head dimension vs. projection dimension.
        # The QKV projection outputs vectors of size projection_head_dim
        # (= hidden_size / num_attention_heads). The KV cache stores larger
        # vectors (projection_head_dim + control_dims) for exact attention
        # masking of control tokens. The expanded size is communicated to
        # vLLM via a custom ModelArchConfigConvertor (registered in vllm/__init__.py)
        # so that hybrid page-size calculations use the correct value.
        # We do NOT set head_dim here because HF's RoPE also reads it.
        # Use explicit head_dim from kwargs when available (some models have
        # head_dim != hidden_size // num_attention_heads).
        explicit_head_dim = kwargs.get("head_dim")
        self.projection_head_dim = (
            explicit_head_dim
            if explicit_head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )

        # Validate and store adapter configuration
        if num_adapters > 0:
            if adapter_ranks is None:
                raise ValueError("adapter_ranks must be provided when num_adapters > 0")

            if len(adapter_ranks) != num_adapters:
                raise ValueError(
                    f"adapter_ranks length ({len(adapter_ranks)}) must equal num_adapters ({num_adapters})"
                )

            if max(adapter_ranks) != max_lora_rank:
                raise ValueError(
                    f"max(adapter_ranks)={max(adapter_ranks)} must equal max_lora_rank={max_lora_rank}"
                )

        self.max_lora_rank = max_lora_rank
        self.adapter_ranks = adapter_ranks

        # Default LoRA target module groups.
        # Dynamically determined based on model architecture.
        # Empty when num_adapters == 0 (no LoRA to apply).
        if lora_target_modules is None:
            lora_target_modules = []

            if self.num_adapters > 0:
                # Attention modules (present in all attention layers)
                if any(lt == "attention" for lt in self.layer_types):
                    lora_target_modules.extend([
                        "qkv_proj",     # Q/K/V fused
                        "o_proj",       # O projection
                    ])

                # MLP modules: all Granite 4 models use shared_mlp naming
                lora_target_modules.extend([
                    "shared_input_linear",   # shared_mlp input_linear (fused gate+up)
                    "shared_output_linear",  # shared_mlp output_linear
                ])

        self.lora_target_modules = lora_target_modules

    @property
    def expanded_head_dim(self) -> int:
        """KV cache head dimension: projection_head_dim + control_dims when adapters are active."""
        if self.num_adapters > 0 and self.control_dims > 0:
            return self.projection_head_dim + self.control_dims
        return self.projection_head_dim

    @property
    def use_token_exchange(self) -> bool:
        """True when control tokens are replaced with substitute embeddings (vs. KV hiding)."""
        return (
            self.num_adapters > 0
            and self.control_dims == 0
            and self.adapter_substitute_token_ids is not None
        )

    @property
    def num_hiding_groups(self) -> int:
        """Number of hiding groups (each uses one control dimension)."""
        if self.hiding_groups is None:
            return 0
        return len(self.hiding_groups)

    @property
    def hiding_group_names(self) -> List[str]:
        """Ordered list of hiding group names (determines control dim indices)."""
        if self.hiding_groups is None:
            return []
        return list(self.hiding_groups.keys())

    def get_hiding_group_token_ids(self) -> Dict[int, List[int]]:
        """Map group index → list of token IDs in that group.

        Resolves adapter names to their activating token IDs using
        adapter_names and adapter_token_ids.

        Returns empty dict if no hiding groups configured.
        """
        if self.hiding_groups is None or self.adapter_names is None:
            return {}
        if self.adapter_token_ids is None:
            return {}

        # Build name → token ID mapping (no offset for SingleSwitch)
        name_to_token_id = {}
        for i, name in enumerate(self.adapter_names):
            name_to_token_id[name] = self.adapter_token_ids[i]

        result = {}
        for group_idx, group_name in enumerate(self.hiding_group_names):
            adapter_names_in_group = self.hiding_groups[group_name]
            token_ids = []
            for name in adapter_names_in_group:
                if name in name_to_token_id:
                    token_ids.append(name_to_token_id[name])
            result[group_idx] = token_ids
        return result

    def get_third_party_adapter_mask(self) -> List[bool]:
        """Return per-adapter-slot boolean: True if the adapter is third-party.

        Index 0 = base (never third-party). Index 1+ = real adapters.
        Length = num_adapters + 1 (one slot per adapter index including base).

        Third-party adapters were not trained with control tokens in their
        vocabulary, which affects KV hiding policy.

        Returns all-False list if adapter_third_party is not configured.
        """
        num_slots = self.num_adapters + 1  # base + adapters
        if not self.adapter_third_party or not self.adapter_names:
            return [False] * num_slots

        tp_set = set(self.adapter_third_party)
        # Index 0 = base (never third-party)
        mask = [False]
        for name in self.adapter_names:
            mask.append(name in tp_set)
        return mask

    def get_adapter_hiding_policy_matrix(self) -> List[List[bool]]:
        """Build adapter hiding policy matrix: [num_adapter_slots][num_groups].

        Index 0 = base adapter. Index 1+ = real adapters (matching adapter_names order).
        Each entry is True if that adapter hides that group.

        Returns empty list if no hiding policy configured.
        """
        if self.hiding_policy is None or self.adapter_names is None:
            return []

        num_groups = self.num_hiding_groups
        group_names = self.hiding_group_names

        # Build ordered adapter list: [base, adapter_0, adapter_1, ...]
        all_adapter_names = ["base"] + list(self.adapter_names)
        num_slots = len(all_adapter_names)

        matrix = []
        for adapter_name in all_adapter_names:
            groups_to_hide = self.hiding_policy.get(adapter_name, [])
            row = [gn in groups_to_hide for gn in group_names]
            matrix.append(row)
        return matrix
