# SPDX-License-Identifier: Apache-2.0
"""Architecture descriptors for Granite Switch model building.

``ModuleDescriptor`` is the single source of truth for each module group.
Adapter remapping (previously duplicated in YAML) is now derived
programmatically from the descriptor list.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleDescriptor:
    """Single source of truth for one module group.

    Consolidates module group metadata that was previously spread across
    multiple dicts, YAML adapter mappings, and hardcoded patterns.

    Attributes:
        name: Group key, e.g., ``"qkv_proj"``, ``"shared_input_linear"``.
        peft_modules: PEFT source module names, e.g., ``["q_proj", "k_proj", "v_proj"]``.
        parent: Parent module in **switch** model hierarchy, e.g., ``"self_attn"``,
            ``"shared_mlp"``.
        attr_name: Model attribute name if different from *name* (e.g.,
            ``"input_linear"`` for group key ``"shared_input_linear"``).
            ``None`` means attribute name equals *name*.
        source_parent: Parent module in the **base** model, when different from
            *parent*.  E.g., ``"mlp"`` when mapping a Granite 3.x dense MLP to
            the switch model's ``"shared_mlp"``.  ``None`` means same as *parent*.
        num_switch_slices: Override slice count for split modules (e.g., ``2``
            for MoE input_linear where 1 PEFT module produces 2 slices).
            ``None`` means auto-derive from ``peft_modules`` length.
        target_inner_path: Extra path segment inserted before lora params.
    """

    name: str
    peft_modules: list[str]
    parent: str
    attr_name: str | None = None
    source_parent: str | None = None
    num_switch_slices: int | None = None
    target_inner_path: str = ""

    @property
    def effective_attr_name(self) -> str:
        """Model attribute name (falls back to *name*)."""
        return self.attr_name if self.attr_name is not None else self.name

    @property
    def effective_source_parent(self) -> str:
        """Parent module in the **base** model (falls back to *parent*)."""
        return self.source_parent if self.source_parent is not None else self.parent

    @property
    def effective_num_slices(self) -> int:
        """Number of lora slices.

        - Explicit ``num_switch_slices`` takes priority (split modules).
        - Multi-PEFT groups (qkv, gate_up) → ``len(peft_modules)``.
        - Single non-split modules → ``0`` (uses plain ``lora_A`` / ``lora_B``).
        """
        if self.num_switch_slices is not None:
            return self.num_switch_slices
        if len(self.peft_modules) > 1:
            return len(self.peft_modules)
        return 0

    @property
    def is_split(self) -> bool:
        """True when 1 PEFT module maps to N slices (e.g., MoE input_linear)."""
        return (
            self.num_switch_slices is not None
            and self.num_switch_slices > 1
            and len(self.peft_modules) == 1
        )

    @property
    def is_sliced(self) -> bool:
        """True when this group uses ``lora_A_slices`` / ``lora_B_slices`` naming."""
        return self.effective_num_slices > 0

    @property
    def is_base_fusion(self) -> bool:
        """True when base model has separate weights per peft_module that need concatenation.

        E.g., qkv_proj fuses q_proj + k_proj + v_proj; gate_up_proj fuses gate + up.
        MoE split modules (``num_switch_slices`` set) are NOT fusions — the base
        model already has a single fused weight.
        """
        return len(self.peft_modules) > 1 and self.num_switch_slices is None


@dataclass
class ArchDescriptor:
    """Architecture descriptor for building Switch models from a base architecture.

    Primary data is ``groups: List[ModuleDescriptor]``.
    """

    # Primary data — single source of truth
    groups: list[ModuleDescriptor]

    # Config fields to copy from base_config -> switch_config
    required_config_fields: list[str]
    optional_config_fields: dict[str, Any]

    # Weight name patterns
    layer_pattern: str = r"layers\.(\d+)\."
    peft_source_prefix: str = "base_model.model.model."

    # LoRA parameter keywords (for filtering)
    lora_keywords: list[str] = field(default_factory=lambda: ["lora_A", "lora_B"])

    # Non-LoRA buffer keywords to exclude from base validation
    buffer_keywords: list[str] = field(
        default_factory=lambda: [
            "adapter_token_ids",
            "adapter_scalings",
            "control_to_substitute_lut",
        ]
    )

    @property
    def switch_to_peft(self) -> dict[str, list[str]]:
        """Map ``parent.attr`` keys to their PEFT source module names.

        Keys match what :meth:`extract_module_key` returns, e.g.,
        ``{"self_attn.qkv_proj": ["q_proj", "k_proj", "v_proj"], ...}``.
        """
        return {
            f"{g.parent}.{g.effective_attr_name}": list(g.peft_modules)
            for g in self.groups
        }

    @property
    def all_peft_modules(self) -> list[str]:
        """Flat list of all PEFT module names across all groups."""
        return [mod for g in self.groups for mod in g.peft_modules]

    @property
    def parent_names(self) -> list[str]:
        """Ordered list of unique parent module names."""
        seen = []
        for g in self.groups:
            if g.parent not in seen:
                seen.append(g.parent)
        return seen

    def extract_module_key(self, param_name: str) -> str | None:
        """Extract ``parent.attr`` module key from a parameter name.

        E.g., ``"model.layers.0.self_attn.qkv_proj.lora_A"`` →
        ``"self_attn.qkv_proj"``.

        Returns None if no known parent is found.
        """
        parts = param_name.split(".")
        for parent in self.parent_names:
            if parent in parts:
                idx = parts.index(parent)
                if idx + 1 < len(parts):
                    return f"{parent}.{parts[idx + 1]}"
        return None

    def build_adapter_remapper(self):
        """Create an :class:`AdapterRemapper` from this descriptor's groups.

        Returns:
            An ``AdapterRemapper`` that can remap PEFT adapter names to
            Switch target names based on the module descriptors.
        """
        from .weight_remapper import AdapterRemapper

        return AdapterRemapper(self.groups, self.peft_source_prefix)


# ---------------------------------------------------------------------------
# Common ModuleDescriptor instances
# ---------------------------------------------------------------------------


def _common_attn_groups() -> list[ModuleDescriptor]:
    """Attention module groups shared by all architectures."""
    return [
        ModuleDescriptor(
            name="qkv_proj",
            peft_modules=["q_proj", "k_proj", "v_proj"],
            parent="self_attn",
        ),
        ModuleDescriptor(
            name="o_proj",
            peft_modules=["o_proj"],
            parent="self_attn",
        ),
    ]


def _dense_mlp_to_shared_groups() -> list[ModuleDescriptor]:
    """Map dense MLP (gate/up/down) to switch model's shared_mlp naming.

    Used for Granite 3.x whose base model uses ``mlp.gate_proj`` /
    ``mlp.up_proj`` / ``mlp.down_proj`` but whose switch model uses
    ``shared_mlp.input_linear`` / ``shared_mlp.output_linear``
    (the ``GraniteMoeHybridMLP`` layout).
    """
    return [
        ModuleDescriptor(
            name="shared_input_linear",
            peft_modules=["gate_proj", "up_proj"],
            parent="shared_mlp",
            source_parent="mlp",
            attr_name="input_linear",
        ),
        ModuleDescriptor(
            name="shared_output_linear",
            peft_modules=["down_proj"],
            parent="shared_mlp",
            source_parent="mlp",
            attr_name="output_linear",
        ),
    ]


def _moe_shared_mlp_groups() -> list[ModuleDescriptor]:
    """MoE shared_mlp groups (fused input_linear split into 2 slices + output_linear)."""
    return [
        ModuleDescriptor(
            name="shared_input_linear",
            peft_modules=["input_linear"],
            parent="shared_mlp",
            attr_name="input_linear",
            num_switch_slices=2,
        ),
        ModuleDescriptor(
            name="shared_output_linear",
            peft_modules=["output_linear"],
            parent="shared_mlp",
            attr_name="output_linear",
        ),
    ]


# ---------------------------------------------------------------------------
# Common config fields
# ---------------------------------------------------------------------------

_COMMON_REQUIRED_FIELDS: list[str] = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "tie_word_embeddings",
]

_COMMON_OPTIONAL_FIELDS: dict[str, Any] = {
    "rope_theta": 10000,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "torch_dtype": None,
}


# ---------------------------------------------------------------------------
# Granite optional config fields
# ---------------------------------------------------------------------------

_GRANITE_OPTIONAL_FIELDS: dict[str, Any] = {
    **_COMMON_OPTIONAL_FIELDS,
    "residual_multiplier": 1.0,
    "embedding_multiplier": 1.0,
    "logits_scaling": 1.0,
    "attention_multiplier": 1.0,
    # Granite/GraniteMoeHybrid vLLM classes use separate add-then-norm.
    "fused_add_norm": False,
}

# MoE fields (propagated when num_local_experts > 0)
_MOE_OPTIONAL_FIELDS: dict[str, Any] = {
    "num_local_experts": 0,
    "num_experts_per_tok": 1,
    "shared_intermediate_size": None,
}

# Layer type fields (propagated for hybrid models)
_HYBRID_OPTIONAL_FIELDS: dict[str, Any] = {
    "layer_types": None,
    "position_embedding_type": "rope",
}


# ---------------------------------------------------------------------------
# Architecture factory functions
# ---------------------------------------------------------------------------


def granite_moe_hybrid_arch(base_config=None) -> ArchDescriptor:
    """Granite 4.x MoE-Hybrid architecture (GraniteMoeHybrid config).

    All Granite 4 models are GraniteMoeHybrid configs and use ``shared_mlp``
    module naming (``shared_input_linear``, ``shared_output_linear``), even
    dense models with ``num_local_experts=0``.
    """
    optional_fields = dict(_GRANITE_OPTIONAL_FIELDS)
    optional_fields.update(_MOE_OPTIONAL_FIELDS)
    optional_fields.update(_HYBRID_OPTIONAL_FIELDS)

    return ArchDescriptor(
        groups=list(_common_attn_groups()) + list(_moe_shared_mlp_groups()),
        required_config_fields=list(_COMMON_REQUIRED_FIELDS),
        optional_config_fields=optional_fields,
    )


def granite_dense_arch(base_config=None) -> ArchDescriptor:
    """Granite 3.x dense architecture (Granite multipliers, mapped to shared_mlp).

    Granite 3.x uses the same ``mlp.gate_proj`` / ``mlp.up_proj`` /
    ``mlp.down_proj`` naming, but has Granite-specific config fields
    (``attention_multiplier``, ``residual_multiplier``, etc.) that are
    propagated from the base config.
    """
    return ArchDescriptor(
        groups=list(_common_attn_groups()) + list(_dense_mlp_to_shared_groups()),
        required_config_fields=list(_COMMON_REQUIRED_FIELDS),
        optional_config_fields=dict(_GRANITE_OPTIONAL_FIELDS),
    )


# ---------------------------------------------------------------------------
# Registry and auto-detection
# ---------------------------------------------------------------------------

_ARCH_REGISTRY = {
    "granite": granite_dense_arch,
    "granitemoehybrid": granite_moe_hybrid_arch,
}


def resolve_arch(model_name_or_path: str, base_config=None) -> ArchDescriptor:
    """Auto-detect architecture from HF config's ``model_type`` field.

    Args:
        model_name_or_path: HF model ID or local path.
        base_config: Optional pre-loaded config (avoids re-loading).

    Returns:
        ArchDescriptor for the detected architecture.

    Raises:
        ValueError: If the model_type is not supported.
    """
    if base_config is None:
        base_config = load_base_config(model_name_or_path)

    model_type = getattr(base_config, "model_type", None)
    if model_type is None:
        raise ValueError(f"Cannot determine model_type for {model_name_or_path}")

    # Normalize: granite_switch -> granite (handle our own model type)
    normalized = model_type.replace("_switch", "")

    factory = _ARCH_REGISTRY.get(normalized)
    if factory is None:
        raise ValueError(
            f"Unsupported architecture '{model_type}' for {model_name_or_path}. "
            f"Only Granite models are supported "
            f"(granite, granitemoehybrid)."
        )

    return factory(base_config=base_config)


def load_base_config(model_name_or_path: str):
    """Load a base model config.

    Equivalent to ``AutoConfig.from_pretrained()``.

    Returns:
        A ``PretrainedConfig`` for the model.
    """
    from transformers import AutoConfig as _AC

    return _AC.from_pretrained(model_name_or_path)
