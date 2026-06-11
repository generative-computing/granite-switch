# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Granite architecture weight mapping (skinning)."""

from types import SimpleNamespace

import pytest

from granite_switch.composer.arch import (
    ArchDescriptor,
    ModuleDescriptor,
    _COMMON_OPTIONAL_FIELDS,
    _GRANITE_OPTIONAL_FIELDS,
    granite_dense_arch,
)
from granite_switch.composer.weight_transfer import _classify_base_weights
from granite_switch.composer.weight_remapper import AdapterRemapper

pytestmark = pytest.mark.local_fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_granite_base_state_dict(num_layers=2):
    """Create a mock base state dict with Granite-style weight names."""
    import torch

    d = {}
    # Embeddings / head / norms
    d["model.embed_tokens.weight"] = torch.zeros(100, 128)
    d["model.norm.weight"] = torch.zeros(128)
    d["lm_head.weight"] = torch.zeros(100, 128)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        # Attention
        d[f"{prefix}.self_attn.q_proj.weight"] = torch.zeros(128, 128)
        d[f"{prefix}.self_attn.k_proj.weight"] = torch.zeros(32, 128)
        d[f"{prefix}.self_attn.v_proj.weight"] = torch.zeros(32, 128)
        d[f"{prefix}.self_attn.o_proj.weight"] = torch.zeros(128, 128)
        # MLP (dense Granite naming)
        d[f"{prefix}.mlp.gate_proj.weight"] = torch.zeros(256, 128)
        d[f"{prefix}.mlp.up_proj.weight"] = torch.zeros(256, 128)
        d[f"{prefix}.mlp.down_proj.weight"] = torch.zeros(128, 256)
        # Layer norms
        d[f"{prefix}.input_layernorm.weight"] = torch.zeros(128)
        d[f"{prefix}.post_attention_layernorm.weight"] = torch.zeros(128)

    return d


def _make_granite_config(hidden_size=128, num_attention_heads=4):
    """Create a mock Granite base config."""
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        model_type="granite",
    )


# ---------------------------------------------------------------------------
# Test 1: Base weight classification for Granite
# ---------------------------------------------------------------------------


class TestBaseWeightClassificationGranite:
    """Verify _classify_base_weights maps Granite MLP names to shared_mlp."""

    def test_fused_collections_have_shared_input_linear(self):
        base_sd = _make_granite_base_state_dict(num_layers=1)
        arch = granite_dense_arch()
        lora_targets = ["shared_input_linear", "shared_output_linear", "qkv_proj", "o_proj"]

        fused, direct = _classify_base_weights(base_sd, arch, lora_targets)

        # gate_proj + up_proj should fuse into shared_input_linear
        key = ("0", "shared_input_linear")
        assert key in fused, f"Expected {key} in fused_collections, got {list(fused.keys())}"
        assert "gate_proj" in fused[key]
        assert "up_proj" in fused[key]

    def test_standalone_down_proj_maps_to_shared_output_linear(self):
        base_sd = _make_granite_base_state_dict(num_layers=1)
        arch = granite_dense_arch()
        lora_targets = ["shared_input_linear", "shared_output_linear", "qkv_proj", "o_proj"]

        _fused, direct = _classify_base_weights(base_sd, arch, lora_targets)

        base_name = "model.layers.0.mlp.down_proj.weight"
        assert base_name in direct, f"down_proj not in direct mappings"
        expected = "model.layers.0.shared_mlp.output_linear.base_layer.weight"
        assert direct[base_name] == expected, (
            f"Expected {expected}, got {direct[base_name]}"
        )

    def test_o_proj_still_maps_correctly(self):
        base_sd = _make_granite_base_state_dict(num_layers=1)
        arch = granite_dense_arch()
        lora_targets = ["shared_input_linear", "shared_output_linear", "qkv_proj", "o_proj"]

        _fused, direct = _classify_base_weights(base_sd, arch, lora_targets)

        base_name = "model.layers.0.self_attn.o_proj.weight"
        assert base_name in direct
        expected = "model.layers.0.self_attn.o_proj.base_layer.weight"
        assert direct[base_name] == expected

    def test_qkv_fused_correctly(self):
        base_sd = _make_granite_base_state_dict(num_layers=1)
        arch = granite_dense_arch()
        lora_targets = ["shared_input_linear", "shared_output_linear", "qkv_proj", "o_proj"]

        fused, _direct = _classify_base_weights(base_sd, arch, lora_targets)

        key = ("0", "qkv_proj")
        assert key in fused
        assert "q_proj" in fused[key]
        assert "k_proj" in fused[key]
        assert "v_proj" in fused[key]


# ---------------------------------------------------------------------------
# Test 2: Granite arch descriptor
# ---------------------------------------------------------------------------


class TestGraniteDenseArchDescriptor:
    """Verify granite_dense_arch produces expected groups and config fields."""

    def test_granite_dense_has_granite_multiplier_fields(self):
        arch = granite_dense_arch()
        opt = arch.optional_config_fields
        assert "attention_multiplier" in opt
        assert "residual_multiplier" in opt
        assert "embedding_multiplier" in opt
        assert "logits_scaling" in opt

    def test_granite_dense_uses_shared_mlp_groups(self):
        arch = granite_dense_arch()
        group_names = [g.name for g in arch.groups]
        assert "shared_input_linear" in group_names
        assert "shared_output_linear" in group_names

    def test_has_expected_groups(self):
        arch = granite_dense_arch()
        group_names = [g.name for g in arch.groups]
        assert group_names == [
            "qkv_proj", "o_proj", "shared_input_linear", "shared_output_linear",
        ]

    def test_granite_dense_preserves_multiplier_defaults(self):
        arch = granite_dense_arch()
        # Granite optional fields have defaults (1.0) from _GRANITE_OPTIONAL_FIELDS
        assert arch.optional_config_fields["attention_multiplier"] == 1.0
        assert arch.optional_config_fields["residual_multiplier"] == 1.0
        assert arch.optional_config_fields["embedding_multiplier"] == 1.0
        assert arch.optional_config_fields["logits_scaling"] == 1.0

    def test_fused_add_norm_false(self):
        """Granite uses separate add-then-norm."""
        arch = granite_dense_arch()
        assert arch.optional_config_fields["fused_add_norm"] is False


# ---------------------------------------------------------------------------
# Test 3: Adapter remapping for Granite
# ---------------------------------------------------------------------------


class TestAdapterRemappingGranite:
    """Verify AdapterRemapper maps PEFT names from Granite adapters correctly."""

    def test_gate_proj_lora_a_maps_to_shared_input_linear_slice_0(self):
        arch = granite_dense_arch()
        remapper = arch.build_adapter_remapper()

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == (
            "model.layers.0.shared_mlp.input_linear.lora_A_slices.0"
        )

    def test_up_proj_lora_b_maps_to_shared_input_linear_slice_1(self):
        arch = granite_dense_arch()
        remapper = arch.build_adapter_remapper()

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == (
            "model.layers.0.shared_mlp.input_linear.lora_B_slices.1"
        )

    def test_down_proj_lora_a_maps_to_shared_output_linear(self):
        arch = granite_dense_arch()
        remapper = arch.build_adapter_remapper()

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == (
            "model.layers.0.shared_mlp.output_linear.lora_A"
        )

    def test_q_proj_lora_a_maps_to_qkv_slice_0(self):
        arch = granite_dense_arch()
        remapper = arch.build_adapter_remapper()

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == (
            "model.layers.0.self_attn.qkv_proj.lora_A_slices.0"
        )

    def test_o_proj_lora_b_maps_correctly(self):
        arch = granite_dense_arch()
        remapper = arch.build_adapter_remapper()

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.2.self_attn.o_proj.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.2.self_attn.o_proj.lora_B"


# ---------------------------------------------------------------------------
# Test 4: rope_scaling propagation
# ---------------------------------------------------------------------------


class TestRopeScalingPropagation:
    """Verify rope_scaling is in common optional fields and gets propagated."""

    def test_in_common_optional_fields(self):
        assert "rope_scaling" in _COMMON_OPTIONAL_FIELDS
        assert _COMMON_OPTIONAL_FIELDS["rope_scaling"] is None

    def test_propagated_to_granite_dense(self):
        arch = granite_dense_arch()
        assert "rope_scaling" in arch.optional_config_fields


# ---------------------------------------------------------------------------
# Test 5: ModuleDescriptor properties
# ---------------------------------------------------------------------------


class TestModuleDescriptorProperties:
    """Test ModuleDescriptor property calculations."""

    def test_effective_attr_name_uses_name_when_none(self):
        md = ModuleDescriptor(
            name="qkv_proj",
            peft_modules=["q_proj", "k_proj", "v_proj"],
            parent="self_attn",
        )
        assert md.effective_attr_name == "qkv_proj"

    def test_effective_attr_name_uses_attr_name_when_set(self):
        md = ModuleDescriptor(
            name="shared_input_linear",
            peft_modules=["gate_proj", "up_proj"],
            parent="shared_mlp",
            attr_name="input_linear",
        )
        assert md.effective_attr_name == "input_linear"

    def test_effective_source_parent_uses_parent_when_none(self):
        md = ModuleDescriptor(
            name="qkv_proj",
            peft_modules=["q_proj", "k_proj", "v_proj"],
            parent="self_attn",
        )
        assert md.effective_source_parent == "self_attn"

    def test_effective_source_parent_uses_source_parent_when_set(self):
        md = ModuleDescriptor(
            name="shared_input_linear",
            peft_modules=["gate_proj", "up_proj"],
            parent="shared_mlp",
            source_parent="mlp",
        )
        assert md.effective_source_parent == "mlp"

    def test_effective_num_slices_for_multi_peft(self):
        md = ModuleDescriptor(
            name="qkv_proj",
            peft_modules=["q_proj", "k_proj", "v_proj"],
            parent="self_attn",
        )
        assert md.effective_num_slices == 3

    def test_effective_num_slices_for_single_peft(self):
        md = ModuleDescriptor(
            name="o_proj",
            peft_modules=["o_proj"],
            parent="self_attn",
        )
        assert md.effective_num_slices == 0

    def test_is_sliced_true_for_multi_peft(self):
        md = ModuleDescriptor(
            name="qkv_proj",
            peft_modules=["q_proj", "k_proj", "v_proj"],
            parent="self_attn",
        )
        assert md.is_sliced is True

    def test_is_sliced_false_for_single_peft(self):
        md = ModuleDescriptor(
            name="o_proj",
            peft_modules=["o_proj"],
            parent="self_attn",
        )
        assert md.is_sliced is False

    def test_is_base_fusion_true_for_multi_peft(self):
        md = ModuleDescriptor(
            name="qkv_proj",
            peft_modules=["q_proj", "k_proj", "v_proj"],
            parent="self_attn",
        )
        assert md.is_base_fusion is True

    def test_is_base_fusion_false_for_single_peft(self):
        md = ModuleDescriptor(
            name="o_proj",
            peft_modules=["o_proj"],
            parent="self_attn",
        )
        assert md.is_base_fusion is False


# ---------------------------------------------------------------------------
# Test 6: ArchDescriptor properties
# ---------------------------------------------------------------------------


class TestArchDescriptorProperties:
    """Test ArchDescriptor property calculations."""

    def test_switch_to_peft_mapping(self):
        arch = granite_dense_arch()
        s2p = arch.switch_to_peft
        assert "self_attn.qkv_proj" in s2p
        assert s2p["self_attn.qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert "shared_mlp.input_linear" in s2p
        assert s2p["shared_mlp.input_linear"] == ["gate_proj", "up_proj"]

    def test_all_peft_modules(self):
        arch = granite_dense_arch()
        all_mods = arch.all_peft_modules
        assert "q_proj" in all_mods
        assert "k_proj" in all_mods
        assert "v_proj" in all_mods
        assert "o_proj" in all_mods
        assert "gate_proj" in all_mods
        assert "up_proj" in all_mods
        assert "down_proj" in all_mods

    def test_parent_names(self):
        arch = granite_dense_arch()
        parents = arch.parent_names
        assert "self_attn" in parents
        assert "shared_mlp" in parents

    def test_extract_module_key(self):
        arch = granite_dense_arch()
        key = arch.extract_module_key("model.layers.0.self_attn.qkv_proj.lora_A")
        assert key == "self_attn.qkv_proj"

    def test_extract_module_key_unknown_parent(self):
        arch = granite_dense_arch()
        key = arch.extract_module_key("model.layers.0.unknown.weight")
        assert key is None
