# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AdapterRemapper and RemapResult."""

import pytest

from granite_switch.composer.weight_remapper import AdapterRemapper, RemapResult
from granite_switch.composer.arch import ModuleDescriptor

pytestmark = pytest.mark.local_fast


class TestRemapResult:
    """Tests for RemapResult dataclass."""

    def test_basic_remap_result(self):
        """Basic RemapResult with just target_name."""
        result = RemapResult(target_name="model.layers.0.self_attn.qkv_proj.lora_A")
        assert result.target_name == "model.layers.0.self_attn.qkv_proj.lora_A"
        assert result.split_slices is None
        assert result.split_type is None

    def test_split_remap_result_duplicate(self):
        """RemapResult with split metadata (duplicate type)."""
        result = RemapResult(
            target_name="model.layers.0.shared_mlp.input_linear.lora_A_slices",
            split_slices=2,
            split_type="duplicate",
        )
        assert result.target_name == "model.layers.0.shared_mlp.input_linear.lora_A_slices"
        assert result.split_slices == 2
        assert result.split_type == "duplicate"

    def test_split_remap_result_chunk(self):
        """RemapResult with split metadata (chunk_dim0 type)."""
        result = RemapResult(
            target_name="model.layers.0.shared_mlp.input_linear.lora_B_slices",
            split_slices=2,
            split_type="chunk_dim0",
        )
        assert result.split_slices == 2
        assert result.split_type == "chunk_dim0"


class TestAdapterRemapperMakePattern:
    """Tests for AdapterRemapper._make_pattern static method."""

    def test_make_pattern_basic(self):
        """Test pattern matches expected PEFT weight names."""
        pattern = AdapterRemapper._make_pattern(
            prefix="base_model.model.model.",
            parent="self_attn",
            peft_mod="q_proj",
            ab="lora_A",
        )
        # Should match
        assert pattern.match("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
        assert pattern.match("base_model.model.model.layers.15.self_attn.q_proj.lora_A.weight")
        assert pattern.match("base_model.model.model.layers.123.self_attn.q_proj.lora_A.weight")

    def test_make_pattern_no_match_wrong_module(self):
        """Pattern should not match different module names."""
        pattern = AdapterRemapper._make_pattern(
            prefix="base_model.model.model.",
            parent="self_attn",
            peft_mod="q_proj",
            ab="lora_A",
        )
        # Should not match k_proj
        assert pattern.match("base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight") is None

    def test_make_pattern_no_match_wrong_lora_type(self):
        """Pattern should not match different lora type."""
        pattern = AdapterRemapper._make_pattern(
            prefix="base_model.model.model.",
            parent="self_attn",
            peft_mod="q_proj",
            ab="lora_A",
        )
        # Should not match lora_B
        assert pattern.match("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight") is None

    def test_make_pattern_extracts_layer_index(self):
        """Pattern should extract layer index via named group."""
        pattern = AdapterRemapper._make_pattern(
            prefix="base_model.model.model.",
            parent="self_attn",
            peft_mod="q_proj",
            ab="lora_A",
        )
        match = pattern.match("base_model.model.model.layers.42.self_attn.q_proj.lora_A.weight")
        assert match is not None
        assert match.group("layer") == "42"

    def test_make_pattern_different_prefix(self):
        """Pattern should work with different prefixes."""
        pattern = AdapterRemapper._make_pattern(
            prefix="model.",
            parent="self_attn",
            peft_mod="q_proj",
            ab="lora_A",
        )
        assert pattern.match("model.layers.0.self_attn.q_proj.lora_A.weight")
        assert pattern.match("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight") is None

    def test_make_pattern_mlp_module(self):
        """Pattern should work for MLP modules."""
        pattern = AdapterRemapper._make_pattern(
            prefix="base_model.model.model.",
            parent="mlp",
            peft_mod="gate_proj",
            ab="lora_B",
        )
        assert pattern.match("base_model.model.model.layers.5.mlp.gate_proj.lora_B.weight")


class TestAdapterRemapper:
    """Tests for AdapterRemapper class."""

    @pytest.fixture
    def qkv_groups(self):
        """Create QKV module descriptors for testing."""
        return [
            ModuleDescriptor(
                name="qkv_proj",
                peft_modules=["q_proj", "k_proj", "v_proj"],
                parent="self_attn",
            ),
        ]

    @pytest.fixture
    def o_proj_group(self):
        """Create single non-sliced module descriptor."""
        return [
            ModuleDescriptor(
                name="o_proj",
                peft_modules=["o_proj"],
                parent="self_attn",
            ),
        ]

    @pytest.fixture
    def split_mlp_group(self):
        """Create split module descriptor (1 PEFT -> N slices)."""
        return [
            ModuleDescriptor(
                name="input_linear",
                peft_modules=["input_linear"],
                parent="shared_mlp",
                num_switch_slices=2,
            ),
        ]

    @pytest.fixture
    def dense_mlp_groups(self):
        """Create dense MLP groups that map to shared_mlp."""
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

    def test_remap_qkv_sliced(self, qkv_groups):
        """Test remapping sliced QKV modules."""
        remapper = AdapterRemapper(qkv_groups)

        # q_proj -> slice 0
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.self_attn.qkv_proj.lora_A_slices.0"
        assert result.split_slices is None

        # k_proj -> slice 1
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.self_attn.qkv_proj.lora_A_slices.1"

        # v_proj -> slice 2
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.self_attn.qkv_proj.lora_B_slices.2"

    def test_remap_non_sliced_module(self, o_proj_group):
        """Test remapping non-sliced modules (plain lora_A/lora_B)."""
        remapper = AdapterRemapper(o_proj_group)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.5.self_attn.o_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.5.self_attn.o_proj.lora_A"
        assert result.split_slices is None

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.5.self_attn.o_proj.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.5.self_attn.o_proj.lora_B"

    def test_remap_split_module(self, split_mlp_group):
        """Test remapping split modules (1 PEFT -> N slices with metadata)."""
        remapper = AdapterRemapper(split_mlp_group)

        # lora_A -> duplicate split
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.shared_mlp.input_linear.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.shared_mlp.input_linear.lora_A_slices"
        assert result.split_slices == 2
        assert result.split_type == "duplicate"

        # lora_B -> chunk_dim0 split
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.shared_mlp.input_linear.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.shared_mlp.input_linear.lora_B_slices"
        assert result.split_slices == 2
        assert result.split_type == "chunk_dim0"

    def test_remap_dense_mlp_to_shared(self, dense_mlp_groups):
        """Test remapping dense MLP (gate/up/down) to shared_mlp naming."""
        remapper = AdapterRemapper(dense_mlp_groups)

        # gate_proj -> shared_mlp.input_linear slice 0
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.shared_mlp.input_linear.lora_A_slices.0"

        # up_proj -> shared_mlp.input_linear slice 1
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.shared_mlp.input_linear.lora_A_slices.1"

        # down_proj -> shared_mlp.output_linear (non-sliced)
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.shared_mlp.output_linear.lora_B"

    def test_remap_no_match(self, qkv_groups):
        """Test that unknown parameter names return None."""
        remapper = AdapterRemapper(qkv_groups)

        # Unknown module
        assert remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.unknown_proj.lora_A.weight"
        ) is None

        # Wrong prefix
        assert remapper.remap_adapter_name(
            "wrong_prefix.layers.0.self_attn.q_proj.lora_A.weight"
        ) is None

        # Non-lora parameter
        assert remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.q_proj.weight"
        ) is None

    def test_remap_different_layer_indices(self, qkv_groups):
        """Test that layer indices are correctly extracted and used."""
        remapper = AdapterRemapper(qkv_groups)

        for layer_idx in [0, 1, 10, 99, 127]:
            result = remapper.remap_adapter_name(
                f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"
            )
            assert result is not None
            assert f"layers.{layer_idx}" in result.target_name

    def test_custom_prefix(self, qkv_groups):
        """Test AdapterRemapper with custom PEFT source prefix."""
        remapper = AdapterRemapper(qkv_groups, peft_source_prefix="custom.prefix.")

        # Should match custom prefix
        result = remapper.remap_adapter_name(
            "custom.prefix.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert result is not None

        # Should not match default prefix
        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert result is None
