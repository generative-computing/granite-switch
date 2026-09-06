# SPDX-License-Identifier: Apache-2.0
"""Tests for SR-specific compose logic: cross_stream and MLP group remapping.

SR adapters are composed into the same fused layout as LoRA/aLoRA. The only
SR-specific piece at the remapper level is ``cross_stream`` — a layer-level
group with no parent. Their MLP spelling (``mlp.gate_proj`` / ``mlp.up_proj`` /
``mlp.down_proj``) is the same one every adapter trained against a dense Granite
base uses, and the dense arch already maps it onto the fused ``shared_mlp``.
"""

import pytest

from granite_switch.composer.arch import (
    _cross_stream_groups,
    _moe_shared_mlp_groups,
    granite_dense_sr_arch,
    granite_moe_sr_arch,
)
from granite_switch.composer.weight_remapper import AdapterRemapper

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def dense_sr_groups():
    """The real dense-Granite SR arch: fused groups + cross_stream."""
    return granite_dense_sr_arch().groups


@pytest.fixture
def moe_mlp_groups():
    """MoE shared-MLP groups (pre-fused ``shared_mlp.*`` spelling)."""
    return _moe_shared_mlp_groups()


@pytest.fixture
def granitemoe_sr_groups():
    """Pure sparse MoE (``granitemoe``) SR arch: attention + cross_stream only."""
    return granite_moe_sr_arch().groups


# ════════════════════════════════════════════════════════════════════
# 1. Dense: mlp.* source keys land in the fused shared_mlp slices
# ════════════════════════════════════════════════════════════════════


class TestDenseMLPRemapping:
    """``mlp.gate_proj``/``up_proj`` are slices of the fused input_linear."""

    def test_gate_and_up_become_input_linear_slices(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)

        for slice_idx, mod in enumerate(["gate_proj", "up_proj"]):
            result = remapper.remap_adapter_name(
                f"base_model.model.model.layers.0.mlp.{mod}.lora_A.weight"
            )
            assert result is not None, f"Failed to remap mlp.{mod}"
            assert result.target_name == (
                f"model.layers.0.shared_mlp.input_linear.lora_A_slices.{slice_idx}"
            )

    def test_down_proj_becomes_unsliced_output_linear(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.39.mlp.down_proj.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == ("model.layers.39.shared_mlp.output_linear.lora_B")

    def test_layer_level_keys_not_matched(self, dense_sr_groups):
        """Keys without the mlp parent (legacy pattern) are not matched."""
        remapper = AdapterRemapper(dense_sr_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.gate_proj.lora_A.weight"
        )
        assert result is None


# ════════════════════════════════════════════════════════════════════
# 2. MoE: the pre-fused spelling is split at load
# ════════════════════════════════════════════════════════════════════


class TestMoEMLPRemapping:
    def test_prefused_input_linear_is_split_at_load(self, moe_mlp_groups):
        """Granite-4 MoE LoRA spelling: one pre-fused tensor, split into 2 slices."""
        remapper = AdapterRemapper(moe_mlp_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.shared_mlp.input_linear.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == (
            "model.layers.0.shared_mlp.input_linear.lora_A_slices"
        )
        assert result.split_slices == 2

    def test_output_linear_stays_unsliced(self, moe_mlp_groups):
        remapper = AdapterRemapper(moe_mlp_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.3.shared_mlp.output_linear.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == ("model.layers.3.shared_mlp.output_linear.lora_A")

    def test_dense_spelling_does_not_match_moe_groups(self, moe_mlp_groups):
        """``mlp.*`` is the dense arch's job — the MoE groups must not claim it."""
        remapper = AdapterRemapper(moe_mlp_groups)

        assert (
            remapper.remap_adapter_name(
                "base_model.model.model.layers.3.mlp.gate_proj.lora_B.weight"
            )
            is None
        )
        assert (
            remapper.remap_adapter_name(
                "base_model.model.model.layers.3.mlp.down_proj.lora_A.weight"
            )
            is None
        )


# ════════════════════════════════════════════════════════════════════
# 2b. granitemoe: no shared MLP, so no MLP spelling resolves at all
# ════════════════════════════════════════════════════════════════════


class TestGraniteMoeSRRemapping:
    """SR over a pure sparse MoE base: cross_stream and attention, nothing else."""

    def test_cross_stream_and_q_proj_remap(self, granitemoe_sr_groups):
        remapper = AdapterRemapper(granitemoe_sr_groups)

        cs = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.cross_stream.lora_A.weight"
        )
        assert cs is not None
        assert cs.target_name == "model.layers.0.cross_stream.lora_A"

        q = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert q is not None
        assert q.target_name == "model.layers.0.self_attn.qkv_proj.lora_A_slices.0"

    def test_no_mlp_spelling_resolves(self, granitemoe_sr_groups):
        """Both the dense and the pre-fused shared-MLP spellings must miss.

        An adapter that trained MLP LoRA against a granitemoe base trained
        something that does not exist here, and must not be silently absorbed.
        """
        remapper = AdapterRemapper(granitemoe_sr_groups)

        for key in (
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
            "base_model.model.model.layers.0.shared_mlp.input_linear.lora_A.weight",
            "base_model.model.model.layers.0.shared_mlp.output_linear.lora_A.weight",
        ):
            assert remapper.remap_adapter_name(key) is None, (
                f"{key} unexpectedly mapped"
            )


# ════════════════════════════════════════════════════════════════════
# 3. Cross-stream (layer-level, no parent)
# ════════════════════════════════════════════════════════════════════


class TestCrossStreamRemapping:
    """_cross_stream_groups() has parent="" — matches keys at layer level."""

    def test_cross_stream_lora_A(self):
        remapper = AdapterRemapper(_cross_stream_groups())

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.cross_stream.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.cross_stream.lora_A"
        assert result.split_slices is None

    def test_cross_stream_lora_B(self):
        remapper = AdapterRemapper(_cross_stream_groups())

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.15.cross_stream.lora_B.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.15.cross_stream.lora_B"

    def test_cross_stream_no_false_matches(self):
        """cross_stream should not match other layer-level keys."""
        remapper = AdapterRemapper(_cross_stream_groups())

        assert (
            remapper.remap_adapter_name(
                "base_model.model.model.layers.0.gate_proj.lora_A.weight"
            )
            is None
        )


# ════════════════════════════════════════════════════════════════════
# 4. Full SR arch: attention is fused too
# ════════════════════════════════════════════════════════════════════


class TestFullSRGroupRemapping:
    """The SR arch is the fused arch plus cross_stream — nothing unfused."""

    def test_q_maps_to_qkv_slice_zero(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == (
            "model.layers.0.self_attn.qkv_proj.lora_A_slices.0"
        )

    def test_k_and_v_map_to_slices_one_and_two(self, dense_sr_groups):
        """A plain LoRA's k/v still remap; SR adapters simply never ship them."""
        remapper = AdapterRemapper(dense_sr_groups)

        for slice_idx, mod in enumerate(["k_proj", "v_proj"], start=1):
            result = remapper.remap_adapter_name(
                f"base_model.model.model.layers.0.self_attn.{mod}.lora_A.weight"
            )
            assert result is not None, f"Failed to remap self_attn.{mod}"
            assert result.target_name == (
                f"model.layers.0.self_attn.qkv_proj.lora_A_slices.{slice_idx}"
            )

    def test_o_proj_stays_unsliced(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.0.self_attn.o_proj.lora_A"

    def test_cross_stream_in_full_set(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)

        result = remapper.remap_adapter_name(
            "base_model.model.model.layers.7.cross_stream.lora_A.weight"
        )
        assert result is not None
        assert result.target_name == "model.layers.7.cross_stream.lora_A"


# ════════════════════════════════════════════════════════════════════
# 5. No-match cases
# ════════════════════════════════════════════════════════════════════


class TestSRNoMatch:
    """Keys that should NOT match any SR group."""

    def test_unknown_module_name(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)
        assert (
            remapper.remap_adapter_name(
                "base_model.model.model.layers.0.self_attn.unknown_proj.lora_A.weight"
            )
            is None
        )

    def test_wrong_prefix(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)
        assert (
            remapper.remap_adapter_name(
                "wrong.prefix.layers.0.self_attn.q_proj.lora_A.weight"
            )
            is None
        )

    def test_non_lora_weight(self, dense_sr_groups):
        remapper = AdapterRemapper(dense_sr_groups)
        assert (
            remapper.remap_adapter_name(
                "base_model.model.model.layers.0.self_attn.q_proj.weight"
            )
            is None
        )

    def test_mlp_with_wrong_parent(self, dense_sr_groups):
        """gate_proj under self_attn should not match."""
        remapper = AdapterRemapper(dense_sr_groups)
        assert (
            remapper.remap_adapter_name(
                "base_model.model.model.layers.0.self_attn.gate_proj.lora_A.weight"
            )
            is None
        )
