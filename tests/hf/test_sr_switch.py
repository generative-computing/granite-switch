# SPDX-License-Identifier: Apache-2.0
"""Shadow Residual (dual-stream) tests for GraniteSwitchForCausalLM.

Shadow Residual is no longer a separate model class: one
``GraniteSwitchForCausalLM`` covers both stream modes and picks its decoder layer
class — :class:`SRSwitchDecoderLayer` or
:class:`GraniteSwitchAttentionDecoderLayer` — from ``config.dual_stream``.  A
checkpoint is entirely one or the other.

These tests cover instantiation, forward shapes and cross-stream injection. All
tests run on CPU with random weights.
"""

import json

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf.modeling_granite_switch import (
    GraniteSwitchAttentionDecoderLayer,
    GraniteSwitchForCausalLM,
    SRSwitchDecoderLayer,
)

# ── Helpers ────────────────────────────────────────────────────────


def _set_adapter_token_ids(model, token_ids):
    """Populate model.model.adapter_token_ids from a list of ints."""
    model.model.adapter_token_ids.data = torch.tensor(token_ids, dtype=torch.long)


def _set_nonzero_lora(model, scale=0.1):
    """Set non-zero lora_B on every LoRA layer so adapters produce visible deltas."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.data = torch.randn_like(param) * scale


def _set_nonzero_cross_stream(model, scale=0.1):
    """Populate cross_stream LoRA for every adapter slot."""
    with torch.no_grad():
        for layer in model.model.layers:
            cs = layer.cross_stream
            cs.lora_A.data = torch.randn_like(cs.lora_A) * scale
            cs.lora_B.data = torch.randn_like(cs.lora_B) * scale


# ── Fixtures ───────────────────────────────────────────────────────

_BASE_KWARGS = dict(
    vocab_size=300,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,  # 1 switch + 2 decoder
    # Pinned: sized for a 1-slot switch; the default switch_type is now "multi".
    switch_type="single",
    num_attention_heads=4,
    num_key_value_heads=4,
    max_lora_rank=4,
    switch_head_dim=16,
)


@pytest.fixture
def sr_config():
    """One Shadow Residual adapter."""
    return GraniteSwitchConfig(
        num_adapters=1,
        adapter_token_ids=[250],
        adapter_substitute_token_ids=[1],
        adapter_names=["sr_adapter"],
        adapter_ranks=[4],
        dual_stream=True,
        cross_stream_rank=4,
        **_BASE_KWARGS,
    )


@pytest.fixture
def sr_config_multi_adapter():
    """Two Shadow Residual adapters."""
    return GraniteSwitchConfig(
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["sr_adapter_1", "sr_adapter_2"],
        adapter_ranks=[4, 4],
        dual_stream=True,
        cross_stream_rank=4,
        **_BASE_KWARGS,
    )


@pytest.fixture
def single_only_config():
    """Two plain LoRA adapters — no Shadow Residual anywhere."""
    return GraniteSwitchConfig(
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["lora_1", "lora_2"],
        adapter_ranks=[4, 4],
        **_BASE_KWARGS,
    )


# ════════════════════════════════════════════════════════════════════
# 1. Model instantiation
# ════════════════════════════════════════════════════════════════════


class TestSRModelInstantiation:
    def test_sr_model_creates_with_shared_kv(self, sr_config):
        model = GraniteSwitchForCausalLM(sr_config)
        assert model.model.switch is not None
        assert len(model.model.layers) == 2  # num_hidden_layers - 1 switch

    def test_layer_class_follows_dual_stream(self, sr_config, single_only_config):
        """One model class, two layer classes, chosen from ``dual_stream``."""
        sr_model = GraniteSwitchForCausalLM(sr_config)
        for layer in sr_model.model.layers:
            assert isinstance(layer, SRSwitchDecoderLayer)

        plain_model = GraniteSwitchForCausalLM(single_only_config)
        for layer in plain_model.model.layers:
            assert isinstance(layer, GraniteSwitchAttentionDecoderLayer)
            assert not isinstance(layer, SRSwitchDecoderLayer)

    def test_config_dual_stream_is_primary(self, sr_config, single_only_config):
        assert sr_config.dual_stream is True
        assert single_only_config.dual_stream is False

    def test_cross_stream_exists_with_zeroed_base(self, sr_config):
        model = GraniteSwitchForCausalLM(sr_config)
        for layer in model.model.layers:
            # base_layer weight must be zeros: cross_stream contributes only
            # the selected adapter's LoRA delta.
            assert torch.all(layer.cross_stream.base_layer.weight == 0)

    def test_fused_mlp_modules(self, sr_config):
        """Fused shared_mlp is the single canonical layout, SR included."""
        model = GraniteSwitchForCausalLM(sr_config)
        for layer in model.model.layers:
            assert hasattr(layer, "shared_mlp")
            assert not hasattr(layer, "mlp")

    def test_fused_qkv_projection(self, sr_config):
        """SR uses the fused qkv_proj, not separate q/k/v projections."""
        model = GraniteSwitchForCausalLM(sr_config)
        for layer in model.model.layers:
            assert hasattr(layer.self_attn, "qkv_proj")
            assert not hasattr(layer.self_attn, "q_proj")

    def test_no_cross_stream_without_sr(self, single_only_config):
        """Pure LoRA/aLoRA checkpoints allocate no cross_stream parameters."""
        model = GraniteSwitchForCausalLM(single_only_config)
        for layer in model.model.layers:
            assert not hasattr(layer, "cross_stream")
        assert not any("cross_stream" in n for n in model.state_dict())

    def test_no_unexpected_non_persistent_buffers(self, sr_config):
        """Non-persistent buffers do not survive save/load — allowlist them.

        Permitted:
          * the rotary tables (``inv_freq`` / ``original_inv_freq``) —
            transformers owns them and recomputes them from config during init;
          * ``adapter_token_ids`` — registered ``persistent=False`` so
            ``device_map="auto"`` does not reject an unmapped buffer, and the
            switch sources the control ids from ``config.adapter_token_ids`` (not
            this buffer) at forward time, so its zeros-after-``from_pretrained``
            value never reaches stream routing.

        Any OTHER non-persistent buffer is a bug: one such buffer once kept its
        uninitialized meta-device placeholder after ``from_pretrained``, silently
        corrupting stream routing.
        """
        model = GraniteSwitchForCausalLM(sr_config)
        offenders = [
            f"{mod_name}.{buf}" if mod_name else buf
            for mod_name, mod in model.named_modules()
            for buf in mod._non_persistent_buffers_set
            if buf not in ("inv_freq", "original_inv_freq", "adapter_token_ids")
        ]
        assert offenders == [], f"non-persistent buffers will not reload: {offenders}"

    def test_legacy_unfused_checkpoint_rejected(self):
        """unfused_qkv=True marks a pre-fusion SR checkpoint."""
        with pytest.raises(ValueError, match="Re-compose"):
            GraniteSwitchConfig(
                num_adapters=1,
                adapter_token_ids=[250],
                adapter_substitute_token_ids=[1],
                adapter_ranks=[4],
                dual_stream=True,
                cross_stream_rank=4,
                unfused_qkv=True,
                **_BASE_KWARGS,
            )

    def test_dual_stream_requires_cross_stream_rank(self):
        with pytest.raises(ValueError, match="cross_stream_rank is required"):
            GraniteSwitchConfig(
                num_adapters=1,
                adapter_token_ids=[250],
                adapter_substitute_token_ids=[1],
                adapter_ranks=[4],
                dual_stream=True,
                **_BASE_KWARGS,
            )

    def test_cross_stream_rank_rejected_without_dual_stream(self):
        with pytest.raises(ValueError, match="cross_stream_rank must be None"):
            GraniteSwitchConfig(
                num_adapters=1,
                adapter_token_ids=[250],
                adapter_substitute_token_ids=[1],
                adapter_ranks=[4],
                cross_stream_rank=4,
                **_BASE_KWARGS,
            )

    def test_sr_config_round_trips(self, sr_config, tmp_path):
        sr_config.save_pretrained(tmp_path)
        reloaded = GraniteSwitchConfig.from_pretrained(tmp_path)
        assert reloaded.dual_stream is True
        assert reloaded.cross_stream_rank == 4

    def test_unfused_checkpoint_rejected_on_reload(self, sr_config, tmp_path):
        """A stale config.json on disk must be rejected, not silently accepted."""
        sr_config.save_pretrained(tmp_path)
        config_file = tmp_path / "config.json"
        payload = json.loads(config_file.read_text())
        payload["unfused_qkv"] = True
        config_file.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Re-compose"):
            GraniteSwitchConfig.from_pretrained(tmp_path)


# ════════════════════════════════════════════════════════════════════
# 2. Forward output shape
# ════════════════════════════════════════════════════════════════════


class TestSRForwardOutputShape:
    def test_basic_output_shape(self, sr_config):
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 8, sr_config.vocab_size)

    def test_batch_output_shape(self, sr_config):
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (2, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (2, 8, sr_config.vocab_size)

    def test_returns_causal_lm_output(self, sr_config):
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert isinstance(output, CausalLMOutputWithPast)

    def test_labels_produce_loss(self, sr_config):
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=input_ids)
        assert output.loss is not None
        assert output.loss.dim() == 0


# ════════════════════════════════════════════════════════════════════
# 3. Adapter-inactive behavior
# ════════════════════════════════════════════════════════════════════


class TestSRAdapterInactive:
    """With adapter_indices==0 everywhere, no LoRA and no injection apply."""

    def test_no_adapter_produces_deterministic_output(self, sr_config):
        torch.manual_seed(42)
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        with torch.no_grad():
            out1 = model(input_ids=input_ids).logits
            out2 = model(input_ids=input_ids).logits
        torch.testing.assert_close(out1, out2)

    def test_cross_stream_zero_when_no_adapter(self, sr_config):
        """No adapter active → control index 0 → cross_stream contributes nothing."""
        torch.manual_seed(42)
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        _set_nonzero_cross_stream(model)

        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        with torch.no_grad():
            out_with_cs = model(input_ids=input_ids).logits

        with torch.no_grad():
            for layer in model.model.layers:
                layer.cross_stream.lora_A.data.zero_()
                layer.cross_stream.lora_B.data.zero_()

        with torch.no_grad():
            out_without_cs = model(input_ids=input_ids).logits

        torch.testing.assert_close(out_with_cs, out_without_cs)


# ════════════════════════════════════════════════════════════════════
# 4. Adapter-active behavior (dual-stream)
# ════════════════════════════════════════════════════════════════════


class TestSRAdapterActive:
    """When the control token fires, the adapter should change the logits."""

    def test_control_token_activates_adapter(self, sr_config):
        torch.manual_seed(42)
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        _set_nonzero_lora(model)

        with_ctrl = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])
        no_ctrl = torch.tensor([[10, 20, 100, 30, 40, 50, 60, 70]])

        with torch.no_grad():
            logits_ctrl = model(input_ids=with_ctrl).logits
            logits_text = model(input_ids=no_ctrl).logits

        # Pre-control positions (0, 1): identical (causal, can't see position 2)
        torch.testing.assert_close(logits_ctrl[0, :2], logits_text[0, :2])

        # Post-control positions (3+): must differ (adapter active via LoRA)
        assert not torch.allclose(logits_ctrl[0, 3:], logits_text[0, 3:]), (
            "Post-control logits should differ when SR adapter is active"
        )

    def test_cross_stream_affects_output_when_active(self, sr_config):
        torch.manual_seed(42)
        model = GraniteSwitchForCausalLM(sr_config).eval()
        _set_adapter_token_ids(model, sr_config.adapter_token_ids)
        _set_nonzero_lora(model)

        input_ids = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])

        with torch.no_grad():
            out_before = model(input_ids=input_ids).logits.clone()

        _set_nonzero_cross_stream(model, scale=0.5)

        with torch.no_grad():
            out_after = model(input_ids=input_ids).logits

        assert not torch.allclose(out_before[0, 3:], out_after[0, 3:]), (
            "Cross-stream injection should change output when adapter is active"
        )


# ════════════════════════════════════════════════════════════════════
# 5. Multi-adapter SR model
# ════════════════════════════════════════════════════════════════════


class TestSRMultiAdapter:
    """Multiple SR adapters in one model (all dual-stream)."""

    def test_multi_adapter_forward_succeeds(self, sr_config_multi_adapter):
        model = GraniteSwitchForCausalLM(sr_config_multi_adapter).eval()
        _set_adapter_token_ids(model, sr_config_multi_adapter.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 8, sr_config_multi_adapter.vocab_size)

    def test_different_adapters_produce_different_logits(self, sr_config_multi_adapter):
        torch.manual_seed(42)
        model = GraniteSwitchForCausalLM(sr_config_multi_adapter).eval()
        _set_adapter_token_ids(model, sr_config_multi_adapter.adapter_token_ids)
        _set_nonzero_lora(model)

        seq_a1 = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])  # adapter 1
        seq_a2 = torch.tensor([[10, 20, 251, 30, 40, 50, 60, 70]])  # adapter 2

        with torch.no_grad():
            logits_a1 = model(input_ids=seq_a1).logits
            logits_a2 = model(input_ids=seq_a2).logits

        assert not torch.allclose(logits_a1[0, 3:], logits_a2[0, 3:]), (
            "Different adapters should produce different post-control logits"
        )


# ════════════════════════════════════════════════════════════════════
# 6. Save / reload and generation
# ════════════════════════════════════════════════════════════════════


class TestSRRoundTrip:
    @staticmethod
    def _prepared_model(config):
        torch.manual_seed(7)
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)
        _set_nonzero_lora(model)
        _set_nonzero_cross_stream(model, scale=0.3)
        return model

    def test_logits_survive_save_and_reload(self, sr_config_multi_adapter, tmp_path):
        """An SR checkpoint must behave identically after a save/load cycle.

        Regression test for SR state that used to be lost on reload — the model
        then ran with uninitialized routing state and silently corrupted every
        adapter's output.
        """
        model = self._prepared_model(sr_config_multi_adapter)

        rows = torch.tensor(
            [
                [10, 20, 30, 40, 50, 60],  # base
                [10, 250, 30, 40, 50, 60],  # SR adapter 1
                [10, 251, 30, 40, 50, 60],  # SR adapter 2
            ]
        )
        with torch.no_grad():
            before = [model(input_ids=rows[i : i + 1]).logits for i in range(3)]

        model.save_pretrained(tmp_path)
        reloaded = GraniteSwitchForCausalLM.from_pretrained(tmp_path).eval()

        assert reloaded.config.dual_stream is True
        for layer in reloaded.model.layers:
            assert isinstance(layer, SRSwitchDecoderLayer)

        with torch.no_grad():
            after = [reloaded(input_ids=rows[i : i + 1]).logits for i in range(3)]
        for i in range(3):
            torch.testing.assert_close(before[i], after[i], atol=0.0, rtol=0.0)

    def test_generation_with_each_adapter(self, sr_config_multi_adapter):
        """Dual-stream decode works across cached steps."""
        model = self._prepared_model(sr_config_multi_adapter)
        for control_token in (250, 251):
            prompt = torch.tensor([[10, control_token, 30, 40]])
            with torch.no_grad():
                out = model.generate(
                    input_ids=prompt,
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=True,
                )
            assert out.shape[1] == prompt.shape[1] + 4
