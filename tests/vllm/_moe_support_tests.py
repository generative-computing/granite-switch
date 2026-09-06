# SPDX-License-Identifier: Apache-2.0
"""Sparse-MoE support in the vLLM backend (construction + weight loading).
Inner file — run by test_moe_support.py in a subprocess.

This is the first coverage the vLLM MoE path has ever had.  Every other
``tests/vllm/`` fixture sets ``num_local_experts=0``, so both the decoder layers'
expert branch and the HF-stacked -> ``FusedMoE`` remap in the weight loaders were
entirely unexercised.  They become load-bearing for a ``granitemoe`` base: a
*pure sparse* MoE with no dense shared MLP — upstream's
``shared_intermediate_size == 0``.

What is deliberately NOT here: numeric assertions on an expert forward.  Top-k
expert outputs accumulate through CUDA ``index_add`` atomics, so upstream's own
``block_sparse_moe(x)`` differs from itself run twice on identical input.  The
distribution-equivalence gate in ``test_generation_equivalence.py`` is where
expert numerics belong; here we pin *structure* and *weight provenance*, which
are exact.

Also deliberately NOT here: tensor parallelism.  ``_build`` goes through
``tests.shared.vllm_distributed.ensure_distributed``, which initializes the
process group with ``world_size=1`` — every test in this file is single-rank *by
construction*, and passing a ``tp_size`` here would shard nothing.  Real TP needs
the ``LLM`` engine; see ``TestTPGraniteMoe`` in ``test_tp_integration.py``.

Requires CUDA GPU and vLLM installed (``FusedMoE`` constructs on device). All
tests are skipped otherwise.
"""

import json
import os
import tempfile

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm.config import VllmConfig  # noqa: F401

        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

from tests.shared.vllm_distributed import ensure_distributed as _ensure_distributed

# ── Constants ────────────────────────────────────────────────────────

HIDDEN = 64
# Per-expert width. Kept distinct from the shared-MLP width below so a test can
# tell which module a tensor came from by its shape alone.
EXPERT_INTERMEDIATE = 32
SHARED_INTERMEDIATE = 96
NUM_EXPERTS = 4
TOP_K = 2
NUM_LAYERS = 3  # 1 switch layer + 2 decoder layers
NUM_DECODER_LAYERS = NUM_LAYERS - 1
# Must be in the fused SWITCH kernel's SUPPORTED_RANKS; finalize_weights asserts.
LORA_RANK = 16
CROSS_RANK = 16
SEED = 1234


# ── Helpers ──────────────────────────────────────────────────────────


def _moe_config(*, shared_intermediate_size, num_local_experts=NUM_EXPERTS, **extra):
    """A GraniteSwitchConfig with an arbitrary combination of MLP paths.

    ``shared_intermediate_size=0`` is the pure-sparse (``granitemoe``) shape; a
    positive value with experts is the Granite 4.x hybrid.
    """
    kwargs = dict(
        vocab_size=300,
        hidden_size=HIDDEN,
        intermediate_size=EXPERT_INTERMEDIATE,
        shared_intermediate_size=shared_intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_adapters=1,
        adapter_token_ids=[250],
        adapter_names=["adapter_1"],
        adapter_substitute_token_ids=[1],
        max_lora_rank=LORA_RANK,
        adapter_ranks=[LORA_RANK],
        # Attention-only, exactly like the validated adapter on a pure sparse
        # base: LoRA on the fused 3D expert parameters is not supported.
        lora_target_modules=["qkv_proj", "o_proj"],
        switch_head_dim=32,
        max_position_embeddings=512,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )
    kwargs.update(extra)
    return GraniteSwitchConfig(**kwargs)


def _sr_config(*, shared_intermediate_size, **extra):
    return _moe_config(
        shared_intermediate_size=shared_intermediate_size,
        dual_stream=True,
        cross_stream_rank=CROSS_RANK,
        **extra,
    )


def _build(config):
    """Instantiate the vLLM model on-device, as vLLM's own loader does.

    Each SwitchedLoRALinear caches its device at construction, so building under
    ``with torch.device("cuda")`` is required — building on CPU then ``.to()``
    leaves that cached device stale.
    """
    from granite_switch.vllm import register

    _ensure_distributed()
    register()

    tmpdir = tempfile.mkdtemp(prefix="granite_switch_moe_test_")
    config_dict = config.to_dict()
    config_dict["architectures"] = ["GraniteSwitchForCausalLM"]
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    model_config = ModelConfig(
        model=tmpdir,
        dtype="bfloat16",
        max_model_len=config.max_position_embeddings,
        enforce_eager=True,
    )
    vllm_config = VllmConfig(model_config=model_config)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            with torch.device("cuda"):
                model = GraniteSwitchForCausalLM(vllm_config=vllm_config)
    finally:
        torch.set_default_dtype(old_dtype)

    # The layer names this model registered in the shared static forward context
    # must not leak into the next build in the same process.
    return model, vllm_config


def _release(model, vllm_config):
    sfc = vllm_config.compilation_config.static_forward_context
    for name in [n for n, m in model.named_modules() if hasattr(m, "kv_cache")]:
        sfc.pop(name, None)
    sfc.clear()


def _decoder_layers(model):
    return model.model.layers


def _hf_moe_weights(config, *, num_decoder_layers=NUM_DECODER_LAYERS):
    """The three stacked expert tensors an HF/composed MoE checkpoint carries.

    Named exactly as ``save_pretrained`` writes them — these are what
    ``_try_load_stacked_moe`` has to recognise and fan out.
    """
    torch.manual_seed(SEED)
    out = {}
    for i in range(num_decoder_layers):
        moe = f"model.layers.{i}.block_sparse_moe"
        out[f"{moe}.input_linear.weight"] = torch.randn(
            NUM_EXPERTS, 2 * EXPERT_INTERMEDIATE, config.hidden_size
        )
        out[f"{moe}.output_linear.weight"] = torch.randn(
            NUM_EXPERTS, config.hidden_size, EXPERT_INTERMEDIATE
        )
        out[f"{moe}.router.layer.weight"] = torch.randn(NUM_EXPERTS, config.hidden_size)
    return out


def _unpadded_vocab_rows(model):
    """Row counts a ``VocabParallelEmbedding`` weight loader will accept.

    Its parameter is padded up to a multiple of 64, but the loader asserts the
    INCOMING tensor has exactly ``org_vocab_size`` rows — no checkpoint ships the
    padding. So the embedding and the LM head are the two places where a
    synthetic checkpoint sized from ``param.shape`` is rejected.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    return {
        f"{name}.weight": mod.org_vocab_size
        for name, mod in model.named_modules()
        if isinstance(mod, VocabParallelEmbedding)
    }


def _full_checkpoint(model, config):
    """Every parameter the model wants, with the expert bank in HF stacked form.

    Built by reading the model's own ``named_parameters()`` so the audit added in
    the loaders sees a complete checkpoint — then the expert entries are replaced
    by their three stacked HF tensors, which is the only remap under test.
    """
    vocab_rows = _unpadded_vocab_rows(model)
    weights = {}
    for name, param in model.named_parameters():
        if ".experts.w13_weight" in name or ".experts.w2_weight" in name:
            continue
        if ".block_sparse_moe.gate.weight" in name:
            continue
        rows = vocab_rows.get(name)
        if rows is None:
            weights[name] = torch.randn_like(param, dtype=torch.float32)
        else:
            weights[name] = torch.randn(rows, *param.shape[1:], dtype=torch.float32)
    weights.update(_hf_moe_weights(config))
    return weights


# ── Plain LoRA: construction ─────────────────────────────────────────


class TestLoRAMoEConstruction:
    """Which MLP paths a plain-LoRA decoder layer builds, for each base shape."""

    def test_pure_sparse_has_no_shared_mlp(self):
        """``granitemoe``: expert bank only.

        Before the gate, ``GraniteMoeSharedMLP`` was built unconditionally, and it
        sets ``self.hidden_size = config.shared_intermediate_size`` — so this
        registered ``[0, H]`` / ``[H, 0]`` weights no checkpoint ships and then
        added their output to the MoE result.
        """
        model, vc = _build(_moe_config(shared_intermediate_size=0))
        try:
            for layer in _decoder_layers(model):
                assert layer.has_experts
                assert not layer.has_shared_mlp
                assert layer.shared_mlp is None
                assert hasattr(layer, "block_sparse_moe")
        finally:
            _release(model, vc)

    def test_pure_sparse_has_no_zero_width_params(self):
        """The regression the gate exists to prevent, stated directly."""
        model, vc = _build(_moe_config(shared_intermediate_size=0))
        try:
            for name, param in model.named_parameters():
                assert "shared_mlp" not in name, f"unexpected shared_mlp param {name}"
                assert 0 not in param.shape, (
                    f"zero-width param {name}: {tuple(param.shape)}"
                )
        finally:
            _release(model, vc)

    def test_hybrid_keeps_both_paths(self):
        """Granite 4.x MoE hybrid must be unchanged by the gate."""
        model, vc = _build(_moe_config(shared_intermediate_size=SHARED_INTERMEDIATE))
        try:
            for layer in _decoder_layers(model):
                assert layer.has_experts
                assert layer.has_shared_mlp
                assert layer.shared_mlp is not None
        finally:
            _release(model, vc)

    def test_dense_only_keeps_shared_mlp(self):
        """Granite 4.0/4.1: shared MLP alone, no expert bank."""
        model, vc = _build(
            _moe_config(
                shared_intermediate_size=SHARED_INTERMEDIATE, num_local_experts=0
            )
        )
        try:
            for layer in _decoder_layers(model):
                assert not layer.has_experts
                assert layer.has_shared_mlp
                assert not hasattr(layer, "block_sparse_moe")
        finally:
            _release(model, vc)

    def test_neither_path_is_rejected(self):
        """A layer with no MLP at all is a config error, not a silent no-op."""
        with pytest.raises(ValueError, match="at least one MLP path"):
            _build(_moe_config(shared_intermediate_size=0, num_local_experts=0))


# ── Shadow Residual: construction ────────────────────────────────────


class TestSRMoEConstruction:
    """SR used to refuse every MoE base outright; it now serves them."""

    def test_pure_sparse_builds(self):
        model, vc = _build(_sr_config(shared_intermediate_size=0))
        try:
            for layer in _decoder_layers(model):
                assert layer.has_experts
                assert not layer.has_shared_mlp
                assert layer.shared_mlp is None
                # Shared routing needs the routing/apply seam that
                # GraniteMoeMoE.forward hides. If upstream stops exposing these
                # as public attributes, the SR forward is silently wrong.
                assert hasattr(layer.block_sparse_moe, "gate")
                assert hasattr(layer.block_sparse_moe, "experts")
        finally:
            _release(model, vc)

    def test_hybrid_builds_with_lora_on_the_shared_mlp(self):
        """Only routing is ever shared — the shared MLP stays per-stream LoRA."""
        from granite_switch.vllm.core.lora import SwitchedLoRALinear

        model, vc = _build(_sr_config(shared_intermediate_size=SHARED_INTERMEDIATE))
        try:
            for layer in _decoder_layers(model):
                assert layer.has_experts
                assert layer.has_shared_mlp
                assert isinstance(layer.shared_mlp.input_linear, SwitchedLoRALinear)
                assert isinstance(layer.shared_mlp.output_linear, SwitchedLoRALinear)
        finally:
            _release(model, vc)

    def test_dense_sr_is_unaffected(self):
        model, vc = _build(
            _sr_config(
                shared_intermediate_size=SHARED_INTERMEDIATE,
                num_local_experts=0,
            )
        )
        try:
            for layer in _decoder_layers(model):
                assert not layer.has_experts
                assert layer.has_shared_mlp
        finally:
            _release(model, vc)


# ── Weight loading: the HF stacked -> FusedMoE remap ─────────────────


class _WeightLoadBase:
    """Shared assertions over whichever decoder interface the config selects."""

    def _load(self, config):
        model, vc = _build(config)
        weights = _full_checkpoint(model, config)
        loaded = model.load_weights(list(weights.items()))
        return model, vc, weights, loaded

    def _assert_experts_populated(self, model, weights):
        """Each stacked HF tensor must land in the right FusedMoE slot.

        ``w13_weight`` is ``[E, 2I, H]`` with w1 (gate) above w3 (up) — the same
        split HF stores, so the remap is a straight per-expert copy and can be
        checked exactly.
        """
        for i, layer in enumerate(_decoder_layers(model)):
            moe = f"model.layers.{i}.block_sparse_moe"
            experts = layer.block_sparse_moe.experts
            src_in = weights[f"{moe}.input_linear.weight"]
            src_out = weights[f"{moe}.output_linear.weight"]
            src_gate = weights[f"{moe}.router.layer.weight"]

            for e in range(NUM_EXPERTS):
                w1, w3 = src_in[e].chunk(2, dim=0)
                torch.testing.assert_close(
                    experts.w13_weight[e, :EXPERT_INTERMEDIATE].float().cpu(),
                    w1,
                    atol=1e-2,
                    rtol=1e-2,
                )
                torch.testing.assert_close(
                    experts.w13_weight[e, EXPERT_INTERMEDIATE:].float().cpu(),
                    w3,
                    atol=1e-2,
                    rtol=1e-2,
                )
                torch.testing.assert_close(
                    experts.w2_weight[e].float().cpu(),
                    src_out[e],
                    atol=1e-2,
                    rtol=1e-2,
                )

            torch.testing.assert_close(
                layer.block_sparse_moe.gate.weight.float().cpu(),
                src_gate,
                atol=1e-2,
                rtol=1e-2,
            )


class TestLoRAMoEWeightLoad(_WeightLoadBase):
    def test_stacked_experts_land_in_fused_moe(self):
        config = _moe_config(shared_intermediate_size=0)
        model, vc, weights, _ = self._load(config)
        try:
            self._assert_experts_populated(model, weights)
        finally:
            _release(model, vc)

    def test_missing_expert_bank_raises(self):
        """The audit added alongside this remap is what makes a typo visible.

        Dropping ``input_linear`` used to produce a ``logger.warning`` and a model
        that served uninitialised memory as plausible-looking tokens.
        """
        config = _moe_config(shared_intermediate_size=0)
        model, vc = _build(config)
        try:
            weights = {
                k: v
                for k, v in _full_checkpoint(model, config).items()
                if not k.endswith(".block_sparse_moe.input_linear.weight")
            }
            with pytest.raises(ValueError, match="UNINITIALIZED"):
                model.load_weights(list(weights.items()))
        finally:
            _release(model, vc)

    def test_missing_lora_delta_is_tolerated(self):
        """A zeroed LoRA parameter is a *correct* absence, not an error.

        Every ``lora_A``/``lora_B`` is constructed zeroed, so an adapter that does
        not target a fused slice legitimately ships nothing for it. The audit must
        keep warning rather than raising here, or it fires on good checkpoints.
        """
        config = _moe_config(shared_intermediate_size=0)
        model, vc = _build(config)
        try:
            weights = {
                k: v
                for k, v in _full_checkpoint(model, config).items()
                if "lora_B" not in k
            }
            assert model.load_weights(list(weights.items())) is not None
        finally:
            _release(model, vc)


class TestSRMoEWeightLoad(_WeightLoadBase):
    def test_stacked_experts_land_in_fused_moe(self):
        """The SR loader had no ``block_sparse_moe`` handling whatsoever.

        A composed SR checkpoint is saved from the HF SR model, whose layer holds
        the same ``GraniteMoeHybridMoE``, so the tensor names and stacked shapes
        are identical to the LoRA case — hence one shared remap helper rather than
        two copies that can drift.
        """
        config = _sr_config(shared_intermediate_size=0)
        model, vc, weights, _ = self._load(config)
        try:
            self._assert_experts_populated(model, weights)
        finally:
            _release(model, vc)

    def test_missing_expert_bank_raises(self):
        config = _sr_config(shared_intermediate_size=0)
        model, vc = _build(config)
        try:
            weights = {
                k: v
                for k, v in _full_checkpoint(model, config).items()
                if not k.endswith(".block_sparse_moe.output_linear.weight")
            }
            with pytest.raises(ValueError, match="UNINITIALIZED"):
                model.load_weights(list(weights.items()))
        finally:
            _release(model, vc)
