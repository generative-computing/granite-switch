# SPDX-License-Identifier: Apache-2.0
"""HF → vLLM weight compatibility integration tests.

The core workflow of granite-switch is "build/train with HF, deploy with vLLM".
Parameter names match exactly by design (confirmed by code analysis), but this
was previously untested.  These tests verify forward equivalence: same input
sequence, same logits from both backends.

Pipeline:
1. Create a tiny HF GraniteSwitchForCausalLM with seeded random weights
2. Set non-zero LoRA weights so adapters produce visible deltas
3. save_pretrained() to a temp directory
4. Load in vLLM: construct VllmConfig from saved config.json, build model,
   call load_weights() with the safetensors file
5. Run the same input through both models and compare logits

Requires CUDA GPU and vLLM installed.  All tests are skipped otherwise.
"""

import os

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm.config import VllmConfig  # noqa: F401
        from vllm.forward_context import (  # noqa: F401
            ForwardContext,
            override_forward_context,
        )
        from vllm.model_executor.layers.attention import Attention  # noqa: F401

        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from safetensors.torch import load_file
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM as HFModel
    from granite_switch.vllm import register
    from granite_switch.vllm.granite_switch_model import (
        GraniteSwitchForCausalLM as VLLMModel,
    )

# ── Constants ────────────────────────────────────────────────────────

BLOCK_SIZE = 16
MAX_TOKENS = 1024
SEED = 42


# ── Distributed init ─────────────────────────────────────────────────

from tests.shared.vllm_distributed import ensure_distributed as _ensure_distributed

# ── Helpers ──────────────────────────────────────────────────────────


def _set_adapter_token_ids_hf(model, token_ids):
    """Populate HF model.model.adapter_token_ids from a list of ints."""
    model.model.adapter_token_ids.data = torch.tensor(token_ids, dtype=torch.long)


def _set_nonzero_lora_hf(model, scale=0.1):
    """Set non-zero LoRA weights on every LoRA layer in the HF model.

    HF uses kaiming for lora_A by default, but we set both lora_A and lora_B
    explicitly for reproducibility.
    """
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.shared_mlp
            # QKV (MergedSwitchedLoRALinear)
            if hasattr(attn.qkv_proj, "lora_A_slices"):
                for a in attn.qkv_proj.lora_A_slices:
                    a.data = torch.randn_like(a) * scale
            if hasattr(attn.qkv_proj, "lora_B_slices"):
                for b in attn.qkv_proj.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            # O proj (SwitchedLoRALinear)
            if hasattr(attn.o_proj, "lora_A"):
                attn.o_proj.lora_A.data = torch.randn_like(attn.o_proj.lora_A) * scale
                attn.o_proj.lora_B.data = torch.randn_like(attn.o_proj.lora_B) * scale
            # input_linear (MergedSwitchedLoRALinear)
            if hasattr(mlp.input_linear, "lora_A_slices"):
                for a in mlp.input_linear.lora_A_slices:
                    a.data = torch.randn_like(a) * scale
            if hasattr(mlp.input_linear, "lora_B_slices"):
                for b in mlp.input_linear.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            # output_linear (SwitchedLoRALinear)
            if hasattr(mlp.output_linear, "lora_A"):
                mlp.output_linear.lora_A.data = (
                    torch.randn_like(mlp.output_linear.lora_A) * scale
                )
                mlp.output_linear.lora_B.data = (
                    torch.randn_like(mlp.output_linear.lora_B) * scale
                )


def _make_vllm_config(tmpdir, config):
    """Create a VllmConfig from a saved config.json directory.

    Reuses the _make_vllm_config pattern from tests/vllm/test_model_forward.py
    but uses an existing directory (where HF already saved config.json).
    """
    register()

    model_config = ModelConfig(
        model=tmpdir,
        dtype="bfloat16",
        max_model_len=config.max_position_embeddings,
        enforce_eager=True,
    )

    vllm_config = VllmConfig(model_config=model_config)
    return vllm_config


def _load_safetensors_as_iterable(tmpdir):
    """Load all safetensors files from a directory as (name, tensor) pairs."""
    import glob

    safetensors_files = sorted(glob.glob(os.path.join(tmpdir, "*.safetensors")))
    for sf_path in safetensors_files:
        state_dict = load_file(sf_path)
        yield from state_dict.items()


# ── Base test class ──────────────────────────────────────────────────


class _HFToVLLMWeightTestBase:
    """Base class for HF→vLLM weight compatibility tests.

    Subclasses provide _config() and _input_ids() for specific switch types.
    """

    def _config(self):
        """Return a GraniteSwitchConfig for this test."""
        raise NotImplementedError

    def _input_ids(self):
        """Return a list of token IDs for the test sequence."""
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        _ensure_distributed()

        self.device = torch.device("cuda")
        self.config = self._config()
        self.tmpdir = str(tmp_path / "model")

        # ── Step 1: Create HF model, save to disk ──
        torch.manual_seed(SEED)
        hf_model = HFModel(self.config).eval()
        _set_adapter_token_ids_hf(hf_model, self.config.adapter_token_ids)
        _set_nonzero_lora_hf(hf_model)
        hf_model.save_pretrained(self.tmpdir)

        # Keep HF model on CPU for forward pass
        self.hf_model = hf_model

        # ── Step 2: Create vLLM model, load weights from saved checkpoint ──
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        try:
            self.vllm_config = _make_vllm_config(self.tmpdir, self.config)
            with set_current_vllm_config(self.vllm_config):
                self.vllm_model = VLLMModel(
                    vllm_config=self.vllm_config,
                ).to(self.device)
        finally:
            torch.set_default_dtype(old_dtype)

        # Load weights from the HF checkpoint
        weights = list(_load_safetensors_as_iterable(self.tmpdir))
        self.vllm_model.load_weights(weights)

        # adapter_token_ids are now sourced from config in
        # vLLM model __init__ (no longer nn.Parameters).

        # ── Step 3: Set up vLLM KV caches ──
        self._setup_vllm_kv_caches()

        yield

        # Clean up static_forward_context entries
        sfc = self.vllm_config.compilation_config.static_forward_context
        for name in list(self._layer_names):
            sfc.pop(name, None)

    def _setup_vllm_kv_caches(self):
        """Allocate KV caches for all attention layers in the vLLM model."""
        self._kv_caches = []
        self._attention_map = {}

        # Subtract the switch's cache slot count to get real decoder layers.
        # SingleSwitch uses 1 cache slot.
        layer_offset = self.vllm_model.model.switch.num_cache_layers
        num_decoder_layers = self.config.num_hidden_layers - layer_offset
        num_blocks = (MAX_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

        # Switch attention
        switch = self.vllm_model.model.switch
        self._setup_single_attn(switch.attn, "switch.layers.0", num_blocks)

        # Decoder attention layers
        for i in range(num_decoder_layers):
            layer_attn = self.vllm_model.model.layers[i].self_attn.attn
            layer_name = f"model.layers.{i}.self_attn.attn"
            self._setup_single_attn(layer_attn, layer_name, num_blocks)

        self._layer_names = list(self._attention_map.keys())

    def _setup_single_attn(self, attn, layer_name, num_blocks):
        """Configure a single Attention layer with KV cache."""
        attn.kv_cache_torch_dtype = torch.bfloat16
        cache_shape = attn.attn_backend.get_kv_cache_shape(
            num_blocks,
            BLOCK_SIZE,
            attn.num_kv_heads,
            attn.head_size,
        )
        kv_cache = torch.zeros(cache_shape, device=self.device, dtype=torch.bfloat16)
        attn.kv_cache = kv_cache
        self._kv_caches.append(kv_cache)
        self._attention_map[layer_name] = attn

    def _run_hf_forward(self, input_ids_list):
        """Run HF model forward, return logits [1, seq, vocab] on CPU."""
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)
        with torch.no_grad():
            output = self.hf_model(input_ids=input_ids)
        return output.logits  # [1, seq, vocab]

    def _run_vllm_forward(self, input_ids_list):
        """Run vLLM model forward + compute_logits, return logits [tokens, vocab] on CUDA."""
        for kv_cache in self._kv_caches:
            kv_cache.zero_()

        seq_len = len(input_ids_list)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)

        # Build metadata
        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        num_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table = torch.arange(
            num_blocks_needed,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)
        query_start_loc = torch.tensor(
            [0, seq_len],
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=self.device)

        backend_name = next(iter(self._attention_map.values())).attn_backend.get_name()
        if backend_name == "FLASH_ATTN":
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

            metadata = FlashAttentionMetadata(
                num_actual_tokens=seq_len,
                max_query_len=seq_len,
                query_start_loc=query_start_loc,
                max_seq_len=seq_len,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=slot_mapping,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
                causal=True,
            )
        else:
            pytest.skip(
                f"Backend {backend_name}: metadata not implemented for this test"
            )

        attn_metadata = {name: metadata for name in self._layer_names}
        slot_mapping_dict = {name: slot_mapping for name in self._layer_names}

        forward_ctx = ForwardContext(
            no_compile_layers=self.vllm_config.compilation_config.static_forward_context,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping_dict,
        )

        old_direct = {}
        for name, attn in self._attention_map.items():
            old_direct[name] = attn.use_direct_call
            attn.use_direct_call = True

        try:
            with override_forward_context(forward_ctx):
                hidden_states = self.vllm_model.forward(
                    input_ids=input_ids,
                    positions=positions,
                )
        finally:
            for name, attn in self._attention_map.items():
                attn.use_direct_call = old_direct[name]

        logits = self.vllm_model.compute_logits(hidden_states)
        return logits  # [tokens, vocab]

    def test_forward_logit_equivalence(self):
        """Same input → same logits from HF and vLLM backends.

        Compares HF logits [1, seq, vocab] with vLLM logits [tokens, vocab]
        after reshaping.  Uses bf16 tolerance since vLLM operates in bf16
        while HF uses fp32 by default — we cast HF logits to bf16 for comparison.

        Reports the achieved tolerance (max absolute and relative differences)
        so regressions are visible even when the test passes.
        """
        self.hf_model.eval()
        self.vllm_model.eval()

        input_ids_list = self._input_ids()

        with torch.no_grad():
            hf_logits = self._run_hf_forward(input_ids_list)  # [1, seq, vocab]
            vllm_logits = self._run_vllm_forward(input_ids_list)  # [tokens, vocab]

        # Reshape HF logits to [tokens, vocab] for comparison
        hf_logits_2d = hf_logits[0].to(self.device)  # [seq, vocab]

        # Cast HF logits to bf16 for apples-to-apples comparison
        # (HF runs in fp32 by default, vLLM runs in bf16)
        hf_logits_bf16 = hf_logits_2d.to(torch.bfloat16)

        # Compute and report achieved tolerances
        abs_diff = (hf_logits_bf16 - vllm_logits).abs()
        max_atol = abs_diff.max().item()

        # Relative difference: |a - b| / max(|a|, |b|), avoiding div-by-zero
        denom = torch.maximum(hf_logits_bf16.abs(), vllm_logits.abs())
        rel_diff = abs_diff / denom.clamp(min=1e-10)
        max_rtol = rel_diff.max().item()
        mean_atol = abs_diff.mean().item()

        print(
            f"\n  Logit equivalence: max_atol={max_atol:.2e}, "
            f"mean_atol={mean_atol:.2e}, max_rtol={max_rtol:.2e}"
        )

        torch.testing.assert_close(
            hf_logits_bf16,
            vllm_logits,
            atol=1e-2,
            rtol=1e-2,
            msg=f"HF and vLLM logits diverged: max_atol={max_atol:.2e}, max_rtol={max_rtol:.2e}",
        )


# ════════════════════════════════════════════════════════════════════
# SingleSwitch forward equivalence
# ════════════════════════════════════════════════════════════════════


class TestSingleSwitchForwardEquivalence(_HFToVLLMWeightTestBase):
    """Verify HF→vLLM weight loading produces equivalent logits for SingleSwitch."""

    def _config(self):
        return GraniteSwitchConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_adapters=2,
            adapter_token_ids=[250, 251],
            adapter_names=["adapter_1", "adapter_2"],
            adapter_substitute_token_ids=[1, 1],
            max_lora_rank=4,
            adapter_ranks=[4, 4],
            switch_head_dim=32,
            max_position_embeddings=512,
            attention_multiplier=1.0,
            embedding_multiplier=1.0,
            residual_multiplier=1.0,
            logits_scaling=1.0,
        )

    def _input_ids(self):
        # Include a control token (250 = adapter 1) at position 2
        return [10, 20, 250, 30, 40, 50, 60, 70]
