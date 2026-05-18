# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward pass tests for GraniteSwitchForCausalLM (vLLM backend).
Inner file — run by test_model_forward.py in subprocess.

Tests the full model wiring: switch -> adapter_indices -> decoder layers.
All tests run on a single CUDA GPU with random weights (no pretrained checkpoint needed).

Requires CUDA GPU and vLLM installed. All tests are skipped otherwise.
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
        from vllm.model_executor.layers.attention.attention import Attention  # noqa: F401
        from vllm.forward_context import ForwardContext, override_forward_context  # noqa: F401
        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM
    from granite_switch.vllm.switch.single import SingleSwitch

# ── Constants ────────────────────────────────────────────────────────

BLOCK_SIZE = 16
MAX_TOKENS = 1024
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────

def _tiny_vllm_config():
    """Minimal GraniteSwitchConfig for single-GPU vLLM tests."""
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_names=["adapter_1", "adapter_2"],
        hiding_groups={"all_controls": ["adapter_1", "adapter_2"]},
        hiding_policy={"base": ["all_controls"], "adapter_1": ["all_controls"], "adapter_2": ["all_controls"]},
        adapter_third_party=["adapter_1", "adapter_2"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=32,
        control_dims=32,
        max_position_embeddings=512,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


def _tiny_vllm_mixed_tp_config():
    """SingleSwitch config where only adapter_1 is third-party."""
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_names=["adapter_1", "adapter_2"],
        hiding_groups={"all_controls": ["adapter_1", "adapter_2"]},
        hiding_policy={"base": ["all_controls"], "adapter_1": ["all_controls"], "adapter_2": ["all_controls"]},
        adapter_third_party=["adapter_1"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=32,
        control_dims=32,
        max_position_embeddings=512,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


def _tiny_vllm_config_no_adapters():
    """Minimal GraniteSwitchConfig with no adapters."""
    return GraniteSwitchConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_adapters=0,
        max_position_embeddings=512,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


from tests.shared.vllm_distributed import ensure_distributed as _ensure_distributed


def _make_vllm_config(config):
    """Create a VllmConfig with a real ModelConfig from our GraniteSwitchConfig."""
    from vllm.config import ModelConfig
    from granite_switch.vllm import register
    register()

    tmpdir = tempfile.mkdtemp(prefix="granite_switch_test_")
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
    return vllm_config


def _init_model_weights(model):
    """Initialize base model weights that vLLM leaves uninitialized."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if 'lora_A' in name or 'lora_B' in name:
                continue
            if 'layernorm' in name or 'norm' in name:
                continue
            param.data.normal_(0, 0.02)


def _set_nonzero_lora(model, scale=0.1):
    """Set non-zero lora_A and lora_B on every LoRA layer."""
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.shared_mlp
            if hasattr(attn.qkv_proj, "lora_A_slices"):
                for a in attn.qkv_proj.lora_A_slices:
                    a.data = torch.randn_like(a) * scale
                for b in attn.qkv_proj.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            if hasattr(attn.o_proj, "lora_A"):
                attn.o_proj.lora_A.data = torch.randn_like(attn.o_proj.lora_A) * scale
                attn.o_proj.lora_B.data = torch.randn_like(attn.o_proj.lora_B) * scale
            if hasattr(mlp.input_linear, "lora_A_slices"):
                for a in mlp.input_linear.lora_A_slices:
                    a.data = torch.randn_like(a) * scale
                for b in mlp.input_linear.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            if hasattr(mlp.output_linear, "lora_A"):
                mlp.output_linear.lora_A.data = torch.randn_like(mlp.output_linear.lora_A) * scale
                mlp.output_linear.lora_B.data = torch.randn_like(mlp.output_linear.lora_B) * scale


# ── Base test class ──────────────────────────────────────────────────

class _VLLMModelTestBase:
    """Base class providing model creation and full forward pass machinery."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        _ensure_distributed()

        self.device = torch.device("cuda")
        self.config = _tiny_vllm_config()

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        try:
            self.vllm_config = _make_vllm_config(self.config)
            with set_current_vllm_config(self.vllm_config):
                self.model = GraniteSwitchForCausalLM(
                    vllm_config=self.vllm_config,
                ).to(self.device)
        finally:
            torch.set_default_dtype(old_dtype)

        torch.manual_seed(SEED)
        _init_model_weights(self.model)

        self.model_config = self.model.config
        self._setup_kv_caches()

        yield

        sfc = self.vllm_config.compilation_config.static_forward_context
        for name in list(self._layer_names):
            sfc.pop(name, None)

    def _setup_kv_caches(self):
        self._kv_caches = []
        self._attention_map = {}

        num_decoder_layers = self.config.num_hidden_layers - 1
        num_blocks = (MAX_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

        switch_attn = self.model.model.switch.attn
        self._setup_single_attn(switch_attn, "switch.layers.0", num_blocks)

        for i in range(num_decoder_layers):
            layer_attn = self.model.model.layers[i].self_attn.attn
            layer_name = f"model.layers.{i}.self_attn.attn"
            self._setup_single_attn(layer_attn, layer_name, num_blocks)

        self._layer_names = list(self._attention_map.keys())

    def _setup_single_attn(self, attn, layer_name, num_blocks):
        attn.kv_cache_torch_dtype = torch.bfloat16
        cache_shape = attn.attn_backend.get_kv_cache_shape(
            num_blocks, BLOCK_SIZE, attn.num_kv_heads, attn.head_size,
        )
        kv_cache = torch.zeros(cache_shape, device=self.device, dtype=torch.bfloat16)
        attn.kv_cache = kv_cache
        self._kv_caches.append(kv_cache)
        self._attention_map[layer_name] = attn

    def _build_metadata(self, seq_len):
        device = self.device
        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
        num_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table = torch.arange(
            num_blocks_needed, dtype=torch.int32, device=device,
        ).unsqueeze(0)
        query_start_loc = torch.tensor(
            [0, seq_len], dtype=torch.int32, device=device,
        )
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

        backend_name = list(self._attention_map.values())[0].attn_backend.get_name()
        if backend_name == "FLASH_ATTN":
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

            # scheduler_metadata is FA3-only — passing it on FA2 (Ampere/A100)
            # forces FA3 kernel dispatch and crashes with "no kernel image
            # is available". Only compute it when get_flash_attn_version() == 3
            # (Hopper SM90+).
            scheduler_metadata = None
            try:
                from vllm.v1.attention.backends.fa_utils import (
                    get_flash_attn_version,
                    get_scheduler_metadata,
                )
                if get_flash_attn_version() == 3:
                    first_attn = list(self._attention_map.values())[0]
                    scheduler_metadata = get_scheduler_metadata(
                        batch_size=1,
                        max_seqlen_q=seq_len,
                        max_seqlen_k=seq_len,
                        num_heads_q=first_attn.num_heads,
                        num_heads_kv=first_attn.num_kv_heads,
                        headdim=first_attn.head_size,
                        cache_seqlens=seq_lens,
                        qkv_dtype=torch.bfloat16,
                        cu_seqlens_q=query_start_loc,
                        page_size=BLOCK_SIZE,
                        causal=True,
                        num_splits=0,
                    )
            except ImportError:
                pass

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
                scheduler_metadata=scheduler_metadata,
            )
        else:
            pytest.skip(f"Backend {backend_name}: metadata not implemented for this test")

        return metadata, slot_mapping

    def _run_forward(self, input_ids_list):
        for kv_cache in self._kv_caches:
            kv_cache.zero_()

        seq_len = len(input_ids_list)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)

        metadata, slot_mapping = self._build_metadata(seq_len)

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
                hidden_states = self.model.forward(
                    input_ids=input_ids,
                    positions=positions,
                )
        finally:
            for name, attn in self._attention_map.items():
                attn.use_direct_call = old_direct[name]

        return hidden_states

    def _run_forward_and_logits(self, input_ids_list):
        hidden_states = self._run_forward(input_ids_list)
        logits = self.model.compute_logits(hidden_states)
        return logits


# ════════════════════════════════════════════════════════════════════
# 1. Model instantiation
# ════════════════════════════════════════════════════════════════════

class TestModelInstantiation(_VLLMModelTestBase):

    def test_single_switch_model_creates(self):
        assert isinstance(self.model.model.switch, SingleSwitch)
        num_decoder_layers = self.config.num_hidden_layers - 1
        assert len(self.model.model.layers) == num_decoder_layers

    def test_no_adapter_model_creates(self):
        config = _tiny_vllm_config_no_adapters()
        vllm_config = _make_vllm_config(config)

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(vllm_config):
                model = GraniteSwitchForCausalLM(
                    vllm_config=vllm_config,
                ).to(self.device)
        finally:
            torch.set_default_dtype(old_dtype)

        assert model.model.switch is None
        assert len(model.model.layers) == 2


# ════════════════════════════════════════════════════════════════════
# 2. Forward output shape
# ════════════════════════════════════════════════════════════════════

class TestForwardOutputShape(_VLLMModelTestBase):

    def test_basic_output_shape(self):
        self.model.eval()
        input_ids_list = [10, 20, 30, 40, 50, 60, 70, 80]
        with torch.no_grad():
            hidden_states = self._run_forward(input_ids_list)
        assert hidden_states.shape == (8, self.config.hidden_size)

    def test_longer_sequence_shape(self):
        self.model.eval()
        input_ids_list = list(range(1, 33))
        with torch.no_grad():
            hidden_states = self._run_forward(input_ids_list)
        assert hidden_states.shape == (32, self.config.hidden_size)


# ════════════════════════════════════════════════════════════════════
# 4. Adapter indices wiring
# ════════════════════════════════════════════════════════════════════

class TestAdapterIndicesWiring(_VLLMModelTestBase):

    def test_control_token_activates_adapter(self):
        torch.manual_seed(SEED)
        _set_nonzero_lora(self.model)
        self.model.eval()

        with_ctrl = [10, 20, 250, 30, 40, 50, 60, 70]
        no_ctrl = [10, 20, 100, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_ctrl = self._run_forward_and_logits(with_ctrl)
            logits_text = self._run_forward_and_logits(no_ctrl)

        torch.testing.assert_close(logits_ctrl[:2], logits_text[:2])

        assert not torch.allclose(logits_ctrl[3:], logits_text[3:]), \
            "Post-control logits should differ when adapter is active"

    def test_different_adapters_produce_different_logits(self):
        torch.manual_seed(SEED)
        _set_nonzero_lora(self.model)
        self.model.eval()

        seq_a1 = [10, 20, 250, 30, 40, 50, 60, 70]
        seq_a2 = [10, 20, 251, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_a1 = self._run_forward_and_logits(seq_a1)
            logits_a2 = self._run_forward_and_logits(seq_a2)

        torch.testing.assert_close(logits_a1[:2], logits_a2[:2])

        assert not torch.allclose(logits_a1[3:], logits_a2[3:]), \
            "Different adapters should produce different post-control logits"


# ════════════════════════════════════════════════════════════════════
# 5. Control token KV invisibility
# ════════════════════════════════════════════════════════════════════

class TestControlTokenKVInvisibility(_VLLMModelTestBase):

    def test_control_token_invisible_to_future_positions(self):
        torch.manual_seed(SEED)
        self.model.eval()

        seq = [10, 20, 250, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_a = self._run_forward_and_logits(seq)

        with torch.no_grad():
            perturbation = torch.randn(
                self.config.hidden_size, device=self.device, dtype=torch.bfloat16
            ) * 10.0
            self.model.model.embed_tokens.weight.data[250] += perturbation

        with torch.no_grad():
            logits_b = self._run_forward_and_logits(seq)

        torch.testing.assert_close(
            logits_a[:2], logits_b[:2],
            msg="Pre-control logits should be identical"
        )

        assert not torch.allclose(logits_a[2], logits_b[2]), \
            "Control token logits should differ after perturbation"

        torch.testing.assert_close(
            logits_a[3:], logits_b[3:],
            msg="Post-control logits should be identical "
                "(control token KV masked by control_dims)"
        )


# ════════════════════════════════════════════════════════════════════
# 6. KV visibility tests
# ════════════════════════════════════════════════════════════════════

class TestKVVisibility(_VLLMModelTestBase):

    def test_adapter_token_kv_invisible(self):
        torch.manual_seed(SEED)
        self.model.eval()

        seq = [10, 20, 250, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_a = self._run_forward_and_logits(seq)

        with torch.no_grad():
            perturbation = torch.randn(
                self.config.hidden_size, device=self.device, dtype=torch.bfloat16
            ) * 10.0
            self.model.model.embed_tokens.weight.data[250] += perturbation

        with torch.no_grad():
            logits_b = self._run_forward_and_logits(seq)

        torch.testing.assert_close(
            logits_a[3:], logits_b[3:],
            msg="Post-adapter-token logits should be identical"
        )
