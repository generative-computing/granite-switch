# SPDX-License-Identifier: Apache-2.0
"""NaN regression tests — control token at sequence position 0 (vLLM backend).
Inner file — run by test_position_zero_nan.py in subprocess.

Combines:
  - vLLM-specific unit tests for GraniteLoRAEmbeddedAttention._expand_with_control_dimensions
    (flat token layout: [num_tokens, num_heads * head_dim])
  - Shared SDPANaNCases and ModelFinitenessCases from tests/shared/position_zero_nan_cases.py

Requires CUDA GPU and vLLM installed.
"""

import types
import json
import os
import tempfile

import pytest
import torch

from tests.shared.position_zero_nan_cases import ModelFinitenessCases, SDPANaNCases

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
    from vllm.config import VllmConfig, ModelConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM
    from granite_switch.vllm.core.decoder import GraniteLoRAEmbeddedAttention

from tests.shared.vllm_distributed import ensure_distributed as _ensure_distributed

BLOCK_SIZE = 16
MAX_TOKENS = 512
SEED = 42


# ── vLLM config + ctrl token ID ────────────────────────────────────


def _make_config():
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
        hiding_policy={
            "base": ["all_controls"],
            "adapter_1": ["all_controls"],
            "adapter_2": ["all_controls"],
        },
        adapter_third_party=["adapter_1", "adapter_2"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=32,
        control_dims=32,
        max_position_embeddings=MAX_TOKENS,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


_CTRL_TOKEN = 250


# ── vLLM model runner ───────────────────────────────────────────────


def _make_vllm_config(config):
    from granite_switch.vllm import register
    register()
    tmpdir = tempfile.mkdtemp(prefix="gs_nan_test_")
    cfg_dict = config.to_dict()
    cfg_dict["architectures"] = ["GraniteSwitchForCausalLM"]
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg_dict, f)
    model_config = ModelConfig(
        model=tmpdir,
        dtype="bfloat16",
        max_model_len=config.max_position_embeddings,
        enforce_eager=True,
    )
    return VllmConfig(model_config=model_config)


def _init_weights(model):
    torch.manual_seed(SEED)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if "lora_A" in name or "lora_B" in name:
                continue
            if "layernorm" in name or "norm" in name:
                continue
            param.data.normal_(0, 0.02)


def _setup_kv_caches(model, config, vllm_config, device):
    kv_caches = []
    attention_map = {}
    num_blocks = (MAX_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

    def _add(attn, name):
        attn.kv_cache_torch_dtype = torch.bfloat16
        shape = attn.attn_backend.get_kv_cache_shape(
            num_blocks, BLOCK_SIZE, attn.num_kv_heads, attn.head_size,
        )
        kv = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        attn.kv_cache = kv
        kv_caches.append(kv)
        attention_map[name] = attn

    sw = model.model.switch
    _add(sw.attn, "switch.layers.0")
    num_decoder = config.num_hidden_layers - sw.num_cache_layers
    for i in range(num_decoder):
        _add(model.model.layers[i].self_attn.attn, f"model.layers.{i}.self_attn.attn")

    return kv_caches, attention_map


def _build_metadata(attention_map, seq_len, device):
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    backend_name = list(attention_map.values())[0].attn_backend.get_name()
    if backend_name == "FLASH_ATTN":
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        # FA3 requires scheduler_metadata; compute it when available.
        scheduler_metadata = None
        try:
            from vllm.v1.attention.backends.fa_utils import (
                get_scheduler_metadata,
            )
            first_attn = list(attention_map.values())[0]
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


def _run_vllm_forward_is_finite(ctrl_pos, seq_len, seed):
    """Create a vLLM switch model and check for finite output at the given ctrl_pos."""
    _ensure_distributed()
    device = torch.device("cuda")
    config = _make_config()

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        vllm_config = _make_vllm_config(config)
        with set_current_vllm_config(vllm_config):
            model = GraniteSwitchForCausalLM(vllm_config=vllm_config).to(device)
    finally:
        torch.set_default_dtype(old_dtype)

    _init_weights(model)
    kv_caches, attention_map = _setup_kv_caches(model, config, vllm_config, device)

    # Build input: ctrl_token at ctrl_pos, random content elsewhere
    torch.manual_seed(seed)
    ctrl_id = _CTRL_TOKEN
    content = torch.randint(0, 100, (seq_len,)).tolist()
    ids_list = content[:ctrl_pos] + [ctrl_id] + content[ctrl_pos:]
    total_len = len(ids_list)

    input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
    positions = torch.arange(total_len, dtype=torch.long, device=device)
    metadata, slot_mapping = _build_metadata(attention_map, total_len, device)

    layer_names = list(attention_map.keys())
    attn_metadata = {n: metadata for n in layer_names}
    slot_mapping_dict = {n: slot_mapping for n in layer_names}

    forward_ctx = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping_dict,
    )

    old_direct = {n: attention_map[n].use_direct_call for n in layer_names}
    for n in layer_names:
        attention_map[n].use_direct_call = True

    try:
        for kv in kv_caches:
            kv.zero_()
        with override_forward_context(forward_ctx):
            hidden = model.forward(input_ids=input_ids, positions=positions)
        logits = model.compute_logits(hidden)
    finally:
        for n in layer_names:
            attention_map[n].use_direct_call = old_direct[n]

    sfc = vllm_config.compilation_config.static_forward_context
    for n in layer_names:
        sfc.pop(n, None)

    return bool(logits.isfinite().all())


# ════════════════════════════════════════════════════════════════════
# 1. vLLM-specific unit tests: _expand_with_control_dimensions
#    Tensor layout: [num_tokens, num_heads * head_dim] (flat)
# ════════════════════════════════════════════════════════════════════


def _vllm_stub(num_heads=2, num_kv_heads=2, head_dim=32, control_dims=1):
    """Minimal namespace for vLLM _expand_with_control_dimensions."""
    return types.SimpleNamespace(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        control_dims=control_dims,
        expanded_head_dim=head_dim + control_dims,
    )


def _vllm_expand(stub, q, k, v, membership, suppression):
    return GraniteLoRAEmbeddedAttention._expand_with_control_dimensions(
        stub, q, k, v, membership, suppression
    )


def _vllm_qkv(stub, num_tokens):
    """Create flat vLLM-layout Q/K/V tensors."""
    q = torch.randn(num_tokens, stub.num_heads * stub.head_dim)
    k = torch.randn(num_tokens, stub.num_kv_heads * stub.head_dim)
    v = torch.randn(num_tokens, stub.num_kv_heads * stub.head_dim)
    return q, k, v


class TestExpandControlDimensions:
    """Direct tests of _expand_with_control_dimensions (vLLM flat tensor layout).

    Input shape:  [num_tokens, num_heads * head_dim]
    Output shape: [num_tokens, num_heads * expanded_head_dim]
    """

    _HEAD_DIM = 32
    _CTRL_DIMS = 1

    def test_control_token_q_hide_zero_at_position_zero(self):
        """Core fix: control token at pos 0 must not activate Q-side hiding."""
        stub = _vllm_stub(control_dims=self._CTRL_DIMS)
        membership = torch.ones(1, 1, dtype=torch.bool)
        suppression = torch.ones(1, 1, dtype=torch.bool)
        q, k, v = _vllm_qkv(stub, num_tokens=1)

        q_exp, _, _ = _vllm_expand(stub, q, k, v, membership, suppression)

        q_reshaped = q_exp.view(1, stub.num_heads, stub.expanded_head_dim)
        q_ctrl = q_reshaped[0, :, self._HEAD_DIM:]
        assert q_ctrl.eq(0).all(), f"Control token at pos 0: q_control must be 0, got {q_ctrl}"

    def test_adapter_generated_tokens_q_hide_one(self):
        """Adapter-generated tokens (non-members) keep q_control=1."""
        stub = _vllm_stub(control_dims=self._CTRL_DIMS)
        num_tokens = 5
        membership = torch.zeros(num_tokens, 1, dtype=torch.bool)
        membership[0, 0] = True   # control token at pos 0
        suppression = torch.ones(num_tokens, 1, dtype=torch.bool)
        q, k, v = _vllm_qkv(stub, num_tokens)

        q_exp, _, _ = _vllm_expand(stub, q, k, v, membership, suppression)
        q_reshaped = q_exp.view(num_tokens, stub.num_heads, stub.expanded_head_dim)

        assert q_reshaped[0, :, self._HEAD_DIM:].eq(0).all(), "Control token: q_control must be 0"
        for pos in range(1, num_tokens):
            assert q_reshaped[pos, :, self._HEAD_DIM:].eq(1).all(), (
                f"Adapter-generated token at pos {pos}: q_control must be 1"
            )

    def test_k_side_finfo_min_for_control_token(self):
        """K-side branding is unaffected by the fix."""
        stub = _vllm_stub(control_dims=self._CTRL_DIMS)
        membership = torch.ones(1, 1, dtype=torch.bool)
        q, k, v = _vllm_qkv(stub, num_tokens=1)

        _, k_exp, _ = _vllm_expand(stub, q, k, v, membership, None)

        k_reshaped = k_exp.view(1, stub.num_kv_heads, stub.expanded_head_dim)
        k_ctrl = k_reshaped[0, :, self._HEAD_DIM:]
        expected_min = torch.finfo(k.dtype).min
        torch.testing.assert_close(k_ctrl, torch.full_like(k_ctrl, expected_min))

    def test_both_none_leaves_all_control_dims_zero(self):
        """With both tensors None, all control dims remain zero."""
        stub = _vllm_stub(control_dims=2)
        q, k, v = _vllm_qkv(stub, num_tokens=4)
        q_exp, k_exp, v_exp = _vllm_expand(stub, q, k, v, None, None)

        exp_head = stub.expanded_head_dim
        assert q_exp.view(4, stub.num_heads, exp_head)[:, :, self._HEAD_DIM:].eq(0).all()
        assert k_exp.view(4, stub.num_kv_heads, exp_head)[:, :, self._HEAD_DIM:].eq(0).all()
        assert v_exp.view(4, stub.num_kv_heads, exp_head)[:, :, self._HEAD_DIM:].eq(0).all()

    def test_original_dimensions_preserved(self):
        """Original head dims are unchanged; only control dims appended."""
        stub = _vllm_stub(control_dims=2)
        q, k, v = _vllm_qkv(stub, num_tokens=3)
        q_exp, k_exp, v_exp = _vllm_expand(stub, q, k, v, None, None)

        exp_head = stub.expanded_head_dim
        torch.testing.assert_close(
            q_exp.view(3, stub.num_heads, exp_head)[:, :, :self._HEAD_DIM],
            q.view(3, stub.num_heads, stub.head_dim),
        )


# ════════════════════════════════════════════════════════════════════
# 2. Shared SDPA cases
# ════════════════════════════════════════════════════════════════════


class TestSDPANaN(SDPANaNCases):
    pass


# ════════════════════════════════════════════════════════════════════
# 3. Shared model finiteness cases — vLLM backend
# ════════════════════════════════════════════════════════════════════


class TestModelFiniteness(ModelFinitenessCases):
    def _assert_no_nan(self, switch_type, ctrl_pos, seq_len, seed):
        is_finite = _run_vllm_forward_is_finite(ctrl_pos, seq_len, seed)
        assert is_finite, (
            f"[vLLM] ctrl_pos={ctrl_pos}: logits contain NaN/Inf"
        )


# ════════════════════════════════════════════════════════════════════
# 4. Mutation test — proves TestModelFiniteness is sensitive to the fix
# ════════════════════════════════════════════════════════════════════


def _buggy_expand(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_group_membership,
    query_group_suppression,
) -> tuple:
    """Pre-fix version: omits q_hide *= (1 - membership), causing NaN at ctrl_pos=0."""
    num_tokens = q.size(0)
    device = q.device
    dtype = q.dtype

    q = q.view(num_tokens, self.num_heads, self.head_dim)
    k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
    v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

    q_control = torch.zeros(num_tokens, self.num_heads, self.control_dims, device=device, dtype=dtype)
    k_control = torch.zeros(num_tokens, self.num_kv_heads, self.control_dims, device=device, dtype=dtype)
    v_control = torch.zeros(num_tokens, self.num_kv_heads, self.control_dims, device=device, dtype=dtype)

    if token_group_membership is not None:
        num_groups = token_group_membership.shape[-1]
        hiding_constant = torch.finfo(dtype).min
        k_control[:, :, :num_groups] = (
            token_group_membership.unsqueeze(1)
            .expand(-1, self.num_kv_heads, -1)
            .to(dtype) * hiding_constant
        )

    if query_group_suppression is not None:
        num_groups = query_group_suppression.shape[-1]
        q_hide = query_group_suppression.to(dtype)
        # BUG: missing `q_hide *= (1 - token_group_membership)` — control token
        # at position 0 gets q_ctrl=1, causing softmax([-inf]) = NaN.
        q_control[:, :, :num_groups] = q_hide.unsqueeze(1).expand(-1, self.num_heads, -1)

    q = torch.cat([q, q_control], dim=-1)
    k = torch.cat([k, k_control], dim=-1)
    v = torch.cat([v, v_control], dim=-1)

    q = q.view(num_tokens, self.num_heads * self.expanded_head_dim)
    k = k.view(num_tokens, self.num_kv_heads * self.expanded_head_dim)
    v = v.view(num_tokens, self.num_kv_heads * self.expanded_head_dim)

    return q, k, v


class TestFixSensitivity:
    """Mutation test: revert the fix and confirm NaN is produced.

    Patches _expand_with_control_dimensions with the pre-fix (buggy) version.
    If _run_vllm_forward_is_finite still returns True, the model-level tests
    are not actually sensitive to the fix and must be reconsidered.
    """

    def test_buggy_expand_produces_nan_at_ctrl_pos_zero(self):
        """Without the fix, ctrl_pos=0 must produce non-finite logits in vLLM."""
        from granite_switch.vllm.core.decoder import GraniteLoRAEmbeddedAttention
        from unittest.mock import patch

        with patch.object(
            GraniteLoRAEmbeddedAttention,
            "_expand_with_control_dimensions",
            _buggy_expand,
        ):
            is_finite = _run_vllm_forward_is_finite(ctrl_pos=0, seq_len=8, seed=99)

        assert not is_finite, (
            "[vLLM] Expected NaN with buggy expand at ctrl_pos=0, "
            "but output was finite — test is not sensitive to the fix"
        )
