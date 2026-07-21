# SPDX-License-Identifier: Apache-2.0
"""Upstream equivalence tests (inner file — run by test_upstream_equivalence.py).

Verify vLLM GraniteSwitch (sans LoRA) produces equivalent logits to upstream.
Requires CUDA GPU and vLLM installed.
"""

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm import LLM  # noqa: F401

        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from tests.shared.granite4_equivalence import assert_close
    from tests.shared.vllm_equivalence import run_equivalence_integration


_COMMON_CONFIG = dict(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    shared_intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_local_experts=0,
    num_experts_per_tok=0,
    layer_types=["attention", "attention", "attention", "attention"],
    hidden_act="silu",
    max_position_embeddings=512,
    attention_bias=False,
    mlp_bias=False,
    embedding_multiplier=1.0,
    residual_multiplier=1.0,
    attention_multiplier=0.125,
    logits_scaling=1.0,
)


class TestAttentionOnlyNoMoE:
    def test_logits_match(self, tmp_path):
        cfg = {
            **_COMMON_CONFIG,
            "layer_types": ["attention", "attention", "attention", "attention"],
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "shared_intermediate_size": _COMMON_CONFIG["intermediate_size"],
        }
        upstream, switch = run_equivalence_integration(
            cfg,
            seq_len=16,
            tmpdir=tmp_path,
        )
        assert_close(
            switch,
            upstream,
            atol=1e-2,
            rtol=1e-2,
            msg="Attention-only (no MoE): vLLM logprobs diverge",
        )
