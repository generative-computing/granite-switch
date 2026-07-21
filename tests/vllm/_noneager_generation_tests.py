# SPDX-License-Identifier: Apache-2.0
"""Non-eager generation tests (inner file — run by test_noneager_generation.py).

Smoke test: GraniteSwitch generation through vLLM's serving pipeline.
Requires CUDA GPU and vLLM installed.
"""

import os

import pytest
import torch

from tests.shared.generation_models import (
    HYBRID_CFG,
    basic_overrides,
    save_switch_model,
)

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


def _generate(model_dir, enforce_eager=False):
    import gc

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    llm = LLM(
        model=model_dir,
        enforce_eager=enforce_eager,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=64,
        gpu_memory_utilization=0.3,
    )

    input_ids = list(range(10, 30))
    prompt = TokensPrompt(prompt_token_ids=input_ids)
    params = SamplingParams(max_tokens=16, temperature=0.8)

    outputs = llm.generate(prompt, sampling_params=params)
    generated = outputs[0].outputs[0].token_ids

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return generated


class TestNoSwitch:
    def test_generates_tokens(self, tmp_path):
        import gc

        from granite_switch.vllm import register as register_granite_switch
        from tests.shared.granite4_equivalence import GRANITE4_MINI
        from tests.shared.vllm_equivalence import (
            save_switch_model,
            save_upstream_model,
        )

        register_granite_switch()

        cfg = GRANITE4_MINI["4.0-350m"]
        upstream_dir, upstream_sd = save_upstream_model(
            cfg,
            seed=0,
            tmpdir=tmp_path,
        )
        switch_dir = save_switch_model(upstream_sd, cfg, tmpdir=tmp_path)
        del upstream_sd
        gc.collect()

        generated = _generate(switch_dir, enforce_eager=False)
        assert len(generated) == 16, (
            f"Expected 16 generated tokens, got {len(generated)}"
        )


class TestSingleSwitch:
    def test_generates_tokens(self, tmp_path):
        model_dir = save_switch_model(
            HYBRID_CFG,
            basic_overrides(HYBRID_CFG),
            tmpdir=tmp_path,
        )
        generated = _generate(model_dir, enforce_eager=False)
        assert len(generated) == 16, (
            f"Expected 16 generated tokens, got {len(generated)}"
        )
