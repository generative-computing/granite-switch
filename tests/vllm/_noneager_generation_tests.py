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
    single_overrides,
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


def _generate_batch(model_dir, prompts, max_tokens=8, enforce_eager=False):
    """Submit several prompts of different lengths in one ``generate`` call.

    Returns the generated token ids per prompt, in request order.
    """
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

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=list(p)) for p in prompts],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    generated = [out.outputs[0].token_ids for out in outputs]

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return generated


# Ragged prompts with the control token at a different offset in each, plus one
# row with no control token at all.  A batch that mixes adapters with base is the
# only shape where the expert bank gathers tokens belonging to different adapter
# contexts into a single grouped matmul.
_MOE_PROMPTS = (
    [10, 11, 250, 12, 13, 14, 15, 16, 17, 18],
    [30, 31, 32, 251, 33],
    [40, 41, 42, 43, 44, 45, 46],
)
_MOE_MAX_TOKENS = 8


def _sparse_moe_model(tmp_path):
    """A pure sparse MoE switch checkpoint: experts, no dense ``shared_mlp``."""
    from tests.shared.granite4_equivalence import GRANITEMOE_MINI

    cfg = GRANITEMOE_MINI["moe-20b"]
    return save_switch_model(cfg, single_overrides(cfg), tmpdir=tmp_path)


def _assert_batch_generated(generated, vocab_size):
    """Liveness only -- deliberately not a numeric comparison.

    Top-k expert dispatch reduces through ``index_add`` with duplicate indices,
    i.e. GPU atomics, so an MoE forward is not reproducible even against itself
    and the generated ids are not a stable function of the batch composition.
    Numeric agreement between backends is gated by per-position top-k JSD in
    ``test_generation_equivalence.py``; what is asserted here is that a ragged
    multi-request batch decodes to completion at all, which is the failure mode
    a zero-width or mis-shaped expert path produces.
    """
    assert len(generated) == len(_MOE_PROMPTS)
    for i, ids in enumerate(generated):
        assert len(ids) == _MOE_MAX_TOKENS, (
            f"prompt {i}: expected {_MOE_MAX_TOKENS} tokens, got {len(ids)}"
        )
        assert all(0 <= t < vocab_size for t in ids), f"prompt {i}: id out of range"


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


class TestMultiSwitch:
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


class TestSparseMoEBatchCudaGraph:
    """Batched generation over a pure sparse MoE base, with graph capture on.

    ``shared_intermediate_size == 0`` is the newest MLP shape and the only one
    with no dense path to fall back on, so this is the case where a mis-built
    expert bank has nothing masking it.  Capture is left enabled on purpose: the
    captured shapes are sized from token counts, and the MoE path has never been
    traced.
    """

    def test_generates_tokens(self, tmp_path):
        from tests.shared.granite4_equivalence import GRANITEMOE_MINI

        model_dir = _sparse_moe_model(tmp_path)
        generated = _generate_batch(
            model_dir,
            _MOE_PROMPTS,
            max_tokens=_MOE_MAX_TOKENS,
            enforce_eager=False,
        )
        _assert_batch_generated(generated, GRANITEMOE_MINI["moe-20b"]["vocab_size"])


class TestSparseMoEBatchEager:
    """The same batch with ``enforce_eager=True``.

    Run as its own case so that a capture-only failure is distinguishable from a
    broken forward -- otherwise the two are one red test with one traceback.
    """

    def test_generates_tokens(self, tmp_path):
        from tests.shared.granite4_equivalence import GRANITEMOE_MINI

        model_dir = _sparse_moe_model(tmp_path)
        generated = _generate_batch(
            model_dir,
            _MOE_PROMPTS,
            max_tokens=_MOE_MAX_TOKENS,
            enforce_eager=True,
        )
        _assert_batch_generated(generated, GRANITEMOE_MINI["moe-20b"]["vocab_size"])
