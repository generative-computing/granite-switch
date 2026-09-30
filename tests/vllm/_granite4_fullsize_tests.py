# SPDX-License-Identifier: Apache-2.0
"""GPU test classes from test_granite4_fullsize.py (inner file — run in subprocess).

Full-size Granite 4 equivalence tests via vllm.LLM.
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

from tests.shared.granite4_equivalence import (
    GRANITE4_FULLSIZE,
    assert_close,
    get_tolerances,
)

_MODEL_NAMES = sorted(GRANITE4_FULLSIZE.keys())
_SEQ_LEN = 8


class TestGranite4FullSizeEquivalence:
    """Full-size integration equivalence via vllm.LLM."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_logits_match(self, model_name, tmp_path):
        from tests.shared.vllm_equivalence import run_equivalence_integration

        cfg = GRANITE4_FULLSIZE[model_name]
        layer_types = cfg.get("layer_types", [])

        upstream, switch = run_equivalence_integration(
            cfg,
            seq_len=_SEQ_LEN,
            tmpdir=tmp_path,
            max_model_len=64,
            gpu_memory_utilization=0.4,
            # enforce_eager is REQUIRED, not an optimisation.  Without it this was
            # the only vLLM equivalence test that left torch.compile, CUDA-graph
            # capture and Triton autotuning live, and [4.0-micro] went red roughly
            # 1 run in 7 with a bit-reproducible 130708 elements over tol,
            # worst diff=2.9306e-03 at |expected|=11.279.
            #
            # Measured 2026-09-29, both on one A100-80GB node at this exact config:
            #   * Upstream against ITSELF through two separate engines, and switch
            #     against upstream, are 0.0000e+00 over 702464 elements for all
            #     three full-size configs.  There is no bf16 noise floor here to
            #     hide behind and nothing wrong with the 1e-5 in get_tolerances.
            #   * The red reproduces with this file run ALONE on a cleared compile
            #     cache, so it is not contamination from a preceding test; and it
            #     does not reproduce on 13 of 14 otherwise-identical runs.
            # Two discrete, bit-reproducible outcomes (2.9306e-03 / 2.9297e-03)
            # from an unpinned compiler is kernel selection, which autotune makes
            # by timing and then caches.  2.9e-3 absolute at |expected|=11.28 is
            # 2.6e-4 relative, under one bf16 ulp (0.031) -- a reduction-order
            # difference, not a logic error.  An equivalence test must not measure
            # the compiler as well as the model.  _granite4_mini_tests.py already
            # pins eager for the config that needed it.
            enforce_eager=True,
            # Set, not defaulted, so the two sides are configured alike.  vLLM
            # 0.26 resolves `default_prefix_caching = is_prefix_caching_supported
            # and not is_hybrid`, and upstream GraniteMoeHybridForCausalLM
            # declares IsHybrid while GraniteSwitchForCausalLM deliberately does
            # not (it must not -- see granite_switch_model.py, or vLLM sizes the
            # KV cache for zero attention layers).  So upstream silently
            # resolved False and switch True.  Measured 2026-09-29 that
            # asymmetry is numerically INERT -- upstream with the flag forced on
            # is bit-identical to upstream with it off -- so this is not the fix
            # for anything.  But a comparison should have one variable in it.
            enable_prefix_caching=False,
        )

        tol = get_tolerances(layer_types)
        if tol is None:
            torch.testing.assert_close(
                switch,
                upstream,
                atol=0.0,
                rtol=0.0,
                msg=f"{model_name}: logprobs should be bit-exact",
            )
        else:
            assert_close(
                switch,
                upstream,
                atol=tol[0],
                rtol=tol[1],
                msg=f"{model_name}: full-size logprobs diverge",
            )
