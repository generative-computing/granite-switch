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
