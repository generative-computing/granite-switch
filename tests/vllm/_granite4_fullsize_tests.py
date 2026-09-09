# SPDX-License-Identifier: Apache-2.0
"""GPU test classes from test_granite4_fullsize.py (inner file — run in subprocess).

Full-size Granite 4 equivalence tests via vllm.LLM.
Requires CUDA GPU and vLLM installed.

Runs eager. These are the bit-exact tests, and bit-exactness is a property of
the module tree, not of whatever inductor decided to fuse in each of the two
graphs — see the comment on ``enforce_eager`` below.
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
            # Compare the two module trees, not two compilations of them.
            # torch.compile/inductor pick fusions and kernels per graph, and
            # the switch graph is not the upstream graph, so at production
            # dimensions the compiled paths can land a bf16 last bit
            # differently while the eager math is identical. That is what
            # broke 4.0-micro under vLLM 0.20 and not 0.19 (see the note in
            # get_tolerances). enforce_eager removes the compiler, the
            # CUDA-graph capture and vLLM's on-disk compile cache from the
            # comparison, which restores a meaningful bit-exact gate.
            # The compiled path is still covered, by the tests that are
            # tolerant of a flipped last bit:
            # tests/composer/test_skinning_equivalence.py (real weights) and
            # tests/vllm/test_generation_equivalence.py (distribution).
            enforce_eager=True,
        )

        tol = get_tolerances(layer_types)
        atol, rtol = (0.0, 0.0) if tol is None else tol
        assert_close(
            switch,
            upstream,
            atol=atol,
            rtol=rtol,
            msg=f"{model_name}: full-size logprobs diverge",
        )

        # Print the margin on a pass, not only into the failure message. A
        # bit-exact gate that has quietly started riding at one ULP is a
        # regression in progress, and this is the only place it shows.
        delta = (switch - upstream).abs()
        finite = delta[delta.isfinite()]
        print(
            f"{model_name}: {finite.numel()} finite logprob entries, "
            f"max |delta| = {finite.max().item():.3e} "
            f"(atol={atol:.1e} rtol={rtol:.1e})"
        )
