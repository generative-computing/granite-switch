# SPDX-License-Identifier: Apache-2.0
"""GPU test classes from test_granite4_mini.py (inner file — run in subprocess).

Equivalence tests via vllm.LLM for miniaturized Granite 4 configs.
Requires CUDA GPU and vLLM installed.
"""

import pytest
import torch

from tests.shared.granite4_equivalence import (
    GRANITE4_MINI,
    assert_close,
    get_tolerances,
    get_visible_mask,
    make_active_adapter_input,
)

_UPSTREAM_EAGER_CONFIGS = {"4.0-h-350m"}


def _eager_kwargs_if_needed(model_name):
    if model_name in _UPSTREAM_EAGER_CONFIGS:
        return {"enforce_eager": True}
    return {}


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

_MODEL_NAMES = sorted(GRANITE4_MINI.keys())


class TestGranite4FamilyEquivalence:
    """Integration equivalence: vLLM GraniteSwitch sans LoRA == upstream."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_logits_short(self, model_name, tmp_path):
        from tests.shared.vllm_equivalence import run_equivalence_integration

        cfg = GRANITE4_MINI[model_name]
        layer_types = cfg.get("layer_types", [])

        upstream, switch = run_equivalence_integration(
            cfg,
            seq_len=16,
            tmpdir=tmp_path,
            **_eager_kwargs_if_needed(model_name),
        )

        tol = get_tolerances(layer_types, long_sequence=False)
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
                msg=f"{model_name}: short sequence logprobs diverge",
            )

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_logits_long(self, model_name, tmp_path):
        from tests.shared.vllm_equivalence import run_equivalence_integration

        cfg = GRANITE4_MINI[model_name]
        layer_types = cfg.get("layer_types", [])

        upstream, switch = run_equivalence_integration(
            cfg,
            seq_len=64,
            tmpdir=tmp_path,
            **_eager_kwargs_if_needed(model_name),
        )

        tol = get_tolerances(layer_types, long_sequence=True)
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
                msg=f"{model_name}: long sequence logprobs diverge",
            )


class TestZeroAdapterNoHiding:
    """Zero LoRA weights, adapter infrastructure active."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_no_control_tokens(self, model_name, tmp_path):
        from tests.shared.vllm_equivalence import run_zero_adapter_no_hiding_equivalence

        cfg = GRANITE4_MINI[model_name]
        upstream, switch = run_zero_adapter_no_hiding_equivalence(
            cfg,
            use_control_tokens=False,
            seq_len=16,
            tmpdir=tmp_path,
            **_eager_kwargs_if_needed(model_name),
        )

        # SingleSwitch is bit-exact (no counting head, no position perturbation)
        torch.testing.assert_close(
            switch,
            upstream,
            atol=0.0,
            rtol=0.0,
            msg=f"{model_name}: should be bit-exact with no control tokens",
        )


class TestZeroAdapterEquivalence:
    """Integration equivalence with active switching and zero LoRA weights."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_logits_short(self, model_name, tmp_path):
        from tests.shared.vllm_equivalence import run_zero_adapter_equivalence

        cfg = GRANITE4_MINI[model_name]
        layer_types = cfg.get("layer_types", [])
        seq_len = 16

        upstream, switch = run_zero_adapter_equivalence(
            cfg,
            seq_len=seq_len,
            tmpdir=tmp_path,
            **_eager_kwargs_if_needed(model_name),
        )

        input_ids = make_active_adapter_input(1, seq_len, seed=42)
        visible = get_visible_mask(input_ids)[0, :-1]

        tol = get_tolerances(layer_types, long_sequence=False, has_kv_hidden=True)
        if tol is None:
            torch.testing.assert_close(
                switch[visible],
                upstream[visible],
                atol=0.0,
                rtol=0.0,
                msg=f"{model_name}: logprobs should be bit-exact",
            )
        else:
            assert_close(
                switch[visible],
                upstream[visible],
                atol=tol[0],
                rtol=tol[1],
                msg=f"{model_name}: short sequence logprobs diverge (zero-adapter)",
            )

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_logits_long(self, model_name, tmp_path):
        from tests.shared.vllm_equivalence import run_zero_adapter_equivalence

        cfg = GRANITE4_MINI[model_name]
        layer_types = cfg.get("layer_types", [])
        seq_len = 64

        upstream, switch = run_zero_adapter_equivalence(
            cfg,
            seq_len=seq_len,
            tmpdir=tmp_path,
            **_eager_kwargs_if_needed(model_name),
        )

        input_ids = make_active_adapter_input(1, seq_len, seed=42)
        visible = get_visible_mask(input_ids)[0, :-1]

        tol = get_tolerances(layer_types, long_sequence=True, has_kv_hidden=True)
        if tol is None:
            torch.testing.assert_close(
                switch[visible],
                upstream[visible],
                atol=0.0,
                rtol=0.0,
                msg=f"{model_name}: logprobs should be bit-exact",
            )
        else:
            assert_close(
                switch[visible],
                upstream[visible],
                atol=tol[0],
                rtol=tol[1],
                msg=f"{model_name}: long sequence logprobs diverge (zero-adapter)",
            )
