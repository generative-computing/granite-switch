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

_SHORT, _LONG = 16, 64


def _random_prompt(seq_len, high):
    torch.manual_seed(42)
    return torch.randint(0, high, (seq_len,)).tolist()


def _adapter_prompt(seq_len):
    return make_active_adapter_input(1, seq_len, seed=42)[0].tolist()


@pytest.fixture(scope="module", params=_MODEL_NAMES)
def scored(request, tmp_path_factory):
    """Every logprob these tests compare, for one variant, from three engines.

    The suite compares three models: the upstream reference ``upstream``, the
    adapter-free switch ``switch`` (num_adapters=0), and the zero-LoRA switch
    ``zero`` (adapter tiers live, deltas zero). It used to boot a fresh pair of
    engines per test -- ten boots per variant, five of them the identical upstream
    model -- and an engine boot (torch.compile + CUDA-graph capture) costs far more
    than scoring a prompt. So each model is built and booted ONCE here and scores
    every prompt its tests need.

    The checkpoints, seeds and prompts are the ones the per-test runners built,
    and each prompt still gets its own ``generate`` call (run_vllm_logprobs_many),
    so the numbers are the ones the tests compared before.

    The outer test_granite4_mini.py runs this file's classes for a variant in a
    single pytest process, so the module scope is what shares the engines.
    """
    import gc

    from granite_switch.vllm import register as register_granite_switch
    from tests.shared.vllm_equivalence import (
        run_vllm_logprobs_many,
        save_switch_model,
        save_upstream_model,
        save_zero_adapter_model,
    )

    register_granite_switch()
    model_name = request.param
    cfg = GRANITE4_MINI[model_name]
    vocab_size = cfg["vocab_size"]
    tmpdir = tmp_path_factory.mktemp(f"granite4_mini_{model_name}")

    upstream_dir, upstream_sd = save_upstream_model(cfg, 0, tmpdir)
    switch_dir = save_switch_model(upstream_sd, cfg, tmpdir)
    zero_dir = save_zero_adapter_model(upstream_sd, cfg, tmpdir)
    del upstream_sd
    gc.collect()

    # name -> (prompt, the models that score it)
    prompts = {
        # TestGranite4FamilyEquivalence: random ids over the full vocabulary
        "family_short": (_random_prompt(_SHORT, vocab_size), ("upstream", "switch")),
        "family_long": (_random_prompt(_LONG, vocab_size), ("upstream", "switch")),
        # TestZeroAdapterNoHiding: random ids clear of the control tokens (101+)
        "no_ctrl": (_random_prompt(_SHORT, 100), ("upstream", "zero")),
        # TestZeroAdapterEquivalence: control tokens that trigger switching
        "ctrl_short": (_adapter_prompt(_SHORT), ("upstream", "zero")),
        "ctrl_long": (_adapter_prompt(_LONG), ("upstream", "zero")),
    }
    dirs = {"upstream": upstream_dir, "switch": switch_dir, "zero": zero_dir}
    out = {}
    for model, model_dir in dirs.items():
        names = [n for n, (_, models) in prompts.items() if model in models]
        tensors = run_vllm_logprobs_many(
            model_dir,
            [prompts[n][0] for n in names],
            vocab_size,
            tmpdir=tmpdir,
        )
        for name, tensor in zip(names, tensors):
            out[(model, name)] = tensor
    return {"model_name": model_name, "cfg": cfg, "logprobs": out}


def _pair(scored, prompt, switch_model):
    lp = scored["logprobs"]
    return lp[("upstream", prompt)], lp[(switch_model, prompt)]


class TestGranite4FamilyEquivalence:
    """Integration equivalence: vLLM GraniteSwitch sans LoRA == upstream."""

    def test_logits_short(self, scored):
        model_name = scored["model_name"]
        layer_types = scored["cfg"].get("layer_types", [])
        upstream, switch = _pair(scored, "family_short", "switch")

        # ULP-equivalent, not bit-exact: the inert switch runs its projections
        # through the fused SWITCH kernel, whose reduction order differs from
        # vLLM's native linear (see get_tolerances). tol is never None now.
        tol = get_tolerances(layer_types, long_sequence=False)
        assert_close(
            switch,
            upstream,
            atol=tol[0],
            rtol=tol[1],
            msg=f"{model_name}: short sequence logprobs diverge",
        )

    def test_logits_long(self, scored):
        model_name = scored["model_name"]
        layer_types = scored["cfg"].get("layer_types", [])
        upstream, switch = _pair(scored, "family_long", "switch")

        # ULP-equivalent, not bit-exact (see get_tolerances / test_logits_short).
        tol = get_tolerances(layer_types, long_sequence=True)
        assert_close(
            switch,
            upstream,
            atol=tol[0],
            rtol=tol[1],
            msg=f"{model_name}: long sequence logprobs diverge",
        )


class TestZeroAdapterNoHiding:
    """Zero LoRA weights, adapter infrastructure active."""

    def test_no_control_tokens(self, scored):
        model_name = scored["model_name"]
        layer_types = scored["cfg"].get("layer_types", [])
        upstream, switch = _pair(scored, "no_ctrl", "zero")

        # Skinned + no control token = inert switch. ULP-equivalent to upstream,
        # not bit-exact: the fused SWITCH kernel reduces in a different order than
        # the GraniteMoeHybrid reference's native linear on the same transferred
        # weights (~1 bf16 ULP; see get_tolerances).
        tol = get_tolerances(layer_types, has_kv_hidden=False)
        assert_close(
            switch,
            upstream,
            atol=tol[0],
            rtol=tol[1],
            msg=f"{model_name}: inert switch (no control token) logprobs diverge",
        )


class TestZeroAdapterEquivalence:
    """Integration equivalence with active switching and zero LoRA weights."""

    @pytest.mark.parametrize(
        "prompt, seq_len, long_sequence",
        [("ctrl_short", _SHORT, False), ("ctrl_long", _LONG, True)],
        ids=["short", "long"],
    )
    def test_logits(self, scored, prompt, seq_len, long_sequence):
        model_name = scored["model_name"]
        layer_types = scored["cfg"].get("layer_types", [])
        upstream, switch = _pair(scored, prompt, "zero")

        input_ids = make_active_adapter_input(1, seq_len, seed=42)
        visible = get_visible_mask(input_ids)[0, :-1]

        tol = get_tolerances(
            layer_types, long_sequence=long_sequence, has_kv_hidden=True
        )
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
                msg=(
                    f"{model_name}: {'long' if long_sequence else 'short'} "
                    "sequence logprobs diverge (zero-adapter)"
                ),
            )
