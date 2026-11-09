# SPDX-License-Identifier: Apache-2.0
"""TP integration tests — require 2+ GPUs with vLLM installed.

Verifies the TP plumbing in SwitchedLoRALinear is structurally correct by
loading a Granite Switch model at TP=1 and TP=2 and checking that the
first-generated-token logprob distribution agrees within bf16 numerical
tolerance. We do NOT assert byte-equality of generated text; see the comment
above TOPK_OVERLAP_MIN for why that is not a well-defined invariant.

Each step (build, run@TP=1, run@TP=2) runs in a separate subprocess to avoid
CUDA fork issues — follows the same pattern as test_generation_equivalence.py.

Test cases:
  1. granite-4.0-micro with adapters from ibm-granite/granitelib-rag-r1.0: a fake
     on every PR run, the published model in the release tier
  2. a synthetic pure sparse MoE (``granitemoe``) base, plain LoRA and Shadow
     Residual — the only committed coverage of expert sharding
  3. the sensitivity guard: the fake case again with a sharding bug injected at
     TP=2, which the comparison MUST reject. Without it, "TP=1 agrees with TP=2"
     could be true merely because the sharded layers do not reach the output.

Skip automatically if fewer than 2 GPUs or vLLM is not installed.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.shared import model_sources
from tests.shared.fake_bases import LIVE_LAYERS, TP_DIMS

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

pytestmark = [
    pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM"),
    pytest.mark.skipif(_NUM_GPUS < 2, reason="requires at least 2 GPUs"),
]

WORKER = Path(__file__).parent / "_tp_integration_worker.py"
TIMEOUT = 1500


def _run_step(step_name, *cmd_args, timeout=TIMEOUT):
    """Run a single worker step as a subprocess and assert success."""
    cmd = [sys.executable, str(WORKER), *cmd_args]
    print(f"\n{'=' * 60}")
    print(f"  Step: {step_name}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, (
        f"TP integration step '{step_name}' failed (exit {result.returncode}).\n"
        f"STDOUT (last 2000):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 1000):\n{result.stderr[-1000:]}"
    )


# Tolerances for TP=1 vs TP=2 equivalence check.
#
# We compare the first-generated-token logprob distribution, not the greedy
# text, because byte-equality of greedy decoding is not an invariant in bf16:
# TP>1 reorders all-reduce summations, which flips low-bit rounding, which
# (after 40 layers of compounding) can flip near-tie argmaxes on the rare
# prompt that lands on such a tie. See docs/TENSOR_PARALLEL_FIX.md.
#
# What we DO want to catch: structural bugs — wrong sharding, missing
# all-reduce, bias applied on wrong rank, etc. — which would cause
# order-of-magnitude drifts, not 1 ULP.
#
# - TOPK_OVERLAP_MIN: how many of the top-K tokens must be the same set.
#   For K=20 a structural bug would scramble the list; fp noise only shuffles
#   a handful of near-ties.
# - TOP1_LOGPROB_ATOL: absolute tolerance on the top-1 token's logprob.
#   Observed bf16 noise at final_norm_out was ~4 in hidden-state units, which
#   after logits_scaling=8 and a softmax can produce ~1.0 logprob drift on
#   near-ties. 2.0 is comfortably above noise and well below any real bug.
TOPK = 20
TOPK_OVERLAP_MIN = 15
TOP1_LOGPROB_ATOL = 0.5


def _compare_topk(label, prompt_idx, rec1, rec2):
    """Assert TP=1 and TP=2 first-token logprob distributions agree within
    numerical tolerance. Raises AssertionError with a diagnostic message."""
    topk1 = rec1["first_token_topk"]
    topk2 = rec2["first_token_topk"]
    assert topk1 is not None and topk2 is not None, (
        f"[{label}] Prompt {prompt_idx}: missing first_token_topk in output"
    )

    ids1 = [tid for tid, _ in topk1[:TOPK]]
    ids2 = [tid for tid, _ in topk2[:TOPK]]
    overlap = len(set(ids1) & set(ids2))

    # Top-1 logprobs should agree within tolerance (we don't require the
    # top-1 *token* to match — a bf16 tie flip is acceptable; a logprob
    # that differs by more than TOP1_LOGPROB_ATOL is not).
    top1_lp1 = topk1[0][1]
    top1_lp2 = topk2[0][1]
    top1_diff = abs(top1_lp1 - top1_lp2)

    msg = (
        f"[{label}] Prompt {prompt_idx} TP divergence beyond bf16 noise:\n"
        f"  TP=1 text: {rec1['text']!r}\n"
        f"  TP=2 text: {rec2['text']!r}\n"
        f"  Top-{TOPK} token-id overlap: {overlap}/{TOPK} "
        f"(require >= {TOPK_OVERLAP_MIN})\n"
        f"  TP=1 top-5 (id, logprob): {topk1[:5]}\n"
        f"  TP=2 top-5 (id, logprob): {topk2[:5]}\n"
        f"  Top-1 logprob diff: {top1_diff:.4f} "
        f"(require <= {TOP1_LOGPROB_ATOL})"
    )
    # Printed on a pass too, not only into the failure message: how far a run sits
    # from the gate is the only thing that says whether these tolerances are
    # measured or merely unbroken, and it is what a reviewer without a GPU has to
    # go on. Requires ``-s``.
    print(
        f"  [{label}] prompt {prompt_idx}: overlap {overlap}/{TOPK} "
        f"(min {TOPK_OVERLAP_MIN})   top-1 logprob diff {top1_diff:.4f} "
        f"(max {TOP1_LOGPROB_ATOL})"
    )
    assert overlap >= TOPK_OVERLAP_MIN, msg
    assert top1_diff <= TOP1_LOGPROB_ATOL, msg


def _build_and_compare(
    work_dir, build_args, label, intrinsic_name=None, token_prompts=False
):
    """Build a model, generate with TP=1 and TP=2, assert distributions match.

    Not an exact-text check — see comment above TOPK_OVERLAP_MIN for why.
    """
    records_tp1, records_tp2 = _build_and_run(
        work_dir, build_args, label, intrinsic_name, token_prompts
    )
    for i, (r1, r2) in enumerate(zip(records_tp1, records_tp2)):
        _compare_topk(label, i, r1, r2)


def _build_and_run(
    work_dir, build_args, label, intrinsic_name=None, token_prompts=False, fault=None
):
    """Build a model, generate at TP=1 and TP=2, return both runs' records.

    ``fault`` (see _tp_integration_worker.FAULTS) is injected into the TP=2 run
    only; it acts on one rank's shard, so TP=1 is unaffected either way.
    """
    model_dir = os.path.join(work_dir, "switch-model")
    _run_step(f"build ({label})", *build_args, "--output-dir", model_dir)

    tp1_out = os.path.join(work_dir, "tp1.json")
    tp2_out = os.path.join(work_dir, "tp2.json")

    run_extra = []
    if intrinsic_name:
        run_extra = ["--intrinsic-name", intrinsic_name]
    if token_prompts:
        run_extra = [*run_extra, "--token-prompts"]

    _run_step(
        f"generate TP=1 ({label})",
        "run",
        "--model-path",
        model_dir,
        "--tp-size",
        "1",
        "--output-path",
        tp1_out,
        *run_extra,
    )
    _run_step(
        f"generate TP=2 ({label}{f', fault={fault}' if fault else ''})",
        "run",
        "--model-path",
        model_dir,
        "--tp-size",
        "2",
        "--output-path",
        tp2_out,
        *run_extra,
        *(["--fault", fault] if fault else []),
    )

    with open(tp1_out) as f:
        records_tp1 = json.load(f)
    with open(tp2_out) as f:
        records_tp2 = json.load(f)

    assert len(records_tp1) == len(records_tp2), (
        f"[{label}] prompt count differs: tp1={len(records_tp1)} tp2={len(records_tp2)}"
    )
    return records_tp1, records_tp2


class TestTPRealAdapters:
    """TP=1 vs TP=2 with adapters from granite-lib-rag (granite-4.0-micro).

    Exercises the coded MultiSwitch engine (the only engine), which owns TWO
    extra attention layers with their own KV cache slots (counting + memory) --
    the obvious thing tensor-parallel sharding can get wrong. Includes a
    chat-template prompt that activates the answerability adapter via
    intrinsic_name, so adapter control-token placement is exercised under
    sharding too. TP=1 and TP=2 logprobs must agree.

    (A former TestTPMultiSwitch built the identical checkpoint: it was the
    MultiSwitch twin of a SingleSwitch case, and SingleSwitch is gone.)
    """

    @pytest.mark.parametrize("source", model_sources.sources())
    def test_tp_logprobs_agree(self, tmp_path, source):
        _build_and_compare(
            str(tmp_path),
            build_args=_rag_build_args(source),
            label=f"granite-4.0-micro-rag-{source}",
            intrinsic_name="answerability",
        )


_BASE_NAME = "granite-4.0-micro"
_RAG_LIBRARY = "ibm-granite/granitelib-rag-r1.0"

#: The fake for these tests (tests/shared/fake_bases.py). LIVE_LAYERS because on
#: the default REALISTIC preset the decoder layers barely reach the output, so a
#: sharding bug inside them would leave TP=2 agreeing with TP=1 anyway. TP_DIMS
#: (8 query / 4 KV heads) so each rank holds several KV heads, which is where GQA
#: sharding bugs show; the default geometry leaves each rank one.
_FAKE_KW = {"preset": LIVE_LAYERS, "dims": TP_DIMS}


def _rag_build_args(source):
    """Compose args for 4.0-micro + the RAG library, per source.

    ``fake`` (every PR run) is a tiny base and library built from captured copies
    of the real ones (tests/shared/model_sources.py), with :data:`_FAKE_KW`.
    ``real`` runs in the release tier.
    """
    kw = _FAKE_KW if source == "fake" else {}
    return [
        "build-compose",
        "--base-model",
        model_sources.base_model(source, _BASE_NAME, **kw),
        "--adapter-repos",
        model_sources.adapter_library(source, _RAG_LIBRARY, _BASE_NAME, **kw),
    ]


def _comparison_fails(label, records_tp1, records_tp2):
    """Why the TP comparison rejects these runs, or None if it accepts them."""
    failures = []
    for i, (r1, r2) in enumerate(zip(records_tp1, records_tp2)):
        try:
            _compare_topk(label, i, r1, r2)
        except AssertionError as e:
            failures.append(f"prompt {i}: {str(e).splitlines()[0]}")
    return failures or None


class TestTPDetectsShardingBugs:
    """The TP comparison must REJECT a TP=2 run with a sharding bug in it.

    Agreement between TP=1 and TP=2 only means something if a broken shard would
    break it. Each case injects one realistic bug into the TP=2 run of the fake
    case above (_tp_integration_worker.FAULTS; it acts on one rank's shard, so
    TP=1 is unaffected) and requires the same _compare_topk gate that the
    passing case uses to reject at least one prompt. If a faulted run still
    passes, the TP tests are vacuous for that class of bug, and this fails with
    that message.

    Covers both LoRA (adapter path) and base-weight sharding; the base fault
    bites with no adapter active, so it also proves the sharded layers reach the
    output at all.
    """

    @pytest.mark.parametrize(
        "fault",
        [
            "lora_b_wrong_shard",
            "lora_a_wrong_shard",
            "skip_row_reduce",
            "base_o_proj_shard",
        ],
    )
    def test_injected_fault_is_detected(self, tmp_path, fault):
        label = f"granite-4.0-micro-rag-fake-fault-{fault}"
        records_tp1, records_tp2 = _build_and_run(
            str(tmp_path),
            build_args=_rag_build_args("fake"),
            label=label,
            intrinsic_name="answerability",
            fault=fault,
        )
        failures = _comparison_fails(label, records_tp1, records_tp2)
        print(f"\n  [{label}] rejected prompts: {failures}")
        assert failures, (
            f"TP comparison is VACUOUS for {fault!r}: a TP=2 run with this sharding "
            f"bug injected still agreed with TP=1 on every prompt, so "
            f"test_tp_logprobs_agree cannot catch it. Make the sharded path reach "
            f"the output (fake preset/geometry) or tighten the gate."
        )


class TestTPGraniteMoe:
    """TP=1 vs TP=2 on a pure sparse MoE base (``granitemoe``), both adaptations.

    Every other arm in this file is a *dense* base, so the expert bank has never
    been sharded by any committed test.  Two things are at stake and neither is
    reachable from the dense arms:

    * the HF-stacked -> ``FusedMoE`` remap in ``vllm/decoder/interface.py``
      (``input_linear`` ``[E, 2I, H]`` -> ``w13_weight``, ``output_linear``
      ``[E, H, I]`` -> ``w2_weight``) must land the right bytes when ``FusedMoE``
      splits the expert bank across ranks;
    * Shadow Residual hands the MoE a ``[2M, H]`` doubled token dim, so the
      ``sr`` arm is the only committed coverage of an expert path under both the
      doubling and sharding at once.

    The sharding arithmetic itself is upstream's — ``_load_expert`` only calls
    ``param.weight_loader(...)`` and never writes a tensor — so what this pins is
    our remap and our plumbing, not ``FusedMoE``'s math.

    Scope note: this is a TP-agreement test, not an adapter-activation test.  That
    the control token fires at all is gated on CPU by
    ``tests/composer/test_granitemoe_compose_e2e.py`` and structurally by
    ``tests/vllm/test_moe_support.py``.  What is guarded here is that the two runs
    are not agreeing *vacuously*: ``build-granitemoe`` fails the build if any
    ``lora_B`` (or, for ``sr``, ``cross_stream.lora_B``) is all zero, which is
    exactly what ``save_switch_model`` would have produced.

    ``tests/vllm/test_sr_tp_equivalence.py`` remains the gate on a *real* composed
    checkpoint; it is env-var gated and never runs in CI, which is why this one
    exists.
    """

    @pytest.mark.parametrize("variant", ["lora", "sr"])
    def test_tp_logprobs_agree_granitemoe(self, tmp_path, variant):
        _build_and_compare(
            str(tmp_path),
            build_args=["build-granitemoe", "--variant", variant],
            label=f"granitemoe-{variant}",
            token_prompts=True,
        )
