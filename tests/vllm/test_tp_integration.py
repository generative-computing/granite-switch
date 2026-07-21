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
  1. granite-4.0-micro with real adapters from ibm-granite/granitelib-rag-r1.0

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
    assert overlap >= TOPK_OVERLAP_MIN, msg
    assert top1_diff <= TOP1_LOGPROB_ATOL, msg


def _build_and_compare(work_dir, build_args, label, intrinsic_name=None):
    """Build a model, generate with TP=1 and TP=2, assert distributions match.

    Not an exact-text check — see comment above TOPK_OVERLAP_MIN for why.
    """
    model_dir = os.path.join(work_dir, "switch-model")
    _run_step(f"build ({label})", *build_args, "--output-dir", model_dir)

    tp1_out = os.path.join(work_dir, "tp1.json")
    tp2_out = os.path.join(work_dir, "tp2.json")

    run_extra = []
    if intrinsic_name:
        run_extra = ["--intrinsic-name", intrinsic_name]

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
        f"generate TP=2 ({label})",
        "run",
        "--model-path",
        model_dir,
        "--tp-size",
        "2",
        "--output-path",
        tp2_out,
        *run_extra,
    )

    with open(tp1_out) as f:
        records_tp1 = json.load(f)
    with open(tp2_out) as f:
        records_tp2 = json.load(f)

    assert len(records_tp1) == len(records_tp2), (
        f"[{label}] prompt count differs: tp1={len(records_tp1)} tp2={len(records_tp2)}"
    )

    for i, (r1, r2) in enumerate(zip(records_tp1, records_tp2)):
        _compare_topk(label, i, r1, r2)


class TestTPRealAdapters:
    """TP=1 vs TP=2 with real adapters from granite-lib-rag (granite-4.0-micro).

    Includes a chat-template prompt that activates the answerability adapter
    via intrinsic_name, testing that adapter control tokens are handled
    correctly under tensor parallelism.
    """

    def test_tp_logprobs_agree(self, tmp_path):
        _build_and_compare(
            str(tmp_path),
            build_args=[
                "build-compose",
                "--base-model",
                "ibm-granite/granite-4.0-micro",
                "--adapter-repos",
                "ibm-granite/granitelib-rag-r1.0",
            ],
            label="granite-4.0-micro-rag",
            intrinsic_name="answerability",
        )
