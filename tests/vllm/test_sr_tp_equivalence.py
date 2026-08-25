# SPDX-License-Identifier: Apache-2.0
"""SR tensor-parallel equivalence — TP=1 vs TP=N on a composed Shadow-Residual checkpoint.

Loads the SAME composed SR checkpoint under vLLM at TP=1 and at TP=SR_TP_SIZE (each in its
own subprocess — vLLM does not free GPU memory on ``del``) and checks that the decision-position
top-k next-token distribution agrees. It is the identical model and identical input, so the only
admissible difference is all-reduce reduction order; a structural TP bug (wrong sharding, missing
reduce, bad head geometry) would shift the distribution far beyond that.

Gate (per swept k, mean over prompts) — JSD is the primary agreement metric; Jaccard is a
loose guard:
    mean(JSD in bits)         <= SR_TP_JSD_THRESH   (default 0.02)   AND
    mean(1 - Jaccard(top-k))  <= SR_TP_JACC_THRESH  (default 0.30)
Top-k *set membership* is noisy for near-tie decisions: mid-rank tokens flip in/out of the
top-k across TP reduction orders even when the distribution is identical (vLLM TP is not
bit-deterministic run-to-run), so JSD — which compares probability mass — is the real gate.
Jaccard only catches gross top-k scrambling; cf. test_tp_integration.py, which tolerates
5/20 top-k overlap loss plus a top-1 logprob tolerance.

gpu-marked; skipped without vLLM, >=2 GPUs, or a readable SR_COMPOSED_DIR. Model-agnostic: point
SR_COMPOSED_DIR at a granite-4.1 or granite-4.2 SR checkpoint — the worker auto-detects the 4.2
chat template. See tests/unit/test_sr_doubled_q_tp_mapping.py for the CPU head-mapping invariant.

Env:
  SR_COMPOSED_DIR (required)  composed SR checkpoint dir
  SR_TP_SIZE      (default 2) the N in TP=1-vs-TP=N (needs that many GPUs)
  SR_TP_TOPK      (default 200) logprobs captured per decision position
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.shared.logit_metrics import jaccard as _jaccard
from tests.shared.logit_metrics import jsd_bits as _jsd_bits
from tests.shared.logit_metrics import topk_ids as _topk_ids

try:
    import torch

    _NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    _NUM_GPUS = 0
_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM"),
    pytest.mark.skipif(_NUM_GPUS < 2, reason="requires at least 2 GPUs"),
]

COMPOSED = os.environ.get("SR_COMPOSED_DIR", "")
TP_SIZE = int(os.environ.get("SR_TP_SIZE", "2"))
WORKER = Path(__file__).parent / "_sr_tp_equivalence_worker.py"
K_SWEEP = [1, 5, 10, 20, 50, 100]
JACC_THRESH = float(os.environ.get("SR_TP_JACC_THRESH", "0.30"))
JSD_THRESH = float(os.environ.get("SR_TP_JSD_THRESH", "0.02"))
TIMEOUT = 1500


def _skip_checks():
    if not COMPOSED or not os.path.exists(os.path.join(COMPOSED, "config.json")):
        pytest.skip(f"SR_COMPOSED_DIR not set/readable ({COMPOSED!r})")
    if _NUM_GPUS < TP_SIZE:
        pytest.skip(f"SR_TP_SIZE={TP_SIZE} needs {TP_SIZE} GPUs, have {_NUM_GPUS}")


def _run_worker(tp, out_path):
    result = subprocess.run(
        [sys.executable, str(WORKER), str(tp), out_path],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, (
        f"SR TP worker (tp={tp}) failed (exit {result.returncode}).\n"
        f"STDERR (last 1500):\n{result.stderr[-1500:]}"
    )


def test_sr_tp_equivalence(tmp_path):
    """TP=1 vs TP=SR_TP_SIZE decision-logit equivalence on the composed SR checkpoint."""
    _skip_checks()
    ref = str(tmp_path / "tp1.json")
    cmp = str(tmp_path / f"tp{TP_SIZE}.json")
    _run_worker(1, ref)
    _run_worker(TP_SIZE, cmp)

    R = json.load(open(ref))["decisions"]
    C = json.load(open(cmp))["decisions"]
    assert R and len(R) == len(C), f"decision count mismatch: {len(R)} vs {len(C)}"
    n = len(R)

    print(f"\nSR TP EQUIVALENCE  tp=1 vs tp={TP_SIZE}  n={n} prompts")
    print(f"  {'k':>5}{'1-Jaccard':>14}{'JSD(bits)':>14}")
    failures = []
    for k in K_SWEEP:
        jd = (
            sum(
                1.0 - _jaccard(_topk_ids(R[i], k), _topk_ids(C[i], k)) for i in range(n)
            )
            / n
        )
        js = (
            sum(
                _jsd_bits(
                    R[i], C[i], list(set(_topk_ids(R[i], k)) | set(_topk_ids(C[i], k)))
                )
                for i in range(n)
            )
            / n
        )
        print(f"  {k:>5}{jd:>14.6f}{js:>14.6f}")
        if jd > JACC_THRESH:
            failures.append(f"k={k}: mean(1-Jaccard)={jd:.4f} > {JACC_THRESH}")
        if js > JSD_THRESH:
            failures.append(f"k={k}: mean(JSD)={js:.6f} > {JSD_THRESH}")

    assert not failures, (
        f"SR TP=1 vs TP={TP_SIZE} diverged beyond bf16 reduction-order noise:\n  "
        + "\n  ".join(failures)
    )
