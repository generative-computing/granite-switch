# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill coverage for the vLLM classifier verdict exit.

``enable_chunked_prefill`` is on by default in serving. The prompt is laid out as
the chat template renders it, with the classifier control token as the very last
token, so it always lands in the final chunk. The same prompt served at
``max_num_batched_tokens`` in {unchunked, 256, 101, 48} must emit the same label
token.

Not asserted: logprob or hidden-state equality across budgets. Chunked and unchunked
serving differ by bf16 reduction order in the attention/KV path (a magnitude that is
hardware- and version-specific), so a numeric band would be flaky.

Each budget runs in its own subprocess so only one vLLM engine is resident.

Requires GPU + vLLM.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vLLM classifier serving test requires a GPU"
)

WORKER = os.path.join(
    os.path.dirname(__file__), "_classifier_chunked_prefill_worker.py"
)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(WORKER)))
BASE_MODEL = os.environ.get("CLS_TEST_BASE", "ibm-granite/granite-4.1-3b")

# One budget per distinct final-chunk shape: ``None`` disables chunked prefill
# (single pass); 256 leaves a wide final chunk; 101 a narrow one; 48 splits into many
# passes. Budgets differing only in pass count exercise no additional path.
BUDGETS = [None, 256, 101, 48]


def _env():
    env = dict(os.environ)
    env["VLLM_USE_V1"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTHONPATH", _REPO)
    return env


def _compose(tmp_path_factory, mode, tag):
    """Compose a classifier checkpoint in a CPU subprocess, before any CUDA init."""
    outdir = str(tmp_path_factory.mktemp(tag))
    proc = subprocess.run(
        [sys.executable, WORKER, mode],
        env={**_env(), "CLS_BASE": BASE_MODEL, "CLS_OUT": outdir},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [l for l in (proc.stdout or "").splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        pytest.fail(
            f"{mode} failed:\nstdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}",
            pytrace=False,
        )
    meta = json.loads(lines[-1])
    if "error" in meta:
        pytest.fail(f"{mode} error: {meta['error'][:2000]}", pytrace=False)
    result = meta["result"]
    result["pids_path"] = os.path.join(outdir, "pids.json")
    json.dump(result["prompt_ids"], open(result["pids_path"], "w"))
    result["outdir"] = outdir
    return result


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    """Multi-label classifier over a varied prompt."""
    return _compose(tmp_path_factory, "compose-multilabel", "cls_ckpt")


def _run_worker(composed, argv, out_name, label, timeout):
    """Run one worker invocation in its own subprocess; return the parsed result."""
    out_path = os.path.join(composed["outdir"], out_name)
    proc = subprocess.run(
        [sys.executable, WORKER, *argv, out_path],
        env={
            **_env(),
            "CLS_CKPT": composed["ckpt"],
            "CLS_PIDS": composed["pids_path"],
        },
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if not os.path.exists(out_path):
        pytest.fail(
            f"{label}: worker wrote no result\n"
            f"stdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-1500:]}",
            pytrace=False,
        )
    res = json.load(open(out_path))
    if "fatal" in res:
        pytest.fail(f"{label}: worker crashed:\n{res['fatal'][:2000]}", pytrace=False)
    return res


def _run(composed, budget):
    """Serve the prompt at one chunk budget (``None`` = chunked prefill disabled)."""
    tag = "none" if budget is None else str(budget)
    return _run_worker(
        composed, ["run", tag], f"run_{tag}.json", f"budget={budget}", 1200
    )


def test_classifier_token_stable_across_chunk_budgets(composed):
    """Emitted verdict token identical across chunk budgets."""
    label_ids = set(composed["label_token_ids"])
    assert len(composed["prompt_ids"]) > min(b for b in BUDGETS if b), (
        "the prompt must be longer than the smallest budget so it chunks"
    )

    results = {b: _run(composed, b) for b in BUDGETS}

    ref = results[None]
    assert ref["token_id"] in label_ids, (
        f"unchunked verdict {ref['token_id']} is not a label token {label_ids}; "
        f"the classifier exit did not fire"
    )
    for b, r in results.items():
        assert r["token_id"] == ref["token_id"], (
            f"budget={b} emitted {r['token_id']} ({r['text']!r}) != unchunked "
            f"{ref['token_id']} ({ref['text']!r})"
        )
