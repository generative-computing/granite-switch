# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill coverage for the vLLM classifier verdict exit.

Every other vLLM classifier path is HF-only and single-pass, yet
``enable_chunked_prefill`` is on by default in serving. Two tests here:

The prompt places the classifier control token immediately after the last
content token (marker at position ``n-1``), and the verdict is read at
``marker - 1`` (the last content token, ``n-2``) inside
``GraniteSwitchModel._classifier_read_points``.

1. ``test_classifier_token_stable_across_chunk_budgets`` — the same prompt served at
   ``max_num_batched_tokens`` in {unchunked, 512, 256, 128, 101, 64, 48, n_prompt-1}
   emits the same verdict token, and each budget's consumed verdict is read from
   ``marker - 1``. ``_classifier_read_points`` locates the marker per pass and reads
   its predecessor; when a chunk boundary splits the marker from its predecessor, a
   single-token cross-pass stash supplies the previous pass's last hidden row. The
   ``n_prompt - 1`` budget is the arm that forces that split (marker alone in the
   final chunk) — the only budget here that exercises the look-back at all.

2. ``test_wrong_read_corrupts_position_and_often_verdict`` — the inverse: forcing the
   read to wrong positions lands at a wrong global position (asserted) and usually
   flips the label (reported; see that test).

Not asserted: logprob or hidden-state equality across budgets. Chunked and unchunked
serving differ by bf16 reduction order in the attention/KV path (a magnitude that is
hardware- and version-specific), so a numeric band would be flaky. Token stability
still catches a wrong read position or a weight/logit corruption.

Each budget runs in its own subprocess so only one vLLM engine is resident, with the
engine core in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the read-index hook
lands on the running model.

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

# One budget per distinct final-chunk shape, which is what the read point depends on:
# ``None`` disables chunked prefill (single pass); 256 leaves a wide final chunk; 48
# splits into many passes with a narrow one; 101 leaves exactly two tokens, so the
# marker and its read point are the last pair in the chunk. The lone-marker budget
# (final chunk of one) is derived per prompt and appended by the test below. Budgets
# differing only in pass count exercise no additional path.
BUDGETS = [None, 256, 101, 48]


def _lone_marker_budget(n_prompt: int) -> int:
    """Budget that puts the marker ALONE in the final chunk.

    The marker is the prompt's last token, so it is the first (and only) token of
    the final chunk exactly when ``n_prompt % budget == 1`` -- i.e. budget
    ``n_prompt - 1``, which splits the prompt into [0, n-2] then [n-1]. That is
    the one layout where the read point (``marker - 1``) lands in the *previous*
    pass, exercising the cross-pass look-back in
    ``GraniteSwitchModel._classifier_read_points``. Computed from ``n_prompt`` so a
    tokenizer change that shifts it cannot silently stop testing this case.
    """
    return n_prompt - 1


def _env():
    env = dict(os.environ)
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine core
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
    result["pids_trailing_path"] = os.path.join(outdir, "pids_trailing.json")
    json.dump(result["prompt_ids_trailing"], open(result["pids_trailing_path"], "w"))
    result["outdir"] = outdir
    return result


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    """Multi-label classifier over a varied prompt: the argmax is position-sensitive,
    which both tests need (stable-across-budgets and the forced-read sweep)."""
    return _compose(tmp_path_factory, "compose-multilabel", "cls_ckpt")


def _run_worker(composed, argv, out_name, label, timeout, pids_key="pids_path"):
    """Run one worker invocation in its own subprocess; return the parsed result."""
    out_path = os.path.join(composed["outdir"], out_name)
    proc = subprocess.run(
        [sys.executable, WORKER, *argv, out_path],
        env={
            **_env(),
            "CLS_CKPT": composed["ckpt"],
            "CLS_PIDS": composed[pids_key],
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


def _run(composed, budget, pids_key="pids_path"):
    """Serve the prompt at one chunk budget (``None`` = chunked prefill disabled)."""
    tag = "none" if budget is None else str(budget)
    suffix = "" if pids_key == "pids_path" else "_trailing"
    return _run_worker(
        composed,
        ["run", tag],
        f"run_{tag}{suffix}.json",
        f"budget={budget}{suffix}",
        1200,
        pids_key=pids_key,
    )


def _run_fault_sweep(composed):
    """Run the full-prompt forced-read sweep in one subprocess."""
    return _run_worker(
        composed, ["fault-sweep"], "fault_sweep.json", "fault-sweep", 1800
    )


def test_classifier_token_stable_across_chunk_budgets(composed):
    """Emitted verdict token identical across chunk budgets; every budget's
    consumed verdict is read from ``marker - 1`` (the last content token).

    The prompt is ``[*body, control_id]`` (marker at ``n_prompt - 1``), so the
    correct read point is ``marker - 1 == n_prompt - 2``.

    Budgets include ``n_prompt - 1``, the one budget that leaves the marker ALONE
    in the final chunk (``n_prompt % budget == 1``). There the read point sits in
    the *previous* pass, so this arm is what exercises the cross-pass look-back;
    every other budget keeps marker and read point in the same chunk."""
    n_prompt = len(composed["prompt_ids"])
    read_point = n_prompt - 2  # marker is at n_prompt - 1; read point = marker - 1
    label_ids = set(composed["label_token_ids"])

    # The lone-marker budget (n_prompt % budget == 1) is the only layout that
    # splits the marker from its read point across a chunk boundary, so it is the
    # one arm that exercises the cross-pass look-back. Derived from n_prompt.
    lone = _lone_marker_budget(n_prompt)
    budgets = [*BUDGETS, lone]
    assert n_prompt % lone == 1, (
        f"budget {lone} does not leave the marker alone in the final chunk for "
        f"n_prompt={n_prompt} (n_prompt % budget == {n_prompt % lone}, want 1)"
    )

    results = {b: _run(composed, b) for b in budgets}

    ref = results[None]
    assert ref["token_id"] in label_ids, (
        f"unchunked verdict {ref['token_id']} is not a label token {label_ids}; "
        f"the classifier exit did not fire"
    )

    # Consumed verdict reads the last content token (marker - 1), for every budget.
    for b, r in results.items():
        assert r["global_read_idx"] == read_point, (
            f"budget={b}: consumed verdict read from global index "
            f"{r['global_read_idx']}, expected {read_point} (marker-1) "
            f"(passes={r['n_passes']})"
        )

    # Verdict token stable across budgets.
    for b, r in results.items():
        assert r["token_id"] == ref["token_id"], (
            f"budget={b} emitted {r['token_id']} ({r['text']!r}) != unchunked "
            f"{ref['token_id']} ({ref['text']!r})"
        )

    # The smallest budget actually chunked.
    assert results[48]["n_passes"] > 1, (
        f"budget=48 ran in {results[48]['n_passes']} pass(es); expected chunking."
    )


def test_wrong_read_corrupts_position_and_often_verdict(composed):
    """A wrong read position corrupts the verdict.

    Forces the verdict to read positions strided across the prompt and records the
    emitted token at each, against a correct run reading ``marker - 1`` (the last
    content token).
    Asserts a wrong read always lands at a wrong global position (the correct run
    reads ``n_prompt - 2``; every forced read reports a different position).

    The label-flip rate is reported, not asserted: a head can map two hidden states to
    the same argmax label, so some wrong reads produce the correct label by chance, and
    a mismatch threshold would be flaky. The multi-label head keeps the argmax
    position-sensitive so most wrong reads do flip the label in practice.
    """
    n_prompt = len(composed["prompt_ids"])
    read_point = n_prompt - 2  # marker at n_prompt - 1; correct read = marker - 1
    label_ids = set(composed["label_token_ids"])

    sweep = _run_fault_sweep(composed)
    correct = sweep["correct"]
    faulted = sweep["faulted"]

    assert correct["pos"] == read_point, (
        f"correct run read global pos {correct['pos']}, expected last content token "
        f"{read_point} (marker - 1)"
    )
    assert correct["token_id"] in label_ids, (
        f"correct verdict {correct['token_id']} is not a label token {label_ids}; "
        f"the classifier exit did not fire"
    )
    # Expect a spread of distinct positions, every one inside 0..n-2 and none at
    # the correct read point.
    assert len(faulted) >= 3, (
        f"fault sweep produced only {len(faulted)} forced reads; expected a spread"
    )
    positions_set = {f["pos"] for f in faulted}
    assert all(0 <= p <= n_prompt - 2 for p in positions_set), (
        f"forced reads landed outside the prompt pool 0..{n_prompt - 2}: "
        f"{sorted(positions_set)}"
    )
    for f in faulted:
        assert f["pos"] != read_point, (
            f"forced read reported global pos {f['pos']} == correct read point "
            f"{read_point}; the forced index did not move the read"
        )

    # Report the label-flip rate (not asserted; see docstring).
    n_mismatch = sum(1 for f in faulted if f["token_id"] != correct["token_id"])
    hist = {}
    for f in faulted:
        hist[f["text"]] = hist.get(f["text"], 0) + 1
    positions = sorted(f["pos"] for f in faulted)
    frac = n_mismatch / len(faulted) if faulted else 0.0
    print(
        f"\nfault-sweep: {n_mismatch}/{len(faulted)} wrong-read positions "
        f"(global {positions[0]}..{positions[-1]}) produced a label != correct "
        f"{correct['text']!r} ({frac:.0%}); labels={hist}"
    )


def test_verdict_carries_when_marker_is_not_in_the_sampling_chunk(composed):
    """A marker in an earlier chunk still produces its verdict.

    A rendered chat prompt continues past the classifier marker (turn close plus
    generation prompt), so the marker is interior. With a budget that closes a chunk
    at the marker, the pass vLLM samples on contains no marker at all -- the verdict
    was resolved one pass earlier and has to be carried forward.

    The prompt is ``[*body, marker, *tail]``, so the marker sits at ``n_tail`` tokens
    from the end and its read point is ``marker - 1``.
    """
    n_prompt = len(composed["prompt_ids_trailing"])
    n_tail = composed["n_tail"]
    marker_pos = n_prompt - n_tail - 1
    read_point = marker_pos - 1
    label_ids = set(composed["label_token_ids"])
    assert n_tail > 0, "the trailing variant must continue past the marker"

    # Budget that ends a chunk exactly at the marker: the next pass starts at the
    # first tail token and holds no marker.
    budget = marker_pos + 1
    res = _run(composed, budget, pids_key="pids_trailing_path")

    assert res["n_passes"] > 1, (
        f"budget={budget} ran in {res['n_passes']} pass(es); expected the prompt to "
        f"split after the marker"
    )
    assert res["token_id"] in label_ids, (
        f"verdict {res['token_id']} is not a label token {label_ids}; the verdict "
        f"resolved before the sampling chunk was not carried forward"
    )
    assert res["global_read_idx"] == read_point, (
        f"carried verdict was read from global index {res['global_read_idx']}, "
        f"expected {read_point} (marker at {marker_pos}, {n_tail} tail tokens)"
    )

    # Same prompt unchunked must agree.
    ref = _run(composed, None, pids_key="pids_trailing_path")
    assert res["token_id"] == ref["token_id"], (
        f"chunked verdict {res['token_id']} != unchunked {ref['token_id']} on the "
        f"trailing-token prompt"
    )
