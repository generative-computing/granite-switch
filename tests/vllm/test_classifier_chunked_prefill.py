# SPDX-License-Identifier: Apache-2.0
"""GPU coverage for the vLLM classifier verdict exit: FULL capture/replay,
chunked prefill, intermediate-layer reads, and two slots in one checkpoint.

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


def _compose(tmp_path_factory, argv, tag):
    """Compose a classifier checkpoint in a CPU subprocess, before any CUDA init."""
    outdir = str(tmp_path_factory.mktemp(tag))
    proc = subprocess.run(
        [sys.executable, WORKER, *argv],
        env={**_env(), "CLS_BASE": BASE_MODEL, "CLS_OUT": outdir},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [l for l in (proc.stdout or "").splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        pytest.fail(
            f"{argv[0]} failed:\nstdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}",
            pytrace=False,
        )
    meta = json.loads(lines[-1])
    if "error" in meta:
        pytest.fail(f"{argv[0]} error: {meta['error'][:2000]}", pytrace=False)
    result = meta["result"]
    result["outdir"] = outdir
    return result


def _write_pids(result):
    path = os.path.join(result["outdir"], "pids.json")
    json.dump(result["prompt_ids"], open(path, "w"))
    return path


def _run_worker(outdir, argv, env, timeout=1800):
    """Run one worker invocation in its own subprocess; return the parsed result."""
    out_path = os.path.join(outdir, "_".join(argv) + ".json")
    proc = subprocess.run(
        [sys.executable, WORKER, *argv, out_path],
        env={**_env(), **env},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    label = " ".join(argv)
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


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    """Multi-label classifier over a varied prompt."""
    result = _compose(tmp_path_factory, ["compose-multilabel"], "cls_ckpt")
    result["pids_path"] = _write_pids(result)
    return result


def _composed_env(composed, **extra):
    return {"CLS_CKPT": composed["ckpt"], "CLS_PIDS": composed["pids_path"], **extra}


def test_full_cudagraph_mode_does_not_crash_and_matches_eager(composed):
    """Exercise FULL startup and changed inputs, including mixed requests.

    The original path copied marker IDs to CUDA during capture and crashed.
    Compare the selected tokens with the same prompts served eagerly.
    """
    label_ids = set(composed["label_token_ids"])
    env = _composed_env(
        composed,
        CLS_CONTROL_ID=str(composed["control_id"]),
        CLS_LABEL_IDS=json.dumps(composed["label_token_ids"]),
    )
    full = _run_worker(composed["outdir"], ["run-full-cudagraph"], env)
    eager = _run_worker(composed["outdir"], ["run-eager-reference"], env)

    assert len(full["results"]) == len(eager["results"]) == 6
    for i, (f, e) in enumerate(zip(full["results"], eager["results"])):
        assert f["is_label"], (
            f"call {i}: FULL-cudagraph verdict {f['token_id']} ({f['text']!r}) "
            f"is not a label token {label_ids}; the classifier exit did not "
            f"fire under real CUDA-graph capture/replay"
        )
        assert f["token_id"] == e["token_id"], (
            f"call {i}: FULL-cudagraph emitted {f['token_id']} ({f['text']!r}) "
            f"!= eager reference {e['token_id']} ({e['text']!r})"
        )

    # Ordinary generation may legitimately choose a label word. Check that
    # non-label tokens remain possible, rather than banning those words.
    assert full["plain_result"]["has_non_label_support"]
    assert full["plain_result"]["token_id"] == eager["plain_result"]["token_id"]
    mixed = full["mixed_results"]
    assert len(mixed) == len(eager["mixed_results"]) == 3
    for i, (actual, reference) in enumerate(zip(mixed, eager["mixed_results"])):
        assert actual["token_id"] == reference["token_id"], f"mixed row {i}"
        assert actual["has_non_label_support"] == (i == 1)
        if i != 1:
            assert actual["is_label"]


def test_classifier_token_stable_across_chunk_budgets(composed):
    """Emitted verdict token identical across chunk budgets."""
    label_ids = set(composed["label_token_ids"])
    assert len(composed["prompt_ids"]) > min(b for b in BUDGETS if b), (
        "the prompt must be longer than the smallest budget so it chunks"
    )

    results = {
        b: _run_worker(
            composed["outdir"],
            ["run", "none" if b is None else str(b)],
            _composed_env(composed),
            timeout=1200,
        )
        for b in BUDGETS
    }

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


# Score-gap tolerance against the fp32 HF reference; this comparison also
# includes HF/vLLM backbone differences, not only classifier arithmetic.
SCORE_DIFF_ATOL = 0.3


def _check_served_scores(
    name, served, ref_scores, ref_label_id, label_token_ids, min_label_margin
):
    """Print one served verdict beside its CPU reference; return failure messages.

    Reference values are raw head scores; served values are logprobs, so only
    label-to-label differences are comparable. The winning-label check runs
    only when the reference margin is at least ``min_label_margin`` (``None``
    always runs it); below that a near-tie makes it inconclusive.
    """
    ref_sorted = sorted(ref_scores, reverse=True)
    margin = ref_sorted[0] - ref_sorted[1]
    label_logprobs = served["label_logprobs"]
    print(f"\n[{name}] label token ids={label_token_ids}")
    print(f"  reference raw scores (fp32): {ref_scores}")
    print(f"  reference label={ref_label_id}, decision margin={margin:.4f}")
    print(f"  served label logprobs:       {label_logprobs}")
    print(f"  served selected token={served['token_id']} ({served['text']!r})")

    failures = []
    if any(lp is None for lp in label_logprobs):
        failures.append(
            f"{name}: not every label token appeared in the top-"
            f"{len(label_token_ids)} logprobs ({label_logprobs})"
        )
    else:
        for j in range(1, len(label_token_ids)):
            served_diff = label_logprobs[j] - label_logprobs[0]
            ref_diff = ref_scores[j] - ref_scores[0]
            ok = served_diff == pytest.approx(ref_diff, abs=SCORE_DIFF_ATOL)
            print(
                f"  label[{j}]-label[0]: served logprob diff={served_diff:.4f}, "
                f"reference score diff={ref_diff:.4f} -> {'ok' if ok else 'MISMATCH'}"
            )
            if not ok:
                failures.append(
                    f"{name}: served logprob diff label[{j}]-label[0] = "
                    f"{served_diff:.4f} != reference score diff {ref_diff:.4f} "
                    f"+/- {SCORE_DIFF_ATOL}"
                )

    if min_label_margin is not None and margin < min_label_margin:
        print(
            f"  winning-label check INCONCLUSIVE: reference margin "
            f"{margin:.4f} < {min_label_margin} (near-tie)"
        )
    elif served["token_id"] != ref_label_id:
        failures.append(
            f"{name}: served token {served['token_id']} ({served['text']!r}) "
            f"!= reference label {ref_label_id} (margin={margin:.4f})"
        )
    return failures


# Allowance for fp32 log-softmax subtraction only (fp32 spacing is ~6e-5 at
# |logit| < 512); not a numerical-equivalence tolerance.
LOGPROB_DIFF_ATOL = 1e-3


def _check_transport(name, served, label_token_ids):
    """Packed scores -> rewritten label logits -> returned logprobs."""
    packed = served["packed_scores"]
    rewritten = served["rewritten_label_logits"]
    label_logprobs = served["label_logprobs"]
    print(f"\n[{name}] transport")
    print(f"  packed head scores (raw):     {packed}")
    print(f"  rewritten label logits (raw): {rewritten}")
    print(f"  served label logprobs:        {label_logprobs}")

    failures = []
    if rewritten != packed:
        failures.append(
            f"{name}: rewritten label logits {rewritten} != packed scores {packed}"
        )
    if any(lp is None for lp in label_logprobs):
        failures.append(
            f"{name}: not every label token appeared in the top-"
            f"{len(label_token_ids)} logprobs ({label_logprobs})"
        )
        return failures
    for j in range(1, len(label_token_ids)):
        lp_diff = label_logprobs[j] - label_logprobs[0]
        logit_diff = rewritten[j] - rewritten[0]
        if lp_diff != pytest.approx(logit_diff, abs=LOGPROB_DIFF_ATOL):
            failures.append(
                f"{name}: logprob diff label[{j}]-label[0] = {lp_diff} != "
                f"rewritten logit diff {logit_diff} +/- {LOGPROB_DIFF_ATOL}"
            )
    return failures


def _check_final_layer_arithmetic(name, served):
    """Packed scores against the loaded head applied to the actual activation."""
    weight = torch.tensor(served["loaded_weight"])
    bias = torch.tensor(served["loaded_bias"])
    hidden = torch.tensor(served["served_hidden"])
    recomputed = weight @ hidden + bias
    rounded = recomputed.to(torch.bfloat16).float().tolist()
    print(f"\n[{name}] arithmetic on the served activation")
    print(f"  loaded head in fp32:          {recomputed.tolist()}")
    print(f"  ... rounded to bf16:          {rounded}")
    print(f"  packed head scores (raw):     {served['packed_scores']}")
    if rounded != served["packed_scores"]:
        return [
            f"{name}: packed scores {served['packed_scores']} != loaded head "
            f"on served activation, rounded to bf16, {rounded}"
        ]
    return []


def _print_final_layer_activation(name, served, reference_hidden, reference_scores):
    """Record the HF/vLLM final-activation difference; no acceptance threshold."""
    weight = torch.tensor(served["loaded_weight"])
    bias = torch.tensor(served["loaded_bias"])
    hf = torch.tensor(reference_hidden, dtype=torch.float32)
    vllm = torch.tensor(served["served_hidden"])

    def gap(scores):
        return float(scores[1] - scores[0])

    hf_gap = gap(weight @ hf + bias)
    vllm_gap = gap(weight @ vllm + bias)
    print(f"\n[{name}] HF/vLLM final activation (recorded, not asserted)")
    print(f"  HF / vLLM norm:               {hf.norm():.4f} / {vllm.norm():.4f}")
    print(
        f"  relative L2 / max abs diff:   "
        f"{(vllm - hf).norm() / hf.norm():.6f} / {(vllm - hf).abs().max():.4f}"
    )
    print(
        f"  reference score gap (fp32 head, HF activation): {gap(reference_scores):.4f}"
    )
    print(f"  loaded bf16 head, HF activation:   {hf_gap:.4f}")
    print(f"  loaded bf16 head, vLLM activation: {vllm_gap:.4f}")
    print(f"  activation contribution:           {vllm_gap - hf_gap:.4f}")


def test_classifier_reads_an_intermediate_layer_correctly(tmp_path_factory):
    """A layer-20 slot under eager and FULL, against an fp32 HF layer-20 reference.

    The fixed seed-0 head's reference margin (about 0.07) is below the score
    tolerance, so the winning label is inconclusive and this does not identify
    layer 20 against nearby layers; it shows the path serves in both modes with
    a score gap within tolerance of the layer-20 reference.
    """
    read_layer = 20  # a middle layer of the 40-layer base
    result = _compose(
        tmp_path_factory, ["compose-at-layer", str(read_layer)], "cls_layer_ckpt"
    )
    label_token_ids = result["label_token_ids"]
    env = {
        "CLS_CKPT": result["ckpt"],
        "CLS_PIDS": _write_pids(result),
        "CLS_LABEL_IDS": json.dumps(label_token_ids),
    }
    # Serve both modes before asserting so one failure cannot hide the other.
    served_by_mode = {
        mode: _run_worker(result["outdir"], [mode], env)
        for mode in ("run-at-layer", "run-at-layer-full-cudagraph")
    }
    failures = []
    for mode, served in served_by_mode.items():
        failures += _check_served_scores(
            f"{mode} (layer={read_layer})",
            served,
            result["ground_truth_scores"],
            result["ground_truth_label_id"],
            label_token_ids,
            min_label_margin=1.0,
        )
    assert not failures, "\n".join(failures)


def test_two_classifier_slots_replay_correctly_in_one_captured_shape(tmp_path_factory):
    """Serve a final-layer and a layer-15 slot under eager and FULL.

    Classifier arithmetic and transport are checked against the head's
    actual input, recorded in-process at ``compute_logits``:

    - slot 1 (final layer): packed scores equal the loaded bf16 head applied
      in fp32 to the served final activation, rounded to bf16 (exact);
    - both slots: rewritten label logits equal packed scores (exact), and
      label logprob differences equal logit differences up to fp32
      log-softmax rounding.

    Slot 2's input (layer 15) is not observable at ``compute_logits``, so its
    source-selection coverage remains the comparison against the layer-15 HF
    reference. Slot 1's HF/vLLM activation difference is printed, with no
    acceptance threshold. The test does not prove which captured graph replayed.
    """
    read_layer = 15
    result = _compose(
        tmp_path_factory, ["compose-two-slot", str(read_layer)], "cls_two_slot_ckpt"
    )
    label_token_ids = result["label_token_ids"]
    env = {
        "CLS_CKPT": result["ckpt"],
        "CLS_PIDS_1": json.dumps(result["slot1_prompt_ids"]),
        "CLS_PIDS_2": json.dumps(result["slot2_prompt_ids"]),
        "CLS_LABEL_IDS": json.dumps(label_token_ids),
    }
    # Serve both modes before asserting so one failure cannot hide another.
    served_by_mode = {
        mode: _run_worker(result["outdir"], [mode], env)
        for mode in ("run-two-slot-eager", "run-two-slot-full-cudagraph")
    }
    failures = []
    for mode, served in served_by_mode.items():
        slot1, slot2 = served["slot1"], served["slot2"]
        failures += _check_transport(f"{mode} slot1", slot1, label_token_ids)
        failures += _check_final_layer_arithmetic(f"{mode} slot1", slot1)
        _print_final_layer_activation(
            f"{mode} slot1",
            slot1,
            result["slot1_reference_hidden"],
            result["slot1_ground_truth_scores"],
        )
        if slot1["token_id"] != result["slot1_ground_truth_label_id"]:
            failures.append(
                f"{mode} slot1: served token {slot1['token_id']} "
                f"({slot1['text']!r}) != reference label "
                f"{result['slot1_ground_truth_label_id']}"
            )
        failures += _check_transport(f"{mode} slot2", slot2, label_token_ids)
        failures += _check_served_scores(
            f"{mode} slot2 (read_layer={read_layer})",
            slot2,
            result["slot2_ground_truth_scores"],
            result["slot2_ground_truth_label_id"],
            label_token_ids,
            min_label_margin=None,
        )
    assert not failures, "\n".join(failures)
