# SPDX-License-Identifier: Apache-2.0
"""PRESERVE under interleaved traffic, and under cache eviction.

The A/B prefix-cache suite runs one conversation at a time against a cache large
enough to hold it. Production is neither. Two things it therefore cannot answer:

CONCURRENT  two PRESERVE conversations in flight together. Each has its own token
    prefix, so each should hit its own blocks and route exactly as it does alone.
    The engine's batch-invariance is covered for a single forward; this covers two
    conversations, each with its own transcript, whose turns interleave across many
    forwards. The arm fails loudly if the scheduler never actually co-batched them,
    because then it proves nothing about concurrency.

EVICT  a cache far too small to keep both conversations resident. PRESERVE's
    *saving* must degrade -- that is inherent, and not a bug -- while its
    *correctness* must not. The same ids are recomputed and recomputation
    reproduces the same routing. If correctness tracked cache pressure the policy
    would be unusable in production, where pressure is the normal condition.
    Pressure comes from a second conversation rather than from one conversation
    outgrowing the cache; see the test's own docstring for why that is the only
    satisfiable form.

Attribution under concurrency is the part that has produced false results before,
so it lives in ``attribute_spans`` and is unit-tested on CPU in
``tests/unit/test_conversation_span_attribution.py``. What remains here is
plumbing around a live server.

Markers: slow + requires_model + gpu, gated on ``GRANITE_SWITCH_E2E_MODELS=1``.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WORKER = Path(__file__).parent / "_conversation_concurrency_worker.py"
SERVE_TIMEOUT = 3600

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="serves a real composed checkpoint; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]


def _model_dir():
    d = os.environ.get("GRANITE_SWITCH_CONV_DIR")
    if not d or not os.path.exists(os.path.join(d, "config.json")):
        pytest.skip(
            "set GRANITE_SWITCH_CONV_DIR to a checkpoint composed with "
            "--switch-type multi and at least two aLoRA adapters"
        )
    return d


def _run(command, model_dir, extra=()):
    with tempfile.TemporaryDirectory() as wd:
        out = str(Path(wd) / f"{command}.json")
        cmd = [
            sys.executable,
            str(WORKER),
            command,
            "--model-path",
            model_dir,
            "--output-path",
            out,
            *extra,
        ]
        print(f"\n{'=' * 72}\n  {' '.join(cmd)}\n{'=' * 72}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SERVE_TIMEOUT)
        if r.stdout:
            print(r.stdout[-2500:])
        if r.stderr:
            print("STDERR (tail):", r.stderr[-8000:])
        assert r.returncode == 0, (
            f"{command} failed (exit {r.returncode})\n{r.stderr[-3000:]}"
        )
        return json.loads(Path(out).read_text())


def test_two_preserve_conversations_do_not_disturb_each_other():
    data = _run("concurrent", _model_dir())
    assert not data.get("error"), data["error"]

    solo, conc = data["arms"]["solo"], data["arms"]["conc"]
    print(
        f"\n  forwards      solo {solo['forwards']:>4d}   conc {conc['forwards']:>4d}"
    )
    print(
        f"  co-batched    solo {solo['multi_request_forwards']:>4d}   "
        f"conc {conc['multi_request_forwards']:>4d}"
    )
    for arm_name, arm in (("solo", solo), ("conc", conc)):
        for label, c in arm["conversations"].items():
            print(
                f"  {arm_name}/{label}  prompt {c['prompt_len']:>4d}  "
                f"attributed {len(c['routing']):>4d}  "
                f"decode {c['decode_indices']}  want {c['expected_index']}"
            )

    # ── Anti-vacuity: the arm has to have been concurrent ────────────────────
    assert conc["decode_windows_disjoint"], (
        "the two conversations' prompt lengths are closer together than "
        f"max_new_tokens={conc['max_new_tokens']}, so their decode seq_len windows "
        "overlap and a decode row could belong to either. Attribution below would "
        "be a coin toss. Widen the fillers so the prompt lengths differ by more "
        "than max_new_tokens."
    )
    assert conc["multi_request_forwards"] > 0, (
        "no forward carried more than one request, so the scheduler never "
        "co-batched the two conversations and this arm says nothing about "
        "concurrency. Raise max_num_seqs, or make the turns overlap for longer."
    )

    for label, c in conc["conversations"].items():
        assert c["routing"], (
            f"conversation {label}: no prompt position could be attributed, so the "
            "comparison below would be vacuous"
        )
        assert c["decode_indices"] == [c["expected_index"]], (
            f"conversation {label}: its answer was generated with adapter "
            f"index(es) {c['decode_indices']}, wanted {c['expected_index']}. Under "
            "concurrency a wrong index means one conversation's adapter leaked into "
            "the other's generation."
        )

    # ── The claim: interleaving changes scheduling, not routing ──────────────
    for label in conc["conversations"]:
        s = solo["conversations"][label]["routing"]
        c = conc["conversations"][label]["routing"]
        shared = sorted(set(s) & set(c), key=int)
        assert shared, (
            f"conversation {label}: the solo and concurrent arms attributed no "
            "position in common, so they cannot be compared. Cache state differs "
            "between arms, so the sets differ -- but not to the point of disjoint."
        )
        disagreements = {p: (s[p], c[p]) for p in shared if s[p] != c[p]}
        assert not disagreements, (
            f"conversation {label} routed differently when run concurrently: "
            f"{dict(list(disagreements.items())[:8])} (position: solo, conc). "
            "Routing must depend only on the request's own tokens; a difference "
            "here means a neighbour's request changed it."
        )


def test_correctness_survives_cache_eviction():
    """Cache pressure must cost recomputation, never routing correctness.

    Two conversations share one deliberately tiny cache, rather than one
    conversation growing past it. That is forced by arithmetic, not taste. vLLM
    will not start unless the KV cache holds one whole ``max_model_len`` request,
    so every servable turn satisfies ``capacity >= max_model_len >= prompt +
    max_tokens``; and under PRESERVE a turn's prompt *contains* the previous
    turn's transcript as its own prefix, i.e. the same blocks. So a single
    conversation that the server can serve is one whose history still fits, and
    nothing is ever evicted -- while the first turn that would overflow the cache
    is the one the server refuses. The earlier single-conversation form of this
    test sat in exactly that gap and died at engine init (a 256-token cache
    against ``max_model_len=4096``), which is why the fix is a second
    conversation rather than a different ``--blocks``.

    A second conversation with a different filler has a disjoint prefix and thus
    its own blocks, so it applies pressure the first conversation did not create.
    Measured at turn 3 the two prompts are ~173 and ~148 tokens: each alone needs
    189 of the 256 available and stays servable, while together they need ~321
    and evict each other.
    """
    data = _run("evict", _model_dir(), extra=["--blocks", "16", "--turns", "4"])
    assert not data.get("error"), data["error"]

    capacity = data["blocks"] * 16
    print(
        f"\n  cache: {data['blocks']} blocks ({capacity} tokens), "
        f"max_model_len {data.get('max_model_len', capacity)}"
    )
    hdr = (
        f"\n  {'turn':>4s} {'conv':>4s} {'prompt':>7s} {'hits':>6s} {'ceil':>6s} "
        f"{'short':>6s} {'want':>5s}  decode  history"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for t in data["turns"]:
        short = t["reusable_ceiling"] - t["hits_delta"]
        print(
            f"  {t['turn']:>4d} {t.get('label', '-'):>4s} {t['prompt_len']:>7d} "
            f"{t['hits_delta']:>6.0f} {t['reusable_ceiling']:>6d} {short:>6.0f} "
            f"{t['expected_index']:>5d}  {t['decode_indices']!s:>8s}  "
            f"{t['history_routing']}"
        )

    turns = data["turns"]
    assert turns, "no turns were run"

    def _name(t):
        return f"turn {t['turn']}/{t.get('label', '-')}"

    # ── Anti-vacuity: eviction has to have actually bitten ──────────────────
    # NOT "the final prompt exceeds the cache": a request larger than the whole
    # cache cannot be served at all, so that condition can never hold on a run
    # that produced results -- it made the first version of this test unsatisfiable
    # rather than strict. Eviction is instead evidenced directly: hits below what
    # the block arithmetic made available means blocks that should have been
    # resident were not.
    shortfalls = {
        _name(t): t["reusable_ceiling"] - t["hits_delta"]
        for t in turns
        if t["hits_delta"] < t["reusable_ceiling"]
    }
    assert shortfalls, (
        f"every turn reused everything the block arithmetic allowed, so the "
        f"{capacity}-token cache never evicted anything and this test measures the "
        "same thing as the un-evicted suite. Add turns or lower --blocks -- but "
        "not below one full request, or the server cannot serve at all."
    )
    print(
        f"  eviction bit on {sorted(shortfalls)} "
        f"(shortfall vs the block-math ceiling: {shortfalls})"
    )

    # ── The claim: the saving degrades, the routing does not ────────────────
    for t in turns:
        assert t["decode_indices"] == [t["expected_index"]], (
            f"{_name(t)}: generated with adapter index(es) "
            f"{t['decode_indices']}, wanted {t['expected_index']}. Cache pressure "
            "must cost recomputation, never correctness -- evicted blocks are "
            "recomputed from the same ids and must reproduce the same routing."
        )
        assert t["answer_ids_source"] == "logprobs", (
            f"{_name(t)}: the answer was re-encoded from text, so the "
            "transcript may not match what was sent and any hit rate below is "
            "unattributable to eviction"
        )
