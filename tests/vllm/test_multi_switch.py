# SPDX-License-Identifier: Apache-2.0
"""vLLM MultiSwitch tests (Kerdock/DG coded-memory engine).

Shared cases live in ``tests/shared/multi_switch_cases.py``; the coded engine
runs them on the vLLM backend here. This file provides:

- CUDA/vLLM availability gating (lightweight, no CUDA context in the parent).
- A long-lived subprocess worker (``_multi_switch_worker.py``) that owns the
  GPU. ``multi`` (coded) builds 2 vLLM.Attention layers (counting + memory), and
  the worker discovers and wires a KV cache + ForwardContext for all of them.
- ``_run(seq, adapter_token_ids)`` that delegates to the worker.

Requires a CUDA GPU + vLLM. All tests skip otherwise. All GPU work happens in
the subprocess worker — the parent pytest process never creates a CUDA context
(required for Exclusive_Process GPU mode).

Coverage note: the coded engine needs two live vLLM.Attention kernels; if the
worker cannot start it (e.g. an FA kernel-image mismatch on this GPU), the worker
returns a structured fatal message and every case fails with that one clear
reason instead of a false pass.
"""

import atexit
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="requires vLLM installed (GPU checked by worker)",
)

from tests.shared.multi_switch_cases import (
    A_TOK,
    ATOK_NO_BASE,
    B_TOK,
    TEXT_TOKEN,
    MultiSwitchEdgeCases,
    MultiSwitchReturnToBaseCases,
    MultiSwitchShapeCorrectnessCases,
    MultiSwitchStickyCases,
    MultiSwitchTransitionCases,
)

SWITCH_TYPES = ["multi"]
EXPECTED_CACHE_LAYERS = {"multi": 2}
EXPECTED_ATTN_LAYERS = {"multi": 2}

_WORKER_PATH = Path(__file__).parent / "_multi_switch_worker.py"

# One worker subprocess per switch_type (they take switch_type as argv[1]).
_workers = {}  # switch_type -> Popen
_worker_locks = {st: threading.Lock() for st in SWITCH_TYPES}
_fatal_startup = {}  # switch_type -> sticky fatal message


def _ensure_worker(switch_type):
    """Lazily start (or reuse) the worker subprocess for ``switch_type``."""
    if switch_type in _fatal_startup:
        pytest.fail(_fatal_startup[switch_type], pytrace=False)
    proc = _workers.get(switch_type)
    if proc is not None and proc.poll() is None:
        return proc

    proc = subprocess.Popen(
        [sys.executable, str(_WORKER_PATH), switch_type],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready_line = proc.stdout.readline()
    if not ready_line:
        stderr = proc.stderr.read()
        raise RuntimeError(f"Worker ({switch_type}) failed to start:\n{stderr}")
    ready = json.loads(ready_line)
    if "fatal" in ready:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr_tail = (proc.stderr.read() or "")[-2000:]
        msg = (
            f"vLLM {switch_type} worker cannot start: {ready['fatal']}\n"
            f"Backend: {ready.get('backend_name', 'unknown')}\n"
            f"Hint: {ready.get('hint', '')}\n"
            f"--- worker stderr (tail) ---\n{stderr_tail}"
        )
        _fatal_startup[switch_type] = msg
        pytest.fail(msg, pytrace=False)
    assert ready.get("ready"), f"Unexpected ready message: {ready}"
    _workers[switch_type] = proc
    atexit.register(_shutdown_all_workers)
    return proc


def _shutdown_all_workers():
    for st, proc in list(_workers.items()):
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        _workers[st] = None


def _send(switch_type, req):
    proc = _ensure_worker(switch_type)
    with _worker_locks[switch_type]:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        resp_line = proc.stdout.readline()
    if not resp_line:
        stderr = proc.stderr.read()
        raise RuntimeError(f"Worker ({switch_type}) died unexpectedly:\n{stderr}")
    resp = json.loads(resp_line)
    if "error" in resp:
        raise RuntimeError(f"Worker error ({switch_type}):\n{resp['error']}")
    return resp["result"]


# ── Module-scoped teardown: release the GPU when this module is done ──


@pytest.fixture(autouse=True, scope="module")
def _worker_lifecycle():
    yield
    _shutdown_all_workers()


# ── _run adapter (coded engine via the worker) ──────────────────────


class _VLLMMultiSwitchBase:
    """Provides ``_run()`` delegating to the worker."""

    switch_type = None  # overridden per subclass

    def _run(self, seq, adapter_token_ids):
        return _send(
            self.switch_type,
            {
                "seq": seq,
                "adapter_token_ids": list(adapter_token_ids),
            },
        )


# ── Shared cases ─────────────────────────────────────────────────────


class TestTransitionsCoded(_VLLMMultiSwitchBase, MultiSwitchTransitionCases):
    switch_type = "multi"


class TestReturnToBaseCoded(_VLLMMultiSwitchBase, MultiSwitchReturnToBaseCases):
    switch_type = "multi"


class TestStickyCoded(_VLLMMultiSwitchBase, MultiSwitchStickyCases):
    switch_type = "multi"


class TestEdgeCasesCoded(_VLLMMultiSwitchBase, MultiSwitchEdgeCases):
    switch_type = "multi"


class TestShapeCoded(_VLLMMultiSwitchBase, MultiSwitchShapeCorrectnessCases):
    switch_type = "multi"


# ── vLLM-specific structural tests ──────────────────────────────────


class TestGeometry:
    """Verify the coded engine's cache-layer / Attention-layer counts on vLLM."""

    @pytest.mark.parametrize("switch_type", SWITCH_TYPES)
    def test_num_cache_and_attn_layers(self, switch_type):
        info = _send(switch_type, {"command": "query_geometry"})
        assert info["switch_type"] == switch_type
        assert info["num_cache_layers"] == EXPECTED_CACHE_LAYERS[switch_type]
        assert info["num_attn_layers"] == EXPECTED_ATTN_LAYERS[switch_type]

    def test_coded_builds_two_attention_layers(self):
        """The coded engine builds exactly the counting + memory Attention layers."""
        info = _send("multi", {"command": "query_geometry"})
        assert info["num_attn_layers"] == 2
        # Layer names come from the coded engine's Attention prefixes.
        assert len(set(info["attn_layer_names"])) == 2


class TestBatchlessSingleRequest:
    """The vLLM coded engine routes a flattened single-request stream correctly."""

    @pytest.mark.parametrize("switch_type", SWITCH_TYPES)
    def test_base_a_b_flat(self, switch_type):
        seq = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN]
        result = _send(switch_type, {"seq": seq, "adapter_token_ids": ATOK_NO_BASE})
        assert result == [0, 1, 1, 2, 2]


# Import the base-reset token for return-to-base batching cases.
from tests.shared.multi_switch_cases import ATOK_BASE_RESET, BASE_TOK


class TestContinuousBatching:
    """vLLM continuous batching: multiple independent requests in ONE flat forward.

    This is the property every previous MultiSwitch implementation (dual-residual,
    fix/multiswitch-kv-cache, and the coded fork's own docstring) left untested or
    scoped out: the single ``positions == 0`` counting anchor is actually PER
    REQUEST (vLLM resets positions per request) and the two vLLM.Attention heads
    mask per request, so a batch must route each request exactly as if it were run
    alone — no cross-request contamination of the 1/(1+n) count.
    """

    def _batch(self, seqs, atok):
        return _send(
            "multi",
            {"command": "forward_batch", "seqs": seqs, "adapter_token_ids": atok},
        )

    def _single(self, seq, atok):
        return _send("multi", {"seq": seq, "adapter_token_ids": atok})

    def test_batch_matches_individual_no_base(self):
        """Each request in a batch routes identically to running it alone."""
        seqs = [
            [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN],  # -> 0,1,1,2,2
            [A_TOK, TEXT_TOKEN, TEXT_TOKEN],  # -> 1,1,1
            [TEXT_TOKEN, TEXT_TOKEN],  # -> 0,0
            [B_TOK, A_TOK],  # -> 2,1
        ]
        batched = self._batch(seqs, ATOK_NO_BASE)
        individual = [self._single(s, ATOK_NO_BASE) for s in seqs]
        assert batched == individual, (
            f"batched routing diverged from per-request:\n"
            f"  batched:    {batched}\n  individual: {individual}"
        )

    def test_batch_return_to_base_isolated(self):
        """Return-to-base in one request must not leak into its neighbors."""
        seqs = [
            [TEXT_TOKEN, A_TOK, TEXT_TOKEN, BASE_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN],
            [A_TOK, TEXT_TOKEN],
            [TEXT_TOKEN, B_TOK, BASE_TOK, TEXT_TOKEN],
        ]
        batched = self._batch(seqs, ATOK_BASE_RESET)
        individual = [self._single(s, ATOK_BASE_RESET) for s in seqs]
        assert batched == individual

    def test_batch_second_request_anchor_independent(self):
        """A control-heavy request 1 must not shift request 2's count/anchor.

        The failure mode of a GLOBAL anchor: request 2's tokens would keep
        counting request 1's control tokens, mis-recovering n and routing wrong.
        Per-request anchoring makes request 2 count from its own position 0.
        """
        r1 = [A_TOK, B_TOK, A_TOK, B_TOK, A_TOK]  # many writes
        r2 = [TEXT_TOKEN, B_TOK, TEXT_TOKEN]  # -> 0,2,2 regardless of r1
        batched = self._batch([r1, r2], ATOK_NO_BASE)
        assert batched[1] == self._single(r2, ATOK_NO_BASE)
        assert batched[1] == [0, 2, 2]

    def test_single_request_batch_equals_batchless(self):
        """A 'batch' of one must equal the batchless path (metadata sanity)."""
        seq = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN]
        assert self._batch([seq], ATOK_NO_BASE) == [self._single(seq, ATOK_NO_BASE)]

    def test_batch_many_controls_vs_ground_truth(self):
        """>=3 controls per request, checked against EXPLICIT ground truth.

        Batched-vs-individual comparison alone is too weak: a wrong write address
        can still resolve to the same adapter, and with <3 control tokens an
        off-by-one address frequently does exactly that -- so the defect hides.
        3+ controls per request is where it becomes reliably visible.

        Ground truth is latest-wins: each position takes the expert written by
        the most recent control token at or before it.
        """
        cases = [
            # 3 controls
            (
                [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN],
                [0, 1, 1, 2, 2, 1, 1],
            ),
            # 4 controls, alternating
            ([A_TOK, B_TOK, A_TOK, B_TOK, TEXT_TOKEN], [1, 2, 1, 2, 2]),
            # 5 controls with text interleaved
            (
                [TEXT_TOKEN, B_TOK, TEXT_TOKEN, A_TOK, B_TOK, TEXT_TOKEN, A_TOK, B_TOK],
                [0, 2, 2, 1, 2, 2, 1, 2],
            ),
        ]
        seqs = [s for s, _ in cases]
        expected = [e for _, e in cases]
        batched = self._batch(seqs, ATOK_NO_BASE)
        assert batched == expected, (
            "batched routing diverged from latest-wins ground truth with >=3 "
            f"controls per request:\n  got:      {batched}\n  expected: {expected}\n"
            "A global (non-per-request) counting anchor recovers a wrong write "
            "address for every request after the first."
        )

    def test_batch_ground_truth_survives_request_order(self):
        """The same request routes to ground truth wherever it sits in the batch.

        A global anchor makes a request's routing depend on how many tokens (and
        control tokens) precede it in the flat batch, so moving it changes the
        result. Per-request anchoring makes batch position irrelevant.
        """
        target = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, A_TOK, TEXT_TOKEN]
        truth = [0, 1, 1, 2, 1, 1]
        filler_short = [TEXT_TOKEN, B_TOK]
        filler_long = [A_TOK, B_TOK, A_TOK, B_TOK, A_TOK, B_TOK, TEXT_TOKEN]

        for idx, batch in enumerate(
            (
                [target, filler_short, filler_long],
                [filler_short, target, filler_long],
                [filler_long, filler_short, target],
            )
        ):
            pos = batch.index(target)
            got = self._batch(batch, ATOK_NO_BASE)[pos]
            assert got == truth, (
                f"arrangement {idx}: target at batch slot {pos} routed {got}, "
                f"expected {truth} -- routing must not depend on batch position"
            )


class TestCountingCeiling:
    """Where bf16 stops recovering a write address, measured on the real engine.

    ``Conversation.MAX_RETAINED_CONTROL_TOKENS`` refuses a request carrying more
    than 188 control tokens, on the grounds that the coded switch's ``1/(1+n)``
    counting signal is produced in the model dtype and bf16 stops inverting
    exactly above that. ``tests/unit/test_counting_ceiling.py`` pins the
    arithmetic; this pins the engine, where the signal comes out of a real
    attention kernel rather than a hand-built tensor.

    The sweep alternates A and B deliberately. With a single repeated adapter an
    off-by-one address retrieves the same expert by luck and aliasing stays
    invisible -- the same coincidence that let three hand-picked SingleSwitch
    cases pass while a full sweep found 62% wrong. Alternating makes a wrong
    address produce a wrong adapter.
    """

    def _sweep(self, n_controls):
        """``[t, <A>, t, <B>, t, ...]`` with ``n_controls`` control tokens."""
        seq, truth, cur = [TEXT_TOKEN], [0], 0
        for i in range(n_controls):
            ctl = A_TOK if i % 2 == 0 else B_TOK
            cur = 1 if i % 2 == 0 else 2
            seq += [ctl, TEXT_TOKEN]
            truth += [cur, cur]
        return seq, truth

    def _first_wrong(self, counts):
        """Smallest control-token count whose routing diverges from latest-wins."""
        for n in counts:
            seq, truth = self._sweep(n)
            got = _send("multi", {"seq": seq, "adapter_token_ids": ATOK_NO_BASE})
            if got != truth:
                first = next(i for i, (g, t) in enumerate(zip(got, truth)) if g != t)
                return n, first, got[first], truth[first]
        return None, None, None, None

    def test_bound_is_safe_on_the_real_engine(self):
        """Routing must be exact at the bound the guard admits.

        Sampled rather than exhaustive: a full 1..188 sweep is 188 forwards of a
        growing sequence, and the failure mode is monotone in n -- precision is
        lost once and stays lost -- so the boundary and a few points below it are
        what carry information.
        """
        from granite_switch.conversation import MAX_RETAINED_CONTROL_TOKENS

        for n in (1, 2, 8, 64, 128, MAX_RETAINED_CONTROL_TOKENS):
            seq, truth = self._sweep(n)
            got = _send("multi", {"seq": seq, "adapter_token_ids": ATOK_NO_BASE})
            assert got == truth, (
                f"{n} control tokens: routing diverged from latest-wins ground "
                f"truth at or below MAX_RETAINED_CONTROL_TOKENS="
                f"{MAX_RETAINED_CONTROL_TOKENS}, so the guard admits requests the "
                f"engine cannot route.\n  got:      {got[:24]}...\n"
                f"  expected: {truth[:24]}..."
            )

    def test_aliasing_does_appear_above_the_bound(self):
        """The guard must not be protecting against nothing.

        If the engine stayed exact far past the bound, the constant would be
        refusing usable conversations. Reported rather than hard-bounded: the
        exact boundary is an IEEE 754 rounding property, and a kernel that
        accumulated the softmax differently could move it.
        """
        from granite_switch.conversation import MAX_RETAINED_CONTROL_TOKENS

        probes = [MAX_RETAINED_CONTROL_TOKENS + d for d in (1, 2, 4, 16, 64, 256)]
        n, pos, got, want = self._first_wrong(probes)
        print(
            f"\n  guard admits          {MAX_RETAINED_CONTROL_TOKENS}"
            f"\n  first wrong at        {n} control tokens"
            f"\n  first wrong position  {pos} (got {got}, wanted {want})"
        )
        assert n is not None, (
            f"routing stayed exact through {probes[-1]} control tokens, so "
            f"MAX_RETAINED_CONTROL_TOKENS={MAX_RETAINED_CONTROL_TOKENS} is refusing "
            "conversations this engine can serve. Re-derive the bound against the "
            "kernel rather than against the bf16 model of it."
        )
