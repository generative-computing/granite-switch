# SPDX-License-Identifier: Apache-2.0
"""Serving-shape coverage for the coded MultiSwitch: chunked prefill, decode, mixed.

``tests/vllm/test_multi_switch.py::TestContinuousBatching`` covers whole-request
prefills packed into one flat forward. Real vLLM serving also produces shapes where
a request's ``query_len < seq_len``, which that harness cannot express:

* **chunked prefill** — a long prompt is split across forwards. Only the FIRST chunk
  contains the request's position-0 counting anchor; later chunks must recover the
  count from the control tokens already in the KV cache.
* **decode** — ``query_len == 1`` over a full history.
* **mixed** — some requests prefilling (or chunk-prefilling) while others decode in
  the SAME forward, at different sequence lengths.

Why these are the risky shapes: the coded engine's ``1/(1+n)`` count is anchored at
``positions == 0``. That anchor exists once per request, in its first chunk. Every
later chunk and every decode step has NO anchor in its own query rows and must reach
back through the paged KV cache. Chunked prefill is therefore the same class of
assumption as the original batching bug (an anchor that happens to be present for
whole-request prefill), and it is disabled in every other test in the suite.

Ground truth throughout is latest-wins computed in Python from the token stream, so a
comparison never depends on the engine agreeing with itself.

Also covered here (previously untested):
* routing invariance to a request's SLOT in the batch (the exact regression shape:
  >=3 control tokens at a non-first slot),
* long sequences co-batched (all other batch tests use short synthetic ones),
* return-to-base under batching and across chunk boundaries,
* stale-slot reuse: a fresh request landing on a recycled block range must not
  inherit the previous request's routing,
* codebook capacity: many control tokens in one request.

Requires GPU + vLLM (same gate as the sibling module).
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from tests.shared.multi_switch_cases import (
    A_TOK,
    ATOK_BASE_RESET,
    ATOK_NO_BASE,
    B_TOK,
    BASE_TOK,
    TEXT_TOKEN,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vLLM MultiSwitch tests require a GPU"
)

WORKER = os.path.join(os.path.dirname(__file__), "_multi_switch_worker.py")
_PROC = None


def _worker():
    """Start the worker and CONSUME its startup handshake line.

    The worker emits ``{"ready": true, ...}`` (or ``{"fatal": ...}``) before any
    response. Not reading it leaves every later read off by one: the first _send
    returns the handshake, which has neither "result" nor "error".
    """
    global _PROC
    if _PROC is None:
        proc = subprocess.Popen(
            [sys.executable, WORKER, "multi"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready_line = proc.stdout.readline()
        if not ready_line:
            stderr = proc.stderr.read()
            raise RuntimeError(f"worker failed to start:\n{stderr}")
        ready = json.loads(ready_line)
        if "fatal" in ready:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            tail = (proc.stderr.read() or "")[-2000:]
            pytest.fail(
                f"vLLM multi worker cannot start: {ready['fatal']}\n"
                f"Backend: {ready.get('backend_name', 'unknown')}\n"
                f"Hint: {ready.get('hint', '')}\n--- stderr ---\n{tail}",
                pytrace=False,
            )
        assert ready.get("ready"), f"unexpected handshake: {ready}"
        _PROC = proc
    return _PROC


def teardown_module(_module):
    global _PROC
    if _PROC is not None:
        try:
            _PROC.stdin.close()
            _PROC.wait(timeout=30)
        except Exception:
            _PROC.kill()
        _PROC = None


def _send(payload):
    p = _worker()
    p.stdin.write(json.dumps(payload) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        raise RuntimeError("worker died")
    resp = json.loads(line)
    if "error" in resp:
        raise AssertionError(f"worker error:\n{resp['error']}")
    if "result" not in resp:
        # e.g. the startup handshake read out of turn, which would otherwise
        # surface as a bare KeyError and desync every later read.
        raise AssertionError(f"malformed worker reply (no 'result'): {resp}")
    return resp["result"]


def _steps(steps, atok, num_requests, seq_capacity):
    return _send(
        {
            "command": "forward_steps",
            "steps": steps,
            "adapter_token_ids": atok,
            "num_requests": num_requests,
            "seq_capacity": seq_capacity,
        }
    )


def _expected(tokens, atok):
    """Latest-wins ground truth: adapter active at each position.

    Mirrors the engine's two layouts: with ``len(atok) == num_adapters + 1`` slot 0
    is the base-reset token and the expert id IS the slot index; otherwise slot i
    fires adapter i+1.
    """
    offset = 0 if len(atok) == 3 else 1  # shared cases use num_adapters == 2
    out, cur = [], 0
    for t in tokens:
        if t in atok:
            cur = atok.index(t) + offset
        out.append(cur)
    return out


def _chunks(tokens, sizes):
    """Split a token list into consecutive chunks of the given sizes."""
    out, off = [], 0
    for n in sizes:
        out.append(tokens[off : off + n])
        off += n
    if off < len(tokens):
        out.append(tokens[off:])
    return [c for c in out if c]


class TestChunkedPrefill:
    """A request's prompt split across forwards; only chunk 0 holds the anchor."""

    def _run_chunked(self, tokens, sizes, atok):
        """Prefill ``tokens`` in chunks; return the concatenated routing."""
        chunks = _chunks(tokens, sizes)
        steps, cached = [], 0
        for c in chunks:
            steps.append([{"req": 0, "tokens": c, "cached": cached}])
            cached += len(c)
        res = _steps(steps, atok, num_requests=1, seq_capacity=len(tokens) + 8)
        return [i for step in res for i in step["0"]]

    def test_chunked_matches_whole_prefill(self):
        """Chunked prefill must route identically to one whole-request prefill."""
        tokens = (
            [TEXT_TOKEN] * 3 + [A_TOK] + [TEXT_TOKEN] * 4 + [B_TOK] + [TEXT_TOKEN] * 3
        )
        whole = self._run_chunked(tokens, [len(tokens)], ATOK_NO_BASE)
        assert whole == _expected(tokens, ATOK_NO_BASE), (
            f"whole prefill wrong: {whole} != {_expected(tokens, ATOK_NO_BASE)}"
        )
        chunked = self._run_chunked(tokens, [4, 4, 4], ATOK_NO_BASE)
        assert chunked == whole, (
            f"chunked prefill diverged from whole prefill:\n"
            f"  whole:   {whole}\n  chunked: {chunked}\n"
            f"  (later chunks carry no positions==0 anchor and must recover the "
            f"count from control tokens already in the paged KV cache)"
        )

    def test_control_token_in_earlier_chunk(self):
        """A control token in chunk 0 must still drive routing in chunk 1+."""
        tokens = [A_TOK] + [TEXT_TOKEN] * 7
        # Split so the control token is alone in the first chunk.
        got = self._run_chunked(tokens, [1, 3, 4], ATOK_NO_BASE)
        assert got == _expected(tokens, ATOK_NO_BASE), (
            f"adapter set in chunk 0 did not carry into later chunks: {got}"
        )

    def test_control_token_on_chunk_boundary(self):
        """Control token as the LAST token of a chunk, and as the FIRST of one."""
        tokens = [TEXT_TOKEN] * 3 + [A_TOK] + [TEXT_TOKEN] * 4
        last_of_chunk = self._run_chunked(tokens, [4, 4], ATOK_NO_BASE)
        assert last_of_chunk == _expected(tokens, ATOK_NO_BASE), (
            f"control token last-in-chunk: {last_of_chunk}"
        )
        first_of_chunk = self._run_chunked(tokens, [3, 5], ATOK_NO_BASE)
        assert first_of_chunk == _expected(tokens, ATOK_NO_BASE), (
            f"control token first-in-chunk: {first_of_chunk}"
        )

    def test_many_small_chunks(self):
        """One token per forward — the pathological chunking."""
        tokens = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN, TEXT_TOKEN]
        got = self._run_chunked(tokens, [1] * len(tokens), ATOK_NO_BASE)
        assert got == _expected(tokens, ATOK_NO_BASE), (
            f"single-token chunks: {got} != {_expected(tokens, ATOK_NO_BASE)}"
        )

    def test_chunked_return_to_base(self):
        """Base-reset in a later chunk must return routing to base."""
        tokens = [A_TOK] + [TEXT_TOKEN] * 3 + [BASE_TOK] + [TEXT_TOKEN] * 3
        got = self._run_chunked(tokens, [3, 3, 2], ATOK_BASE_RESET)
        assert got == _expected(tokens, ATOK_BASE_RESET), (
            f"chunked return-to-base: {got} != {_expected(tokens, ATOK_BASE_RESET)}"
        )

    def test_chunked_three_controls(self):
        """>=3 control tokens across chunk boundaries (the sensitive count)."""
        tokens = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN, A_TOK] + [
            TEXT_TOKEN
        ] * 4
        got = self._run_chunked(tokens, [2, 2, 3, 3], ATOK_NO_BASE)
        assert got == _expected(tokens, ATOK_NO_BASE), (
            f"3 controls across chunks: {got} != {_expected(tokens, ATOK_NO_BASE)}"
        )


class TestDecode:
    """query_len == 1 over a cached history: the control token is not in input_ids."""

    def _prefill_then_decode(self, prompt, gen_tokens, atok):
        """Prefill ``prompt``, then feed ``gen_tokens`` one per step."""
        steps = [[{"req": 0, "tokens": list(prompt), "cached": 0}]]
        cached = len(prompt)
        for t in gen_tokens:
            steps.append([{"req": 0, "tokens": [t], "cached": cached}])
            cached += 1
        res = _steps(
            steps, atok, num_requests=1, seq_capacity=len(prompt) + len(gen_tokens) + 8
        )
        return res

    def test_decode_carries_adapter(self):
        """Adapter set during prefill stays active through every decode step."""
        prompt = [TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        gen = [TEXT_TOKEN] * 6
        res = self._prefill_then_decode(prompt, gen, ATOK_NO_BASE)
        decode_idx = [step["0"][0] for step in res[1:]]
        assert decode_idx == [1] * len(gen), (
            f"decode did not carry the prefill adapter: {decode_idx} "
            f"(expected all 1). The control token lives only in the KV cache at "
            f"this point, so this is the cache-carry property."
        )

    def test_control_token_generated_during_decode(self):
        """A control token EMITTED mid-generation switches the adapter."""
        prompt = [TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        gen = [TEXT_TOKEN, B_TOK, TEXT_TOKEN, TEXT_TOKEN]
        res = self._prefill_then_decode(prompt, gen, ATOK_NO_BASE)
        decode_idx = [step["0"][0] for step in res[1:]]
        assert decode_idx == [1, 2, 2, 2], (
            f"a control token generated during decode did not switch routing: "
            f"{decode_idx} (expected [1, 2, 2, 2]) — this is the agentic "
            f"multi-turn case"
        )

    def test_decode_return_to_base(self):
        """A base-reset token generated during decode returns routing to base."""
        prompt = [A_TOK, TEXT_TOKEN]
        gen = [TEXT_TOKEN, BASE_TOK, TEXT_TOKEN]
        res = self._prefill_then_decode(prompt, gen, ATOK_BASE_RESET)
        decode_idx = [step["0"][0] for step in res[1:]]
        assert decode_idx == [1, 0, 0], (
            f"decode return-to-base: {decode_idx} (expected [1, 0, 0])"
        )

    def test_decode_from_base_stays_base(self):
        """No control token anywhere: every decode step routes to base."""
        res = self._prefill_then_decode(
            [TEXT_TOKEN] * 4, [TEXT_TOKEN] * 5, ATOK_NO_BASE
        )
        decode_idx = [step["0"][0] for step in res[1:]]
        assert decode_idx == [0] * 5, f"base-only decode: {decode_idx}"


class TestMixedBatches:
    """Prefill, chunked prefill and decode requests in the SAME forward."""

    def test_mixed_prefill_and_decode(self):
        """One request decoding while another prefills, in one forward."""
        # Step 0: both prefill (A-request and B-request).
        a_prompt = [TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        b_prompt = [B_TOK, TEXT_TOKEN]
        steps = [
            [
                {"req": 0, "tokens": a_prompt, "cached": 0},
                {"req": 1, "tokens": b_prompt, "cached": 0},
            ]
        ]
        # Step 1: req 0 decodes; req 2 arrives and prefills.
        c_prompt = [TEXT_TOKEN, TEXT_TOKEN, A_TOK]
        steps.append(
            [
                {"req": 0, "tokens": [TEXT_TOKEN], "cached": len(a_prompt)},
                {"req": 2, "tokens": c_prompt, "cached": 0},
            ]
        )
        # Step 2: reqs 0 and 2 both decode, req 1 decodes too.
        steps.append(
            [
                {"req": 0, "tokens": [TEXT_TOKEN], "cached": len(a_prompt) + 1},
                {"req": 1, "tokens": [TEXT_TOKEN], "cached": len(b_prompt)},
                {"req": 2, "tokens": [TEXT_TOKEN], "cached": len(c_prompt)},
            ]
        )
        res = _steps(steps, ATOK_NO_BASE, num_requests=3, seq_capacity=16)

        assert res[0]["0"] == _expected(a_prompt, ATOK_NO_BASE), res[0]["0"]
        assert res[0]["1"] == _expected(b_prompt, ATOK_NO_BASE), res[0]["1"]
        # req 0 decoding (adapter 1) alongside req 2 prefilling.
        assert res[1]["0"] == [1], f"decode row in mixed forward: {res[1]['0']}"
        assert res[1]["2"] == _expected(c_prompt, ATOK_NO_BASE), res[1]["2"]
        # All three decoding: 1, 2, 1 respectively.
        assert res[2]["0"] == [1], f"req0 decode: {res[2]['0']}"
        assert res[2]["1"] == [2], f"req1 decode: {res[2]['1']}"
        assert res[2]["2"] == [1], f"req2 decode: {res[2]['2']}"

    def test_mixed_chunked_prefill_and_decode(self):
        """A chunk-prefilling request co-batched with decoding requests."""
        long_prompt = (
            [TEXT_TOKEN] * 2 + [B_TOK] + [TEXT_TOKEN] * 3 + [A_TOK] + [TEXT_TOKEN] * 2
        )
        short = [A_TOK, TEXT_TOKEN]
        steps = [
            [
                {"req": 0, "tokens": short, "cached": 0},
                {"req": 1, "tokens": long_prompt[:4], "cached": 0},
            ]
        ]
        steps.append(
            [
                {"req": 0, "tokens": [TEXT_TOKEN], "cached": len(short)},
                {"req": 1, "tokens": long_prompt[4:], "cached": 4},
            ]
        )
        res = _steps(steps, ATOK_NO_BASE, num_requests=2, seq_capacity=16)

        exp_long = _expected(long_prompt, ATOK_NO_BASE)
        assert res[0]["1"] == exp_long[:4], (
            f"chunk 0 of co-batched request: {res[0]['1']} != {exp_long[:4]}"
        )
        assert res[1]["1"] == exp_long[4:], (
            f"chunk 1 of a request co-batched WITH A DECODE row: {res[1]['1']} "
            f"!= {exp_long[4:]} — this chunk has no anchor of its own"
        )
        assert res[0]["0"] == _expected(short, ATOK_NO_BASE), res[0]["0"]
        assert res[1]["0"] == [1], f"decode alongside chunked prefill: {res[1]['0']}"

    def test_decode_only_batch(self):
        """A pure decode batch: no request has a position-0 anchor at all."""
        prompts = [
            [TEXT_TOKEN, A_TOK, TEXT_TOKEN],
            [B_TOK, TEXT_TOKEN],
            [TEXT_TOKEN, TEXT_TOKEN],
        ]
        steps = [[{"req": i, "tokens": p, "cached": 0} for i, p in enumerate(prompts)]]
        steps.append(
            [
                {"req": i, "tokens": [TEXT_TOKEN], "cached": len(p)}
                for i, p in enumerate(prompts)
            ]
        )
        res = _steps(steps, ATOK_NO_BASE, num_requests=3, seq_capacity=12)
        got = [res[1][str(i)][0] for i in range(3)]
        assert got == [1, 2, 0], (
            f"pure decode batch routed {got}, expected [1, 2, 0] — no request "
            f"contributes a position-0 row in this forward"
        )


class TestSlotAndOrderInvariance:
    """Routing must not depend on batch slot, request order, or slot reuse."""

    def test_routing_invariant_to_batch_slot(self):
        """A >=3-control request routes the same at every slot in the batch.

        This is the exact regression shape: with a fabricated flat arange only the
        FIRST request had an anchor, so a request's routing depended on its slot.
        """
        target = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN, A_TOK] + [
            TEXT_TOKEN
        ] * 3
        filler = [TEXT_TOKEN] * 5
        exp = _expected(target, ATOK_NO_BASE)
        for slot in range(4):
            seqs = [filler] * 4
            seqs[slot] = target
            steps = [[{"req": i, "tokens": s, "cached": 0} for i, s in enumerate(seqs)]]
            res = _steps(steps, ATOK_NO_BASE, num_requests=4, seq_capacity=16)
            assert res[0][str(slot)] == exp, (
                f"request routing depended on its batch slot: at slot {slot} got "
                f"{res[0][str(slot)]}, expected {exp}"
            )

    def test_stale_slot_not_inherited(self):
        """A new request reusing a block range must not inherit prior routing.

        vLLM recycles decode slots; the Mamba precedent is a new request reading
        the previous occupant's leftover state.
        """
        # Step 0: req 0 activates adapter 2.
        steps = [[{"req": 0, "tokens": [B_TOK, TEXT_TOKEN, TEXT_TOKEN], "cached": 0}]]
        # Step 1: a *fresh* prefill on the SAME request slot, no control token.
        # cached=0 marks it as a new sequence occupying the same blocks.
        steps.append([{"req": 0, "tokens": [TEXT_TOKEN] * 3, "cached": 0}])
        res = _steps(steps, ATOK_NO_BASE, num_requests=1, seq_capacity=12)
        assert res[1]["0"] == [0, 0, 0], (
            f"fresh request on a recycled slot inherited stale routing: "
            f"{res[1]['0']} (expected all base). Prior occupant used adapter 2."
        )


class TestLongAndCapacity:
    """Long sequences co-batched, and many control tokens in one request."""

    def test_long_sequences_batched(self):
        """Long, unequal, co-batched requests with mid-sequence controls."""
        a = [TEXT_TOKEN] * 120 + [A_TOK] + [TEXT_TOKEN] * 80
        b = (
            [TEXT_TOKEN] * 40
            + [B_TOK]
            + [TEXT_TOKEN] * 200
            + [A_TOK]
            + [TEXT_TOKEN] * 30
        )
        c = [TEXT_TOKEN] * 15
        steps = [
            [
                {"req": 0, "tokens": a, "cached": 0},
                {"req": 1, "tokens": b, "cached": 0},
                {"req": 2, "tokens": c, "cached": 0},
            ]
        ]
        res = _steps(steps, ATOK_NO_BASE, num_requests=3, seq_capacity=320)
        for i, seq in enumerate((a, b, c)):
            exp = _expected(seq, ATOK_NO_BASE)
            assert res[0][str(i)] == exp, (
                f"long co-batched request {i} mis-routed; first mismatch at "
                f"{next(k for k in range(len(exp)) if res[0][str(i)][k] != exp[k])}"
            )

    def test_many_control_tokens_one_request(self):
        """40 alternating control tokens in a single request (codebook writes)."""
        seq = []
        for i in range(40):
            seq.append(A_TOK if i % 2 == 0 else B_TOK)
            seq.extend([TEXT_TOKEN] * 2)
        steps = [[{"req": 0, "tokens": seq, "cached": 0}]]
        res = _steps(steps, ATOK_NO_BASE, num_requests=1, seq_capacity=len(seq) + 8)
        assert res[0]["0"] == _expected(seq, ATOK_NO_BASE), (
            "40-transition request mis-routed"
        )

    def test_many_controls_chunked(self):
        """Many transitions AND chunked prefill together."""
        seq = []
        for i in range(20):
            seq.append(A_TOK if i % 2 == 0 else B_TOK)
            seq.extend([TEXT_TOKEN] * 3)
        steps, cached = [], 0
        for chunk in _chunks(seq, [16] * (len(seq) // 16 + 1)):
            steps.append([{"req": 0, "tokens": chunk, "cached": cached}])
            cached += len(chunk)
        res = _steps(steps, ATOK_NO_BASE, num_requests=1, seq_capacity=len(seq) + 8)
        got = [i for step in res for i in step["0"]]
        assert got == _expected(seq, ATOK_NO_BASE), (
            "20 transitions across chunk boundaries mis-routed"
        )


class TestBaseResetLayout:
    """The num_adapters+1 layout, and the consequence of composing without it."""

    def test_base_reset_only_in_plus_one_layout(self):
        """With no base slot there is no token that can return to base.

        ``len(adapter_token_ids) == num_adapters`` sets ``_expert_id_offset = 1``,
        so slot i fires adapter i+1 and NO token maps to expert 0; a checkpoint with
        that layout structurally cannot return to base mid-sequence.

        SCOPE — read this before citing the test. It builds both layouts BY HAND via
        the worker's mock config, so it verifies the ENGINE's handling of each. It
        does NOT exercise a real composed checkpoint. The composer CAN now produce
        the +1 layout — compose with ``--base-reset-token`` and
        ``add_control_tokens(base_reset=True)`` prepends ``<|base_reset|>`` — and
        ``tests/composer/test_base_reset_token.py`` drives the engine with exactly
        those compose-produced ids. What is still caller-side is *emitting* the
        token: the chat template only emits control tokens for adapters, and
        base-reset is not an adapter, so the id is placed in the prompt by whoever
        builds the request. The base-reset token needs NO adapter weights, since
        expert 0 IS the base model.
        """
        seq = [A_TOK, TEXT_TOKEN, BASE_TOK, TEXT_TOKEN]
        # Base-reset layout: BASE_TOK returns to base.
        with_base = _steps(
            [[{"req": 0, "tokens": seq, "cached": 0}]],
            ATOK_BASE_RESET,
            num_requests=1,
            seq_capacity=12,
        )
        assert with_base[0]["0"] == [1, 1, 0, 0], with_base[0]["0"]
        # No-base layout: the same id is just another adapter, never base.
        no_base = _steps(
            [[{"req": 0, "tokens": seq, "cached": 0}]],
            ATOK_NO_BASE,
            num_requests=1,
            seq_capacity=12,
        )
        assert 0 not in no_base[0]["0"][1:], (
            f"no-base-slot layout unexpectedly returned to base: {no_base[0]['0']}"
        )

    def test_base_reset_batched_and_isolated(self):
        """Base-reset in one request must not reset a co-batched request."""
        resetting = [A_TOK, TEXT_TOKEN, BASE_TOK, TEXT_TOKEN]
        holding = [B_TOK, TEXT_TOKEN, TEXT_TOKEN, TEXT_TOKEN]
        steps = [
            [
                {"req": 0, "tokens": resetting, "cached": 0},
                {"req": 1, "tokens": holding, "cached": 0},
            ]
        ]
        res = _steps(steps, ATOK_BASE_RESET, num_requests=2, seq_capacity=12)
        assert res[0]["0"] == _expected(resetting, ATOK_BASE_RESET), res[0]["0"]
        assert res[0]["1"] == _expected(holding, ATOK_BASE_RESET), (
            f"a base-reset in another request leaked: {res[0]['1']}"
        )
