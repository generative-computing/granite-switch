# SPDX-License-Identifier: Apache-2.0
"""PRESERVE_MIXED_HISTORY must actually save work, not just route differently.

The routing claim is settled on CPU. This is the only test that checks the reason
the policy exists: keeping turn 1's ids should let the prefix cache serve turn 1's
blocks -- adapter-1 blocks included -- so turn 2 prefills only the new turn.

Measured two ways, since either alone can mislead:

* **rows routed** -- how many prompt positions the switch was asked to route for
  turn 2, captured inside the server process. A position served from cache never
  reaches the model, so this counts recomputation directly.
* **vLLM's own counters** -- ``vllm:prefix_cache_hits_total`` around each request,
  so a suspiciously low row count cannot be mistaken for reuse when the real
  cause was a request that never ran.

Anti-vacuity guards, each corresponding to a way this could report a false pass:
the checkpoint must really be ``switch_type=multi`` with >=2 aLoRA adapters; the
trace hook must have fired; prefix caching must really be on (queries > 0); and
turn 2 under PRESERVE must really carry two control tokens.

Markers: slow + requires_model + gpu, gated on ``GRANITE_SWITCH_E2E_MODELS=1``.

STATUS: green on 1x A100 (vLLM 0.19.x, granite-4.1-3b + granitelib-rag aLoRA).
Measured there: PRESERVE recomputed 24 of 104 turn-2 positions against
RE_PREFILL's 38 of 102, with 80 cache hits against 64 -- one extra 16-token block.
Both figures landed exactly where the block arithmetic predicts: RE_PREFILL diverges
at its old control token (index 77 -> floor(77/16)*16 = 64) and PRESERVE at its
transcript boundary (89 -> floor(89/16)*16 = 80).

Those absolute numbers are from before the two arms' openings were made equal in
length, so a rerun's lengths are a few tokens shorter; the assertions do not depend
on them. Each arm's expected hit count is derived from that run's own reported
lengths, so what is checked is the arithmetic, not the constants.

The first run of this file FAILED at 31 vs 31, and usefully: the answer was being
re-encoded from the returned text, which capped reuse at floor(74/16)*16 = 64, the
last block boundary before the answer. Hence the ids-not-text assertion below --
without it the test passes while measuring nothing. That failure is also why reuse
is now asserted to the block: it left the direction right and the number one block
low, which the earlier "PRESERVE routed fewer positions" check would have accepted.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WORKER = Path(__file__).parent / "_conversation_prefix_cache_worker.py"
BUILD_TIMEOUT = 9000
SERVE_TIMEOUT = 3600

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="composes and serves a real ~3B checkpoint; set "
        "GRANITE_SWITCH_E2E_MODELS=1",
    ),
]


def _step(name, *cmd_args, timeout):
    cmd = [sys.executable, str(WORKER), *cmd_args]
    print(f"\n{'=' * 72}\n  Step: {name}\n{'=' * 72}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        print(r.stdout[-2500:])
    if r.stderr:
        print("STDERR (tail):", r.stderr[-12000:])
    assert r.returncode == 0, (
        f"step '{name}' failed (exit {r.returncode}).\n"
        f"STDOUT:\n{r.stdout[-1500:]}\nSTDERR:\n{r.stderr[-4000:]}"
    )


def test_preserve_reuses_the_adapter_history():
    external = os.environ.get("GRANITE_SWITCH_ALORA_DIR")

    with tempfile.TemporaryDirectory() as wd:
        model_dir = external or str(Path(wd) / "conv_model")
        out_json = str(Path(wd) / "conv.json")

        if not external:
            _step(
                "build (compose aLoRA multi)",
                "build",
                "--output-dir",
                model_dir,
                timeout=BUILD_TIMEOUT,
            )
        _step(
            "serve (prefix caching ENABLED, both policies)",
            "serve",
            "--model-path",
            model_dir,
            "--output-path",
            out_json,
            timeout=SERVE_TIMEOUT,
        )

        with open(out_json) as f:
            data = json.load(f)

    assert not data.get("error"), data["error"]

    print("\n" + "=" * 66)
    print("CHECKPOINT")
    print("=" * 66)
    print(f"  num_adapters  : {data['num_adapters']}")
    print(f"  aLoRA adapters: {data['alora_names']}")
    print(f"  chosen        : {data.get('chosen')}")

    hdr = (
        f"\n  {'policy':24s} {'turn':>4s} {'prompt':>7s} {'routed':>7s} "
        f"{'hits':>7s} {'queries':>8s} {'ids':>9s} {'sent':>5s} {'ext':>5s}  ctl@"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for policy, turns in data["policies"].items():
        for t in turns:
            print(
                f"  {policy:24s} {t['turn']:>4d} {t['prompt_len']:>7d} "
                f"{t['prompt_rows_routed']:>7d} {t['hits_delta']:>7.0f} "
                f"{t['queries_delta']:>8.0f} {t.get('answer_ids_source', '?'):>9s} "
                f"{t.get('sent_len', '-')!s:>5s} "
                f"{t.get('sent_extends_prompt', '-')!s:>5s}"
                f"  {t['control_token_positions']}"
            )

    dhdr = (
        f"\n  {'policy':24s} {'turn':>4s} {'want':>5s} {'rows':>5s} "
        f"{'decode idx':>12s}  decode n"
    )
    print(dhdr)
    print("  " + "-" * (len(dhdr) - 3))
    for policy, turns in data["policies"].items():
        for t in turns:
            print(
                f"  {policy:24s} {t['turn']:>4d} "
                f"{t.get('expected_adapter_index', -1):>5d} "
                f"{t.get('decode_rows', 0):>5d} "
                f"{t.get('decode_adapter_indices', [])!s:>12s}"
                f"  {t.get('decode_write_addresses', [])}"
            )

    # ── Anti-vacuity ────────────────────────────────────────────────────────
    assert len(data["alora_names"]) >= 2, (
        f"need >=2 aLoRA adapters for a two-adapter conversation; got "
        f"{data['alora_names']}. Without them PRESERVE and RE_PREFILL send the "
        "same thing and every assertion below is vacuous."
    )

    preserve = data["policies"]["preserve_mixed_history"]
    reprefill = data["policies"]["re_prefill"]
    p_turn2, r_turn2 = preserve[-1], reprefill[-1]

    assert p_turn2["prompt_rows_routed"] > 0, (
        "the in-server trace hook never fired for turn 2, so nothing was measured"
    )
    assert p_turn2["answer_ids_source"] == "logprobs", (
        "the answer was re-encoded from text rather than taken as ids. "
        "encode(detokenize(ids)) does not always reproduce ids, and one wrong "
        "token in the answer makes the block straddling the prompt/answer seam "
        "miss, capping reuse at the last boundary before the answer -- exactly "
        "the saving PRESERVE exists to produce. Needs a vLLM build supporting "
        "--return-tokens-as-token-ids."
    )
    assert p_turn2["sent_extends_prompt"] is not False, (
        "the transcript does not begin with what was actually sent, so the cache "
        "cannot match it and no reuse is possible by construction"
    )
    assert p_turn2["queries_delta"] > 0, (
        "vLLM reported zero prefix-cache queries, so caching was not actually on "
        "and the comparison below measures nothing"
    )

    # ── Decode: which adapter actually generated the answer ─────────────────
    # Prompt positions say how much was recomputed; they say nothing about the
    # tokens the user reads. Those come from decode rows, and under PRESERVE the
    # tail sits at a different codebook address than under RE_PREFILL (two control
    # tokens vs one), so the address must move while the adapter must not.
    for label, turn2 in (("preserve", p_turn2), ("re_prefill", r_turn2)):
        assert turn2["decode_rows"] > 0, (
            f"{label}: no decode rows were traced for turn 2, so 'the answer was "
            "generated by the adapter we asked for' is untested -- the assertion "
            "below would pass on an empty set"
        )
        want = turn2["expected_adapter_index"]
        assert turn2["decode_adapter_indices"] == [want], (
            f"{label}: turn 2's answer was generated with adapter index(es) "
            f"{turn2['decode_adapter_indices']}, wanted only {want} "
            f"({turn2['adapter']!r}). Every generated token must route to the "
            "adapter the caller asked for; a wrong index here means the user read "
            "output from another adapter, which no prompt-side measurement can see."
        )

    p_addr = p_turn2["decode_write_addresses"]
    r_addr = r_turn2["decode_write_addresses"]
    print(f"\n  decode write address   PRESERVE {p_addr}   RE_PREFILL {r_addr}")
    # Not `if p_addr and r_addr:`. The addresses come from MultiSwitch's
    # _debug_write_addresses, which is written under a
    # `not torch.compiler.is_compiling()` guard -- so it exists only when the server
    # runs eager, and the compiled graph omits it entirely. Skipping the assertion on
    # empty input made the one check of the counting MECHANISM (as opposed to its
    # outcome) vanish the moment someone dropped --enforce-eager, and pass.
    assert p_addr and r_addr, (
        f"no decode write addresses were traced (PRESERVE {p_addr}, RE_PREFILL "
        f"{r_addr}). MultiSwitch only records them outside torch.compile, so this "
        "means the server was not started with --enforce-eager -- the count-carry "
        "assertion below cannot run, and must not be skipped quietly."
    )
    assert min(p_addr) > min(r_addr), (
        f"PRESERVE turn 2 carries two control tokens and RE_PREFILL one, so "
        f"the recovered write address must be higher under PRESERVE; got "
        f"{p_addr} vs {r_addr}. Equal addresses mean the retained control "
        "token was not counted, which is the L1 count-carry failure: routing "
        "would happen to be right here while the mechanism behind it is not."
    )

    # ── The routing difference the policies exist for ───────────────────────
    assert len(p_turn2["control_token_positions"]) == 2, (
        f"PRESERVE turn 2 should carry both control tokens, got "
        f"{p_turn2['control_token_positions']}"
    )
    assert len(r_turn2["control_token_positions"]) == 1, (
        f"RE_PREFILL turn 2 should carry only the current turn's control token, got "
        f"{r_turn2['control_token_positions']}"
    )

    # ── The economic claim ──────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("REUSE")
    print("=" * 66)
    print(
        f"  turn-2 positions recomputed  PRESERVE {p_turn2['prompt_rows_routed']:>5d}"
        f"   RE_PREFILL {r_turn2['prompt_rows_routed']:>5d}"
    )
    print(
        f"  turn-2 prefix-cache hits     PRESERVE {p_turn2['hits_delta']:>5.0f}"
        f"   RE_PREFILL {r_turn2['hits_delta']:>5.0f}"
    )

    assert p_turn2["prompt_rows_routed"] < r_turn2["prompt_rows_routed"], (
        "PRESERVE recomputed at least as many turn-2 positions as RE_PREFILL "
        f"({p_turn2['prompt_rows_routed']} vs {r_turn2['prompt_rows_routed']}). "
        "Keeping the ids is supposed to make turn 1's blocks reusable; if it does "
        "not, the policy costs a control token and buys nothing."
    )
    assert p_turn2["prompt_rows_routed"] < p_turn2["prompt_len"], (
        f"PRESERVE routed {p_turn2['prompt_rows_routed']} of {p_turn2['prompt_len']} "
        "prompt positions, i.e. nothing was served from cache at all"
    )

    # ── The same claim, to the block ─────────────────────────────────────────
    # The inequality above is satisfied by "PRESERVE reused something", including
    # by reuse that stops a whole block early -- which is precisely the failure
    # this file has already seen once (the answer re-encoded from text, capping
    # reuse at the boundary before the answer). So predict each arm's hit count
    # from the block arithmetic and check the number, not the direction.
    BLOCK = 16  # vLLM's default block_size; launch_vllm passes no override

    p_turn1, r_turn1 = preserve[0], reprefill[0]

    # Equal-cost openings ("Case A. " / "Case B. ", see CASE_MARKER in the worker).
    # Turn 1 is otherwise identical in both arms, so a length difference here means
    # the arms' absolute counts are not comparable and the contrast below is
    # measuring the marker as much as the policy.
    assert p_turn1["prompt_len"] == r_turn1["prompt_len"], (
        f"turn-1 prompts differ in length between the arms "
        f"({p_turn1['prompt_len']} vs {r_turn1['prompt_len']}), so the two "
        "policies were not given the same opening"
    )

    # A prompt position is either served from cache or routed by the switch, never
    # both and never neither. Holds whenever the turn has new content -- a fully
    # cached prompt still recomputes its last token, and there is none here.
    for label, t in (("preserve", p_turn2), ("re_prefill", r_turn2)):
        assert t["hits_delta"] + t["prompt_rows_routed"] == t["prompt_len"], (
            f"{label}: {t['hits_delta']:.0f} cached + {t['prompt_rows_routed']} "
            f"routed != {t['prompt_len']} prompt positions. The two measurements "
            "are of the same quantity from opposite sides; if they disagree, one "
            "of them is not counting what the numbers below assume."
        )

    # PRESERVE reuses up to the last whole block turn 1 left resident. Turn 1 put
    # prompt_len + answer_len - 1 positions in the cache: every prompt position,
    # plus every generated token except the last, whose KV is never computed
    # because it is never fed back in as an input. The transcript is longer than
    # that by the turn-end tokens, which were appended locally and never sent, so
    # the prediction is a two-sided bound that collapses to one value unless a
    # block boundary falls between them.
    assert p_turn1["answer_ids_source"] == "logprobs", (
        "turn 1's answer was re-encoded from text, so what the transcript claims "
        "is cached is not what the server computed, and the bound below is unsound"
    )
    computed1 = p_turn1["prompt_len"] + p_turn1["answer_len"] - 1
    lo = (computed1 // BLOCK) * BLOCK
    hi = (p_turn1["sent_len"] // BLOCK) * BLOCK
    print(
        f"  PRESERVE predicted hits      {lo if lo == hi else f'{lo}..{hi}'}"
        f"  (turn 1 computed {computed1}, transcript {p_turn1['sent_len']})"
    )
    assert lo <= p_turn2["hits_delta"] <= hi, (
        f"PRESERVE turn 2 reused {p_turn2['hits_delta']:.0f} positions; the block "
        f"arithmetic says {lo} (turn 1 left {computed1} positions cached, "
        f"transcript {p_turn1['sent_len']}). Below it means reuse stopped at an "
        "earlier boundary than the ids allow -- the transcript and the cache "
        "disagree somewhere inside turn 1. Above it means the counter is not "
        "measuring this request."
    )

    # RE_PREFILL diverges from the cache at turn 1's control token: the re-render
    # drops it, so every position from there on shifts. Reuse therefore ends at
    # the last whole block before it -- the structural reason the two policies
    # differ, expressed as a number rather than as "less".
    assert len(r_turn1["control_token_positions"]) == 1, (
        f"RE_PREFILL turn 1 should carry exactly one control token, got "
        f"{r_turn1['control_token_positions']}"
    )
    ctl1 = r_turn1["control_token_positions"][0]
    assert r_turn2["hits_delta"] == (ctl1 // BLOCK) * BLOCK, (
        f"RE_PREFILL turn 2 reused {r_turn2['hits_delta']:.0f} positions; turn 1's "
        f"control token sat at index {ctl1}, so the re-render's first difference is "
        f"there and reuse must end at {(ctl1 // BLOCK) * BLOCK}. A different number "
        "means the divergence point is not where dropping the control token puts "
        "it, and the PRESERVE/RE_PREFILL gap is not the one being explained."
    )

    print("\nMEASUREMENT COMPLETE.")
