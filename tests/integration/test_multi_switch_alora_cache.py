# SPDX-License-Identifier: Apache-2.0
"""aLoRA + prefix caching: does a previous turn's adapter contaminate the next turn?

Nothing else in the suite covers this, and it is the real aLoRA serving path.

The question is well-founded rather than hypothetical:

* ``grep`` finds NO prefix-cache invalidation, re-prefill, or cache-salt logic anywhere
  in ``src/granite_switch/``, and ``MultiSwitch.forward`` takes only
  ``(input_ids, adapter_token_ids, positions)`` -- it cannot see that a cache hit
  occurred, how much was reused, or which adapter produced the reused entries.
* aLoRA places its control token LATE (just before the invocation sequence), precisely
  so the conversation history prefills with BASE weights and is reusable. That is what
  makes aLoRA cheap -- and it also guarantees a turn-2 request shares a long prefix with
  turn 1 and WILL hit the cache.
* LoRA places its control token at position 0, so a different adapter changes the prefix
  immediately and misses. Safe, but by accident, not design.

Measured leak paths:
  L1 COUNT   turn 1's control token remains in the reused KV, so the coded counting head
             may include it and shift turn 2's recovered ``n`` (hence every codebook
             address).
  L2 ROUTING does turn 2 route to the adapter it asked for, or to turn 1's?
  L3 OUTPUT  does a warm turn 2 differ from the same request served cold? That is the
             only leak a user can see.

Prefix caching is deliberately ENABLED (every other MultiSwitch test disables it),
because reuse is the condition under test. A/B and A/A turn pairs are both run, so
"contaminated" is distinguishable from "same adapter anyway".

This test is CHARACTERIZATION. Routing correctness at the turn-2 tail is asserted; the
output-level contamination is reported, because whether it is acceptable is a product
decision (an adapter's own earlier output arguably IS legitimate context), not a
correctness question this test can settle.

Markers: slow + requires_model + gpu, gated on ``GRANITE_SWITCH_E2E_MODELS=1``.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WORKER = Path(__file__).parent / "_multi_switch_alora_cache_worker.py"
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
        print("STDERR (tail):", r.stderr[-14000:])
    assert r.returncode == 0, (
        f"step '{name}' failed (exit {r.returncode}).\n"
        f"STDOUT:\n{r.stdout[-1500:]}\nSTDERR:\n{r.stderr[-4000:]}"
    )


def test_multi_switch_alora_prefix_cache():
    ext = os.environ.get("GRANITE_SWITCH_ALORA_DIR")

    with tempfile.TemporaryDirectory() as wd:
        model_dir = ext or str(Path(wd) / "alora_model")
        keys_json = str(Path(wd) / "keys.json")
        out_json = str(Path(wd) / "alora.json")

        # ── Paper check first: what does vLLM hash a cache block on? ─────────
        _step(
            "inspect vLLM cache-key construction",
            "inspect-keys",
            "--output-path",
            keys_json,
            timeout=600,
        )
        with open(keys_json) as f:
            keys = json.load(f)

        print("\n" + "=" * 66)
        print("PART 0 — what vLLM hashes a cache block on (paper check)")
        print("=" * 66)
        if keys.get("error"):
            print(f"  could not read: {keys['error']}")
        else:
            print(f"  module: {keys.get('module_file')}")
            for fn in (
                "hash_block_tokens",
                "generate_block_hash_extra_keys",
                "need_extra_keys",
            ):
                src = keys.get(fn)
                if src:
                    print(f"\n  --- {fn} ---")
                    for line in src.splitlines()[:28]:
                        print(f"    {line}")

        if not ext:
            _step(
                "build (compose MIXED alora+lora)",
                "build",
                "--output-dir",
                model_dir,
                timeout=BUILD_TIMEOUT,
            )
        _step(
            "serve (prefix caching ENABLED, 2-turn session)",
            "serve",
            "--model-path",
            model_dir,
            "--output-path",
            out_json,
            timeout=SERVE_TIMEOUT,
        )

        with open(out_json) as f:
            d = json.load(f)

        sc = d["scenarios"]
        chosen = d.get("chosen", {})

        print("\n" + "=" * 66)
        print("CHECKPOINT — is it really mixed aLoRA + LoRA?")
        print("=" * 66)
        techs = d.get("adapter_technologies") or []
        n_alora = sum(1 for t in techs if t == "alora")
        print(f"  num_adapters   : {d.get('num_adapters')}")
        print(f"  technologies   : {techs}")
        print(f"  aLoRA count    : {n_alora}")
        print(f"  chosen A/B     : {chosen}")

        print("\n" + "=" * 66)
        print("PART 1/2 — routing + recovered n at the turn-2 tail")
        print("=" * 66)
        hdr = (
            f"\n{'scenario':16s} {'want':>5s} {'ctl@':>6s} {'attr':>6s} "
            f"{'tail':>5s}  {'tail adapters':16s} {'tail n':16s} {'wrong':10s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for name in ("turn1_A", "turn2_B_warm", "turn2_B_cold", "turn2_A_warm"):
            s = sc.get(name)
            if not s:
                continue
            print(
                f"{name:16s} {s['want_adapter']:>5d} {s['control_pos']:>6d} "
                f"{s['attributed_total']:>6d} {s['attributed_tail']:>5d}  "
                f"{s['tail_adapters']!s:16s} {s['tail_n']!s:16s} "
                f"{s['tail_wrong']!s:10s}"
            )

        print("\n" + "=" * 66)
        print("PART 3 — output-level contamination (what a user sees)")
        print("=" * 66)
        cont = d.get("contamination", {})
        print(
            f"  warm-B text == cold-B text : {cont.get('warm_B_vs_cold_B_text_same')}"
        )
        print(
            f"  warm-B text == warm-A text : {cont.get('warm_B_vs_warm_A_text_same')}"
        )
        for k, v in (d.get("texts") or {}).items():
            print(f"    {k:16s}: {str(v)[:90]!r}")
        if d.get("server_prefix_cache_lines"):
            print("\n  server prefix-cache log lines:")
            for ln in d["server_prefix_cache_lines"]:
                print(f"    {ln.strip()[:120]}")

        # ── Interpretation ──────────────────────────────────────────────────
        print("\n" + "=" * 66)
        print("INTERPRETATION")
        print("=" * 66)
        warm = sc.get("turn2_B_warm") or {}
        cold = sc.get("turn2_B_cold") or {}
        n_shift = warm.get("tail_n") != cold.get("tail_n")
        if warm.get("tail_wrong"):
            print(
                "  L2 ROUTING LEAK: warm turn-2 routed the WRONG adapter at "
                f"{len(warm['tail_wrong'])} tail positions -> a previous turn's "
                "control token in the reused KV is changing this turn's routing."
            )
        else:
            print("  L2 routing at the turn-2 tail is CORRECT even with a warm cache.")
        if n_shift:
            print(
                f"  L1 COUNT CARRY: recovered n differs warm {warm.get('tail_n')} vs "
                f"cold {cold.get('tail_n')} -- turn 1's control token is still being "
                "counted. Harmless only while it does not cross an adapter boundary."
            )
        else:
            print("  L1 recovered n is the same warm vs cold.")
        if cont.get("warm_B_vs_cold_B_text_same") is False:
            print(
                "  L3 OUTPUT differs warm vs cold. Expected to some degree: the two "
                "requests genuinely have different history. Judge it against the "
                "routing rows above -- identical routing plus different text means "
                "context effect, not mis-routing."
            )

        # ── Assertions ──────────────────────────────────────────────────────
        assert n_alora > 0, (
            f"no aLoRA adapters detected (technologies={techs}); the aLoRA half of this "
            "test would be vacuous.\n"
            "NOTE: technology is INFERRED from adapter_substitute_token_ids, because "
            "the composer does not persist an adapter_technologies field -- it picks "
            "alora>lora at compose time and keeps only the consequence. A LoRA "
            "adapter's substitute is the chat template's start-of-turn token (same for "
            "all of them); an aLoRA adapter's substitute is the first token of ITS "
            "invocation sequence, so it differs. If every substitute is identical, the "
            "compose really did resolve everything to LoRA."
        )
        assert warm.get("attributed_tail", 0) > 0, (
            f"no turn-2 tail positions attributable ({warm}) -- cannot judge routing."
        )
        # Routing is the switch's contract and must hold regardless of cache reuse.
        assert not warm.get("tail_wrong"), (
            "with prefix caching ENABLED, turn 2 routed to the wrong adapter at "
            f"positions {warm['tail_wrong']} (wanted {warm['want_adapter']}, saw "
            f"{warm['tail_adapters']}) -- a previous turn's adapter is leaking through "
            "the reused KV cache."
        )
        print("\nCHARACTERIZATION COMPLETE.")
