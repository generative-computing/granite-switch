# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: verify generation equivalence between upstream and zero-adapter switch models.

Three modes, each invoked as a separate subprocess so only one vLLM model is
ever resident on GPU at a time::

    python worker.py build   --model <name> --work-dir <dir>
    python worker.py run     --model <name-or-path> --work-dir <dir> --tag <tag>
    python worker.py compare --work-dir <dir> --label <model_name>

A GraniteSwitch model with a single ZERO-weight built-in adapter must be
equivalent to the upstream base (zero LoRA delta => it *is* the base). We check
that as DISTRIBUTION equivalence, not exact greedy-token equality:

  build   Deterministic 64-token prompt; build the zero-adapter switch model.
  run     Load in vLLM and capture per-position top-k next-token distributions
          over ``[prompt + continuation]`` (teacher-forced ``prompt_logprobs``),
          where the continuation is the REFERENCE model's greedy output — so both
          models are scored on the SAME token sequence. This covers the prompt
          positions AND the generation-decision positions (including the first
          generated token). The model's own greedy output is also captured, for
          informational divergence reporting only.
  compare Gate: mean JSD(bits) <= MEAN_JSD_THRESH AND per-position max JSD <=
          MAX_JSD_THRESH AND mean top-k Jaccard-distance <= JACC_THRESH, swept
          over k. JSD is primary; the max-JSD gate stops one catastrophically
          divergent position from being averaged away; Jaccard is a loose guard.

Why distribution equivalence rather than greedy token match: the SWITCH kernel's
fused projections use a different float reduction order than vLLM's native linear
(not bit-exact by design). On a near-tie position that tiny difference can flip
the greedy argmax, and strict token-for-token comparison then amplifies the single
flip into a fully divergent sequence — a false failure sensitive to the vLLM
version's numerics (it passed under 0.19 but flipped the first token under 0.20.2,
where the top candidates were within ~0.06 logprob). Gating probability-mass
agreement is robust to ties while the max-JSD + tightened mean gates still catch
real logit/weight regressions (a localized corruption, or a systematic scaling/bias).

Thresholds are calibrated to the OBSERVED fused-vs-native noise floor on this
comparison (same GPU, TP=1, zero delta), measured on granite-4.0-micro under vLLM
0.19 and 0.20.2; see the constants below for the recorded margins.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig

# Make tests.shared importable when run as a bare subprocess (cwd-independent).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests.shared.logit_metrics import captured_mass, jaccard, jsd_bits, topk_ids

# Top-k logprobs captured per position. Kept >= 2*max(K_SWEEP) so every swept
# top-k UNION is fully captured on BOTH sides — otherwise a union member missing
# from one side's capture is scored as prob 0 and inflates JSD spuriously.
TOPK = int(os.environ.get("GEN_EQUIV_TOPK", "64"))
K_SWEEP = [int(x) for x in os.environ.get("GEN_EQUIV_KS", "1,5,10,20").split(",")]
assert max(K_SWEEP) <= TOPK, f"max(K_SWEEP)={max(K_SWEEP)} must be <= TOPK={TOPK}"
GEN_TOKENS = int(os.environ.get("GEN_EQUIV_GEN_TOKENS", "32"))

# Gates. Calibrated to the MEASURED fused-vs-native floor on granite-4.0-micro,
# both vLLM 0.19.1 and 0.20.2 (95 positions = 63 prompt + 32 generation): mean
# JSD <= 5e-4, per-position max JSD <= 3.5e-3, mean 1-Jaccard <= 0.08. Thresholds
# sit ~6-15x above that floor so benign tie-flips pass (BOTH versions PASS), while
# a localized corruption (JSD up to 1.0 bit at one position) or a systematic ~30%
# logit scaling (mean JSD ~0.012) is caught -- verified locally against both.
MEAN_JSD_THRESH = float(os.environ.get("GEN_EQUIV_MEAN_JSD_THRESH", "0.003"))
MAX_JSD_THRESH = float(os.environ.get("GEN_EQUIV_MAX_JSD_THRESH", "0.05"))
JACC_THRESH = float(os.environ.get("GEN_EQUIV_JACC_THRESH", "0.20"))


def _native_dtype(config):
    """Determine the model's native dtype from its HuggingFace config."""
    dt = getattr(config, "torch_dtype", None)
    if dt is None:
        return torch.float32
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        return getattr(torch, dt, torch.float32)
    return torch.float32


def _dtype_str(dtype):
    """Convert torch.dtype to vLLM dtype string."""
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, "auto")


# ── build mode ────────────────────────────────────────────────────


def cmd_build(args):
    """Build a GraniteSwitch model with 1 zero-weight built-in adapter."""
    from granite_switch.composer import GraniteSwitchComposer

    model_name = args.model
    work_dir = args.work_dir

    print(f"Loading config for {model_name}...")
    base_config = AutoConfig.from_pretrained(model_name)
    dtype = _native_dtype(base_config)
    vocab_size = base_config.vocab_size
    print(
        f"  model_type={base_config.model_type}  native_dtype={dtype}  vocab_size={vocab_size}"
    )

    # Deterministic prompt (no control tokens — all IDs in [1, 1000))
    torch.manual_seed(42)
    max_tok = min(vocab_size, 1000)
    prompt_ids = torch.randint(1, max_tok, (64,)).tolist()

    # Adapter token IDs placed far from the prompt range
    adapter_token_id = vocab_size - 100

    inputs_path = os.path.join(work_dir, "inputs.json")
    with open(inputs_path, "w") as f:
        json.dump(
            {
                "prompt_ids": prompt_ids,
                "adapter_token_id": adapter_token_id,
                "vocab_size": vocab_size,
            },
            f,
        )
    print(f"  saved inputs to {inputs_path}")

    print("\nBuilding GraniteSwitch (1 built-in adapter)...")
    skin_dir = os.path.join(work_dir, "switch")
    model = GraniteSwitchComposer.from_base_and_adapters(
        model_name,
        built_in_adapter_names=["test"],
        adapter_names=["test"],
        adapter_token_ids=[adapter_token_id],
        adapter_substitute_token_ids=[1],
        torch_dtype=dtype,
    )

    print("  zeroing all LoRA weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.zero_()
                print(f"    zeroed {name} {tuple(param.shape)}")

    print(f"  saving switch model to {skin_dir}...")
    model.save_pretrained(skin_dir)
    del model
    print("  build complete")
    return 0


# ── run mode ──────────────────────────────────────────────────────


def _dists_over(llm, seq):
    """Teacher-forced per-position top-k distributions over token sequence `seq`."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    out = llm.generate(
        TokensPrompt(prompt_token_ids=list(seq)),
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK),
    )
    pl = out[0].prompt_logprobs or []
    return [
        None if d is None else {str(int(t)): float(v.logprob) for t, v in d.items()}
        for d in pl
    ]


def cmd_run(args):
    """Load a model in vLLM; capture teacher-forced per-position distributions."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    work_dir, tag = args.work_dir, args.tag
    with open(os.path.join(work_dir, "inputs.json")) as f:
        prompt_ids = json.load(f)["prompt_ids"]

    dtype_s = _dtype_str(_native_dtype(AutoConfig.from_pretrained(args.model)))
    print(f"Creating vLLM LLM for {args.model} (dtype={dtype_s})...")
    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        dtype=dtype_s,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_logprobs=TOPK,
    )

    greedy_sp = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, ignore_eos=True)

    # The teacher-forcing continuation is the REFERENCE model's greedy output, so
    # both models are scored on the SAME [prompt + continuation] sequence. The
    # reference run produces it; the switch run reads it back from ref.json.
    if tag == "ref":
        g = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), greedy_sp)
        cont_ids = list(g[0].outputs[0].token_ids)
        own_greedy = cont_ids
    else:
        with open(os.path.join(work_dir, "ref.json")) as f:
            cont_ids = json.load(f)["cont_ids"]
        g = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), greedy_sp)
        own_greedy = list(g[0].outputs[0].token_ids)

    seq = list(prompt_ids) + list(cont_ids)
    dists = _dists_over(llm, seq)

    n_pos = sum(d is not None for d in dists)
    print(
        f"  {tag}: {n_pos} teacher-forced dists over {len(seq)} tokens "
        f"({len(prompt_ids)} prompt + {len(cont_ids)} continuation); greedy[:10]={own_greedy[:10]}"
    )
    with open(os.path.join(work_dir, f"{tag}.json"), "w") as f:
        json.dump(
            {
                "cont_ids": cont_ids,
                "n_prompt": len(prompt_ids),
                "dists": dists,
                "greedy": own_greedy,
            },
            f,
        )
    del llm
    return 0


# ── compare mode ──────────────────────────────────────────────────


def cmd_compare(args):
    """Gate distribution equivalence (mean + max JSD, mean Jaccard) over all positions."""
    ref = json.load(open(os.path.join(args.work_dir, "ref.json")))
    sw = json.load(open(os.path.join(args.work_dir, "switch.json")))
    R, C = ref["dists"], sw["dists"]
    label = args.label
    n_prompt = ref["n_prompt"]

    if len(R) != len(C):
        print(
            f"\nFAIL: {label} — position count mismatch: ref={len(R)}, switch={len(C)}"
        )
        return 1

    idx = [i for i in range(len(R)) if R[i] and C[i]]
    if not idx:
        print(f"\nFAIL: {label} — no comparable positions with logprobs")
        return 1
    n_cont = sum(1 for i in idx if i >= n_prompt)
    print(
        f"\nGEN-EQUIVALENCE (distribution) {label}  positions={len(idx)} "
        f"({len(idx) - n_cont} prompt + {n_cont} generation)"
    )
    print(f"  {'k':>4}{'mean(1-Jacc)':>15}{'mean JSD':>12}{'max JSD':>12}{'@pos':>7}")

    failures = []
    for k in K_SWEEP:
        jds, jss = [], []
        for i in idx:
            ids = list(set(topk_ids(R[i], k)) | set(topk_ids(C[i], k)))
            jds.append(1.0 - jaccard(topk_ids(R[i], k), topk_ids(C[i], k)))
            jss.append(jsd_bits(R[i], C[i], ids))
        mean_jd = sum(jds) / len(jds)
        mean_js = sum(jss) / len(jss)
        argmax = max(range(len(jss)), key=lambda j: jss[j])
        max_js, max_pos = jss[argmax], idx[argmax]
        print(f"  {k:>4}{mean_jd:>15.6f}{mean_js:>12.6f}{max_js:>12.6f}{max_pos:>7}")
        if mean_jd > JACC_THRESH:
            failures.append(f"k={k}: mean(1-Jaccard)={mean_jd:.4f} > {JACC_THRESH}")
        if mean_js > MEAN_JSD_THRESH:
            failures.append(f"k={k}: mean JSD={mean_js:.6f} > {MEAN_JSD_THRESH}")
        if max_js > MAX_JSD_THRESH:
            failures.append(
                f"k={k}: max JSD={max_js:.6f} @pos {max_pos} > {MAX_JSD_THRESH}"
            )

    # Informational: captured top-k mass agreement (tail-redistribution signal) and
    # greedy-token divergence (the brittle signal the old token-equality gated on).
    mass_diff = sum(abs(captured_mass(R[i]) - captured_mass(C[i])) for i in idx) / len(
        idx
    )
    rg, cg = ref.get("greedy", []), sw.get("greedy", [])
    gdiv = next((i for i, (a, b) in enumerate(zip(rg, cg)) if a != b), None)
    print(f"  mean |captured top-{TOPK} mass diff| = {mass_diff:.6f}  (informational)")
    print(f"  greedy first-divergence step: {gdiv}  (informational, not gated)")
    print(f"    ref greedy[:8]   = {rg[:8]}")
    print(f"    switch greedy[:8]= {cg[:8]}")

    if failures:
        print(
            f"\nFAIL: {label} — distributions diverged beyond the fused-vs-native floor:\n  "
            + "\n  ".join(failures)
        )
        return 1
    print(
        f"\nPASS: {label} — distribution equivalence over {len(idx)} positions "
        f"(mean JSD <= {MEAN_JSD_THRESH}, max JSD <= {MAX_JSD_THRESH}, mean 1-Jaccard <= {JACC_THRESH})"
    )
    return 0


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_build = sub.add_parser("build", help="Build switch model and save inputs")
    p_build.add_argument(
        "--model", required=True, help="HuggingFace model name or path"
    )
    p_build.add_argument(
        "--work-dir", required=True, help="Working directory for outputs"
    )

    p_run = sub.add_parser("run", help="Load model in vLLM, capture distributions")
    p_run.add_argument("--model", required=True, help="Model name or path to load")
    p_run.add_argument(
        "--work-dir", required=True, help="Working directory with inputs.json"
    )
    p_run.add_argument("--tag", required=True, help="Output tag (ref or switch)")

    p_compare = sub.add_parser("compare", help="Gate distribution equivalence")
    p_compare.add_argument(
        "--work-dir", required=True, help="Working dir with ref.json and switch.json"
    )
    p_compare.add_argument("--label", required=True, help="Model label for output")

    args = parser.parse_args()
    if args.mode == "build":
        return cmd_build(args)
    elif args.mode == "run":
        return cmd_run(args)
    elif args.mode == "compare":
        return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
