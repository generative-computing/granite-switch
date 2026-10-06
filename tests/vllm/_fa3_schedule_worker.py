# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: FA3 AOT-schedule correctness under FULL CUDA graphs + prefix caching.

Reproduces and gates the bug tracked in
``docs/FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md`` / issue #139: on vLLM 0.26, a
Shadow-Residual (SR) MultiSwitch checkpoint served with
``cudagraph_mode=FULL`` AND prefix caching ON emits tokens outside the
adapter's allowed set and distributions that diverge from the reference — but
only on small, prefix-cached decode steps. Turning off EITHER FULL or prefix
caching matches the reference exactly.

Root cause (summary): FA3's ahead-of-time schedule is sized from the model
config's attention shape, but SR's decoder doubles its query heads
(``Attention(2 * num_heads, ...)``), so ``num_heads_q`` is half the real value
for that group. Under FULL the mis-sized schedule is frozen into the captured
graph; a 1-token prefix-cached decode is where it goes wrong. See the doc for
the full trace.

Each mode is a separate subprocess so only one vLLM model is ever resident on
GPU at a time::

    python worker.py build   --work-dir <dir>
    python worker.py run     --work-dir <dir> --tag <tag> --cudagraph <mode> --prefix <on|off>
    python worker.py compare --work-dir <dir> --ref <tag> --cand <tag> --label <label>

``build`` composes a small SR MultiSwitch checkpoint on CPU (tiny random base,
no download) through ``GraniteSwitchComposer`` — model construction must go
through the composer (CLAUDE.md gotcha #5). ``run`` loads it in vLLM under one
(cudagraph_mode, prefix-caching) setting and captures (a) the greedy decode
token ids over a growing prefix-cached history and (b) teacher-forced
per-position top-k distributions. ``compare`` gates a candidate run against a
reference run: every decoded token must be in the adapter's allowed set, and
the distributions must match within the fused-vs-native floor.

The FULL-cudagraph run uses real CUDA graphs (NOT enforce_eager), so the
MultiSwitch eager-only debug attributes do not exist (CLAUDE.md gotcha #11);
this worker asserts on generated tokens and logprobs, never on ``_debug_*``.
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

# Make tests.shared importable when run as a bare subprocess (cwd-independent).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests.shared.logit_metrics import jaccard, jsd_bits, topk_ids

# ── small SR MultiSwitch checkpoint geometry ──────────────────────────────
# Kept tiny so build is a fast CPU step with no download. The FA3 bug is about
# attention SHAPE (SR's doubled query heads), not model quality, so a random
# base reproduces it. head_dim is >= 32 so FlashAttention accepts it, and
# large enough that FA3 (not a fallback) is selected on GPU.
VOCAB_SIZE = 320
HIDDEN = 256
INTERMEDIATE = 512
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 64
LORA_RANK = 4
CROSS_STREAM_RANK = 8

# Two SR adapters, each with its own control token near the top of the vocab.
ADAPTER_TOKEN_IDS = [VOCAB_SIZE - 10, VOCAB_SIZE - 9]
ADAPTER_SUBSTITUTE_TOKEN_IDS = [1, 2]
ADAPTER_NAMES = ["sr_a", "sr_b"]

# Distribution-equivalence gates. Same shape as the generation-equivalence
# worker; a wrong AOT schedule produces a gross divergence (and out-of-set
# tokens), far above these, so the floor-calibrated values are sufficient.
TOPK = int(os.environ.get("FA3_TOPK", "64"))
K_SWEEP = [int(x) for x in os.environ.get("FA3_KS", "1,5,10,20").split(",")]
assert max(K_SWEEP) <= TOPK, f"max(K_SWEEP)={max(K_SWEEP)} must be <= TOPK={TOPK}"
MEAN_JSD_THRESH = float(os.environ.get("FA3_MEAN_JSD_THRESH", "0.003"))
MAX_JSD_THRESH = float(os.environ.get("FA3_MAX_JSD_THRESH", "0.05"))
JACC_THRESH = float(os.environ.get("FA3_JACC_THRESH", "0.20"))

# Growing history: a shared prefix (so prefix caching actually caches) followed
# by a short per-adapter suffix, then one decode step per batch. This is the
# "one game served alone on a growing history" regime from the issue.
GEN_TOKENS = int(os.environ.get("FA3_GEN_TOKENS", "16"))


# ── build mode (CPU) ───────────────────────────────────────────────────────


def _write_base_model(path):
    """Write a tiny Granite base checkpoint on disk (random weights, no download)."""
    base_cfg = {
        "model_type": "granite",
        "architectures": ["GraniteForCausalLM"],
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "attention_multiplier": 1.0,
        "logits_scaling": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "torch_dtype": "bfloat16",
        "head_dim": HEAD_DIM,
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(base_cfg, f)

    torch.manual_seed(0)
    sd = {"model.embed_tokens.weight": torch.randn(VOCAB_SIZE, HIDDEN)}
    qkv_out = NUM_HEADS * HEAD_DIM
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(qkv_out, HIDDEN)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(HIDDEN, qkv_out)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(HIDDEN, INTERMEDIATE)
        sd[f"{p}.input_layernorm.weight"] = torch.ones(HIDDEN)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(HIDDEN)
    sd["model.norm.weight"] = torch.ones(HIDDEN)
    sd["lm_head.weight"] = torch.randn(VOCAB_SIZE, HIDDEN)
    save_file(sd, os.path.join(path, "model.safetensors"))


def _write_sr_adapter(path):
    """Write a mock SR adapter (cross_stream weights) on disk."""
    os.makedirs(path, exist_ok=True)
    config = {
        "r": LORA_RANK,
        "lora_alpha": LORA_RANK,
        "target_modules": [
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "cross_stream",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "rank_pattern": {"cross_stream": CROSS_STREAM_RANK},
        "alpha_pattern": {"cross_stream": CROSS_STREAM_RANK},
    }
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(config, f)

    torch.manual_seed(42)
    sd = {}
    prefix = "base_model.model.model."
    for i in range(NUM_LAYERS):
        lp = f"{prefix}layers.{i}"
        for mod in ["q_proj", "o_proj"]:
            sd[f"{lp}.self_attn.{mod}.lora_A.weight"] = torch.randn(LORA_RANK, HIDDEN)
            sd[f"{lp}.self_attn.{mod}.lora_B.weight"] = torch.randn(HIDDEN, LORA_RANK)
        for mod in ["gate_proj", "up_proj"]:
            sd[f"{lp}.mlp.{mod}.lora_A.weight"] = torch.randn(LORA_RANK, HIDDEN)
            sd[f"{lp}.mlp.{mod}.lora_B.weight"] = torch.randn(INTERMEDIATE, LORA_RANK)
        sd[f"{lp}.mlp.down_proj.lora_A.weight"] = torch.randn(LORA_RANK, INTERMEDIATE)
        sd[f"{lp}.mlp.down_proj.lora_B.weight"] = torch.randn(HIDDEN, LORA_RANK)
        sd[f"{lp}.cross_stream.lora_A.weight"] = torch.randn(CROSS_STREAM_RANK, HIDDEN)
        sd[f"{lp}.cross_stream.lora_B.weight"] = torch.randn(HIDDEN, CROSS_STREAM_RANK)
    save_file(sd, os.path.join(path, "adapter_model.safetensors"))


def cmd_build(args):
    """Compose a small SR MultiSwitch checkpoint through GraniteSwitchComposer."""
    from granite_switch.composer import GraniteSwitchComposer

    work_dir = args.work_dir
    base_dir = os.path.join(work_dir, "base")
    out_dir = os.path.join(work_dir, "sr_switch")

    print("Writing tiny Granite base + SR adapters (CPU, no download)...")
    _write_base_model(base_dir)
    adapter_dirs = []
    for name in ADAPTER_NAMES:
        ad = os.path.join(work_dir, f"adapter_{name}")
        _write_sr_adapter(ad)
        adapter_dirs.append(ad)

    print("Composing SR MultiSwitch via GraniteSwitchComposer...")
    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base_dir,
        adapter_paths=adapter_dirs,
        adapter_token_ids=ADAPTER_TOKEN_IDS,
        adapter_substitute_token_ids=ADAPTER_SUBSTITUTE_TOKEN_IDS,
        adapter_names=ADAPTER_NAMES,
    )
    model.save_pretrained(out_dir)

    cfg = json.loads(open(os.path.join(out_dir, "config.json")).read())
    assert cfg["dual_stream"] is True, "composed checkpoint is not SR (dual_stream)"
    assert cfg["model_type"] == "granite_switch"
    print(f"  dual_stream={cfg['dual_stream']} cross_stream_rank={cfg['cross_stream_rank']}")

    # Deterministic shared prefix (prefix-cacheable) in the plain-token range.
    # The per-adapter continuation and the "allowed set" are NOT hardcoded here:
    # they are defined by the REFERENCE run's actual output (see cmd_run), so
    # "in-set" means "matches the known-good reference", not an arbitrary range.
    torch.manual_seed(7)
    plain_max = min(VOCAB_SIZE - 20, 300)
    shared_prefix = torch.randint(3, plain_max, (48,)).tolist()
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(
            {
                "shared_prefix": shared_prefix,
                "adapter_token_ids": ADAPTER_TOKEN_IDS,
                "plain_max": plain_max,
            },
            f,
        )
    del model
    print("  build complete")
    return 0


# ── run mode (GPU) ─────────────────────────────────────────────────────────


def _make_llm(model_dir, cudagraph, prefix_on):
    """Construct the vLLM LLM under one (cudagraph_mode, prefix-caching) setting."""
    from vllm import LLM

    kwargs = dict(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enable_prefix_caching=bool(prefix_on),
        max_logprobs=TOPK,
        gpu_memory_utilization=float(os.environ.get("FA3_GPU_MEM_UTIL", "0.45")),
    )
    if cudagraph == "eager":
        kwargs["enforce_eager"] = True
    else:
        # FULL and FULL_AND_PIECEWISE are compiled CUDA-graph modes. enforce_eager
        # must stay False/unset so graphs are actually captured.
        from vllm.config import CompilationConfig, CUDAGraphMode

        mode = {
            "FULL": CUDAGraphMode.FULL,
            "FULL_AND_PIECEWISE": CUDAGraphMode.FULL_AND_PIECEWISE,
            "PIECEWISE": CUDAGraphMode.PIECEWISE,
        }[cudagraph]
        kwargs["compilation_config"] = CompilationConfig(cudagraph_mode=mode)
    return LLM(**kwargs)


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


def _probe_fa_version():
    """Log the FlashAttention version actually selected (3 => FULL cudagraph works).

    The #139 bug only exists under FA3 + FULL. FA3 requires an SM90 (Hopper) GPU
    AND a vllm_flash_attn build whose FA3 kernel loads for the installed
    torch/CUDA. When FA3 is absent, vLLM falls back to FA2, whose cudagraph
    support is UNIFORM_BATCH, so cudagraph_mode=FULL is silently downgraded to
    FULL_AND_PIECEWISE and the bug cannot reproduce. Printing this makes a moot
    run obvious instead of a misleading pass/fail.
    """
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        v = get_flash_attn_version()
        print(f"  FA_VERSION={v}  (FULL cudagraph requires FA_VERSION=3)")
        if v != 3:
            print("  WARNING: FA_VERSION != 3 — cudagraph_mode=FULL will DOWNGRADE; "
                  "this run cannot reproduce issue #139.")
    except Exception as e:  # import path or probe may move across vLLM versions
        print(f"  FA_VERSION probe failed ({type(e).__name__}: {e})")


def cmd_run(args):
    """Load the SR MultiSwitch model under one setting; capture decodes + dists.

    The REFERENCE run (tag 'ref_*') is the source of truth: it greedily generates
    each adapter's continuation and records it. Those continuation ids ARE the
    adapter's allowed set, and every CANDIDATE run is teacher-forced over the
    exact same [prefix + ctrl + reference-continuation] sequence — so distribution
    comparison is apples-to-apples and "out of allowed set" means "diverged from
    the known-good reference", not an arbitrary range.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    work_dir, tag = args.work_dir, args.tag
    is_ref = tag.startswith("ref")
    model_dir = os.path.join(work_dir, "sr_switch")
    inp = json.load(open(os.path.join(work_dir, "inputs.json")))
    shared_prefix = inp["shared_prefix"]
    adapter_token_ids = inp["adapter_token_ids"]

    print(
        f"Creating vLLM LLM tag={tag} cudagraph={args.cudagraph} "
        f"prefix_caching={args.prefix}..."
    )
    llm = _make_llm(model_dir, args.cudagraph, args.prefix == "on")
    _probe_fa_version()

    greedy = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, ignore_eos=True)

    # The reference writes the continuation; candidates read it back so every
    # run is scored over the identical token sequence.
    if is_ref:
        continuations = {}
    else:
        continuations = json.load(open(os.path.join(work_dir, "ref_cont.json")))

    decodes = {}
    dists = {}
    for ai, ctrl in enumerate(adapter_token_ids):
        seq = list(shared_prefix) + [ctrl]
        # Each adapter decodes against the shared (prefix-cached) prefix — the
        # exact "same history, different adapter suffix" regime from the issue.
        g = llm.generate(TokensPrompt(prompt_token_ids=seq), greedy)
        gen_ids = list(g[0].outputs[0].token_ids)
        decodes[str(ctrl)] = gen_ids
        cont = gen_ids if is_ref else continuations[str(ctrl)]
        if is_ref:
            continuations[str(ctrl)] = gen_ids
        # Teacher-force over [prefix + ctrl + REFERENCE continuation] in all runs.
        dists[str(ctrl)] = _dists_over(llm, seq + list(cont))
        print(f"  adapter {ADAPTER_NAMES[ai]} (ctrl={ctrl}) greedy[:8]={gen_ids[:8]}")

    if is_ref:
        with open(os.path.join(work_dir, "ref_cont.json"), "w") as f:
            json.dump(continuations, f)

    with open(os.path.join(work_dir, f"{tag}.json"), "w") as f:
        json.dump({"decodes": decodes, "dists": dists}, f)
    del llm
    return 0


# ── compare mode (CPU) ───────────────────────────────────────────────────────


def _gate_distributions(R, C, label):
    """Return a list of failure strings for the JSD/Jaccard gate (empty = pass)."""
    idx = [i for i in range(min(len(R), len(C))) if R[i] and C[i]]
    if len(R) != len(C):
        return [f"{label}: position count mismatch ref={len(R)} cand={len(C)}"]
    if not idx:
        return [f"{label}: no comparable positions with logprobs"]
    failures = []
    for k in K_SWEEP:
        jds, jss = [], []
        for i in idx:
            ids = list(set(topk_ids(R[i], k)) | set(topk_ids(C[i], k)))
            jds.append(1.0 - jaccard(topk_ids(R[i], k), topk_ids(C[i], k)))
            jss.append(jsd_bits(R[i], C[i], ids))
        mean_jd = sum(jds) / len(jds)
        mean_js = sum(jss) / len(jss)
        max_js = max(jss)
        if mean_jd > JACC_THRESH:
            failures.append(f"{label} k={k}: mean(1-Jacc)={mean_jd:.4f} > {JACC_THRESH}")
        if mean_js > MEAN_JSD_THRESH:
            failures.append(f"{label} k={k}: mean JSD={mean_js:.6f} > {MEAN_JSD_THRESH}")
        if max_js > MAX_JSD_THRESH:
            failures.append(f"{label} k={k}: max JSD={max_js:.6f} > {MAX_JSD_THRESH}")
    return failures


def cmd_compare(args):
    """Gate a candidate run against the reference: allowed-set + distributions.

    The reference's own greedy output per adapter defines that adapter's allowed
    set. A correct candidate (any setting that is NOT the broken FA3+FULL path)
    reproduces the reference exactly, so its decoded ids are a subset of the
    reference's and its teacher-forced distributions match within the fused-vs-
    native floor. The broken path emits ids the reference never produced and
    diverges in distribution.
    """
    work_dir = args.work_dir
    ref = json.load(open(os.path.join(work_dir, f"{args.ref}.json")))
    cand = json.load(open(os.path.join(work_dir, f"{args.cand}.json")))
    label = args.label

    # Allowed set per adapter = the set of ids the REFERENCE emitted for it.
    allowed = {ctrl: set(ids) for ctrl, ids in ref["decodes"].items()}

    print(f"\nFA3-SCHEDULE compare: {label}  (cand '{args.cand}' vs ref '{args.ref}')")

    failures = []

    # 1. Allowed-set gate: every candidate token must be one the reference also
    #    produced for that adapter. This is the analogue of the production
    #    out-of-allowed-set token (the token-0 KeyError).
    out_of_set = 0
    for ctrl, gen_ids in cand["decodes"].items():
        aset = allowed.get(ctrl, set())
        bad = [t for t in gen_ids if t not in aset]
        if bad:
            out_of_set += len(bad)
            failures.append(
                f"{label}: adapter ctrl={ctrl} emitted {len(bad)} token(s) the "
                f"reference never produced (e.g. {bad[:5]})"
            )
    print(f"  tokens outside the reference's allowed set: {out_of_set}")

    # 2. Distribution gate: candidate must match the reference within the floor,
    #    scored over the identical [prefix + ctrl + ref-continuation] sequence.
    for ctrl in ref["dists"]:
        if ctrl not in cand["dists"]:
            failures.append(f"{label}: candidate missing adapter ctrl={ctrl}")
            continue
        failures += _gate_distributions(
            ref["dists"][ctrl], cand["dists"][ctrl], f"ctrl={ctrl}"
        )

    if failures:
        print(f"\nFAIL: {label}\n  " + "\n  ".join(failures))
        return 1
    print(f"\nPASS: {label} — all decodes in-set and distributions match reference")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_build = sub.add_parser("build", help="Compose SR MultiSwitch (CPU)")
    p_build.add_argument("--work-dir", required=True)

    p_run = sub.add_parser("run", help="Load in vLLM under one setting (GPU)")
    p_run.add_argument("--work-dir", required=True)
    p_run.add_argument("--tag", required=True, help="Output tag for this run")
    p_run.add_argument(
        "--cudagraph",
        required=True,
        choices=["FULL", "FULL_AND_PIECEWISE", "PIECEWISE", "eager"],
    )
    p_run.add_argument("--prefix", required=True, choices=["on", "off"])

    p_cmp = sub.add_parser("compare", help="Gate candidate vs reference (CPU)")
    p_cmp.add_argument("--work-dir", required=True)
    p_cmp.add_argument("--ref", required=True, help="Reference run tag")
    p_cmp.add_argument("--cand", required=True, help="Candidate run tag")
    p_cmp.add_argument("--label", required=True)

    args = parser.parse_args()
    return {"build": cmd_build, "run": cmd_run, "compare": cmd_compare}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
