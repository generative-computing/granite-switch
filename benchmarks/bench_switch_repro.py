#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit f48054ac) and kept unchanged but for one
# internal link, so this benchmark measures throughput exactly as it does.
# Update it only by copying it again.
"""Decode / prefill throughput driver for the SWITCH.md reproduction.

A fork of the reference harness's ``bench_toks_per_sec.py`` (from
an internal repository, ``feature/fused-lora-clean``), which
produced the numbers in SWITCH.md sections 3.1-3.3.

**What is reproduced verbatim.** Everything that decides *what gets measured*:
the prompt construction (tiled filler from one seed string per index), the
+/-20% length jitter, the four prefill patterns, the arm-independent RNG seeding
that makes the per-prompt adapter assignment byte-identical across arms, the
2-warmup / 5-measured structure, and the ``iters_ms`` record schema. Those
functions are copied unchanged and marked ``VERBATIM``; do not "improve" them,
because their exact behaviour is the thing that makes our numbers comparable to
the published ones.

**What is added.** Only things that observe, never things that change timing:

* **Realized-vs-intended token accounting.** The reference builds a prompt as a
  token-id list and then hands vLLM ``tokenizer.decode(ids)`` -- a string. The
  re-encode is not guaranteed to return the same number of tokens, so the
  ``tokens`` field it records (and divides by, for prefill tok/s) is the
  *intended* count. SWITCH.md never mentions this. We keep the reference's code
  path exactly, and additionally record ``tokens_realized`` by re-encoding, plus
  ``tokens_drift_pct``. If drift is zero the reproduction is exact; if it is not,
  the report says so with a number instead of inheriting an unstated assumption.
* **Per-cell provenance** -- GPU, driver, versions, resolved attention backend,
  cudagraph-vs-eager, scheduler config, and clock/throttle state around the
  measured window. An eager fallback on one arm is a silent 5-10x that would read
  as "that arm is slow"; SWITCH.md records none of this.
* **Preflight gates** that hard-fail rather than produce a fast-looking arm with
  no live adapters. Both loaders drop unrecognized weights silently, so "fast"
  and "broken" are indistinguishable without a gate.
* **Explicit scheduler pinning.** vLLM picks ``max_num_batched_tokens`` /
  ``max_num_seqs`` *by GPU model*, so prefill chunking would otherwise depend on
  which node we land on. Pinned identically for every arm.

Arms. ``--arm`` is the report label; ``--model-type`` is the mechanism. They are
separate because two arms can share a mechanism and differ only in which venv
(and therefore which ``granite_switch.vllm`` backend) is installed:

    arm          model-type      what it is
    gs-sr-vllm          granite-switch  SR adapters, SWITCH kernel
    gs-lora-vllm        granite-switch  LoRA adapters, SWITCH kernel
    gs-lora-vllm-oldkernel      granite-switch  the SAME gs-lora-vllm checkpoint on the pre-SWITCH
                                 Punica backend (rev e2c101d, installed in a
                                 second venv). This is SWITCH.md's own
                                 ``switch-main`` arm: "the *same* switched-LoRA
                                 checkpoints run on the pre-kernel backend",
                                 only the backend differs.
    native-lora      native-lora     stock vLLM multi-LoRA, 7-linear adapters
    native-sr     native-lora     stock vLLM multi-LoRA serving the SAME SR
                                 fleet gs-sr-vllm uses -- byte-identical adapter files,
                                 shunt included. Stock vLLM has no layer-level
                                 LoRA site, so it cannot bind ``cross_stream``
                                 and would in fact RAISE on those tensors; the
                                 arm passes ``--lora-skip-prefixes cross_stream``
                                 so they are skipped at load instead. What the
                                 arm therefore shows is SR's adapters served
                                 without SR's architecture: no second stream, no
                                 doubled Q against shared base K/V, no shunt.
                                 Expected to land on native-lora within noise (see
                                 the report) -- the value is demonstrating that
                                 rather than asserting it, so it runs at N=1 and
                                 N=12 only, not swept.
"""

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

PREFILL_PATTERNS = ["all_base", "all_single", "mostly_base", "mixed"]

#: Report label -> the mechanism it runs on. Kept explicit so a typo in --arm
#: cannot silently mislabel a block in the dataset.
ARMS = {
    "gs-sr-vllm": "granite-switch",
    "gs-lora-vllm": "granite-switch",
    "gs-lora-vllm-oldkernel": "granite-switch",
    "native-lora": "native-lora",
    "native-sr": "native-lora",
    "baseline": "base",
}


# ---------------------------------------------------------------------------
# VERBATIM from the reference harness -- prompt construction.
# Do not modify: identical prompts are what make our numbers comparable.
# ---------------------------------------------------------------------------


def _generate_batch_lengths(target_seq_len, batch_size, pattern):
    """Generate reproducible per-prompt lengths (±20% jitter). Same across model types."""
    rng = random.Random(f"lengths_{target_seq_len}_{batch_size}")
    lo = int(0.8 * target_seq_len)
    hi = int(1.2 * target_seq_len)
    return [rng.randint(lo, hi) for _ in range(batch_size)]


def _make_unique_filler(tokenizer, length, index, base_text):
    """Generate a unique token-id list of exactly `length` tokens for prompt `index`."""
    seed_text = f"Benchmark request {index}: " + base_text
    toks = tokenizer.encode(seed_text)
    repeated = (toks * (length // len(toks) + 1))[:length]
    return repeated


def _build_prefill_all_base(tokenizer, length, index, base_text):
    return _make_unique_filler(tokenizer, length, index, base_text)


def _build_prefill_batch(
    tokenizer,
    pattern,
    batch_size,
    target_seq_len,
    adapter_token_ids,
    is_base_model,
    base_text,
    num_adapters,
    prepend_control=True,
):
    """Build one prefill batch. Returns (prompts, total_tokens, adapter_slots).

    ``adapter_slots[i]`` is the integer adapter slot intended active for prompt
    i, or None for base. Drawn from RNGs seeded only by batch shape -- never by
    arm -- so the switch realizes slot s as control token adapter_token_ids[s]
    and native as lora_requests[s] on byte-identical prompts.
    """
    lengths = _generate_batch_lengths(target_seq_len, batch_size, pattern)

    pat_rng = random.Random(f"prefill_pat_{target_seq_len}_{batch_size}_{pattern}")
    slot_rng = random.Random(f"prefill_slot_{target_seq_len}_{batch_size}_{pattern}")
    pos_rng = random.Random(f"prefill_pos_{target_seq_len}_{batch_size}_{pattern}")

    prompts = []
    adapter_slots = []
    total_tokens = 0
    for i, length in enumerate(lengths):
        if pattern == "mixed":
            pat = pat_rng.choice(["all_base", "all_single", "mostly_base"])
        else:
            pat = pattern

        adapter_intent = (not is_base_model) and pat != "all_base"
        slot = (
            slot_rng.randrange(num_adapters)
            if (adapter_intent and num_adapters > 0)
            else None
        )

        if not adapter_intent or not prepend_control:
            toks = _build_prefill_all_base(tokenizer, length, i, base_text)
        elif pat == "all_single":
            filler = _make_unique_filler(tokenizer, length - 1, i, base_text)
            toks = [adapter_token_ids[slot]] + filler  # noqa: RUF005 (VERBATIM)
        else:  # mostly_base: control token late in the sequence
            filler = _make_unique_filler(tokenizer, length - 1, i, base_text)
            lo_pos = int(0.8 * length)
            hi_pos = max(lo_pos + 1, int(0.95 * length))
            insert_pos = pos_rng.randint(lo_pos, hi_pos)
            toks = filler[:insert_pos] + [adapter_token_ids[slot]] + filler[insert_pos:]

        prompts.append(tokenizer.decode(toks))
        adapter_slots.append(slot)
        total_tokens += length
    return prompts, total_tokens, adapter_slots


# ---------------------------------------------------------------------------
# Added: observation only.
# ---------------------------------------------------------------------------


def _jsonable(value):
    """Coerce provenance values json.dumps cannot handle (e.g. torch.dtype).

    Learned the hard way: an un-coerced ``model_config.dtype`` raised *after*
    every cell in a block had been measured, discarding the whole block.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def realized_tokens(tokenizer, prompts):
    """``(token count, host encode seconds)`` after the reference's decode() round-trip.

    The count is what vLLM actually sees; the reference divides by the *intended*
    sum instead (see the module docstring).

    The timing is not incidental. The reference hands vLLM strings rebuilt from
    token ids it already had, and vLLM drains its input-preprocess generator to
    completion before starting the engine -- so the host re-encode of the entire
    batch runs serially INSIDE the measured window instead of overlapping GPU work.
    Passing a TokensPrompt would remove it, but that changes the code path whose
    numbers we are reproducing. So the path stays and the cost is recorded, the same
    choice made for token drift: measure the artefact rather than silently inherit
    or silently fix it. It is identical across arms, so it cannot bias any Δ%.
    """
    t0 = time.perf_counter()
    total = sum(len(tokenizer.encode(p)) for p in prompts)
    return total, time.perf_counter() - t0


def nvidia_smi():
    """Clocks, temperature, power and throttle reasons, or None if unavailable."""
    fields = (
        "clocks.current.sm,clocks.current.memory,temperature.gpu,"
        "power.draw,clocks_throttle_reasons.active"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return [c.strip() for c in out.splitlines()[0].split(",")] if out else None


def git_dirty():
    """sha256 of the uncommitted diff, or None if the tree is clean.

    The gs-sr-vllm / gs-lora-vllm arms execute whatever is in the working tree, so a
    committed-SHA-only provenance record can describe code that never ran.
    """
    import hashlib

    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if not diff.strip():
        return None
    return {
        "diff_sha256_16": hashlib.sha256(diff.encode()).hexdigest()[:16],
        "diff_lines": diff.count("\n"),
    }


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sub_config(llm, name):
    """One of vLLM's sub-configs, or None if this version does not expose it.

    v1's ``LLMEngine`` sets only ``vllm_config`` and ``model_config``; the cache,
    scheduler and LoRA configs hang off ``vllm_config``, NOT off the engine. Going
    straight to ``llm_engine.cache_config`` raises AttributeError, and
    ``getattr(llm_engine, "lora_config", None)`` quietly returns None -- which
    would have disabled the max_loras gate below without any sign. Try the nested
    path first, then a direct attribute for other engine versions.
    """
    vc = getattr(llm.llm_engine, "vllm_config", None)
    if vc is not None and getattr(vc, name, None) is not None:
        return getattr(vc, name)
    return getattr(llm.llm_engine, name, None)


class _SetLoraSkipPrefixes:
    """Set ``lora_skip_prefixes`` on a model, in whatever process holds it.

    A module-level callable class rather than the closure this used to be:
    ``apply_model`` dispatches through ``collective_rpc``, which pickles the callable
    to reach each worker process. A function defined inside ``main()`` is a local
    object and unpicklable, so at TP=1 (worker in-process, no pickling) it worked and
    at TP>1 every native-sr block died with::

        AttributeError: Can't get local object 'main.<locals>._set_skip'
    """

    def __init__(self, prefixes):
        self.prefixes = list(prefixes)

    def __call__(self, model):
        model.lora_skip_prefixes = list(self.prefixes)
        return getattr(model, "lora_skip_prefixes", None)


def engine_provenance(llm, args):
    """Everything needed to tell a real regression from a changed environment."""
    prov = {
        "hostname": socket.gethostname(),
        "gs_commit": git_sha(),
        "gs_dirty": git_dirty(),
        "python": sys.version.split()[0],
        "arm": args.arm,
        "model_type": args.model_type,
        "num_adapters_declared": args.num_adapters,
        "lora_skip_prefixes": args.lora_skip_prefixes,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    try:
        import torch

        prov["torch"] = torch.__version__
        if torch.cuda.is_available():
            prov["gpu_name"] = torch.cuda.get_device_name(0)
            prov["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        prov["torch_error"] = str(exc)
    for mod in ("vllm", "triton", "transformers"):
        try:
            prov[mod] = __import__(mod).__version__
        except Exception:
            prov[mod] = None
    try:
        mc = llm.llm_engine.model_config
        cc = sub_config(llm, "cache_config")
        sc = sub_config(llm, "scheduler_config")
        cp = sub_config(llm, "compilation_config")
        prov.update(
            {
                "dtype": mc.dtype,
                "num_hidden_layers": mc.hf_config.num_hidden_layers,
                "vocab_size": getattr(mc.hf_config, "vocab_size", None),
                "hf_num_adapters": getattr(mc.hf_config, "num_adapters", None),
                "dual_stream": getattr(mc.hf_config, "dual_stream", None),
                "cross_stream_rank": getattr(mc.hf_config, "cross_stream_rank", None),
                "max_model_len": mc.max_model_len,
                "enable_prefix_caching": getattr(cc, "enable_prefix_caching", None),
                # Profiled in the engine-core process and not propagated back to
                # the front end, so this is None under the default MP engine.
                # Recorded anyway for the single-process case; a null here means
                # "not visible from the front end", not "zero blocks".
                "num_gpu_blocks_frontend": getattr(cc, "num_gpu_blocks", None),
                "max_cudagraph_capture_size": getattr(
                    cp, "max_cudagraph_capture_size", None
                ),
                "cudagraph_mode": str(getattr(cp, "cudagraph_mode", None)),
                "max_num_batched_tokens": getattr(sc, "max_num_batched_tokens", None),
                "max_num_seqs": getattr(sc, "max_num_seqs", None),
            }
        )
    except Exception as exc:
        prov["engine_introspect_error"] = str(exc)
    return _jsonable(prov)


def preflight(llm, args, atids, lora_requests, tokenizer):
    """Hard-fail unless this arm demonstrably has live adapters.

    Both loaders drop weights they cannot match *silently* -- our composer with a
    bare ``continue``, stock vLLM with ``reset_lora(); continue`` behind a
    ``logger.debug``. A dropped-adapter arm is therefore fast and wrong, and
    looks exactly like a win. Refusing to run is the only safe default.
    """
    from vllm import SamplingParams

    if args.model_type == "base":
        print("Preflight: base arm, no adapter to verify.")
        return {"checked": False}

    cfg = llm.llm_engine.model_config.hf_config
    cache_cfg = sub_config(llm, "cache_config")
    if cache_cfg is None:
        raise SystemExit(
            "FATAL preflight: cannot read the resolved cache config, so prefix "
            "caching cannot be verified OFF. It defaults to ON in vLLM, and an arm "
            "sharing a prefix that a position-0-control-token arm cannot share is "
            "a first-order fake result in either direction. Refusing to guess."
        )
    resolved_pc = getattr(cache_cfg, "enable_prefix_caching", None)
    if resolved_pc is not False:
        raise SystemExit(
            f"FATAL preflight: prefix caching resolved to {resolved_pc!r}, not "
            "False. It lets an arm share a prefix that a "
            "position-0-control-token arm cannot, which is a first-order fake "
            "result in either direction."
        )
    print("Preflight: prefix caching resolved OFF")

    # max_cudagraph_capture_size is a request, not a pin: vLLM clamps it to
    # max_num_batched_tokens and re-snaps it onto its own generated size list with
    # only a logger.warning, and LLM() coerces the compilation_config dict while
    # dropping keys it does not recognise -- the same silent-swallow shape as the
    # lora_config bug. The report asserts every arm ran at the same capture size, so
    # verify it instead of asserting it.
    comp = sub_config(llm, "compilation_config")
    resolved_cap = getattr(comp, "max_cudagraph_capture_size", None)
    if resolved_cap != args.cudagraph_capture_size:
        raise SystemExit(
            f"FATAL preflight: requested max_cudagraph_capture_size="
            f"{args.cudagraph_capture_size} but vLLM resolved {resolved_cap!r}. "
            "Capture size changes the decode regime, so arms measured at different "
            "sizes are not comparable. Lower --max-num-batched-tokens or pick a "
            "size on vLLM's generated list."
        )
    print(f"Preflight: cudagraph capture size resolved {resolved_cap}")

    if args.model_type == "granite-switch":
        got = int(getattr(cfg, "num_adapters", 0))
        if got != args.num_adapters:
            raise SystemExit(
                f"FATAL preflight: checkpoint carries num_adapters={got} but the "
                f"sweep declared N={args.num_adapters}. The N axis would be wrong."
            )
        if len(atids) != got:
            raise SystemExit(
                f"FATAL preflight: {len(atids)} control tokens for {got} adapters."
            )
    elif len(lora_requests) != args.num_adapters:
        raise SystemExit(
            f"FATAL preflight: {len(lora_requests)} LoRARequests but the sweep "
            f"declared N={args.num_adapters}."
        )
    if args.model_type == "native-lora":
        lc = sub_config(llm, "lora_config")
        if lc is None:
            raise SystemExit(
                "FATAL preflight: cannot read the resolved LoRA config, so "
                "max_loras cannot be verified. vLLM's v1 scheduler SKIPS a "
                "request whose LoRA id exceeds max_loras (default 1) instead of "
                "erroring, so an unverified value can silently turn an "
                "N-adapter cell into a 1-adapter one."
            )
        resolved = getattr(lc, "max_loras", None)
        if resolved is None or resolved < args.num_adapters:
            raise SystemExit(
                f"FATAL preflight: resolved max_loras={resolved} < N="
                f"{args.num_adapters}. The v1 scheduler silently SKIPS requests "
                "whose LoRA id exceeds max_loras, so this cell would quietly "
                "serve fewer adapters than it claims."
            )
        print(f"Preflight: resolved max_loras={resolved} covers N={args.num_adapters}")

    # The adapter must measurably change the next-token DISTRIBUTION.
    #
    # This deliberately does not compare greedy text. The first version did, and it
    # was a false pass on the switch arms: it built the adapted prompt as
    # `[control] + probe[:-1]`, dropping a token to hold the length constant, which
    # shifts the repeating filler by one position. Every arm then "differed from
    # base" by resuming the tile at a different phase, whether or not any adapter
    # loaded. Observed on a real run -- base ": Explain the theory of relativity in
    # de" against adapted "0: Explain the theory of relativity in d", a pure
    # one-character shift.
    #
    # Greedy text is also too blunt in the other direction: on tiled filler the
    # copy pattern dominates the distribution, so a real adapter need not move the
    # argmax at all. Top-k logprobs catch both cases.
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    probe = _make_unique_filler(tokenizer, args.preflight_tokens, 0, args.prompt)

    def _dist(text, lora=None):
        out = (
            llm.generate([text], sp, lora_request=lora)
            if lora is not None
            else llm.generate([text], sp)
        )[0].outputs[0]
        if not out.logprobs:
            raise SystemExit(
                "FATAL preflight: vLLM returned no logprobs, so the adapter cannot "
                "be verified live. Refusing to guess."
            )
        return {
            int(t): math.exp(getattr(lp, "logprob", lp))
            for t, lp in out.logprobs[0].items()
        }

    def _jsd(p, q):
        keys = set(p) | set(q)
        ps = [p.get(k, 0.0) for k in keys]
        qs = [q.get(k, 0.0) for k in keys]
        sp_, sq = sum(ps) or 1.0, sum(qs) or 1.0
        ps, qs = [x / sp_ for x in ps], [x / sq for x in qs]

        def kl(a, b):
            return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)

        m = [(x + y) / 2 for x, y in zip(ps, qs)]
        return 0.5 * kl(ps, m) + 0.5 * kl(qs, m)

    base_dist = _dist(tokenizer.decode(probe))
    if args.model_type == "granite-switch":
        # PREPEND and drop nothing, so the final token -- and thus the prediction
        # context -- is identical to base. One extra token still shifts positions
        # slightly, which is why the floor is a floor and not an equality test.
        adapted_dist = _dist(tokenizer.decode([atids[0], *probe]))
    else:
        # Native arms are the clean case: the prompt is byte-identical and the
        # adapter arrives out of band via the LoRARequest.
        adapted_dist = _dist(tokenizer.decode(probe), lora=lora_requests[0])

    divergence = _jsd(adapted_dist, base_dist)
    if divergence < args.min_preflight_jsd:
        raise SystemExit(
            f"FATAL preflight: adapter-active distribution is indistinguishable "
            f"from base (JSD {divergence:.3e} < {args.min_preflight_jsd:.3e}). The "
            "adapter is not firing -- most likely its weights were silently dropped "
            "at load. Refusing to record a meaningless arm."
        )
    print(f"Preflight OK: adapter shifts the distribution, JSD={divergence:.6f}")
    return {
        "checked": True,
        "probe_tokens": args.preflight_tokens,
        "preflight_jsd": divergence,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=sorted(ARMS),
        help="Report label. Determines --model-type unless overridden.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Composed checkpoint (switch arms) or base model "
        "(native / baseline arms).",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["granite-switch", "base", "native-lora"],
        help="Mechanism (default: implied by --arm).",
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Comma-separated PEFT dirs for native arms; one "
        "LoRARequest each, max_loras=N.",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        required=True,
        help="N for this block. Cross-checked against the "
        "checkpoint / LoRARequest count in preflight.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the theory of relativity in detail.",
        help="Filler seed text. VERBATIM default from the "
        "reference -- changing it changes every prompt.",
    )
    # --- cells (reference defaults for sections 3.1-3.3) ---
    parser.add_argument("--decode-sweep", action="store_true")
    parser.add_argument("--prefill-sweep", action="store_true")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--adapter-fractions", default="0,100")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--seq-lens", default="512,2048")
    parser.add_argument(
        "--patterns", default=None, help=f"Subset of {','.join(PREFILL_PATTERNS)}."
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    # --- engine, pinned identically for every arm ---
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cudagraph-capture-size", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--lora-skip-prefixes",
        default=None,
        help="Comma-separated module-name prefixes for stock vLLM to skip when "
        "loading an adapter (e.g. cross_stream). vLLM's WorkerLoRAManager reads "
        "this as a plain attribute on the model instance and honours it BEFORE its "
        "expected-modules check, so it turns an otherwise-fatal ValueError into a "
        "deliberate, recorded omission. Required for any stock-vLLM arm served an "
        "SR adapter.",
    )
    parser.add_argument(
        "--max-cpu-loras",
        type=int,
        default=None,
        help="Native arms: vLLM requires max_cpu_loras >= max_loras (default: N).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    # --- output ---
    parser.add_argument(
        "--dump-iters", default=None, help="Append one JSONL record per cell here."
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Block label stored in every record, e.g. srA_decode_N4.",
    )
    parser.add_argument("--preflight-tokens", type=int, default=256)
    parser.add_argument(
        "--min-preflight-jsd",
        type=float,
        default=1e-5,
        help="Minimum Jensen-Shannon divergence between the adapter-active and "
        "base next-token distributions for the adapter to count as live. Near "
        "zero on purpose: the question is whether the adapter does anything at "
        "all, not how strongly.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Escape hatch for debugging only. Never use for a "
        "recorded run: the gates it skips are what "
        "distinguish a fast arm from a broken one.",
    )
    args = parser.parse_args()

    args.model_type = args.model_type or ARMS[args.arm]
    if args.model_type == "native-lora" and not args.lora_path:
        parser.error("native arms require --lora-path")
    if not args.model:
        parser.error("--model is required")
    if not (args.decode_sweep or args.prefill_sweep):
        parser.error("pass --decode-sweep and/or --prefill-sweep")
    # Setting min_tokens == max_tokens defeats vLLM's length cap: a request that
    # cannot reach max_tokens within max_model_len is never marked
    # FINISHED_LENGTH_CAPPED, gets clamped to zero new tokens, and is rescheduled
    # forever while generate() spins on has_unfinished_requests(). It HANGS rather
    # than truncating. Safe at the defaults (512+128 vs 4096); this guard is for
    # the moment someone raises the token counts or lowers max_model_len.
    if args.decode_sweep:
        need = args.input_tokens + args.decode_tokens
        if need >= args.max_model_len - 1:
            parser.error(
                f"--input-tokens {args.input_tokens} + --decode-tokens "
                f"{args.decode_tokens} = {need} must be < --max-model-len - 1 "
                f"({args.max_model_len - 1}). min_tokens == max_tokens defeats "
                "vLLM's length cap, so this hangs instead of truncating."
            )
    if args.prefill_sweep:
        # _generate_batch_lengths jitters up to +20%, and a control token adds one.
        worst = int(1.2 * max(int(s) for s in args.seq_lens.split(","))) + 1
        if worst >= args.max_model_len:
            parser.error(
                f"--seq-lens jitters up to {worst} tokens (+20% and a control "
                f"token), which is >= --max-model-len {args.max_model_len}."
            )
    if args.lora_skip_prefixes and args.model_type != "native-lora":
        parser.error(
            "--lora-skip-prefixes only does anything on a native-lora arm: vLLM "
            "builds the WorkerLoRAManager that reads it only when enable_lora is "
            "set, so on a switch arm it would set an attribute nobody reads while "
            "printing as though the skip were active."
        )

    if args.model_type == "granite-switch":
        from granite_switch.vllm import register

        register()

    from vllm import LLM, SamplingParams

    lora_requests = []
    lora_kwargs = {}
    if args.model_type == "native-lora":
        from vllm.lora.request import LoRARequest

        paths = [p.strip() for p in args.lora_path.split(",") if p.strip()]
        ranks = [
            int(json.load(open(os.path.join(p, "adapter_config.json")))["r"])
            for p in paths
        ]
        lora_requests = [LoRARequest(f"a{j}", j + 1, p) for j, p in enumerate(paths)]
        # vLLM's max_lora_rank is a pydantic Literal over fixed tiers, so an
        # off-tier value from an adapter_config.json raises ValidationError at
        # LLM() construction. Snap UP to the next legal tier: padding is what vLLM
        # does internally anyway, and it keeps the arm runnable instead of dying.
        legal = (1, 8, 16, 32, 64, 128, 256, 320, 512)
        raw_rank = max(ranks)
        if raw_rank > legal[-1]:
            parser.error(f"adapter rank {raw_rank} exceeds vLLM's max of {legal[-1]}")
        lora_rank = next(r for r in legal if r >= raw_rank)
        # max_loras defaults to 1, and the v1 scheduler SKIPS (does not error
        # on) a request whose LoRA id exceeds it -- so an unset max_loras turns an
        # N-adapter cell into a 1-adapter cell with no diagnostic at all. Pinned
        # to N, and asserted against the resolved config in preflight.
        lora_kwargs = {
            "enable_lora": True,
            "max_lora_rank": lora_rank,
            "max_loras": len(lora_requests),
            "max_cpu_loras": args.max_cpu_loras or len(lora_requests),
        }
        print(
            f"Native LoRA: N={len(lora_requests)} ranks={ranks} "
            f"max_lora_rank={lora_rank}"
            + (f" (snapped up from {raw_rank})" if lora_rank != raw_rank else "")
        )

    print(f"Loading [{args.arm} / {args.model_type}] {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Off for every arm: see preflight. The reference derives this from the
        # sweep flags; we pin it so it cannot drift.
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        compilation_config={"max_cudagraph_capture_size": args.cudagraph_capture_size},
        **lora_kwargs,
    )
    print("Model loaded.")

    if args.lora_skip_prefixes and args.model_type == "native-lora":
        prefixes = [p.strip() for p in args.lora_skip_prefixes.split(",") if p.strip()]

        got = llm.llm_engine.apply_model(_SetLoraSkipPrefixes(prefixes))
        if not got or any(g != prefixes for g in got):
            raise SystemExit(
                f"FATAL: lora_skip_prefixes={prefixes} did not take on every "
                f"rank (got {got}). Without it vLLM raises ValueError on the "
                "skipped tensors, so proceeding would either crash mid-sweep or "
                "silently load a different adapter set than intended."
            )
        print(f"lora_skip_prefixes={prefixes} set on {len(got)} rank(s)")

    tokenizer = llm.get_tokenizer()
    num_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    atids = (
        # `or []`: the config leaves adapter_token_ids as None when num_adapters is
        # 0, and a bare list(None) would TypeError after paying the model load.
        # Preflight's count check still catches a genuinely wrong list.
        list(
            getattr(llm.llm_engine.model_config.hf_config, "adapter_token_ids", None)
            or []
        )
        if args.model_type == "granite-switch"
        else []
    )
    if args.model_type == "granite-switch":
        num_adapters = len(atids)
    elif args.model_type == "native-lora":
        num_adapters = len(lora_requests)
    else:
        num_adapters = 0

    prov = engine_provenance(llm, args)
    print(
        "Provenance: "
        + json.dumps(
            {
                k: prov[k]
                for k in (
                    "gpu_name",
                    "vllm",
                    "torch",
                    "triton",
                    "dtype",
                    "num_hidden_layers",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "enable_prefix_caching",
                )
                if k in prov
            }
        )
    )

    pf = (
        {"checked": False, "skipped": True}
        if args.skip_preflight
        else preflight(llm, args, atids, lora_requests, tokenizer)
    )

    def dump(record):
        if not args.dump_iters:
            return
        record["tag"] = args.tag
        record["arm"] = args.arm
        record["N"] = args.num_adapters
        record["provenance"] = prov
        record["preflight"] = pf
        with open(args.dump_iters, "a") as handle:
            handle.write(json.dumps(_jsonable(record), default=str) + "\n")

    def gen(prompts, sp, adapter_slots=None):
        """VERBATIM semantics: native threads a per-prompt LoRARequest, others don't."""
        if args.model_type == "native-lora":
            reqs = (
                lora_requests[0]
                if adapter_slots is None
                else [
                    lora_requests[s] if s is not None else None for s in adapter_slots
                ]
            )
            return llm.generate(prompts, sp, lora_request=reqs)
        return llm.generate(prompts, sp)

    def measure(prompts, sp, gen_slots):
        """2 warmup + N measured perf_counter runs, plus clock state around them."""
        for _ in range(args.warmup_runs):
            gen(prompts, sp, gen_slots)
        smi_before = nvidia_smi()
        iters = []
        for _ in range(args.num_runs):
            t0 = time.perf_counter()
            gen(prompts, sp, gen_slots)
            iters.append(time.perf_counter() - t0)
        return iters, smi_before, nvidia_smi()

    # ---------------- decode ----------------
    if args.decode_sweep:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        fractions = (
            [0]
            if args.model_type == "base"
            else [int(f) for f in args.adapter_fractions.split(",")]
        )
        is_base = args.model_type == "base"

        sp = SamplingParams(
            min_tokens=args.decode_tokens,
            max_tokens=args.decode_tokens,
            temperature=0.0,
        )
        # Companion single-token generate on the identical batch, so the report can
        # subtract the prompt-prefill share out of the "decode" cell. SWITCH.md's
        # decode number includes it.
        sp1 = SamplingParams(min_tokens=1, max_tokens=1, temperature=0.0)

        def build_prompt(num_tokens, adapter_token_id=None, index=0):
            """VERBATIM from the reference (control token replaces the last id)."""
            toks = _make_unique_filler(tokenizer, num_tokens, index, args.prompt)
            if adapter_token_id is not None:
                toks = [adapter_token_id] + list(toks[:-1])  # noqa: RUF005 (VERBATIM)
            return tokenizer.decode(toks)

        def build_mixed_batch(B, frac):
            """VERBATIM: arm-independent RNGs for the use-adapter and slot draws."""
            rng = random.Random(f"decode_{args.input_tokens}_{B}_{frac}")
            slot_rng = random.Random(f"decode_slot_{args.input_tokens}_{B}_{frac}")
            prompts, slots = [], []
            for i in range(B):
                use = not is_base and frac > 0 and rng.random() < frac / 100
                slot = (
                    slot_rng.randrange(num_adapters)
                    if (use and num_adapters > 0)
                    else None
                )
                aid = atids[slot] if (use and atids) else None
                prompts.append(
                    build_prompt(args.input_tokens, adapter_token_id=aid, index=i)
                )
                slots.append(slot)
            return prompts, slots

        for frac in fractions:
            for B in batch_sizes:
                prompts, slots = build_mixed_batch(B, frac)
                gen_slots = slots if args.model_type == "native-lora" else None
                iters, smi0, smi1 = measure(prompts, sp, gen_slots)
                iters1, _, _ = measure(prompts, sp1, gen_slots)
                intended = B * args.input_tokens
                realized, encode_s = realized_tokens(tokenizer, prompts)
                dump(
                    {
                        "phase": "decode",
                        "batch": B,
                        "frac": frac,
                        "decode_tokens": args.decode_tokens,
                        "input_tokens": args.input_tokens,
                        # Part of the cell's IDENTITY, not just provenance: the plot
                        # generator keys on it. Absent, a TP=2 cell collides exactly
                        # with the published TP=1 cell of the same shape and silently
                        # replaces it.
                        "tp": args.tensor_parallel_size,
                        "iters_ms": [t * 1000 for t in iters],
                        "prefill_only_iters_ms": [t * 1000 for t in iters1],
                        "prompt_tokens_intended": intended,
                        "prompt_tokens_realized": realized,
                        "tokens_drift_pct": 100.0 * (realized - intended) / intended,
                        "host_encode_ms": encode_s * 1000.0,
                        "adapted_prompts": sum(s is not None for s in slots),
                        "num_layers": num_layers,
                        "smi_before": smi0,
                        "smi_after": smi1,
                    }
                )
                med = sorted(iters)[len(iters) // 2]
                print(
                    f"  decode B={B} frac={frac}% "
                    f"median={med * 1000:.1f}ms "
                    f"tok/s={B * args.decode_tokens / med:.0f}",
                    flush=True,
                )

    # ---------------- prefill ----------------
    if args.prefill_sweep:
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        patterns = (
            [p.strip() for p in args.patterns.split(",")]
            if args.patterns
            else PREFILL_PATTERNS
        )
        is_base = args.model_type == "base"
        prepend_control = args.model_type == "granite-switch"
        sp = SamplingParams(max_tokens=1, temperature=0.0)

        for sl in seq_lens:
            for B in batch_sizes:
                for pattern in patterns:
                    prompts, intended, slots = _build_prefill_batch(
                        tokenizer,
                        pattern,
                        B,
                        sl,
                        atids,
                        is_base,
                        args.prompt,
                        num_adapters,
                        prepend_control=prepend_control,
                    )
                    gen_slots = slots if args.model_type == "native-lora" else None
                    iters, smi0, smi1 = measure(prompts, sp, gen_slots)
                    realized, encode_s = realized_tokens(tokenizer, prompts)
                    dump(
                        {
                            "phase": "prefill",
                            "tp": args.tensor_parallel_size,
                            "seq_len": sl,
                            "batch": B,
                            "pattern": pattern,
                            # "tokens" keeps the reference's meaning (intended) so the
                            # published formula reproduces; the realized count sits
                            # beside it rather than replacing it.
                            "tokens": intended,
                            "tokens_realized": realized,
                            "tokens_drift_pct": 100.0
                            * (realized - intended)
                            / intended,
                            "iters_ms": [t * 1000 for t in iters],
                            "host_encode_ms": encode_s * 1000.0,
                            "adapted_prompts": sum(s is not None for s in slots),
                            "num_layers": num_layers,
                            "smi_before": smi0,
                            "smi_after": smi1,
                        }
                    )
                    med = sorted(iters)[len(iters) // 2]
                    print(
                        f"  prefill seq={sl} B={B} {pattern} "
                        f"tokens={intended} (realized {realized}) "
                        f"median={med * 1000:.1f}ms "
                        f"tok/s={intended / med:.0f}",
                        flush=True,
                    )

    print(f"BLOCK COMPLETE: {args.tag or args.arm}")


if __name__ == "__main__":
    main()
