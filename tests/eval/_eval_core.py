# SPDX-License-Identifier: Apache-2.0
"""Capture and compare adapter-conditioned output distributions for the eval gate.

One module, used by BOTH `tests/eval/test_adapter_evals.py` (which compares against
a committed baseline) and the skill's `record_eval_baseline.py` (which writes one).
Sharing it is the point: if recording and checking were separate implementations,
the baseline would slowly stop describing what the check measures and the gate
would pass forever.

What is measured, per eval config:

  continuations  the greedy token ids the baseline run produced, per prompt
  adapter_dists  top-k logprobs at each of those positions, with the adapter's
                 control token prepended to the prompt
  base_dists     the same prompts with NO control token

MEASUREMENT IS TEACHER-FORCED, which is load-bearing. Both the baseline and the
check feed the BASELINE's `continuations` through the model and read the
distribution at each position, so position i is always conditioned on a byte-
identical prefix. Only recording generates; the check never lets the model
condition on its own output.

Why: the first version of this gate compared position-aligned distributions
across a self-conditioning greedy rollout, and `granite42_vllm` failed its own
recording line (vLLM 0.30) with mean JSD 0.95 on prompt 3. The cause was ONE
near-tie argmax flip at position 1 -- `527` vs `16` -- after which the two runs
were generating different sentences and the metric was comparing unrelated
contexts. The numbers gave it away: mean JSD 0.953644 and mean(1-Jaccard)
0.9583 are both exactly 23/24, the fraction of positions at or after the flip,
and they were byte-identical on 0.26 and 0.30. Positions before the flip agreed
to JSD < 0.01.

A rollout gate therefore cannot distinguish "this vLLM line computes different
logits" from "this vLLM line broke a tie the other way" -- and tolerating that
tie is the whole reason this is a distribution metric and not a token-exact one
(see below). Teacher forcing removes the cascade without weakening anything: a
real logit or weight regression still moves the distribution at the very first
position it touches.

`adapter_dists` vs the baseline's `adapter_dists` is the regression gate: it answers
"does this vLLM line still produce the output we signed off on". `adapter_dists` vs
`base_dists` is the liveness gate: it answers "did the adapter fire at all".

The liveness gate exists because of a real incident. vLLM 0.29 deleted the
multimodal hook the ASR path was overriding, so the override stopped being called
and no transcript was ever produced again -- with every test still green. An
adapter that silently stops being applied is the same failure shape, and only a
comparison against the un-adapted run can see it.

Metric is top-k JSD plus top-k Jaccard distance, reused from
`tests/shared/logit_metrics.py` -- the same metric the generation-equivalence suite
already gates on. It is deliberately NOT token-exact: the fused SWITCH Triton
kernel's float-reduction order differs from vLLM's native linear, and that sub-ULP
difference can flip a near-tie greedy argmax between vLLM lines. A token-exact gate
would flag that as a regression; a distribution gate does not, while still catching
a real logit or weight break.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.shared.logit_metrics import jaccard, jsd_bits, topk_ids

HERE = Path(__file__).parent
CASES = HERE / "cases"
BASELINES = HERE / "baselines"

# Same defaults tests/conftest.py applies, set HERE because this module is also
# imported by the skill's record_eval_baseline.py, which runs standalone and so
# never loads conftest. Without it, recording a baseline booted vLLM with the
# FlashInfer sampler, which JIT-compiles through ninja and failed outright on the
# GPU pod (`ninja ... returned non-zero exit status 1`, engine core never
# started) -- while the eval CHECK, running under pytest, had the sampler off.
# Recording and checking must not differ in engine configuration, which is the
# whole reason both sides share this module.
#
# setdefault, so an image with a working toolchain can opt back in.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

#: Generated positions per prompt. Long enough that the adapter's behaviour shows,
#: short enough that four configs fit in a GPU leg alongside everything else.
MAX_TOKENS = int(os.environ.get("GSW_EVAL_MAX_TOKENS", "24"))
#: Top-k captured per position. 20 keeps the baseline files well under the
#: 500 KB pre-commit ceiling while leaving room to sweep k when comparing.
TOP_K = int(os.environ.get("GSW_EVAL_TOP_K", "20"))

#: Defaults, overridable per config in the baseline file. Starting values are the
#: generation-equivalence suite's, which were calibrated against real corruptions:
#: a localized break (JSD ~1.0 bit at one position) and a systematic ~30% logit
#: scaling (mean JSD ~0.012) are both caught.
DEFAULT_THRESHOLDS = {
    "mean_jsd": 0.003,
    "max_jsd": 0.05,
    "mean_jaccard_dist": 0.20,
    # The liveness floor. An adapter whose output is indistinguishable from base
    # is not applied -- or is a no-op, which is equally worth knowing.
    "min_adapter_effect_jsd": 0.01,
}


@dataclass(frozen=True)
class EvalConfig:
    name: str
    env: str
    backend: str  # "vllm" | "hf"
    description: str
    #: Consulted when `env` is unset. Lets the HF-backend eval ride on whichever
    #: checkpoint is already staged instead of needing its own copy.
    fallback_env: str | None = None


#: The four evaluation configs. Granite 4.2 and Granite 5 with LoRA/aLoRA adapters
#: on vLLM, an SR (dual-stream) adapter on vLLM, and one adapter on the HF backend
#: -- so a break that lives in the HF path cannot hide behind a green vLLM leg.
EVAL_CONFIGS: dict[str, EvalConfig] = {
    "granite42_vllm": EvalConfig(
        "granite42_vllm",
        "GSW_EVAL_G42_DIR",
        "vllm",
        "Granite 4.2 + LoRA/aLoRA adapter, vLLM backend",
    ),
    "granite5_vllm": EvalConfig(
        "granite5_vllm",
        "GSW_EVAL_G5_DIR",
        "vllm",
        "Granite 5 + LoRA/aLoRA adapter, vLLM backend",
    ),
    "sr_vllm": EvalConfig(
        "sr_vllm",
        "GSW_EVAL_SR_DIR",
        "vllm",
        "Shadow-Residual dual-stream adapter, vLLM backend",
    ),
    "granite42_hf": EvalConfig(
        "granite42_hf",
        "GSW_EVAL_HF_DIR",
        "hf",
        "One adapter on the HuggingFace backend",
        fallback_env="GSW_EVAL_G42_DIR",
    ),
}


#: Compose recipes, consulted only when no checkpoint is staged. Public
#: `ibm-granite` repos, and the same base+adapters the e2e suites already
#: download, so this adds no new artifact to fetch. `sr_vllm` deliberately has
#: NO recipe: a dual-stream checkpoint needs SR adapters, which are not public,
#: so that config keeps skipping rather than quietly evaluating a non-SR model
#: and reporting it as SR coverage.
COMPOSE_RECIPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "granite42_vllm": (
        "ibm-granite/granite-4.1-3b",
        ("ibm-granite/granitelib-guardian-r1.0", "ibm-granite/granitelib-core-r1.0"),
    ),
    "granite5_vllm": (
        "ibm-granite/granite-4.0-micro",
        ("ibm-granite/granitelib-rag-r1.0",),
    ),
}
#: The HF-backend eval evaluates the same checkpoint as granite42_vllm, on the
#: other backend -- mirroring its `fallback_env`.
COMPOSE_RECIPES["granite42_hf"] = COMPOSE_RECIPES["granite42_vllm"]

#: Shared with tests/hf/test_multi_switch_e2e.py ON PURPOSE -- the same cache
#: root, so a checkpoint composed by either side is warm-reused by the other
#: instead of being built twice in one GPU leg.
_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))


def checkpoint_dir(cfg: EvalConfig) -> Path | None:
    """The checkpoint for `cfg`, composing one when no path is staged.

    An explicit ``GSW_EVAL_*_DIR`` always wins, so a pod with a real checkpoint
    staged -- a 30B, or a variant with no public recipe -- is unaffected.

    WHY THERE IS A FALLBACK AT ALL. Requiring the env var meant the eval suite
    could only ever skip: nothing sets ``GSW_EVAL_*_DIR`` on the GPU pod, and
    pytest exits 0 on an all-skipped suite, so the leg went green having
    evaluated nothing (run 39299824463: 4 collected, 4 skipped, "ALL GPU TESTS
    PASSED", 0.03s). An eval gate that cannot run is worse than no gate, because
    it reads as coverage. The repo already solved this for the e2e suites --
    compose from public HF repos into a warm-reused cache -- so this reuses that
    mechanism rather than inventing a second one.
    """
    for var in (cfg.env, cfg.fallback_env):
        if var and (val := os.environ.get(var)):
            p = Path(val)
            if not p.is_dir():
                raise RuntimeError(
                    f"{var}={val} is set but is not a directory. An eval config "
                    "must either be absent (and skip) or usable -- a bad path "
                    "would otherwise read as 'not staged' and pass vacuously."
                )
            return p
    return compose_for(cfg)


def compose_for(cfg: EvalConfig) -> Path | None:
    """Compose (or warm-reuse) a checkpoint for `cfg`; None if not possible here.

    Returns None -- so the caller SKIPS -- when the config has no recipe, or when
    ``GSW_EVAL_NO_COMPOSE`` is set to force the old staged-only behaviour. A
    compose FAILURE is not a skip: it raises, because a recipe that has stopped
    working is a real break and must not be reported as "this runner does not
    have the checkpoint".
    """
    recipe = COMPOSE_RECIPES.get(cfg.name)
    if recipe is None:
        return None
    # NO `GRANITE_SWITCH_E2E_MODELS=1` GATE HERE, DELIBERATELY. The e2e suites
    # need that opt-in because they are reachable from a plain `pytest tests/`
    # and composing pulls several GB from HF. These evals are not: they are
    # excluded from the default selection by pyproject's `-m "not deep"` plus
    # the `eval` marker, so the only ways in are `-m eval` and /gpu-test-eval.
    # ASKING FOR THE EVAL SUITE IS THE OPT-IN.
    #
    # This is also the only workable place for it. Requiring a second env var
    # would mean forwarding it to the GPU pod, and the pod is launched by
    # /opt/gsw/submit_and_poll.sh, which is baked into the runner image and
    # cannot be given a new flag from this repo. A gate nothing can satisfy is
    # how the suite ended up permanently skipping in the first place.
    if os.environ.get("GSW_EVAL_NO_COMPOSE"):
        return None

    base, adapters = recipe
    out_dir = _E2E_ROOT / f"eval-{cfg.name}"
    if (out_dir / "config.json").exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "granite_switch.composer.compose_granite_switch",
            "--base-model",
            base,
            "--adapters",
            *adapters,
            "--technology-filter",
            "lora",
            "--output",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"compose failed for eval config {cfg.name}:\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )
    if not (out_dir / "config.json").exists():
        raise RuntimeError(f"compose produced no config.json in {out_dir}")
    return out_dir


def prompts() -> list[str]:
    return json.loads((CASES / "prompts.json").read_text())["prompts"]


def fingerprint(model_dir: Path) -> str:
    """Identify the checkpoint a baseline was recorded against.

    config.json byte-for-byte plus the name and size of every weight shard. Not a
    content hash of the weights -- that is minutes of I/O for a 30B checkpoint and
    the shard sizes already change whenever the weights are re-exported.

    Without this, re-staging a different checkpoint under the same path would be
    silently evaluated against the old baseline, and the gate would be measuring
    nothing.
    """
    h = hashlib.sha256()
    h.update((model_dir / "config.json").read_bytes())
    for f in sorted(model_dir.glob("*.safetensors")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


def adapter_of(model_dir: Path) -> tuple[str, int]:
    """(name, control-token id) of the adapter this config evaluates.

    Index 0 by default, overridable by name with GSW_EVAL_ADAPTER. The NAME is
    recorded in the baseline, so re-composing a checkpoint with its adapters in a
    different order is caught rather than quietly evaluating a different adapter.
    """
    cfg = json.loads((model_dir / "config.json").read_text())
    names = cfg.get("adapter_names") or []
    ids = cfg.get("adapter_token_ids") or []
    if not names or not ids:
        raise RuntimeError(
            f"{model_dir} declares no adapters (adapter_names / adapter_token_ids "
            "are empty) -- it is a skinned checkpoint, not an eval target."
        )

    want = os.environ.get("GSW_EVAL_ADAPTER")
    idx = names.index(want) if want else 0
    if want and want not in names:
        raise RuntimeError(f"GSW_EVAL_ADAPTER={want!r} not in {names}")
    return names[idx], ids[idx]


def versions() -> dict[str, str]:
    """What the capture ran against. The baseline's provenance."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("vllm", "torch", "transformers"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "absent"
    return out


# ── capture ───────────────────────────────────────────────────────────────────


def _topk(d: dict[int, float]) -> dict[str, float]:
    """The TOP_K highest-logprob entries, string-keyed for JSON.

    vLLM's `prompt_logprobs` returns the k alternatives PLUS the actual token when
    that token is outside the top k, so a raw dict is sometimes k+1 entries wide.
    Truncating here keeps every captured position exactly k wide, which the
    metric's own k-sweep assumes.
    """
    return {
        str(t): float(v)
        for t, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
    }


def _run_vllm(
    model_dir: Path,
    adapter_seqs: list[list[int]],
    base_seqs: list[list[int]],
    continuations: list[list[int]] | None,
) -> tuple[list[list[int]], list[list[dict]], list[list[dict]]]:
    """Teacher-forced top-k logprobs for both passes; generates if asked to.

    `continuations is None` means "recording": greedily generate from the adapter
    pass first and return what it produced. Either way the DISTRIBUTIONS come from
    the same teacher-forced `prompt_logprobs` path, so a baseline is measured by
    exactly the code that later checks it.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from tests.shared.vllm_gpu_mem import gpu_mem_util, shutdown_llm

    kwargs: dict = {
        "model": str(model_dir),
        "enforce_eager": True,
        # OFF for reproducibility, not speed. The adapter and base passes send
        # near-identical token sequences -- they differ only by the leading
        # control token -- so with prefix caching on, whether a position is
        # recomputed or served from a cached block depends on what ran earlier in
        # the same engine. That made the capture depend on pass order: prompt 0,
        # the longest, was the only one whose numbers moved between two runs of
        # the same vLLM line (granite42_vllm 0.000000 -> 0.001857, granite5_vllm
        # 0.001914 -> 0.003221 across the 0.26 and 0.30 legs of run
        # 39556314476), while the shorter prompts reproduced byte-identically.
        "enable_prefix_caching": False,
    }
    # vLLM measures gpu_memory_utilization against TOTAL memory and refuses to
    # start if that exceeds what is free. These evals start engines back to back,
    # so a fixed value only works on an idle card.
    if (util := gpu_mem_util()) is not None:
        kwargs["gpu_memory_utilization"] = util

    llm = LLM(**kwargs)
    try:
        if continuations is None:
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=s) for s in adapter_seqs],
                SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
            )
            continuations = [list(o.outputs[0].token_ids) for o in outs]

        groups: list[list[list[dict]]] = []
        for seqs in (adapter_seqs, base_seqs):
            outs = llm.generate(
                [
                    TokensPrompt(prompt_token_ids=[*s, *c])
                    for s, c in zip(seqs, continuations, strict=True)
                ],
                # max_tokens=1 because prompt_logprobs is the measurement and vLLM
                # requires at least one generated token; the generated one is unused.
                SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOP_K),
            )
            group: list[list[dict]] = []
            for o, s, c in zip(outs, seqs, continuations, strict=True):
                pl = o.prompt_logprobs
                # Not a skip: a capture missing positions would otherwise be
                # compared as a shorter run and the metric would average over
                # whatever survived.
                if pl is None or len(pl) < len(s) + len(c):
                    raise RuntimeError(
                        "vLLM returned no usable prompt_logprobs "
                        f"(got {0 if pl is None else len(pl)}, need "
                        f"{len(s) + len(c)}). The eval measures distributions via "
                        "prompt_logprobs; without them there is nothing to compare."
                    )
                group.append(
                    [
                        # prompt_logprobs[j] is the distribution over prompt token j
                        # given tokens[:j], so continuation step i -- token index
                        # len(s)+i -- reads at len(s)+i.
                        _topk({t: lp.logprob for t, lp in pl[len(s) + i].items()})
                        for i in range(len(c))
                    ]
                )
            groups.append(group)
        return continuations, groups[0], groups[1]
    finally:
        # Release the device before the next config's engine starts; otherwise the
        # free-memory guard above trips on our own leftovers.
        shutdown_llm(llm)


def _run_hf(
    model_dir: Path,
    adapter_seqs: list[list[int]],
    base_seqs: list[list[int]],
    continuations: list[list[int]] | None,
) -> tuple[list[list[int]], list[list[dict]], list[list[dict]]]:
    """The same measurement through the HF backend: one forward per forced sequence."""
    import torch

    from granite_switch.hf import GraniteSwitchForCausalLM

    model = GraniteSwitchForCausalLM.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    if continuations is None:
        continuations = []
        for seq in adapter_seqs:
            ids = torch.tensor([seq], device=model.device)
            gen = model.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False)
            continuations.append([int(t) for t in gen[0][len(seq) :]])

    groups: list[list[list[dict]]] = []
    for seqs in (adapter_seqs, base_seqs):
        group: list[list[dict]] = []
        for seq, cont in zip(seqs, continuations, strict=True):
            ids = torch.tensor([[*seq, *cont]], device=model.device)
            with torch.no_grad():
                logits = model(ids).logits[0].float()
            steps = []
            for i in range(len(cont)):
                # logits[j] predicts token j+1, so step i -- token index len(seq)+i
                # -- is read at len(seq)+i-1.
                lp = torch.log_softmax(logits[len(seq) + i - 1], dim=-1)
                vals, idx = torch.topk(lp, TOP_K)
                steps.append({int(t): float(v) for t, v in zip(idx, vals, strict=True)})
            group.append([_topk(s) for s in steps])
        groups.append(group)
    return continuations, groups[0], groups[1]


def capture(
    cfg: EvalConfig, model_dir: Path, continuations: list[list[int]] | None = None
) -> dict:
    """Run both the adapter-on and base passes and package the result.

    `continuations` is the baseline's recorded token ids. The CHECK always passes
    them, so it measures the model at exactly the contexts the baseline measured;
    only recording leaves it None and generates.
    """
    from transformers import AutoTokenizer

    name, ctrl = adapter_of(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    texts = prompts()
    base_seqs = [tok(p, add_special_tokens=False)["input_ids"] for p in texts]
    # Control token at the sequence head -- the LoRA placement, and what
    # tests/integration/test_audio_serving_smoke.py does. Index i of
    # adapter_token_ids activates adapter i+1; 0 is base.
    adapter_seqs = [[ctrl, *s] for s in base_seqs]

    run = _run_vllm if cfg.backend == "vllm" else _run_hf
    # Both passes forced with the SAME continuation, so the liveness comparison
    # (adapter vs base) is position-aligned by construction too: it asks what the
    # un-adapted model would have predicted at the adapter's own contexts.
    continuations, adapter_dists, base_dists = run(
        model_dir, adapter_seqs, base_seqs, continuations
    )

    return {
        "config": cfg.name,
        "backend": cfg.backend,
        "description": cfg.description,
        "recorded_on": versions(),
        "checkpoint_fingerprint": fingerprint(model_dir),
        "adapter": {"name": name, "token_id": ctrl},
        "max_tokens": MAX_TOKENS,
        "top_k": TOP_K,
        "prompts": texts,
        "continuations": continuations,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "adapter_dists": adapter_dists,
        "base_dists": base_dists,
    }


# ── compare ───────────────────────────────────────────────────────────────────


def _pairwise(a: list[dict], b: list[dict], ks=(1, 5, 10, 20)) -> dict:
    """mean/max JSD and mean Jaccard distance over aligned positions, swept over k."""
    worst = {
        "mean_jsd": 0.0,
        "max_jsd": 0.0,
        "mean_jaccard_dist": 0.0,
        "max_jsd_pos": -1,
    }
    n = min(len(a), len(b))
    for k in ks:
        if k > TOP_K:
            continue
        jsds, jaccs = [], []
        for i in range(n):
            pa = {int(t): v for t, v in a[i].items()}
            pb = {int(t): v for t, v in b[i].items()}
            ids = list(set(topk_ids(pa, k)) | set(topk_ids(pb, k)))
            jsds.append(jsd_bits(pa, pb, ids))
            jaccs.append(1.0 - jaccard(topk_ids(pa, k), topk_ids(pb, k)))
        if not jsds:
            continue
        mx = max(jsds)
        worst["mean_jsd"] = max(worst["mean_jsd"], sum(jsds) / len(jsds))
        worst["mean_jaccard_dist"] = max(
            worst["mean_jaccard_dist"], sum(jaccs) / len(jaccs)
        )
        if mx > worst["max_jsd"]:
            worst["max_jsd"] = mx
            worst["max_jsd_pos"] = jsds.index(mx)
    return worst


def compare(baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    """(failures, report-lines). Empty failures means the eval passed."""
    fails: list[str] = []
    report: list[str] = []
    th = {**DEFAULT_THRESHOLDS, **baseline.get("thresholds", {})}

    # Structural mismatches first: a threshold comparison against a baseline
    # describing a different checkpoint, adapter or prompt set is meaningless, and
    # reporting it as a quality regression would send the reader hunting a bug that
    # is actually a staging mistake.
    if baseline["prompts"] != current["prompts"]:
        fails.append(
            "prompt set changed since the baseline was recorded -- re-record it "
            "deliberately (tests/eval/cases/prompts.json is committed for this reason)"
        )
        return fails, report
    if baseline["adapter"]["name"] != current["adapter"]["name"]:
        fails.append(
            f"baseline evaluates adapter {baseline['adapter']['name']!r} but the "
            f"checkpoint now offers {current['adapter']['name']!r} first -- the "
            "checkpoint was re-composed with a different adapter order"
        )
        return fails, report
    if "continuations" not in baseline:
        fails.append(
            "baseline predates teacher forcing (no 'continuations' key). It was "
            "recorded from a self-conditioning greedy rollout, where one flipped "
            "near-tie desynchronizes every later position and the metric compares "
            "unrelated contexts. Re-record it on a KNOWN-GOOD line; it cannot be "
            "compared against a forced capture."
        )
        return fails, report
    if baseline["continuations"] != current["continuations"]:
        # Unreachable via test_adapter_evals, which forces the baseline's own
        # continuations. Guards a caller that forgets to pass them -- which would
        # silently reintroduce the rollout bug this measurement exists to avoid.
        fails.append(
            "current capture used different continuations from the baseline, so "
            "the comparison is not position-aligned. capture() must be called "
            "with baseline['continuations']."
        )
        return fails, report
    if baseline["checkpoint_fingerprint"] != current["checkpoint_fingerprint"]:
        fails.append(
            "checkpoint changed since the baseline was recorded "
            f"({baseline['checkpoint_fingerprint']} -> "
            f"{current['checkpoint_fingerprint']}). The gate cannot tell a vLLM "
            "regression from a different checkpoint, so it refuses to guess. "
            "Re-stage the recorded checkpoint, or re-record the baseline."
        )
        return fails, report

    report.append(f"baseline recorded on vllm {baseline['recorded_on'].get('vllm')}")
    report.append(f"current run     on vllm {current['recorded_on'].get('vllm')}")

    # 1. Regression: this run vs the signed-off run.
    for i, prompt in enumerate(current["prompts"]):
        m = _pairwise(baseline["adapter_dists"][i], current["adapter_dists"][i])
        report.append(
            f"  prompt {i}: mean JSD {m['mean_jsd']:.6f}  max {m['max_jsd']:.6f}"
            f" @pos {m['max_jsd_pos']}  mean(1-Jacc) {m['mean_jaccard_dist']:.4f}"
        )
        for key in ("mean_jsd", "max_jsd", "mean_jaccard_dist"):
            if m[key] > th[key]:
                fails.append(
                    f"prompt {i} ({prompt[:40]!r}...): {key}={m[key]:.6f} > {th[key]}"
                )

    # 2. Liveness: the adapter must still change the output.
    effects = [
        _pairwise(current["adapter_dists"][i], current["base_dists"][i])["mean_jsd"]
        for i in range(len(current["prompts"]))
    ]
    eff = sum(effects) / len(effects) if effects else 0.0
    report.append(f"  adapter effect (vs base): mean JSD {eff:.6f}")
    if eff < th["min_adapter_effect_jsd"]:
        fails.append(
            f"adapter effect {eff:.6f} < {th['min_adapter_effect_jsd']}: the "
            f"adapter {current['adapter']['name']!r} is not measurably changing the "
            "output. Either its control token is no longer routing, or the LoRA "
            "path is not being applied. This is the silent-failure mode the eval "
            "exists for -- do NOT re-record the baseline to make it pass."
        )
    if math.isnan(eff):
        fails.append("adapter effect is NaN -- the model produced non-finite logits")

    return fails, report


def baseline_path(name: str) -> Path:
    return BASELINES / f"{name}.json"


def load_baseline(name: str) -> dict | None:
    p = baseline_path(name)
    return json.loads(p.read_text()) if p.is_file() else None
