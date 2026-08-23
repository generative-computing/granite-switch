# SPDX-License-Identifier: Apache-2.0
"""Answerability decode-variant equivalence — shared across the HF and vLLM SR paths.

A composed Granite-Switch checkpoint can be decoded three ways that must agree, because
they carry the SAME embedded adapter:

  vllm           the composed model under the granite-switch vLLM model (this repo), eager
  vllm_compiled  the same, with torch.compile + CUDA graphs ON -- i.e. the path production
                 serving actually runs. Every other leg here is eager, so without this leg an
                 SR bug that only appears under compilation or graph capture would pass
                 unnoticed. That is a live risk for shadow residual specifically: the decoder
                 doubles the token dimension into a [2M,H] stacked stream, and shape-dependent
                 reshaping is exactly what piecewise compile and cudagraph shape bucketing
                 tend to mishandle.
  hf             the composed model under the granite-switch HF backend
  peft           the adapter extracted from that composed model, run standalone under
                 transformers (via the shadow-residual repo's SR-PEFT loader)

and a fourth that must DISAGREE — the bare base model. At the answerability decision
position (right after a teacher-forced opening quote), on the top-k next-token support:

  * PARITY (gated) — every pair among the legs in ``SR_PARITY_GATE_LEGS`` (by default
    every leg that carries the adapter) must be SIMILAR: mean Jaccard distance (1 - Jaccard
    of the two top-k supports) <= SR_JACCARD_DIST_THRESH (default 0.10) AND mean JSD <=
    SR_JSD_THRESH (default 0.02 bits) at every swept k. On a composed SR checkpoint the pairs
    land two to three orders of magnitude inside both gates, so these thresholds bound a real
    equivalence rather than absorbing a known discrepancy — a leg that only just passes should
    be read as a regression, not as within tolerance. A leg outside the gate set is still
    measured and printed, just not asserted.
  * SEPARATION (gated) — every present variant vs base EXCEEDS the Jaccard threshold. Base is
    prompted WITHOUT a control token, so the adapter never fires: it lands ~1 full bit of JSD
    away and never puts the ``unanswerable`` token in its top-k at all.

Real-model behavioral check (env-configured composed checkpoint + eval data), so it is
``gpu``-marked and skips when assets/CUDA are absent. Each leg runs in its OWN subprocess
(vLLM does not free GPU memory on ``del``; process exit does) — same pattern as the other
vLLM tests in this suite. The ``vllm``, ``hf`` and ``base`` legs always run; ``peft`` runs when
the ``shadow-residual`` repo is installed and SR_PEFT_ADAPTER is set. A leg whose backend is
missing prints a SKIP line and drops out of the comparison rather than failing.

The ``peft`` leg needs an adapter whose key paths match the loading model's module paths --
see ``_audit_lora_loaded``, which fails loudly on the silent partial-load that PEFT's
zero-init otherwise hides.

Env: SR_COMPOSED_DIR (req), SR_BASE_MODEL, SR_EVAL_JSONL (req), SR_PEFT_ADAPTER (opt),
     SR_VARIANT_N (rows/class, default 8), SR_VARIANT_KS (default 5,10,20,50),
     SR_PARITY_GATE_LEGS (default vllm,vllm_compiled,hf,peft),
     SR_VLLM_COMPILED_LEG (default 1; set 0 to drop the compiled leg on a tight GPU).
"""

import json
import math
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

COMPOSED = os.environ.get("SR_COMPOSED_DIR", "")
BASE_MODEL = os.environ.get("SR_BASE_MODEL", "ibm-granite/granite-4.1-3b")
EVAL_JSONL = os.environ.get("SR_EVAL_JSONL", "")
PEFT_ADAPTER = os.environ.get("SR_PEFT_ADAPTER", "")
N_PER_CLASS = int(os.environ.get("SR_VARIANT_N", "8"))
KS = [int(x) for x in os.environ.get("SR_VARIANT_KS", "5,10,20,50").split(",")]
K_COLLECT = int(os.environ.get("SR_VARIANT_KCOLLECT", "200"))
JDIST_THRESH = float(os.environ.get("SR_JACCARD_DIST_THRESH", "0.10"))
JSD_THRESH = float(os.environ.get("SR_JSD_THRESH", "0.02"))
# Legs whose mutual parity is hard-gated. They all decode the SAME embedded adapter and agree
# to ~5e-5 JSD in practice. A leg outside this set is measured and printed but not asserted.
PARITY_GATE_LEGS = set(
    x
    for x in os.environ.get("SR_PARITY_GATE_LEGS", "vllm,vllm_compiled,hf,peft").split(
        ","
    )
    if x
)
# The compiled-vLLM leg costs one extra engine load; set 0 to skip it on a tight GPU.
WANT_COMPILED = os.environ.get("SR_VLLM_COMPILED_LEG", "1") not in ("0", "false", "")
LABEL_ANS = os.environ.get("SR_LABEL_ANS", '"answerable"')
LABEL_UNANS = os.environ.get("SR_LABEL_UNANS", '"unanswerable"')
MAX_MODEL_LEN = int(os.environ.get("SR_MAX_MODEL_LEN", "16384"))
_LOG2 = math.log(2.0)


# ─────────────────────────── prompt spec (CPU, orchestrator) ───────────────────────────
def _fixdoc(d):
    if not d:
        return d
    return [
        x if isinstance(x, dict) else {"title": "Context", "text": str(x)} for x in d
    ]


def _select_rows(path, n):
    want = {"answerable": [], "unanswerable": []}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gt = str(row.get("ground_truth", "")).strip().strip('"').lower()
            if gt in want and len(want[gt]) < n:
                want[gt].append(row)
            if all(len(v) >= n for v in want.values()):
                break
    return want["answerable"] + want["unanswerable"]


def _label_branch(tok):
    a = list(tok(LABEL_ANS, add_special_tokens=False).input_ids)
    u = list(tok(LABEL_UNANS, add_special_tokens=False).input_ids)
    n = 0
    while n < min(len(a), len(u)) and a[n] == u[n]:
        n += 1
    assert n < len(a) and n < len(u), f"labels have no branch point: {a} {u}"
    return a[:n]


def _build_spec():
    from transformers import AutoTokenizer

    ccfg = json.load(open(os.path.join(COMPOSED, "config.json")))
    tok = AutoTokenizer.from_pretrained(COMPOSED)
    ctrl = int(ccfg["adapter_token_ids"][0])
    adapter_name = (ccfg.get("adapter_names") or ["sr-answerability"])[0]
    shared = _label_branch(tok)
    base_vocab = min(int(t) for t in ccfg["adapter_token_ids"])
    examples = []
    for row in _select_rows(EVAL_JSONL, N_PER_CLASS):
        kw = dict(
            tools=row.get("tools"),
            documents=_fixdoc(row.get("documents")),
            add_generation_prompt=True,
            tokenize=False,
        )
        on = list(
            tok(
                tok.apply_chat_template(
                    row["messages"], adapter_name=adapter_name, **kw
                ),
                add_special_tokens=False,
            ).input_ids
        )
        off = list(
            tok(
                tok.apply_chat_template(row["messages"], **kw), add_special_tokens=False
            ).input_ids
        )
        assert ctrl in on, (
            "control token not placed — the SR leg would run base-equivalent"
        )
        examples.append({"on": on, "off": off})
    return dict(examples=examples, shared=shared, base_vocab=base_vocab)


# ─────────────────────────── per-leg workers (each its OWN process) ───────────────────────────
def _worker_vllm(spec, model_dir, register_sr, eager=True):
    """One vLLM leg. ``eager=False`` enables torch.compile + CUDA graphs.

    ``enforce_eager`` sets both ``CompilationMode.NONE`` and ``CUDAGraphMode.NONE`` (see
    vllm/config/vllm.py). Eager is the default here because it keeps the measured logprobs a
    property of the model rather than of Inductor's kernel selection, and because the hook-based
    sanity test cannot observe activations through a compiled graph at all. ``eager=False`` is
    the production path and is covered by its own leg.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if register_sr:
        from granite_switch.vllm import register

        register()
    from vllm import LLM, SamplingParams

    kw = dict(
        enforce_eager=eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_logprobs=K_COLLECT,
        gpu_memory_utilization=0.85,
    )
    if register_sr:
        kw["hf_overrides"] = {"architectures": ["ShadowResidualForCausalLM"]}
    llm = LLM(model=model_dir, **kw)
    key = "on" if register_sr else "off"  # base leg: adapter-off prompt (no ctrl)
    reqs = [{"prompt_token_ids": ex[key] + spec["shared"]} for ex in spec["examples"]]
    outs = llm.generate(
        reqs, SamplingParams(max_tokens=1, temperature=0.0, logprobs=K_COLLECT)
    )
    return [
        {str(int(t)): float(v.logprob) for t, v in o.outputs[0].logprobs[0].items()}
        for o in outs
    ]


def _audit_lora_loaded(model):
    """Fail loudly when a whole LoRA module family failed to load.

    PEFT initializes ``lora_B`` to zeros, so a checkpoint key whose path does not match the
    model's module path leaves that module's delta at exactly zero. Nothing warns: the key
    COUNT still matches, generation still works, and the adapter simply runs at reduced
    strength -- which shows up only as a small, systematic distribution shift. A module family
    whose every ``lora_B`` is all-zero is that failure, so check for it directly.
    """
    import collections

    fam = collections.defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        if "lora_B" not in name:
            continue
        key = name.split("layers.")[-1]
        key = ".".join(key.split(".")[1:])
        key = key.replace(".default.weight", "").replace(".weight", "")
        fam[key][0] += 1
        fam[key][1] += int(bool((p == 0).all().item()))
    dead = sorted(k for k, (tot, zero) in fam.items() if tot and zero == tot)
    if dead:
        raise AssertionError(
            f"adapter only partially loaded from {PEFT_ADAPTER}: every lora_B is zero for "
            f"{dead}. Those checkpoint keys do not match the model's module paths, so PEFT "
            f"left them at their zero init and the adapter runs at reduced strength. The MLP "
            f"projections in particular must be nested as `mlp.gate_proj` / `mlp.up_proj` / "
            f"`mlp.down_proj` to match ShadowResidualMLP and upstream HF Granite."
        )


def _audit_sr_dual_stream(model):
    """Fail loudly when the composed checkpoint did not build as dual-stream SR.

    A GraniteSwitch config whose ``dual_stream`` is off builds plain switched-LoRA layers
    instead: the adapter still fires and still classifies correctly, so the only symptom is a
    distribution shift -- the same silent shape as a partially-loaded adapter. Assert the SR
    layer type is actually present rather than trusting the config.
    """
    from granite_switch.hf.modeling_granite_switch import SRSwitchDecoderLayer

    n = sum(1 for m in model.modules() if isinstance(m, SRSwitchDecoderLayer))
    if n == 0:
        raise AssertionError(
            f"{COMPOSED} did not build SR dual-stream layers under granite_switch.hf (no "
            f"SRSwitchDecoderLayer present) -- this leg would be measuring a plain "
            f"switched-LoRA model rather than shadow residual."
        )
    return n


def _worker_hf(spec):
    """The composed checkpoint under the granite-switch HF backend (fused projections)."""
    import torch
    from transformers import AutoModelForCausalLM

    import granite_switch.hf  # noqa: F401  registers granite_switch -> GraniteSwitchForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(COMPOSED, dtype=torch.bfloat16)
    except TypeError:  # transformers < 5 spelling
        model = AutoModelForCausalLM.from_pretrained(
            COMPOSED, torch_dtype=torch.bfloat16
        )
    model = model.to("cuda").eval()
    print(f"hf leg: {type(model).__name__}, {_audit_sr_dual_stream(model)} SR layers")

    dists = []
    for ex in spec["examples"]:
        ids = ex["on"] + spec["shared"]  # control-token prompt, same as the vllm leg
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0]
        lp = torch.log_softmax(logits[len(ids) - 1].float(), dim=-1)
        v, idx = lp.topk(K_COLLECT)
        dists.append({str(int(i)): float(x) for i, x in zip(idx.tolist(), v.tolist())})
    return dists


def _worker_peft(spec):
    import torch
    from shadow_residual.peft_shadow_residual import load_shadow_residual_peft_model

    model = (
        load_shadow_residual_peft_model(
            BASE_MODEL, PEFT_ADAPTER, torch_dtype=torch.bfloat16, shared_base_kv=True
        )
        .to("cuda")
        .eval()
    )
    _audit_lora_loaded(model)
    dists = []
    for ex in spec["examples"]:
        ids = ex["off"] + spec["shared"]  # peft gates on the invocation tokens
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0]
        lp = torch.log_softmax(logits[len(ids) - 1].float(), dim=-1)
        v, idx = lp.topk(K_COLLECT)
        dists.append({str(int(i)): float(x) for i, x in zip(idx.tolist(), v.tolist())})
    return dists


def _run_worker(leg, spec_path, out_path):
    spec = json.load(open(spec_path))
    if leg == "vllm":
        d = _worker_vllm(spec, COMPOSED, register_sr=True)
    elif leg == "vllm_compiled":
        d = _worker_vllm(spec, COMPOSED, register_sr=True, eager=False)
    elif leg == "base":
        d = _worker_vllm(spec, BASE_MODEL, register_sr=False)
    elif leg == "hf":
        d = _worker_hf(spec)
    elif leg == "peft":
        d = _worker_peft(spec)
    else:
        raise SystemExit(f"unknown leg {leg}")
    json.dump({"dists": d}, open(out_path, "w"))


# ─────────────────────────── metrics ───────────────────────────
def _topk(dump, k, vmax):
    it = [(int(t), v) for t, v in dump.items() if int(t) < vmax]
    it.sort(key=lambda kv: -kv[1])
    return [t for t, _ in it[:k]]


def _restrict(dump, support, vmax):
    avail = {int(t): v for t, v in dump.items() if int(t) < vmax}
    bound = min(avail.values()) if avail else -60.0
    raw = [math.exp(avail.get(t, bound)) for t in support]
    tot = sum(raw)
    return [x / tot for x in raw] if tot > 0 else [0.0] * len(support)


def _kl_bits(p, q):
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0.0) / _LOG2


def _jsd_bits(p, q):
    m = [0.5 * (a + b) for a, b in zip(p, q)]
    return 0.5 * _kl_bits(p, m) + 0.5 * _kl_bits(q, m)


def _pair(da, db, k, vmax):
    ta, tb = set(_topk(da, k, vmax)), set(_topk(db, k, vmax))
    support = sorted(ta | tb)
    if not support:
        return 0.0, 0.0
    p, q = _restrict(da, support, vmax), _restrict(db, support, vmax)
    union = ta | tb
    return 1.0 - (len(ta & tb) / len(union) if union else 1.0), _jsd_bits(p, q)


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


# ─────────────────────────── the test ───────────────────────────
def _skip_checks():
    if not COMPOSED or not os.path.exists(os.path.join(COMPOSED, "config.json")):
        pytest.skip(f"SR_COMPOSED_DIR not set/readable ({COMPOSED!r})")
    if not EVAL_JSONL or not os.path.exists(EVAL_JSONL):
        pytest.skip(f"SR_EVAL_JSONL not set/readable ({EVAL_JSONL!r})")
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU required")
    except Exception:
        pytest.skip("torch unavailable")


def test_answerability_variant_equivalence(tmp_path):
    _skip_checks()
    spec = _build_spec()
    n = len(spec["examples"])
    vmax = spec["base_vocab"]
    spec_path = str(tmp_path / "spec.json")
    json.dump(spec, open(spec_path, "w"))

    plan = [("vllm", True), ("base", True)]  # always run
    if WANT_COMPILED:
        plan.append(("vllm_compiled", True))
    else:
        print("SKIP vllm_compiled leg: SR_VLLM_COMPILED_LEG=0")
    try:
        import granite_switch.hf  # noqa: F401
        from granite_switch.hf.modeling_granite_switch import (  # noqa: F401
            SRSwitchDecoderLayer,
        )

        plan.append(("hf", True))
    except Exception as exc:
        print(f"SKIP hf leg: granite_switch.hf SR backend unavailable ({exc})")

    have_peft = False
    if PEFT_ADAPTER and os.path.exists(
        os.path.join(PEFT_ADAPTER, "adapter_config.json")
    ):
        try:
            import shadow_residual.peft_shadow_residual  # noqa: F401

            have_peft = True
        except Exception:
            have_peft = False
    if have_peft:
        plan.append(("peft", True))
    else:
        print(
            "SKIP peft leg: shadow_residual not installed or SR_PEFT_ADAPTER unavailable"
        )
    # All three gated legs decode the SAME embedded adapter, so every pair among them is an
    # equivalence claim about the implementations rather than about the weights.

    legs = {}
    for leg, _ in plan:
        out = str(tmp_path / f"{leg}.json")
        subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                leg,
                spec_path,
                out,
            ],
            check=True,
            env=os.environ,
        )
        legs[leg] = json.load(open(out))["dists"]

    group = [l for l in ("vllm", "vllm_compiled", "hf", "peft") if l in legs]
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            gated = a in PARITY_GATE_LEGS and b in PARITY_GATE_LEGS
            tag = "GATED" if gated else "reference"
            for k in KS:
                pairs = [_pair(legs[a][r], legs[b][r], k, vmax) for r in range(n)]
                jd, js = _mean([p[0] for p in pairs]), _mean([p[1] for p in pairs])
                print(f"PARITY[{tag}] {a}/{b} k={k}: 1-Jaccard={jd:.4f} JSD={js:.6f}")
                if gated:
                    assert jd <= JDIST_THRESH, (
                        f"{a}/{b} k={k}: 1-Jaccard {jd:.4f} > {JDIST_THRESH}"
                    )
                    assert js <= JSD_THRESH, (
                        f"{a}/{b} k={k}: JSD {js:.6f} > {JSD_THRESH}"
                    )
    for a in group:
        seps = [
            max(_pair(legs[a][r], legs["base"][r], k, vmax)[0] for r in range(n))
            for k in KS
        ]
        print(f"SEPARATION {a}/base: min 1-Jaccard over k = {min(seps):.4f}")
        assert min(seps) > JDIST_THRESH, (
            f"{a} not separated from base ({min(seps):.4f})"
        )


if __name__ == "__main__":
    # subprocess worker: --worker <leg> <spec_path> <out_path>
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], sys.argv[3], sys.argv[4])
