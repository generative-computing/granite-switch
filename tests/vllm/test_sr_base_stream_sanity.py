# SPDX-License-Identifier: Apache-2.0
"""vLLM base-model sanity checks for the Shadow-Residual decoder.

Both run the REAL composed SR model under vLLM (env-configured) vs the bare base model —
gpu-marked, skipped when assets/CUDA are absent.

TEST 1 — adapter OFF == base.  With the adapter not triggered (no control token in the
  prompt), the SR path reduces to base: per-position top-1 agreement ~identical. The two
  models run in SEPARATE subprocesses (vLLM does not free GPU memory on ``del``).

TEST 2 — base stream == base at every layer (adapter ON).  W_cross is base->adapter only
  and the base half never receives a delta/shunt, so with the adapter ACTIVE the per-layer
  base half (rows [:M] of the [2M,H] stacked stream, hooked on each
  ShadowResidualDecoderLayer) matches the base model's per-layer hidden states — at every
  layer. Composed vLLM (low mem-util) and the small base HF model co-reside in one process.

Env: SR_COMPOSED_DIR (req), SR_BASE_MODEL, SR_EVAL_JSONL (req), SR_SANITY_N (default 2).
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

COMPOSED = os.environ.get("SR_COMPOSED_DIR", "")
BASE_MODEL = os.environ.get("SR_BASE_MODEL", "ibm-granite/granite-4.1-3b")
EVAL_JSONL = os.environ.get("SR_EVAL_JSONL", "")
N_ROWS = int(os.environ.get("SR_SANITY_N", "2"))
T1_MIN_TOP1 = float(os.environ.get("SR_T1_MIN_TOP1", "0.97"))
# Base stream vs UPSTREAM base HF: every layer but the last matches at cos >= 0.99999;
# the final layer sits at ~0.9988. granite-switch documents its fused QKV/gate-up projections
# as not bit-exact against upstream HF's separate ones (see CLAUDE.md), which is the expected
# source of an accumulating bf16 difference of that size, though this test does not isolate
# it. 0.998 therefore catches a structural divergence — a stream that is not base-equivalent
# at all — without failing on that expected drift.
T2_MIN_COS = float(os.environ.get("SR_T2_MIN_COS", "0.998"))
MAX_MODEL_LEN = int(os.environ.get("SR_MAX_MODEL_LEN", "16384"))


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


def _fixdoc(d):
    if not d:
        return d
    return [
        x if isinstance(x, dict) else {"title": "Context", "text": str(x)} for x in d
    ]


def _prompts(n):
    """Per-row (on = with control token, off = without) using the composed tokenizer."""
    from transformers import AutoTokenizer

    cfg = json.load(open(os.path.join(COMPOSED, "config.json")))
    tok = AutoTokenizer.from_pretrained(COMPOSED)
    name = (cfg.get("adapter_names") or ["sr-answerability"])[0]
    rows = []
    with open(EVAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            kw = dict(
                tools=r.get("tools"),
                documents=_fixdoc(r.get("documents")),
                add_generation_prompt=True,
                tokenize=False,
            )
            on = list(
                tok(
                    tok.apply_chat_template(r["messages"], adapter_name=name, **kw),
                    add_special_tokens=False,
                ).input_ids
            )
            off = list(
                tok(
                    tok.apply_chat_template(r["messages"], **kw),
                    add_special_tokens=False,
                ).input_ids
            )
            rows.append({"on": on, "off": off})
            if len(rows) >= n:
                break
    return rows


# ─────────────────────── TEST 1: adapter-off == base (subprocess per model) ───────────────────────
def _worker_offlogprobs(model_dir, register_sr, spec_path, out_path):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if register_sr:
        from granite_switch.vllm import register

        register()
    from vllm import LLM, SamplingParams

    rows = json.load(open(spec_path))
    kw = dict(
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_logprobs=32,
        gpu_memory_utilization=0.85,
    )
    if register_sr:
        kw["hf_overrides"] = {"architectures": ["ShadowResidualForCausalLM"]}
    llm = LLM(model=model_dir, **kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=20)
    out = []
    for row in rows:
        o = llm.generate([{"prompt_token_ids": row["off"]}], sp)[0]
        per = []
        for ent in o.prompt_logprobs or []:
            per.append(int(max(ent, key=lambda t: ent[t].logprob)) if ent else None)
        out.append(per)
    json.dump(out, open(out_path, "w"))


def test_adapter_off_equals_base(tmp_path):
    _skip_checks()
    rows = _prompts(N_ROWS)
    spec = str(tmp_path / "rows.json")
    json.dump(rows, open(spec, "w"))
    comp_out, base_out = str(tmp_path / "comp.json"), str(tmp_path / "base.json")
    subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--offlogprobs",
            "composed",
            spec,
            comp_out,
        ],
        check=True,
        env=os.environ,
    )
    subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--offlogprobs",
            "base",
            spec,
            base_out,
        ],
        check=True,
        env=os.environ,
    )
    comp, base = json.load(open(comp_out)), json.load(open(base_out))
    agree = tot = 0
    for cr, br in zip(comp, base):
        for ct, bt in zip(cr, br):
            if ct is None or bt is None:
                continue
            tot += 1
            agree += int(ct == bt)
    top1 = agree / tot if tot else 0.0
    print(f"TEST1 adapter-off==base: positions={tot} top1_agreement={top1:.4f}")
    assert top1 >= T1_MIN_TOP1, f"adapter-off vs base top-1 {top1:.4f} < {T1_MIN_TOP1}"


# ─────────────────────── TEST 2: base stream == base per layer (in-process) ───────────────────────
def _find_sr_layers(llm):
    from granite_switch.vllm.decoder.shadow_residual.decoder import (
        ShadowResidualDecoderLayer,
    )

    eng = getattr(llm, "llm_engine", None)
    objs = []
    for path in (
        "model_executor.driver_worker.model_runner.model",
        "engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        "engine_core.model_executor.driver_worker.model_runner.model",
    ):
        obj = eng
        for a in path.split("."):
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            objs.append(obj)
    for m in objs:
        layers = [
            mod for mod in m.modules() if isinstance(mod, ShadowResidualDecoderLayer)
        ]
        if layers:
            return layers
    return []


def test_base_stream_equals_base_all_layers():
    _skip_checks()
    import gc

    import torch
    import torch.nn.functional as F

    row = _prompts(1)[0]
    on, off = row["on"], row["off"]
    # The composed model runs the SR stacked stream on `on` (control token -> adapter ON).
    # The bare base model cannot embed the control token (out of base vocab), so it runs
    # `off` (the same sequence with the original invocation token). on/off differ only at
    # the control-token index; that token is KV-hidden, so every other base-stream position
    # must still match the base model. Compare at the common (identical-token) positions.
    assert len(on) == len(off), "expected SUBSTITUTE placement (equal length on/off)"
    keep = [i for i in range(len(on)) if on[i] == off[i]]
    assert len(keep) >= len(on) - 2, "on/off differ at more than the control token"

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from granite_switch.vllm import register

    register()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=COMPOSED,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.45,
        hf_overrides={"architectures": ["ShadowResidualForCausalLM"]},
    )
    layers = _find_sr_layers(llm)
    if not layers:
        pytest.skip("could not reach the in-process vLLM SR decoder layers to hook")
    cap = {}

    def mk(i):
        def hook(_m, _in, out):
            hs = out[0] if isinstance(out, tuple) else out
            m = hs.shape[0] // 2
            cap[i] = hs[:m].detach().float().cpu()

        return hook

    handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    llm.generate(
        [{"prompt_token_ids": on}], SamplingParams(max_tokens=1, temperature=0.0)
    )
    for h in handles:
        h.remove()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM

    base = (
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    bcap = {}

    def mkb(i):
        def hook(_m, _in, out):
            bcap[i] = (
                (out[0] if isinstance(out, tuple) else out)[0].detach().float().cpu()
            )

        return hook

    bh = [l.register_forward_hook(mkb(i)) for i, l in enumerate(base.model.layers)]
    with torch.no_grad():
        base(input_ids=torch.tensor([off], device="cuda"))
    for h in bh:
        h.remove()

    idx = torch.tensor(keep)
    worst = 1.0
    nl = min(len(cap), len(bcap))
    assert nl > 0, "no captured layers"
    for i in range(nl):
        a, b = cap[i].index_select(0, idx), bcap[i].index_select(0, idx)
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        worst = min(worst, cos)
        if i < 3 or i >= nl - 2 or cos < T2_MIN_COS:
            print(f"TEST2 layer {i:2d}: cos={cos:.6f}")
    print(
        f"TEST2 base-stream==base: worst per-layer cos={worst:.6f} "
        f"over {nl} layers, {len(keep)}/{len(on)} positions"
    )
    assert worst >= T2_MIN_COS, (
        f"base stream diverges from base (worst cos {worst:.6f})"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--offlogprobs":
        _worker_offlogprobs(
            COMPOSED if sys.argv[2] == "composed" else BASE_MODEL,
            sys.argv[2] == "composed",
            sys.argv[3],
            sys.argv[4],
        )
