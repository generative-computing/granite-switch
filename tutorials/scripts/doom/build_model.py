# SPDX-License-Identifier: Apache-2.0
"""Stage the demo's adapters and compose them into one Granite Switch checkpoint.

Stand-in adapters (random weights, the real shape) for the latency test, which
does not need trained behavior. ``--kind lora`` builds the same six adapters as
plain LoRA, the baseline aLoRA is measured against::

    python build_model.py standin --base /path/to/granite-4.1-3b --out models/standin
    python build_model.py standin --kind lora --base /path/to/granite-4.1-3b \
        --out models/standin-lora

Trained adapters, as written by ``train_alora.py`` to ``<runs>/<name>/``::

    python build_model.py compose --runs runs/round0 --base /path/to/granite-4.1-3b \
        --out models/doom-switch

Each adapter is staged in the library layout the composer expects,
``<stage>/<name>/<target_model>/{alora,lora}/`` with an ``io.yaml``; the
directory name tells the composer which kind it is (aLoRA: control token before
the assistant header; LoRA: at position 0). Then the stock compose CLI runs
with the local paths.

Adapter settings follow IBM's shipped Granite aLoRAs: rank 32 on every linear
layer, attention and MLP.

Parity of the composed checkpoint against PEFT, on each adapter's held-out
states (``heldout_preds.jsonl`` from ``train_alora.py``); the gate is >= 99%::

    python build_model.py verify --runs runs/round0 --model models/doom-switch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy import ADAPTERS, alora_invocation_ids

DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_RANK = 32
MARGIN = 0.05  # verify: PEFT top-1 minus top-2 probability for a "clear" state
KINDS = ("alora", "lora")


def io_yaml(name: str) -> dict:
    return {
        "name": name,
        "model": None,
        "response_format": None,
        "transformations": None,
        "instruction": None,
        "parameters": {"max_completion_tokens": 1, "temperature": 0.0},
        "sentence_boundaries": None,
    }


def stage(
    adapter_dir: Path, name: str, stage_root: Path, target_model: str, kind: str
) -> Path:
    """Copy a PEFT adapter into ``<stage_root>/<name>/<target_model>/<kind>/``."""
    dst = stage_root / name / target_model / kind
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(adapter_dir / f, dst / f)
    (dst / "io.yaml").write_text(yaml.safe_dump(io_yaml(name), sort_keys=False))
    return dst


def compose(
    paths: list[Path],
    base: str,
    out: Path,
    asr_model: str | None = None,
    asr_device="cpu",
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--adapters",
        *map(str, paths),
        "--base-model",
        base,
        "--output",
        str(out),
    ]
    if asr_model:
        # Speech in through the checkpoint's own ASR cascade (docs/AUDIO.md). A
        # conformer's BatchNorm cannot run in float16, so the ASR stays float32.
        cmd += ["--enable-audio", "--asr-model", asr_model, "--asr-dtype", "float32"]
        cmd += ["--asr-device", asr_device]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def make_standins(
    base: str, out_dir: Path, rank: int, targets: tuple[str, ...], seed: int, kind: str
) -> dict[str, Path]:
    """Write one random-weight adapter per name (same shape as the real ones)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    extra = (
        {"alora_invocation_tokens": alora_invocation_ids(tok)}
        if kind == "alora"
        else {}
    )
    cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=rank,
        target_modules=list(targets),
        lora_dropout=0.0,
        bias="none",
        **extra,
    )
    peft_model = get_peft_model(model, cfg)
    gen = torch.Generator().manual_seed(seed)
    written = {}
    for name in ADAPTERS:
        with torch.no_grad():
            for pname, p in peft_model.named_parameters():
                if "lora_" in pname:
                    p.copy_((torch.randn(p.shape, generator=gen) * 0.02).to(p.dtype))
        d = out_dir / "peft" / name
        peft_model.save_pretrained(str(d))
        written[name] = d
        print(f"stand-in {kind} {name}: r={rank} targets={','.join(targets)} -> {d}")
    return written


def verify(
    runs: Path, model: str, router_runs: Path | None, limit: int, layout: str = "log"
) -> dict:
    """Argmax agreement: composed checkpoint in vLLM vs the PEFT adapter, on the
    held-out rows each adapter's training run wrote (history ids + state)."""
    from policy import ROUTER, VLLMPolicy

    pol = VLLMPolicy(model, warmup=5, max_num_seqs=64, layout=layout)
    report = {}
    for name in ADAPTERS:
        root = router_runs if (name == ROUTER and router_runs) else runs
        path = root / name / "heldout_preds.jsonl"
        rows = [json.loads(line) for line in open(path)][:limit]
        if name == ROUTER:
            routes = [pol.route(r["state"]) for r in rows]
            got, dists = [x.adapter for x in routes], [x.probs for x in routes]
        else:
            got, dists = [], []
            for i in range(0, len(rows), 64):
                chunk = rows[i : i + 64]
                decs = pol.decide_games(
                    [(r["history_ids"], r["state"]) for r in chunk],
                    [[name]] * len(chunk),
                )
                got += [d[name].action for d in decs]
                dists += [d[name].probs for d in decs]
        agree = sum(g == r["peft"] for g, r in zip(got, rows)) / max(1, len(rows))
        acc = sum(g == r["label"] for g, r in zip(got, rows)) / max(1, len(rows))
        rep = {
            "n": len(rows),
            "agree_with_peft": round(agree, 4),
            "acc_vs_label": round(acc, 4),
        }
        if rows and "peft_probs" in rows[0]:
            # Disagreements between two bf16 implementations should sit on
            # near-ties; the distance between whole distributions says more.
            tv, clear = [], []
            for g, r, q in zip(got, rows, dists):
                p = r["peft_probs"]
                tv.append(0.5 * sum(abs(p[c] - q.get(c, 0.0)) for c in p))
                top = sorted(p.values(), reverse=True)
                if len(top) < 2 or top[0] - top[1] >= MARGIN:
                    clear.append(g == r["peft"])
            rep["mean_tv_distance"] = round(sum(tv) / len(tv), 4)
            rep["agree_when_peft_margin_ge"] = {
                "margin": MARGIN,
                "n": len(clear),
                "agree": round(sum(clear) / max(1, len(clear)), 4),
            }
        report[name] = rep
        extra = ""
        if "mean_tv_distance" in rep:
            c = rep["agree_when_peft_margin_ge"]
            extra = (
                f"  mean TV {rep['mean_tv_distance']:.4f}  "
                f"agree at margin>={MARGIN} {100 * c['agree']:6.2f}% (n={c['n']})"
            )
        print(
            f"{name:<10} n={len(rows):>5}  composed==PEFT {100 * agree:6.2f}%  "
            f"accuracy vs label {100 * acc:6.2f}%  {'PASS' if agree >= 0.99 else 'FAIL'}"
            + extra,
            flush=True,
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("standin", help="Random-weight adapters for the latency test")
    s.add_argument("--rank", type=int, default=DEFAULT_RANK)
    s.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compose", help="Compose trained adapters from train_alora.py")
    c.add_argument(
        "--runs", type=Path, required=True, help="Dir holding <name>/ adapters"
    )
    c.add_argument(
        "--router-runs", type=Path, help="Where the router adapter lives if elsewhere"
    )
    c.add_argument(
        "--asr-model",
        help="Enable audio input with this HF speech-recognition model, e.g. "
        "ibm-granite/granite-speech-5.0-470m-turboctc",
    )
    c.add_argument("--asr-device", default="cpu", help="cpu or cuda:0")
    s.add_argument("--kind", choices=KINDS, default="alora")
    # The composer reads Shadow Residual from the weights (a cross_stream LoRA),
    # not from the staging directory, so SR adapters are staged under lora/.
    c.add_argument("--kind", choices=(*KINDS, "sr"), default="alora")
    for p in (s, c):
        p.add_argument("--base", default="ibm-granite/granite-4.1-3b")
        p.add_argument(
            "--out", type=Path, required=True, help="Composed checkpoint dir"
        )
    v = sub.add_parser("verify", help="Composed checkpoint vs PEFT on held-out states")
    v.add_argument("--runs", type=Path, required=True)
    v.add_argument("--router-runs", type=Path)
    v.add_argument("--model", required=True)
    v.add_argument("--limit", type=int, default=5000)
    v.add_argument("--layout", default="log", help="The adapters' prompt layout")
    v.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.cmd == "verify":
        rep = verify(args.runs, args.model, args.router_runs, args.limit, args.layout)
        if args.json:
            args.json.write_text(json.dumps(rep, indent=1))
        return

    target_model = args.base.rstrip("/").split("/")[-1]
    stage_root = args.out.parent / f"{args.out.name}-stage"
    if args.cmd == "standin":
        adapters = make_standins(
            args.base,
            stage_root,
            args.rank,
            tuple(args.targets.split(",")),
            args.seed,
            args.kind,
        )
    else:
        adapters = {}
        for name in ADAPTERS:
            root = (
                args.router_runs
                if (name == "router" and args.router_runs)
                else args.runs
            )
            adapters[name] = root / name
            if not (adapters[name] / "adapter_config.json").exists():
                raise SystemExit(f"missing trained adapter: {adapters[name]}")
    kind = "lora" if args.kind == "sr" else args.kind
    paths = [stage(d, n, stage_root, target_model, kind) for n, d in adapters.items()]
    compose(
        paths,
        args.base,
        args.out,
        getattr(args, "asr_model", None),
        getattr(args, "asr_device", "cpu"),
    )
    print(f"\ncomposed {len(paths)} adapters ({', '.join(adapters)}) -> {args.out}")


if __name__ == "__main__":
    main()
