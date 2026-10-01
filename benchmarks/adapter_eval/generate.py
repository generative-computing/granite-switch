# SPDX-License-Identifier: Apache-2.0
"""Greedy vLLM generation for every adapter in one composed checkpoint.

Run as a subprocess by ``run_benchmark.py``, one process per composed model,
so each gets a clean CUDA context::

    python -m benchmarks.adapter_eval.generate <jobs.json>

``jobs.json``::

    {
      "model_dir": "...", "status_path": "...",
      "llm": {"max_model_len": 32768, "enforce_eager": false,
              "enable_prefix_caching": false, "gpu_memory_utilization": 0.85,
              "enable_chunked_prefill": true, "max_num_batched_tokens": null},
      "jobs": [{"key": "answerability/lora", "adapter_name": "answerability_lora",
                "eval_path": "...", "out_path": "...", "limit": null,
                "max_new_tokens": 200, "documents": "native",
                "chat_template_kwargs": {}}]
    }

Each job writes its eval rows plus ``generated_content`` and ``prompt_tokens``
to ``out_path``.
The status file records per-job counts, or the reason a job failed; one
failing job does not stop the others.

Prompts are rendered by the composed tokenizer's chat template with
``adapter_name``, in the job's document style and chat-template options
(``prompts.py``). An unknown name silently renders the base-model prompt, so
every prompt is checked to contain the adapter's control token.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from .prompts import chat_text
from .staged import read_jsonl, write_jsonl


class ControlTokenMissing(Exception):
    pass


def render(tok, row: dict, job: dict, control_token_id: int | None) -> list[int]:
    adapter_name = job["adapter_name"]
    text = chat_text(
        tok,
        row,
        job.get("documents", "native"),
        job.get("chat_template_kwargs", {}),
        adapter_name=adapter_name,
    )
    if f"<|{adapter_name}|>" not in text:
        raise ControlTokenMissing(f"control token for {adapter_name} not rendered")
    ids = list(tok(text, add_special_tokens=False).input_ids)
    if control_token_id is not None and control_token_id not in ids:
        raise ControlTokenMissing(f"control token id for {adapter_name} not in prompt")
    return ids


def control_token_ids(model_dir: Path) -> dict[str, int]:
    cfg = json.loads((model_dir / "config.json").read_text())
    names = cfg.get("adapter_names") or []
    ids = cfg.get("adapter_token_ids") or []
    return dict(zip(names, ids, strict=False))


def scheduler_settings(llm) -> dict | None:
    """The engine's effective token budget and chunked-prefill mode, if readable."""
    try:
        cfg = llm.llm_engine.vllm_config.scheduler_config
        return {
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "max_num_seqs": cfg.max_num_seqs,
            "chunked_prefill": bool(cfg.enable_chunked_prefill),
        }
    except AttributeError:
        return None


LENGTH_BUCKETS = (1024, 2048, 4096, 8192, 16384)


def truncation_by_length(rows: list[dict], outputs) -> str:
    """``"<1024: 3/120, <2048: ..."``: rows that hit max_tokens per prompt length."""
    counts: dict[str, list[int]] = {}
    for row, out in zip(rows, outputs, strict=True):
        n = row["prompt_tokens"]
        label = next(
            (f"<{b}" for b in LENGTH_BUCKETS if n < b), f">={LENGTH_BUCKETS[-1]}"
        )
        hit, total = counts.setdefault(label, [0, 0])
        counts[label] = [hit + (out.outputs[0].finish_reason == "length"), total + 1]
    order = [f"<{b}" for b in LENGTH_BUCKETS] + [f">={LENGTH_BUCKETS[-1]}"]
    return ", ".join(
        f"{k}: {counts[k][0]}/{counts[k][1]}" for k in order if k in counts
    )


def run(spec: dict) -> dict:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_dir = Path(spec["model_dir"])
    llm_kw = spec["llm"]
    max_model_len = int(llm_kw["max_model_len"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    token_ids = control_token_ids(model_dir)

    status: dict = {
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "jobs": {},
    }

    # Render everything before loading the model: a template problem then
    # costs seconds, not a model load.
    prepared = []
    for job in spec["jobs"]:
        key = job["key"]
        try:
            rows = read_jsonl(Path(job["eval_path"]), job.get("limit"))
            kept, prompts, params = [], [], []
            too_long = 0
            for row in rows:
                ids = render(tok, row, job, token_ids.get(job["adapter_name"]))
                budget = min(int(job["max_new_tokens"]), max_model_len - len(ids))
                if budget < 1:
                    too_long += 1
                    continue
                kept.append(row)
                prompts.append({"prompt_token_ids": ids})
                params.append(SamplingParams(temperature=0.0, max_tokens=budget))
            if not kept:
                raise ValueError("no rows left to generate")
            prepared.append((job, kept, prompts, params))
            status["jobs"][key] = {
                "ok": True,
                "n_rows": len(rows),
                "too_long": too_long,
            }
        except ControlTokenMissing as e:
            print(f"[generate] {key}: {e}", flush=True)
            status["jobs"][key] = {"ok": False, "reason": "control token not rendered"}
        except Exception:
            traceback.print_exc()
            status["jobs"][key] = {"ok": False, "reason": "prompt rendering failed"}

    if not prepared:
        return status

    # Scheduler settings are passed only when set, so the default run keeps
    # vLLM's own defaults (the path a server takes).
    scheduler = {}
    if llm_kw.get("max_num_batched_tokens"):
        scheduler["max_num_batched_tokens"] = int(llm_kw["max_num_batched_tokens"])
    if llm_kw.get("enable_chunked_prefill") is False:
        scheduler["enable_chunked_prefill"] = False
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=float(llm_kw.get("gpu_memory_utilization", 0.85)),
        enforce_eager=bool(llm_kw.get("enforce_eager", False)),
        enable_prefix_caching=bool(llm_kw.get("enable_prefix_caching", False)),
        **scheduler,
    )
    status["scheduler"] = scheduler_settings(llm)
    print(f"[generate] scheduler: {status['scheduler']}", flush=True)

    for job, kept, prompts, params in prepared:
        key = job["key"]
        t0 = time.time()
        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
            truncated = 0
            for row, prompt, out in zip(kept, prompts, outputs, strict=True):
                completion = out.outputs[0]
                row["generated_content"] = completion.text
                row["prompt_tokens"] = len(prompt["prompt_token_ids"])
                truncated += completion.finish_reason == "length"
            write_jsonl(Path(job["out_path"]), kept)
            status["jobs"][key].update(
                n_generated=len(kept),
                truncated=truncated,
                seconds=round(time.time() - t0, 1),
            )
            print(
                f"[generate] {key}: {len(kept)} rows in {time.time() - t0:.0f}s "
                f"({truncated} hit max_tokens; by prompt length: "
                f"{truncation_by_length(kept, outputs)})",
                flush=True,
            )
        except Exception:
            traceback.print_exc()
            status["jobs"][key] = {"ok": False, "reason": "generation failed"}
    return status


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[0]).read_text())
    status = run(spec)
    Path(spec["status_path"]).write_text(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
