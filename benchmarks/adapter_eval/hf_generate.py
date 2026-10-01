# SPDX-License-Identifier: Apache-2.0
"""Greedy HF + PEFT generation for one reference cell, without granite-switch.

Run as a subprocess by ``reference.py``, one process per cell on one GPU, so
each gets a clean CUDA context and a failure stays in its cell::

    CUDA_VISIBLE_DEVICES=<gpu> python -m benchmarks.adapter_eval.hf_generate <job.json>

``job.json``::

    {"key": "answerability/alora", "column": "alora", "base_model": "...",
     "adapter_dir": "..." | null, "eval_path": "...", "out_path": "...",
     "status_path": "...", "limit": null, "max_new_tokens": 200,
     "max_model_len": 32768, "token_budget": 262144, "max_batch": 64,
     "documents": "native", "chat_template_kwargs": {}}

The model per column:

* ``base``: the base model, no adapter.
* ``lora`` / ``alora``: the base model with the checkpoint loaded by PEFT.
  PEFT turns an aLoRA on at its invocation tokens.
* ``sr``: the shadow-residual repo's model (``build_sr_base``, shipped to the
  pod at a pinned commit) with the checkpoint loaded by PEFT, as that repo's
  own generation does. The checkpoint's invocation tokens were dropped first
  (``staged.peft_sr_copy``).

Prompts are rendered by the base tokenizer's chat template, with the same
documents, tools, document style and chat-template options as the
granite-switch run (``prompts.py``), and generated greedily in bfloat16. The token budgets match ``generate.py``: a row gets
``min(max_new_tokens, max_model_len - prompt length)`` new tokens, a row with
no room is left out (``too_long``), and a row that does not reach EOS in its
budget counts as truncated.

Two checks turn a silent fallback into a failed cell:

* every checkpoint weight must be in the loaded model, with its saved value.
  PEFT loads a weight that names no module of the model without an error.
* an aLoRA's invocation tokens must be in every prompt. Without them PEFT
  runs the base model.

Rows are generated shortest first, in batches of at most ``max_batch`` rows
that fit ``token_budget`` padded tokens. Out of memory halves the batch size
for the rest of the cell.

Writes the kept eval rows plus ``generated_content`` and ``prompt_tokens`` to
``out_path``, in eval-file order, and the counts, or the reason the cell
failed, to ``status_path``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from .prompts import chat_text
from .staged import CONFIG_FILE, WEIGHTS_FILE, read_jsonl, write_jsonl


class CellFailed(Exception):
    """A failure whose message is a publishable reason."""


def contains(ids: list[int], seq: list[int]) -> bool:
    """Whether ``seq`` is a contiguous part of ``ids``."""
    n = len(seq)
    return any(ids[i : i + n] == seq for i in range(len(ids) - n + 1))


def batch_size(lengths: list[int], new_tokens: int, token_budget: int, cap: int) -> int:
    """How many rows from the start of ``lengths`` (ascending) to batch.

    A batch is padded to its longest prompt plus ``new_tokens``. At least one
    row is taken, even when it alone exceeds the budget.
    """
    n = 1
    while (
        n < min(cap, len(lengths))
        and (n + 1) * (lengths[n] + new_tokens) <= token_budget
    ):
        n += 1
    return n


def cut(new_ids: list[int], budget: int, eos_ids: set[int]) -> tuple[list[int], bool]:
    """A row's own output: up to its first EOS within ``budget`` new tokens.

    Returns the tokens and whether the row was truncated (no EOS in budget).
    Rows of a batch share the batch's largest budget, so a row can run past
    its own.
    """
    for i, t in enumerate(new_ids[:budget]):
        if t in eos_ids:
            return new_ids[: i + 1], False
    return new_ids[:budget], True


def loaded_name(key: str) -> str | None:
    """The name PEFT gives a saved LoRA weight once loaded, or None if not LoRA."""
    for tag in (".lora_A.", ".lora_B."):
        if tag in key:
            return key.replace(tag, f"{tag}default.", 1)
    return None


def unloaded_weights(saved: dict, state: dict) -> list[str]:
    """Saved LoRA weights missing from, or different in, the loaded model.

    ``saved`` is the checkpoint's tensors, ``state`` the model's state dict.
    Values are compared exactly after casting both to float32: PEFT may keep
    the adapter in float32 on a bfloat16 model.
    """
    bad = []
    for key, tensor in saved.items():
        name = loaded_name(key)
        if name is None:
            continue
        loaded = state.get(name)
        if loaded is None or tuple(loaded.shape) != tuple(tensor.shape):
            bad.append(key)
        elif not loaded.detach().float().cpu().equal(tensor.float()):
            bad.append(key)
    return bad


def eos_token_ids(model, tok) -> set[int]:
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def load_model(job: dict):
    import torch
    from transformers import AutoModelForCausalLM

    column = job["column"]
    base_model = job["base_model"]
    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if column == "base":
        return AutoModelForCausalLM.from_pretrained(base_model, **kw).to("cuda").eval()

    from peft import PeftConfig, PeftModel

    adapter_dir = Path(job["adapter_dir"])
    if column == "sr":
        from shadow_residual.shadow_residual.build import build_sr_base

        adapter_config = json.loads((adapter_dir / CONFIG_FILE).read_text())
        model = build_sr_base(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            # Read from the adapter, as the shadow-residual repo does; False
            # when absent, and a no-op on a dense base.
            share_moe_routing=bool(adapter_config.get("share_moe_routing", False)),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kw)
    config = PeftConfig.from_pretrained(adapter_dir)
    # Without a task type PEFT returns its generic wrapper, whose generate
    # skips the aLoRA activation.
    config.task_type = config.task_type or "CAUSAL_LM"
    model = PeftModel.from_pretrained(model, adapter_dir, config=config)
    return model.to("cuda").eval()


def invocation_tokens(job: dict) -> list[int] | None:
    if job["column"] != "alora":
        return None
    config = json.loads((Path(job["adapter_dir"]) / CONFIG_FILE).read_text())
    return list(config["alora_invocation_tokens"])


def generate_rows(model, tok, prompts: list[list[int]], budgets: list[int], job: dict):
    """Generate every prompt; returns (new token ids per prompt, truncated count)."""
    import gc

    import torch

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_ids = eos_token_ids(model, tok)
    max_new = int(job["max_new_tokens"])
    token_budget = int(job.get("token_budget", 262144))
    cap = int(job.get("max_batch", 64))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    lengths = [len(prompts[i]) for i in order]
    outputs: list[list[int] | None] = [None] * len(prompts)
    truncated = 0
    start = 0
    while start < len(order):
        n = batch_size(lengths[start:], max_new, token_budget, cap)
        rows = order[start : start + n]
        width = max(len(prompts[i]) for i in rows)
        input_ids = torch.full((n, width), pad, dtype=torch.long)
        attention_mask = torch.zeros((n, width), dtype=torch.long)
        for j, i in enumerate(rows):  # left padding
            input_ids[j, width - len(prompts[i]) :] = torch.tensor(prompts[i])
            attention_mask[j, width - len(prompts[i]) :] = 1
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids.to("cuda"),
                    attention_mask=attention_mask.to("cuda"),
                    max_new_tokens=max(budgets[i] for i in rows),
                    do_sample=False,
                    pad_token_id=pad,
                )
        except torch.cuda.OutOfMemoryError:
            if n == 1:
                raise
            del input_ids, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
            cap = max(1, n // 2)
            print(
                f"[hf_generate] {job['key']}: out of memory; batch cap {cap}",
                flush=True,
            )
            continue
        new = out[:, width:].tolist()
        for j, i in enumerate(rows):
            outputs[i], hit = cut(new[j], budgets[i], eos_ids)
            truncated += hit
        start += n
        print(f"[hf_generate] {job['key']}: {start} / {len(order)}", flush=True)
    return outputs, truncated


def run(job: dict) -> dict:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    key = job["key"]
    max_model_len = int(job["max_model_len"])
    tok = AutoTokenizer.from_pretrained(job["base_model"])
    invocation = invocation_tokens(job)

    rows = read_jsonl(Path(job["eval_path"]), job.get("limit"))
    kept, prompts, budgets = [], [], []
    too_long = 0
    for row in rows:
        text = chat_text(
            tok,
            row,
            job.get("documents", "native"),
            job.get("chat_template_kwargs", {}),
        )
        ids = list(tok(text, add_special_tokens=False).input_ids)
        if invocation and not contains(ids, invocation):
            raise CellFailed("invocation tokens not in prompt")
        budget = min(int(job["max_new_tokens"]), max_model_len - len(ids))
        if budget < 1:
            too_long += 1
            continue
        kept.append(row)
        prompts.append(ids)
        budgets.append(budget)
    if not kept:
        raise CellFailed("no rows left to generate")

    t0 = time.time()
    model = load_model(job)
    if job.get("adapter_dir"):
        saved = load_file(str(Path(job["adapter_dir"]) / WEIGHTS_FILE))
        bad = unloaded_weights(saved, model.state_dict())
        if bad:
            print(f"[hf_generate] {key}: {len(bad)} weights not loaded, e.g. {bad[:3]}")
            raise CellFailed("adapter weights not loaded")
    print(f"[hf_generate] {key}: model loaded in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    outputs, truncated = generate_rows(model, tok, prompts, budgets, job)
    for row, ids, new in zip(kept, prompts, outputs, strict=True):
        row["generated_content"] = tok.decode(new, skip_special_tokens=True)
        row["prompt_tokens"] = len(ids)
    write_jsonl(Path(job["out_path"]), kept)
    seconds = round(time.time() - t0, 1)
    print(
        f"[hf_generate] {key}: {len(kept)} rows in {seconds:.0f}s "
        f"({truncated} hit max_new_tokens)",
        flush=True,
    )
    return {
        "ok": True,
        "n_rows": len(rows),
        "too_long": too_long,
        "n_generated": len(kept),
        "truncated": truncated,
        "seconds": seconds,
        "gpu": torch.cuda.get_device_name(0),
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    job = json.loads(Path(argv[0]).read_text())
    try:
        status = run(job)
    except CellFailed as e:
        print(f"[hf_generate] {job['key']}: {e}", flush=True)
        status = {"ok": False, "reason": str(e)}
    except Exception:
        traceback.print_exc()
        status = {"ok": False, "reason": "generation failed"}
    Path(job["status_path"]).write_text(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
