# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker for the vLLM classifier chunked-prefill test.

Each sub-command is a separate process invocation so only one vLLM engine is ever
resident (same discipline as ``_generation_equivalence_worker``)::

    python worker.py compose                      # CPU only; prints {"result": meta}
    python worker.py compose-multilabel           # CPU only; multi-label head
    python worker.py run <budget|none> <out.json> # one engine, one budget
    python worker.py fault-sweep <out.json>       # forced-read sweep over the prompt

The engine core runs in-process (the driver sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
so the hooks installed here land on the model the engine actually runs.

The read-index hook records, per forward pass, the global position the verdict read.
``_classifier_read_points`` reads at the marker (a flat within-pass index); the hook
maps that index through ``positions``, vLLM's own per-token coordinate, so the recorded
position reflects the token actually read.
"""

import json
import os
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _install_read_index_hook():
    """Record, per forward pass, the global position the classifier verdict actually
    read.

    Wraps ``GraniteSwitchModel._classifier_read_points`` to capture, per request, the
    flat within-pass index it read (the marker). The ``forward`` wrapper maps that
    index through ``positions`` to a global position.
    """
    from granite_switch.vllm.granite_switch_model import (
        GraniteSwitchForCausalLM,
        GraniteSwitchModel,
    )

    passes = []
    orig_fwd = GraniteSwitchForCausalLM.forward
    orig_read = GraniteSwitchModel._classifier_read_points

    # Per-pass scratch for request 0 (these tests use a single request):
    #   flat_read_idx -- within-pass flat index the verdict read (the marker), a
    #                    sentinel -1 when this pass merely carried a verdict resolved
    #                    earlier, or None.
    scratch = {"flat_read_idx": None}

    def read_points(self, classifier_indices, hidden_states, input_ids):
        # Snapshot before the call: it replaces _classifier_resolved_verdict with
        # this pass's map, and the carry check needs the previous pass's.
        carried_before = dict(getattr(self, "_classifier_resolved_verdict", None) or {})
        res = orig_read(self, classifier_indices, hidden_states, input_ids)
        try:
            from vllm.forward_context import get_forward_context

            am = get_forward_context().attn_metadata
            if isinstance(am, list):
                am = am[0] if am else None
            if isinstance(am, dict):
                am = next(iter(am.values()), None)
            qsl = getattr(am, "query_start_loc", None)
            if qsl is not None:
                qsl = qsl.tolist()
                start, end = qsl[0], qsl[1]  # request 0
                # The marker is the last control-token id match in this slice.
                import torch as _t

                mids = _t.tensor(
                    self.config.classifier_control_token_ids,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                hits = _t.isin(input_ids[start:end], mids).nonzero(as_tuple=True)[0]
                if hits.numel() == 0:
                    # No marker in this slice. If a verdict was resolved in an
                    # earlier pass it is carried forward, and that position is not
                    # addressable by this pass's ``positions``; report it directly.
                    # Otherwise this request is still mid-prefill.
                    sql = getattr(am, "seq_lens", None)
                    pre = (
                        int(sql.tolist()[0]) - (end - start)
                        if sql is not None
                        else None
                    )
                    if pre is not None and pre in carried_before:
                        scratch["flat_read_idx"] = -1
                    else:
                        scratch["flat_read_idx"] = end - 1
                else:
                    # The marker is its own read point, always in this pass.
                    marker = start + int(hits[-1])
                    scratch["flat_read_idx"] = marker
                    scratch["resolving_flat"] = marker
        except Exception:  # a hook must never break a forward
            scratch["flat_read_idx"] = None
        return res

    def fwd(self, input_ids, positions, *args, **kwargs):
        scratch["flat_read_idx"] = None
        out = orig_fwd(self, input_ids, positions, *args, **kwargs)
        try:
            has_cls = getattr(self.model, "classifier_head", None) is not None
            flat = scratch["flat_read_idx"]
            if has_cls and positions is not None and flat is not None:
                if flat == -1:
                    read_pos = scratch.get("carried_global")  # resolved earlier
                else:
                    read_pos = int(positions[flat].item())
                if scratch.get("resolving_flat") is not None:
                    # Remember where the verdict was resolved, so a later pass that
                    # merely carries it can report the same global position.
                    scratch["carried_global"] = int(
                        positions[scratch["resolving_flat"]].item()
                    )
                    scratch["resolving_flat"] = None
                passes.append(
                    {"read_global_pos": read_pos, "n_query": int(positions.shape[0])}
                )
        except Exception:  # a hook must never break a forward
            pass
        return out

    GraniteSwitchModel._classifier_read_points = read_points
    GraniteSwitchForCausalLM.forward = fwd
    return passes


# Single-token label candidates, in priority order (only single-token ones are kept).
_LABEL_CANDIDATES = [
    "safe",
    "unsafe",
    "yes",
    "no",
    "true",
    "false",
    "good",
    "bad",
    "high",
    "low",
    "hot",
    "cold",
]


def compose(base, outdir, control_offset=5, num_labels=6, scale=0.5):
    """Compose a single-classifier-slot checkpoint.

    ``num_labels`` single-token labels are taken from ``_LABEL_CANDIDATES`` and
    ``scale`` is the std of the random head weight. Both are set high enough, over a
    non-repetitive prompt, that the argmax varies across read positions -- which is
    what makes a wrong read observable in the fault sweep.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer

    from granite_switch.composer.compose_utils import GraniteSwitchComposer
    from granite_switch.composer.weight_transfer import CLASSIFIER_HEAD_FILE

    tok = AutoTokenizer.from_pretrained(base)
    hidden = AutoConfig.from_pretrained(base).hidden_size

    labels, ids = [], []
    for c in _LABEL_CANDIDATES:
        e = tok.encode(c, add_special_tokens=False)
        if len(e) == 1 and e[0] not in ids:
            labels.append(c)
            ids.append(e[0])
        if len(labels) == num_labels:
            break
    if len(labels) < num_labels:
        raise RuntimeError(
            f"only {len(labels)} single-token labels available from "
            f"{_LABEL_CANDIDATES}, need {num_labels}"
        )
    nl = len(labels)

    g = torch.Generator().manual_seed(0)
    # Weight-based (not bias-only) head so the verdict depends on the hidden state.
    weight = torch.randn((nl, hidden), generator=g, dtype=torch.float32) * scale
    bias = torch.zeros((nl,), dtype=torch.float32)

    slot = os.path.join(outdir, "detect")
    os.makedirs(slot, exist_ok=True)
    save_file(
        {"weight": weight.contiguous(), "bias": bias.contiguous()},
        os.path.join(slot, CLASSIFIER_HEAD_FILE),
    )
    json.dump(
        {"kind": "classifier", "labels": labels},
        open(os.path.join(slot, "adapter_config.json"), "w"),
    )
    open(os.path.join(slot, "io.yaml"), "w").write("name: detect\nmodel: ~\n")

    control_id = tok.vocab_size - control_offset
    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base,
        adapter_paths=[slot],
        adapter_token_ids=[control_id],
        adapter_substitute_token_ids=[control_id],
        adapter_names=["detect"],
        adapter_kinds=["classifier"],
        classifier_label_token_ids=[ids],
    )
    ckpt = os.path.join(outdir, "composed")
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    # Non-repetitive body so mid-prompt hidden states differ across positions.
    words = (
        "machine translation quietly reshaped how distant villages argued "
        "about rainfall while engineers debated whether copper cables or "
        "glass fibre carried gossip faster than a startled horse could run "
        "downhill past the old mill where children once traded marbles for "
        "stories about comets volcanoes submarines and the strange arithmetic "
        "of tides that neither king nor merchant ever fully trusted yet "
        "everyone quoted at dinner as though the ocean kept a ledger of debts "
        "owed to the moon each evening without fail or apparent complaint "
        "although sailors insisted otherwise over cheap wine and louder songs"
    )
    body = tok.encode(f"Is this text safe? {words}", add_special_tokens=False)
    # End-locator layout, matching the chat template: the marker follows the last
    # content token, and is itself the read point.
    prompt_ids = [*body, control_id]
    # Same layout, but continuing past the marker the way a rendered chat prompt
    # does (turn close + generation prompt). The marker is then interior, so a
    # chunk budget can close the prompt after it -- the pass that samples holds no
    # marker at all and must consume the verdict resolved in an earlier pass.
    tail = tok.encode(
        "\n<|start_of_role|>assistant<|end_of_role|>", add_special_tokens=False
    )
    prompt_ids_trailing = [*body, control_id, *tail]
    return {
        "ckpt": ckpt,
        "labels": labels,
        "control_id": control_id,
        "label_token_ids": ids,
        "prompt_ids": prompt_ids,
        "prompt_ids_trailing": prompt_ids_trailing,
        "n_tail": len(tail),
    }


def run_budget(ckpt, prompt_ids, budget, out_path):
    import torch  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    passes = _install_read_index_hook()
    kw = dict(
        model=ckpt,
        enforce_eager=True,
        gpu_memory_utilization=0.55,
        max_model_len=4096,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    if budget is None:
        kw["enable_chunked_prefill"] = False
    else:
        kw["max_num_batched_tokens"] = budget
        kw["enable_chunked_prefill"] = True
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
    o = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0].outputs[0]

    # The consumed verdict is the final pass's; single request => one read per pass.
    last = passes[-1] if passes else None
    global_idx = None
    if last and last["read_global_pos"] is not None:
        rp = last["read_global_pos"]
        global_idx = int(rp[0]) if isinstance(rp, list) else int(rp)

    out = {
        "budget": budget,
        "token_id": int(o.token_ids[0]),
        "text": o.text,
        "n_passes": len(passes),
        "global_read_idx": global_idx,
        "n_prompt": len(prompt_ids),
    }
    json.dump(out, open(out_path, "w"))


def run_fault_sweep(ckpt, prompt_ids, out_path):
    """Force the classifier verdict to read positions swept across the whole prompt
    and record the resulting token at each.

    The correct read point is the marker itself, located inside
    ``_classifier_read_points``. To sweep wrong reads, that method is replaced with a
    forcing version: it takes the real per-request slice but reads
    ``hidden_states[start + forced[0]]`` and reports slot 1 (the classifier slot) so the
    verdict still fires. ``forced[0] is None`` runs the real method.

    The prefill runs unchunked (a single pass over all ``n`` tokens), so the within-pass
    flat index equals the global position and a forced ``start + i`` reaches global
    position ``i`` for any ``0 <= i <= n-1`` in one resident engine. A mutable
    ``forced[0]`` cell, closed over by the monkeypatch, changes the forced index between
    ``generate`` calls without rebuilding the engine."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.forward_context import get_forward_context
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm.granite_switch_model import GraniteSwitchModel

    n = len(prompt_ids)

    forced = [None]  # None => real method (the marker); int => forced slice index
    read_global = [None]  # global position the forced read landed on
    orig_read = GraniteSwitchModel._classifier_read_points

    def patched_read(self, classifier_indices, hidden_states, input_ids):
        if forced[0] is None:
            return orig_read(self, classifier_indices, hidden_states, input_ids)
        am = get_forward_context().attn_metadata
        if isinstance(am, list):
            am = am[0] if am else None
        if isinstance(am, dict):
            am = next(iter(am.values()), None)
        qsl = getattr(am, "query_start_loc", None)
        if qsl is None:
            return None
        qsl = qsl.tolist()
        start = qsl[0]  # single request in these tests
        flat = start + int(forced[0])
        # Unchunked single pass => flat within-pass index == global position.
        read_global[0] = int(forced[0])
        req_hidden = hidden_states[flat : flat + 1]  # [1, hidden]
        # Slot 1 = the classifier slot, so the verdict fires like a real read.
        req_slot = torch.ones(1, dtype=torch.long, device=hidden_states.device)
        return req_hidden, req_slot

    GraniteSwitchModel._classifier_read_points = patched_read
    try:
        passes = _install_read_index_hook()  # wraps the (now patched) method + forward
        llm = LLM(
            model=ckpt,
            enforce_eager=True,
            gpu_memory_utilization=0.55,
            max_model_len=4096,
            enable_prefix_caching=False,
            dtype="bfloat16",
            enable_chunked_prefill=False,
        )
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
        pt = TokensPrompt(prompt_token_ids=prompt_ids)

        def one(forced_idx):
            forced[0] = forced_idx
            read_global[0] = None
            passes.clear()
            o = llm.generate([pt], sp)[0].outputs[0]
            if forced_idx is None:
                # Correct run: read position comes from the hook (the marker).
                last = passes[-1] if passes else None
                rp = last["read_global_pos"] if last else None
                pos = int(rp) if rp is not None else None
            else:
                # Forced run: the global position is the forced index directly.
                pos = read_global[0]
            return {"pos": pos, "token_id": int(o.token_ids[0]), "text": o.text}

        # Correct run: reads the marker (global pos n-1).
        correct = one(None)

        # Faulted sweep at a fixed stride over the prompt, excluding the correct
        # read point (n-2) so every forced read is genuinely a wrong position.
        step = max(1, (n - 1) // 10)
        sweep_idx = sorted({i for i in range(0, n - 1, step)} - {n - 2})
        faulted = [one(i) for i in sweep_idx]
    finally:
        GraniteSwitchModel._classifier_read_points = orig_read

    out = {"correct": correct, "faulted": faulted, "n_prompt": n}
    json.dump(out, open(out_path, "w"))


def main():
    mode = sys.argv[1]
    if mode == "compose-multilabel":
        try:
            meta = compose(os.environ["CLS_BASE"], os.environ["CLS_OUT"])
            print(json.dumps({"result": meta}))
        except Exception as e:
            print(json.dumps({"error": f"{e}\n{traceback.format_exc()}"}))
        return
    if mode == "fault-sweep":
        out_path = sys.argv[2]
        prompt_ids = json.load(open(os.environ["CLS_PIDS"]))
        try:
            run_fault_sweep(os.environ["CLS_CKPT"], prompt_ids, out_path)
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    if mode == "run":
        budget = None if sys.argv[2] == "none" else int(sys.argv[2])
        out_path = sys.argv[3]
        prompt_ids = json.load(open(os.environ["CLS_PIDS"]))
        try:
            run_budget(os.environ["CLS_CKPT"], prompt_ids, budget, out_path)
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
