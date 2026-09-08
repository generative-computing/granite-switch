# SPDX-License-Identifier: Apache-2.0
"""aLoRA + prefix-cache cross-turn contamination probe (real model, real serving).

Answers a question nothing else in the suite touches: **when a multi-turn session
reuses the KV cache, does a previous turn's adapter contaminate the next turn?**

Why this is a real question, not a hypothetical:

* ``grep`` finds NO prefix-cache invalidation, re-prefill, or cache-salt logic anywhere
  in ``src/granite_switch/``, and ``MultiSwitch.forward`` receives only
  ``(input_ids, adapter_token_ids, positions)`` -- it cannot see that a cache hit
  happened, how many tokens were reused, or which adapter produced them.
* aLoRA deliberately places its control token LATE (immediately before the invocation
  sequence, ``tokenizer_setup.py``), so the long conversation history prefills with BASE
  weights and is reusable. That is the whole economic point of aLoRA. But it also means
  a turn-2 request shares a long prefix with turn 1 and WILL hit the cache.
* LoRA places its control token at the sequence START, so a differing adapter changes
  the prefix at position 0 and should miss -- safe, but by accident rather than design.

Three leak paths this probe measures, rather than reasons about:

  L1 COUNT CARRY   turn 1's control token stays in the reused KV, so the coded engine's
                   counting head may include it: turn 2's recovered ``n`` would be
                   shifted, moving every codebook address.
  L2 ROUTING       does turn 2 route to the adapter IT asked for, or to turn 1's?
  L3 OUTPUT        does turn 2's output differ from the SAME turn-2 prompt served with
                   no history at all? That difference IS the contamination, and it is
                   the only one a user sees.

Design: prefix caching is deliberately ENABLED here (every other test in the suite
disables it), because reuse is the condition under test. Each scenario is run twice --
once as turn 2 of a session (cache warm from turn 1) and once standalone (cold) -- and
the two are compared. A/A and A/B turn pairs are both covered, so "contaminated" can be
distinguished from "same adapter anyway".

Commands (JSON to stdout; diagnostics to stderr):
  build        --output-dir <dir>     compose WITHOUT --technology-filter so the
                                     composer's alora>lora preference yields a MIXED
                                     aLoRA+LoRA checkpoint
  inspect-keys --output-path <json>   read vLLM's own cache-key construction (no GPU)
  serve        --model-path <dir> --output-path <json>
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_MODEL = "ibm-granite/granite-4.1-3b"
# No --technology-filter: composer prefers alora>lora, so this yields a MIXED
# checkpoint. 10 aLoRA adapters exist across these three libraries.
ADAPTER_REPOS = [
    "ibm-granite/granitelib-core-r1.0",
    "ibm-granite/granitelib-guardian-r1.0",
    "ibm-granite/granitelib-rag-r1.0",
]
DEFAULT_PORT = 8131
MAX_TOKENS = 24


def _compose(out_dir):
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        BASE_MODEL,
        "--adapters",
        *ADAPTER_REPOS,
        "--output",
        out_dir,
    ]
    print("composing (MIXED alora+lora):", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-3000:], file=sys.stderr)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"compose failed ({r.returncode})")


def _infer_technologies(cfg):
    """Infer per-adapter technology from the SUBSTITUTE token ids.

    ``adapter_technologies`` is NOT persisted -- the composer decides alora-vs-lora at
    compose time and keeps only the consequence. That consequence is the substitute:
      * LoRA  -> the chat template's start-of-turn token, identical for every LoRA
                 adapter (probed once via _probe_lora_substitute_token_id).
      * aLoRA -> the FIRST TOKEN OF THAT ADAPTER'S INVOCATION SEQUENCE
                 (get_alora_first_invocation_token_id), so it differs from the
                 template token.
    So the modal substitute is the LoRA one, and any adapter whose substitute differs
    is aLoRA. Observed on a real mixed compose: subs=[100264, 27, 27, 27, 27, 27, 27,
    100264, ...] -> the six 27s are exactly the adapters that have alora variants on
    the Hub.
    """
    subs = list(cfg.get("adapter_substitute_token_ids") or [])
    if not subs:
        return []
    from collections import Counter

    lora_sub = Counter(subs).most_common(1)[0][0]
    return ["lora" if s == lora_sub else "alora" for s in subs]


def cmd_build(args):
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"warm-reuse {args.output_dir}", file=sys.stderr)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        _compose(args.output_dir)
    with open(os.path.join(args.output_dir, "config.json")) as f:
        cfg = json.load(f)
    names = cfg.get("adapter_names") or []
    techs = _infer_technologies(cfg)
    print(f"num_adapters={cfg.get('num_adapters')}", file=sys.stderr)
    print(f"names={names}", file=sys.stderr)
    print(f"technologies={techs}", file=sys.stderr)
    print(f"ctrl={cfg.get('adapter_token_ids')}", file=sys.stderr)
    print(f"subs={cfg.get('adapter_substitute_token_ids')}", file=sys.stderr)
    assert "ms_code_m" in cfg, (
        "expected a MultiSwitch (coded) checkpoint (ms_code_m marker)"
    )
    print("BUILD_OK")
    return 0


# ── PAPER CHECK: what does vLLM actually hash a cache block on? ──────────────
def cmd_inspect_keys(args):
    """Read vLLM's own cache-key construction. No GPU, no model.

    The question: for a MultiSwitch checkpoint the adapters are INSIDE the model and
    selected by control tokens in the token stream, not by vLLM's LoRA id. So if the
    block hash covers only token ids (+ vLLM's own lora/mm extras), then two requests
    sharing a token prefix hit the cache even when that prefix was computed under a
    different embedded adapter -- because vLLM has no idea the routing differed.
    """
    out = {}
    try:
        import inspect as _inspect

        from vllm.v1.core import kv_cache_utils as kcu

        out["module_file"] = kcu.__file__
        for fn in (
            "hash_block_tokens",
            "hash_request_tokens",
            "generate_block_hash_extra_keys",
            "init_none_hash",
            "need_extra_keys",
        ):
            f = getattr(kcu, fn, None)
            if f is None:
                continue
            try:
                out[fn] = _inspect.getsource(f)
            except Exception as e:
                out[fn] = f"<source unavailable: {e!r}>"
        # Which fields make up a block key at all?
        for cls in ("BlockHash", "BlockHashWithGroupId", "KVCacheBlock"):
            c = getattr(kcu, cls, None)
            if c is not None:
                out[f"class::{cls}"] = str(getattr(c, "__annotations__", {}))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    with open(args.output_path, "w") as f:
        json.dump(out, f)
    print(f"inspect keys: {sorted(out)}", file=sys.stderr)
    print("INSPECT_OK")
    return 0


# ── in-server instrumentation ───────────────────────────────────────────────
_SITECUSTOMIZE = r"""
import importlib.abc
import importlib.util
import json
import os
import sys

_T = os.environ.get("GS_ALORA_TRACE")
if _T:
    _TARGET = "granite_switch.vllm.switch.multi"

    def _patch(mod):
        cls = getattr(mod, "MultiSwitch", None)
        if cls is None or getattr(cls, "_gs_alora_patched", False):
            return
        orig = cls.forward

        def traced(self, input_ids, adapter_token_ids, positions=None):
            ai, modified = orig(
                self,
                input_ids=input_ids,
                adapter_token_ids=adapter_token_ids,
                positions=positions,
            )
            try:
                qsl = seq_lens = None
                try:
                    from vllm.forward_context import get_forward_context

                    fc = get_forward_context()
                    md = getattr(fc, "attn_metadata", None)
                    if isinstance(md, dict) and md:
                        md = next(iter(md.values()))
                    q = getattr(md, "query_start_loc", None)
                    if q is not None:
                        qsl = [int(x) for x in q.detach().cpu().tolist()]
                    s = getattr(md, "seq_lens", None)
                    if s is not None:
                        try:
                            seq_lens = [int(x) for x in s.detach().cpu().tolist()]
                        except AttributeError:
                            seq_lens = [int(x) for x in s]
                except Exception:
                    pass
                # Present only outside torch.compile: MultiSwitch guards this write
                # with `not torch.compiler.is_compiling()`, and a compiled graph does
                # not contain the branch at all. Holds here because this worker runs
                # with enforce_eager=True -- drop that and every write_addresses below
                # becomes None, silently.
                wa = getattr(self, "_debug_write_addresses", None)
                rec = {
                    "tag": os.environ.get("GS_ALORA_TAG", ""),
                    "input_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
                    "positions": (
                        None
                        if positions is None
                        else [int(x) for x in positions.detach().cpu().tolist()]
                    ),
                    "adapter_indices": [int(x) for x in ai.detach().cpu().tolist()],
                    "write_addresses": (
                        None if wa is None
                        else [int(x) for x in wa.detach().cpu().tolist()]
                    ),
                    "query_start_loc": qsl,
                    "seq_lens": seq_lens,
                }
                with open(_T, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                sys.stderr.write("gs-alora trace error: %r\n" % (e,))
            return ai, modified

        cls.forward = traced
        cls._gs_alora_patched = True
        sys.stderr.write("gs-alora: patched in pid %d\n" % os.getpid())

    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:

        class _F(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name != _TARGET:
                    return None
                sys.meta_path.remove(self)
                try:
                    real = importlib.util.find_spec(name)
                finally:
                    sys.meta_path.insert(0, self)
                if real is None:
                    return None
                re_ = real.loader.exec_module

                def exec_module(mod, _r=re_):
                    _r(mod)
                    _patch(mod)

                real.loader.exec_module = exec_module
                return real

        sys.meta_path.insert(0, _F())
"""


def _write_sc(d):
    with open(os.path.join(d, "sitecustomize.py"), "w") as f:
        f.write(_SITECUSTOMIZE)


def _read(p, start=0):
    out = []
    if not os.path.exists(p):
        return out, start
    with open(p, "rb") as f:
        f.seek(start)
        raw = f.read()
        end = f.tell()
    for ln in raw.split(b"\n"):
        ln = ln.strip().strip(b"\x00")
        if not ln:
            continue
        try:
            out.append(json.loads(ln.decode("utf-8")))
        except Exception:
            pass
    return out, end


def _supported_flags(candidates):
    """Drop flags this vLLM build rejects (api_server exits on unknown flags)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        help_text = (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        print(f"flag probe failed ({e!r}); unfiltered", file=sys.stderr)
        return list(candidates)
    kept, dropped, i = [], [], 0
    while i < len(candidates):
        flag = candidates[i]
        if flag.startswith("--"):
            if flag in help_text:
                kept.append(flag)
                if i + 1 < len(candidates) and not candidates[i + 1].startswith("--"):
                    kept.append(candidates[i + 1])
                    i += 1
            else:
                dropped.append(flag)
                if i + 1 < len(candidates) and not candidates[i + 1].startswith("--"):
                    i += 1
        i += 1
    if dropped:
        print(f"WARNING: skipped unsupported {dropped}", file=sys.stderr)
    return kept


def _post(port, payload, timeout=240):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _wait_ready(port, proc, log_file, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            if os.path.exists(log_file):
                with open(log_file) as f:
                    tail = f.read()[-3000:]
            raise RuntimeError(f"server exited early (rc={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server not ready within {timeout}s")


_HISTORY = (
    "User: I am comparing storage options for a small analytics service. "
    "Assistant: The main tradeoff is between row stores and column stores. "
    "Row stores favour point lookups and transactional writes, while column "
    "stores compress better and scan large ranges far more cheaply. "
    "User: What about indexing strategy for mostly-append workloads? "
    "Assistant: For append-heavy tables a clustered index on the append key keeps "
    "inserts sequential, and secondary indexes should be kept few because each one "
    "adds write amplification. Consider partitioning by time so old partitions can "
    "be compacted independently. "
)


def cmd_serve(args):
    import tempfile

    from transformers import AutoTokenizer

    from granite_switch.tutorials.vllm_server import (
        kill_stale_vllm_processes,
        launch_vllm,
    )

    port = args.port
    d = tempfile.mkdtemp(prefix="gs_alora_")
    _write_sc(d)
    trace = os.path.join(d, "trace.jsonl")
    log_file = os.path.join(d, "server.log")

    with open(os.path.join(args.model_path, "config.json")) as f:
        cfg = json.load(f)
    ctrl = list(cfg["adapter_token_ids"])
    names = list(cfg.get("adapter_names") or [])
    techs = _infer_technologies(cfg)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    os.environ["GS_ALORA_TRACE"] = trace
    os.environ["PYTHONPATH"] = d + os.pathsep + os.environ.get("PYTHONPATH", "")

    kill_stale_vllm_processes()
    proc = launch_vllm(
        model=args.model_path,
        port=port,
        log_file=log_file,
        max_num_seqs=8,
        enforce_eager=True,
        max_model_len=4096,
        # PREFIX CACHING ON -- reuse is the condition under test. Every other
        # MultiSwitch test disables it; here it must be enabled or there is nothing
        # to measure.
        extra_args=_supported_flags(
            [
                "--enable-prefix-caching",
                "--enable-chunked-prefill",
            ]
        ),
    )

    result = {
        "num_adapters": cfg.get("num_adapters"),
        "adapter_names": names,
        "adapter_technologies": techs,
        "adapter_token_ids": ctrl,
        "scenarios": {},
    }
    try:
        _wait_ready(port, proc, log_file)
        print("server ready (prefix caching ENABLED)", file=sys.stderr)

        hist_ids = tok(_HISTORY)["input_ids"]
        q2 = tok("User: Which option would you pick for this workload? ")["input_ids"]

        def ask(token_ids, tag):
            os.environ["GS_ALORA_TAG"] = tag
            return _post(
                port,
                {
                    "model": args.model_path,
                    "prompt": token_ids,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                },
            )

        # Pick two DIFFERENT adapters; prefer aLoRA ones when present so the
        # late-control-token placement (the reusable-history case) is exercised.
        alora_idx = [i for i, t in enumerate(techs) if t == "alora"]
        pick = alora_idx if len(alora_idx) >= 2 else list(range(len(ctrl)))
        a_i, b_i = pick[0], pick[1]
        A, B = ctrl[a_i], ctrl[b_i]
        result["chosen"] = {
            "A": {
                "idx": a_i,
                "token": A,
                "name": names[a_i] if a_i < len(names) else None,
                "tech": techs[a_i] if a_i < len(techs) else None,
            },
            "B": {
                "idx": b_i,
                "token": B,
                "name": names[b_i] if b_i < len(names) else None,
                "tech": techs[b_i] if b_i < len(techs) else None,
            },
        }
        print(f"chosen adapters: {result['chosen']}", file=sys.stderr)

        # aLoRA-style placement: control token LATE, just before the final question.
        def turn2(ctok):
            return [*hist_ids, ctok, *q2]

        off = os.path.getsize(trace) if os.path.exists(trace) else 0

        def run(tag, token_ids):
            nonlocal off
            r = ask(token_ids, tag)
            recs, off = _read(trace, off)
            return r["choices"][0]["text"], recs

        # ── Scenario 1: A then B (different adapter, warm cache) ─────────────
        # turn 1 warms the cache with A active on the tail.
        t1a_text, t1a_recs = run("s1_turn1_A", [*hist_ids, A, *q2])
        # turn 2 asks for B over the SAME history -> long prefix hit.
        warm_B_text, warm_B_recs = run("s1_turn2_B_warm", turn2(B))

        # ── Scenario 2: the SAME turn-2 request, COLD ───────────────────────
        # A fresh server would be ideal but is expensive; instead alter the history
        # so no prefix is shared, then ask B. Its own tail is identical, so a
        # difference isolates the contribution of the reused prefix.
        cold_hist = tok("User: Unrelated opening about bicycle maintenance. ")[
            "input_ids"
        ]
        cold_B_text, cold_B_recs = run("s2_turn2_B_cold", [*cold_hist, B, *q2])

        # ── Scenario 3: A then A (same adapter, warm) — control ──────────────
        warm_A_text, warm_A_recs = run("s3_turn2_A_warm", turn2(A))

        def summarize(recs, token_ids, ctok):
            """Routing + recovered n at the CONTROL token and the tokens after it."""
            want_idx = ctrl.index(ctok) + (
                0 if len(ctrl) == (cfg.get("num_adapters") or 0) + 1 else 1
            )
            ctl_pos = len(token_ids) - len(q2) - 1  # control token's own position
            found = {}
            for r in recs:
                pos = r.get("positions")
                ids = r.get("input_ids") or []
                ai = r["adapter_indices"]
                wa = r.get("write_addresses")
                qsl = r.get("query_start_loc")
                if not pos:
                    continue
                ranges = (
                    [(qsl[i], qsl[i + 1]) for i in range(len(qsl) - 1)]
                    if qsl and len(qsl) >= 2
                    else [(0, len(ids))]
                )
                for lo, hi in ranges:
                    if hi - lo < 2 or hi > len(ids) or hi > len(pos):
                        continue
                    ok = all(
                        0 <= pos[j] < len(token_ids) and ids[j] == token_ids[pos[j]]
                        for j in range(lo, hi)
                    )
                    if not ok:
                        continue
                    for j in range(lo, hi):
                        found[int(pos[j])] = {
                            "ai": int(ai[j]),
                            "n": None if wa is None else int(wa[j]),
                        }
            tail = {p: v for p, v in found.items() if p >= ctl_pos}
            return {
                "want_adapter": want_idx,
                "control_pos": ctl_pos,
                "prompt_len": len(token_ids),
                "attributed_total": len(found),
                "attributed_tail": len(tail),
                "tail_adapters": sorted({v["ai"] for v in tail.values()}),
                "tail_n": sorted({v["n"] for v in tail.values() if v["n"] is not None}),
                "tail_wrong": sorted(p for p, v in tail.items() if v["ai"] != want_idx)[
                    :8
                ],
            }

        result["scenarios"] = {
            "turn1_A": summarize(t1a_recs, [*hist_ids, A, *q2], A),
            "turn2_B_warm": summarize(warm_B_recs, turn2(B), B),
            "turn2_B_cold": summarize(cold_B_recs, [*cold_hist, B, *q2], B),
            "turn2_A_warm": summarize(warm_A_recs, turn2(A), A),
        }
        result["texts"] = {
            "turn1_A": t1a_text,
            "turn2_B_warm": warm_B_text,
            "turn2_B_cold": cold_B_text,
            "turn2_A_warm": warm_A_text,
        }
        # L3: does a warm turn-2 differ from the same request served cold?
        result["contamination"] = {
            "warm_B_vs_cold_B_text_same": warm_B_text == cold_B_text,
            "warm_B_vs_warm_A_text_same": warm_B_text == warm_A_text,
        }
        # Server-side cache-hit evidence.
        if os.path.exists(log_file):
            with open(log_file) as f:
                log = f.read()
            hits = [ln for ln in log.splitlines() if "prefix cache" in ln.lower()]
            result["server_prefix_cache_lines"] = hits[-8:]
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        if os.path.exists(log_file):
            with open(log_file) as f:
                print("--- server log tail ---\n" + f.read()[-3000:], file=sys.stderr)

    with open(args.output_path, "w") as f:
        json.dump(result, f)
    print(f"scenarios: {json.dumps(result['scenarios'], indent=1)}", file=sys.stderr)
    print("SERVE_OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output-dir", required=True)
    k = sub.add_parser("inspect-keys")
    k.add_argument("--output-path", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model-path", required=True)
    s.add_argument("--output-path", required=True)
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    return {"build": cmd_build, "inspect-keys": cmd_inspect_keys, "serve": cmd_serve}[
        a.command
    ](a)


if __name__ == "__main__":
    sys.exit(main())
