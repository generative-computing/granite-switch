# SPDX-License-Identifier: Apache-2.0
"""PRESERVE under interleaved traffic, and under cache eviction.

Two questions the A/B prefix-cache suite cannot answer, because it runs one
conversation at a time against a cache large enough to hold it:

  CONCURRENT  two PRESERVE conversations in flight together. Each has its own
              token prefix, so each should hit its own blocks and route exactly as
              it does alone -- but "should" is what this repo has learned to
              measure. The engine's batch-invariance is covered for a single
              forward; this covers two conversations whose turns interleave across
              many forwards, with their own transcripts.

  EVICT       a cache too small to keep a conversation's early blocks. PRESERVE's
              *saving* must degrade -- that is inherent -- while its *correctness*
              must not: the same ids are simply recomputed, and recomputation
              reproduces the same routing. If correctness moved with cache
              pressure, the policy would be unusable in production, where
              pressure is the normal condition.

Attribution under concurrency is the part that is easy to get wrong, so it lives
in ``attribute_spans`` -- a pure function, unit-tested on CPU in
``tests/unit/test_conversation_span_attribution.py``. Everything else here is
plumbing around a live server.

Commands (JSON to stdout, diagnostics to stderr):
  concurrent --model-path <dir> --output-path <json>
  evict      --model-path <dir> --output-path <json> [--blocks N]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.integration._conversation_prefix_cache_worker import (
    _generated_ids,
    _metrics,
    _post,
    _read_trace,
    _supported_flags,
    _wait_ready,
)

MAX_TOKENS = 16
DEFAULT_PORT = 8137
# Distinct lengths, so a span can be attributed to one conversation rather than the
# other even when both are mid-flight. Prose that differs from the first token also
# keeps their block hashes disjoint, so neither primes the other's cache.
FILLERS = {
    "A": "Storage tradeoffs: row stores favour point lookups while column stores "
    "compress better and scan wide ranges far more cheaply. ",
    "B": "Indexing for append-heavy tables: a clustered index on the append key "
    "keeps inserts sequential, and each secondary index adds write "
    "amplification, so keep them few. Partition by time so old partitions "
    "compact independently and vacuum does not walk the whole table. ",
}


# ── the trace hook: like the sibling worker's, plus per-request row spans ─────
_SITECUSTOMIZE = r"""
import importlib.abc, importlib.util, json, os, sys

_T = os.environ.get("GS_CONC_TRACE")
if _T:
    _TARGET = "granite_switch.vllm.switch.multi"

    def _patch(mod):
        cls = getattr(mod, "MultiSwitch", None)
        if cls is None or getattr(cls, "_gs_conc_patched", False):
            return
        orig = cls.forward

        def traced(self, input_ids, adapter_token_ids, positions=None):
            ai, modified = orig(self, input_ids=input_ids,
                                adapter_token_ids=adapter_token_ids,
                                positions=positions)
            try:
                qsl = seq_lens = None
                try:
                    from vllm.forward_context import get_forward_context
                    md = getattr(get_forward_context(), "attn_metadata", None)
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
                rec = {
                    "tag": os.environ.get("GS_CONC_TAG", ""),
                    "input_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
                    "positions": (None if positions is None else
                                  [int(x) for x in positions.detach().cpu().tolist()]),
                    "adapter_indices": [int(x) for x in ai.detach().cpu().tolist()],
                    "query_start_loc": qsl,
                    "seq_lens": seq_lens,
                }
                with open(_T, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                sys.stderr.write("gs-conc trace error: %r\n" % (e,))
            return ai, modified

        cls.forward = traced
        cls._gs_conc_patched = True

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


# ── the part worth unit-testing ───────────────────────────────────────────────
def attribute_spans(records, prompt_tokens):
    """Routing for ``prompt_tokens``, stitched from however many forwards carried it.

    Attribution is per SPAN, never per token. Each forward's ``query_start_loc``
    delimits one contiguous run of rows per request; a span belongs to this prompt
    only if EVERY row in it matches the prompt at the row's own recorded position.
    Requiring the whole span to agree is what makes this unambiguous when two
    concurrent prompts share individual token values, which prose reliably does.

    Single-row spans are refused outright. A decode row is one row wide, and its
    token can coincidentally equal this prompt's token at the position it reports --
    which is how a phantom "wrong routing" appears at an isolated position and
    moves between runs. Prompt positions may only be attributed from multi-token
    prefill spans.

    Returns ``{position: adapter_index}`` for prompt positions only.
    """
    got = {}
    n = len(prompt_tokens)
    for rec in records:
        ids = rec.get("input_ids") or []
        pos = rec.get("positions")
        ai = rec.get("adapter_indices") or []
        qsl = rec.get("query_start_loc")
        if not pos or not ids:
            continue
        if qsl and len(qsl) >= 2:
            ranges = [(qsl[i], qsl[i + 1]) for i in range(len(qsl) - 1)]
        else:
            ranges = [(0, len(ids))]
        for lo, hi in ranges:
            if hi - lo < 2:  # decode row: never attributable to a prompt
                continue
            if lo < 0 or hi > len(ids) or hi > len(pos) or hi > len(ai):
                continue
            if not all(
                0 <= pos[j] < n and ids[j] == prompt_tokens[pos[j]]
                for j in range(lo, hi)
            ):
                continue
            for j in range(lo, hi):
                got[int(pos[j])] = int(ai[j])
    return got


def windows_disjoint(prompt_lens, max_new):
    """Whether decode windows ``(L, L + max_new]`` cannot overlap for these prompts.

    The attribution below depends on it, so the caller must be able to check it
    rather than hope. Two prompt lengths closer together than ``max_new`` produce
    overlapping windows, and a decode row in the overlap belongs to either.
    """
    ordered = sorted(prompt_lens)
    return all(b - a > max_new for a, b in pairwise(ordered))


def decode_indices_by_seqlen(records, prompt_len, max_new):
    """Adapter indices on decode rows belonging to a request of this prompt length.

    A tag cannot do this job. The trace hook reads its tag from the *server's*
    environment, and the client sets that variable in its own process long after
    the server started, so the value never arrives -- the rows carry whatever the
    tag was at launch. (This is why the first GPU run reported ``[]`` for every
    conversation.) Under concurrency there is no single correct tag anyway, since
    two conversations are in flight at once.

    ``seq_lens`` can. During decode a request's seq_len is its prompt length plus
    however many tokens it has generated so far, so a row belongs to this
    conversation exactly when ``prompt_len < seq_len <= prompt_len + max_new``.
    That is what the deliberately distinct prompt lengths buy: with the lengths
    further apart than ``max_new`` the windows are disjoint, which
    ``windows_disjoint`` lets the caller assert.

    The window also excludes prefill rows for free: a chunk's seq_len is at most
    the prompt length, so it falls below the window even when the chunk happens to
    be one token wide.
    """
    out = set()
    lo, hi = prompt_len, prompt_len + max_new
    for rec in records:
        ids = rec.get("input_ids") or []
        ai = rec.get("adapter_indices") or []
        qsl = rec.get("query_start_loc")
        seq_lens = rec.get("seq_lens")
        if not ids or not ai:
            continue
        if qsl and len(qsl) >= 2:
            spans = [(qsl[i], qsl[i + 1], i) for i in range(len(qsl) - 1)]
        else:
            spans = [(0, len(ids), 0)]
        for start, end, req in spans:
            if end - start != 1:  # not a decode row
                continue
            if not seq_lens or req >= len(seq_lens) or start >= len(ai):
                continue
            if lo < seq_lens[req] <= hi:
                out.add(int(ai[start]))
    return sorted(out)


# ── shared server plumbing ────────────────────────────────────────────────────
def _launch(model_path, port, workdir, extra):
    from granite_switch.tutorials.vllm_server import (
        kill_stale_vllm_processes,
        launch_vllm,
    )

    with open(os.path.join(workdir, "sitecustomize.py"), "w") as f:
        f.write(_SITECUSTOMIZE)
    trace = os.path.join(workdir, "trace.jsonl")
    log_file = os.path.join(workdir, "server.log")
    os.environ["GS_CONC_TRACE"] = trace
    os.environ["PYTHONPATH"] = workdir + os.pathsep + os.environ.get("PYTHONPATH", "")
    kill_stale_vllm_processes()
    proc = launch_vllm(
        model=model_path,
        port=port,
        log_file=log_file,
        max_num_seqs=8,
        enforce_eager=True,
        max_model_len=4096,
        extra_args=_supported_flags(
            ["--enable-prefix-caching", "--return-tokens-as-token-ids", *extra]
        ),
    )
    return proc, trace, log_file


def _setup(model_path):
    from transformers import AutoTokenizer

    from granite_switch import GraniteSwitchConfig
    from tests.integration._conversation_prefix_cache_worker import _parse_adapter_map

    tok = AutoTokenizer.from_pretrained(model_path)
    config = GraniteSwitchConfig.from_pretrained(model_path)
    amap = _parse_adapter_map(tok.chat_template or "")
    aloras = [n for n, (_t, tech, _i) in amap.items() if tech == "alora"]
    ordered = list(config.adapter_token_ids or [])
    off = 0 if len(ordered) == (config.num_adapters or 0) + 1 else 1

    def expected(name):
        return ordered.index(tok.convert_tokens_to_ids(amap[name][0])) + off

    return tok, config, amap, aloras, expected


def _user_text(base, invocation):
    if invocation and "start_of_role" not in invocation:
        return f"{base} {invocation}"
    return base


def _run_turn(port, model_path, conv, question, adapter, tag):
    os.environ["GS_CONC_TAG"] = tag
    prompt = conv.build_prompt(adapter=adapter)
    choice = _post(
        port,
        {
            "model": model_path,
            "prompt": list(prompt),
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "logprobs": 1,
        },
    )["choices"][0]
    ids = _generated_ids(choice)
    conv.record_answer(ids or choice["text"], adapter=adapter)
    return prompt, ("logprobs" if ids else "text")


# ── CONCURRENT ────────────────────────────────────────────────────────────────
def cmd_concurrent(args):
    import tempfile

    from granite_switch import Conversation, KVHistoryPolicy

    tok, config, amap, aloras, expected = _setup(args.model_path)
    result = {"switch_type": config.switch_type, "alora_names": aloras, "arms": {}}
    if len(aloras) < 2:
        result["error"] = f"need >=2 alora adapters, got {aloras}"
        Path(args.output_path).write_text(json.dumps(result))
        print("CONCURRENT_OK")
        return 0

    a_name, b_name = aloras[0], aloras[1]
    a_inv, b_inv = amap[a_name][2], amap[b_name][2]
    workdir = tempfile.mkdtemp(prefix="gs_conc_")
    proc, trace, log_file = _launch(args.model_path, args.port, workdir, [])
    try:
        _wait_ready(args.port, proc, log_file)
        offset = os.path.getsize(trace) if os.path.exists(trace) else 0

        def build(label, arm):
            """Turn 1 for one conversation; returns it plus its turn-2 question."""
            conv = Conversation(
                tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
            )
            # Arm in the marker as well as the conversation label: the solo arm
            # must not leave blocks that the concurrent arm can hit, or the
            # comparison measures caching rather than concurrency.
            conv.user(
                _user_text(
                    f"Case {arm}-{label}. {FILLERS[label]}Which techniques apply?",
                    a_inv,
                )
            )
            _run_turn(args.port, args.model_path, conv, None, a_name, f"{arm}{label}1")
            conv.user(_user_text("Now give me that as JSON.", b_inv))
            return conv

        for arm in ("solo", "conc"):
            convs = {label: build(label, arm) for label in ("A", "B")}
            _read_trace(trace, offset)  # discard turn-1 rows
            offset = os.path.getsize(trace)
            before = _metrics(args.port)

            if arm == "solo":
                prompts = {
                    label: _run_turn(
                        args.port,
                        args.model_path,
                        convs[label],
                        None,
                        b_name,
                        f"{arm}{label}2",
                    )[0]
                    for label in ("A", "B")
                }
            else:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    futs = {
                        label: ex.submit(
                            _run_turn,
                            args.port,
                            args.model_path,
                            convs[label],
                            None,
                            b_name,
                            f"{arm}{label}2",
                        )
                        for label in ("A", "B")
                    }
                    prompts = {label: f.result()[0] for label, f in futs.items()}

            after = _metrics(args.port)
            records, offset = _read_trace(trace, offset)
            widths = [
                (
                    [
                        r["query_start_loc"][i + 1] - r["query_start_loc"][i]
                        for i in range(len(r["query_start_loc"]) - 1)
                    ]
                    if r.get("query_start_loc") and len(r["query_start_loc"]) >= 2
                    else [len(r["input_ids"])]
                )
                for r in records
            ]
            result["arms"][arm] = {
                "multi_request_forwards": sum(1 for w in widths if len(w) > 1),
                "forwards": len(records),
                "max_new_tokens": MAX_TOKENS,
                # The attribution below is only sound if these do not overlap.
                "decode_windows_disjoint": windows_disjoint(
                    [len(prompts[la]) for la in ("A", "B")], MAX_TOKENS
                ),
                "hits_delta": (
                    after.get("vllm:prefix_cache_hits_total", 0.0)
                    - before.get("vllm:prefix_cache_hits_total", 0.0)
                ),
                "conversations": {
                    label: {
                        "prompt_len": len(prompts[label]),
                        "expected_index": expected(b_name),
                        "routing": {
                            str(k): v
                            for k, v in attribute_spans(
                                records, list(prompts[label])
                            ).items()
                        },
                        "decode_indices": decode_indices_by_seqlen(
                            records, len(prompts[label]), MAX_TOKENS
                        ),
                    }
                    for label in ("A", "B")
                },
            }
    finally:
        _shutdown(proc, log_file)

    Path(args.output_path).write_text(json.dumps(result))
    print("CONCURRENT_OK")
    return 0


# ── EVICT ─────────────────────────────────────────────────────────────────────
def cmd_evict(args):
    import tempfile

    from granite_switch import Conversation, KVHistoryPolicy

    tok, config, amap, aloras, expected = _setup(args.model_path)
    result = {"switch_type": config.switch_type, "blocks": args.blocks, "turns": []}
    if len(aloras) < 2:
        result["error"] = f"need >=2 alora adapters, got {aloras}"
        Path(args.output_path).write_text(json.dumps(result))
        print("EVICT_OK")
        return 0

    a_name, b_name = aloras[0], aloras[1]
    workdir = tempfile.mkdtemp(prefix="gs_evict_")
    # A cache far too small to keep the conversation resident, so early blocks are
    # evicted between turns. The point is that correctness must not follow the hit
    # rate down.
    proc, trace, log_file = _launch(
        args.model_path,
        args.port,
        workdir,
        ["--num-gpu-blocks-override", str(args.blocks)],
    )
    try:
        _wait_ready(args.port, proc, log_file)
        offset = os.path.getsize(trace) if os.path.exists(trace) else 0
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        for turn in range(args.turns):
            adapter = a_name if turn % 2 == 0 else b_name
            question = (
                f"{FILLERS['B']}Turn {turn}: which techniques apply?"
                if turn == 0
                else f"Turn {turn}: restate that as JSON."
            )
            conv.user(_user_text(question, amap[adapter][2]))
            sent_before = len(conv.sent_token_ids)
            before = _metrics(args.port)
            prompt, src = _run_turn(
                args.port, args.model_path, conv, None, adapter, f"evict{turn}"
            )
            after = _metrics(args.port)
            records, offset = _read_trace(trace, offset)
            result["turns"].append(
                {
                    "turn": turn,
                    "adapter": adapter,
                    "expected_index": expected(adapter),
                    "prompt_len": len(prompt),
                    "answer_ids_source": src,
                    # What the block arithmetic makes available for THIS turn: the
                    # transcript as it stood before the turn, rounded down to whole
                    # 16-token blocks. Hits below it mean blocks that should have been
                    # resident were not -- direct evidence of eviction, rather than an
                    # inference from cache size.
                    "reusable_ceiling": (sent_before // 16) * 16,
                    "sent_len": len(conv.sent_token_ids),
                    "hits_delta": (
                        after.get("vllm:prefix_cache_hits_total", 0.0)
                        - before.get("vllm:prefix_cache_hits_total", 0.0)
                    ),
                    "decode_indices": decode_indices_by_seqlen(
                        records, len(prompt), MAX_TOKENS
                    ),
                    "history_routing": sorted(
                        set(attribute_spans(records, list(prompt)).values())
                    ),
                }
            )
    finally:
        _shutdown(proc, log_file)

    Path(args.output_path).write_text(json.dumps(result))
    print("EVICT_OK")
    return 0


def _shutdown(proc, log_file):
    try:
        proc.terminate()
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    if os.path.exists(log_file):
        print(
            "--- server log tail ---\n" + open(log_file).read()[-2500:], file=sys.stderr
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("concurrent")
    c.add_argument("--model-path", required=True)
    c.add_argument("--output-path", required=True)
    c.add_argument("--port", type=int, default=DEFAULT_PORT)
    e = sub.add_parser("evict")
    e.add_argument("--model-path", required=True)
    e.add_argument("--output-path", required=True)
    e.add_argument("--port", type=int, default=DEFAULT_PORT + 1)
    e.add_argument("--blocks", type=int, default=16)
    e.add_argument("--turns", type=int, default=4)
    args = ap.parse_args()
    return {"concurrent": cmd_concurrent, "evict": cmd_evict}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
