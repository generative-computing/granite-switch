# SPDX-License-Identifier: Apache-2.0
"""Real-model SERVING e2e worker for the coded MultiSwitch.

Every other MultiSwitch test either (a) uses a mock switch config with synthetic
geometry and no real weights, or (b) drives the real checkpoint through OFFLINE
``llm.generate``. Neither exercises vLLM as a *server*: requests arriving
independently over HTTP, the scheduler interleaving them, chunked prefill splitting
long prompts, and genuine mixed prefill+decode forwards.

This worker runs the real thing:

  1. compose granite-4.1-3b + granitelib LoRA adapters (``--switch-type multi``),
  2. launch ``vllm.entrypoints.openai.api_server`` on that checkpoint with
     ``--enable-chunked-prefill`` and a small ``--max-num-batched-tokens``, so the
     SCHEDULER naturally produces chunked prefill and mixed forwards,
  3. fire CONCURRENT HTTP completion requests of varied length and adapter content,
  4. capture, from inside the server process, every ``MultiSwitch.forward``
     call — its ``adapter_indices``, recovered count ``n``, ``query_start_loc`` and
     ``seq_lens`` — via a ``sitecustomize.py`` on ``PYTHONPATH`` (CPython imports it
     in every process, which is the only way to instrument a separately-launched
     server),
  5. report both the HTTP responses (what a user sees) and the routing trace (what
     the switch actually did), plus which forward SHAPES actually occurred.

Both signals matter: a routing bug that does not change the text is real and
invisible to output-only assertions (previously observed: identical output with
23/23 wrong decode routing). Conversely a routing trace that never saw a chunked or
multi-request forward proves nothing about serving.

Commands (JSON to stdout; diagnostics to stderr):
  build --output-dir <dir>
  serve --model-path <dir> --output-path <json> [--port N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_MODEL = "ibm-granite/granite-4.1-3b"
ADAPTER_REPOS = [
    "ibm-granite/granitelib-core-r1.0",
    "ibm-granite/granitelib-guardian-r1.0",
    "ibm-granite/granitelib-rag-r1.0",
]
DEFAULT_PORT = 8123
MAX_TOKENS = 24
# Small enough that a long prompt MUST be split across forwards.
MAX_NUM_BATCHED_TOKENS = 256


def _compose(out_dir):
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        BASE_MODEL,
        "--adapters",
        *ADAPTER_REPOS,
        "--technology-filter",
        "lora",
        "--output",
        out_dir,
    ]
    print("composing:", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-2500:], file=sys.stderr)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"compose failed ({r.returncode})")


def cmd_build(args):
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"warm-reuse {args.output_dir}", file=sys.stderr)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        _compose(args.output_dir)
    with open(os.path.join(args.output_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("switch_type") == "multi", (
        f"switch_type={cfg.get('switch_type')!r}, expected 'multi'"
    )
    print(
        f"num_adapters={cfg.get('num_adapters')} ctrl={cfg.get('adapter_token_ids')}",
        file=sys.stderr,
    )
    print("BUILD_OK")
    return 0


# ── in-server instrumentation ───────────────────────────────────────────────
# The API server is a separate process we launch, so an in-process monkeypatch is
# useless. sitecustomize.py is imported by CPython at interpreter startup in EVERY
# process, including the server and its EngineCore child.
_SITECUSTOMIZE = r"""
import importlib.abc
import importlib.util
import json
import os
import sys

_T = os.environ.get("GS_SERVE_TRACE")
if _T:
    _TARGET = "granite_switch.vllm.switch.multi"

    def _patch(mod):
        cls = getattr(mod, "MultiSwitch", None)
        if cls is None or getattr(cls, "_gs_serve_patched", False):
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
                    "pid": os.getpid(),
                    "input_ids": [int(x) for x in input_ids.detach().cpu().tolist()],
                    "positions": (
                        None
                        if positions is None
                        else [int(x) for x in positions.detach().cpu().tolist()]
                    ),
                    "adapter_indices": [int(x) for x in ai.detach().cpu().tolist()],
                    "write_addresses": (
                        None
                        if wa is None
                        else [int(x) for x in wa.detach().cpu().tolist()]
                    ),
                    "query_start_loc": qsl,
                    "seq_lens": seq_lens,
                }
                with open(_T, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                sys.stderr.write("gs-serve trace error: %r\n" % (e,))
            return ai, modified

        cls.forward = traced
        cls._gs_serve_patched = True
        sys.stderr.write("gs-serve: patched in pid %d\n" % os.getpid())

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


def _write_sitecustomize(d):
    with open(os.path.join(d, "sitecustomize.py"), "w") as f:
        f.write(_SITECUSTOMIZE)


def _read_trace(p, start_offset=0):
    """Read records from ``start_offset`` bytes onward; return (records, new_offset).

    NEVER truncate this file. The server process holds it open in append mode, so
    truncating from here does not reset the server's file offset: it keeps writing at
    its old position, leaving a hole of NUL bytes and losing records. An earlier
    revision truncated between passes and saw routing coverage collapse to 2-33% of
    prompt positions (worse the longer the prompt) for exactly this reason.

    Pass boundaries are therefore byte offsets into one append-only file.
    """
    out = []
    if not os.path.exists(p):
        return out, start_offset
    with open(p, "rb") as f:
        f.seek(start_offset)
        raw = f.read()
        end = f.tell()
    for ln in raw.split(b"\n"):
        ln = ln.strip().strip(b"\x00")
        if not ln:
            continue
        try:
            rec = json.loads(ln.decode("utf-8"))
            # Stamp a stable index so attribution can report WHICH record claimed a
            # position. Provenance is what distinguishes real mis-routing from a span
            # coincidentally matched off another concurrent request.
            rec["idx"] = len(out)
            out.append(rec)
        except Exception:
            pass
    return out, end


def _post(port, payload, timeout=180):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _supported_flags(candidates):
    """Keep only flags this vLLM build's api_server actually accepts.

    api_server exits immediately on an unknown flag (a renamed
    --disable-log-requests already cost one run), and flag names for prefix caching
    and log verbosity have churned across versions. Probe --help once and filter,
    so a rename degrades to "that option was skipped" instead of "the server never
    started".
    """
    try:
        r = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        help_text = (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        print(f"flag probe failed ({e!r}); passing flags unfiltered", file=sys.stderr)
        return list(candidates)
    kept, dropped = [], []
    i = 0
    while i < len(candidates):
        flag = candidates[i]
        if flag.startswith("--"):
            if flag in help_text:
                kept.append(flag)
                # Keep an immediately-following value argument with its flag.
                if i + 1 < len(candidates) and not candidates[i + 1].startswith("--"):
                    kept.append(candidates[i + 1])
                    i += 1
            else:
                dropped.append(flag)
                if i + 1 < len(candidates) and not candidates[i + 1].startswith("--"):
                    i += 1
        i += 1
    if dropped:
        print(
            f"WARNING: api_server does not accept {dropped}; skipped", file=sys.stderr
        )
    return kept


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


# REAL-LANGUAGE filler, never repeated single tokens. A prompt like
# [50, 50, ..., ctrl, 50, 50] is badly out of distribution and greedy decoding on it
# sits on near-ties, so a request can return different (equally meaningless) text
# solo vs concurrently while its ADAPTER ROUTING is byte-identical. Asserting on
# generated text over junk input tests the sampler, not the switch. (This is the
# lesson recorded in tests/integration/test_multi_switch_vllm_generate.py.)
# Enough DISTINCT sentences that a long prompt is varied prose rather than the same
# few sentences cycled. Repetition is the failure mode to avoid: a prompt built by
# repeating a handful of sentences is still low-entropy, greedy decoding on it sits on
# near-ties, and the returned text can differ solo vs concurrent while routing is
# byte-identical. An earlier revision cycled 6 sentences and its 614-token prompt did
# exactly that, which would make a text-stability assertion flaky rather than
# meaningful.
_PROSE = [
    "A hash function maps data of arbitrary size to a fixed-size value. ",
    "Entropy measures the number of microscopic configurations of a system. ",
    "A binary search tree keeps keys ordered so lookups take logarithmic time. ",
    "Gradient descent iteratively moves parameters against the gradient. ",
    "A cache stores recently used values so repeated reads are cheaper. ",
    "Compilers translate source code into a lower-level representation. ",
    "Photosynthesis converts light energy into chemical energy in plants. ",
    "The printing press made written knowledge far cheaper to reproduce. ",
    "Ocean currents redistribute heat between the equator and the poles. ",
    "A vaccine trains the immune system using a harmless antigen. ",
    "Plate tectonics explains why mountain ranges form where they do. ",
    "Public-key cryptography lets strangers agree on a shared secret. ",
    "Antibiotic resistance spreads when selection pressure is sustained. ",
    "Bridges are designed so that expected loads stay well inside tolerance. ",
    "The water cycle moves moisture between oceans, air, and land. ",
    "Sorting a list first often makes later queries dramatically cheaper. ",
    "Sound travels faster through water than through air. ",
    "Bees communicate the direction of food using a waggle dance. ",
    "Glaciers carve valleys by grinding rock beneath enormous weight. ",
    "A transformer changes voltage without changing frequency. ",
    "Sedimentary layers record the order in which material was deposited. ",
    "Latency and bandwidth are independent properties of a network link. ",
    "Yeast produces carbon dioxide, which makes bread dough rise. ",
    "Telescopes gather more light as their aperture increases. ",
    "Erosion reshapes coastlines over decades and centuries. ",
    "A pendulum's period depends on its length, not its mass. ",
    "Deciduous trees shed leaves to conserve water in winter. ",
    "Radio waves diffract around obstacles more readily than light. ",
    "Fermentation preserves food by lowering its pH. ",
    "Migratory birds navigate using both stars and magnetic cues. ",
]


def _build_prompts(ctrl, tokenizer):
    """Real-language prompts with control tokens at known positions.

    Ground truth is latest-wins over the token stream. Requirements encoded here:

    * **real prose, not repeated filler** — see ``_PROSE`` above,
    * **DISTINCT total lengths** so a decode row can be attributed unambiguously,
    * **>=3 control tokens** on several prompts: with fewer, an off-by-one codebook
      address can still resolve to the same adapter and hide a defect,
    * **prompts longer than ``MAX_NUM_BATCHED_TOKENS``** so the scheduler MUST split
      them into chunked prefills.
    """
    offset = 1  # composer emits exactly num_adapters control tokens (no base slot)
    acts = list(ctrl)

    # ~1500 prompt tokens are needed in total (prompts must exceed
    # MAX_NUM_BATCHED_TOKENS to force chunking), which is far more text than a
    # hand-written list supplies. Compose each sentence from the base sentence plus a
    # varying qualifier, giving a large pool of DISTINCT sentences without cycling.
    # Cycling a few sentences produces low-entropy text that sits on greedy near-ties,
    # which would make the text-stability assertion flaky (see _PROSE).
    _QUALIFIERS = [
        "This holds in most practical cases.",
        "Engineers rely on this when planning capacity.",
        "The effect is larger at higher temperatures.",
        "Measurements confirmed this in the 1950s.",
        "It follows directly from conservation of energy.",
        "Textbooks usually introduce this early.",
        "The magnitude depends on local conditions.",
        "Careful experiments separate cause from correlation.",
        "Small changes accumulate over long periods.",
        "The same reasoning applies at larger scales.",
        "Field data broadly match the prediction.",
        "Exceptions arise when pressure is extreme.",
    ]
    pool = [f"{s.strip()} {q} " for q in _QUALIFIERS for s in _PROSE]
    cursor = {"i": 0}
    reuse = {"n": 0}

    def prose_ids(min_tokens):
        """>= min_tokens tokens of DISTINCT prose, advancing a global cursor."""
        ids = []
        while len(ids) < min_tokens:
            i = cursor["i"]
            if i >= len(pool):
                # Pool exhausted: wrap, but record it so the caller can see that this
                # run reused text and treat text-stability results with suspicion.
                reuse["n"] += 1
                cursor["i"] = 0
                i = 0
            ids.extend(tokenizer(pool[i])["input_ids"])
            cursor["i"] = i + 1
        return ids[:min_tokens]

    def seq(spec):
        ids, exp, cur = [], [], 0
        for n, ctok in spec:
            chunk = prose_ids(n)
            ids.extend(chunk)
            exp.extend([cur] * len(chunk))
            if ctok is not None:
                ids.append(ctok)
                cur = acts.index(ctok) + offset
                exp.append(cur)
        return ids, exp

    specs = [
        ("base_short", [(24, None)]),
        ("one_ctl", [(40, acts[0]), (20, None)]),
        (
            "three_ctl_long",
            [
                (150, acts[1]),
                (120, acts[2]),
                (100, acts[3]),
                (40, None),
            ],
        ),
        (
            "four_ctl_longer",
            [
                (200, acts[4]),
                (150, acts[5]),
                (120, acts[0]),
                (90, acts[2]),
                (50, None),
            ],
        ),
        ("many_ctl", [(60, acts[i % len(acts)]) for i in range(6)] + [(30, None)]),
    ]
    out = []
    for name, spec in specs:
        ids, exp = seq(spec)
        assert len(ids) == len(exp), "ground truth length mismatch"
        out.append(
            {
                "name": name,
                "token_ids": ids,
                "expected": exp,
                "final_adapter": exp[-1],
                "n_controls": sum(1 for _, c in spec if c is not None),
            }
        )
    lens = [len(p["token_ids"]) for p in out]
    assert len(set(lens)) == len(lens), f"lengths must be distinct: {lens}"
    assert max(lens) > MAX_NUM_BATCHED_TOKENS, (
        f"longest prompt {max(lens)} <= max_num_batched_tokens "
        f"{MAX_NUM_BATCHED_TOKENS}; nothing would be chunked"
    )
    print(
        f"prompts: {[(p['name'], len(p['token_ids'])) for p in out]}", file=sys.stderr
    )
    # Reuse means the prose pool ran out and text repeated, which reintroduces the
    # greedy near-tie hazard the pool exists to avoid. Report it rather than hide it.
    print(
        f"prose pool: {len(pool)} distinct sentences, wrapped {reuse['n']}x "
        f"(must be 0; wrapping repeats text and reintroduces greedy near-ties)",
        file=sys.stderr,
    )
    assert reuse["n"] == 0, (
        f"prose pool exhausted ({len(pool)} sentences) and wrapped {reuse['n']}x, so "
        "prompt text repeats. Repeated text sits on greedy near-ties, which makes the "
        "solo-vs-concurrent TEXT assertion flaky rather than meaningful. Enlarge "
        "_PROSE/_QUALIFIERS instead of accepting the reuse."
    )
    return out, reuse["n"]


def _find_subseq(hay, needle):
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if hay[i : i + m] == needle:
            return i
    return -1


def _classify_shapes(recs, prompts):
    """What forward SHAPES did the scheduler actually produce?

    Without this the run is unfalsifiable: a trace that only ever saw whole-request
    single-prefill forwards says nothing about serving.
    """
    shapes = {
        "total_forwards": len(recs),
        "chunked_prefill_forwards": 0,  # a span whose seq_len > query_len (>1 query)
        "pure_decode_forwards": 0,  # all spans == 1
        "mixed_forwards": 0,  # some span == 1 and some > 1
        "multi_request_forwards": 0,  # >= 2 requests in one forward
        "max_requests_in_forward": 0,
        "max_flat_tokens": 0,
        "pids": sorted({r.get("pid") for r in recs if r.get("pid")}),
    }
    for r in recs:
        qsl, sl = r.get("query_start_loc"), r.get("seq_lens")
        shapes["max_flat_tokens"] = max(
            shapes["max_flat_tokens"], len(r.get("input_ids") or [])
        )
        if not qsl or len(qsl) < 2:
            continue
        spans = [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]
        nreq = len(spans)
        shapes["max_requests_in_forward"] = max(shapes["max_requests_in_forward"], nreq)
        if nreq >= 2:
            shapes["multi_request_forwards"] += 1
        if all(s == 1 for s in spans):
            shapes["pure_decode_forwards"] += 1
        elif any(s == 1 for s in spans) and any(s > 1 for s in spans):
            shapes["mixed_forwards"] += 1
        # Chunked: a multi-token span whose request already has cached tokens.
        if sl and len(sl) == nreq:
            for span, seqlen in zip(spans, sl):
                if span > 1 and seqlen > span:
                    shapes["chunked_prefill_forwards"] += 1
                    break
    return shapes


def _routing_for_prompt(recs, prompt_tokens):
    """Stitch this prompt's prefill routing across however many forwards carried it.

    Attribution is per SPAN, not per token. Each forward's ``query_start_loc``
    delimits one contiguous run of rows per request; a span belongs to this prompt
    only if EVERY row in it matches the prompt at the row's own recorded absolute
    position. Requiring the whole span to agree makes attribution unambiguous even
    when concurrent prompts share individual token values — a per-token match (an
    earlier revision) both missed rows and could attribute a row to the wrong
    request.

    Under chunked prefill a prompt arrives as several multi-token spans across
    forwards; under decode its rows are single-token spans at positions >= len(prompt)
    which are ignored here (prefill positions only).
    """
    got = {}
    n = len(prompt_tokens)
    for rec in recs:
        ids = rec.get("input_ids") or []
        pos = rec.get("positions")
        ai = rec["adapter_indices"]
        wa = rec.get("write_addresses")
        qsl = rec.get("query_start_loc")
        if pos is None or not ids:
            continue
        # Row ranges for this forward: real spans when qsl is present, else one span.
        if qsl and len(qsl) >= 2:
            ranges = [(qsl[i], qsl[i + 1]) for i in range(len(qsl) - 1)]
        else:
            ranges = [(0, len(ids))]
        for lo, hi in ranges:
            if lo >= hi or hi > len(ids) or hi > len(pos):
                continue
            # A PREFILL span is multi-token. A single-token span is a DECODE row for
            # some request, and its token can coincidentally equal this prompt's token
            # at the position it reports -- which is exactly how positions 67 and 71 of
            # a 614-token prompt got claimed from one_ctl's decode rows (width=1,
            # spans=[1,1,1,1,1], seq_lens identifying a 68-token sequence), producing a
            # phantom "wrong routing" that moved between runs. Prompt positions may
            # only be attributed from multi-token prefill spans.
            if hi - lo < 2:
                continue
            # Every row in the span must sit inside the prompt and match it there.
            ok = True
            for j in range(lo, hi):
                p = pos[j]
                if not (0 <= p < n) or ids[j] != prompt_tokens[p]:
                    ok = False
                    break
            if not ok:
                continue
            for j in range(lo, hi):
                p = int(pos[j])
                got[p] = {
                    "adapter_index": int(ai[j]),
                    "n": (None if wa is None else int(wa[j])),
                    # PROVENANCE: which record/span claimed this position, and what
                    # else was in that forward. A wrong adapter at an isolated
                    # position could be real mis-routing OR a span coincidentally
                    # matched from ANOTHER concurrent request (prose-similar prompts
                    # can agree over a short run). These fields decide which:
                    # a span whose seq_lens/width belong to a different request's
                    # chunk is theft; one consistent with this prompt's own segment
                    # structure is real.
                    "src": {
                        "rec": rec.get("idx"),
                        "span": [lo, hi],
                        "span_width": hi - lo,
                        "n_ids": len(ids),
                        "seq_lens": rec.get("seq_lens"),
                        "all_spans": (
                            [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]
                            if qsl and len(qsl) >= 2
                            else None
                        ),
                    },
                }
    return got


def cmd_serve(args):
    import tempfile

    from granite_switch.tutorials.vllm_server import (
        kill_stale_vllm_processes,
        launch_vllm,
    )

    port = args.port
    probe_dir = tempfile.mkdtemp(prefix="gs_serve_")
    _write_sitecustomize(probe_dir)
    trace = os.path.join(probe_dir, "trace.jsonl")
    log_file = os.path.join(probe_dir, "server.log")

    with open(os.path.join(args.model_path, "config.json")) as f:
        cfg = json.load(f)
    ctrl = list(cfg["adapter_token_ids"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts, prose_reuse = _build_prompts(ctrl, tokenizer)

    # The server inherits these: the hook writes the trace, PYTHONPATH makes
    # sitecustomize importable in the server AND its EngineCore child.
    env_pp = probe_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["GS_SERVE_TRACE"] = trace
    os.environ["PYTHONPATH"] = env_pp

    kill_stale_vllm_processes()
    proc = launch_vllm(
        model=args.model_path,
        port=port,
        log_file=log_file,
        max_num_seqs=8,
        enforce_eager=True,
        max_model_len=4096,
        # Only flags that are stable across vLLM versions. (An earlier revision
        # passed --disable-log-requests, which this build renamed to
        # --no-enable-log-requests; api_server exits immediately on an unknown
        # flag, so keep this list minimal and functional.)
        extra_args=_supported_flags(
            [
                "--enable-chunked-prefill",
                "--max-num-batched-tokens",
                str(MAX_NUM_BATCHED_TOKENS),
                # MUST disable prefix caching. It is ON by default in vLLM V1, and the
                # solo pass prefills every prompt; the concurrent pass then sends the
                # SAME prompts, so their prefill is served from the prefix cache and only
                # a small tail chunk re-runs the switch. That is what capped routing
                # coverage at 1-33% (8-13 attributed positions on a 614-token prompt) and
                # made the concurrent pass mostly decode. Disabling it forces every pass
                # to prefill for real, which is also the honest comparison: solo vs
                # concurrent must differ only in scheduling, not in how much work runs.
                "--no-enable-prefix-caching",
            ]
        ),
    )
    result = {"prompts": {}, "shapes": {}, "http": {}}
    try:
        _wait_ready(port, proc, log_file)
        print("server ready", file=sys.stderr)

        def one(p, extra=None):
            payload = {
                "model": args.model_path,
                "prompt": p["token_ids"],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "logprobs": 0,
            }
            if extra:
                payload.update(extra)
            return _post(port, payload)

        # Pass boundaries are byte offsets into one APPEND-ONLY trace file. Do not
        # truncate: the server holds the file open in append mode and would keep
        # writing at its old offset, losing records (see _read_trace).
        off = os.path.getsize(trace) if os.path.exists(trace) else 0

        # ── PASS 1: solo, sequential (reference) ────────────────────────────
        solo = {}
        for p in prompts:
            r = one(p)
            recs, off = _read_trace(trace, off)
            solo[p["name"]] = {
                "text": r["choices"][0]["text"],
                "routing": _routing_for_prompt(recs, p["token_ids"]),
                "records": len(recs),
            }
        # ── PASS 2: all CONCURRENT (real serving: interleaved scheduling) ────
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            futs = {ex.submit(one, p): p["name"] for p in prompts}
            conc = {}
            for fut, name in [(f, futs[f]) for f in futs]:
                conc[name] = fut.result()
        conc_recs, off = _read_trace(trace, off)
        result["shapes"] = _classify_shapes(conc_recs, prompts)
        # Total queried rows vs total prompt tokens. If prefix caching is serving the
        # prefill (it is ON by default in vLLM V1), the concurrent pass mostly decodes
        # and this ratio collapses -- which is exactly how coverage silently sat at
        # 1-33%. Surfacing it makes that failure mode self-evident next time.
        rows = sum(len(r.get("input_ids") or []) for r in conc_recs)
        prompt_tokens = sum(len(p["token_ids"]) for p in prompts)
        result["shapes"]["queried_rows_total"] = rows
        result["shapes"]["prompt_tokens_total"] = prompt_tokens
        print(
            f"queried rows {rows} vs prompt tokens {prompt_tokens} "
            f"(ratio {rows / max(1, prompt_tokens):.2f}; well under 1.0 means prefill "
            f"was served from cache rather than recomputed)",
            file=sys.stderr,
        )

        # ── DIAGNOSTIC: what do the records actually look like? ───────────────
        # Prefill rows have been missing from the trace across several runs
        # (max_flat_tokens 45 against 614-token prompts), so dump the raw shape of
        # every record rather than guessing which attribution step drops them.
        diag = []
        for k, r in enumerate(conc_recs):
            ids = r.get("input_ids") or []
            pos = r.get("positions")
            qsl = r.get("query_start_loc")
            spans = (
                [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]
                if qsl and len(qsl) >= 2
                else None
            )
            diag.append(
                {
                    "i": k,
                    "pid": r.get("pid"),
                    "n_ids": len(ids),
                    "spans": spans,
                    "seq_lens": r.get("seq_lens"),
                    "pos_first": (pos[:4] if pos else None),
                    "pos_last": (pos[-4:] if pos else None),
                    "pos_is_none": pos is None,
                }
            )
        result["diag_records"] = diag
        print(f"DIAG: {len(diag)} records", file=sys.stderr)
        for d in diag[:30]:
            print(f"  DIAG {d}", file=sys.stderr)
        # Per-prompt: how many positions did span-attribution actually claim, and
        # would a looser per-token match have claimed more?
        for p in prompts:
            span_got = _routing_for_prompt(conc_recs, p["token_ids"])
            loose = {}
            for r in conc_recs:
                ids = r.get("input_ids") or []
                pos = r.get("positions")
                if not pos:
                    continue
                for j, pp in enumerate(pos):
                    if 0 <= pp < len(p["token_ids"]) and j < len(ids):
                        if ids[j] == p["token_ids"][pp]:
                            loose[int(pp)] = 1
            print(
                f"  DIAG-ATTR {p['name']:16s} len={len(p['token_ids']):4d} "
                f"span_claimed={len(span_got):4d} loose_claimed={len(loose):4d}",
                file=sys.stderr,
            )

        # ── PASS 3: determinism — repeat one prompt concurrently x3 ──────────
        rep = prompts[2]
        with ThreadPoolExecutor(max_workers=3) as ex:
            reps = [ex.submit(one, rep) for _ in range(3)]
            rep_texts = [f.result()["choices"][0]["text"] for f in reps]

        for p in prompts:
            name = p["name"]
            result["prompts"][name] = {
                "len": len(p["token_ids"]),
                "n_controls": p["n_controls"],
                # Needed to judge a wrong routing: a non-base adapter BEFORE the
                # first control token cannot be latest-wins under any count.
                "control_positions": [
                    i for i, t in enumerate(p["token_ids"]) if t in ctrl
                ],
                "expected": p["expected"],
                "final_adapter": p["final_adapter"],
                "solo_text": solo[name]["text"],
                "solo_routing": solo[name]["routing"],
                "conc_text": conc[name]["choices"][0]["text"],
                "conc_routing": _routing_for_prompt(conc_recs, p["token_ids"]),
            }
        result["http"] = {
            "repeat_texts_identical": len(set(rep_texts)) == 1,
            "repeat_prompt": rep["name"],
            "n_concurrent": len(prompts),
            "prose_pool_size": len(_PROSE),
            "prose_reuse_wraps": prose_reuse,
        }
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        if os.path.exists(log_file):
            with open(log_file) as f:
                tail = f.read()[-4000:]
            print("--- server log tail ---\n" + tail, file=sys.stderr)

    with open(args.output_path, "w") as f:
        json.dump(result, f)
    print(f"shapes: {result['shapes']}", file=sys.stderr)
    print("SERVE_OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output-dir", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model-path", required=True)
    s.add_argument("--output-path", required=True)
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    return {"build": cmd_build, "serve": cmd_serve}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
