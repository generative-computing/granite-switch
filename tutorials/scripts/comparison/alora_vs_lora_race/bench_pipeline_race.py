#!/usr/bin/env python3
"""Live terminal race: ALORA vs LORA.

Pipeline per turn (5 turns per conversation, 32 conversations per server):
  harm ─▶ rewrite ─▶ retrieve ─▶ answerability ─▶ clarify ─▶ generate

Conversation scripts are loaded from govt_conversations.json.
Each script has 5 pre-scripted user turns:
  Q1 — happy path, Q2 — ambiguous, Q3 — follow-up,
  Q4 — unanswerable, Q5 — guardian block

Usage:
  # Simultaneous race (requires both servers running):
  python bench_pipeline_race.py

  # Sequential — run one server at a time (Colab-friendly):
  python bench_pipeline_race.py --mode sequential --server "ALORA (8111)"
  python bench_pipeline_race.py --mode sequential --server "LORA (8112)"

Results are written to race_results.json (appended per server in sequential mode).
"""

import argparse
import contextlib
import copy
import datetime
import json
import logging
import os
import re
import shutil
import statistics
import threading
import time
import urllib.request as _urllib_req
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
import torch
from chromadb import Documents, Embeddings, EmbeddingFunction
from huggingface_hub import hf_hub_download
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from transformers import AutoModel, AutoTokenizer

from mellea.backends import ModelOption
from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components import Document as MelleaDocument
from mellea.stdlib.components.chat import Message as MelleaMessage
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.components.intrinsic.guardian import guardian_check
from mellea.stdlib.context import ChatContext
import mellea.stdlib.functional as mfuncs

logging.getLogger("mellea").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)


from IPython.utils import coloransi
if not hasattr(coloransi.TermColors, 'Green'):
    coloransi.TermColors.Green = '\033[0;32m'
    coloransi.TermColors.Blue = '\033[0;34m'
    coloransi.TermColors.Normal = '\033[0m'

def _detect_notebook():
    """Return True when running inside Jupyter / Colab."""
    try:
        return get_ipython().__class__.__name__ != "TerminalInteractiveShell"
    except NameError:
        return False


# ── Shared config ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"
CHROMA_PATH        = str(_HERE / "govt_chroma")
CONVERSATIONS_PATH = str(_HERE / "govt_conversations.json")
TOP_K                  = None  # set by main()
RUNS                   = None  # set by main()
CONCURRENCY_PER_SERVER = None  # set by main()

GUARDIAN_HARM_CRITERIA = "harm"

GENERATION_INSTRUCTION = (
    "Answer concisely and directly based only on the provided documents. "
    "Do not repeat the question or add unnecessary preamble."
)

ALORA_MODEL = "ibm-granite/granite-switch-4.1-3b-preview"
LORA_MODEL  = "./granite-switch-lora-only"  # override with --lora-model

SERVERS = {
    "ALORA (8111)": {
        "base_url":    "http://localhost:8111/v1",
        "model":       ALORA_MODEL,
        "source":      ALORA_MODEL,
        "metrics_url": "http://localhost:8111/metrics",
        "log_file":    str(_HERE / "vllm_alora.log"),
    },
    "LORA (8112)": {
        "base_url":    "http://localhost:8112/v1",
        "model":       LORA_MODEL,
        "source":      LORA_MODEL,
        "metrics_url": "http://localhost:8112/metrics",
        "log_file":    str(_HERE / "vllm_lora.log"),
    },
}

# ── Load conversation scripts ─────────────────────────────────────────────────
with open(CONVERSATIONS_PATH) as _f:
    _raw_convs = json.load(_f)

CONVERSATIONS = [
    [(t["label"], t["query"]) for t in conv["turns"]]
    for conv in _raw_convs
]
print(f"Loaded {len(CONVERSATIONS)} conversation scripts from {CONVERSATIONS_PATH}")


def get_queries_for_run(run_idx):
    return CONVERSATIONS[run_idx % len(CONVERSATIONS)]


# ── Shared ChromaDB collection ────────────────────────────────────────────────
class GraniteEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_id=EMBEDDING_MODEL_ID, batch_size=64):
        device = "cpu"
        self._device    = device
        self._batch     = batch_size
        self._lock      = threading.Lock()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model     = AutoModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, input: Documents) -> Embeddings:
        with self._lock:
            all_embs = []
            for i in range(0, len(input), self._batch):
                batch = list(input[i : i + self._batch])
                enc = self._tokenizer(batch, return_tensors="pt",
                                      truncation=True, max_length=512, padding=True)
                enc = {k: v.to(self._device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self._model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                all_embs.extend(emb.cpu().float().tolist())
            return all_embs


print("Loading embedding model...")
_embed_fn  = GraniteEmbeddingFunction()
_chroma    = chromadb.PersistentClient(path=CHROMA_PATH)
collection = _chroma.get_or_create_collection(
    name="govt", embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)
print(f"  Embedding on {next(_embed_fn._model.parameters()).device}, "
      f"{collection.count():,} docs\n")

# ── Helpers for local-path or HF-hub sources ─────────────────────────────────
def _resolve_adapter_index(source):
    """Load adapter_index.json from a local directory or HF Hub repo."""
    local = Path(source) / "adapter_index.json"
    if local.exists():
        return json.loads(local.read_text())
    with open(hf_hub_download(repo_id=source, filename="adapter_index.json")) as f:
        return json.load(f)


# ── Mellea helpers ────────────────────────────────────────────────────────────
def _discover_vllm_model(base_url):
    """Query /v1/models to get the model name the vLLM server is actually using."""
    try:
        url = base_url.rstrip("/") + "/models"
        with _urllib_req.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return None


def make_backend(cfg):
    backend = OpenAIBackend(model_id=cfg["model"], base_url=cfg["base_url"], api_key="unused")
    source = cfg["source"]
    if Path(source).is_dir():
        backend.register_embedded_adapter_model(source)
    else:
        for a in EmbeddedIntrinsicAdapter.from_hub(source):
            backend.add_adapter(a)
    return backend


def build_context(history):
    ctx = ChatContext()
    for m in history:
        docs = ([MelleaDocument(doc_id=str(i), text=t)
                 for i, t in enumerate(m["documents"])]
                if m.get("documents") else None)
        ctx = ctx.add(MelleaMessage(m["role"], m["content"], documents=docs))
    return ctx


def to_mellea_docs(texts):
    return [MelleaDocument(doc_id=str(i), text=t) for i, t in enumerate(texts)]


# ── Intrinsic error dumping ───────────────────────────────────────────────────
def _serialize_ctx(ctx):
    return [
        {"role": m.role, "content": m.content,
         "documents": [{"doc_id": d.doc_id, "text": d.text} for d in (m._docs or [])]}
        for m in ctx.as_list()
    ]


def _dump_intrinsic_error(step, ctx, conv_json_idx, turn_idx):
    dump_dir = os.environ.get("INTRINSIC_DUMP_DIR", "")
    if not dump_dir:
        return
    try:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        idx  = len(list(dump_path.glob("intrinsic_request_*.json")))
        path = dump_path / f"intrinsic_request_{idx:04d}.json"
        path.write_text(json.dumps({
            "step": step, "conv_json_idx": conv_json_idx, "turn_idx": turn_idx,
            "messages": _serialize_ctx(ctx),
        }, indent=2))
        print(f"[intrinsic error] {step} failed (conv={conv_json_idx}, turn={turn_idx}) — dumped to {path}")
    except Exception as e:
        print(f"[intrinsic error] dump failed: {e}")


def _call_or_dump(step, ctx, conv_json_idx, turn_idx, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        _dump_intrinsic_error(step, ctx, conv_json_idx, turn_idx)
        raise


# ── Pipeline ──────────────────────────────────────────────────────────────────
def retrieve_documents(query, top_k=None):
    if top_k is None:
        top_k = TOP_K
    return collection.query(query_texts=[query], n_results=top_k)["documents"][0]


def run_timed_pipeline(query, history, backend, conv_json_idx=None, turn_idx=None, state_cb=None):
    """Run one pipeline turn. Returns timings, work metadata, and result."""
    timings = {}
    work    = {"history_turns": len(history), "query_len": len(query)}

    history_ctx = build_context(history)

    if state_cb: state_cb("harm")
    ctx = history_ctx.add(MelleaMessage("user", query))
    t0  = time.perf_counter()
    harm_score = _call_or_dump("guardian_harm", ctx, conv_json_idx, turn_idx,
                               guardian_check, ctx, backend, GUARDIAN_HARM_CRITERIA, target_role="user")
    timings["guardian_harm"] = time.perf_counter() - t0
    if harm_score >= 0.5:
        work["exit"] = "harm_blocked"
        return {"blocked": True, "timings": timings, "total": sum(timings.values()), "work": work}

    if state_cb: state_cb("rewrite")
    t0 = time.perf_counter()
    rewritten = _call_or_dump("query_rewrite", history_ctx, conv_json_idx, turn_idx,
                              rag.rewrite_question, query, history_ctx, backend)
    timings["query_rewrite"] = time.perf_counter() - t0
    work["rewritten_len"] = len(rewritten)

    if state_cb: state_cb("retrieve")
    t0 = time.perf_counter()
    documents = retrieve_documents(rewritten)
    timings["retrieval"] = time.perf_counter() - t0
    work["num_docs"]        = len(documents)
    work["total_doc_chars"] = sum(len(d) for d in documents)

    if state_cb: state_cb("answer?")
    t0 = time.perf_counter()
    ad_score = _call_or_dump("answerability", history_ctx, conv_json_idx, turn_idx,
                             rag.check_answerability, query, to_mellea_docs(documents), history_ctx, backend)
    timings["answerability"] = time.perf_counter() - t0
    if ad_score == "unanswerable":
        work["exit"] = "unanswerable"
        return {"unanswerable": True, "documents": documents,
                "timings": timings, "total": sum(timings.values()), "work": work}

    if state_cb: state_cb("clarify")
    t0 = time.perf_counter()
    clarification = _call_or_dump("clarification", history_ctx, conv_json_idx, turn_idx,
                                  rag.clarify_query, query, to_mellea_docs(documents), history_ctx, backend)
    timings["clarification"] = time.perf_counter() - t0
    work["clarification_len"] = len(clarification)
    if not clarification.strip().upper().startswith("CLEAR"):
        work["exit"] = "needs_clarification"
        return {"needs_clarification": True, "clarification": clarification,
                "documents": documents, "timings": timings, "total": sum(timings.values()), "work": work}

    if state_cb: state_cb("generate")
    gen_msg = MelleaMessage("user", rewritten + "\n\n" + GENERATION_INSTRUCTION,
                            documents=to_mellea_docs(documents) if documents else None)
    t0  = time.perf_counter()
    out, _ = _call_or_dump("generation", history_ctx.add(gen_msg), conv_json_idx, turn_idx,
                           mfuncs.act, gen_msg, history_ctx, backend,
                           model_options={ModelOption.TEMPERATURE: 0.0})
    timings["generation"] = time.perf_counter() - t0
    answer = str(out)
    work["answer_len"] = len(answer)
    work["exit"]       = "full_pipeline"

    return {"answer": answer, "documents": documents,
            "timings": timings, "total": sum(timings.values()), "work": work}


# ── Prometheus scraping ───────────────────────────────────────────────────────
def _parse_prom(text):
    gauges, hists = {}, {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            brace_end = line.index("}")
            name      = line[:line.index("{")]
            labels    = dict(re.findall(r'(\w+)="([^"]*)"', line[line.index("{")+1:brace_end]))
            val_str   = line[brace_end+1:].strip().split()[0]
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, val_str, labels = parts[0], parts[1], {}
        try:
            val = float(val_str)
        except ValueError:
            continue

        if name.endswith("_bucket"):
            base   = name[:-7]
            le_str = labels.get("le", "+Inf")
            le     = float("inf") if le_str == "+Inf" else float(le_str)
            h      = hists.setdefault(base, {"sum": 0.0, "count": 0.0, "bd": {}})
            h["bd"][le] = h["bd"].get(le, 0.0) + val
        elif name.endswith("_sum"):
            hists.setdefault(name[:-4], {"sum": 0.0, "count": 0.0, "bd": {}})["sum"] += val
        elif name.endswith("_count"):
            hists.setdefault(name[:-6], {"sum": 0.0, "count": 0.0, "bd": {}})["count"] += val
        else:
            gauges[name] = gauges.get(name, 0.0) + val

    for h in hists.values():
        h["buckets"] = sorted(h.pop("bd").items())
    return {"gauges": gauges, "hists": hists}


def _scrape(url):
    try:
        with _urllib_req.urlopen(url, timeout=2) as r:
            return _parse_prom(r.read().decode())
    except Exception:
        return None


def parse_kv_cache_from_log(log_path):
    try:
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", open(log_path).read())
        return int(m.group(1).replace(",", "")) if m else None
    except Exception:
        return None


_vllm_metrics_lock = threading.Lock()
_vllm_metrics      = {label: None for label in SERVERS}
_metrics_baseline  = {label: {} for label in SERVERS}


def _metrics_scraper_loop():
    while True:
        for label, cfg in SERVERS.items():
            parsed = _scrape(cfg["metrics_url"])
            if parsed is not None:
                with _vllm_metrics_lock:
                    _vllm_metrics[label] = parsed

        if _race_start is not None:
            snap = _get_metrics_snapshot()
            vllm_data = {}
            for label in SERVERS:
                parsed = snap.get(label)
                if parsed:
                    g = parsed["gauges"]
                    vllm_data[label] = {
                        "kv_hit":       round(_pc_hit_rate(parsed, label), 1),
                        "running":      int(g.get("vllm:num_requests_running", 0)),
                        "waiting":      int(g.get("vllm:num_requests_waiting", 0)),
                        "ttft_avg":     round(_race_avg("vllm:time_to_first_token_seconds", parsed, label), 3),
                        "e2e_avg":      round(_race_avg("vllm:e2e_request_latency_seconds",  parsed, label), 3),
                        "prompt_avg":   round(_race_avg("vllm:request_prompt_tokens",        parsed, label), 1),
                    }
            _emit_event("metrics", vllm=vllm_data)

        time.sleep(1.5)


def _get_metrics_snapshot():
    with _vllm_metrics_lock:
        return copy.deepcopy(_vllm_metrics)


def _record_metrics_baseline():
    snap = _get_metrics_snapshot()
    for label, parsed in snap.items():
        if parsed:
            _metrics_baseline[label] = {
                name: {"sum": h["sum"], "count": h["count"], "buckets_dict": dict(h["buckets"])}
                for name, h in parsed["hists"].items()
            }
            _metrics_baseline[label]["_counters"] = dict(parsed["gauges"])


def _race_counter(name, parsed, label):
    if not parsed:
        return 0.0
    return max(parsed["gauges"].get(name, 0.0)
               - _metrics_baseline.get(label, {}).get("_counters", {}).get(name, 0.0), 0.0)


def _pc_hit_rate(parsed, label):
    queries = _race_counter("vllm:prefix_cache_queries_total", parsed, label)
    hits    = _race_counter("vllm:prefix_cache_hits_total",    parsed, label)
    return hits / queries * 100.0 if queries > 0 else 0.0


# ── Shared race state ─────────────────────────────────────────────────────────
_state_lock   = threading.Lock()
_race_state   = {}
_server_finish = {}
_race_start   = None

_race_events  = []
_events_lock  = threading.Lock()
EVENTS_PATH   = str(_HERE / "race_events.json")


def _update_state(label, conv_idx, **kwargs):
    with _state_lock:
        key = (label, conv_idx)
        if key not in _race_state:
            _race_state[key] = {
                "turns_done": 0, "current_step": "", "current_turn": "",
                "done": False, "wall_time": 0.0,
            }
        _race_state[key].update(kwargs)


def _get_state_snapshot():
    with _state_lock:
        return dict(_race_state)


def _emit_event(event_type, **kwargs):
    """Record a timestamped race event (no-op before race_start is set)."""
    if _race_start is None:
        return
    evt = {"t": round(time.perf_counter() - _race_start, 4), "ev": event_type, **kwargs}
    with _events_lock:
        _race_events.append(evt)


# ── Conversation runner ───────────────────────────────────────────────────────
def run_conversation(run_idx, backend, label):
    queries          = get_queries_for_run(run_idx)
    conv_json_idx    = run_idx % len(CONVERSATIONS)
    history          = []
    query_results    = []
    conv_start       = time.perf_counter()
    _emit_event("conv_start", srv=label, conv=run_idx)

    try:
        for qi, (ql, query) in enumerate(queries):
            turn_label = ql.split(" — ")[0]

            def state_cb(step, _l=label, _i=run_idx, _t=turn_label):
                _update_state(_l, _i, current_turn=_t, current_step=step)
                _emit_event("step_start", srv=_l, conv=_i, turn=_t, step=step)

            state_cb("harm")
            r = run_timed_pipeline(query, history, backend,
                                   conv_json_idx=conv_json_idx, turn_idx=qi, state_cb=state_cb)

            docs = r.get("documents")
            if r.get("needs_clarification"):
                history.append({"role": "user",      "content": query,             "documents": docs})
                history.append({"role": "assistant",  "content": r["clarification"]})
            elif r.get("answer"):
                history.append({"role": "user",      "content": query,             "documents": docs})
                history.append({"role": "assistant",  "content": r["answer"]})

            query_results.append((ql, r))
            _update_state(label, run_idx, turns_done=qi + 1)
            _emit_event("turn_done", srv=label, conv=run_idx, turns_done=qi + 1)
    finally:
        conv_wall = time.perf_counter() - conv_start
        _update_state(label, run_idx, done=True, wall_time=conv_wall, current_step="done")
        _emit_event("conv_done", srv=label, conv=run_idx, wall_time=round(conv_wall, 4))

    return {
        "run_idx":       run_idx,
        "query_results": query_results,
        "conv_sum":      sum(r["total"] for _, r in query_results),
        "conv_wall":     conv_wall,
    }


# ── Rich live display ─────────────────────────────────────────────────────────

BAR_WIDTH   = 20
STEP_COLORS = {
    "harm": "bright_blue", "rewrite": "bright_cyan",
    "retrieve": "bright_yellow", "answer?": "bright_magenta", "clarify": "bright_magenta",
    "generate": "bright_red", "done": "bright_green",
}


def _bar(frac, width=18, color="bright_green"):
    frac   = max(0.0, min(frac, 1.0))
    filled = int(frac * width)
    txt    = Text()
    txt.append("█" * filled,          style=f"bold {color}")
    txt.append("░" * (width - filled), style="dim")
    return txt


def _race_avg(hist_name, parsed, label):
    if not parsed:
        return 0.0
    h      = parsed["hists"].get(hist_name, {})
    bl     = _metrics_baseline.get(label, {}).get(hist_name, {})
    d_sum  = h.get("sum",   0.0) - bl.get("sum",   0.0)
    d_cnt  = h.get("count", 0.0) - bl.get("count", 0.0)
    return d_sum / d_cnt if d_cnt > 0 else 0.0


def _build_vllm_table(labels, m_snap):
    t = Table(show_header=True, show_lines=False, pad_edge=False, expand=True, title="vLLM")
    for label in labels:
        t.add_column(label, ratio=1)

    def gauge_line(label_str, val, max_val, fmt, color_thresholds):
        frac  = val / max_val if max_val > 0 else 0.0
        color = color_thresholds[0][1]
        for pct, c in color_thresholds:
            if frac >= pct:
                color = c
        txt = Text(f"{label_str:<12s}")
        txt.append_text(_bar(frac, color=color))
        txt.append(f"  {fmt(val)}", style=color)
        return txt

    def cell_vals(label):
        parsed  = m_snap.get(label)
        g       = parsed["gauges"] if parsed else {}
        return (
            _pc_hit_rate(parsed, label),
            int(g.get("vllm:num_requests_running", 0)),
            int(g.get("vllm:num_requests_waiting", 0)),
            _race_avg("vllm:time_to_first_token_seconds", parsed, label),
            _race_avg("vllm:e2e_request_latency_seconds",  parsed, label),
            _race_avg("vllm:request_prompt_tokens",        parsed, label),
        )

    all_vals   = {label: cell_vals(label) for label in labels}
    ttft_max   = max(max(all_vals[l][3] for l in labels) * 1.5, 0.5)
    e2e_max    = max(max(all_vals[l][4] for l in labels) * 1.5, 2.0)
    prompt_max = max(max(all_vals[l][5] for l in labels) * 1.5, 200.0)
    lat_colors = [(0, "bright_cyan"), (0.5, "bright_yellow"), (0.8, "bright_red")]

    cells = []
    for label in labels:
        kv_hit, running, waiting, ttft, e2e, prompt = all_vals[label]
        txt = Text()
        txt.append_text(gauge_line("KV hit rate", kv_hit,  100.0,
                                   lambda v: f"{v:.0f}%",
                                   [(0, "bright_red"), (0.4, "bright_yellow"), (0.7, "bright_green")]))
        txt.append("\n")
        txt.append_text(gauge_line("Running",     running, CONCURRENCY_PER_SERVER,
                                   lambda v: f"{int(v):2d}/{CONCURRENCY_PER_SERVER}",
                                   [(0, "bright_green"), (0.5, "bright_yellow"), (0.9, "bright_red")]))
        txt.append("\n")
        txt.append_text(gauge_line("Waiting",     waiting, CONCURRENCY_PER_SERVER,
                                   lambda v: f"{int(v):2d}",
                                   [(0, "dim"), (0.01, "bright_yellow"), (0.1, "bright_red")]))
        txt.append("\n")
        txt.append_text(gauge_line("TTFT avg",    ttft,    ttft_max,
                                   lambda v: f"{v:.2f}s", lat_colors))
        txt.append("\n")
        txt.append_text(gauge_line("E2E avg",     e2e,     e2e_max,
                                   lambda v: f"{v:.1f}s",  lat_colors))
        txt.append("\n")
        txt.append_text(gauge_line("Prompt avg",  prompt,  prompt_max,
                                   lambda v: f"{int(v):,} tok", lat_colors))
        cells.append(txt)

    t.add_row(*cells)
    return t


def build_display(labels):
    snap    = _get_state_snapshot()
    elapsed = time.perf_counter() - _race_start if _race_start else 0.0

    table = Table(
        title=f"ALORA vs LORA — Live Race (govt) — {RUNS} conversations x 5 turns  [{elapsed:.1f}s]",
        show_header=True, show_lines=False, pad_edge=False, expand=True,
    )
    for label in labels:
        table.add_column(label, ratio=1)

    for i in range(RUNS):
        cells = []
        for label in labels:
            s      = snap.get((label, i), {"turns_done": 0, "current_step": "",
                                            "done": False, "wall_time": 0.0, "current_turn": ""})
            filled = int(s["turns_done"] / 5 * BAR_WIDTH)
            bar    = "█" * filled + "░" * (BAR_WIDTH - filled)

            if s["done"]:
                txt = Text(f"{i+1:2d} {bar} {s['wall_time']:.1f}s")
                txt.stylize("bold green")
            elif s["turns_done"] > 0 or s["current_step"]:
                step  = s.get("current_step", "")
                turn  = s.get("current_turn", "")
                color = STEP_COLORS.get(step, "white")
                txt   = Text(f"{i+1:2d} ")
                txt.append(bar[:filled],      style=f"bold {color}")
                txt.append(bar[filled:],      style="dim")
                txt.append(f" {turn}:{step}", style=color)
            else:
                txt = Text(f"{i+1:2d} {bar}", style="dim")
            cells.append(txt)
        table.add_row(*cells)

    table.add_section()
    footers = []
    for label in labels:
        done_count = sum(1 for i in range(RUNS) if snap.get((label, i), {}).get("done", False))
        if label in _server_finish:
            txt = Text(f"DONE  {done_count}/{RUNS}  {_server_finish[label]:.1f}s elapsed",
                       style="bold green")
        else:
            txt = Text(f"  {done_count}/{RUNS} done   {elapsed:.1f}s elapsed")
        footers.append(txt)
    table.add_row(*footers)

    m_snap = _get_metrics_snapshot()
    return Group(table, _build_vllm_table(labels, m_snap))


# ── Race runner ───────────────────────────────────────────────────────────────
def run_race(backends, labels, console, no_live=False):
    global _race_start
    all_conv_results   = {label: [] for label in labels}
    server_done_count  = {label: 0  for label in labels}
    _race_start        = time.perf_counter()

    pools = {label: ThreadPoolExecutor(max_workers=CONCURRENCY_PER_SERVER) for label in labels}
    with contextlib.ExitStack() as stack:
        for label in labels:
            stack.enter_context(pools[label])

        futures = {}
        for run_idx in range(RUNS):
            for label in labels:
                f = pools[label].submit(run_conversation, run_idx, backends[label], label)
                futures[f] = (label, run_idx)

        seen = set()
        use_live = not no_live

        def _drain_futures():
            for f in [f for f in futures if f.done() and id(f) not in seen]:
                seen.add(id(f))
                label, run_idx = futures[f]
                try:
                    all_conv_results[label].append(f.result())
                except Exception as e:
                    console.print(f"[red]ERROR: {label} conv {run_idx}: {e}[/red]")
                server_done_count[label] += 1
                if label not in _server_finish and server_done_count[label] == RUNS:
                    _server_finish[label] = time.perf_counter() - _race_start

        if use_live:
            with Live(build_display(labels), console=console, refresh_per_second=5) as live:
                while len(seen) < len(futures):
                    live.update(build_display(labels))
                    _drain_futures()
                    time.sleep(0.2)
                live.update(build_display(labels))
        else:
            # Notebook mode: print a simple counter instead of Rich Live
            _last_print = [0]
            total_turns = len(futures) * 5
            while len(seen) < len(futures):
                _drain_futures()
                done = sum(server_done_count.values())
                now = time.time()
                if now - _last_print[0] >= 2.0 or done == len(futures):
                    snap = _get_state_snapshot()
                    elapsed = time.perf_counter() - _race_start
                    parts = []
                    for l in labels:
                        turns = sum(snap.get((l, i), {}).get("turns_done", 0) for i in range(RUNS))
                        active = sum(1 for i in range(RUNS)
                                     if snap.get((l, i), {}).get("current_step") and
                                     not snap.get((l, i), {}).get("done"))
                        parts.append(f"{l}: {server_done_count[l]}/{RUNS} done, "
                                     f"{turns}/{RUNS*5} turns, {active} active")
                    print(f"  [{elapsed:5.1f}s]  {' | '.join(parts)}")
                    _last_print[0] = now
                time.sleep(0.2)

    return all_conv_results, time.perf_counter() - _race_start


# ── Stats + report ────────────────────────────────────────────────────────────
STEPS = ["guardian_harm", "query_rewrite", "retrieval", "answerability", "clarification", "generation"]


def collect_stats(all_conv_results, labels):
    server_results = {}
    for label in labels:
        all_conv_results[label].sort(key=lambda c: c["run_idx"])
        step_times = {s: [] for s in STEPS}
        conv_walls, conv_sums, all_work = [], [], []
        for cr in all_conv_results[label]:
            conv_walls.append(cr["conv_wall"])
            conv_sums.append(cr["conv_sum"])
            for _, r in cr["query_results"]:
                for step, t in r["timings"].items():
                    step_times[step].append(t)
                if "work" in r:
                    all_work.append(r["work"])
        server_results[label] = {
            "wall_total": max(conv_walls) if conv_walls else 0,
            "conv_walls": conv_walls,
            "conv_sums":  conv_sums,
            "step_times": step_times,
            "all_work":   all_work,
        }
    return server_results


def print_report(server_results, adapter_tech, all_conv_results, labels, race_wall):
    la, ll = labels[0], labels[1]
    ra, rl = server_results[la], server_results[ll]

    print("\n" + "=" * 120)
    print(f"SIMULTANEOUS RACE — {RUNS} conversations x 5 turns, {CONCURRENCY_PER_SERVER} concurrent per server")
    print("=" * 120)
    print(f"\n  {'':40s} {la:>14s}  {ll:>14s}  {'Diff':>10s}")

    def _row(label, va, vl, fmt="13.1f", unit="s"):
        diff = ((vl - va) / va * 100) if va > 0 else 0
        sign = "+" if diff > 0 else ""
        print(f"  {label:40s} {va:{fmt}}{unit}  {vl:{fmt}}{unit}  {sign}{diff:.1f}%")

    _row("Race wall-clock (both servers)",       race_wall,               race_wall)
    _row("Server wall-clock (last conv done)",   ra["wall_total"],        rl["wall_total"])
    if not ra["conv_walls"] or not rl["conv_walls"]:
        print("\n  No successful conversations — are the servers running?")
        return
    _row("Median conv wall-clock",               statistics.median(ra["conv_walls"]),
                                                 statistics.median(rl["conv_walls"]))
    _row("Max conv wall-clock (tail)",           max(ra["conv_walls"]),   max(rl["conv_walls"]))
    _row("Median conv CPU-time (sum of steps)",  statistics.median(ra["conv_sums"]),
                                                 statistics.median(rl["conv_sums"]))

    print(f"\n{'Step':20s} | {'Adapter':>15s} | {'n':>4s} | {'median':>8s} | {'mean':>8s} |"
          f" {'Adapter':>15s} | {'n':>4s} | {'median':>8s} | {'mean':>8s} | {'Diff(med)':>10s}")
    print("-" * 120)
    for step in STEPS:
        ta = [t * 1000 for t in ra["step_times"][step]]
        tl = [t * 1000 for t in rl["step_times"][step]]
        if not ta and not tl:
            continue
        med_a  = statistics.median(ta) if ta else 0
        med_l  = statistics.median(tl) if tl else 0
        mean_a = statistics.mean(ta)   if ta else 0
        mean_l = statistics.mean(tl)   if tl else 0
        diff   = ((med_l - med_a) / med_a * 100) if med_a > 0 else 0
        sign   = "+" if diff > 0 else ""
        print(f"{step:20s} | {adapter_tech[la].get(step,'-'):>15s} | {len(ta):4d} |"
              f" {med_a:7.1f}ms | {mean_a:7.1f}ms |"
              f" {adapter_tech[ll].get(step,'-'):>15s} | {len(tl):4d} |"
              f" {med_l:7.1f}ms | {mean_l:7.1f}ms | {sign}{diff:.1f}%")

    print(f"\n{'=' * 120}\nWORKLOAD COMPARISON\n{'=' * 120}")
    wa, wl       = ra["all_work"], rl["all_work"]
    total_queries = RUNS * 5
    exits_a = {w["exit"]: exits_a.get(w["exit"], 0) + 1 for w in wa for exits_a in [{}]}
    exits_l = {w["exit"]: exits_l.get(w["exit"], 0) + 1 for w in wl for exits_l in [{}]}

    # rebuild cleanly
    exits_a, exits_l = {}, {}
    for w in wa: exits_a[w["exit"]] = exits_a.get(w["exit"], 0) + 1
    for w in wl: exits_l[w["exit"]] = exits_l.get(w["exit"], 0) + 1
    exit_order = ["harm_blocked", "unanswerable", "needs_clarification", "full_pipeline"]
    all_exits  = sorted(set(exits_a) | set(exits_l),
                        key=lambda x: exit_order.index(x) if x in exit_order else 99)

    print(f"\n  Exit reason distribution ({total_queries} queries each):")
    print(f"  {'Exit reason':25s}  {'ALORA':>6s}  {'%':>5s}  {'LORA':>6s}  {'%':>5s}  {'Match':>6s}")
    for ex in all_exits:
        ca, cl = exits_a.get(ex, 0), exits_l.get(ex, 0)
        print(f"  {ex:25s}  {ca:6d}  {ca/total_queries*100:4.1f}%"
              f"  {cl:6d}  {cl/total_queries*100:4.1f}%  {'YES' if ca == cl else 'NO':>6s}")

    def _stat_row(label, key):
        va = [w[key] for w in wa if key in w]
        vl = [w[key] for w in wl if key in w]
        if not va or not vl:
            return
        ma, ml = statistics.median(va), statistics.median(vl)
        diff   = ((ml - ma) / ma * 100) if ma > 0 else 0
        sign   = "+" if diff > 0 else ""
        print(f"  {label:<26s} {la} median={ma:.0f} (n={len(va)})  "
              f"{ll} median={ml:.0f} (n={len(vl)})  ({sign}{diff:.1f}%)")

    print()
    total_a = sum(len(ra["step_times"][s]) for s in STEPS)
    total_l = sum(len(rl["step_times"][s]) for s in STEPS)
    diff    = ((total_l - total_a) / total_a * 100) if total_a > 0 else 0
    print(f"  Total intrinsic calls:    {la}={total_a}   {ll}={total_l}   "
          f"({'+'if diff>0 else ''}{diff:.1f}%)")
    _stat_row("Rewritten query length:", "rewritten_len")
    _stat_row("Answer output:",          "answer_len")
    _stat_row("Clarification output:",   "clarification_len")

    print(f"\n{'=' * 120}\nPer-conversation wall-clock (seconds):")
    print(f"  {'Conv':>4s}  {la+' wall':>11s}  {la+' sum':>10s}  {ll+' wall':>10s}  {ll+' sum':>10s}")
    by_idx_a = {c["run_idx"]: c for c in all_conv_results[la]}
    by_idx_l = {c["run_idx"]: c for c in all_conv_results[ll]}
    for i in range(RUNS):
        ca, cl = by_idx_a.get(i), by_idx_l.get(i)
        print(f"  {i+1:4d}  "
              f"{ca['conv_wall']:>10.1f}s  {ca['conv_sum']:>9.1f}s  " if ca else "       ERR        ERR  ",
              f"{cl['conv_wall']:>9.1f}s  {cl['conv_sum']:>9.1f}s"   if cl else "      ERR       ERR")


# ── Telemetry ─────────────────────────────────────────────────────────────────
TELEMETRY_PATH = str(_HERE / "race_results.json")


def write_telemetry(server_results, adapter_tech, all_conv_results, labels, race_wall, mode):
    """Write (or merge) race_results.json. In sequential mode each server run appends."""
    existing = {}
    if mode == "sequential" and Path(TELEMETRY_PATH).exists():
        try:
            existing = json.loads(Path(TELEMETRY_PATH).read_text())
        except Exception:
            pass

    # Only keep existing server data for servers that are actually configured
    servers_block = {k: v for k, v in existing.get("servers", {}).items() if k in SERVERS}
    for label in labels:
        sr = server_results[label]
        servers_block[label] = {
            "model":        SERVERS[label]["model"],
            "adapter_tech": adapter_tech[label],
            "wall_total":   sr["wall_total"],
            "conv_walls":   sr["conv_walls"],
            "conv_sums":    sr["conv_sums"],
            "step_times":   {s: sr["step_times"][s] for s in STEPS},
            "work":         sr["all_work"],
        }

    out = {
        "metadata": {
            "mode":        mode,
            "runs":        RUNS,
            "concurrency": CONCURRENCY_PER_SERVER,
            "timestamp":   datetime.datetime.utcnow().isoformat() + "Z",
            "race_wall":   race_wall,
        },
        "servers": servers_block,
    }
    Path(TELEMETRY_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nTelemetry written to {TELEMETRY_PATH}")

    # ── Events file ──────────────────────────────────────────────────────────
    # In sequential mode, keep events from servers that (a) are not in the
    # current run AND (b) already have results in the telemetry file.
    # This prevents stale/shipped events from bleeding into a fresh run.
    existing_events = []
    if mode == "sequential" and Path(EVENTS_PATH).exists():
        try:
            prev = json.loads(Path(EVENTS_PATH).read_text())
            current_srv_set = set(labels)
            keep_srvs = {s for s in servers_block if s not in current_srv_set}
            existing_events = [e for e in prev.get("events", [])
                                if e.get("srv") in keep_srvs
                                or (e.get("ev") == "metrics" and keep_srvs)]
        except Exception:
            pass

    with _events_lock:
        new_events = list(_race_events)

    all_events = existing_events + new_events
    all_events.sort(key=lambda e: e["t"])

    events_out = {
        "metadata":  out["metadata"],
        "race_wall": race_wall,
        "servers":   list(servers_block.keys()),
        "events":    all_events,
    }
    Path(EVENTS_PATH).write_text(json.dumps(events_out))
    print(f"Events  written to {EVENTS_PATH}")

    # Embed into race_live.html so it opens without an HTTP server.
    # If race_live.html doesn't exist yet, copy the template from sample_run/.
    live_path = _HERE / "race_live.html"
    if not live_path.exists():
        template = _HERE / "sample_run" / "race_live.html"
        if template.exists():
            shutil.copy2(template, live_path)
            print(f"Copied template from {template}")
    if live_path.exists():
        try:
            html_txt = live_path.read_text()
            new_line = (f"const RACE_EVENTS_EMBEDDED = {json.dumps(events_out)};"
                        " // <<RACE_EVENTS>>")
            html_txt = re.sub(
                r"const RACE_EVENTS_EMBEDDED = .*?; // <<RACE_EVENTS>>",
                new_line,
                html_txt,
                flags=re.DOTALL,
            )
            live_path.write_text(html_txt)
            print(f"Events embedded in {live_path.name}")
        except Exception as e:
            print(f"Warning: could not embed events in race_live.html: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global RUNS, CONCURRENCY_PER_SERVER, TOP_K

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["race", "sequential"], default="race",
                        help="race: both servers simultaneously (default); "
                             "sequential: one server at a time")
    parser.add_argument("--server", default=None,
                        help="In sequential mode, which server label to run "
                             "(e.g. 'ALORA (8111)'). Omit to run all servers.")
    parser.add_argument("--no-live", action="store_true",
                        help="Disable Rich Live display (use for Jupyter/Colab)")
    parser.add_argument("-n", "--runs", type=int, default=16,
                        help="Number of conversations to run (default: 16; H100 can handle 32)")
    parser.add_argument("-c", "--concurrency", type=int, default=8,
                        help="Max concurrent requests per server (default: 8; H100 can handle 24)")
    parser.add_argument("-k", "--top-k", type=int, default=10,
                        help="Number of documents to retrieve per query (default: 10; H100 can handle 15)")
    parser.add_argument("--alora-model", default=None,
                        help="Override ALORA model (HF repo or local path)")
    parser.add_argument("--lora-model", default=None,
                        help="Override LORA model (HF repo or local path)")
    args = parser.parse_args()
    RUNS = args.runs
    CONCURRENCY_PER_SERVER = args.concurrency
    TOP_K = args.top_k
    if args.alora_model is not None:
        SERVERS["ALORA (8111)"]["model"]  = args.alora_model
        SERVERS["ALORA (8111)"]["source"] = args.alora_model
    if args.lora_model is not None:
        SERVERS["LORA (8112)"]["model"]  = args.lora_model
        SERVERS["LORA (8112)"]["source"] = args.lora_model

    if args.mode == "sequential" and args.server and args.server not in SERVERS:
        parser.error(f"Unknown server '{args.server}'. Choose from: {list(SERVERS)}")

    labels = ([args.server] if args.mode == "sequential" and args.server
              else list(SERVERS.keys()))

    for label in labels:
        cfg = SERVERS[label]
        idx = _resolve_adapter_index(cfg["source"])
        adapters = idx["adapters"]
        aloras = sum(1 for a in adapters if a.get("technology") == "alora")
        loras  = sum(1 for a in adapters if a.get("technology") == "lora")
        print(f"{label}: {aloras} ALORAs, {loras} LORAs")

    for label, cfg in SERVERS.items():
        kv_tokens = parse_kv_cache_from_log(cfg.get("log_file", ""))
        kv_str    = f"{kv_tokens:,} tokens" if kv_tokens else "unknown"
        print(f"  {label}: KV cache budget {kv_str}")
    print()

    scraper = threading.Thread(target=_metrics_scraper_loop, daemon=True, name="metrics-scraper")
    scraper.start()
    time.sleep(2)
    _record_metrics_baseline()

    backends = {}
    for label in labels:
        cfg = SERVERS[label]
        discovered = _discover_vllm_model(cfg["base_url"])
        if discovered and discovered != cfg["model"]:
            print(f"  {label}: server reports model '{discovered}' (config had '{cfg['model']}')")
            cfg["model"] = discovered
        print(f"Connecting to {label} (model={cfg['model']})...")
        backends[label] = make_backend(cfg)
    print()

    adapter_tech = {}
    step_to_adapter = {
        "guardian_harm": "guardian-core",
        "query_rewrite": "query_rewrite",
        "answerability": "answerability",
        "clarification": "query_clarification",
    }
    for label in labels:
        idx = _resolve_adapter_index(SERVERS[label]["source"])
        tech_map          = {a["adapter_name"]: a["technology"] for a in idx["adapters"]}
        adapter_tech[label] = {step: tech_map.get(adapter, "base")
                               for step, adapter in step_to_adapter.items()}
        adapter_tech[label]["generation"] = "base"
        adapter_tech[label]["retrieval"]  = "local"

    no_live = _detect_notebook() or args.no_live
    console = Console()
    all_conv_results, race_wall  = run_race(backends, labels, console, no_live=no_live)
    server_results               = collect_stats(all_conv_results, labels)
    write_telemetry(server_results, adapter_tech, all_conv_results, labels, race_wall, args.mode)

    if len(labels) >= 2:
        print_report(server_results, adapter_tech, all_conv_results, labels, race_wall)


if __name__ == "__main__":
    main()
