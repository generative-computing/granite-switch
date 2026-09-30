# SPDX-License-Identifier: Apache-2.0
"""Train one demo adapter with PEFT: loss on the single output token only.

Adapted from ``peft/examples/alora_finetuning/alora_finetuning.py``. The prompt
is assembled exactly like the demo's inference prompt (:class:`policy.PromptBuilder`,
base-model form): system prompt, the row's windowed 5 Hz history (rebuilt from
the match's history stream), the current state, the assistant header. The loss
is cross-entropy at the last position over the adapter's output vocabulary.
Labels are the teacher's whole distribution when it has one (``soft``, from the
RL teacher; this is KL to the teacher up to a constant), else one-hot.

Adapters and their labels (rows from ``collect.py``):

* ``fighter`` / ``cautious`` / ``collector``: the teacher's move, every tic.
* ``arms``: the teacher's weapon slot, planner tics only.
* ``critic``: ``low`` / ``mid`` / ``high``, the outcome of the next second.
* ``probe``: where the last enemy in the history was (history-only question;
  used for the aLoRA-vs-LoRA comparison, not composed into the demo).
* ``router``: rows from ``router_data.py`` (instruction, label); no history.
* ``narrator``: a whole spoken line, the one generating adapter. Rows are the
  lines ``narrate_ivr.py`` wrote that passed every check, each in its match's
  conversation (chat layout, :func:`load_narration_rows`); the loss is
  cross-entropy on the line's tokens and the end of the turn.

``--kind lora`` trains a plain LoRA on the same data and prompts: the baseline
the aLoRA is compared against. ``--kind sr`` trains a Shadow Residual adapter
(github.ibm.com/generative-computing/shadow-residual, its ``src`` on
``PYTHONPATH``): a second, adapter stream through every layer that reads the
frozen base stream's K/V, with LoRA on Q, O and the MLP (never K/V) plus a
per-layer low-rank cross-stream from the base. The adapter never writes K/V, so
with a one-token answer only the last position's adapter stream matters, and
training it always-active is exactly inference with the control token on the
last prompt token. Settings follow IBM's shipped aLoRAs: rank 32 on every
linear layer (for SR, every linear layer but K and V).

::

    python train_alora.py --adapter fighter --data data/d0 data/d1 \
        --base /path/granite-4.1-3b --out runs/d1/fighter
    python train_alora.py --adapter router --data data/router/train.jsonl \
        --eval-data data/router/heldout.jsonl --epochs 4 --out runs/router/router

Writes the PEFT adapter to ``--out``, ``metrics.json`` and
``heldout_preds.jsonl`` (history ids, state and the PEFT argmax per held-out
row), which ``build_model.py verify`` compares against the composed checkpoint.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import math
import os
import random
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import TIC_HZ
from history import PROBE_WORDS, History, said_entry
from policy import (
    _EOR,
    _EOT,
    ARMS,
    CRITIC,
    DANGER_LEVELS,
    LAYOUTS,
    NARRATOR,
    OUTPUTS,
    ROUTER,
    ROUTER_SYSTEM_PROMPT,
    PromptBuilder,
    alora_invocation_ids,
    output_token_ids,
    route_token_ids,
    spoken_entry,
    system_prompt,
    talk_extra,
)
from talk import brief

PROBE = "probe"
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SR_CROSS = "cross_stream"
VOCAB = {**OUTPUTS, PROBE: PROBE_WORDS}


# ── Data ───────────────────────────────────────────────────────────────────────
class Stream:
    """One match's history entries, tokenized once; ``ids(n)`` is the windowed
    history after the first ``n`` entries (the same rule as History.append)."""

    def __init__(self, entries: list[str], tok):
        h = History(tok)
        self.starts, self.entry_ids, self.cum = [0], [], [0]
        for e in entries:
            h.append(e)
            self.starts.append(len(self.entry_ids) + 1 - len(h.entries))
            self.entry_ids.append(h.entry_ids[-1])
            self.cum.append(self.cum[-1] + len(h.entry_ids[-1]))

    def ids(self, n: int) -> list[int]:
        return [i for e in self.entry_ids[self.starts[n] : n] for i in e]

    def length(self, n: int) -> int:
        return self.cum[n] - self.cum[self.starts[n]]


def row_target(r: dict, adapter: str) -> dict[str, float] | None:
    """The adapter's label distribution for one collect.py row, or None to skip."""
    if adapter == ARMS:
        if r.get("weapon") is None:
            return None
        return r.get("weapon_soft") or {r["weapon"]: 1.0}
    if adapter == CRITIC:
        return {r["critic"]: 1.0}
    if adapter == PROBE:
        return {r["probe"]: 1.0} if r.get("probe") else None
    return r.get("soft") or {r["expert"]: 1.0}


@dataclass
class TalkAug:
    """Spoken lines to insert into training histories (see :func:`with_talk`)."""

    lines: list[str]  # the player's own
    user_lines: list[str]  # the watcher's words; may be empty
    rate: float  # lines per history entry in a talking match (0.07: ~one per 3 s)
    frac: float  # share of matches that talk
    seed: int = 0
    layout: str = "log"  # "log": me:/user: entries; "chat": real turns


def with_talk(
    entries: list[str], aug: TalkAug, key, states: list[tuple[int, str]] = ()
) -> tuple[list[str], list[int]]:
    """A match's history with spoken lines inserted between entries, as the live
    game appends them. Log layout: ``me:``/``user:`` entries
    (``history.said_entry``). Chat layout: closed turns (``policy.spoken_entry``:
    a brief, sometimes the watcher's words, the state at that tick from
    ``states``, then the line as an assistant turn). Returns the new entries and,
    for every original prefix length n, the new prefix length, so a row keeps
    its moment. Seeded per match: the same match talks the same way every run."""
    rng = random.Random(zlib.crc32(repr((key, aug.seed)).encode()))
    if rng.random() >= aug.frac:
        return entries, list(range(len(entries) + 1))
    rate = aug.rate * rng.uniform(0.5, 1.5)
    ticks = [t for t, _ in states]
    out, where = [], [0]
    for i, e in enumerate(entries):
        out.append(e)
        if rng.random() < rate:
            tick = round(float(e.split()[0][1:]) * TIC_HZ)
            user = bool(aug.user_lines) and rng.random() < (
                0.2 if aug.layout == "chat" else 0.15
            )
            if aug.layout == "chat":
                j = bisect.bisect_right(ticks, tick) - 1
                if j < 0 or tick - ticks[j] > TIC_HZ:
                    where.append(len(out))
                    continue  # no recorded state near this tick
                state = states[j][1]
                extra = brief(entries[: i + 1], state) + "\n"
                if user:
                    extra += f"Player: {rng.choice(aug.user_lines)}\n"
                out.append(spoken_entry(state, rng.choice(aug.lines), extra))
            else:
                pool = aug.user_lines if user else aug.lines
                out.append(said_entry(tick, "user" if user else "me", rng.choice(pool)))
        where.append(len(out))
    return out, where


def load_game_rows(
    dirs: list[Path],
    adapter: str,
    styles: list[str],
    tok,
    every: int,
    talk: TalkAug | None = None,
):
    """Rows as (stream, hist_n, state, target, episode key); history streams are
    tokenized once per match. With ``talk``, histories carry spoken lines."""
    out, streams, where = [], {}, {}
    for d in dirs:
        for style in styles:
            hp, rp = d / f"{style}_history.jsonl", d / f"{style}.jsonl"
            if not rp.exists():
                continue
            rows = [json.loads(line) for line in open(rp)]
            states: dict = {}  # per match: (tick, state) for the chat layout's turns
            if talk is not None and talk.layout == "chat":
                for r in rows:
                    states.setdefault(r["ep"], []).append((r["t"], r["state"]))
                for v in states.values():
                    v.sort()
            for line in open(hp):
                h = json.loads(line)
                key = (str(d), style, h["ep"])
                entries = h["entries"]
                if talk is not None:
                    entries, where[key] = with_talk(
                        entries, talk, key, states.get(h["ep"], [])
                    )
                streams[key] = Stream(entries, tok)
            for r in rows:
                if r["t"] % every:
                    continue
                target = row_target(r, adapter)
                if target is None:
                    continue
                key = (str(d), style, r["ep"])
                n = where[key][r["hist_n"]] if key in where else r["hist_n"]
                out.append((streams[key], n, r["state"], target, key))
    return out


def load_router_rows(paths: list[Path]):
    out = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            out.append(
                (None, 0, r["text"], {r["label"]: 1.0}, (str(p), r.get("ep", 0)))
            )
    return out


def load_narration_rows(paths: list[Path], moments: Path, tok) -> list:
    """Narrator rows as (stream, n, (state, closing extra), target ids, match
    key, meta), one per written line that passed every check.

    A match's conversation is rebuilt as the chat layout's live game builds it:
    its log entries, and at each earlier speaking moment the line written there
    (passing or not: it is what the writer had said, and saw) as a closed turn,
    ``policy.spoken_entry``, under the same 10 s window. The prompt is that
    history up to the moment, closed by the moment's brief, what the person
    watching said if anything (``partner_ivr.py`` rows' ``player``) and the
    state (``PromptBuilder.turn_ids``); the target is the line and the end of
    the turn. ``moments``: ``talk.py moments`` output, for each moment's state,
    brief and last log lines."""
    ctx = {}
    for x in open(moments):
        m = json.loads(x)
        ctx[(m["data"], m["ep"])] = {mm["t"]: mm for mm in m["moments"]}
    by_match: dict = {}
    for p in paths:
        for x in open(p):
            r = json.loads(x)
            by_match.setdefault((r["data"], r["style"], r["ep"]), []).append(r)
    entries_of = {}
    for data, style in {k[:2] for k in by_match}:
        for x in open(Path(data) / f"{style}_history.jsonl"):
            h = json.loads(x)
            entries_of[(data, style, h["ep"])] = h["entries"]
    eot = tok.encode(_EOT, add_special_tokens=False)
    out = []
    for key, rows in by_match.items():
        rows.sort(key=lambda r: r["t"])
        entries, conv, at, i = entries_of[key], [], [], 0
        for r in rows:
            m = ctx[(key[0], key[2])][r["t"]]
            conv += entries[i : r["hist_n"]]
            i = r["hist_n"]
            extra = talk_extra(m["brief"], r.get("player"))
            at.append((len(conv), m, extra, r))
            if r["line"]:
                conv.append(spoken_entry(m["state"], r["line"], extra))
        stream = Stream(conv, tok)
        for n, m, extra, r in at:
            if not (r["ok"] and r["line"]):
                continue
            line = " ".join(r["line"].split())
            meta = {
                "data": key[0],
                "ep": key[2],
                "t": r["t"],
                "brief": m["brief"],
                "recent": m["recent"],
                "prev": r["prev"],
                "player": r.get("player"),
                "line": line,
            }
            target = tok.encode(line, add_special_tokens=False) + eot
            out.append((stream, n, (m["state"], extra), target, key, meta))
    return out


def split_by_episode(rows: list, val_frac: float, seed: int) -> tuple[list, list]:
    """Hold out whole matches: consecutive tics are near-duplicates."""
    eps = sorted({r[4] for r in rows})
    random.Random(seed).shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac)) if len(eps) > 1 else 0
    val_eps = set(eps[:n_val])
    return [r for r in rows if r[4] not in val_eps], [
        r for r in rows if r[4] in val_eps
    ]


def batches(items: list, size: int, shuffle: bool, seed: int, lengths=None):
    """Batches of ``size``. With ``lengths``, rows are bucketed: shuffled, cut
    into chunks of 50 batches, sorted by length within a chunk, and the batches
    shuffled, so a batch holds prompts of similar length (less padding)."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    if shuffle and lengths is not None:
        chunk = size * 50
        groups = []
        for c in range(0, len(idx), chunk):
            part = sorted(idx[c : c + chunk], key=lengths.__getitem__)
            groups += [part[i : i + size] for i in range(0, len(part), size)]
        rng.shuffle(groups)
        for g in groups:
            yield [items[j] for j in g]
        return
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i : i + size]]


# ── Model helpers ──────────────────────────────────────────────────────────────
def collate(prompts: list[list[int]], targets: list[list[float]], pad_id: int, device):
    """Left-pad so every prompt ends at the last position; explicit position ids
    keep RoPE identical to the unpadded prompt the engine sees at inference."""
    import torch

    n = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), n), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, n - len(p) :] = torch.tensor(p)
        mask[i, n - len(p) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    tgt = torch.tensor(targets, dtype=torch.float32)
    return input_ids.to(device), mask.to(device), pos.to(device), tgt.to(device)


def last_logits(model, input_ids, mask, pos, allowed):
    out = model(
        input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=1
    )
    return out.logits[:, -1, :][:, allowed].float()


def collate_lines(prompts: list[list[int]], targets: list[list[int]], pad_id, device):
    """Prompt + target, left-padded so every target ends at the last position.
    ``labels`` covers the last ``max target length`` positions, -100 where a
    shorter target's prompt shows through."""
    import torch

    seqs = [p + t for p, t in zip(prompts, targets)]
    n, k = max(len(s) for s in seqs), max(len(t) for t in targets)
    input_ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    labels = torch.full((len(seqs), k), -100, dtype=torch.long)
    for i, (s, t) in enumerate(zip(seqs, targets)):
        input_ids[i, n - len(s) :] = torch.tensor(s)
        mask[i, n - len(s) :] = 1
        labels[i, k - len(t) :] = torch.tensor(t)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return input_ids.to(device), mask.to(device), pos.to(device), labels.to(device)


def line_logits(model, input_ids, mask, pos, k: int):
    """Full-vocabulary logits predicting the last ``k`` tokens (only those
    positions are projected to the vocabulary)."""
    out = model(
        input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=k + 1
    )
    return out.logits[:, :-1, :].float()


def generate_lines(
    model, tok, prompts, pad_id, device, *, adapter=True, temperature=0.8, batch=8
) -> list[str]:
    """One sampled line per prompt, as the demo samples talk (temperature
    only, at most 32 new tokens, up to the end of the turn or a newline);
    ``adapter=False`` gives the base model's line for the same prompt."""
    import torch

    eot = tok.convert_tokens_to_ids(_EOT)
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        n = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, n - len(p) :] = torch.tensor(p)
            mask[j, n - len(p) :] = 1
        off = contextlib.nullcontext() if adapter else model.disable_adapter()
        with torch.no_grad(), off:
            g = model.generate(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                max_new_tokens=32,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_k=0,
                top_p=1.0,
                eos_token_id=eot,
                pad_token_id=pad_id,
            )
        for row in g[:, n:]:
            text = tok.decode(row, skip_special_tokens=True)
            out.append(text.strip().split("\n")[0].strip().strip('"'))
    return out


def train_window(curve: list[dict], steps: int) -> str:
    """Mean training loss over the last ``steps`` steps (one pass: each batch is
    fresh, so this is the training distribution's own held-out loss)."""
    last = [c["loss"] for c in curve[-max(1, steps // 50) :]]
    return f", train loss {sum(last) / len(last):.4f}" if last else ""


def auc(scores: list[float], positives: list[bool]) -> float | None:
    """Rank AUC (probability a random positive outscores a random negative)."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # average ranks over ties
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rank_sum += r * sum(y for _, y in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, choices=sorted([*VOCAB, NARRATOR]))
    ap.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="collect.py dirs (router: jsonl; narrator: narrate_ivr.py jsonl)",
    )
    ap.add_argument(
        "--moments",
        type=Path,
        help="Narrator: the talk.py moments file the lines were written for",
    )
    ap.add_argument(
        "--gen-n",
        type=int,
        default=0,
        help="Narrator: held-out moments to sample a line for after training, "
        "from the adapter and from the base model (heldout_gen.jsonl)",
    )
    ap.add_argument("--eval-data", type=Path, nargs="*", help="Explicit held-out set")
    ap.add_argument(
        "--styles",
        nargs="+",
        help="Row files to read (default: the adapter's own, "
        "or every style for arms/critic/probe)",
    )
    ap.add_argument("--kind", choices=("alora", "lora", "sr"), default="alora")
    ap.add_argument(
        "--cross-rank", type=int, default=32, help="SR: rank of the cross-stream"
    )
    ap.add_argument(
        "--init",
        type=Path,
        help="Warm-start from this adapter (e.g. the previous DAgger round's); "
        "its config (kind, rank, targets) is used as saved",
    )
    ap.add_argument("--base", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=32)
    # alpha = rank (scale 1) and lr 1e-4: at alpha 64 / lr 2e-4 a rank-32
    # all-linear adapter collapsed to the label marginal mid-run.
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8, help="Micro-batch (memory)")
    ap.add_argument("--every", type=int, default=3, help="Keep every k-th tic")
    ap.add_argument("--max-examples", type=int, default=150_000)
    ap.add_argument("--max-heldout", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--eval-n", type=int, default=1000, help="Held-out rows per mid-run evaluation"
    )
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--keep-best",
        action="store_true",
        help="Also save the adapter with the lowest held-out soft CE (to <out>/best)",
    )
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--talk-lines",
        type=Path,
        help="Spoken lines (talk.py output, one JSON with 'line' per row) to "
        "insert into training histories as 'me:' entries",
    )
    ap.add_argument(
        "--talk-user-lines",
        type=Path,
        help="Instructions (JSON rows with 'text', e.g. the router data) for "
        "'user:' entries",
    )
    ap.add_argument(
        "--layout",
        default="log",
        choices=sorted(LAYOUTS),
        help="Game prompt layout: log (one user turn) or chat (spoken lines are "
        "assistant turns; the engine must use the same)",
    )
    ap.add_argument("--talk-rate", type=float, default=0.07, help="Lines per entry")
    ap.add_argument("--talk-frac", type=float, default=0.7, help="Matches that talk")
    ap.add_argument(
        "--talk-eval", action="store_true", help="Held-out histories talk too"
    )
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    # One process per GPU under torchrun; each takes a slice of every batch.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp = world > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cuda")
    main_rank = rank == 0
    tok = AutoTokenizer.from_pretrained(args.base)
    a = args.adapter
    lines = a == NARRATOR
    if a == ROUTER:
        label_ids = route_token_ids(tok)
        pb = PromptBuilder(tok, ROUTER_SYSTEM_PROMPT, align=False)
        rows = load_router_rows(args.data)
        val_rows = load_router_rows(args.eval_data) if args.eval_data else None
    elif lines:
        if args.moments is None:
            raise SystemExit("--moments is required for the narrator")
        label_ids = {}
        pb = PromptBuilder(tok, system_prompt("chat"))
        rows = load_narration_rows(args.data, args.moments, tok)
        val_rows = (
            load_narration_rows(args.eval_data, args.moments, tok)
            if args.eval_data
            else None
        )
    else:
        label_ids = output_token_ids(tok, VOCAB[a])
        pb = PromptBuilder(tok, system_prompt(args.layout))
        styles = args.styles or (
            [a]
            if a in OUTPUTS and OUTPUTS[a] == OUTPUTS["fighter"]
            else ["fighter", "cautious", "collector"]
        )
        talk = None
        if args.talk_lines:
            talk = TalkAug(
                lines=[json.loads(x)["line"] for x in open(args.talk_lines)],
                user_lines=[json.loads(x)["text"] for x in open(args.talk_user_lines)]
                if args.talk_user_lines
                else [],
                rate=args.talk_rate,
                frac=args.talk_frac,
                seed=args.seed,
                layout=args.layout,
            )
        rows = load_game_rows(args.data, a, styles, tok, args.every, talk)
        val_rows = (
            load_game_rows(
                args.eval_data,
                a,
                styles,
                tok,
                args.every,
                talk if args.talk_eval else None,
            )
            if args.eval_data
            else None
        )
    classes = list(label_ids)
    allowed = torch.tensor(list(label_ids.values()), device=device)
    if val_rows is None:
        train_rows, val_rows = split_by_episode(rows, args.val_frac, args.seed)
    else:
        train_rows = rows
    rng = random.Random(args.seed)
    if len(train_rows) > args.max_examples:
        train_rows = rng.sample(train_rows, args.max_examples)
    # Random order even when nothing is dropped: rows arrive grouped by match, so
    # the mid-run slice (--eval-n) would otherwise cover only the first few matches.
    val_rows = rng.sample(val_rows, min(len(val_rows), args.max_heldout))

    def prompt(r) -> list[int]:
        stream, n, state = r[0], r[1], r[2]
        if stream is None:
            return pb.ids(state, None)
        if lines:
            return pb.turn_ids(stream.ids(n), *state)
        return pb.game_ids(stream.ids(n), state, [None])[0]

    def target(r) -> list[float]:
        t = [r[3].get(c, 0.0) for c in classes]
        s = sum(t)
        return [x / s for x in t]

    lens = [len(prompt(r)) for r in train_rows[:500]]
    if lines:
        tl = [len(r[3]) for r in train_rows]
        mix = f"line {min(tl)}-{max(tl)} tokens (mean {sum(tl) / len(tl):.1f})"
    else:
        mix = f"label mix {Counter(max(r[3], key=r[3].get) for r in train_rows).most_common(6)}"
    print(
        f"{a} ({args.kind}): {len(train_rows)} train / {len(val_rows)} held-out rows, "
        f"prompt {min(lens)}-{max(lens)} tokens (first 500); {mix}",
        flush=True,
    )

    if args.kind == "sr":
        if args.init is not None:
            raise SystemExit("--init is not supported for sr yet")
        from shadow_residual.training.factory import get_shadow_residual_peft_model

        # K/V always come from the base stream: no LoRA can land there.
        targets = [t for t in args.targets.split(",") if t not in ("k_proj", "v_proj")]
        cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=[*targets, SR_CROSS],
            rank_pattern={SR_CROSS: args.cross_rank},
            bias="none",
        )
        model = get_shadow_residual_peft_model(
            args.base, cfg, torch_dtype=torch.bfloat16
        ).to(device)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16
        ).to(device)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if args.init is not None:
            from peft import PeftModel

            # Onto this rank's GPU: the default ("cuda") is device 0, which
            # the other ranks cannot open in exclusive-process mode.
            model = PeftModel.from_pretrained(
                model, str(args.init), is_trainable=True, torch_device=str(device)
            )
        else:
            extra = (
                {"alora_invocation_tokens": alora_invocation_ids(tok)}
                if args.kind == "alora"
                else {}
            )
            cfg = LoraConfig(
                task_type="CAUSAL_LM",
                r=args.rank,
                lora_alpha=args.alpha,
                lora_dropout=args.dropout,
                target_modules=args.targets.split(","),
                bias="none",
                **extra,
            )
            model = get_peft_model(model, cfg)
    raw = model  # the PEFT model; ``model`` is DDP-wrapped when distributed
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index]
        )
    if main_rank:
        raw.print_trainable_parameters()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def evaluate_lines(items: list) -> tuple[dict, list[tuple[list, list]]]:
        """Teacher-forced: per-token CE and argmax accuracy over the held-out
        lines' tokens (``soft_ce`` / ``acc``, the keys the loop reads), and each
        line's argmax ids with their top-1 minus top-2 probability (what
        ``build_model.py verify`` compares)."""
        import torch.nn.functional as F

        raw.eval()
        ce_sum, n_tok, hit, first, preds = 0.0, 0, 0, 0, []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(items, args.micro, False, 0):
                ids, m, p, lab = collate_lines(
                    [prompt(r) for r in b], [r[3] for r in b], pad_id, device
                )
                logits = line_logits(raw, ids, m, p, lab.shape[1])
                ce = F.cross_entropy(
                    logits.transpose(1, 2), lab, ignore_index=-100, reduction="none"
                )
                am, ok = logits.argmax(-1), lab != -100
                top2 = torch.softmax(logits, -1).topk(2, -1).values
                margin = top2[..., 0] - top2[..., 1]  # near-ties flip between kernels
                ce_sum += ce[ok].sum().item()
                n_tok += int(ok.sum())
                hit += int((am == lab)[ok].sum())
                for i, r in enumerate(b):
                    k = len(r[3])
                    preds.append((am[i, -k:].tolist(), margin[i, -k:].tolist()))
                    first += int(preds[-1][0][0] == r[3][0])
        raw.train()
        ce = ce_sum / max(1, n_tok)
        return {
            "n": len(items),
            "tokens": n_tok,
            "acc": round(hit / max(1, n_tok), 4),
            "soft_ce": round(ce, 4),
            "ppl": round(math.exp(ce), 3),
            "first_token_acc": round(first / max(1, len(items)), 4),
        }, preds

    def evaluate(items: list) -> tuple[dict, list[list[float]]]:
        if lines:
            return evaluate_lines(items)
        # On the unwrapped model (rank 0 only): DDP's forward would wait for
        # the other ranks.
        raw.eval()
        probs: list[list[float]] = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(items, args.micro, False, 0):
                ids, m, p, _ = collate(
                    [prompt(r) for r in b], [target(r) for r in b], pad_id, device
                )
                probs += torch.softmax(
                    last_logits(raw, ids, m, p, allowed), -1
                ).tolist()
        raw.train()
        tg = [target(r) for r in items]
        gold = [max(range(len(classes)), key=t.__getitem__) for t in tg]
        pred = [max(range(len(classes)), key=q.__getitem__) for q in probs]
        ce = sum(
            -sum(t_i * math.log(max(q_i, 1e-9)) for t_i, q_i in zip(t, q))
            for t, q in zip(tg, probs)
        ) / max(1, len(items))
        maj = Counter(gold).most_common(1)[0]
        ev = {
            "n": len(items),
            "acc": round(
                sum(g == q for g, q in zip(gold, pred)) / max(1, len(items)), 4
            ),
            "soft_ce": round(ce, 4),
            "majority_class": classes[maj[0]],
            "majority_baseline": round(maj[1] / max(1, len(items)), 4),
            "per_class": {
                c: {
                    "n": sum(g == ci for g in gold),
                    "acc": round(
                        sum(g == q == ci for g, q in zip(gold, pred))
                        / max(1, sum(g == ci for g in gold)),
                        4,
                    ),
                }
                for ci, c in enumerate(classes)
                if any(g == ci for g in gold)
            },
        }
        if a == CRITIC:
            hi, lo = DANGER_LEVELS.index("high"), DANGER_LEVELS.index("low")
            ev["auc_damage_1s"] = auc(
                [1 - q[classes.index(DANGER_LEVELS[lo])] for q in probs],
                [g != classes.index("low") for g in gold],
            )
            ev["auc_high"] = auc(
                [q[classes.index(DANGER_LEVELS[hi])] for q in probs],
                [g == classes.index("high") for g in gold],
            )
        return ev, probs

    steps_per_epoch = math.ceil(len(train_rows) / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    step, t0, run_loss, history = 0, time.time(), 0.0, []
    train_curve: list[dict] = []  # training loss, every 50 steps
    best = {"soft_ce": math.inf, "step": None}
    lengths = [
        (r[0].length(r[1]) if r[0] is not None else 0)
        + len(r[2] if isinstance(r[2], str) else "".join(r[2])) // 3
        for r in train_rows
    ]
    model.train()
    epoch = 0
    while step < total:
        for b in batches(train_rows, args.batch, True, args.seed + epoch, lengths):
            share = b[rank::world]  # this rank's slice of the global batch
            if len(b) < world:
                continue  # a last short batch cannot feed every rank
            micros = [
                share[k : k + args.micro] for k in range(0, len(share), args.micro)
            ]
            share_tokens = sum(len(r[3]) for r in share) if lines else 0
            for k, mb in enumerate(micros):
                # Sync gradients once per step, on the last micro-batch.
                sync = (
                    model.no_sync()
                    if ddp and k < len(micros) - 1
                    else contextlib.nullcontext()
                )
                if lines:
                    ids, m, p, lab = collate_lines(
                        [prompt(r) for r in mb], [r[3] for r in mb], pad_id, device
                    )
                    with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = line_logits(model, ids, m, p, lab.shape[1])
                        # Token mean over this rank's share; DDP averages the ranks.
                        loss = (
                            torch.nn.functional.cross_entropy(
                                logits.transpose(1, 2),
                                lab,
                                ignore_index=-100,
                                reduction="sum",
                            )
                            / share_tokens
                        )
                        loss.backward()
                    run_loss += loss.item()
                    continue
                ids, m, p, tgt = collate(
                    [prompt(r) for r in mb], [target(r) for r in mb], pad_id, device
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    logp = torch.log_softmax(last_logits(model, ids, m, p, allowed), -1)
                    # Mean over this rank's share; DDP averages the ranks.
                    loss = -(tgt * logp).sum(-1).mean() * len(mb) / len(share)
                    loss.backward()
                run_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 and main_rank:
                print(
                    f"step {step}/{total} loss {run_loss / 50:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                    flush=True,
                )
                train_curve.append({"step": step, "loss": round(run_loss / 50, 5)})
                run_loss = 0.0
            if step % args.eval_every == 0 and val_rows and main_rank:
                ev, _ = evaluate(val_rows[: args.eval_n])
                history.append({"step": step, **{k: ev[k] for k in ("acc", "soft_ce")}})
                if args.keep_best and ev["soft_ce"] < best["soft_ce"]:
                    best = {"soft_ce": ev["soft_ce"], "acc": ev["acc"], "step": step}
                    raw.save_pretrained(str(args.out / "best"))
                ref = (
                    f"ppl {ev['ppl']:.3f}"
                    if lines
                    else f"majority {ev['majority_baseline']:.4f}"
                )
                print(
                    f"  held-out acc {ev['acc']:.4f} soft CE {ev['soft_ce']:.4f} "
                    f"({ref}) at step {step}"
                    + train_window(train_curve, args.eval_every),
                    flush=True,
                )
            if step >= total:
                break
        epoch += 1

    if ddp:
        torch.distributed.barrier()
    if not main_rank:
        torch.distributed.destroy_process_group()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    raw.save_pretrained(str(args.out))
    if args.kind == "sr":
        # What the Granite Switch composer reads to place an SR control token:
        # it replaces this single token, the last one of the prompt.
        path = args.out / "adapter_config.json"
        acfg = json.loads(path.read_text())
        acfg["last_context_token"] = _EOR
        acfg["last_context_token_id"] = tok.encode(_EOR, add_special_tokens=False)[0]
        acfg["share_moe_routing"] = True  # no-op on a dense base; recorded by SR
        path.write_text(json.dumps(acfg, indent=2))
    if lines and best["step"] is not None:
        # A line model can overfit a few thousand lines: the narrator is the
        # checkpoint with the lowest held-out CE, not the last one.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        best_w = load_file(str(args.out / "best" / "adapter_model.safetensors"))
        set_peft_model_state_dict(raw, best_w)
        raw.save_pretrained(str(args.out))
    ev, probs = evaluate(val_rows)
    with open(args.out / "heldout_preds.jsonl", "w") as f:
        for r, q in zip(val_rows, probs):
            stream, n, state = r[0], r[1], r[2]
            if lines:
                row = {
                    "history_ids": stream.ids(n),
                    "state": state[0],
                    "extra": state[1],
                    "line": r[5]["line"],
                    "target_ids": r[3],
                    "peft_ids": q[0],
                    "peft_margin": [round(x, 4) for x in q[1]],
                }
            else:
                row = {
                    "history_ids": stream.ids(n) if stream is not None else None,
                    "state": state,
                    "label": max(r[3], key=r[3].get),
                    "peft": classes[max(range(len(q)), key=q.__getitem__)],
                    "peft_probs": {c: round(x, 5) for c, x in zip(classes, q)},
                }
            f.write(json.dumps(row) + "\n")
    if lines and args.gen_n:
        # The same held-out moments, a line from the adapter and one from the
        # base model: narrate_ivr.py --judge scores both.
        gen = val_rows[: args.gen_n]
        ps = [prompt(r) for r in gen]
        said = {
            k: generate_lines(raw, tok, ps, pad_id, device, adapter=k == "adapter")
            for k in ("adapter", "base")
        }
        with open(args.out / "heldout_gen.jsonl", "w") as f:
            for i, r in enumerate(gen):
                row = {**r[5], "adapter": said["adapter"][i], "base": said["base"][i]}
                f.write(json.dumps(row) + "\n")
        for i in range(min(8, len(gen))):
            print(
                f"  {gen[i][5]['brief']}\n    written: {gen[i][5]['line']}\n"
                f"    adapter: {said['adapter'][i]}\n    base:    {said['base'][i]}"
            )
    metrics = {
        "adapter": a,
        "kind": args.kind,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_examples": len(train_rows),
        "steps": step,
        "seconds": round(time.time() - t0, 1),
        "heldout": ev,
        "history": history,
        "train_curve": train_curve,
        "best": best if args.keep_best else None,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    ref = (
        f"ppl {ev['ppl']:.3f}, first-token acc {ev['first_token_acc']:.4f}"
        if lines
        else f"vs majority {ev['majority_baseline']:.4f} ({ev['majority_class']})"
    )
    print(
        f"\n{a} ({args.kind}): held-out acc {ev['acc']:.4f} soft CE {ev['soft_ce']:.4f} "
        + ref
        + (
            f"; AUC damage-in-1s {ev['auc_damage_1s']:.3f}"
            if a == CRITIC and ev.get("auc_damage_1s")
            else ""
        )
        + f"; saved -> {args.out}"
    )
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
