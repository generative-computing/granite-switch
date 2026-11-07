# SPDX-License-Identifier: Apache-2.0
"""Train a fixed-point loop head for Sudoku over a frozen Granite Switch model.

The checkpoint is composed with a ``kind: loop`` slot (make_slot.py, compose
``--untrained-loop-heads``). This script loads it, freezes everything but that
slot's loop head, and trains it on Sudoku-Extreme as in FPRM:

* a batch of slots carries puzzles across steps (FPRM's carry). Each step is
  one supervision round: ``--n-back`` fixed-point steps with gradients, one
  loss, one update, then the state is detached (truncated backprop);
* a slot gets a fresh puzzle once its residual is below ``fp_thresh`` or it
  has had ``--max-sup`` rounds; the base model reads only the fresh puzzles,
  under no_grad (the head never writes back, so a read never changes);
* AdamW, constant LR after warm-up, an EMA of the head's weights for eval.

Evaluation reads the head's answer through the real HF forward pass (the logit
rows at the 81 cells), so it measures the served path, not a training shortcut.

Multi-GPU: run under torchrun; each rank trains on its own shard and the head's
gradients are all-reduced (the base model is replicated, frozen).
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from sudoku_data import check

import granite_switch.hf  # noqa: F401
from granite_switch.hf import GraniteSwitchForCausalLM


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
class Prompt:
    """Token ids for one puzzle, laid out by the probe's chat template.

    ``apply_chat_template(..., adapter_name=<slot>)`` closes the user turn and
    places the probe marker last; the 81 cells end the user's message. Cells
    are single tokens: '.' for a blank, '1'..'9' for a given. They are looked up
    as tokens, never encoded as a string (the tokenizer merges digit runs).
    """

    INTRO = "Solve this sudoku. The grid is given row by row, '.' for a blank:\n"
    SENTINEL = "\x01"

    def __init__(self, tokenizer, slot_name, end_offset):
        msgs = [{"role": "user", "content": self.INTRO + self.SENTINEL}]
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, adapter_name=slot_name
        )
        head, tail = rendered.split(self.SENTINEL)
        self.head = tokenizer.encode(head, add_special_tokens=False)
        self.tail = tokenizer.encode(tail, add_special_tokens=False)
        self.marker = tokenizer.convert_tokens_to_ids(f"<|{slot_name}|>")
        if not self.tail or self.tail[-1] != self.marker:
            raise ValueError(f"the template does not end with the marker: {tail!r}")
        if len(self.tail) - 1 != end_offset:
            raise ValueError(
                f"{len(self.tail) - 1} tokens sit between the grid and the marker, "
                f"but the slot's end_offset is {end_offset}"
            )
        self.cell_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(".")]
            + [tokenizer.convert_tokens_to_ids(str(d)) for d in range(1, 10)]
        )
        self.grid_start = len(self.head)

    def batch(self, boards: torch.Tensor) -> torch.Tensor:
        """``[B, 81]`` boards (0 = blank) -> ``[B, L]`` input ids."""
        b = boards.shape[0]
        cells = self.cell_ids.to(boards.device)[boards.long()]
        head = torch.tensor(self.head, device=boards.device).expand(b, -1)
        tail = torch.tensor(self.tail, device=boards.device).expand(b, -1)
        return torch.cat([head, cells, tail], dim=1)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def stablemax_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """FPRM's stablemax cross-entropy, per token, in float64."""
    x = logits.double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    logp = torch.log(s / s.sum(-1, keepdim=True))
    return -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def loss_fn(logits, labels, kind):
    if kind == "stablemax":
        return stablemax_ce(logits, labels).mean()
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())


# --------------------------------------------------------------------------- #
# Base-model read
# --------------------------------------------------------------------------- #
@torch.no_grad()
def base_read(model, input_ids, grid_start, read_layer):
    """The base model's hidden states at the 81 grid cells (no logits).

    The prompt is read WITHOUT its trailing probe marker: with the marker, the
    model would also run the slot's own loop head (its full fixed-point solve)
    inside this forward, and training does that part itself. Causal attention
    makes the grid's states identical either way.
    """
    out = model.model(
        input_ids=input_ids[:, :-1],
        use_cache=False,
        output_hidden_states=read_layer is not None,
    )
    if read_layer is None:
        hidden = out.last_hidden_state
    else:
        # hidden_states[i] is the input to decoder layer i; layer L's output is
        # i = L + 1, and read_layer -1 (the input embeddings) is i = 0.
        hidden = out.hidden_states[read_layer + 1]
    return hidden[:, grid_start : grid_start + 81]


# --------------------------------------------------------------------------- #
# Evaluation through the served HF path
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, prompt, boards, sols, batch_size, digit_ids, device):
    """Exact/valid/cell accuracy through the served HF path.

    Under torchrun every rank scores its own shard of the puzzles and the
    counts are summed, so a long fixed-point eval is split across the GPUs.
    """
    model.eval()
    ddp = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    n = len(boards)
    lo, hi = rank * n // world, (rank + 1) * n // world
    cells = exact = valid = blank_right = blanks = 0
    iters = []
    for i in range(lo, hi, batch_size):
        j = min(hi, i + batch_size)
        bq = torch.as_tensor(boards[i:j], device=device)
        ba = sols[i:j]
        out = model(input_ids=prompt.batch(bq), use_cache=False)
        rows = out.logits[:, prompt.grid_start : prompt.grid_start + 81]
        pred = (rows.argmax(-1).unsqueeze(-1) == digit_ids).float().argmax(
            -1
        ) + 1  # token -> digit
        pred = pred.cpu().numpy()
        cells += int((pred == ba).sum())
        given = bq.cpu().numpy() > 0
        blank_right += int(((pred == ba) & ~given).sum())
        blanks += int((~given).sum())
        exact += int((pred == ba).all(axis=1).sum())
        valid += sum(check(bq[k].cpu().numpy(), pred[k]) for k in range(len(pred)))
        it = getattr(model.model, "_last_loop_iters", None)
        if it is not None:
            iters.append(it.float().cpu())
    it = torch.cat(iters) if iters else torch.zeros(0)  # a rank's shard can be empty
    stats = torch.tensor(
        [
            cells,
            exact,
            valid,
            float(it.sum()),
            float(it.numel()),
            float(it.max()) if it.numel() else 0.0,
            blank_right,
            blanks,
        ],
        dtype=torch.float64,
        device=device,
    )
    if ddp:
        top = stats[5:6].clone()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(top, op=dist.ReduceOp.MAX)
        stats[5] = top[0]
    cells, exact, valid, it_sum, it_n, it_max, blank_right, blanks = stats.tolist()
    return {
        "solver_iters_mean": it_sum / max(1.0, it_n),
        "solver_iters_max": it_max,
        "cell_acc": cells / (81 * n),
        # Only the cells the puzzle leaves blank (the givens come back as given).
        "blank_cell_acc": blank_right / max(1.0, blanks),
        "exact_acc": exact / n,
        "valid_acc": valid / n,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", required=True, help="composed checkpoint with a loop slot"
    )
    p.add_argument("--slot-name", default="sudoku")
    p.add_argument("--data", required=True, help="folder with train.npz and test.npz")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--steps",
        type=int,
        default=78125,
        help="supervision rounds, one update each (FPRM: 60000 epochs x 1000 groups / 768)",
    )
    p.add_argument("--batch", type=int, default=768, help="global batch size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument(
        "--n-back", type=int, default=6, help="fixed-point steps per supervision round"
    )
    p.add_argument(
        "--max-sup",
        type=int,
        default=12,
        help="supervision rounds before a puzzle is replaced",
    )
    p.add_argument("--loss", choices=["stablemax", "softmax"], default="stablemax")
    p.add_argument(
        "--train-decay",
        type=float,
        default=0.9,
        help="step-size decay in training (FPRM: 0.9)",
    )
    p.add_argument(
        "--train-patience",
        type=int,
        default=5,
        help="non-improving steps before decay in training (FPRM: 5)",
    )
    p.add_argument(
        "--min-sup",
        type=int,
        default=0,
        help="rounds before a converged puzzle may leave (FPRM: 0)",
    )
    p.add_argument(
        "--q-loss-coeff",
        type=float,
        default=0.5,
        help="weight of the halt head's BCE (FPRM: 0.5)",
    )
    p.add_argument(
        "--clip", type=float, default=0.0, help="gradient-norm clip (FPRM: none)"
    )
    p.add_argument(
        "--sample",
        choices=["group", "uniform"],
        default="group",
        help="group: one random augmentation of a random base puzzle per draw (FPRM)",
    )
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument(
        "--eval-n", type=int, default=2000, help="test puzzles per periodic eval"
    )
    p.add_argument(
        "--final-eval-n", type=int, default=0, help="0 = the whole test split"
    )
    p.add_argument(
        "--eval-max-iter", type=int, default=None, help="override the spec's max_iter"
    )
    p.add_argument(
        "--eval-min-iter", type=int, default=None, help="override the spec's min_iter"
    )
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="stop after this many evals without a new best (0: never)",
    )
    p.add_argument(
        "--read-dropout",
        type=float,
        default=0.0,
        help="feature dropout on the base read, one mask per puzzle for its whole trajectory",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="run the head in fp32 (default: bf16 autocast, as FPRM)",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="resume.pt of an earlier run: continue after its last eval",
    )
    args = p.parse_args()

    # torchrun sets LOCAL_RANK. The cluster's job operator sets RANK/WORLD_SIZE on
    # every pod (for multi-node jobs), so RANK alone does not mean torchrun.
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        # Evals run long fixed-point solves; keep the collective timeout above them.
        # gloo serves CPU plumbing checks.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=datetime.timedelta(hours=3))
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=not args.fp32
    )
    ck = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    start = ck["step"] + 1 if ck else 1
    torch.manual_seed(args.seed + rank + 7919 * (start - 1))
    rng = np.random.default_rng(
        args.seed + rank if start == 1 else [args.seed, rank, start]
    )
    is_main = rank == 0
    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = GraniteSwitchForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device)
    cfg = model.config
    slot = cfg.adapter_names.index(args.slot_name) + 1
    spec = (cfg.loop_specs or [None] * cfg.num_adapters)[slot - 1]
    if spec is None:
        raise SystemExit(f"slot {args.slot_name!r} is not a loop slot")
    if args.eval_max_iter is not None:
        # The served path reads the solver budget from the config.
        spec["max_iter"] = args.eval_max_iter
    if args.eval_min_iter is not None:
        spec["min_iter"] = args.eval_min_iter
    head = model.model.loop_heads.head(slot)
    # The head trains in fp32; the frozen base runs in bf16.
    head.float()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(f"model.loop_heads.heads.{slot - 1}."))
    trainable = [p_ for p_ in head.parameters()]
    n_params = sum(p_.numel() for p_ in trainable)
    prompt = Prompt(tok, args.slot_name, spec.get("end_offset", 0))
    digit_ids = torch.tensor(spec["output_token_ids"], device=device)
    assert prompt.cell_ids[1:].tolist() == digit_ids.tolist(), (
        "cell tokens must match the head's outputs"
    )

    train = np.load(Path(args.data) / "train.npz")
    test = np.load(Path(args.data) / "test.npz")
    tx, ty = train["x"], train["y"]
    ex, ey = test["x"], test["y"]
    if is_main:
        print(
            f"train {len(tx)} test {len(ex)} head params {n_params / 1e6:.2f}M world {world}",
            flush=True,
        )

    # FPRM: decay every trainable param except those tagged _no_weight_decay
    # (the residual-scale alphas and the conv).
    no_decay = [
        p_ for p_ in head.parameters() if getattr(p_, "_no_weight_decay", False)
    ]
    decay = [
        p_ for p_ in head.parameters() if not getattr(p_, "_no_weight_decay", False)
    ]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    if ddp:
        # The gradients are averaged, so identical starts keep the ranks identical.
        for t in head.state_dict().values():
            dist.broadcast(t, 0)
    ema = copy.deepcopy(head).requires_grad_(False)

    local_batch = args.batch // world
    log_path = out_dir / "train_log.jsonl"
    t0 = time.time()
    order = rng.permutation(len(tx))
    cursor = 0

    # build() stores each base puzzle followed by its augmentations, and records
    # the group size (1 + num_aug).
    group_size = int(train["group_size"]) if "group_size" in train.files else 1
    groups = len(tx) // group_size
    if args.sample == "group":
        order = rng.permutation(groups)

    def draw(n):
        """The next ``n`` training puzzles.

        FPRM: one pass over the base puzzles (groups) per epoch, in a fresh
        order, taking one random augmentation of each.
        """
        nonlocal order, cursor
        size = groups if args.sample == "group" else len(tx)
        while cursor + n > len(order):  # passes run back to back
            order = np.concatenate([order[cursor:], rng.permutation(size)])
            cursor = 0
        pick = order[cursor : cursor + n]
        cursor += n
        idx = (
            pick * group_size + rng.integers(0, group_size, n)
            if args.sample == "group"
            else pick
        )
        return (
            torch.as_tensor(tx[idx], device=device),
            torch.as_tensor(ty[idx], device=device).long() - 1,  # digits 1..9 -> 0..8
        )

    def read_of(boards):
        r = base_read(
            model, prompt.batch(boards), prompt.grid_start, spec["read_layer"]
        ).float()
        # Drawn once when a puzzle enters its slot, so the mask is fixed across
        # the puzzle's supervision rounds (variational dropout on the input).
        return (
            F.dropout(r, args.read_dropout, training=True) if args.read_dropout else r
        )

    # FPRM's carry: each step is one supervision round (n_back solver steps, one
    # loss, one update) over a fixed batch of slots. A slot keeps its puzzle,
    # read and solver state across steps until it converges or has had max_sup
    # rounds; then it gets a fresh puzzle from the next batch.
    boards, labels = draw(local_batch)
    read = read_of(boards)
    head.train()
    with amp:
        state = head.init_state(head.inject(read))
    state["z"] = state["z"].float()
    rounds = torch.zeros(local_batch, dtype=torch.long, device=device)
    halt_solved, halt_rounds, halt_res = [], [], []
    best_exact, best_step, stale, best_state, best_score = -1.0, 0, 0, None, None
    if ck:
        # The carry restarts with fresh puzzles; everything else continues.
        head.load_state_dict(ck["head"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        best_score, best_exact, best_step, stale, best_state = ck["best"]
        if is_main:
            print(
                f"resumed after step {start - 1}: best exact {best_exact:.4f} at step {best_step}",
                flush=True,
            )
    stop = torch.zeros(1, device=device)
    for step in range(start, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr

        with amp:
            x = head.inject(read)
            for _k in range(args.n_back):
                state = head.step(
                    state, x, args.train_decay, args.train_patience, spec["fp_thresh"]
                )
            logits = head.logits(state["z"])
            q = head.halt_logit(state["z"])
        lm_loss = loss_fn(logits, labels, args.loss)
        with torch.no_grad():
            solved_now = (logits.argmax(-1) == labels).all(-1).float()
        q_loss = F.binary_cross_entropy_with_logits(q, solved_now)
        loss = lm_loss + args.q_loss_coeff * q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ddp:
            for p_ in trainable:
                if p_.grad is not None:
                    dist.all_reduce(p_.grad)  # SUM, then mean: gloo has no AVG
                    p_.grad.div_(world)
        if args.clip:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), head.parameters()):
                pe.lerp_(pm, 1 - args.ema)

        # Carry: detach, then replace the slots that halted.
        with torch.no_grad():
            state = {k: v.detach() for k, v in state.items()}
            rounds += 1
            correct = (logits.argmax(-1) == labels).all(-1)
            # A puzzle leaves after max_sup rounds, or once converged (residual
            # below fp_thresh, or a collapsed step size) after at least min_sup.
            converged = (state["residual"] < spec["fp_thresh"]) | (
                state["stepsize"].view(-1) < 1e-3
            )
            halted = (rounds >= args.max_sup) | (converged & (rounds >= args.min_sup))
            halt_solved.append(correct[halted].float().cpu())
            halt_rounds.append(rounds[halted].float().cpu())
            halt_res.append(state["residual"][halted].float().cpu())
            n_new = int(halted.sum())
            if n_new:
                nb, nl = draw(n_new)
                boards[halted], labels[halted] = nb, nl
                read[halted] = read_of(nb)
                with amp:
                    fresh = head.init_state(head.inject(read[halted]))
                for k in state:
                    state[k][halted] = fresh[k].to(state[k].dtype)
                rounds[halted] = 0

        if is_main and (step % 50 == 0 or step == 1):
            with torch.no_grad():
                acc = (logits.argmax(-1) == labels).float().mean().item()
                # Diagnostics: the update's size and the head's learned scales.
                grads = [p_.grad.norm() for p_ in trainable if p_.grad is not None]
                grad_norm = torch.stack(grads).norm().item() if grads else 0.0
                a1, _b1, a2, _b2 = head.scales()
                x_rms = x.float().pow(2).mean(-1).sqrt()
            hs, hr, hres = (
                torch.cat(v) if v else torch.zeros(1)
                for v in (halt_solved, halt_rounds, halt_res)
            )
            rec = {
                "step": step,
                "loss": loss.item(),
                "cell_acc": acc,
                # Puzzles that left their slot since the last log: solved share,
                # rounds used, and residual at the time they left.
                "halted": int(hs.numel()) if halt_solved else 0,
                "halted_solved": hs.mean().item(),
                "halted_rounds_mean": hr.mean().item(),
                "halted_residual_median": hres.median().item(),
                "q_loss": q_loss.item(),
                "grad_norm": grad_norm,
                "alpha_1_median": a1.median().item(),
                "alpha_2_median": a2.median().item(),
                "x_rms_grid": x_rms[:, head.prefix_len :].mean().item(),
                "x_rms_prefix": x_rms[:, : head.prefix_len].mean().item(),
                "lr": lr,
                "elapsed_s": round(time.time() - t0, 1),
            }
            halt_solved, halt_rounds, halt_res = [], [], []
            print("TRAIN " + json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if step % args.eval_every == 0 or step == args.steps:
            # Every rank evaluates its shard with the EMA head; rank 0 records.
            t_eval = time.time()
            live = copy.deepcopy(head.state_dict())
            head.load_state_dict(ema.state_dict())
            with amp:
                res = evaluate(
                    model,
                    prompt,
                    ex[: args.eval_n],
                    ey[: args.eval_n],
                    args.eval_batch,
                    digit_ids,
                    device,
                )
            head.load_state_dict(live)
            res["step"] = step
            res["eval_s"] = round(time.time() - t_eval, 1)
            t0 += time.time() - t_eval  # elapsed_s counts training time only
            if is_main:
                print("EVAL " + json.dumps(res), flush=True)
                with open(out_dir / "eval_log.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                save_head(ema, out_dir / "loop_head.safetensors")
                # Keep every eval's head, and the best one by exact accuracy.
                save_head(ema, out_dir / f"loop_head.step{step}.safetensors")
            # Ties keep the earlier head; cell accuracy breaks a tie at exact 0.
            score = (res["exact_acc"], res["cell_acc"])
            if best_state is None or score > best_score:
                best_score = score
                best_exact, best_step, stale = res["exact_acc"], step, 0
                best_state = copy.deepcopy(ema.state_dict())
                if is_main:
                    save_head(ema, out_dir / "loop_head.best.safetensors")
            else:
                stale += 1
            if is_main:
                torch.save(
                    {
                        "step": step,
                        "head": head.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "best": [best_score, best_exact, best_step, stale, best_state],
                    },
                    out_dir / "resume.pt",
                )
        if args.patience and stale >= args.patience:  # same decision on every rank
            if is_main:
                print(
                    f"EARLY_STOP at step {step}: best exact {best_exact:.4f} at step {best_step}",
                    flush=True,
                )
            stop.fill_(1)
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            dist.barrier()
        if stop.item():
            break

    # The final test scores the best head by the periodic eval (2,000 test
    # puzzles), on a larger slice of the test split. Every rank evaluates its
    # shard; rank 0 records.
    head.load_state_dict(best_state if best_state is not None else ema.state_dict())
    # Disjoint from the periodic evals, which choose the head: test puzzles
    # [eval_n, eval_n + final_eval_n) are never seen by any decision.
    lo = args.eval_n
    hi = len(ex) if not args.final_eval_n else min(len(ex), lo + args.final_eval_n)
    with amp:
        res = evaluate(
            model, prompt, ex[lo:hi], ey[lo:hi], args.eval_batch, digit_ids, device
        )
    if is_main:
        res["test_slice"] = [lo, hi]
        res["step"] = best_step if best_state is not None else step
        res["final"] = True
        print("FINAL " + json.dumps(res), flush=True)
        (out_dir / "final_eval.json").write_text(json.dumps(res, indent=2))
        save_head(head, out_dir / "loop_head.safetensors")
        save_head(head, out_dir / "loop_head.best.safetensors")
    if ddp:
        dist.destroy_process_group()


def save_head(head, path):
    """Write the loop head's weights: the slot folder's loop_head.safetensors."""
    from safetensors.torch import save_file

    sd = {
        k: v.detach().float().contiguous().cpu() for k, v in head.state_dict().items()
    }
    save_file(sd, str(path))


if __name__ == "__main__":
    main()
