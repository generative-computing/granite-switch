# SPDX-License-Identifier: Apache-2.0
"""Serving a loop slot's answer as one block of output tokens (backend-agnostic).

A loop request ends with its slot's control token, and the grid its head solves
is the ``grid_size`` prompt tokens that end ``end_offset`` tokens before that
marker. The HF backend rewrites those grid rows' logits in a single forward
pass. A serving engine computes logits only where it samples, so the vLLM
backend (``granite_switch.vllm.loop``) works differently:

1. As the prompt is prefilled, the read layer's state at each grid token is
   stored in a per-request buffer (prefill may span several engine steps).
2. In the step that completes the prompt, the head solves the stored grid and
   the request's output for that step is the whole answer: one token per grid
   cell (the head's best output token) followed by end of text.

This module holds the parts that need no vLLM: recognizing a loop request,
planning where each prefilled grid token's state goes, and the answer's tokens
and log-probabilities.
"""

from dataclasses import dataclass

import torch


@dataclass
class LoopRequest:
    """One loop request's grid and how its answer is emitted."""

    slot: int  # 1-based adapter index of the loop slot
    grid_start: int  # sequence position of the grid's first token
    grid_size: int
    seed: int  # the solver's starting point (see LoopHead.solve_rows)
    eos_token_id: int | None  # appended after the cells; None: not appended
    emitted: bool = False  # the answer is out; the request decodes normally after


def loop_markers(config) -> dict[int, int]:
    """``{control token id: 1-based adapter index}`` over the loop slots."""
    token_ids = config.adapter_token_ids or []
    offset = len(token_ids) - config.num_adapters
    return {
        token_ids[i + offset]: i + 1
        for i, kind in enumerate(config.adapter_kinds or [])
        if kind == "loop" and token_ids
    }


def find_loop_request(
    prompt_token_ids, markers, loop_specs, seed=None, eos_token_id=None
) -> LoopRequest | None:
    """The request's grid, or ``None`` when it is not a loop request.

    A loop request's last prompt token is a loop slot's control token. The HF
    backend raises when that marker sits anywhere else or the prompt is too
    short for the grid; a server must not fail on one request, so such a
    request is served as an ordinary one instead.
    """
    if not prompt_token_ids:
        return None
    slot = markers.get(int(prompt_token_ids[-1]))
    if slot is None:
        return None
    spec = loop_specs[slot - 1]
    grid_end = len(prompt_token_ids) - 1 - spec.get("end_offset", 0)
    grid_start = grid_end - spec["grid_size"]
    if grid_start < 0:
        return None
    return LoopRequest(
        slot=slot,
        grid_start=grid_start,
        grid_size=spec["grid_size"],
        seed=0 if seed is None else int(seed),
        eos_token_id=eos_token_id,
    )


def plan_grid_writes(rows, row_size: int) -> tuple[list[int], list[int]]:
    """Which of this step's tokens are grid tokens, and where each one goes.

    ``rows`` holds ``(buffer_row, request, num_computed, num_scheduled,
    query_start)`` per loop request in the batch: this step processes sequence
    positions ``[num_computed, num_computed + num_scheduled)`` of the request,
    starting at flat token ``query_start``. A grid position ``p`` goes to
    buffer row ``buffer_row * row_size + (p - grid_start)``.

    Returns ``(token_indices, buffer_indices)``, two equal-length lists.
    """
    tokens: list[int] = []
    dest: list[int] = []
    for buffer_row, req, computed, scheduled, start in rows:
        lo = max(computed, req.grid_start)
        hi = min(computed + scheduled, req.grid_start + req.grid_size)
        for pos in range(lo, hi):
            tokens.append(start + pos - computed)
            dest.append(buffer_row * row_size + pos - req.grid_start)
    return tokens, dest


def answer_token_ids(
    scores: torch.Tensor, output_token_ids, eos_token_id: int | None, limit: int
) -> list[int]:
    """The tokens a loop request emits: each cell's best output token, then EOS.

    ``scores`` is the head's ``[grid, outputs]`` matrix. Each cell emits its
    highest-scoring output token, which is what greedy decoding of the HF
    backend's rewritten logits gives. ``limit`` caps the count (the request's
    ``max_tokens`` and the model length).
    """
    ids = torch.as_tensor(output_token_ids)[scores.argmax(-1).cpu()].tolist()
    if eos_token_id is not None:
        ids.append(int(eos_token_id))
    return ids[: max(limit, 0)]


def answer_logprobs(
    scores: torch.Tensor,
    output_token_ids,
    eos_token_id: int | None,
    num_logprobs: int,
    logits_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per emitted token: ``(token ids [n, 1+k], values [n, 1+k], ranks [n])``.

    Column 0 is the emitted token, columns ``1..k`` the top ``k``, as vLLM lays
    out sample logprobs. A cell's logits are ``-inf`` except at the output
    tokens, where they are the head's scores centered at their maximum (the HF
    backend's rewritten row), so its log-probabilities are a log-softmax over
    the output tokens; ``logits_mode`` returns the centered scores instead. The
    end-of-text token is forced: value 0, rank 1. Columns past the number of
    output tokens hold token 0 at ``-inf``.
    """
    centered = scores.float() - scores.float().amax(-1, keepdim=True)
    values = centered if logits_mode else torch.log_softmax(centered, -1)
    cells, num_outputs = values.shape
    out_ids = torch.as_tensor(output_token_ids, device=values.device)
    rows = cells + (eos_token_id is not None)
    ids = torch.zeros(rows, 1 + num_logprobs, dtype=torch.long, device=values.device)
    vals = torch.full(
        (rows, 1 + num_logprobs),
        float("-inf"),
        dtype=torch.float32,
        device=values.device,
    )
    best = values.argmax(-1)
    ids[:cells, 0] = out_ids[best]
    vals[:cells, 0] = values.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    top = min(num_logprobs, num_outputs)
    if top:
        top_vals, top_idx = values.topk(top, dim=-1)
        ids[:cells, 1 : 1 + top] = out_ids[top_idx]
        vals[:cells, 1 : 1 + top] = top_vals
    if eos_token_id is not None:
        ids[cells, 0] = int(eos_token_id)
        vals[cells, 0] = 0.0
        if num_logprobs:
            ids[cells, 1] = int(eos_token_id)
            vals[cells, 1] = 0.0
    ranks = torch.ones(rows, dtype=torch.long, device=values.device)
    return ids, vals, ranks
