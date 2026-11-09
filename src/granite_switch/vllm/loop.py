# SPDX-License-Identifier: Apache-2.0
"""Loop slots on vLLM: the whole answer as one step's output tokens.

Background. The HF backend serves a loop request in one forward pass: the head
solves the grid it reads and rewrites the grid rows' logits. vLLM computes
logits only where it samples (one row per request per step) and keeps nothing
but K/V between steps, so neither half carries over. This module serves the
answer the way vLLM serves block-diffusion models (DiffusionGemma): a
model-specific state and sampler, installed through Model Runner V2's hooks
(``get_model_state_cls`` and ``ModelState.custom_sampler``), produce a block
of tokens for one request in one step, while the other requests in the batch
decode as usual.

How a loop request runs (prompt ``... grid ... <close> <|loop marker|>``):

1. **Read.** While the prompt is prefilled, the compiled forward copies the
   read layer's state at each grid token into a per-request buffer
   (``LoopGridReads``). The grid may span several prefill chunks; the buffer
   outlives each forward and keeps only the grid rows.
2. **Solve and emit.** In the step that completes the prompt, the sampler
   solves each such request's grid (``LoopHead.solve_rows``: a per-request
   seeded start and per-request stopping, so the answer does not depend on
   the batch) and returns that request's whole answer: one token per grid
   cell (the head's best output token), then end of text.
3. **Everything else** in the batch goes through vLLM's own sampler.

What is installed, and only for a checkpoint with a loop slot:

* ``LoopModelState`` (vLLM's default model state, extended) recognizes loop
  requests, plans the grid copy before each forward, and installs
  ``LoopSampler``.
* ``LoopSampler`` wraps vLLM's sampler and adds the answers.
* ``make_loop_scheduler`` builds vLLM's scheduler (sync or async, whichever the
  engine runs) with two changes: loop requests never read the prefix cache (a
  hit would skip the grid's forward, and the head reads the grid there), and a
  loop request ends with its answer, with the async scheduler's placeholder
  count covering the extra tokens.
* ``loop_config.GraniteSwitchConfigHook`` (vLLM's per-architecture config
  hook, registered by ``register``) selects that scheduler.

Model Runner V2 is required (vLLM 0.26 and 0.30 both have its hooks).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.library import Library
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from ..loop_serving import (
    LoopRequest,
    answer_logprobs,
    answer_token_ids,
    find_loop_request,
    loop_markers,
    plan_grid_writes,
)

logger = init_logger(__name__)

# --------------------------------------------------------------------------- #
# The grid read: a buffer the compiled forward writes into
# --------------------------------------------------------------------------- #

_loop_lib = Library("granite_switch_loop", "FRAGMENT")


def _loop_grid_capture(
    hidden: torch.Tensor,
    token_index: torch.Tensor,
    dest_index: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """``rows[dest_index[j]] = hidden[token_index[j]]`` for each planned ``j``.

    Unplanned entries point at the last row of ``rows`` (a sink), and entries
    past this forward's token count are sent there too, so the op is
    branch-free, shape-static and safe to capture in a CUDA graph.
    """
    n = hidden.shape[0]
    sink = rows.shape[0] - 1
    src = token_index.clamp(min=0, max=max(n - 1, 0))
    dest = torch.where(token_index < n, dest_index, torch.full_like(dest_index, sink))
    rows.index_copy_(0, dest, hidden[src].to(rows.dtype))


def _loop_grid_capture_fake(hidden, token_index, dest_index, rows) -> None:
    return


direct_register_custom_op(
    op_name="loop_grid_capture",
    op_func=_loop_grid_capture,
    mutates_args=["rows"],
    fake_impl=_loop_grid_capture_fake,
    target_lib=_loop_lib,
)
loop_grid_capture = torch.ops.granite_switch_loop.loop_grid_capture


class LoopGridReads(nn.Module):
    """Per-request copies of the read layer's state at the grid tokens.

    ``rows[r * grid + i]`` holds grid cell ``i`` of the request in model-runner
    slot ``r``. Before each forward the model state writes this step's plan
    into ``token_index`` / ``dest_index`` (in place, so their addresses stay
    fixed for CUDA-graph replay), and the forward calls this module on the read
    layer's ``[T, H]`` state.
    """

    def __init__(self, max_num_reqs, grid_size, hidden_size, max_tokens, dtype):
        super().__init__()
        self.grid_size = grid_size
        self.sink = max_num_reqs * grid_size
        self.capacity = min(max_tokens, max_num_reqs * grid_size)
        self.register_buffer(
            "rows",
            torch.zeros(self.sink + 1, hidden_size, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "token_index",
            torch.zeros(self.capacity, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "dest_index",
            torch.full((self.capacity,), self.sink, dtype=torch.long),
            persistent=False,
        )

    def forward(self, hidden: torch.Tensor) -> None:
        loop_grid_capture(hidden, self.token_index, self.dest_index, self.rows)

    def set_plan(self, tokens: list[int], dest: list[int]) -> None:
        """This step's copies, as ``plan_grid_writes`` lists them."""
        n = len(tokens)
        if n > self.capacity:
            raise RuntimeError(f"loop grid plan of {n} tokens exceeds {self.capacity}")
        self.dest_index.fill_(self.sink)
        if n:
            dev = self.token_index.device
            self.token_index[:n].copy_(torch.tensor(tokens, device=dev))
            self.dest_index[:n].copy_(torch.tensor(dest, device=dev))

    def grid(self, slot: int) -> torch.Tensor:
        start = slot * self.grid_size
        return self.rows[start : start + self.grid_size]


def loop_read_layers(config) -> dict[int | None, list[int]]:
    """``{read layer: [1-based loop slots]}`` (``-1``: embeddings, None: final)."""
    out: dict[int | None, list[int]] = {}
    for i, kind in enumerate(config.adapter_kinds or []):
        if kind == "loop":
            out.setdefault(config.loop_specs[i]["read_layer"], []).append(i + 1)
    return out


def check_loop_serving(config, vllm_config) -> None:
    """Raise for setups this backend does not serve loop slots in."""
    slots = [i for i, k in enumerate(config.adapter_kinds or []) if k == "loop"]
    grids = {config.loop_specs[i]["grid_size"] for i in slots}
    problems = []
    if len(grids) > 1:
        problems.append(f"loop slots with different grid sizes {sorted(grids)}")
    if len(loop_read_layers(config)) > 1:
        problems.append("loop slots reading different layers")
    if not vllm_config.use_v2_model_runner:
        problems.append("the V1 model runner (unset VLLM_USE_V2_MODEL_RUNNER=0)")
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        problems.append("pipeline parallelism")
    if vllm_config.speculative_config is not None:
        problems.append("speculative decoding")
    if getattr(vllm_config.parallel_config, "enable_batch_sharded_sampling", False):
        problems.append("batch-sharded sampling")
    if getattr(vllm_config.model_config, "return_sampling_mask", False):
        problems.append("sampling-mask replay")
    if problems:
        raise ValueError(
            "Granite Switch loop slots on vLLM do not support: " + "; ".join(problems)
        )


# --------------------------------------------------------------------------- #
# Model Runner V2: model state and sampler
# --------------------------------------------------------------------------- #


class LoopModelState(DefaultModelState):
    """Model Runner V2 state for a checkpoint with loop slots.

    Tracks the running loop requests by the runner's slot index (which is also
    the request's row in the grid buffer), plans the grid copy before each
    forward, and installs ``LoopSampler``.
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.config = model.config
        self.markers = loop_markers(self.config)
        self.reads: LoopGridReads = model.model.loop_reads
        self.loop_requests: dict[int, LoopRequest] = {}
        self.loop_req_ids: dict[int, str] = {}
        self.loop_slot_of: dict[str, int] = {}

    def add_request(self, req_index: int, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        self._forget(req_index)
        params = new_req_data.sampling_params
        if params is None:
            return
        req = find_loop_request(
            new_req_data.prompt_token_ids,
            self.markers,
            self.config.loop_specs,
            seed=params.seed,
            eos_token_id=params.eos_token_id,
        )
        if req is None:
            return
        prompt_len = len(new_req_data.prompt_token_ids)
        token_ids = new_req_data.prefill_token_ids or new_req_data.prompt_token_ids
        # A request resumed after its answer went out decodes normally.
        req.emitted = len(token_ids) > prompt_len
        limit = self.max_model_len - prompt_len
        if params.max_tokens is not None:
            limit = min(limit, params.max_tokens)
        req.max_tokens = limit
        self.loop_requests[req_index] = req
        self.loop_req_ids[req_index] = new_req_data.req_id
        self.loop_slot_of[new_req_data.req_id] = req_index

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        slot = self.loop_slot_of.pop(req_id, None)
        if slot is not None:
            self._forget(slot)

    def _forget(self, slot: int) -> None:
        self.loop_requests.pop(slot, None)
        req_id = self.loop_req_ids.pop(slot, None)
        if req_id is not None:
            self.loop_slot_of.pop(req_id, None)

    def pending(self, input_batch):
        """``(batch row, slot, request)`` of this batch's loop requests yet to answer.

        Matched by request id as well as slot: a dummy batch (profiling, DP
        padding) reuses slot numbers.
        """
        if not self.loop_requests:
            return []
        out = []
        for b, slot in enumerate(input_batch.idx_mapping_np.tolist()):
            req = self.loop_requests.get(slot)
            if (
                req is not None
                and not req.emitted
                and b < len(input_batch.req_ids)
                and input_batch.req_ids[b] == self.loop_req_ids.get(slot)
            ):
                out.append((b, slot, req))
        return out

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        rows = [
            (
                slot,
                req,
                int(input_batch.num_computed_tokens_np[b]),
                int(input_batch.num_scheduled_tokens[b]),
                int(input_batch.query_start_loc_np[b]),
            )
            for b, slot, req in self.pending(input_batch)
        ]
        self.reads.set_plan(*plan_grid_writes(rows, self.reads.grid_size))
        return super().prepare_inputs(input_batch, req_states)

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        self.reads.set_plan([], [])
        return super().prepare_dummy_inputs(num_reqs, num_tokens)

    def custom_sampler(self, sampler):
        return LoopSampler(sampler, self), None


class LoopSampler:
    """vLLM's sampler, plus the whole answer of each loop request that completes.

    Every row goes through the wrapped sampler. A loop request whose prompt
    completes this step is then solved, and its row's tokens are replaced by
    the answer (``grid_size`` cells and end of text, capped by ``max_tokens``
    and the model length). A loop request still prefilling gets no tokens, as
    the wrapped sampler already decides.
    """

    def __init__(self, sampler, state: LoopModelState):
        self.base = sampler
        self.state = state

    def __getattr__(self, name):  # sampling_states, penalties_state, ...
        return getattr(self.base, name)

    def add_request(self, *args, **kwargs) -> None:
        self.base.add_request(*args, **kwargs)

    def apply_staged_writes(self) -> None:
        self.base.apply_staged_writes()

    def __call__(self, logits, input_batch, *args, **kwargs) -> SamplerOutput:
        out = self.base(logits, input_batch, *args, **kwargs)
        done = [
            (b, slot, req)
            for b, slot, req in self.state.pending(input_batch)
            if int(input_batch.num_computed_tokens_np[b])
            + int(input_batch.num_scheduled_tokens[b])
            >= int(input_batch.prefill_len_np[b])
        ]
        if not done:
            return out
        return self._with_answers(out, done)

    @torch.no_grad()
    def _with_answers(self, out: SamplerOutput, done) -> SamplerOutput:
        config = self.state.config
        heads = self.state.model.model.loop_heads
        reads = self.state.reads
        answers: dict[int, tuple[torch.Tensor, LoopRequest]] = {}
        by_slot: dict[int, list] = {}
        for item in done:
            by_slot.setdefault(item[2].slot, []).append(item)
        for loop_slot, group in by_slot.items():
            spec = config.loop_specs[loop_slot - 1]
            grids = torch.stack([reads.grid(slot) for _, slot, _ in group])
            amp = spec.get("solve_dtype") == "bfloat16"
            with torch.autocast(grids.device.type, dtype=torch.bfloat16, enabled=amp):
                scores, state = heads.head(loop_slot).solve_rows(
                    grids,
                    seeds=[req.seed for _, _, req in group],
                    max_iter=spec["max_iter"],
                    fp_thresh=spec["fp_thresh"],
                    decay=spec["stepsize_decay"],
                    patience=spec["decay_patience"],
                    min_iter=spec.get("min_iter", 0),
                )
            iters = state["iters"].tolist()
            for k, (b, _, req) in enumerate(group):
                answers[b] = (scores[k].float(), req)
                req.iters = iters[k]
            logger.info(
                "loop slot %d: solved %d grid(s) in %s solver iterations",
                loop_slot,
                len(group),
                iters,
            )

        sampled = out.sampled_token_ids
        num_reqs = sampled.shape[0]
        counts = (
            out.num_sampled.tolist()
            if out.num_sampled is not None
            else [sampled.shape[1]] * num_reqs
        )
        rows = [sampled[b, : counts[b]].tolist() for b in range(num_reqs)]
        for b, (scores, req) in answers.items():
            spec = config.loop_specs[req.slot - 1]
            rows[b] = answer_token_ids(
                scores, spec["output_token_ids"], req.eos_token_id, req.max_tokens
            )
            req.emitted = True

        width = max(1, max(len(r) for r in rows))
        dev = sampled.device
        new_sampled = torch.zeros(num_reqs, width, dtype=sampled.dtype, device=dev)
        for b, r in enumerate(rows):
            if r:
                new_sampled[b, : len(r)] = torch.tensor(
                    r, dtype=sampled.dtype, device=dev
                )
        num_sampled = torch.tensor(
            [len(r) for r in rows], dtype=torch.int32, device=dev
        )
        num_rejected = (
            out.num_rejected
            if out.num_rejected is not None
            else torch.zeros_like(num_sampled)
        )
        return SamplerOutput(
            sampled_token_ids=new_sampled,
            logprobs_tensors=self._logprobs(out, answers, rows),
            num_nans=out.num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _logprobs(self, out, answers, rows) -> LogprobsTensors | None:
        """Sample logprobs, one row per emitted token (spec decoding's layout)."""
        base = out.logprobs_tensors
        if base is None:
            return None
        k = base.logprob_token_ids.shape[1] - 1
        logits_mode = self.base.logprobs_mode in ("raw_logits", "processed_logits")
        ids, vals, ranks, cu = [], [], [], [0]
        for b, r in enumerate(rows):
            n = len(r)
            if b in answers:
                scores, req = answers[b]
                spec = self.state.config.loop_specs[req.slot - 1]
                a_ids, a_vals, a_ranks = answer_logprobs(
                    scores, spec["output_token_ids"], req.eos_token_id, k, logits_mode
                )
                ids.append(a_ids[:n].to(base.logprob_token_ids))
                vals.append(a_vals[:n].to(base.logprobs))
                ranks.append(a_ranks[:n].to(base.selected_token_ranks))
            else:
                ids.append(base.logprob_token_ids[b : b + n])
                vals.append(base.logprobs[b : b + n])
                ranks.append(base.selected_token_ranks[b : b + n])
            cu.append(cu[-1] + n)
        return LogprobsTensors(
            logprob_token_ids=torch.cat(ids),
            logprobs=torch.cat(vals),
            selected_token_ranks=torch.cat(ranks),
            cu_num_generated_tokens=cu,
        )


# --------------------------------------------------------------------------- #
# Scheduler and config hook
# --------------------------------------------------------------------------- #


class _LoopSchedulerMixin:
    """Loop-request handling on top of vLLM's scheduler (sync or async).

    * A loop request skips reading the prefix cache: a hit would skip the
      grid's forward, and the head reads the grid there.
    * A loop request's output is its answer: it ends with it even when the
      answer carries no end of text (``ignore_eos``), so its tokens never need
      K/V. The async scheduler expects one token per step and counts the
      extra ones against its placeholders first.
    """

    _async = False

    def _loop_setup(self) -> None:
        self._loop_markers = loop_markers(self.vllm_config.model_config.hf_config)

    def _is_loop(self, request) -> bool:
        ids = request.prompt_token_ids
        return bool(ids) and int(ids[-1]) in self._loop_markers

    def add_request(self, request) -> None:
        if self._is_loop(request):
            request.skip_reading_prefix_cache = True
        super().add_request(request)

    def _update_request_with_output(self, request, new_token_ids, *args, **kwargs):
        is_answer = (
            len(new_token_ids) > 1
            and request.num_output_tokens == 0
            and self._is_loop(request)
        )
        stale = kwargs.get("is_stale", args[0] if args else False)
        if is_answer and self._async and not stale:
            request.num_output_placeholders += len(new_token_ids) - 1
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids, *args, **kwargs
        )
        if is_answer and not stopped:
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True
        return new_token_ids, stopped


class LoopScheduler(_LoopSchedulerMixin, Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop_setup()


class LoopAsyncScheduler(_LoopSchedulerMixin, AsyncScheduler):
    _async = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop_setup()


def make_loop_scheduler(*args, **kwargs):
    """vLLM's scheduler (sync or async, as the engine runs) with loop handling.

    Named as ``scheduler_cls`` by ``loop_config.GraniteSwitchConfigHook``. The
    engine core calls it with its final config, so it follows the engine's own
    ``async_scheduling`` choice.
    """
    vllm_config = kwargs.get("vllm_config", args[0] if args else None)
    cls = (
        LoopAsyncScheduler
        if vllm_config.scheduler_config.async_scheduling
        else LoopScheduler
    )
    return cls(*args, **kwargs)
