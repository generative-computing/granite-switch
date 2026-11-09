# SPDX-License-Identifier: Apache-2.0
"""Loop slots on vLLM: emit a loop head's whole answer in the step that reads it.

A loop slot's head solves a grid of prompt positions (e.g. the 81 cells of a
Sudoku) from one layer of the base model. Its answer is one token per cell.
vLLM samples one token per request per step, and the grid's hidden states
exist only in the step that prefills the prompt, so the answer cannot be
spread over later steps. Instead:

* the compiled forward copies the read layer of every scheduled token into a
  persistent buffer (``GraniteSwitchModel._loop_read_<layer>``);
* :class:`LoopBurstSampler`, which vLLM's V2 model runner gets from
  :meth:`LoopSlotModelState.custom_sampler`, finds the loop requests whose
  prefill completes in the step, gathers each one's grid from that buffer,
  solves them, and returns each whole answer plus EOS as that request's
  sampled tokens. Every other request samples as usual.

The V2 runner already commits any number of sampled tokens per request:
``post_update`` writes ``num_sampled`` tokens into the request's history, and
the scheduler appends them one by one and stops at EOS. A loop request
therefore finishes in its prefill step.

The worker cannot reject a request without stopping the engine, so a loop
request the burst cannot serve (logprobs, structured output, ignore_eos, ...)
is logged and decoded as an ordinary request instead.

See docs/LOOP_VLLM_MULTI_TOKEN.md for the design and its limits.
"""

from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

logger = init_logger(__name__)


@dataclass
class LoopRequest:
    """What the sampler needs to answer one loop request."""

    slot: int  # 1-based adapter index of the loop slot
    eos_token_id: int | None
    max_new_tokens: int  # how many tokens the request may produce


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def loop_request_slot(prompt_token_ids, loop_marker_to_slot: dict[int, int]):
    """The loop slot a prompt asks for, or ``None``.

    A loop request ends with its slot's control token, like a probe request.
    """
    if not prompt_token_ids:
        return None
    return loop_marker_to_slot.get(int(prompt_token_ids[-1]))


def loop_request_problem(spec, prompt_len, sampling_params) -> str | None:
    """Why the answer burst cannot serve this loop request, or ``None``."""
    need = spec["grid_size"] + spec.get("end_offset", 0) + 1
    if prompt_len < need:
        return (
            f"the prompt has {prompt_len} tokens, fewer than the {need} its grid "
            "needs (grid_size + end_offset + the marker)"
        )
    if sampling_params is None:
        return None
    if sampling_params.logprobs is not None:
        return "it asks for logprobs (the answer is the head's argmax, one burst)"
    if getattr(sampling_params, "structured_outputs", None) is not None:
        return "it asks for structured output"
    if sampling_params.ignore_eos or sampling_params.min_tokens > spec["grid_size"] + 1:
        return "it asks to keep generating past the answer (ignore_eos / min_tokens)"
    return None


# --------------------------------------------------------------------------- #
# The burst
# --------------------------------------------------------------------------- #
def burst_tokens(
    answer_ids: torch.Tensor, eos_token_id: int | None, max_new_tokens: int
) -> torch.Tensor:
    """The tokens one loop request emits: its answer, then EOS, capped to fit."""
    tokens = answer_ids
    if eos_token_id is not None:
        tokens = torch.cat([tokens, tokens.new_tensor([eos_token_id])])
    return tokens[: max(1, max_new_tokens)]


def merge_bursts(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    rows: list[int],
    bursts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlay each loop row's burst on the ordinary sampler's output.

    ``sampled`` is ``[num_reqs, 1]`` (one ordinary token per row) and
    ``num_sampled`` ``[num_reqs]``. Returns a ``[num_reqs, W]`` matrix, ``W``
    the longest burst, and the new per-row counts. Ordinary rows keep their
    token in column 0 and their count; the rest of their row is padding the
    runner never reads (it trims each row to ``num_sampled``).
    """
    width = max([sampled.shape[1], *(b.numel() for b in bursts)])
    out = sampled.new_zeros((sampled.shape[0], width))
    out[:, : sampled.shape[1]] = sampled
    counts = num_sampled.clone()
    for row, burst in zip(rows, bursts):
        out[row, : burst.numel()] = burst.to(device=out.device, dtype=out.dtype)
        counts[row] = burst.numel()
    return out, counts


# --------------------------------------------------------------------------- #
# Engine settings (a per-architecture config hook, registered by register())
# --------------------------------------------------------------------------- #
def has_loop_slots(hf_config) -> bool:
    return "loop" in (getattr(hf_config, "adapter_kinds", None) or [])


def configure_engine_for_loop_slots(vllm_config) -> None:
    """Engine settings a loop checkpoint needs; vLLM runs this per architecture.

    * The V2 model runner: only it calls the model's sampler hook. Without it
      a loop request would quietly decode ordinary tokens, so this raises.
    * No async scheduling: the async scheduler counts one output placeholder
      per request per step and asserts when a step returns more.
    * No chunked prefill and no prefix caching: the head reads the grid's
      hidden states from the forward that schedules the marker, so the whole
      prompt must be computed in that one forward.
    """
    if not has_loop_slots(vllm_config.model_config.hf_config):
        return
    if not vllm_config.use_v2_model_runner:
        raise ValueError(
            "Loop slots need vLLM's V2 model runner (set VLLM_USE_V2_MODEL_RUNNER=1, "
            "and drop settings it does not support)."
        )
    if vllm_config.speculative_config is not None:
        raise ValueError("Loop slots do not support speculative decoding.")
    sched = vllm_config.scheduler_config
    if sched.async_scheduling:
        raise ValueError(
            "Loop slots do not support async scheduling yet; serve with "
            "--no-async-scheduling (it is off by default for loop checkpoints)."
        )
    sched.async_scheduling = False
    if sched.enable_chunked_prefill:
        sched.enable_chunked_prefill = False
        max_len = vllm_config.model_config.max_model_len
        if sched.max_num_batched_tokens < max_len:
            sched.max_num_batched_tokens = max_len
        logger.info(
            "Loop slots: chunked prefill off (max_num_batched_tokens=%d), so each "
            "loop request's prompt is computed in one forward.",
            sched.max_num_batched_tokens,
        )
    if vllm_config.cache_config.enable_prefix_caching:
        vllm_config.cache_config.enable_prefix_caching = False
        logger.info(
            "Loop slots: prefix caching off (a cached grid has no hidden states)."
        )


# --------------------------------------------------------------------------- #
# The sampler and the model state
# --------------------------------------------------------------------------- #
class LoopBurstSampler:
    """V2 sampler wrapper that answers loop requests with one token burst.

    Every attribute the runner reads besides ``__call__`` (``add_request``,
    ``apply_staged_writes``, ``sampling_states``, ``penalties_state``, ...)
    is the wrapped sampler's.
    """

    def __init__(self, sampler, model, state):
        self._sampler = sampler
        self._model = model
        self._state = state

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def __call__(self, logits, input_batch):
        out = self._sampler(logits, input_batch)
        rows = self._state.loop_rows(input_batch)
        if not rows:
            return out
        # One logits row per request (no drafts), so a request's row is its
        # last scheduled token: for a loop prefill, the marker.
        markers = input_batch.logits_indices[: input_batch.num_reqs].tolist()
        by_slot: dict[int, list[int]] = {}
        for i, (row, req) in enumerate(rows):
            by_slot.setdefault(req.slot, []).append(i)
        answers: list[torch.Tensor | None] = [None] * len(rows)
        for slot, members in by_slot.items():
            solved = self._model.solve_loop_requests(
                slot, [markers[rows[i][0]] for i in members]
            )
            for i, ans in zip(members, solved):
                answers[i] = ans
        bursts = [
            burst_tokens(ans, req.eos_token_id, req.max_new_tokens)
            for ans, (_, req) in zip(answers, rows)
        ]
        sampled, num_sampled = merge_bursts(
            out.sampled_token_ids, out.num_sampled, [r for r, _ in rows], bursts
        )
        out.sampled_token_ids = sampled
        out.num_sampled = num_sampled
        # num_rejected stays 0: the runner advances num_computed_tokens by
        # query_len - num_rejected, the tokens this step actually computed.
        return out


class LoopSlotModelState(DefaultModelState):
    """V2 model state of a checkpoint with loop slots.

    Records which requests are loop requests as they are added, and supplies
    :class:`LoopBurstSampler`. Everything else is vLLM's default state.
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._specs = vllm_config.model_config.hf_config.loop_specs
        self._marker_to_slot = model.model.loop_marker_to_slot()
        self._requests: dict[int, LoopRequest] = {}
        self._req_index: dict[str, int] = {}

    def custom_sampler(self, sampler):
        return LoopBurstSampler(sampler, self.model.model, self), None

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        self._requests.pop(req_index, None)
        prompt = new_req_data.prompt_token_ids
        slot = loop_request_slot(prompt, self._marker_to_slot)
        if slot is None:
            return
        params = new_req_data.sampling_params
        spec = self._specs[slot - 1]
        problem = loop_request_problem(spec, len(prompt), params)
        if problem is not None:
            logger.warning(
                "Loop request %s decodes as an ordinary request: %s.",
                new_req_data.req_id,
                problem,
            )
            return
        self._req_index[new_req_data.req_id] = req_index
        max_tokens = None if params is None else params.max_tokens
        self._requests[req_index] = LoopRequest(
            slot=slot,
            eos_token_id=None if params is None else params.eos_token_id,
            max_new_tokens=spec["grid_size"] + 1 if max_tokens is None else max_tokens,
        )

    def remove_request(self, req_id: str) -> None:
        idx = self._req_index.pop(req_id, None)
        if idx is not None:
            self._requests.pop(idx, None)
        super().remove_request(req_id)

    def loop_rows(self, input_batch) -> list[tuple[int, LoopRequest]]:
        """The batch rows whose loop request computes its whole prompt this step.

        With chunked prefill and prefix caching off (configure_engine_for_loop_slots)
        a request's prompt is scheduled in one piece, so that is the step
        whose forward holds the grid. A preempted request is re-prefilled
        with its output appended; its marker is no longer last, and it is not
        answered again.
        """
        if not self._requests:
            return []
        rows = []
        n = input_batch.num_reqs
        for row, idx in enumerate(input_batch.idx_mapping_np[:n]):
            req = self._requests.get(int(idx))
            if req is None:
                continue
            computed = int(input_batch.num_computed_tokens_np[row])
            scheduled = int(input_batch.num_scheduled_tokens[row])
            prefill = int(input_batch.prefill_len_np[row])
            if computed >= prefill:
                continue  # decoding: this request's answer was emitted already
            if computed > 0 or scheduled < prefill:
                logger.warning(
                    "Loop request in row %d: its prompt is not computed in one "
                    "forward (computed %d, scheduled %d of %d tokens); it "
                    "decodes as an ordinary request.",
                    row,
                    computed,
                    scheduled,
                    prefill,
                )
                self._requests.pop(int(idx), None)
                continue
            rows.append((row, req))
        return rows
