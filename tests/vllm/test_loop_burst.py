# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the loop-slot answer burst (granite_switch.vllm.loop_burst).

The sampler wrapper, the request bookkeeping and the engine settings are
exercised with stand-ins for vLLM's batch and sampler; serving end to end is
in ``test_loop_serving.py`` (GPU).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from granite_switch.vllm.loop_burst import (
    LoopBurstSampler,
    LoopRequest,
    LoopSlotModelState,
    burst_tokens,
    configure_engine_for_loop_slots,
    loop_request_problem,
    loop_request_slot,
    merge_bursts,
)

SPEC = {"grid_size": 4, "end_offset": 2}
EOS = 99


def _params(**kw):
    base = dict(
        logprobs=None,
        structured_outputs=None,
        ignore_eos=False,
        min_tokens=0,
        max_tokens=16,
        eos_token_id=EOS,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def test_a_loop_request_ends_with_its_marker():
    markers = {251: 3}
    assert loop_request_slot([1, 2, 251], markers) == 3
    assert loop_request_slot([1, 251, 2], markers) is None
    assert loop_request_slot([], markers) is None


def test_loop_request_needs_its_whole_grid():
    assert loop_request_problem(SPEC, 7, _params()) is None  # 4 + 2 + marker
    assert "fewer than" in loop_request_problem(SPEC, 6, _params())


@pytest.mark.parametrize(
    "bad, match",
    [
        (dict(logprobs=1), "logprobs"),
        (dict(structured_outputs=object()), "structured"),
        (dict(ignore_eos=True), "ignore_eos"),
        (dict(min_tokens=6), "min_tokens"),
    ],
)
def test_loop_request_names_what_a_burst_cannot_serve(bad, match):
    assert match in loop_request_problem(SPEC, 7, _params(**bad))


# --------------------------------------------------------------------------- #
# The burst
# --------------------------------------------------------------------------- #
def test_burst_is_the_answer_then_eos_capped_to_max_tokens():
    answer = torch.tensor([16, 17, 18, 19])
    assert burst_tokens(answer, EOS, 16).tolist() == [16, 17, 18, 19, EOS]
    assert burst_tokens(answer, EOS, 3).tolist() == [16, 17, 18]
    assert burst_tokens(answer, None, 16).tolist() == [16, 17, 18, 19]


def test_merge_keeps_ordinary_rows_and_widens_loop_rows():
    sampled = torch.tensor([[5], [6], [7]])
    num_sampled = torch.tensor([1, 0, 1])  # row 1 is a partial prefill
    out, counts = merge_bursts(
        sampled, num_sampled, [2], [torch.tensor([16, 17, 18, EOS])]
    )
    assert out.shape == (3, 4)
    assert out[0, 0] == 5 and out[1, 0] == 6
    assert out[2].tolist() == [16, 17, 18, EOS]
    assert counts.tolist() == [1, 0, 4]
    assert num_sampled.tolist() == [1, 0, 1]  # the input is not modified


# --------------------------------------------------------------------------- #
# The sampler wrapper
# --------------------------------------------------------------------------- #
class _Sampler:
    """Stands in for vLLM's V2 sampler: one token (the row index + 5) per row."""

    sampling_states = "states"

    def __call__(self, logits, input_batch):
        n = input_batch.num_reqs
        return SimpleNamespace(
            sampled_token_ids=torch.arange(5, 5 + n).view(-1, 1),
            num_sampled=torch.ones(n, dtype=torch.int32),
            num_rejected=torch.zeros(n, dtype=torch.int32),
        )

    def add_request(self, *a):
        return "added"


class _Model:
    def __init__(self):
        self.calls = []

    def solve_loop_requests(self, slot, markers):
        self.calls.append((slot, markers))
        return [torch.tensor([16, 17, 18, 19]) + 10 * i for i in range(len(markers))]


class _State:
    def __init__(self, rows):
        self.rows = rows

    def loop_rows(self, input_batch):
        return self.rows


def _batch(num_reqs=3, logits_indices=(10, 25, 40)):
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=0,
        logits_indices=torch.tensor(logits_indices),
    )


def test_sampler_answers_loop_rows_at_their_marker_and_leaves_others():
    model = _Model()
    req = LoopRequest(slot=3, eos_token_id=EOS, max_new_tokens=16)
    sampler = LoopBurstSampler(_Sampler(), model, _State([(1, req)]))
    out = sampler(None, _batch())
    assert model.calls == [(3, [25])]  # row 1's logits row is its marker
    assert out.num_sampled.tolist() == [1, 5, 1]
    assert out.sampled_token_ids[1].tolist() == [16, 17, 18, 19, EOS]
    assert out.sampled_token_ids[:, 0].tolist() == [5, 16, 7]
    assert out.num_rejected.tolist() == [0, 0, 0]


def test_sampler_solves_each_slot_once_for_all_its_requests():
    model = _Model()
    a = LoopRequest(slot=3, eos_token_id=EOS, max_new_tokens=16)
    b = LoopRequest(slot=4, eos_token_id=EOS, max_new_tokens=16)
    sampler = LoopBurstSampler(_Sampler(), model, _State([(0, a), (1, b), (2, a)]))
    out = sampler(None, _batch())
    assert model.calls == [(3, [10, 40]), (4, [25])]
    assert out.sampled_token_ids[0, :4].tolist() == [16, 17, 18, 19]
    assert out.sampled_token_ids[2, :4].tolist() == [26, 27, 28, 29]
    assert out.sampled_token_ids[1, :4].tolist() == [16, 17, 18, 19]
    assert out.num_sampled.tolist() == [5, 5, 5]


def test_sampler_without_loop_rows_returns_the_ordinary_output():
    sampler = LoopBurstSampler(_Sampler(), _Model(), _State([]))
    out = sampler(None, _batch())
    assert out.sampled_token_ids.shape == (3, 1)


def test_sampler_delegates_everything_else():
    sampler = LoopBurstSampler(_Sampler(), _Model(), _State([]))
    assert sampler.sampling_states == "states"
    assert sampler.add_request() == "added"


# --------------------------------------------------------------------------- #
# Which rows are loop requests this step (LoopSlotModelState.loop_rows)
# --------------------------------------------------------------------------- #
def _rows(requests, batch):
    state = SimpleNamespace(_requests=requests)
    return LoopSlotModelState.loop_rows(state, batch)


def _prefill_batch(computed, scheduled, prefill_len, idx):
    return SimpleNamespace(
        num_reqs=len(idx),
        idx_mapping_np=np.array(idx),
        num_computed_tokens_np=np.array(computed),
        num_scheduled_tokens=np.array(scheduled),
        prefill_len_np=np.array(prefill_len),
    )


def test_loop_rows_are_the_loop_requests_prefilled_this_step():
    req = LoopRequest(slot=3, eos_token_id=EOS, max_new_tokens=16)
    batch = _prefill_batch(
        computed=[0, 0, 7], scheduled=[7, 9, 1], prefill_len=[7, 9, 7], idx=[4, 2, 6]
    )
    # Slot 4 prefills its whole prompt now; slot 6 is decoding (already prefilled).
    assert _rows({4: req, 6: req}, batch) == [(0, req)]


def test_a_split_loop_prompt_falls_back_to_ordinary_decoding():
    req = LoopRequest(slot=3, eos_token_id=EOS, max_new_tokens=16)
    requests = {4: req}
    batch = _prefill_batch(computed=[3], scheduled=[4], prefill_len=[7], idx=[4])
    assert _rows(requests, batch) == []
    assert requests == {}  # not answered later either


# --------------------------------------------------------------------------- #
# Engine settings
# --------------------------------------------------------------------------- #
def _vllm_config(kinds=("lora", "loop"), async_scheduling=None, v2=True):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(adapter_kinds=list(kinds)), max_model_len=4096
        ),
        use_v2_model_runner=v2,
        speculative_config=None,
        scheduler_config=SimpleNamespace(
            async_scheduling=async_scheduling,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )


def test_engine_settings_for_a_loop_checkpoint():
    cfg = _vllm_config()
    configure_engine_for_loop_slots(cfg)
    assert cfg.scheduler_config.async_scheduling is False
    assert cfg.scheduler_config.enable_chunked_prefill is False
    assert cfg.scheduler_config.max_num_batched_tokens == 4096
    assert cfg.cache_config.enable_prefix_caching is False


def test_engine_settings_leave_other_checkpoints_alone():
    cfg = _vllm_config(kinds=("lora", "probe"))
    configure_engine_for_loop_slots(cfg)
    assert cfg.scheduler_config.async_scheduling is None
    assert cfg.scheduler_config.enable_chunked_prefill is True
    assert cfg.cache_config.enable_prefix_caching is True


@pytest.mark.parametrize(
    "kw, match",
    [(dict(v2=False), "V2 model runner"), (dict(async_scheduling=True), "async")],
)
def test_engine_settings_refuse_what_cannot_serve_a_burst(kw, match):
    with pytest.raises(ValueError, match=match):
        configure_engine_for_loop_slots(_vllm_config(**kw))
