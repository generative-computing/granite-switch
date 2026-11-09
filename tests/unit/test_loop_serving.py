# SPDX-License-Identifier: Apache-2.0
"""The backend-agnostic half of serving a loop slot's answer as one token block.

vLLM serves a loop request in the step that completes its prompt: the grid's
read-layer states are captured as the prompt is prefilled (possibly over
several chunks), the head solves each request on its own, and the request's
output for that step is the whole answer. These tests cover the parts that need
no vLLM (``granite_switch.loop_serving`` and ``LoopHead.solve_rows``); the
engine wiring runs on GPU in ``tests/vllm/test_loop_serving.py``.

CPU, random weights.
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf.core.loop import LoopHead
from granite_switch.loop_serving import (
    answer_logprobs,
    answer_token_ids,
    find_loop_request,
    loop_markers,
    plan_grid_writes,
)

GRID = 9
OUTPUT_IDS = list(range(20, 29))


def _head(seed=0):
    torch.manual_seed(seed)
    head = LoopHead(
        hidden_size=16,
        width=32,
        num_layers=2,
        num_heads=4,
        expansion=4,
        grid_size=GRID,
        num_outputs=len(OUTPUT_IDS),
        prefix_len=4,
        init_std=1.0,
    )
    head.reset_parameters()
    return head.eval()


def _spec():
    return dict(grid_size=GRID, end_offset=2, output_token_ids=OUTPUT_IDS)


SOLVE = dict(max_iter=60, fp_thresh=1e-3, decay=0.9, patience=5)


# --------------------------------------------------------------------------- #
# Recognizing a loop request
# --------------------------------------------------------------------------- #


def _config():
    kinds = ["lora", "probe", "loop"]
    spec = dict(
        width=32,
        num_layers=2,
        num_heads=4,
        expansion=4,
        grid_size=GRID,
        prefix_len=4,
        conv_kernel_size=3,
        alpha_1_init=0.75,
        alpha_2_init=0.25,
        rope_theta=10000.0,
        init_std=1.0,
        output_token_ids=OUTPUT_IDS,
        read_layer=None,
        max_iter=20,
        fp_thresh=0.1,
        stepsize_decay=0.9,
        decay_patience=5,
        end_offset=2,
    )
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=3,
        adapter_token_ids=[249, 250, 251],
        adapter_substitute_token_ids=[1, 1, 1],
        adapter_names=["a1", "a2", "a3"],
        adapter_kinds=kinds,
        probe_label_token_ids=[None, [100, 200], None],
        loop_specs=[None, None, spec],
        max_lora_rank=4,
        adapter_ranks=[4, 4, 4],
        switch_head_dim=16,
    )


def test_markers_map_loop_control_tokens_to_their_slot():
    assert loop_markers(_config()) == {251: 3}


def test_a_request_ending_in_the_marker_is_a_loop_request():
    specs = _config().loop_specs
    prompt = [10, 11, *range(30, 30 + GRID), 12, 13, 251]
    req = find_loop_request(prompt, {251: 3}, specs, seed=7, eos_token_id=99)
    assert (req.slot, req.grid_start, req.grid_size) == (3, 2, GRID)
    assert (req.seed, req.eos_token_id, req.emitted) == (7, 99, False)


@pytest.mark.parametrize(
    "prompt",
    [
        [10, 11, *range(30, 30 + GRID), 12, 13, 251, 10],  # marker not last
        [10, 11, 12, 251],  # too short for the grid
        [10, 11, *range(30, 30 + GRID), 12, 13, 250],  # a probe marker
        [],
    ],
)
def test_anything_else_is_served_as_an_ordinary_request(prompt):
    assert find_loop_request(prompt, {251: 3}, _config().loop_specs) is None


def test_no_seed_means_seed_zero():
    prompt = [*range(30, 30 + GRID), 12, 13, 251]
    assert find_loop_request(prompt, {251: 3}, _config().loop_specs).seed == 0


# --------------------------------------------------------------------------- #
# Where each prefilled grid token's state goes
# --------------------------------------------------------------------------- #


def _req(grid_start=5):
    prompt = [0] * grid_start + list(range(30, 30 + GRID)) + [12, 13, 251]
    return find_loop_request(prompt, {251: 3}, _config().loop_specs)


def test_one_forward_holding_the_whole_prompt_copies_every_grid_token():
    req = _req(grid_start=5)
    # Buffer row 2, nothing computed yet, 17 tokens scheduled from flat token 40.
    tokens, dest = plan_grid_writes([(2, req, 0, 17, 40)], GRID)
    assert tokens == list(range(45, 45 + GRID))
    assert dest == list(range(2 * GRID, 3 * GRID))


def test_chunks_split_the_grid_and_together_cover_it_once():
    req = _req(grid_start=5)
    seen = []
    for computed, scheduled in [(0, 8), (8, 4), (12, 5)]:
        tokens, dest = plan_grid_writes([(0, req, computed, scheduled, 100)], GRID)
        # Each planned token is the right sequence position of this chunk.
        assert [t - 100 + computed for t in tokens] == [5 + d for d in dest]
        seen += dest
    assert sorted(seen) == list(range(GRID))


def test_several_requests_in_one_batch_go_to_their_own_rows():
    a, b = _req(grid_start=0), _req(grid_start=3)
    tokens, dest = plan_grid_writes([(0, a, 0, 12, 0), (5, b, 0, 15, 12)], GRID)
    assert tokens == list(range(GRID)) + list(range(15, 15 + GRID))
    assert dest == list(range(GRID)) + list(range(5 * GRID, 6 * GRID))


def test_a_chunk_without_grid_tokens_plans_nothing():
    req = _req(grid_start=5)
    assert plan_grid_writes([(0, req, 0, 5, 0)], GRID) == ([], [])
    assert plan_grid_writes([(0, req, 14, 3, 0)], GRID) == ([], [])


# --------------------------------------------------------------------------- #
# The answer's tokens and log-probabilities
# --------------------------------------------------------------------------- #


def test_answer_is_each_cells_best_output_token_then_eos():
    scores = torch.full((GRID, len(OUTPUT_IDS)), -1.0)
    best = [3, 0, 8, 1, 1, 2, 7, 5, 4]
    scores[torch.arange(GRID), best] = 2.0
    ids = answer_token_ids(scores, OUTPUT_IDS, eos_token_id=99, limit=100)
    assert ids == [OUTPUT_IDS[i] for i in best] + [99]


def test_answer_is_cut_at_the_limit_and_eos_is_optional():
    scores = torch.randn(GRID, len(OUTPUT_IDS))
    assert len(answer_token_ids(scores, OUTPUT_IDS, 99, limit=4)) == 4
    assert len(answer_token_ids(scores, OUTPUT_IDS, None, limit=100)) == GRID


def test_answer_logprobs_match_the_hf_rows_log_softmax():
    """A cell's distribution is the HF backend's rewritten row: -inf off the
    output tokens, the head's centered scores on them."""
    scores = torch.randn(GRID, len(OUTPUT_IDS))
    ids, vals, ranks = answer_logprobs(scores, OUTPUT_IDS, 99, num_logprobs=3)
    assert ids.shape == (GRID + 1, 4) and vals.shape == (GRID + 1, 4)
    expected = torch.log_softmax(scores, -1)
    best = expected.argmax(-1)
    assert ids[:GRID, 0].tolist() == [OUTPUT_IDS[i] for i in best]
    torch.testing.assert_close(vals[:GRID, 0], expected.max(-1).values)
    torch.testing.assert_close(vals[:GRID, 1:], expected.topk(3, -1).values)
    assert ranks.tolist() == [1] * (GRID + 1)
    # The forced end of text: certain, rank 1.
    assert ids[GRID, 0].item() == 99 and vals[GRID, 0].item() == 0.0


def test_answer_logprobs_in_logits_mode_are_the_centered_scores():
    scores = torch.randn(GRID, len(OUTPUT_IDS))
    _, vals, _ = answer_logprobs(scores, OUTPUT_IDS, None, 0, logits_mode=True)
    assert vals.shape == (GRID, 1)
    assert torch.all(vals == 0)  # the best score, centered


# --------------------------------------------------------------------------- #
# Solving each request on its own
# --------------------------------------------------------------------------- #


def test_a_single_row_solves_as_the_batch_solver_does():
    """With one row, per-row stopping is the batch rule: same answer, same
    iteration count, when both start from the same seeded draw."""
    head = _head()
    read = torch.randn(1, GRID, 16)
    gen = torch.Generator().manual_seed(11)
    want, want_state = head.solve(read, generator=gen, **SOLVE)
    got, got_state = head.solve_rows(read, seeds=[11], **SOLVE)
    torch.testing.assert_close(got, want)
    assert got_state["iters"].tolist() == want_state["iters"].tolist()


def test_a_rows_answer_does_not_depend_on_its_batch():
    """Checked in float64: the logic is per row, while float32 matmuls round
    differently at different batch sizes, and a row that never converges
    amplifies that rounding over hundreds of updates."""
    head = _head().double()
    torch.manual_seed(1)
    reads = torch.randn(4, GRID, 16, dtype=torch.float64)
    solve = dict(max_iter=300, fp_thresh=1e-2, decay=0.9, patience=5)
    together, state = head.solve_rows(reads, seeds=[1, 2, 3, 4], **solve)
    converged = state["residual"] < solve["fp_thresh"]
    assert converged.any() and not converged.all()  # both kinds of row
    for r in range(4):
        alone, alone_state = head.solve_rows(reads[r : r + 1], seeds=[r + 1], **solve)
        torch.testing.assert_close(together[r], alone[0], atol=1e-12, rtol=0)
        assert state["iters"][r].item() == alone_state["iters"][0].item()


def test_the_same_seed_gives_the_same_answer_and_another_seed_starts_elsewhere():
    head = _head()
    read = torch.randn(1, GRID, 16)
    a, _ = head.solve_rows(
        read, seeds=[5], max_iter=3, fp_thresh=0.0, decay=0.9, patience=5
    )
    b, _ = head.solve_rows(
        read, seeds=[5], max_iter=3, fp_thresh=0.0, decay=0.9, patience=5
    )
    c, _ = head.solve_rows(
        read, seeds=[6], max_iter=3, fp_thresh=0.0, decay=0.9, patience=5
    )
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.allclose(a, c)


def test_rows_stop_on_their_own_and_respect_min_iter_and_the_budget():
    head = _head()
    reads = torch.randn(2, GRID, 16)
    _, state = head.solve_rows(
        reads, seeds=[1, 2], max_iter=7, fp_thresh=0.0, decay=0.9, patience=5
    )
    assert state["iters"].tolist() == [7, 7]  # never converge: the budget
    _, state = head.solve_rows(
        reads,
        seeds=[1, 2],
        max_iter=50,
        fp_thresh=10.0,
        decay=0.9,
        patience=5,
        min_iter=4,
    )
    assert state["iters"].tolist() == [4, 4]  # converge at once: min_iter


def test_the_host_check_interval_does_not_change_the_answer():
    head = _head()
    reads = torch.randn(2, GRID, 16)
    a, sa = head.solve_rows(reads, seeds=[1, 2], check_every=1, **SOLVE)
    b, sb = head.solve_rows(reads, seeds=[1, 2], check_every=32, **SOLVE)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert sa["iters"].tolist() == sb["iters"].tolist()


# --------------------------------------------------------------------------- #
# The vLLM config hook (runs for every Granite Switch checkpoint)
# --------------------------------------------------------------------------- #


def test_config_hook_only_touches_checkpoints_with_a_loop_slot():
    """vLLM calls both hook methods on every Granite Switch checkpoint; only a
    loop slot gets the loop scheduler, and a set scheduler is kept."""
    loop_config = pytest.importorskip("granite_switch.vllm.loop_config")
    from types import SimpleNamespace

    hook = loop_config.GraniteSwitchConfigHook

    def vllm_config(kinds, scheduler_cls=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(adapter_kinds=kinds)
            ),
            scheduler_config=SimpleNamespace(scheduler_cls=scheduler_cls),
        )

    plain = vllm_config(["lora", "probe"])
    hook.verify_and_update_config(plain)
    hook.verify_and_update_model_config(plain.model_config)
    assert plain.scheduler_config.scheduler_cls is None

    loop = vllm_config(["lora", "loop"])
    hook.verify_and_update_config(loop)
    assert loop.scheduler_config.scheduler_cls == loop_config.LOOP_SCHEDULER

    custom = vllm_config(["loop"], scheduler_cls="my.Scheduler")
    hook.verify_and_update_config(custom)
    assert custom.scheduler_config.scheduler_cls == "my.Scheduler"
