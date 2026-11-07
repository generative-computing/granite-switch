# SPDX-License-Identifier: Apache-2.0
"""Loop-slot tests for GraniteSwitchForCausalLM (HF backend).

A loop slot is a sibling of a probe slot: it shares the adapter index, the
control token (which must be the request's last token) and the switch's split
of head positions out of the LoRA stream. Where a probe scores a few labels at
its marker, the loop head runs a small fixed-point transformer over the
``grid_size`` tokens that end ``end_offset`` tokens before the marker, read at
the slot's read layer, and rewrites each grid row's logits: ``-inf`` except at
the spec's ``output_token_ids``.

CPU, random weights, no checkpoint — mirrors tests/hf/test_probe.py.
"""

import math

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

GRID = 9
OUTPUT_IDS = list(range(20, 29))
END_OFFSET = 2


def _loop_spec(**overrides):
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
        end_offset=END_OFFSET,
    )
    spec.update(overrides)
    return spec


def _loop_config(adapter_kinds=("lora", "probe", "loop"), **spec_overrides):
    kinds = list(adapter_kinds)
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 2 switch cache layers + 2 decoder layers
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=len(kinds),
        adapter_token_ids=[249, 250, 251][-len(kinds) :],
        adapter_substitute_token_ids=[1] * len(kinds),
        adapter_names=[f"adapter_{i + 1}" for i in range(len(kinds))],
        adapter_kinds=kinds,
        probe_label_token_ids=[[100, 200] if k == "probe" else None for k in kinds],
        loop_specs=[
            _loop_spec(**spec_overrides) if k == "loop" else None for k in kinds
        ],
        max_lora_rank=4,
        adapter_ranks=[4] * len(kinds),
        switch_head_dim=16,
    )


def _model(config):
    model = GraniteSwitchForCausalLM(config).eval()
    model.model.adapter_token_ids.data = torch.tensor(
        config.adapter_token_ids, dtype=torch.long
    )
    return model


def _loop_request(marker=251, prefix=(10, 11)):
    """``prefix + grid + 2 closing tokens + marker`` (the template's layout)."""
    grid = list(range(30, 30 + GRID))
    return [*prefix, *grid, 12, 13, marker], len(prefix)


def test_loop_heads_built_only_when_a_loop_slot_is_present():
    assert _model(_loop_config(("lora", "probe"))).model.loop_heads is None
    model = _model(_loop_config())
    assert list(model.model.loop_heads.heads) == ["2"]  # slot 3 -> key "2"
    head = model.model.loop_heads.head(3)
    assert head.out_proj.out_features == len(OUTPUT_IDS)


def test_loop_head_keeps_its_own_initialization():
    """post_init must not redraw the head at the base model's init scale."""
    head = _model(_loop_config()).model.loop_heads.head(3)
    # LeCun normal: std 1/sqrt(fan_in); the base model's init would be 0.02.
    expected = 1.0 / math.sqrt(head.in_proj.in_features)
    assert abs(head.in_proj.weight.std().item() - expected) < 0.3 * expected
    assert torch.all(head.q_head.weight == 0)
    assert torch.all(head.q_head.bias == -5)


def test_loop_rewrites_exactly_the_grid_rows():
    model = _model(_loop_config())
    ids, start = _loop_request()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))

    logits = out.logits[0]
    other = [t for t in range(model.config.vocab_size) if t not in OUTPUT_IDS]
    grid_rows = logits[start : start + GRID]
    assert torch.isinf(grid_rows[:, other]).all()
    assert torch.isfinite(grid_rows[:, OUTPUT_IDS]).all()
    # Every grid row's best score is 0 (shifted so bf16 logits keep the argmax).
    torch.testing.assert_close(
        grid_rows[:, OUTPUT_IDS].amax(-1), torch.zeros(GRID), atol=0, rtol=0
    )
    # Rows before and after the grid are the base model's.
    assert torch.isfinite(logits[:start]).all()
    assert torch.isfinite(logits[start + GRID :]).all()


def test_loop_logits_are_the_heads_scores():
    """The grid rows carry the head's own solve on the read layer's states."""
    model = _model(_loop_config())
    ids, start = _loop_request()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        grids, head_logits, _ = model.model._last_loop_logits
        rows, cols = grids[3]
        assert rows.tolist() == [0]
        assert cols.tolist() == [list(range(start, start + GRID))]
        expected = head_logits[3].float()
        expected = expected - expected.amax(-1, keepdim=True)
    torch.testing.assert_close(
        out.logits[0, start : start + GRID][:, OUTPUT_IDS], expected[0]
    )


def test_non_loop_requests_are_untouched():
    model = _model(_loop_config())
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[10, 20, 30, 40, 50, 60]]))
    assert torch.isfinite(out.logits).all()
    assert model.model._last_loop_logits is None


def test_probe_still_works_next_to_a_loop_slot():
    model = _model(_loop_config())
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[10, 20, 30, 250]]))
    finite = torch.isfinite(out.logits[0, -1]).nonzero().flatten().tolist()
    assert finite == [100, 200]
    assert model.model._last_loop_logits is None


def test_loop_positions_leave_the_lora_stream():
    model = _model(_loop_config())
    ids, _ = _loop_request()
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    assert not model.model._last_lora_indices.any()
    assert model.model._last_probe_indices[0, -1].item() == 3


def test_marker_that_is_not_the_last_token_raises():
    model = _model(_loop_config())
    ids, _ = _loop_request()
    with pytest.raises(RuntimeError, match="not their last token"):
        with torch.no_grad():
            model(input_ids=torch.tensor([[*ids, 10]]))


def test_too_short_request_raises():
    model = _model(_loop_config())
    with pytest.raises(RuntimeError, match="grid tokens"):
        with torch.no_grad():
            model(input_ids=torch.tensor([[10, 20, 30, 251]]))


def test_mixed_batch_rewrites_only_the_loop_row():
    model = _model(_loop_config())
    ids, start = _loop_request()
    plain = [10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 12, 13, 14, 15]
    assert len(plain) == len(ids)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([plain, ids]))
    assert torch.isfinite(out.logits[0]).all()
    assert torch.isinf(out.logits[1, start, 0])


@pytest.mark.parametrize("read_layer, state_index", [(0, 1), (None, -1), (-1, 0)])
def test_read_layer_selects_the_state_the_head_solves(read_layer, state_index):
    """The head solves the configured layer's grid states (None: the final norm,
    -1: the input embeddings, which hidden_states[0] holds)."""
    model = _model(_loop_config(read_layer=read_layer))
    head = model.model.loop_heads.head(3)
    seen = []
    original = head.solve

    def recording_solve(read, *args, **kwargs):
        seen.append(read.detach().clone())
        return original(read, *args, **kwargs)

    head.solve = recording_solve
    ids, start = _loop_request()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    # hidden_states[i] is the input to decoder layer i; layer 0's output is i = 1.
    expected = out.hidden_states[state_index][:, start : start + GRID]
    other = out.hidden_states[-1 if state_index == 1 else 1][:, start : start + GRID]
    torch.testing.assert_close(seen[0], expected)
    assert not torch.allclose(seen[0], other)


def test_solver_stops_at_the_budget_and_counts_iterations():
    model = _model(_loop_config(max_iter=7, fp_thresh=0.0))
    ids, _ = _loop_request()
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    assert model.model._last_loop_iters.tolist() == [7]


def test_bf16_solve_dtype_runs_and_keeps_fp32_weights():
    model = _model(_loop_config(solve_dtype="bfloat16"))
    ids, start = _loop_request()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
    assert torch.isfinite(out.logits[0, start : start + GRID][:, OUTPUT_IDS]).all()
    assert {p.dtype for p in model.model.loop_heads.parameters()} == {torch.float32}


def test_save_and_bf16_reload_keep_the_head_in_fp32(tmp_path):
    model = _model(_loop_config())
    model.save_pretrained(tmp_path)
    reloaded = GraniteSwitchForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    before = model.model.loop_heads.state_dict()
    after = reloaded.model.loop_heads.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        assert after[key].dtype == torch.float32, key
        torch.testing.assert_close(after[key], before[key].float(), atol=0, rtol=0)


def test_head_trains_only_its_own_parameters():
    model = _model(_loop_config())
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("model.loop_heads."))
    head = model.model.loop_heads.head(3)
    ids, start = _loop_request()
    with torch.no_grad():
        read = model.model(
            input_ids=torch.tensor([ids[:-1]]), use_cache=False
        ).last_hidden_state[:, start : start + GRID]
    x = head.inject(read)
    state = head.init_state(x)
    for _ in range(3):
        state = head.step(state, x, 0.9, 5, 0.1)
    loss = (
        head.logits(state["z"]).logsumexp(-1).mean()
        + head.halt_logit(state["z"]).mean()
    )
    loss.backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad
    assert all(n.startswith("model.loop_heads.") for n in with_grad)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"read_layer": 5}, "read_layer"),
        ({"read_layer": -2}, "read_layer"),
        ({"output_token_ids": []}, "output_token_ids"),
    ],
)
def test_config_rejects_bad_loop_specs(bad, match):
    with pytest.raises(ValueError, match=match):
        _loop_config(**bad)


def test_config_requires_specs_for_loop_slots():
    with pytest.raises(ValueError, match="loop_specs"):
        GraniteSwitchConfig(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_adapters=1,
            adapter_token_ids=[251],
            adapter_substitute_token_ids=[1],
            adapter_names=["sudoku"],
            adapter_kinds=["loop"],
            max_lora_rank=4,
            adapter_ranks=[4],
            switch_head_dim=16,
        )
