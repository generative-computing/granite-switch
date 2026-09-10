# SPDX-License-Identifier: Apache-2.0
"""End-to-end compose tests for ``granitemoe`` bases.

``granitemoe`` is a *pure sparse* MoE: every layer has an expert bank and **no**
dense ``shared_mlp``.  That makes it the first supported base where
``shared_intermediate_size == 0`` (upstream's own encoding for "no shared MLP"),
so these tests exist mainly to pin down that the shared-MLP module, its LoRA
target groups, and the MLP forward path all disappear together — while the frozen
expert tensors transfer through untouched.

Under Shadow Residual the adapter stream always inherits the base stream's expert
routing, so these tests also pin that the frozen base stream is left exactly as a
lone ``block_sparse_moe`` call would leave it.

All tests run on CPU with random weights — no model download needed.
"""

import json

import pytest
import torch
from safetensors.torch import load_file

import granite_switch.hf  # noqa: F401 — registers AutoModel
from granite_switch.config import SWITCH_CACHE_LAYERS

# The synthetic base/adapter builders live in ``tests/shared`` because the vLLM
# TP suite needs the same composed checkpoint — see ``TestTPGraniteMoe`` in
# ``tests/vllm/test_tp_integration.py``.
from tests.shared.granitemoe_compose import (
    CONTROL_TOKEN_ID,
    CROSS_STREAM_RANK,
    HIDDEN,
    NUM_EXPERTS,
    NUM_LAYERS,
    TOP_K,
    VOCAB_SIZE,
    compose_granitemoe,
    create_lora_adapter,
    create_sr_adapter,
)

INPUT_IDS = torch.tensor([[10, 20, CONTROL_TOKEN_ID, 30, 40, 50, 60, 70]])
PAD_TOKEN_ID = 0

# Two rows of different real length, each carrying its own control token.  Batch
# size > 1 is what the eval harness actually runs (batch 16), and it is the only
# shape where the expert bank gathers tokens from more than one sequence into a
# single grouped matmul.
BATCH_ROWS = (
    [20, CONTROL_TOKEN_ID, 30, 40, 50, 60],
    [10, 20, CONTROL_TOKEN_ID, 30],
)


# ---------------------------------------------------------------------------
# Helpers: decode-path probes
# ---------------------------------------------------------------------------


def _full_prefill_logits(model, input_ids):
    """Single forward pass over the whole sequence.  Returns [B, seq, vocab]."""
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def _incremental_decode_logits(model, input_ids):
    """Feed tokens one at a time, accumulating the KV cache.

    The switch owns cache slot 0 and the decoder layers occupy 1..N, so this is
    what actually exercises that ``DynamicCache`` allocates ``num_hidden_layers
    = real + 1`` slots and that the switch re-reads its own cached control-token
    K/V on every step.  It also runs expert routing per single token rather than
    over a whole prefill, which is the decode-time shape.
    """
    seq_len = input_ids.shape[1]
    all_logits = []
    past_key_values = None

    with torch.no_grad():
        for i in range(seq_len):
            output = model(
                input_ids=input_ids[:, i : i + 1],
                past_key_values=past_key_values,
                cache_position=torch.tensor([i], dtype=torch.long),
                use_cache=True,
            )
            all_logits.append(output.logits)
            past_key_values = output.past_key_values

    return torch.cat(all_logits, dim=1)


# Sparse-MoE decode cannot be as tight as the dense path (1e-5/1e-4).  Experts
# gather their assigned tokens into one batched matmul, so a prefill of N tokens
# and N single-token steps reduce over different groupings — pure float ordering,
# independent of routing.  Observed worst case here: ~9e-5 absolute.  A real
# cache- or routing-misalignment bug drops the adapter entirely and diverges by
# O(1), which is why the argmax check below is the load-bearing assertion.
_MOE_DECODE_ATOL = 1e-3
_MOE_DECODE_RTOL = 1e-3


def _assert_prefill_matches_decode(model, input_ids):
    prefill = _full_prefill_logits(model, input_ids)
    incremental = _incremental_decode_logits(model, input_ids)

    torch.testing.assert_close(
        prefill.argmax(-1),
        incremental.argmax(-1),
        msg="greedy token choice diverged between prefill and decode",
    )
    torch.testing.assert_close(
        prefill,
        incremental,
        atol=_MOE_DECODE_ATOL,
        rtol=_MOE_DECODE_RTOL,
    )


def _left_pad(rows):
    """Left-pad ragged rows into ``(input_ids, attention_mask)``.

    Left padding is not cosmetic here: the last real token of every row must sit
    at the end of the tensor for greedy decode to continue from it, which is why
    the eval harness sets ``padding_side="left"``.
    """
    width = max(len(r) for r in rows)
    ids, mask = [], []
    for row in rows:
        gap = width - len(row)
        ids.append([PAD_TOKEN_ID] * gap + list(row))
        mask.append([0] * gap + [1] * len(row))
    return torch.tensor(ids), torch.tensor(mask)


def _assert_batch_matches_singles(model, rows=BATCH_ROWS, new_tokens=4):
    """Greedy-decode a ragged batch and each row alone; the rows must agree.

    Two independent things can break only at batch > 1 and would be invisible to
    every other test in this file, all of which are batch 1:

    * the switch resolves adapter indices per row, so a batched control-token
      scan that leaked across rows -- or that counted pad positions -- would
      route one row to base;
    * the expert bank gathers assigned tokens from the *whole* batch into one
      grouped matmul, so cross-row contamination lands in the MLP output rather
      than in the routing.

    Both failures are O(1) in the logits, so token equality is a sound gate at
    this size; the argmax is not near a tie with random weights and a 300-token
    vocabulary.
    """
    batched_ids, batched_mask = _left_pad(rows)

    def _gen(input_ids, attention_mask):
        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=PAD_TOKEN_ID,
            )

    batched = _gen(batched_ids, batched_mask)
    assert batched.shape == (len(rows), batched_ids.shape[1] + new_tokens)

    # Every row kept its adapter for every generated token.  Read this before
    # the per-row runs overwrite it.
    routes = model.model._last_adapter_indices
    for i in range(len(rows)):
        route = routes[i].tolist()
        assert route and all(v == 1 for v in route), (
            f"row {i} lost its adapter during batched decode: {route}"
        )

    for i, row in enumerate(rows):
        single = torch.tensor([row])
        alone = _gen(single, torch.ones_like(single))
        torch.testing.assert_close(
            batched[i, -new_tokens:],
            alone[0, -new_tokens:],
            msg=lambda m, i=i: f"row {i} decoded differently in a batch: {m}",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _compose(tmp_path_factory, name: str, make_adapter):
    return compose_granitemoe(
        name=name,
        base_path=tmp_path_factory.mktemp(f"{name}_base"),
        adapter_path=tmp_path_factory.mktemp(f"{name}_adapter"),
        output_dir=tmp_path_factory.mktemp(f"{name}_out") / "moe-model",
        make_adapter=make_adapter,
    )


@pytest.fixture(scope="module")
def lora_build(tmp_path_factory):
    """Attention-only LoRA composed onto a pure-sparse granitemoe base."""
    return _compose(tmp_path_factory, "lora", create_lora_adapter)


@pytest.fixture(scope="module")
def sr_build(tmp_path_factory):
    """SR over a pure-sparse base: the adapter stream inherits the base's routing."""
    return _compose(tmp_path_factory, "sr", create_sr_adapter)


# ---------------------------------------------------------------------------
# Attention-only LoRA over a pure sparse MoE base
# ---------------------------------------------------------------------------


class TestGraniteMoeLoRACompose:
    """A ``granitemoe`` compose must drop the shared MLP entirely."""

    def test_config_records_no_shared_mlp(self, lora_build):
        config = json.loads((lora_build.output_dir / "config.json").read_text())

        assert config["shared_intermediate_size"] == 0
        assert config["num_local_experts"] == NUM_EXPERTS
        assert config["num_experts_per_tok"] == TOP_K
        assert config["dual_stream"] is False

    def test_shared_mlp_lora_groups_omitted(self, lora_build):
        """With no shared MLP there is nothing for its LoRA groups to attach to."""
        config = json.loads((lora_build.output_dir / "config.json").read_text())

        assert set(config["lora_target_modules"]) == {"qkv_proj", "o_proj"}

    def test_layers_have_experts_and_no_shared_mlp(self, lora_build):
        for layer in lora_build.model.model.layers:
            assert layer.has_experts
            assert not layer.has_shared_mlp
            assert layer.shared_mlp is None
            assert hasattr(layer, "block_sparse_moe")

    def test_no_zero_width_shared_params(self, lora_build):
        """A gated-off shared MLP must leave no ``[0, H]`` tensors behind."""
        for name, param in lora_build.model.named_parameters():
            assert "shared_mlp" not in name, f"unexpected shared_mlp param {name}"
            assert 0 not in param.shape, (
                f"zero-width param {name}: {tuple(param.shape)}"
            )

    def test_expert_weights_transfer_by_identity(self, lora_build):
        """Frozen expert tensors are named identically, so they map 1:1."""
        base_sd = load_file(str(lora_build.base_path / "model.safetensors"))

        for i, layer in enumerate(lora_build.model.model.layers):
            moe, src = layer.block_sparse_moe, f"model.layers.{i}.block_sparse_moe"
            torch.testing.assert_close(
                moe.input_linear.weight, base_sd[f"{src}.input_linear.weight"]
            )
            torch.testing.assert_close(
                moe.output_linear.weight, base_sd[f"{src}.output_linear.weight"]
            )
            torch.testing.assert_close(
                moe.router.layer.weight, base_sd[f"{src}.router.layer.weight"]
            )

    def test_switch_layer_reserved_at_front(self, lora_build):
        """The switch reserves its cache slots at the front, so
        ``num_hidden_layers`` grows by ``SWITCH_CACHE_LAYERS`` (MultiSwitch, the
        only engine, owns two: counting + memory)."""
        config = json.loads((lora_build.output_dir / "config.json").read_text())

        assert config["num_hidden_layers"] == NUM_LAYERS + SWITCH_CACHE_LAYERS
        assert len(lora_build.model.model.layers) == NUM_LAYERS

    def test_forward_and_roundtrip(self, lora_build):
        """Compose → save → load produces bit-exact logits."""
        from granite_switch.hf import load_model

        model = lora_build.model.eval()
        loaded = load_model(str(lora_build.output_dir)).eval()

        with torch.no_grad():
            out_built = model(input_ids=INPUT_IDS).logits
            out_loaded = loaded(input_ids=INPUT_IDS).logits

        assert out_built.shape == (1, INPUT_IDS.shape[1], VOCAB_SIZE)
        assert torch.isfinite(out_built).all()
        torch.testing.assert_close(out_built, out_loaded)

    def test_control_token_activates_adapter(self, lora_build):
        """The control token must change the output — the adapter really fires."""
        model = lora_build.model.eval()

        without = INPUT_IDS.clone()
        without[0, 2] = 30

        with torch.no_grad():
            out_ctrl = model(input_ids=INPUT_IDS).logits[:, -1]
            out_plain = model(input_ids=without).logits[:, -1]

        assert not torch.allclose(out_ctrl, out_plain)

    def test_zero_shared_intermediate_survives_reload(self, lora_build):
        """0 must round-trip, not be re-defaulted from ``intermediate_size``.

        Two independent places could resurrect a phantom dense MLP here: the
        composer only writes the key when it is present in ``config_kwargs``, and
        ``GraniteSwitchConfig`` re-derives it from ``intermediate_size`` when it
        is ``None``.  A falsy test in either would rebuild the module the base
        checkpoint has no weights for.
        """
        from granite_switch.hf import load_model

        loaded = load_model(str(lora_build.output_dir))

        assert loaded.config.shared_intermediate_size == 0
        for layer in loaded.model.layers:
            assert layer.shared_mlp is None
        assert not any("shared_mlp" in n for n, _ in loaded.named_parameters())


# ---------------------------------------------------------------------------
# Decode path: cache layout, adapter stickiness, batching, dtype
# ---------------------------------------------------------------------------


class TestGraniteMoeDecode:
    """The composed model must be usable for generation, not just one forward."""

    def test_prefill_matches_incremental_decode(self, lora_build):
        """Prefill and token-by-token decode must agree.

        If the switch's cache slot were misaligned with the 40+1 layer indices,
        or if it stopped seeing its own cached control token, every step after
        the control position would route to base and the logits would diverge
        wildly rather than by rounding.
        """
        _assert_prefill_matches_decode(lora_build.model.eval(), INPUT_IDS)

    def test_adapter_stays_active_across_generate(self, lora_build):
        """The adapter must still be selected on the last generated token.

        There is no ``prepare_inputs_for_generation`` override: stickiness rests
        entirely on the switch attending over its own cached control-token K/V.
        """
        model = lora_build.model.eval()

        with torch.no_grad():
            out = model.generate(
                input_ids=INPUT_IDS,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
            )

        assert out.shape[1] == INPUT_IDS.shape[1] + 4
        route = model.model._last_adapter_indices[0].tolist()
        assert route and all(v == 1 for v in route), (
            f"adapter dropped during decode: {route}"
        )

    def test_left_padded_batch_matches_unpadded(self, lora_build):
        """Left padding with an explicit mask is how the eval harness calls it.

        Batched left-padded greedy generation is the only shape the answerability
        eval ever uses, and nothing else in the suite covers it.
        """
        model = lora_build.model.eval()

        short = torch.tensor([[20, CONTROL_TOKEN_ID, 30, 40]])
        pad = torch.full((1, 4), PAD_TOKEN_ID, dtype=torch.long)
        padded = torch.cat([pad, short], dim=1)
        mask = torch.cat([torch.zeros_like(pad), torch.ones_like(short)], dim=1)

        with torch.no_grad():
            out_plain = model.generate(
                input_ids=short,
                attention_mask=torch.ones_like(short),
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=PAD_TOKEN_ID,
            )
            out_padded = model.generate(
                input_ids=padded,
                attention_mask=mask,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=PAD_TOKEN_ID,
            )

        torch.testing.assert_close(out_padded[:, -4:], out_plain[:, -4:])

    def test_ragged_batch_matches_single_rows(self, lora_build):
        """Batch size > 1 with rows of different length.

        The test above pads a batch of one, so it never puts two sequences in the
        same expert matmul.  This one does, which is the shape the eval harness
        runs at.
        """
        _assert_batch_matches_singles(lora_build.model.eval())

    def test_bf16_forward(self, lora_build):
        """The 20B build runs in bf16; the test suite otherwise only ever uses fp32."""
        model = lora_build.model.eval().to(torch.bfloat16)
        try:
            without = INPUT_IDS.clone()
            without[0, 2] = 30

            with torch.no_grad():
                out_ctrl = model(input_ids=INPUT_IDS).logits
                out_plain = model(input_ids=without).logits

            assert out_ctrl.dtype == torch.bfloat16
            assert torch.isfinite(out_ctrl).all()
            assert not torch.allclose(out_ctrl[:, -1], out_plain[:, -1])
        finally:
            model.to(torch.float32)


# ---------------------------------------------------------------------------
# Shadow Residual over a pure sparse MoE base
# ---------------------------------------------------------------------------


class TestGraniteMoeSR:
    """SR over a pure sparse MoE base.

    The base stream routes on its own hidden states, and the adapter stream reuses
    that partition *and* its gate scalars instead of routing on ``normed_adapt``.
    """

    def test_compose_and_config(self, sr_build):
        config = json.loads((sr_build.output_dir / "config.json").read_text())

        assert config["dual_stream"] is True
        assert config["cross_stream_rank"] == CROSS_STREAM_RANK
        assert config["shared_intermediate_size"] == 0

    def test_cross_stream_lora_transferred(self, sr_build):
        for layer in sr_build.model.model.layers:
            assert torch.all(layer.cross_stream.base_layer.weight == 0)
            assert layer.cross_stream.lora_A[0].abs().sum() > 0
            assert layer.cross_stream.lora_B[0].abs().sum() > 0

    def test_forward_and_roundtrip(self, sr_build):
        from granite_switch.hf import load_model

        model = sr_build.model.eval()
        loaded = load_model(str(sr_build.output_dir)).eval()

        with torch.no_grad():
            out_built = model(input_ids=INPUT_IDS).logits
            out_loaded = loaded(input_ids=INPUT_IDS).logits

        assert torch.isfinite(out_built).all()
        assert out_built.shape == (1, INPUT_IDS.shape[1], VOCAB_SIZE)
        torch.testing.assert_close(out_built, out_loaded)

    def test_route_apply_matches_upstream_moe(self, sr_build):
        """``_apply_experts(_route(x))`` must equal ``block_sparse_moe(x)`` exactly.

        Shared routing needs a seam between routing and expert application that
        upstream does not expose, so ``_apply_experts`` duplicates the second half
        of ``GraniteMoeHybridMoE.forward``.  This is the test that makes that
        duplication safe across the supported ``transformers`` range: if the
        upstream expert path ever changes, it fails here instead of drifting an
        eval score by a fraction of a point.
        """
        torch.manual_seed(3)
        hidden = torch.randn(2, 5, HIDDEN)

        for layer in sr_build.model.model.layers:
            with torch.no_grad():
                expected = layer.block_sparse_moe(hidden)
                actual = layer._apply_experts(hidden, layer._route(hidden))

            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_base_stream_is_unperturbed(self, sr_build):
        """Sharing must change only the adapter stream's MLP, never the frozen one.

        The base stream routes on its own hidden states and applies the experts
        under that same partition, so its MLP output must equal what a lone
        ``block_sparse_moe`` call would produce.  Observed directly: the base
        stream's MLP output is not reachable from the model output (only the
        adapter stream feeds the LM head), so wrap ``_mlp_block`` on the first
        layer and capture the call ``SRSwitchDecoderLayer.forward`` makes with
        ``adapter_indices=None``.
        """
        model = sr_build.model.eval()
        layer = model.model.layers[0]
        original = layer._mlp_block
        captured = []

        def _capture(hidden_states, adapter_indices, routing=None):
            out = original(hidden_states, adapter_indices, routing)
            if adapter_indices is None:
                captured.append((hidden_states.clone(), out.clone()))
            return out

        try:
            layer._mlp_block = _capture
            with torch.no_grad():
                model(input_ids=INPUT_IDS)
        finally:
            layer._mlp_block = original

        assert len(captured) == 1
        normed_base, mlp_base = captured[0]
        with torch.no_grad():
            expected = layer.block_sparse_moe(normed_base)
        torch.testing.assert_close(mlp_base, expected, atol=0.0, rtol=0.0)

    def test_prefill_matches_incremental_decode(self, sr_build):
        """Shared routing is recomputed per step, so decode must still line up.

        A single decode step routes one token where prefill routed the whole
        sequence — the expert groupings differ, hence the loosened tolerance in
        ``_assert_prefill_matches_decode``.
        """
        _assert_prefill_matches_decode(sr_build.model.eval(), INPUT_IDS)

    def test_adapter_stays_active_across_generate(self, sr_build):
        model = sr_build.model.eval()

        with torch.no_grad():
            model.generate(
                input_ids=INPUT_IDS,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
            )

        route = model.model._last_adapter_indices[0].tolist()
        assert route and all(v == 1 for v in route), (
            f"adapter dropped during decode: {route}"
        )

    def test_ragged_batch_matches_single_rows(self, sr_build):
        """Shared routing routes on the base half only, then reuses that partition.

        With more than one row in flight, the half it routes on is
        ``normed[: B*S]`` rather than ``normed[:S]``.  Getting that slice wrong
        gives the adapter stream another row's expert assignment, which no
        batch-1 test can distinguish from the correct one.
        """
        _assert_batch_matches_singles(sr_build.model.eval())
