# SPDX-License-Identifier: Apache-2.0
"""End-to-end compose tests for SR (Shadow Residual) adapters.

Creates mock SR adapters on disk (with cross_stream weights), composes via
GraniteSwitchComposer, saves to disk, then verifies the output directory
structure, config correctness, weight transfer, and model loading.

A checkpoint holds either SR adapters or LoRA/aLoRA adapters, never both, so a
mixed compose is an error — see :class:`TestMixedAdapterRejection`.

All tests run on CPU with random weights — no model download needed.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import granite_switch.hf  # noqa: F401 — registers AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LORA_RANK = 4
CROSS_STREAM_RANK = 8
NUM_LAYERS = 2
HIDDEN = 64
INTERMEDIATE = 128
VOCAB_SIZE = 300
CONTROL_TOKEN_ID = 250


# ---------------------------------------------------------------------------
# Helpers: create mock base model and SR adapters on disk
# ---------------------------------------------------------------------------


def _create_base_model(path: Path):
    """Create a tiny Granite base model checkpoint on disk.

    Uses 'granite' model type (GraniteForCausalLM) — same weight structure
    (self_attn.q/k/v/o_proj, mlp.gate/up/down_proj) that GraniteSwitch expects.
    """
    base_cfg = {
        "model_type": "granite",
        "architectures": ["GraniteForCausalLM"],
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "attention_multiplier": 1.0,
        "logits_scaling": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "torch_dtype": "float32",
        "head_dim": 16,
    }
    (path / "config.json").write_text(json.dumps(base_cfg))

    torch.manual_seed(0)
    state_dict = {}
    state_dict["model.embed_tokens.weight"] = torch.randn(VOCAB_SIZE, HIDDEN)

    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(HIDDEN, INTERMEDIATE)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(HIDDEN)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(HIDDEN)

    state_dict["model.norm.weight"] = torch.ones(HIDDEN)
    state_dict["lm_head.weight"] = torch.randn(VOCAB_SIZE, HIDDEN)

    save_file(state_dict, str(path / "model.safetensors"))


def _create_sr_adapter(
    path: Path,
    cross_stream_rank: int = CROSS_STREAM_RANK,
    lora_rank: int = LORA_RANK,
    rank_pattern_key: str | None = "cross_stream",
    lora_alpha: float | None = None,
    alpha_pattern_key: str | None = "same",
):
    """Create a mock SR adapter with cross_stream weights.

    Args:
        rank_pattern_key: Key under which ``rank_pattern`` records the
            cross-stream rank. Stock PEFT owns this spelling since
            shadow-residual ``feature/stock-peft-sr``, so it may arrive fully
            qualified; ``None`` omits the entry entirely, forcing the rank to be
            derived from the weights.
        lora_alpha: Global ``lora_alpha``. Defaults to *lora_rank*.
        alpha_pattern_key: Key under which ``alpha_pattern`` records the
            cross-stream alpha. ``"same"`` reuses *rank_pattern_key*; ``None``
            leaves ``alpha_pattern`` empty, which is what real stock-peft-sr
            checkpoints ship and which makes the global ``lora_alpha`` apply.
    """
    if lora_alpha is None:
        lora_alpha = lora_rank
    if alpha_pattern_key == "same":
        alpha_pattern_key = rank_pattern_key
    path.mkdir(parents=True, exist_ok=True)

    target_modules = [
        "q_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "cross_stream",
    ]

    config = {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "rank_pattern": (
            {rank_pattern_key: cross_stream_rank} if rank_pattern_key else {}
        ),
        "alpha_pattern": (
            {alpha_pattern_key: cross_stream_rank} if alpha_pattern_key else {}
        ),
        "alora_invocation_tokens": [
            "<|start_of_role|>",
            "available_tools",
            "<|end_of_role|>",
        ],
    }
    (path / "adapter_config.json").write_text(json.dumps(config))

    # Create adapter weights
    torch.manual_seed(42)
    state_dict = {}
    prefix = "base_model.model.model."

    for layer_idx in range(NUM_LAYERS):
        lp = f"{prefix}layers.{layer_idx}"

        # Attention projections (under self_attn)
        for mod in ["q_proj", "o_proj"]:
            state_dict[f"{lp}.self_attn.{mod}.lora_A.weight"] = torch.randn(
                lora_rank, HIDDEN
            )
            state_dict[f"{lp}.self_attn.{mod}.lora_B.weight"] = torch.randn(
                HIDDEN, lora_rank
            )

        # MLP projections — SR stores under mlp parent (layers.X.mlp.gate_proj)
        for mod in ["gate_proj", "up_proj"]:
            state_dict[f"{lp}.mlp.{mod}.lora_A.weight"] = torch.randn(lora_rank, HIDDEN)
            state_dict[f"{lp}.mlp.{mod}.lora_B.weight"] = torch.randn(
                INTERMEDIATE, lora_rank
            )

        state_dict[f"{lp}.mlp.down_proj.lora_A.weight"] = torch.randn(
            lora_rank, INTERMEDIATE
        )
        state_dict[f"{lp}.mlp.down_proj.lora_B.weight"] = torch.randn(HIDDEN, lora_rank)

        # Cross-stream (layer-level, no parent)
        state_dict[f"{lp}.cross_stream.lora_A.weight"] = torch.randn(
            cross_stream_rank, HIDDEN
        )
        state_dict[f"{lp}.cross_stream.lora_B.weight"] = torch.randn(
            HIDDEN, cross_stream_rank
        )

    save_file(state_dict, str(path / "adapter_model.safetensors"))


def _compose_sr_probe(path: Path, read_layers=(None,)):
    """Tiny SR + probe checkpoint, also used by the vLLM test.

    An identity head exposes every component of its input, so its scores pin
    the exact layer and normalization read without random margins.
    """
    from granite_switch.composer import GraniteSwitchComposer
    from granite_switch.composer.weight_transfer import PROBE_HEAD_FILE

    base, sr, probe = (path / name for name in ("base", "sr", "probe"))
    base.mkdir(parents=True)
    probe.mkdir()
    _create_base_model(base)
    _create_sr_adapter(sr)
    save_file(
        {"weight": torch.eye(HIDDEN), "bias": torch.zeros(HIDDEN)},
        str(probe / PROBE_HEAD_FILE),
    )
    return GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base),
        adapter_paths=[str(sr), *[str(probe)] * len(read_layers)],
        adapter_token_ids=list(
            range(CONTROL_TOKEN_ID, CONTROL_TOKEN_ID + 1 + len(read_layers))
        ),
        adapter_substitute_token_ids=[1, *[2] * len(read_layers)],
        adapter_names=["sr", *[f"detect{i}" for i in range(len(read_layers))]],
        adapter_kinds=["lora", *["probe"] * len(read_layers)],
        probe_label_token_ids=[
            None,
            *[list(range(100, 100 + HIDDEN))] * len(read_layers),
        ],
        probe_read_layers=[None, *read_layers],
        switch_head_dim=16,
    ).eval()


@pytest.mark.parametrize("read_layer", [None, 0, NUM_LAYERS - 1])
def test_sr_and_probe_slots_select_separately(tmp_path, capsys, read_layer):
    """Identity scores prove layer and normalization before/after reload.

    The probe marker routes out of the SR stream, so the probe reads the
    base stream at its layer; an SR request in the same checkpoint still adapts.
    """
    from granite_switch.hf import load_model

    model = _compose_sr_probe(tmp_path, read_layers=[read_layer])
    assert "cross_stream slots are all-zero" not in capsys.readouterr().out
    assert model.config.dual_stream
    head = model.model.probe_head
    torch.testing.assert_close(head.weight[1], torch.eye(HIDDEN))
    assert torch.count_nonzero(head.weight[0]) == 0
    for layer in model.model.layers:
        assert torch.count_nonzero(layer.cross_stream.lora_B[0]) > 0
        assert torch.count_nonzero(layer.cross_stream.lora_B[1]) == 0

    def run(m, ids):
        streams = []
        hook = m.model.layers[
            -1 if read_layer is None else read_layer
        ].register_forward_hook(lambda _mod, _args, out: streams.append(out[0]))
        try:
            with torch.no_grad():
                output = m(input_ids=ids, use_cache=False)
        finally:
            hook.remove()
        base, adapted = (
            (m.model.norm(h) if read_layer is None else h)[0, -1] for h in streams[0]
        )
        return output, base, adapted

    labels = model.config.probe_label_token_ids[1]
    non_labels = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    non_labels[labels] = False

    probe_ids = torch.tensor([[10, 21, 20, CONTROL_TOKEN_ID + 1]])
    probe, base, _ = run(model, probe_ids)
    assert model.model._last_lora_indices.tolist() == [[0, 0, 0, 0]]
    assert model.model._last_probe_indices.tolist() == [[0, 0, 0, 2]]
    torch.testing.assert_close(probe.logits[0, -1, labels], base)
    assert torch.isneginf(probe.logits[0, -1, non_labels]).all()

    sr_ids = torch.tensor([[10, CONTROL_TOKEN_ID, 20, 30]])
    sr, base, adapted = run(model, sr_ids)
    assert model.model._last_lora_indices.tolist() == [[0, 1, 1, 1]]
    assert not model.model._last_probe_indices.any()
    assert not torch.allclose(adapted, base), "fixture must exercise SR adaptation"
    assert torch.isfinite(sr.logits[0, -1]).all()

    checkpoint = tmp_path / "composed"
    model.save_pretrained(checkpoint)
    loaded = load_model(str(checkpoint)).eval()
    torch.testing.assert_close(
        loaded.model.switch.adapter_kind_lut, model.model.switch.adapter_kind_lut
    )
    for ids, expected in ((probe_ids, probe), (sr_ids, sr)):
        with torch.no_grad():
            restored = loaded(input_ids=ids, use_cache=False)
        torch.testing.assert_close(restored.logits, expected.logits)
    assert loaded.model._last_lora_indices[0, -1].item() == 1


# ---------------------------------------------------------------------------
# Fixtures: compose once, save to disk, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build_output_shared(tmp_path_factory):
    """Compose SR shared-KV model and save to disk. Returns output dir Path."""
    from granite_switch.composer import GraniteSwitchComposer

    base_path = tmp_path_factory.mktemp("base_model")
    adapter_path = tmp_path_factory.mktemp("sr_shared_adapter")
    output_dir = tmp_path_factory.mktemp("build_shared") / "sr-model"

    _create_base_model(base_path)
    _create_sr_adapter(adapter_path)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base_path),
        adapter_paths=[str(adapter_path)],
        adapter_token_ids=[CONTROL_TOKEN_ID],
        adapter_substitute_token_ids=[1],
        adapter_names=["sr_shared"],
    )
    model.save_pretrained(str(output_dir))

    return output_dir, model


# ---------------------------------------------------------------------------
# Tests: shared-KV build
# ---------------------------------------------------------------------------


class TestSRBuildE2E:
    """Sanity checks on the output of an SR shared-KV compose build."""

    def test_output_files_exist(self, build_output_shared):
        """Check that safetensors and config.json are present."""
        output_dir, _ = build_output_shared

        safetensors = list(output_dir.glob("*.safetensors"))
        assert safetensors, "No .safetensors files found"
        assert (output_dir / "config.json").exists(), "Missing config.json"

    def test_config_architecture_is_unified(self, build_output_shared):
        """SR is no longer its own class — one architectures string for all kinds."""
        output_dir, _ = build_output_shared
        config = json.loads((output_dir / "config.json").read_text())

        assert config["architectures"] == ["GraniteSwitchForCausalLM"]
        assert config["model_type"] == "granite_switch"

    def test_config_has_sr_fields(self, build_output_shared):
        """An SR compose marks the whole checkpoint dual-stream."""
        output_dir, _ = build_output_shared
        config = json.loads((output_dir / "config.json").read_text())

        assert config["dual_stream"] is True
        assert config["cross_stream_rank"] == CROSS_STREAM_RANK
        assert "unfused_qkv" not in config

    def test_fused_attention_projection(self, build_output_shared):
        """SR composes into the fused qkv_proj, same as LoRA/aLoRA."""
        _, model = build_output_shared

        for layer in model.model.layers:
            attn = layer.self_attn
            assert hasattr(attn, "qkv_proj"), "Missing fused qkv_proj"
            for mod in ("q_proj", "k_proj", "v_proj"):
                assert not hasattr(attn, mod), f"Should not have unfused {mod}"

    def test_q_only_lora_lands_in_slice_zero(self, build_output_shared):
        """The adapter trains q_proj only, so slice 0 fires and 1/2 stay zero.

        This is why fusing SR is lossless: the LoRA side of a fused projection
        is stored as independent per-slice (A, B) pairs, never concatenated.
        """
        _, model = build_output_shared

        for layer in model.model.layers:
            qkv = layer.self_attn.qkv_proj
            assert qkv.lora_A_slices[0].abs().sum() > 0, "q slice lora_A is zero"
            assert qkv.lora_B_slices[0].abs().sum() > 0, "q slice lora_B is zero"
            # K and V were never trained (SR reads them from the base stream).
            for slice_idx in (1, 2):
                assert torch.all(qkv.lora_B_slices[slice_idx] == 0), (
                    f"qkv slice {slice_idx} lora_B should be zero"
                )

    def test_fused_mlp_projections(self, build_output_shared):
        """SR's mlp.gate/up/down_proj remap into the fused shared_mlp."""
        _, model = build_output_shared

        for layer in model.model.layers:
            assert hasattr(layer, "shared_mlp"), "Missing shared_mlp"
            assert not hasattr(layer, "mlp"), "Should not have unfused mlp"
            assert hasattr(layer.shared_mlp, "input_linear")
            assert hasattr(layer.shared_mlp, "output_linear")

    def test_cross_stream_base_is_zero(self, build_output_shared):
        """cross_stream.base_layer.weight must be all zeros after compose."""
        _, model = build_output_shared

        for layer in model.model.layers:
            weight = layer.cross_stream.base_layer.weight
            assert torch.all(weight == 0), (
                f"cross_stream base_layer not zero: max={weight.abs().max()}"
            )

    def test_cross_stream_lora_transferred(self, build_output_shared):
        """cross_stream lora_A/lora_B must be non-zero (adapter weights loaded)."""
        _, model = build_output_shared

        for layer in model.model.layers:
            cs = layer.cross_stream
            assert cs.lora_A[0].abs().sum() > 0, "cross_stream lora_A is all zeros"
            assert cs.lora_B[0].abs().sum() > 0, "cross_stream lora_B is all zeros"

    def test_cross_stream_rank_matches_config(self, build_output_shared):
        """cross_stream LoRA rank dimension matches cross_stream_rank from config."""
        _, model = build_output_shared

        for layer in model.model.layers:
            cs = layer.cross_stream
            # lora_A shape: (num_adapters, num_groups, rank, in_features)
            actual_rank = cs.lora_A.shape[2]
            assert actual_rank == CROSS_STREAM_RANK, (
                f"Expected rank {CROSS_STREAM_RANK}, got {actual_rank}"
            )

    def test_sr_mlp_keys_remap_to_fused_slices(self, build_output_shared):
        """Adapter keys under ``mlp.*`` land in the fused shared_mlp slices.

        ``gate_proj``/``up_proj`` become input_linear slices 0/1; ``down_proj``
        becomes the unsliced output_linear.
        """
        _, model = build_output_shared

        state_dict = model.state_dict()
        for layer_idx in range(NUM_LAYERS):
            prefix = f"model.layers.{layer_idx}.shared_mlp"
            expected = [
                f"{prefix}.input_linear.lora_A_slices.0",  # gate_proj
                f"{prefix}.input_linear.lora_B_slices.0",
                f"{prefix}.input_linear.lora_A_slices.1",  # up_proj
                f"{prefix}.input_linear.lora_B_slices.1",
                f"{prefix}.output_linear.lora_A",  # down_proj
                f"{prefix}.output_linear.lora_B",
            ]
            for key in expected:
                assert key in state_dict, f"Missing {key}"
                assert state_dict[key].abs().sum() > 0, f"{key} is all zeros"

            assert not any(
                k.startswith(f"model.layers.{layer_idx}.mlp.") for k in state_dict
            ), "Unfused mlp.* keys should be gone"

    def test_model_loads_roundtrip(self, build_output_shared):
        """Compose → save → load via load_model() produces bit-exact logits."""
        output_dir, model = build_output_shared
        model = model.eval()

        from granite_switch.hf import load_model

        loaded = load_model(str(output_dir)).eval()

        input_ids = torch.tensor([[10, 20, CONTROL_TOKEN_ID, 30, 40, 50, 60, 70]])
        with torch.no_grad():
            out_built = model(input_ids=input_ids).logits
            out_loaded = loaded(input_ids=input_ids).logits

        torch.testing.assert_close(out_built, out_loaded)

    def test_forward_output_shape(self, build_output_shared):
        """Composed SR model forward produces correct logits shape."""
        _, model = build_output_shared
        model = model.eval()

        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output.logits.shape == (1, 8, VOCAB_SIZE)

    def test_adapter_activation_changes_output(self, build_output_shared):
        """Control token activates SR adapter — post-control logits differ."""
        _, model = build_output_shared
        model = model.eval()

        with_ctrl = torch.tensor([[10, 20, CONTROL_TOKEN_ID, 30, 40, 50, 60, 70]])
        no_ctrl = torch.tensor([[10, 20, 100, 30, 40, 50, 60, 70]])

        with torch.no_grad():
            out_ctrl = model(input_ids=with_ctrl).logits
            out_text = model(input_ids=no_ctrl).logits

        # Post-control positions should differ (adapter active)
        assert not torch.allclose(out_ctrl[0, 3:], out_text[0, 3:]), (
            "Post-control logits should differ when SR adapter is active"
        )


# ---------------------------------------------------------------------------
# Tests: multi-adapter with different cross_stream ranks
# ---------------------------------------------------------------------------

CROSS_STREAM_RANK_SMALL = 4
CROSS_STREAM_RANK_LARGE = 16


@pytest.fixture(scope="module")
def build_output_multi_rank(tmp_path_factory):
    """Compose two SR adapters with different cross_stream ranks."""
    from granite_switch.composer import GraniteSwitchComposer

    base_path = tmp_path_factory.mktemp("base_model_mr")
    adapter_path_1 = tmp_path_factory.mktemp("sr_adapter_small_rank")
    adapter_path_2 = tmp_path_factory.mktemp("sr_adapter_large_rank")
    output_dir = tmp_path_factory.mktemp("build_multi_rank") / "sr-model"

    _create_base_model(base_path)
    _create_sr_adapter(adapter_path_1, cross_stream_rank=CROSS_STREAM_RANK_SMALL)
    _create_sr_adapter(adapter_path_2, cross_stream_rank=CROSS_STREAM_RANK_LARGE)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base_path),
        adapter_paths=[str(adapter_path_1), str(adapter_path_2)],
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 2],
        adapter_names=["sr_small", "sr_large"],
    )
    model.save_pretrained(str(output_dir))

    return output_dir, model


class TestSRMultiRankCrossStream:
    """Compose two SR adapters with different cross_stream ranks."""

    def test_cross_stream_rank_is_max(self, build_output_multi_rank):
        """Config cross_stream_rank should be the max across all adapters."""
        output_dir, _ = build_output_multi_rank
        config = json.loads((output_dir / "config.json").read_text())

        assert config["cross_stream_rank"] == CROSS_STREAM_RANK_LARGE

    def test_cross_stream_lora_shape_accommodates_max(self, build_output_multi_rank):
        """cross_stream lora_A rank dimension fits the largest adapter."""
        _, model = build_output_multi_rank

        for layer in model.model.layers:
            cs = layer.cross_stream
            # lora_A shape: (num_adapters, num_groups, rank, in_features)
            actual_rank = cs.lora_A.shape[2]
            assert actual_rank == CROSS_STREAM_RANK_LARGE

    def test_small_rank_adapter_is_zero_padded(self, build_output_multi_rank):
        """Adapter with smaller cross_stream rank should be zero-padded."""
        _, model = build_output_multi_rank

        for layer in model.model.layers:
            cs = layer.cross_stream
            # Adapter index 0 (sr_small) has rank 4, padded to 16
            lora_a = cs.lora_A[0, 0]  # shape: (max_rank, hidden)
            # First CROSS_STREAM_RANK_SMALL rows should be non-zero
            assert lora_a[:CROSS_STREAM_RANK_SMALL].abs().sum() > 0
            # Remaining rows should be zero (padding)
            assert torch.all(lora_a[CROSS_STREAM_RANK_SMALL:] == 0)

    def test_forward_runs_with_both_adapters(self, build_output_multi_rank):
        """Forward pass succeeds with both adapters active at different positions."""
        _, model = build_output_multi_rank
        model = model.eval()

        # Token 250 activates adapter 0 (small rank), 251 activates adapter 1 (large rank)
        input_ids = torch.tensor([[10, 250, 30, 40, 251, 50, 60, 70]])
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output.logits.shape == (1, 8, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Tests: mixing SR and non-SR adapters is rejected
# ---------------------------------------------------------------------------


def _create_standard_lora_adapter(path: Path):
    """Create a mock standard LoRA adapter (no cross_stream)."""
    path.mkdir(parents=True, exist_ok=True)

    config = {
        "r": LORA_RANK,
        "lora_alpha": LORA_RANK,
        "target_modules": ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))

    torch.manual_seed(99)
    state_dict = {}
    prefix = "base_model.model.model."

    for layer_idx in range(NUM_LAYERS):
        lp = f"{prefix}layers.{layer_idx}"
        for mod in ["q_proj", "o_proj"]:
            state_dict[f"{lp}.self_attn.{mod}.lora_A.weight"] = torch.randn(
                LORA_RANK, HIDDEN
            )
            state_dict[f"{lp}.self_attn.{mod}.lora_B.weight"] = torch.randn(
                HIDDEN, LORA_RANK
            )
        for mod in ["gate_proj", "up_proj"]:
            state_dict[f"{lp}.mlp.{mod}.lora_A.weight"] = torch.randn(LORA_RANK, HIDDEN)
            state_dict[f"{lp}.mlp.{mod}.lora_B.weight"] = torch.randn(
                INTERMEDIATE, LORA_RANK
            )
        state_dict[f"{lp}.mlp.down_proj.lora_A.weight"] = torch.randn(
            LORA_RANK, INTERMEDIATE
        )
        state_dict[f"{lp}.mlp.down_proj.lora_B.weight"] = torch.randn(HIDDEN, LORA_RANK)

    save_file(state_dict, str(path / "adapter_model.safetensors"))


class TestMixedAdapterRejection:
    """Composing SR together with non-SR adapters must raise.

    The decoder runs in one stream mode for the whole checkpoint, so there is no
    layout that could serve both kinds at once.
    """

    def test_mixed_sr_and_lora_raises(self, tmp_path):
        from granite_switch.composer import GraniteSwitchComposer

        base_path = tmp_path / "base"
        lora_path = tmp_path / "lora"
        sr_path = tmp_path / "sr"

        base_path.mkdir(parents=True)
        _create_base_model(base_path)
        _create_standard_lora_adapter(lora_path)
        _create_sr_adapter(sr_path)

        with pytest.raises(ValueError, match="Cannot mix Shadow Residual"):
            GraniteSwitchComposer.from_base_and_adapters(
                base_model_name_or_path=str(base_path),
                adapter_paths=[str(lora_path), str(sr_path)],
                adapter_token_ids=[250, 251],
                adapter_substitute_token_ids=[1, 2],
                adapter_names=["plain_lora", "sr_adapter"],
            )


# ---------------------------------------------------------------------------
# Tests: a LoRA-only compose stays free of every SR artifact
# ---------------------------------------------------------------------------


class TestLoRAOnlyComposeIsUnchanged:
    """Adding SR support must not add a single parameter to a pure-LoRA build.

    ``tests/composer/test_compose_e2e.py`` pins an exact BASE_PARAM_COUNT for a
    real granite-4.0-micro build; this is the cheap CPU proof of the same
    invariant.
    """

    def test_no_sr_parameters_or_config_fields(self, tmp_path):
        from granite_switch.composer import GraniteSwitchComposer

        base_path = tmp_path / "base"
        lora_path = tmp_path / "lora"
        output_dir = tmp_path / "lora-only"

        base_path.mkdir(parents=True)
        _create_base_model(base_path)
        _create_standard_lora_adapter(lora_path)

        model = GraniteSwitchComposer.from_base_and_adapters(
            base_model_name_or_path=str(base_path),
            adapter_paths=[str(lora_path)],
            adapter_token_ids=[CONTROL_TOKEN_ID],
            adapter_substitute_token_ids=[1],
            adapter_names=["plain_lora"],
        )
        model.save_pretrained(str(output_dir))

        assert not [n for n in model.state_dict() if "cross_stream" in n]
        for layer in model.model.layers:
            assert not hasattr(layer, "cross_stream")
            assert type(layer).__name__ == "GraniteSwitchAttentionDecoderLayer"

        # The SR fields are always serialized (they are config attributes), but
        # must be inert — no cross_stream site is allocated.
        config = json.loads((output_dir / "config.json").read_text())
        assert config["architectures"] == ["GraniteSwitchForCausalLM"]
        assert config["dual_stream"] is False
        assert config["cross_stream_rank"] is None


# ---------------------------------------------------------------------------
# Tests: the cross-stream rank survives stock PEFT's key spelling
# ---------------------------------------------------------------------------


class TestCrossStreamRankResolution:
    """``rank_pattern``'s cross-stream key is no longer ours to spell.

    Since shadow-residual ``feature/stock-peft-sr`` the cross-stream layer is
    targeted by name through stock PEFT, which decides whether the key lands
    bare or fully qualified. An unresolved rank used to surface far from its
    cause, as ``cross_stream_rank is required when dual_stream is True`` out of
    config validation, and would leave the composed model silently wrong.
    """

    def _compose(self, tmp_path, **adapter_kwargs):
        from granite_switch.composer import GraniteSwitchComposer

        base_path = tmp_path / "base"
        adapter_path = tmp_path / "sr"
        base_path.mkdir(parents=True)
        _create_base_model(base_path)
        _create_sr_adapter(adapter_path, **adapter_kwargs)

        return GraniteSwitchComposer.from_base_and_adapters(
            base_model_name_or_path=str(base_path),
            adapter_paths=[str(adapter_path)],
            adapter_token_ids=[250],
            adapter_substitute_token_ids=[1],
            adapter_names=["sr_adapter"],
        )

    def test_fully_qualified_rank_pattern_key(self, tmp_path):
        model = self._compose(
            tmp_path,
            rank_pattern_key="base_model.model.model.layers.0.cross_stream",
        )
        assert model.config.cross_stream_rank == CROSS_STREAM_RANK

    def test_rank_derived_from_weights_when_absent(self, tmp_path):
        """No cross_stream entry at all — the weights are authoritative.

        ``lora_A.weight`` has shape ``(rank, hidden)``, which cannot drift from
        the tensors actually being loaded.
        """
        model = self._compose(tmp_path, rank_pattern_key=None)
        assert model.config.cross_stream_rank == CROSS_STREAM_RANK

    def test_empty_alpha_pattern_falls_back_to_global_lora_alpha(self, tmp_path):
        """An absent alpha_pattern entry means ``lora_alpha``, never the rank.

        This is PEFT's own rule (``LoraModel._create_and_replace``:
        ``alpha_pattern.get(key, lora_config.lora_alpha)``), and real
        stock-peft-sr checkpoints ship ``alpha_pattern={}`` with a
        ``lora_alpha`` of twice the cross-stream rank. Falling back to the rank
        instead would scale the cross-stream delta by 1.0 rather than 2.0 —
        halving SR's contribution with no error anywhere.
        """
        from granite_switch.composer.adapter_loader import (
            resolve_cross_stream_rank_alpha,
        )

        adapter_path = tmp_path / "sr"
        _create_sr_adapter(
            adapter_path,
            alpha_pattern_key=None,
            lora_alpha=2 * CROSS_STREAM_RANK,
        )

        rank, alpha = resolve_cross_stream_rank_alpha(str(adapter_path))
        assert rank == CROSS_STREAM_RANK
        assert alpha == 2 * CROSS_STREAM_RANK

    def test_alpha_pattern_entry_wins_over_global(self, tmp_path):
        """An explicit cross_stream alpha_pattern entry is authoritative.

        Pre-stock-PEFT SR checkpoints spelled this out, and they must keep
        their original scaling.
        """
        from granite_switch.composer.adapter_loader import (
            resolve_cross_stream_rank_alpha,
        )

        adapter_path = tmp_path / "sr"
        _create_sr_adapter(adapter_path, lora_alpha=999)

        _rank, alpha = resolve_cross_stream_rank_alpha(str(adapter_path))
        assert alpha == CROSS_STREAM_RANK


# ---------------------------------------------------------------------------
# Tests: the pre-activation streams are identical
# ---------------------------------------------------------------------------


class TestStreamsAgreeBeforeActivation:
    """Before the control token the adapter stream must equal the base stream.

    This is what makes control-token placement a pure prompt-level decision: at
    positions where ``adapter_indices == 0`` every switched LoRA falls back to
    its base weight, and ``cross_stream``'s own base weight is zero-initialized,
    so no adapter contribution can leak in early.
    """

    def test_hidden_states_match_before_the_control_token(self, build_output_shared):
        _, model = build_output_shared
        model = model.eval()

        control_id = model.config.adapter_token_ids[0]
        activate_at = 4
        input_ids = torch.tensor([[10, 20, 30, 40, control_id, 50, 60]])
        assert input_ids[0, activate_at].item() == control_id

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)

        # The lm_head reads the adapter stream, so a base/adapter divergence
        # before the control token would show up as a logits difference against
        # a run with no control token at all.
        plain = input_ids.clone()
        plain[0, activate_at] = 20
        with torch.no_grad():
            out_plain = model(input_ids=plain, output_hidden_states=True)

        torch.testing.assert_close(
            out.logits[0, :activate_at],
            out_plain.logits[0, :activate_at],
            msg="adapter stream diverged from the base stream before activation",
        )
