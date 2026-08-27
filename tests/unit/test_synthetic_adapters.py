# SPDX-License-Identifier: Apache-2.0
"""The synthetic adapter generator reproduces the statistics it was fitted to.

tests/shared/synthetic_adapters.py exists because iid Gaussian ``lora_A`` /
``lora_B`` are wrong in the way that matters: a trained delta is nearly rank-one to
rank-three whatever its nominal rank, while an iid product uses a large fraction of
its nominal rank with an almost flat spectrum. These tests pin that the generator
hits its three targets — delta norm relative to the base weight, stable rank, and
the A/B asymmetry — and that an iid product would not.

Deliberately small. The vLLM suite that consumes these adapters re-measures every
one it builds and prints the result, so a broken generator surfaces there too; what
needs pinning here is only what that run cannot tell apart from a runtime bug.

CPU-only and download-free: a tiny synthetic Granite base supplies the weight norms
the profiles are expressed against.
"""

import json

import pytest
import torch
from safetensors.torch import load_file

from tests.shared.base_models import dims_from_config, write_tiny_granite_base
from tests.shared.synthetic_adapters import (
    PROFILES,
    attn_modules,
    base_weight_norms,
    factorize,
    measure_adapter,
    mlp_modules,
    spectrum,
    synthesize_adapter,
)

# Wide enough that the rank-64 profile fits every projection (rank must not exceed
# min(out, in), and kv_width is the smallest).
BASE_KW = dict(
    hidden=128, intermediate=256, num_heads=4, num_kv_heads=4, head_dim=32, num_layers=3
)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """(dims, weight norms) for a tiny dense Granite base."""
    from transformers import AutoConfig

    root = tmp_path_factory.mktemp("synth_base")
    path = write_tiny_granite_base(root / "base", mlp_style="dense", **BASE_KW)
    dims = dims_from_config(AutoConfig.from_pretrained(str(path)))
    return dims, base_weight_norms(path)


def test_spectrum_solves_stable_rank_and_clamps_to_the_achievable_range():
    """Stable rank is bisected to the target, and cannot exceed r or fall below 1."""
    for rank in (16, 32, 64):
        for target in (1.1, 1.5, 3.0, 8.0):
            s = spectrum(rank, target)
            got = float((s**2).sum() / s[0] ** 2)
            assert abs(got - target) < 0.02, f"r={rank} asked {target}, got {got}"
    for rank, asked, expect in ((16, 100.0, 16.0), (16, 0.1, 1.0)):
        s = spectrum(rank, asked)
        assert float((s**2).sum() / s[0] ** 2) == pytest.approx(expect, abs=0.05)


def test_an_iid_product_uses_far_more_directions():
    """The contrast this module exists for.

    Every trained adapter measured came in at stable rank 1.09-3.48 except the r=64
    one (3.53-23.9). An iid product of the same shape uses a large fraction of its
    nominal rank instead; the exact ratio depends on the aspect ratio, so assert the
    gap rather than a fitted constant.
    """
    trained_max = 3.5  # the measured ceiling for the r<=32 adapters
    for rank in (16, 32, 64):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(rank, 512, generator=g)
        b = torch.randn(512, rank, generator=g)
        delta = b @ a
        sv = torch.linalg.svdvals(delta)
        srank = float(torch.linalg.matrix_norm(delta) ** 2 / sv[0] ** 2)
        assert srank > 2 * trained_max, (
            f"iid r={rank} stable rank {srank} is within reach of a trained "
            f"adapter's {trained_max}; the contrast this module rests on does not "
            "hold for these shapes"
        )


def test_factorize_hits_norm_spectrum_and_asymmetry():
    """All three targets exactly, and factors safetensors will accept."""
    a, b = factorize(
        256,
        512,
        rank=32,
        target_fro=3.5,
        stable_rank=1.8,
        b_over_a=0.4,
        generator=torch.Generator().manual_seed(1),
    )
    delta = b @ a
    fro = float(torch.linalg.matrix_norm(delta))
    assert fro == pytest.approx(3.5, rel=1e-3)
    assert float(fro**2 / torch.linalg.svdvals(delta)[0] ** 2) == pytest.approx(
        1.8, abs=0.02
    )
    assert float(b.std() / a.std()) == pytest.approx(0.4, rel=1e-3)
    # The broadcasts inside leave non-contiguous views, which safetensors refuses.
    assert a.is_contiguous() and b.is_contiguous()

    with pytest.raises(ValueError, match="exceeds"):
        factorize(
            32,
            512,
            rank=64,
            target_fro=1.0,
            stable_rank=2.0,
            b_over_a=0.5,
            generator=torch.Generator().manual_seed(3),
        )


@pytest.mark.parametrize("profile_key", ["weak", "strong"])
def test_every_module_and_layer_matches_the_profile(base, profile_key, tmp_path):
    """What was written measures back as what the profile asked for.

    The weakest and strongest profiles bracket the range: a 40x spread in delta norm
    and stable rank 1.53 against 11.63.
    """
    dims, norms = base
    profile = PROFILES[profile_key]
    modules = attn_modules() + mlp_modules(dims)
    out = synthesize_adapter(
        tmp_path / profile_key, dims, norms, profile, seed=5, modules=modules
    )
    layers = (0, dims.num_layers - 1)
    rows = measure_adapter(out, norms, layers=layers)
    assert len(rows) == len(layers) * len(modules)
    for row in rows:
        want = profile.rel_fro(row["layer"], dims.num_layers)
        assert row["rel_fro"] == pytest.approx(want, rel=0.01), row
        assert row["stable_rank"] == pytest.approx(profile.stable_rank, abs=0.05)
        assert row["b_over_a"] == pytest.approx(profile.b_over_a, rel=0.01)
    # The depth ramp: the last layer's target is the larger one.
    first = next(r["rel_fro"] for r in rows if r["layer"] == 0)
    last = next(r["rel_fro"] for r in rows if r["layer"] == layers[-1])
    assert first < last


def test_effect_scale_multiplies_the_target_without_hiding_the_measurement():
    """rel_fro() applies effect_scale; the endpoints stay the measured ones.

    The endpoints are what a trained adapter measured; effect_scale is the empirical
    correction for random-direction deltas being weaker per unit norm. Both have to
    stay separately readable.
    """
    from dataclasses import replace

    p = PROFILES["weak"]
    assert p.effect_scale > 1.0
    assert replace(p, effect_scale=1.0).rel_fro(0, 40) == pytest.approx(p.rel_fro_first)
    assert p.rel_fro(0, 40) == pytest.approx(p.rel_fro_first * p.effect_scale)
    assert p.rel_fro(39, 40) == pytest.approx(p.rel_fro_last * p.effect_scale)


def test_sr_and_alora_configs_are_mutually_exclusive(base, tmp_path):
    """SR carries last_context_token; aLoRA carries invocation tokens.

    The composer probes the aLoRA key FIRST, so an SR adapter that also carried one
    would keep aLoRA placement — which silently moves the control token.
    """
    dims, norms = base
    common = dict(
        dims=dims, norms=norms, profile=PROFILES["weak"], modules=["self_attn.q_proj"]
    )
    sr = synthesize_adapter(
        tmp_path / "cfg_sr",
        seed=8,
        cross_stream_rank=16,
        last_context_token=("<|end_of_role|>", 100265),
        **common,
    )
    cfg = json.loads((sr / "adapter_config.json").read_text())
    assert cfg["last_context_token_id"] == 100265
    assert "alora_invocation_tokens" not in cfg
    assert cfg["rank_pattern"] == {"cross_stream": 16}

    alora = synthesize_adapter(
        tmp_path / "cfg_alora", seed=8, alora_invocation_tokens=[1, 2, 3], **common
    )
    cfg = json.loads((alora / "adapter_config.json").read_text())
    assert cfg["alora_invocation_tokens"] == [1, 2, 3]
    assert "last_context_token" not in cfg


def test_cross_stream_is_sized_against_o_proj(base, tmp_path):
    """The SR shunt is W-less, so its target borrows the o_proj norm."""
    dims, norms = base
    profile = PROFILES["light"]
    out = synthesize_adapter(
        tmp_path / "sr",
        dims,
        norms,
        profile,
        seed=7,
        modules=attn_modules(with_kv=False) + mlp_modules(dims),
        cross_stream_rank=32,
        last_context_token=("<|end_of_role|>", 100265),
    )
    rows = measure_adapter(out, norms, layers=(0,))
    cross = next(r for r in rows if r["module"] == "cross_stream")
    assert cross["rank"] == 32
    assert cross["rel_fro"] == pytest.approx(
        profile.rel_fro(0, dims.num_layers), rel=0.01
    )
    # SR reads K/V from the base stream; the composer rejects an adapter that
    # carries them, so they must be absent.
    assert not [r for r in rows if "k_proj" in r["module"] or "v_proj" in r["module"]]


def test_same_seed_reproduces_bit_identical_weights(base, tmp_path):
    """Slot-invariance testing rests on an adapter being reproducible."""
    dims, norms = base
    kw = dict(
        dims=dims, norms=norms, profile=PROFILES["light"], modules=["self_attn.q_proj"]
    )
    a = synthesize_adapter(tmp_path / "seed_a", seed=11, **kw)
    b = synthesize_adapter(tmp_path / "seed_b", seed=11, **kw)
    c = synthesize_adapter(tmp_path / "seed_c", seed=12, **kw)
    wa, wb, wc = (load_file(str(p / "adapter_model.safetensors")) for p in (a, b, c))
    assert wa.keys() == wb.keys() == wc.keys()
    for key in wa:
        assert torch.equal(wa[key], wb[key]), f"{key} not reproducible"
    assert any(not torch.equal(wa[key], wc[key]) for key in wa), (
        "a different seed produced identical weights"
    )


def test_the_shared_mlp_spelling_is_supported(tmp_path):
    """granitemoehybrid bases fuse gate|up, so ``lora_B`` is ``[2*I, r]``.

    The GPU suite runs against a dense base, so nothing else covers this table.
    """
    from transformers import AutoConfig

    path = write_tiny_granite_base(tmp_path / "shared", mlp_style="shared", **BASE_KW)
    dims = dims_from_config(AutoConfig.from_pretrained(str(path)))
    assert mlp_modules(dims) == ["shared_mlp.input_linear", "shared_mlp.output_linear"]
    norms = base_weight_norms(path)
    profile = PROFILES["medium"]
    out = synthesize_adapter(
        tmp_path / "ad",
        dims,
        norms,
        profile,
        seed=1,
        modules=attn_modules() + mlp_modules(dims),
    )
    weights = load_file(str(out / "adapter_model.safetensors"))
    fused = weights[
        "base_model.model.model.layers.0.shared_mlp.input_linear.lora_B.weight"
    ]
    assert fused.shape == (2 * dims.intermediate, profile.rank)
    for row in measure_adapter(out, norms, layers=(0,)):
        assert row["rel_fro"] == pytest.approx(
            profile.rel_fro(0, dims.num_layers), rel=0.01
        )


def test_unknown_module_is_rejected(base, tmp_path):
    dims, norms = base
    with pytest.raises(ValueError, match="unknown modules"):
        synthesize_adapter(
            tmp_path / "bad",
            dims,
            norms,
            PROFILES["weak"],
            seed=13,
            modules=["self_attn.nonexistent"],
        )
