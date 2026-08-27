# SPDX-License-Identifier: Apache-2.0
"""Slot-invariance of a multi-adapter compose (CPU, no model download).

Establishes the property the vLLM multi-adapter equivalence tests
(tests/vllm/test_multi_adapter_equivalence.py) rest on: **an adapter's behaviour
depends on the adapter, not on which slot it landed in.** Composing the same set of
adapters in two orders must produce, per adapter, byte-identical stacked LoRA
weights and the same forward output when its control token fires. If either fails,
a difference between two orders under vLLM would be a compose artifact rather than
the routing bug the GPU tests look for.

Three configurations, chosen for what each one alone covers: LoRA on the dense MLP
spelling (what granite-4.1-3b uses, and what the GPU suite composes), aLoRA on the
shared spelling (granitemoehybrid's fused gate|up, which the GPU suite never
reaches), and Shadow Residual, whose cross_stream site exists nowhere else. Each
carries a distinct rank per adapter, so the kernel-facing ``adapter_ranks`` ordering
is exercised too.

CPU-only, no download: a tiny synthetic base plus adapters from the fitted profiles
in tests/shared/synthetic_adapters.py.
"""

import json

import pytest
import torch

import granite_switch.hf  # noqa: F401 — registers AutoModel/AutoConfig
from tests.shared.base_models import dims_from_config, write_tiny_granite_base
from tests.shared.synthetic_adapters import (
    PROFILES,
    attn_modules,
    base_weight_norms,
    mlp_modules,
    synthesize_adapter,
)

# Ranks on the SWITCH kernel's supported tiers, distinct per adapter so rank-ordered
# kernel-local remapping differs from global adapter order. Each comes from the
# fitted profile of that rank, so the adapters carry a trained delta's norm,
# spectrum and A/B asymmetry rather than iid noise.
PROFILE_BY_RANK = {16: "light", 32: "medium", 64: "strong"}
RANKS = [16, 32, 64]
CROSS_RANKS = [16, 32, 64]
NUM_ADAPTERS = 3
CONTROL_TOKEN_IDS = [250, 251, 252]
SUBSTITUTE_TOKEN_ID = 1

#: (flavor, MLP spelling). See the module docstring for why these three.
CONFIGURATIONS = [("lora", "dense"), ("alora", "shared"), ("sr", "dense")]


def _adapter_names():
    return [f"ad_{i}" for i in range(NUM_ADAPTERS)]


def _write_adapters(root, flavor, dims, norms):
    """Write NUM_ADAPTERS adapter dirs; returns {name: path}."""
    paths = {}
    for i, name in enumerate(_adapter_names()):
        path = root / f"adapter_{name}"
        profile = PROFILES[PROFILE_BY_RANK[RANKS[i]]]
        if flavor == "sr":
            # SR reads K/V from the base stream and wraps the MLP projections
            # unconditionally, so its adapters cover neither k/v nor less than the
            # whole MLP.
            modules = attn_modules(with_kv=False) + mlp_modules(dims)
            extra = dict(
                cross_stream_rank=CROSS_RANKS[i],
                last_context_token=("<|end_of_role|>", 100265),
            )
        else:
            # The last adapter is attention-only, so the MLP modules have adapters
            # that do not apply to every slot.
            modules = attn_modules() + (
                mlp_modules(dims) if i != NUM_ADAPTERS - 1 else []
            )
            extra = {"alora_invocation_tokens": [1, 2, 3]} if flavor == "alora" else {}
        synthesize_adapter(
            path, dims, norms, profile, seed=1000 + i, modules=modules, **extra
        )
        paths[name] = str(path)
    return paths


def _compose(base_path, adapter_paths_by_name, order):
    """Compose the adapters in ``order`` (a list of names)."""
    from granite_switch.composer import GraniteSwitchComposer

    token_by_name = dict(zip(_adapter_names(), CONTROL_TOKEN_IDS))
    return GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base_path),
        adapter_paths=[adapter_paths_by_name[n] for n in order],
        adapter_names=list(order),
        adapter_token_ids=[token_by_name[n] for n in order],
        adapter_substitute_token_ids=[SUBSTITUTE_TOKEN_ID] * len(order),
    )


@pytest.fixture(scope="module")
def composed(request, tmp_path_factory):
    """(forward-order model, reversed-order model) for one configuration."""
    from transformers import AutoConfig

    flavor, mlp_style = request.param
    root = tmp_path_factory.mktemp(f"multi_{flavor}_{mlp_style}")
    base_path = write_tiny_granite_base(root / "base", mlp_style=mlp_style)
    dims = dims_from_config(AutoConfig.from_pretrained(str(base_path)))
    adapters = _write_adapters(root, flavor, dims, base_weight_norms(base_path))

    names = _adapter_names()
    fwd = _compose(base_path, adapters, names)
    rev = _compose(base_path, adapters, list(reversed(names)))
    return fwd, rev


@pytest.mark.parametrize(
    "composed", CONFIGURATIONS, indirect=True, ids=lambda c: "-".join(c)
)
class TestMultiAdapterSlotInvariance:
    def test_slot_order_is_honoured(self, composed):
        """Every adapter lands in the checkpoint, and adapter_ranks is per slot."""
        fwd, rev = composed
        names = _adapter_names()
        assert fwd.config.num_adapters == NUM_ADAPTERS
        assert rev.config.num_adapters == NUM_ADAPTERS
        assert list(fwd.config.adapter_names) == names
        assert list(rev.config.adapter_names) == list(reversed(names))
        assert list(fwd.config.adapter_ranks) == RANKS
        assert list(rev.config.adapter_ranks) == list(reversed(RANKS))

    def test_per_adapter_weights_are_slot_independent(self, composed):
        """Each adapter's stacked LoRA slice is identical in both composes, and
        distinct adapters differ — so the comparison is not vacuous.

        Only *delta-bearing* slices are compared: a slice whose ``lora_B`` is
        all-zero contributes nothing regardless of its ``lora_A``, and the composer
        leaves such an ``lora_A`` at whatever the module's constructor initialized it
        to (random, and not reproducible across composes). SR's fused-QKV K/V slices
        are the case in point, since SR never adapts K/V. Comparing those would
        assert compose determinism the composer does not promise.
        """
        fwd, rev = composed
        fwd_names = list(fwd.config.adapter_names)
        rev_names = list(rev.config.adapter_names)
        fwd_params = dict(fwd.named_parameters())
        rev_params = dict(rev.named_parameters())

        compared = differing = 0
        for pname, fp in fwd_params.items():
            if "lora_A" not in pname and "lora_B" not in pname:
                continue
            rp = rev_params[pname]
            assert fp.shape == rp.shape, f"{pname}: {fp.shape} vs {rp.shape}"
            b_name = pname.replace("lora_A", "lora_B")
            f_b, r_b = fwd_params[b_name], rev_params[b_name]
            for name in fwd_names:
                f_slot, r_slot = fwd_names.index(name), rev_names.index(name)
                if not f_b[f_slot].any() and not r_b[r_slot].any():
                    continue  # no delta on either side — lora_A is unused
                assert torch.equal(fp[f_slot], rp[r_slot]), (
                    f"{pname}: adapter {name} differs between slot {f_slot} and "
                    f"slot {r_slot}"
                )
                compared += 1
        assert compared > 0, "no LoRA parameters were compared"

        for pname, p in fwd_params.items():
            if "lora_A" not in pname:
                continue
            for i in range(NUM_ADAPTERS):
                differing += sum(
                    not torch.equal(p[i], p[j]) for j in range(i + 1, NUM_ADAPTERS)
                )
        assert differing > 0, "all adapters carry identical LoRA weights"

    def test_forward_is_slot_independent_and_every_adapter_matters(self, composed):
        """Firing adapter i's control token gives the same output in both orders,
        and a different output from base and from every other adapter."""
        fwd, rev = composed
        fwd, rev = fwd.eval(), rev.eval()
        torch.manual_seed(7)
        body = torch.randint(2, 200, (1, 12))
        base_ids = torch.cat([torch.tensor([[SUBSTITUTE_TOKEN_ID]]), body], dim=1)

        outputs = []
        with torch.no_grad():
            base_out = fwd(input_ids=base_ids).logits
            for name, token_id in zip(_adapter_names(), CONTROL_TOKEN_IDS):
                ids = torch.cat([torch.tensor([[token_id]]), body], dim=1)
                a = fwd(input_ids=ids).logits
                b = rev(input_ids=ids).logits
                torch.testing.assert_close(
                    a,
                    b,
                    rtol=0,
                    atol=0,
                    msg=lambda m,
                    n=name: f"adapter {n} output depends on its slot\n{m}",
                )
                outputs.append(a)

        for i, out in enumerate(outputs):
            assert not torch.allclose(out, base_out), f"adapter {i} is a no-op vs base"
        for i in range(NUM_ADAPTERS):
            for j in range(i + 1, NUM_ADAPTERS):
                assert not torch.allclose(outputs[i], outputs[j]), (
                    f"adapters {i} and {j} produce identical output"
                )


def test_multi_adapter_sr_checkpoint_roundtrips(tmp_path):
    """A 3-adapter SR checkpoint saves and reloads (the shape vLLM now accepts).

    The vLLM SR decoder used to reject ``num_adapters > 1`` outright; the composer
    never did. This pins the checkpoint side: three SR adapters with distinct
    cross_stream ranks compose, save and reload with the stacked cross_stream sized
    to the max rank and the narrower adapters zero-padded.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    base_path = write_tiny_granite_base(tmp_path / "base", mlp_style="dense")
    dims = dims_from_config(AutoConfig.from_pretrained(str(base_path)))
    adapters = _write_adapters(tmp_path, "sr", dims, base_weight_norms(base_path))
    model = _compose(base_path, adapters, _adapter_names())

    out_dir = tmp_path / "composed"
    model.save_pretrained(str(out_dir))

    config = json.loads((out_dir / "config.json").read_text())
    assert config["num_adapters"] == NUM_ADAPTERS
    assert config["dual_stream"] is True
    assert config["cross_stream_rank"] == max(CROSS_RANKS)

    reloaded = AutoModelForCausalLM.from_pretrained(str(out_dir))
    assert reloaded.config.num_adapters == NUM_ADAPTERS
    for layer in reloaded.model.layers:
        cs = layer.cross_stream
        assert cs.lora_A.shape[0] == NUM_ADAPTERS
        assert cs.lora_A.shape[2] == max(CROSS_RANKS)
        for slot, rank in enumerate(CROSS_RANKS):
            rows = cs.lora_A[slot, 0]
            assert rows[:rank].abs().sum() > 0, f"slot {slot} cross_stream is empty"
            assert torch.all(rows[rank:] == 0), f"slot {slot} padding is not zero"
