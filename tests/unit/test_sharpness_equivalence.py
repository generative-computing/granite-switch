# SPDX-License-Identifier: Apache-2.0
"""Softmax sharpness equivalence between HF and vLLM SingleSwitch geometries.

SingleSwitch runs under two different computational paths:

- HF backend: 1 head, scaling=1.0, K[0] = ±control_token_gain directly.
  Net softmax logit = ±gain.

- vLLM backend: multi-head (backbone geometry), scaling=attention_multiplier,
  K[0] = ±effective_gain where effective_gain = gain / attention_multiplier.
  The vLLM Attention kernel applies `scaling` internally, so the net softmax
  logit is again ±gain — but via `attention_multiplier × effective_gain`
  in bf16 (e.g. 0.0078125 × 1920.0 for granite-4.0-h-1b).

In exact arithmetic both paths produce the same softmax distribution and the
same recovered adapter ID after `round(weighted_sum_of_V)`. This test verifies
the equivalence holds under bf16 — given the same token sequence, the
attention weight distribution under real (attention_multiplier, effective_gain)
must match the HF-style (1.0, gain) within bf16 tolerance, and the recovered
adapter ID must match exactly.

Pure numerical test — no model, no GPU, runs on CPU in milliseconds.

Note: this test cannot reuse SingleSwitch.forward() directly because the HF
implementation hardcodes scaling=1.0 regardless of config, so calling it only
exercises one of the two geometries we need to compare. The vLLM geometry
needs manual logit construction because vLLM's Attention kernel is GPU-only.
"""

import pytest
import torch

from tests.shared.granite4_constants import (
    DEFAULT_CONTROL_TOKEN_GAIN,
    MAX_POSITION_EMBEDDINGS,
    PRODUCTION_ATTENTION_MULTIPLIERS,
)

# Stress adapter IDs: 1 (smallest), 16 (middle), 32 (largest supported)
ADAPTER_IDS = [1, 16, 32]

# Sequence lengths covering short through model's declared max context.
# Includes 32K and 64K deployment lengths and 131K (max_position_embeddings).
SEQ_LENS = [10, 100, 1000, 10000, 32768, 65536, MAX_POSITION_EMBEDDINGS]


def _build_logits_and_values(
    seq_len: int,
    adapter_id: int,
    attention_multiplier: float,
    ctrl_pos: int = 1,
):
    """Construct pre-softmax logit vectors under HF and vLLM geometries.

    Mirrors SingleSwitch forward's K[0]/V[0] pattern:
      - HF:   K[0] ∈ {-gain, +gain},        scaling = 1.0
      - vLLM: K[0] ∈ {-eff_gain, +eff_gain}, scaling = attention_multiplier
              where eff_gain = gain / attention_multiplier
              Net logit = scaling × Q[0] × K[0] = attention_multiplier × eff_gain = gain

    Returns (hf_logits, vllm_logits, values) — all bf16.
    """
    gain = DEFAULT_CONTROL_TOKEN_GAIN
    gain_bf16 = torch.tensor(gain, dtype=torch.bfloat16)
    mult_bf16 = torch.tensor(attention_multiplier, dtype=torch.bfloat16)
    effective_gain = gain_bf16 / mult_bf16  # exact for power-of-2 multipliers

    # HF: Q[0]=1, K[0]=-gain (text) or +gain (control), scaling=1.0
    hf_logits = torch.full((seq_len,), -gain, dtype=torch.bfloat16)
    hf_logits[ctrl_pos] = gain_bf16

    # vLLM: pre-scale K[0]=±effective_gain, then multiplied by attention_multiplier
    vllm_pre_scale = torch.full(
        (seq_len,), -effective_gain.item(), dtype=torch.bfloat16
    )
    vllm_pre_scale[ctrl_pos] = effective_gain
    vllm_logits = vllm_pre_scale * mult_bf16

    # V[0] = 0 for text, adapter_id (as float) for control
    values = torch.zeros(seq_len, dtype=torch.bfloat16)
    values[ctrl_pos] = float(adapter_id)

    return hf_logits, vllm_logits, values


class TestSoftmaxAdapterRecovery:
    """Both geometries must recover the same adapter ID via round(softmax @ V).

    End-to-end property SingleSwitch relies on: regardless of compensation path,
    the rounded weighted sum must equal the exact adapter ID.
    """

    @pytest.mark.parametrize("attention_multiplier", PRODUCTION_ATTENTION_MULTIPLIERS)
    @pytest.mark.parametrize("adapter_id", ADAPTER_IDS)
    @pytest.mark.parametrize("seq_len", [10, 100, 1000, 10000, 32768, 65536])
    def test_both_geometries_recover_adapter(
        self,
        attention_multiplier,
        adapter_id,
        seq_len,
    ):
        hf_logits, vllm_logits, values = _build_logits_and_values(
            seq_len,
            adapter_id,
            attention_multiplier,
        )

        # Softmax in float32 (matches attention-kernel practice) for fairness;
        # logits were produced in bf16 so bf16 rounding is still exercised.
        hf_weights = torch.softmax(hf_logits.float(), dim=0)
        vllm_weights = torch.softmax(vllm_logits.float(), dim=0)

        hf_recovered = round((hf_weights * values.float()).sum().item())
        vllm_recovered = round((vllm_weights * values.float()).sum().item())

        assert hf_recovered == adapter_id, (
            f"HF path failed: recovered={hf_recovered}, expected={adapter_id} "
            f"at seq_len={seq_len}, mult={attention_multiplier}"
        )
        assert vllm_recovered == adapter_id, (
            f"vLLM path failed: recovered={vllm_recovered}, expected={adapter_id} "
            f"at seq_len={seq_len}, mult={attention_multiplier}"
        )


class TestSoftmaxWeightDistribution:
    """HF and vLLM attention weight vectors must agree within bf16 tolerance.

    Catches drift in gain-compensation math: any rounding divergence between
    `(1.0 × ±gain)` and `(attention_multiplier × ±effective_gain)` would show
    up as a per-position weight difference after softmax.
    """

    @pytest.mark.parametrize("attention_multiplier", PRODUCTION_ATTENTION_MULTIPLIERS)
    @pytest.mark.parametrize("seq_len", [10, 100, 1000, 10000, 32768, 65536])
    def test_weight_distributions_match(self, attention_multiplier, seq_len):
        hf_logits, vllm_logits, _ = _build_logits_and_values(
            seq_len,
            adapter_id=1,
            attention_multiplier=attention_multiplier,
        )

        hf_weights = torch.softmax(hf_logits.float(), dim=0)
        vllm_weights = torch.softmax(vllm_logits.float(), dim=0)

        max_abs_diff = (hf_weights - vllm_weights).abs().max().item()
        assert max_abs_diff < 1e-3, (
            f"HF vs vLLM softmax weights diverged by {max_abs_diff} at "
            f"seq_len={seq_len}, mult={attention_multiplier}"
        )


class TestControlTokenDominance:
    """Control token must hold >99.9% of softmax weight at every tested seq_len.

    Mathematical basis for sharpness — with gain=15 the non-control mass is
    ~(seq_len-1) × exp(-30), which is ~10⁻⁸ even at MAX_POSITION_EMBEDDINGS.
    """

    @pytest.mark.parametrize("attention_multiplier", PRODUCTION_ATTENTION_MULTIPLIERS)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_control_dominates_softmax(self, attention_multiplier, seq_len):
        _, vllm_logits, _ = _build_logits_and_values(
            seq_len,
            adapter_id=1,
            attention_multiplier=attention_multiplier,
        )
        weights = torch.softmax(vllm_logits.float(), dim=0)
        ctrl_weight = weights[1].item()
        assert ctrl_weight > 0.999, (
            f"Control token dominance broken: weight={ctrl_weight} at "
            f"seq_len={seq_len}, mult={attention_multiplier}"
        )
