# SPDX-License-Identifier: Apache-2.0
"""HF generation tests: smoke tests and KV cache consistency.

Validates autoregressive generation for SingleSwitch models.
Tests run on CPU with random weights (no pretrained checkpoint needed).

Key test: prefill-decode consistency verifies that feeding tokens one at a
time through the KV cache produces the same logits as a single full-prefill
forward pass.  This directly validates the past_key_values.update() fix in
SingleSwitch.
"""

import pytest
import torch

from tests.shared.generation_models import (
    DENSE_CFG,
    basic_overrides,
    make_switch_model,
)

pytestmark = pytest.mark.local_fast


# ── Helpers ───────────────────────────────────────────────────────

def _set_nonzero_lora_B(model, scale=0.1):
    """Set non-zero lora_B on every LoRA layer so adapters produce visible deltas."""
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.shared_mlp
            if hasattr(attn.qkv_proj, "lora_B_slices"):
                for b in attn.qkv_proj.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            if hasattr(attn.o_proj, "lora_B"):
                attn.o_proj.lora_B.data = torch.randn_like(attn.o_proj.lora_B) * scale
            if hasattr(mlp.input_linear, "lora_B_slices"):
                for b in mlp.input_linear.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            if hasattr(mlp.output_linear, "lora_B"):
                mlp.output_linear.lora_B.data = torch.randn_like(mlp.output_linear.lora_B) * scale


def _full_prefill_logits(model, input_ids):
    """Single forward pass on the full sequence.  Returns [1, seq, vocab]."""
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def _incremental_decode_logits(model, input_ids):
    """Feed tokens one at a time, accumulating the KV cache.

    Returns [1, seq, vocab] logits assembled from each step.
    If past_key_values.update() is missing from the switch, the switch
    will "forget" previous control tokens and produce wrong adapter indices
    during decode steps.
    """
    seq_len = input_ids.shape[1]
    all_logits = []
    past_key_values = None

    with torch.no_grad():
        for i in range(seq_len):
            token = input_ids[:, i : i + 1]  # [1, 1]
            cache_position = torch.tensor([i], dtype=torch.long)
            output = model(
                input_ids=token,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            all_logits.append(output.logits)  # [1, 1, vocab]
            past_key_values = output.past_key_values

    return torch.cat(all_logits, dim=1)  # [1, seq, vocab]


# ── SingleSwitch ───────────────────────────────────────────────────

class TestSingleSwitchGeneration:

    def _make(self, seed=42):
        model, cfg = make_switch_model(
            DENSE_CFG, basic_overrides(DENSE_CFG), seed=seed,
        )
        return model, cfg

    def test_generates_tokens(self):
        """Smoke: model.generate() produces the requested number of tokens."""
        model, _ = self._make()
        input_ids = torch.randint(0, 200, (1, 10))
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=16,
                do_sample=False,
                eos_token_id=None,
            )
        assert output.shape[1] == 10 + 16

    def test_prefill_decode_consistency(self):
        """Full-prefill logits match incremental-decode logits (with control token).

        This is the core KV cache validation.  Token 250 is adapter_token_ids[0],
        activating adapter 0 at position 2.  If the switch cache update is broken,
        decode steps after position 2 will see adapter_indices=0 instead of 1,
        producing completely different logits.
        """
        model, _ = self._make()
        input_ids = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])

        prefill = _full_prefill_logits(model, input_ids)
        incremental = _incremental_decode_logits(model, input_ids)

        torch.testing.assert_close(prefill, incremental, atol=1e-5, rtol=1e-4)

    def test_prefill_decode_consistency_no_control_token(self):
        """Sanity baseline: cache consistency without any control tokens."""
        model, _ = self._make()
        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])

        prefill = _full_prefill_logits(model, input_ids)
        incremental = _incremental_decode_logits(model, input_ids)

        torch.testing.assert_close(prefill, incremental, atol=1e-5, rtol=1e-4)

    def test_generates_with_control_token_in_prompt(self):
        """Generation with adapter in prompt produces different tokens than without."""
        model, _ = self._make()
        _set_nonzero_lora_B(model, scale=1.0)

        prompt_ctrl = torch.tensor([[10, 20, 250, 30, 40]])
        prompt_text = torch.tensor([[10, 20, 100, 30, 40]])

        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=8, do_sample=False, eos_token_id=None)
            out_ctrl = model.generate(input_ids=prompt_ctrl, **gen_kwargs)
            out_text = model.generate(input_ids=prompt_text, **gen_kwargs)

        assert out_ctrl.shape[1] == 5 + 8
        assert out_text.shape[1] == 5 + 8
        # With nonzero LoRA weights, adapter activation should change the output
        assert not torch.equal(out_ctrl[:, 5:], out_text[:, 5:])
