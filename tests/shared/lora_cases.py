# SPDX-License-Identifier: Apache-2.0
"""Shared test cases for SwitchedLoRALinear (HF and vLLM).

Defines test-case mixins that both backends inherit. Each backend provides
only ``_make_layer()`` and ``_run()`` adapters that abstract away layer
construction and forward-pass invocation.

Adding a test here automatically covers both backends.

SwitchedLoRALinear behaviour recap:
- adapter_indices=0 → base layer only (no LoRA)
- adapter_indices=1+ → base + LoRA for that adapter
- lora_B is pre-scaled by (alpha/rank) at load time; runtime scaling = 1.0
- LoRA math: output = base(x) + x @ A^T @ B^T  (for active adapter tokens)

Contract for subclasses:
- ``_make_layer(in_features, out_features, num_adapters, rank) -> layer``
  Creates a SwitchedLoRALinear with random weights.
- ``_run(layer, x, adapter_indices) -> Tensor``
  Calls forward, returns output tensor.
"""

import pytest
import torch

# ── Defaults ────────────────────────────────────────────────────────

IN_FEATURES = 32
OUT_FEATURES = 16
NUM_ADAPTERS = 4
RANK = 8
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────


def _seeded_input(batch_size, seq_len, features, seed=SEED):
    """Create reproducible random input."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, features)


def _single_sequence_result(run_fn, layer, x_row, adapter_row):
    """Run a single sequence through the layer (1, seq_len, features)."""
    x_single = x_row.unsqueeze(0)
    idx_single = adapter_row.unsqueeze(0)
    return run_fn(layer, x_single, idx_single).squeeze(0)


# ── 1. Base passthrough ────────────────────────────────────────────


class LoRABasePassthroughCases:
    """adapter_indices=0 everywhere → output equals base_layer(x).

    Subclass must implement ``_make_layer`` and ``_run``.
    """

    def test_all_base_equals_base_layer(self):
        """Output matches base_layer(x) exactly when all indices are 0."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x = _seeded_input(2, 5, IN_FEATURES, seed=SEED + 1)
        adapter_indices = torch.zeros(2, 5, dtype=torch.long)

        output = self._run(layer, x, adapter_indices)
        expected = layer.base_layer(x)

        torch.testing.assert_close(output, expected)

    def test_early_exit_no_lora(self):
        """All-base triggers early exit — same result as base_layer."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x = _seeded_input(1, 10, IN_FEATURES, seed=SEED + 2)
        adapter_indices = torch.zeros(1, 10, dtype=torch.long)

        output = self._run(layer, x, adapter_indices)
        expected = layer.base_layer(x)

        torch.testing.assert_close(output, expected)


# ── 2. Adapter activation ──────────────────────────────────────────


class LoRAAdapterActivationCases:
    """adapter_indices > 0 → LoRA modifies output.

    Subclass must implement ``_make_layer`` and ``_run``.
    """

    def test_adapter_modifies_output(self):
        """Tokens with adapter_indices > 0 produce different output than base."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        # lora_B is zero-initialized, so we must set non-zero weights
        # to get a non-zero LoRA delta
        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = _seeded_input(1, 4, IN_FEATURES, seed=SEED + 3)

        base_indices = torch.zeros(1, 4, dtype=torch.long)
        adapter_indices = torch.ones(1, 4, dtype=torch.long)  # adapter 1

        base_output = self._run(layer, x, base_indices)
        adapter_output = self._run(layer, x, adapter_indices)

        assert not torch.allclose(base_output, adapter_output), (
            "Adapter output should differ from base output"
        )

    def test_different_adapters_produce_different_outputs(self):
        """Adapter 1 ≠ adapter 2."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        # lora_B is zero-initialized; set distinct non-zero weights per adapter
        with torch.no_grad():
            for a in range(NUM_ADAPTERS):
                layer.lora_B.data[a] = torch.randn_like(layer.lora_B[a]) * 0.1 * (a + 1)

        x = _seeded_input(1, 4, IN_FEATURES, seed=SEED + 4)

        indices_1 = torch.ones(1, 4, dtype=torch.long)  # adapter 1
        indices_2 = torch.full((1, 4), 2, dtype=torch.long)  # adapter 2

        out_1 = self._run(layer, x, indices_1)
        out_2 = self._run(layer, x, indices_2)

        assert not torch.allclose(out_1, out_2), (
            "Different adapters should produce different outputs"
        )

    def test_base_tokens_unchanged_in_mixed_batch(self):
        """Tokens with index 0 get exact base output even when others use adapters."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        # Set non-zero lora_B so adapter tokens actually differ from base
        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = _seeded_input(1, 6, IN_FEATURES, seed=SEED + 5)

        # Mixed: [0, 1, 0, 2, 0, 0]
        mixed_indices = torch.tensor([[0, 1, 0, 2, 0, 0]], dtype=torch.long)
        all_base_indices = torch.zeros(1, 6, dtype=torch.long)

        mixed_output = self._run(layer, x, mixed_indices)
        base_output = self._run(layer, x, all_base_indices)

        # Tokens at positions 0, 2, 4, 5 should be identical
        base_positions = [0, 2, 4, 5]
        for pos in base_positions:
            torch.testing.assert_close(
                mixed_output[0, pos],
                base_output[0, pos],
                msg=f"Base token at position {pos} should be unchanged",
            )


# ── 3. Batch independence ──────────────────────────────────────────


class LoRABatchIndependenceCases:
    """Batches with mixed adapter assignments — no cross-contamination.

    Oracle: single-sequence result == batched result for each row.

    Subclass must implement ``_make_layer`` and ``_run``.
    """

    def test_batch_each_sequence_different_adapter(self):
        """Batch of 4 sequences, each using a different adapter (0, 1, 2, 3)."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        # Set non-zero lora_B so adapter tokens produce distinct LoRA deltas
        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = _seeded_input(4, 5, IN_FEATURES, seed=SEED + 6)

        adapter_indices = torch.tensor(
            [
                [0, 0, 0, 0, 0],  # all base
                [1, 1, 1, 1, 1],  # all adapter 1
                [2, 2, 2, 2, 2],  # all adapter 2
                [3, 3, 3, 3, 3],  # all adapter 3
            ],
            dtype=torch.long,
        )

        batched_output = self._run(layer, x, adapter_indices)

        for i in range(4):
            single = _single_sequence_result(self._run, layer, x[i], adapter_indices[i])
            torch.testing.assert_close(
                batched_output[i],
                single,
                msg=f"Row {i} batched result should match single-sequence result",
            )

    def test_batch_base_vs_adapter_isolation(self):
        """Batch of 2: one all-base, one all-adapter. No cross-talk."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = _seeded_input(2, 5, IN_FEATURES, seed=SEED + 7)

        adapter_indices = torch.tensor(
            [
                [0, 0, 0, 0, 0],  # all base
                [2, 2, 2, 2, 2],  # all adapter 2
            ],
            dtype=torch.long,
        )

        batched_output = self._run(layer, x, adapter_indices)

        # Base row must be bitwise identical to running base alone
        base_single = _single_sequence_result(
            self._run, layer, x[0], adapter_indices[0]
        )
        torch.testing.assert_close(batched_output[0], base_single)

        # Adapter row must match running adapter alone
        adapter_single = _single_sequence_result(
            self._run, layer, x[1], adapter_indices[1]
        )
        torch.testing.assert_close(batched_output[1], adapter_single)

    def test_batch_mixed_within_sequence(self):
        """Single batch element with per-token adapter switching."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = _seeded_input(1, 5, IN_FEATURES, seed=SEED + 8)

        # Vary adapter per-token: [0, 1, 0, 2, 0]
        adapter_indices = torch.tensor([[0, 1, 0, 2, 0]], dtype=torch.long)

        output = self._run(layer, x, adapter_indices)

        # Verify each token matches single-adapter result
        base_output = layer.base_layer(x)
        for pos in range(5):
            aid = adapter_indices[0, pos].item()
            if aid == 0:
                # Base token
                torch.testing.assert_close(
                    output[0, pos],
                    base_output[0, pos],
                    msg=f"Token {pos} (base) should match base_layer output",
                )
            else:
                # Adapter token — run single-token to get reference
                x_token = x[0, pos : pos + 1].unsqueeze(0)  # (1, 1, features)
                idx_token = torch.tensor([[aid]], dtype=torch.long)
                ref = self._run(layer, x_token, idx_token)
                torch.testing.assert_close(
                    output[0, pos],
                    ref[0, 0],
                    msg=f"Token {pos} (adapter {aid}) should match single-token result",
                )

    def test_batch_all_adapters_simultaneously(self):
        """Batch with every adapter active at once (one sequence per adapter + base)."""
        torch.manual_seed(SEED)
        num_adapters = NUM_ADAPTERS
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, num_adapters, RANK)

        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        seq_len = 4
        batch_size = num_adapters + 1  # base + all adapters

        x = _seeded_input(batch_size, seq_len, IN_FEATURES, seed=SEED + 9)

        # Row 0: all base; row i: all adapter i
        adapter_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i in range(1, batch_size):
            adapter_indices[i] = i  # adapter i (1-indexed)

        batched_output = self._run(layer, x, adapter_indices)

        for i in range(batch_size):
            single = _single_sequence_result(self._run, layer, x[i], adapter_indices[i])
            torch.testing.assert_close(
                batched_output[i],
                single,
                msg=f"Row {i} (adapter={adapter_indices[i, 0].item()}) "
                f"should match single-sequence result",
            )

    def test_batch_all_sequences_switching_within(self):
        """Batch of N sequences where every row has intra-sequence switching."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = _seeded_input(3, 5, IN_FEATURES, seed=SEED + 10)

        adapter_indices = torch.tensor(
            [
                [0, 1, 0, 2, 0],  # seq 0: base/adapter1/base/adapter2/base
                [2, 0, 1, 1, 0],  # seq 1: adapter2/base/adapter1/adapter1/base
                [0, 0, 3, 0, 1],  # seq 2: base/base/adapter3/base/adapter1
            ],
            dtype=torch.long,
        )

        batched_output = self._run(layer, x, adapter_indices)

        # Verify every token in every row matches single-token result
        for row in range(3):
            for pos in range(5):
                aid = adapter_indices[row, pos].item()
                x_token = x[row, pos : pos + 1].unsqueeze(0)
                idx_token = torch.tensor([[aid]], dtype=torch.long)
                ref = self._run(layer, x_token, idx_token)
                torch.testing.assert_close(
                    batched_output[row, pos],
                    ref[0, 0],
                    msg=f"Row {row}, pos {pos} (adapter={aid}) "
                    f"should match single-token result",
                )


# ── 4. Math correctness ────────────────────────────────────────────


class LoRAMathCorrectnessCases:
    """Known weights → verify exact LoRA math.

    output[token] = base(x) + x @ A^T @ B^T  (for adapter tokens)

    Subclass must implement ``_make_layer`` and ``_run``.
    """

    @pytest.mark.parametrize("num_adapters", [1, 2, 4])
    def test_lora_output_matches_manual_computation(self, num_adapters):
        """Manually set lora_A/B, verify output = base(x) + x @ A^T @ B^T."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, num_adapters, RANK)

        # Set known LoRA weights
        with torch.no_grad():
            for a in range(num_adapters):
                layer.lora_A.data[a, 0] = torch.eye(RANK, IN_FEATURES) * 0.1 * (a + 1)
                layer.lora_B.data[a, 0] = torch.eye(OUT_FEATURES, RANK) * 0.2 * (a + 1)

        x = _seeded_input(1, 3, IN_FEATURES, seed=SEED + 11)

        for adapter_id in range(1, num_adapters + 1):
            adapter_indices = torch.full((1, 3), adapter_id, dtype=torch.long)
            output = self._run(layer, x, adapter_indices)

            # Manual computation
            base_out = layer.base_layer(x)
            tensor_idx = adapter_id - 1
            lora_a = layer.lora_A[tensor_idx, 0]  # [rank, in_features]
            lora_b = layer.lora_B[tensor_idx, 0]  # [out_features, rank]
            lora_delta = x @ lora_a.t() @ lora_b.t()
            expected = base_out + lora_delta

            torch.testing.assert_close(
                output,
                expected,
                msg=f"Adapter {adapter_id}: output should match base + x @ A^T @ B^T",
            )

    def test_batch_math_correctness(self):
        """Known-weight verification with multi-row batch, different adapters per row."""
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        # Set known weights
        with torch.no_grad():
            for a in range(NUM_ADAPTERS):
                layer.lora_A.data[a, 0] = torch.eye(RANK, IN_FEATURES) * 0.1 * (a + 1)
                layer.lora_B.data[a, 0] = torch.eye(OUT_FEATURES, RANK) * 0.2 * (a + 1)

        x = _seeded_input(4, 3, IN_FEATURES, seed=SEED + 12)

        # Each row uses a different adapter: 0 (base), 1, 2, 3
        adapter_indices = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],
            dtype=torch.long,
        )

        output = self._run(layer, x, adapter_indices)
        base_out = layer.base_layer(x)

        for row in range(4):
            aid = adapter_indices[row, 0].item()
            if aid == 0:
                expected = base_out[row]
            else:
                tensor_idx = aid - 1
                lora_a = layer.lora_A[tensor_idx, 0]
                lora_b = layer.lora_B[tensor_idx, 0]
                lora_delta = x[row] @ lora_a.t() @ lora_b.t()
                expected = base_out[row] + lora_delta

            torch.testing.assert_close(
                output[row], expected, msg=f"Row {row} (adapter={aid}): math mismatch"
            )


# ── 5. Shape correctness ──────────────────────────────────────────


class LoRAShapeCorrectnessCases:
    """Output shape matches expected for various input shapes.

    Subclass must implement ``_make_layer`` and ``_run``.
    """

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 1),
            (1, 5),
            (1, 20),
            (2, 5),
            (4, 10),
            (8, 3),
        ],
    )
    def test_output_shape_matches_expected(self, batch_size, seq_len):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x = torch.randn(batch_size, seq_len, IN_FEATURES)
        adapter_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)

        output = self._run(layer, x, adapter_indices)
        assert output.shape == (batch_size, seq_len, OUT_FEATURES)
