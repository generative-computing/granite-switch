# SPDX-License-Identifier: Apache-2.0
"""Shared test cases for SingleSwitch (HF and vLLM).

Defines test-case mixins that both backends inherit. Each backend provides
only a ``_run(seq, num_adapters=32) -> list[int]`` adapter that abstracts
away tensor shapes, device placement, and attention setup.

Adding a test here automatically covers both backends.

SingleSwitch behaviour recap:
- Single-switch mechanism: one control token type per sequence
- Differential-gain attention averages all control tokens → mixed types
  give undefined results, so every test uses a single adapter type
- 32 adapters (IDs 1..32) stress-test rounding precision at higher indices
"""

import pytest

NUM_ADAPTERS = 32
TEXT_TOKEN = 50
# Adapter token IDs (hidden flavor): 1000..1031 for adapters 1..32
ADAPTER_TOKEN_IDS_LIST = list(range(1000, 1000 + NUM_ADAPTERS))


class SingleSwitchTokenMatchingCases:
    """Each adapter token → correct index; non-adapter tokens → 0.

    Subclass must implement ``_run(seq, num_adapters=32) -> list[int]``.
    """

    @pytest.mark.parametrize("adapter_idx", range(4))
    def test_each_adapter_maps_to_correct_index(self, adapter_idx):
        """Each adapter token maps to its 1-indexed adapter ID (separate sequences)."""
        expected_id = adapter_idx + 1
        seq = [TEXT_TOKEN, ADAPTER_TOKEN_IDS_LIST[adapter_idx], TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, num_adapters=4)
        assert result[2] == expected_id
        assert result[3] == expected_id

    def test_non_adapter_tokens_produce_zero(self):
        """Non-adapter tokens before any control token → 0."""
        seq = [TEXT_TOKEN, 51, 52, 53, 54]
        result = self._run(seq, num_adapters=4)
        assert all(v == 0 for v in result)

    @pytest.mark.parametrize("adapter_idx", range(NUM_ADAPTERS))
    def test_every_adapter_id_exact(self, adapter_idx):
        """All 32 adapter IDs round-trip exactly under gain compensation."""
        expected_id = adapter_idx + 1
        seq = (
            [TEXT_TOKEN] * 10
            + [ADAPTER_TOKEN_IDS_LIST[adapter_idx]]
            + [TEXT_TOKEN] * 10
        )
        result = self._run(seq)
        assert all(v == 0 for v in result[:10])
        assert all(v == expected_id for v in result[10:])

    def test_unregistered_token_produces_zero(self):
        """Token IDs just outside the adapter range are not recognized."""
        seq = [TEXT_TOKEN, 999, 1032, TEXT_TOKEN]
        result = self._run(seq, num_adapters=4)
        assert all(v == 0 for v in result)


class SingleSwitchAdapterRetrievalCases:
    """Single control token persists through subsequent positions.

    Subclass must implement ``_run(seq, num_adapters=32) -> list[int]``.
    """

    def test_single_switch_persists(self):
        """After one control token, all subsequent positions return its adapter ID."""
        seq = [
            TEXT_TOKEN,
            ADAPTER_TOKEN_IDS_LIST[2],
            TEXT_TOKEN,
            TEXT_TOKEN,
            TEXT_TOKEN,
            TEXT_TOKEN,
        ]
        result = self._run(seq, num_adapters=4)
        assert all(v == 3 for v in result[2:])

    def test_control_token_own_position(self):
        """Adapter ID is correct at the control token's own position."""
        seq = [TEXT_TOKEN, ADAPTER_TOKEN_IDS_LIST[2], TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, num_adapters=4)
        assert result[1] == 3

    def test_duplicate_control_tokens(self):
        """Same adapter token appearing twice still returns correct ID."""
        seq = [
            TEXT_TOKEN,
            ADAPTER_TOKEN_IDS_LIST[0],
            TEXT_TOKEN,
            ADAPTER_TOKEN_IDS_LIST[0],
            TEXT_TOKEN,
        ]
        result = self._run(seq, num_adapters=4)
        assert all(v == 1 for v in result[1:])


class SingleSwitchEdgeCases:
    """Edge cases: position-0 control token, single adapter, minimal lengths.

    Subclass must implement ``_run(seq, num_adapters=32) -> list[int]``.
    """

    def test_control_token_at_position_zero(self):
        """Control token as the very first token in the sequence."""
        seq = [ADAPTER_TOKEN_IDS_LIST[0], TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, num_adapters=4)
        assert all(v == 1 for v in result)

    def test_single_adapter(self):
        """num_adapters=1: minimal adapter configuration."""
        seq = [TEXT_TOKEN, ADAPTER_TOKEN_IDS_LIST[0], TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, num_adapters=1)
        assert result[0] == 0
        assert all(v == 1 for v in result[1:])

    def test_seq_len_one_text(self):
        """Single text token → [0]."""
        result = self._run([TEXT_TOKEN], num_adapters=4)
        assert result == [0]

    def test_seq_len_one_control(self):
        """Single control token → [adapter_id]."""
        result = self._run([ADAPTER_TOKEN_IDS_LIST[2]], num_adapters=4)
        assert result == [3]

    def test_seq_len_two(self):
        """[control, text] → both get the adapter ID."""
        seq = [ADAPTER_TOKEN_IDS_LIST[1], TEXT_TOKEN]
        result = self._run(seq, num_adapters=4)
        assert result == [2, 2]

    def test_control_token_at_last_position(self):
        """Control token as the final token gets the correct adapter ID."""
        seq = [TEXT_TOKEN] * 9 + [ADAPTER_TOKEN_IDS_LIST[2]]
        result = self._run(seq, num_adapters=4)
        assert all(v == 0 for v in result[:9])
        assert result[9] == 3

    def test_mixed_adapters_no_crash(self):
        """Two different adapter tokens: must not crash or produce out-of-range."""
        seq = [
            TEXT_TOKEN,
            ADAPTER_TOKEN_IDS_LIST[0],
            ADAPTER_TOKEN_IDS_LIST[1],
            TEXT_TOKEN,
            TEXT_TOKEN,
        ]
        result = self._run(seq, num_adapters=4)
        assert all(0 <= v <= 4 for v in result)

    def test_long_all_text_produces_zero(self):
        """1000 text tokens with no adapters — no spurious activations."""
        seq = [TEXT_TOKEN] * 1000
        result = self._run(seq, num_adapters=4)
        assert all(v == 0 for v in result)


class SingleSwitchShapeCorrectnessCases:
    """Output length matches input for various sequence lengths.

    Subclass must implement ``_run(seq, num_adapters=32) -> list[int]``.
    """

    @pytest.mark.parametrize("seq_len", [1, 2, 5, 10, 20, 50, 100])
    def test_output_length_matches_input(self, seq_len):
        seq = [TEXT_TOKEN] * seq_len
        result = self._run(seq)
        assert len(result) == seq_len


class SingleSwitchContextLengthSweepCases:
    """Single control token at varying positions across varying context lengths.

    Verifies:
    - Forward persistence: adapter survives long text tails after control
    - Late activation: control token works correctly deep into sequence
    - All 32 adapters: no index-specific rounding failures

    Subclass must implement ``_run(seq, num_adapters=32) -> list[int]``.
    """

    @pytest.mark.parametrize("adapter_idx", range(NUM_ADAPTERS))
    @pytest.mark.parametrize(
        "context_length,control_position",
        [
            (100, "early"),
            (100, "mid"),
            (100, "late"),
            (1000, "early"),
            (1000, "mid"),
            (1000, "late"),
            (10000, "early"),
            (10000, "mid"),
            (10000, "late"),
        ],
    )
    def test_single_switch_at_distance(
        self, context_length, control_position, adapter_idx
    ):
        """One control token, rest text. Verify adapter persists to end."""
        if control_position == "early":
            ctrl_pos = 1
        elif control_position == "mid":
            ctrl_pos = context_length // 2
        else:  # late
            ctrl_pos = context_length - 10

        seq = [TEXT_TOKEN] * context_length
        seq[ctrl_pos] = ADAPTER_TOKEN_IDS_LIST[adapter_idx]
        expected_id = adapter_idx + 1

        result = self._run(seq)

        # Before control token: should be 0 (base)
        if ctrl_pos > 0:
            pre = result[:ctrl_pos]
            failures = sum(1 for v in pre if v != 0)
            assert failures == 0, (
                f"Pre-control: {failures}/{ctrl_pos} non-zero for adapter {expected_id} "
                f"at pos {ctrl_pos} in context {context_length}"
            )

        # At and after control token: should be the adapter ID
        post = result[ctrl_pos:]
        failures = sum(1 for v in post if v != expected_id)
        assert failures == 0, (
            f"Post-control: {failures}/{len(post)} wrong for adapter {expected_id} "
            f"at pos {ctrl_pos} in context {context_length}"
        )

    @pytest.mark.parametrize("adapter_idx", [0, 15, 31])
    @pytest.mark.parametrize("context_length", [32768, 50000, 65536, 100000, 131072])
    def test_long_context_sweep(self, context_length, adapter_idx):
        """Very long context (32K-131K) with early control token.

        Covers common deployment context lengths (32K, 64K) and the model's
        declared max_position_embeddings (131K). Reduced adapter parametrization
        (3 representative adapters) keeps runtime manageable at these lengths.
        """
        ctrl_pos = 1
        seq = [TEXT_TOKEN] * context_length
        seq[ctrl_pos] = ADAPTER_TOKEN_IDS_LIST[adapter_idx]
        expected_id = adapter_idx + 1

        result = self._run(seq)

        assert result[0] == 0, "Position before control should be 0"
        post = result[ctrl_pos:]
        failures = sum(1 for v in post if v != expected_id)
        assert failures == 0, (
            f"Post-control: {failures}/{len(post)} wrong for adapter {expected_id} "
            f"in context {context_length}"
        )

    @pytest.mark.parametrize("adapter_idx", [0, 15, 31])
    def test_high_adapter_at_long_context(self, adapter_idx):
        """Gain-compensated geometry preserves precision at 10K for high adapter IDs."""
        seq = [TEXT_TOKEN] * 10000
        seq[1] = ADAPTER_TOKEN_IDS_LIST[adapter_idx]
        expected_id = adapter_idx + 1

        result = self._run(seq)

        assert result[0] == 0
        post = result[1:]
        failures = sum(1 for v in post if v != expected_id)
        assert failures == 0, (
            f"Post-control: {failures}/{len(post)} wrong for adapter {expected_id}"
        )


class SingleSwitchGainSensitivityCases:
    """Demonstrate that low gain breaks retrieval at long context.

    With gain=1 the softmax logit gap between control and non-control tokens
    is only 2 (= +1 vs -1).  Over a long sequence with many non-control
    positions, the non-control mass overwhelms the single control token and
    the rounded output drifts toward 0.

    Subclass must implement
    ``_run(seq, num_adapters=32, control_token_gain=15.0) -> list[int]``.
    """

    def test_low_gain_fails_at_long_context(self):
        """gain=1, context=10000: adapter signal is washed out."""
        context_length = 10000
        seq = [TEXT_TOKEN] * context_length
        seq[1] = ADAPTER_TOKEN_IDS_LIST[0]

        result = self._run(seq, num_adapters=4, control_token_gain=1.0)

        # With gain=1 the tail positions should NOT reliably return adapter 1.
        # We check that at least some of the tail positions have drifted to 0.
        tail = result[context_length // 2 :]
        wrong = sum(1 for v in tail if v != 1)
        assert wrong > 0, (
            "Expected low-gain degradation at context 10K but all tail "
            "positions returned the correct adapter — gain may be too high"
        )

    def test_high_gain_survives_long_context(self):
        """gain=15 (default), context=10000: adapter signal is preserved."""
        context_length = 10000
        seq = [TEXT_TOKEN] * context_length
        seq[1] = ADAPTER_TOKEN_IDS_LIST[0]

        result = self._run(seq, num_adapters=4, control_token_gain=15.0)

        post = result[1:]
        failures = sum(1 for v in post if v != 1)
        assert failures == 0, (
            f"gain=15 should preserve signal but got {failures}/{len(post)} failures"
        )
