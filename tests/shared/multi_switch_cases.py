# SPDX-License-Identifier: Apache-2.0
"""Shared test cases for the MultiSwitch engine (HF and vLLM).

Mirror of ``tests/shared/single_switch_cases.py`` for the native
multi-transition Kerdock/DG coded-memory engine (``switch_type="multi"``):
coded-memory routing over two tiny attention heads (counting + memory).

The engine adds *multi-transition* semantics on top of SingleSwitch: the
adapter selected at each position is the expert written by the **most recent
control token at or before that position** — exact for arbitrarily many
transitions per request. SingleSwitch's ±gain attention *averages* competing
control tokens, so an A(higher)→B(lower) sequence mis-routes; the multi engine
picks latest-wins exactly.

Each backend provides only a ``_run(seq, adapter_token_ids) -> list[int]``
adapter that abstracts away tensor shapes, device placement, and attention
setup. Adding a test here automatically covers both backends.

Two ``adapter_token_ids`` layouts are exercised (both are valid per the engine
docstrings):

* **No-base-slot** (real composed checkpoints): ``len == num_adapters``.
  ``adapter_token_ids[k]`` fires adapter ``k+1``; index 0 = base and there is
  no way to re-select base mid-sequence.
* **Base-reset** (return-to-base): ``len == num_adapters + 1``. Produced by
  composing with ``--base-reset-token``, which prepends ``<|base_reset|>``;
  without the flag ``add_control_tokens`` emits exactly one token per discovered
  adapter and a checkpoint has ``len == num_adapters`` and offset 1. The cases
  below still build the layout BY HAND, so they validate the ENGINE in isolation
  -- the compose side is covered separately by
  ``tests/composer/test_base_reset_token.py``.
  Ordinary aLoRA chat does not need it (the template emits one control token for
  the current turn only, so prior turns route to base already); an explicit base
  token is needed for agentic per-step switching within one sequence.

  ``adapter_token_ids[0]`` is a base-reset token that fires expert 0; slots
  ``1..N`` fire experts ``1..N``. This lets a control token re-select base.

Token-id conventions (kept disjoint from real vocab text tokens):
- ``TEXT_TOKEN = 50``    — a non-control token.
- ``A_TOK = 101``        — control token firing adapter 1.
- ``B_TOK = 102``        — control token firing adapter 2.
- ``BASE_TOK = 100``     — base-reset control token (base-reset layout only).
- ``ATOK_NO_BASE = [101, 102]``       — no-base-slot layout (num_adapters=2).
- ``ATOK_BASE_RESET = [100, 101, 102]`` — base-reset layout (num_adapters=2).
"""

import pytest

TEXT_TOKEN = 50
BASE_TOK = 100
A_TOK = 101
B_TOK = 102

# num_adapters == 2 in every shared case below.
NUM_ADAPTERS = 2

# No-base-slot layout: adapter_token_ids[k] -> adapter k+1.
ATOK_NO_BASE = [A_TOK, B_TOK]
# Base-reset layout: slot 0 fires expert 0 (base); slots 1..N fire 1..N.
ATOK_BASE_RESET = [BASE_TOK, A_TOK, B_TOK]


class MultiSwitchTransitionCases:
    """Multi-transition routing: latest control token wins at every position.

    These are the cases the coded engine adds over SingleSwitch. Subclass
    must implement ``_run(seq, adapter_token_ids) -> list[int]``.
    """

    def test_base_then_A_then_B(self):
        """base -> A -> B: [50,101,50,102,50] -> [0,1,1,2,2]."""
        seq = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert result == [0, 1, 1, 2, 2]

    def test_higher_then_lower_A2_then_B1(self):
        """A(2) -> B(1) higher-then-lower: [102,50,101,50] -> [2,2,1,1].

        This is the sequence SingleSwitch mis-routes: its ±gain attention
        averages the two control values instead of taking the most recent, so
        the tail drifts toward a blend of 1 and 2 rather than the correct 1.
        The multi engine takes latest-wins, so the answer is exact.
        """
        seq = [B_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert result == [2, 2, 1, 1]

    def test_three_transitions_A_B_A(self):
        """A -> B -> A: latest-wins across three transitions."""
        seq = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert result == [0, 1, 1, 2, 2, 1, 1]

    def test_adjacent_control_tokens(self):
        """Two control tokens back-to-back: the second wins from its position."""
        seq = [TEXT_TOKEN, A_TOK, B_TOK, TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert result == [0, 1, 2, 2, 2]


class MultiSwitchReturnToBaseCases:
    """Base-reset layout: a control token can re-select base mid-sequence.

    Subclass must implement ``_run(seq, adapter_token_ids) -> list[int]``.
    """

    def test_A_then_return_to_base(self):
        """A -> base-reset: [101,50,100,50] -> [1,1,0,0]."""
        seq = [A_TOK, TEXT_TOKEN, BASE_TOK, TEXT_TOKEN]
        result = self._run(seq, ATOK_BASE_RESET)
        assert result == [1, 1, 0, 0]

    def test_A_B_base_A(self):
        """Full round trip: A -> B -> base -> A."""
        seq = [A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN, BASE_TOK, TEXT_TOKEN, A_TOK]
        result = self._run(seq, ATOK_BASE_RESET)
        assert result == [1, 1, 2, 2, 0, 0, 1]

    def test_base_reset_at_position_zero(self):
        """Base-reset token at pos 0 keeps everything at base until a real one."""
        seq = [BASE_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        result = self._run(seq, ATOK_BASE_RESET)
        assert result == [0, 0, 1, 1]


class MultiSwitchStickyCases:
    """A single transition persists to the end of the sequence.

    Subclass must implement ``_run(seq, adapter_token_ids) -> list[int]``.
    """

    def test_single_transition_sticky(self):
        """One control token -> its adapter persists through all later text."""
        seq = [TEXT_TOKEN, TEXT_TOKEN, A_TOK] + [TEXT_TOKEN] * 6
        result = self._run(seq, ATOK_NO_BASE)
        assert result[:2] == [0, 0]
        assert all(v == 1 for v in result[2:])

    def test_control_token_own_position(self):
        """The adapter is active at the control token's own position."""
        seq = [TEXT_TOKEN, B_TOK, TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert result[1] == 2
        assert all(v == 2 for v in result[1:])

    def test_control_token_at_last_position(self):
        """A control token as the final token activates only at that position."""
        seq = [TEXT_TOKEN] * 5 + [A_TOK]
        result = self._run(seq, ATOK_NO_BASE)
        assert result[:5] == [0, 0, 0, 0, 0]
        assert result[5] == 1


class MultiSwitchEdgeCases:
    """Edge cases: all-base, control-at-pos-0, minimal lengths.

    Subclass must implement ``_run(seq, adapter_token_ids) -> list[int]``.
    """

    def test_non_control_all_zero(self):
        """No control tokens anywhere -> every position is base (0)."""
        seq = [TEXT_TOKEN, 51, 52, 53, 54]
        result = self._run(seq, ATOK_NO_BASE)
        assert all(v == 0 for v in result)

    def test_unregistered_tokens_produce_zero(self):
        """Token ids adjacent to the control ids are not recognized."""
        seq = [TEXT_TOKEN, 99, 103, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert all(v == 0 for v in result)

    def test_control_token_at_position_zero(self):
        """Control token as the very first token -> adapter active everywhere."""
        seq = [A_TOK, TEXT_TOKEN, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert all(v == 1 for v in result)

    def test_seq_len_one_text(self):
        """Single text token -> [0]."""
        result = self._run([TEXT_TOKEN], ATOK_NO_BASE)
        assert result == [0]

    def test_seq_len_one_control(self):
        """Single control token -> [adapter_id]."""
        result = self._run([B_TOK], ATOK_NO_BASE)
        assert result == [2]

    def test_seq_len_two_control_text(self):
        """[control, text] -> both get the adapter id."""
        result = self._run([A_TOK, TEXT_TOKEN], ATOK_NO_BASE)
        assert result == [1, 1]

    def test_seq_len_two_text_control(self):
        """[text, control] -> [0, adapter_id]."""
        result = self._run([TEXT_TOKEN, B_TOK], ATOK_NO_BASE)
        assert result == [0, 2]

    def test_long_all_text_produces_zero(self):
        """A long all-text sequence produces no spurious activations."""
        seq = [TEXT_TOKEN] * 500
        result = self._run(seq, ATOK_NO_BASE)
        assert all(v == 0 for v in result)

    def test_duplicate_same_adapter(self):
        """The same adapter token twice keeps that adapter active throughout."""
        seq = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN]
        result = self._run(seq, ATOK_NO_BASE)
        assert result[0] == 0
        assert all(v == 1 for v in result[1:])


class MultiSwitchShapeCorrectnessCases:
    """Output length matches input for various sequence lengths.

    Subclass must implement ``_run(seq, adapter_token_ids) -> list[int]``.
    """

    @pytest.mark.parametrize("seq_len", [1, 2, 5, 10, 20, 50, 100])
    def test_output_length_matches_input(self, seq_len):
        seq = [TEXT_TOKEN] * seq_len
        result = self._run(seq, ATOK_NO_BASE)
        assert len(result) == seq_len

    @pytest.mark.parametrize("seq_len", [3, 8, 33, 64, 129])
    def test_output_length_with_transitions(self, seq_len):
        """A base->A->B sequence padded to seq_len keeps len(result) == seq_len."""
        seq = [TEXT_TOKEN] * seq_len
        if seq_len >= 2:
            seq[1] = A_TOK
        if seq_len >= 3:
            seq[2] = B_TOK
        result = self._run(seq, ATOK_NO_BASE)
        assert len(result) == seq_len
