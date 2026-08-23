# SPDX-License-Identifier: Apache-2.0
"""Attributing traced rows to a conversation, when two are in flight at once.

The concurrency suite is GPU-gated, but the logic that decides *which*
conversation a traced row belongs to is pure and is the part that has produced
false results before. It is exercised here on synthetic records, on CPU, in
milliseconds.

Two traps, both previously observed on real runs:

  A decode row is one row wide, and its single token can coincidentally equal
  another prompt's token at the position it reports -- which produces a phantom
  "wrong routing" at an isolated position that moves between runs. Prompt
  positions may only be attributed from multi-token prefill spans.

  Two concurrent prompts made of prose share individual token values constantly.
  Matching per token would attribute rows to the wrong conversation; requiring
  EVERY row of a span to match the prompt at its own recorded position is what
  makes the decision unambiguous.
"""

import pytest

from tests.integration._conversation_concurrency_worker import (
    attribute_spans,
    decode_indices_by_seqlen,
    windows_disjoint,
)

PROMPT_A = [10, 11, 12, 13, 14, 15]
PROMPT_B = [10, 11, 99, 98, 97, 96]  # shares a two-token opening with A


def _prefill(prompt, adapter, lo=0, hi=None, tag="x"):
    """One forward carrying a whole prompt as a single request's span."""
    hi = len(prompt) if hi is None else hi
    ids = prompt[lo:hi]
    return {
        "tag": tag,
        "input_ids": ids,
        "positions": list(range(lo, hi)),
        "adapter_indices": [adapter] * len(ids),
        "query_start_loc": [0, len(ids)],
        "seq_lens": [hi],
    }


class TestSpanAttribution:
    def test_attributes_a_whole_prefill_span(self):
        got = attribute_spans([_prefill(PROMPT_A, 1)], PROMPT_A)
        assert got == {i: 1 for i in range(len(PROMPT_A))}

    def test_chunked_prefill_stitches_across_forwards(self):
        """A long prompt arrives as several spans; all of them must land."""
        records = [_prefill(PROMPT_A, 1, 0, 3), _prefill(PROMPT_A, 2, 3, 6)]
        got = attribute_spans(records, PROMPT_A)
        assert got == {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}

    def test_other_conversations_span_is_rejected(self):
        """B's span shares A's first two tokens but diverges, so it must not land."""
        got = attribute_spans([_prefill(PROMPT_B, 7)], PROMPT_A)
        assert got == {}, (
            "a span that disagrees with this prompt anywhere must be refused "
            "entirely; accepting its matching head would import another "
            "conversation's routing"
        )

    def test_two_requests_in_one_forward_are_split_by_span(self):
        """A batched forward carries both prompts; each gets only its own rows."""
        rec = {
            "tag": "batched",
            "input_ids": PROMPT_A + PROMPT_B,
            "positions": list(range(len(PROMPT_A))) + list(range(len(PROMPT_B))),
            "adapter_indices": [1] * len(PROMPT_A) + [2] * len(PROMPT_B),
            "query_start_loc": [0, len(PROMPT_A), len(PROMPT_A) + len(PROMPT_B)],
            "seq_lens": [len(PROMPT_A), len(PROMPT_B)],
        }
        assert attribute_spans([rec], PROMPT_A) == {i: 1 for i in range(6)}
        assert attribute_spans([rec], PROMPT_B) == {i: 2 for i in range(6)}

    def test_decode_row_never_claims_a_prompt_position(self):
        """The trap: a one-row span whose token matches the prompt where it says.

        Accepting it produced a phantom wrong-routing finding that moved between
        runs. It must be refused on width alone, before any token comparison.
        """
        decode = {
            "tag": "d",
            "input_ids": [PROMPT_A[3]],
            "positions": [3],
            "adapter_indices": [9],
            "query_start_loc": [0, 1],
            "seq_lens": [40],
        }
        assert attribute_spans([decode], PROMPT_A) == {}
        got = attribute_spans([_prefill(PROMPT_A, 1), decode], PROMPT_A)
        assert got[3] == 1, "the decode row must not overwrite the prefill's answer"

    @pytest.mark.parametrize(
        "bad",
        [
            {"positions": None},
            {"input_ids": []},
            {"adapter_indices": []},
            {"query_start_loc": [0, 99]},  # span longer than the row data
        ],
        ids=["no-positions", "no-ids", "no-indices", "span-past-the-end"],
    )
    def test_malformed_records_are_skipped_not_raised(self, bad):
        """A trace is best-effort telemetry; a missing field must not end the run."""
        rec = dict(_prefill(PROMPT_A, 1))
        rec.update(bad)
        assert attribute_spans([rec], PROMPT_A) == {}

    def test_positions_outside_the_prompt_are_ignored(self):
        """Rows from a longer request must not write past this prompt's end."""
        rec = _prefill([*PROMPT_A, 77, 78], 1)
        assert attribute_spans([rec], PROMPT_A) == {}


def _decode(seq_len, adapter, req=0, n_requests=1):
    """One decode row for request ``req`` of a forward carrying ``n_requests``."""
    return {
        "input_ids": [5] * n_requests,
        "positions": [seq_len - 1] * n_requests,
        "adapter_indices": [adapter + i for i in range(n_requests)],
        "query_start_loc": list(range(n_requests + 1)),
        "seq_lens": [seq_len + i for i in range(n_requests)],
    }


class TestDecodeAttributionBySeqLen:
    """Decode rows are attributed by seq_len, because a tag cannot reach them.

    The trace hook reads its tag from the SERVER's environment; the client sets
    that variable in its own process after the server started, so the rows carry
    whatever it was at launch. The first GPU run reported ``[]`` for every
    conversation because of it. seq_len is carried in the metadata itself.
    """

    def test_decode_row_inside_the_window_is_counted(self):
        """During decode, seq_len is prompt_len + tokens generated so far."""
        records = [_decode(seq_len=42, adapter=3)]
        assert decode_indices_by_seqlen(records, prompt_len=40, max_new=16) == [3]

    def test_prefill_rows_are_excluded_by_the_window(self):
        """A chunk's seq_len is at most the prompt length, so it falls below.

        This is what lets the window double as a prefill filter, even for a chunk
        that happens to be one token wide.
        """
        records = [_prefill(PROMPT_A, 1)]
        assert decode_indices_by_seqlen(records, prompt_len=6, max_new=16) == []

    def test_another_conversations_decode_row_is_excluded(self):
        """A longer conversation's rows sit above this one's window."""
        records = [_decode(seq_len=200, adapter=9)]
        assert decode_indices_by_seqlen(records, prompt_len=40, max_new=16) == []

    def test_rows_beyond_max_new_are_excluded(self):
        """Past prompt_len + max_new the row cannot be this request's."""
        records = [_decode(seq_len=100, adapter=7)]
        assert decode_indices_by_seqlen(records, prompt_len=40, max_new=16) == []

    def test_two_requests_decoding_in_one_forward_are_separated(self):
        """The co-batched case this exists for: one row each, different seq_lens."""
        rec = {
            "input_ids": [5, 6],
            "positions": [41, 220],
            "adapter_indices": [1, 2],
            "query_start_loc": [0, 1, 2],
            "seq_lens": [42, 221],
        }
        assert decode_indices_by_seqlen([rec], prompt_len=40, max_new=16) == [1]
        assert decode_indices_by_seqlen([rec], prompt_len=210, max_new=16) == [2]

    def test_missing_seq_lens_is_skipped_not_guessed(self):
        rec = _decode(seq_len=42, adapter=3)
        rec["seq_lens"] = None
        assert decode_indices_by_seqlen([rec], prompt_len=40, max_new=16) == []


class TestWindowDisjointness:
    """The attribution is only sound when the windows cannot overlap."""

    def test_far_apart_prompts_are_disjoint(self):
        assert windows_disjoint([40, 200], max_new=16) is True

    def test_prompts_closer_than_max_new_overlap(self):
        """40 and 50 with 16 new tokens: seq_len 51 could belong to either."""
        assert windows_disjoint([40, 50], max_new=16) is False

    def test_exactly_max_new_apart_still_overlaps(self):
        """Boundary: a gap equal to max_new leaves the endpoint ambiguous."""
        assert windows_disjoint([40, 56], max_new=16) is False
        assert windows_disjoint([40, 57], max_new=16) is True
