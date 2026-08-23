# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``promote_rank`` — rounding a LoRA rank up to a supported kernel tier.

Pure CPU: no GPU is needed to execute these (the function is plain arithmetic). The
kernels package is imported via ``importorskip`` so the module is skipped rather than
erroring where Triton is unavailable.
"""

import pytest

_gk = pytest.importorskip("granite_switch.kernels")
promote_rank = _gk.promote_rank
SUPPORTED_RANKS = _gk.SUPPORTED_RANKS


@pytest.mark.parametrize(
    "rank,expected",
    [
        (1, 16),  # below the smallest tier -> smallest tier
        (8, 16),  # the sub-16 case exercised end-to-end in test_switched_lora_linear
        (15, 16),
        (16, 16),  # exactly on a tier -> unchanged
        (17, 32),  # just over a tier -> next tier
        (32, 32),
        (33, 64),
        (100, 128),
        (256, 256),
        (257, 512),
        (512, 512),  # the largest tier
    ],
)
def test_promotes_to_next_supported_tier(rank, expected):
    assert promote_rank(rank) == expected


def test_supported_ranks_are_fixed_points():
    for r in SUPPORTED_RANKS:
        assert promote_rank(r) == r


@pytest.mark.parametrize("rank", [0, -1, -8])
def test_non_positive_rank_returned_unchanged(rank):
    # rank <= 0 means "not applicable" (no adapter); returned as-is, never promoted.
    assert promote_rank(rank) == rank


@pytest.mark.parametrize("rank", [513, 1000])
def test_rank_above_largest_tier_raises(rank):
    with pytest.raises(ValueError):
        promote_rank(rank)
