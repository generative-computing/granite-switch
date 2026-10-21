# SPDX-License-Identifier: Apache-2.0
"""The eval metric must not sweep k == TOP_K, the capture-window edge.

CPU-only, no checkpoint: these build the pathological distribution pair by hand,
which is the point -- the inflation this guards against was originally found only
after burning four GPU runs chasing it as a vLLM regression.

At k == TOP_K the compared support is the capture WINDOW rather than a rank band
well inside it, so a token that merely sits on opposite sides of the truncation
boundary in two runs appears in the union present on one side and absent on the
other. `jsd_bits` renormalizes over the union, so such a token contributes
roughly its entire probability mass in bits however small the logit difference
that moved it was. The result reads as a distribution regression and is not one.
"""

import math

import pytest

from tests.eval import _eval_core as core
from tests.shared.logit_metrics import jsd_bits, topk_ids


def test_compare_ks_stay_below_top_k():
    """The invariant itself, so re-adding TOP_K to the sweep fails here."""
    assert core.COMPARE_KS, "the comparison must sweep at least one k"
    assert max(core.COMPARE_KS) < core.TOP_K, (
        f"COMPARE_KS={core.COMPARE_KS} includes k >= TOP_K={core.TOP_K}. At k == "
        "TOP_K the metric measures capture-window churn, not distribution drift "
        "-- see COMPARE_KS in tests/eval/_eval_core.py."
    )


def test_top_k_below_the_sweep_is_rejected_not_silently_vacuous():
    """A TOP_K under the whole sweep would make `_pairwise` return all zeros."""
    assert core.TOP_K > min(core.COMPARE_KS)


def _window(ranked, k):
    """{token: logprob} for the top `k` of `ranked`, as a capture of width k."""
    return {str(t): lp for t, lp in ranked[:k]}


def _tail_swapped_pair(top_k):
    """Two captures agreeing on every shared token but disagreeing at the edge.

    Identical head; the tokens at ranks `top_k-1` and `top_k` are near-ties, and
    the two runs order them oppositely. Exactly one token therefore falls outside
    each capture -- a different one each time -- with no change to any logprob
    that both captures retain.
    """
    head = [(1000 + i, -2.0 - 0.5 * i) for i in range(top_k - 1)]
    a_edge, b_edge = (7001, -9.0), (7002, -9.000001)
    a = dict(_window([*head, a_edge, b_edge], top_k))
    b = dict(_window([*head, b_edge, a_edge], top_k))
    return a, b


def test_edge_churn_inflates_jsd_at_k_equals_top_k():
    """The mechanism, measured: same shared logprobs, wildly different JSD."""
    top_k = core.TOP_K
    a, b = _tail_swapped_pair(top_k)

    shared = set(a) & set(b)
    assert len(shared) == top_k - 1
    for t in shared:
        assert a[t] == b[t], "the pair must not differ on any retained token"

    pa = {int(t): v for t, v in a.items()}
    pb = {int(t): v for t, v in b.items()}

    def swept(k):
        ids = list(set(topk_ids(pa, k)) | set(topk_ids(pb, k)))
        return jsd_bits(pa, pb, ids)

    at_window = swept(top_k)
    inside = max(swept(k) for k in core.COMPARE_KS)

    # Inside the window the two captures are identical, so this is exactly zero.
    assert inside == pytest.approx(0.0, abs=1e-12), (
        f"k<{top_k} should see no difference at all, saw {inside}"
    )
    # At the window edge the same pair reads as a regression. The excluded token
    # holds ~exp(-9) of the mass; the JSD it manufactures is of that order, which
    # is above the 0.003 mean_jsd gate once a few positions carry it.
    assert at_window > 100 * (inside + 1e-9)
    assert at_window > math.exp(-9.0) / 4, (
        "the inflation should be on the order of the excluded token's mass; "
        f"got {at_window}"
    )


def test_pairwise_reports_zero_for_an_edge_churn_only_difference():
    """End to end through `_pairwise`: the configured sweep is immune."""
    a, b = _tail_swapped_pair(core.TOP_K)
    m = core._pairwise([a] * 4, [b] * 4)
    assert m["mean_jsd"] == pytest.approx(0.0, abs=1e-12)
    assert m["max_jsd"] == pytest.approx(0.0, abs=1e-12)
    assert m["mean_jaccard_dist"] == pytest.approx(0.0, abs=1e-12)


def test_pairwise_still_catches_a_real_localized_break():
    """Capping k must not cost sensitivity -- the gate still has to bite."""
    top_k = core.TOP_K
    ranked = [(1000 + i, -2.0 - 0.5 * i) for i in range(top_k)]
    clean = {str(t): lp for t, lp in ranked}

    # Swap the top two by 3 nats at one position out of four: the shape of a
    # genuine logit break, well inside the compared support.
    broken = dict(clean)
    broken[str(ranked[0][0])] = ranked[0][1] - 3.0
    broken[str(ranked[1][0])] = ranked[1][1] + 3.0

    m = core._pairwise([clean] * 4, [broken, clean, clean, clean])
    th = core.DEFAULT_THRESHOLDS
    assert m["max_jsd"] > th["max_jsd"], f"max_jsd {m['max_jsd']} did not trip"
    assert m["max_jsd_pos"] == 0


def test_pairwise_still_catches_a_systematic_logit_scaling():
    """The other calibrated corruption: a global temperature change."""
    top_k = core.TOP_K
    ranked = [(1000 + i, -2.0 - 0.5 * i) for i in range(top_k)]
    clean = {str(t): lp for t, lp in ranked}
    scaled = {t: lp * 1.3 for t, lp in clean.items()}

    m = core._pairwise([clean] * 4, [scaled] * 4)
    assert m["mean_jsd"] > core.DEFAULT_THRESHOLDS["mean_jsd"], (
        f"a 30% logit scaling must trip mean_jsd; got {m['mean_jsd']}"
    )
