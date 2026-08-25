# SPDX-License-Identifier: Apache-2.0
"""Top-k logit-distribution agreement metrics: JSD (bits) and top-k Jaccard.

Shared by the equivalence tests (SR TP/PP and generation-equivalence) so the
metric cannot silently drift between copies. JSD over the renormalized top-k
union is the primary agreement signal; top-k Jaccard-distance is a loose
set-overlap guard.
"""

import math


def topk_ids(dist, k):
    """Ids of the k highest-logprob entries of a {token_id: logprob} dict."""
    return [t for t, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def jsd_bits(p_lp, q_lp, ids):
    """JSD (bits) between two logprob dicts, restricted + renormalized over `ids`.

    Note: this renormalizes over `ids` (the compared support), so it measures the
    SHAPE agreement within that support. Callers that need tail-mass sensitivity
    should also compare captured top-k total mass (see captured_mass).
    """

    def probs(lp):
        v = {t: math.exp(lp[t]) for t in ids if t in lp}
        s = sum(v.values()) or 1.0
        return {t: v.get(t, 0.0) / s for t in ids}

    p, q = probs(p_lp), probs(q_lp)
    m = {t: 0.5 * (p[t] + q[t]) for t in ids}

    def kl(a, b):
        return sum(a[t] * math.log2(a[t] / b[t]) for t in ids if a[t] > 0 and b[t] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def captured_mass(lp):
    """Total probability mass held by the captured (top-k) entries of a logprob dict."""
    return sum(math.exp(v) for v in lp.values())
