# SPDX-License-Identifier: Apache-2.0
"""SR doubled-Q head interleave under simulated tensor parallelism (CPU-only).

Validates the invariant behind SR's tensor-parallel support in
``ShadowResidualAttention``: with per-rank (LOCAL) head counts, base query head *i*
and adapter query head *i* land on the SAME KV head that the vanilla head *i* used —
for BOTH the divisible-KV and the replicated-KV (``num_key_value_heads < tp_size``)
regimes. This is what lets the doubled-Q attention run correctly once the ``tp==1``
guard is replaced by total-vs-local head geometry.

Pure arithmetic + the real ``interleave_q_heads`` / ``deinterleave_heads`` from
``granite_switch.vllm.decoder.shadow_residual._sr_ops`` (loaded by file path so the
vLLM package ``__init__`` is never imported — these ops are deliberately vLLM-free and
CPU-unit-testable). No GPU; the replicated-KV corner is otherwise unreachable on real
hardware for the shipped Granite models (it needs ``tp_size > num_key_value_heads``).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the pure-torch ops by path (no granite_switch.vllm package import → no vLLM/GPU).
_SR_OPS = (
    Path(__file__).resolve().parents[2]
    / "src/granite_switch/vllm/decoder/shadow_residual/_sr_ops.py"
)
assert _SR_OPS.is_file(), f"_sr_ops not found at {_SR_OPS}"
_spec = importlib.util.spec_from_file_location("_sr_ops", _SR_OPS)
_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ops)
interleave_q_heads = _ops.interleave_q_heads
deinterleave_heads = _ops.deinterleave_heads

# (total_num_heads, total_num_kv_heads, tp_size)
CONFIGS = [
    # divisible-KV (real granite-4.1-3b is Hq=40, Hkv=8)
    (40, 8, 1),
    (40, 8, 2),
    (40, 8, 4),
    (40, 8, 8),
    # replicated-KV (num_key_value_heads < tp_size): not reachable on the shipped
    # models at feasible GPU counts, so covered here.
    (4, 2, 4),
    (8, 2, 4),
    (8, 2, 8),
    (4, 1, 4),
]
_IDS = [f"Hq{hq}-Hkv{hkv}-tp{tp}" for hq, hkv, tp in CONFIGS]


def _shard_plan(total_hq, total_hkv, tp):
    """Mirror vLLM's contiguous head sharding: per-rank global q heads owned, the
    global KV head held (a replica in the replicated regime), and local counts."""
    assert total_hq % tp == 0
    local_hq = total_hq // tp
    if total_hkv >= tp:
        assert total_hkv % tp == 0
        local_hkv, replicated = total_hkv // tp, False
    else:
        assert tp % total_hkv == 0
        local_hkv, replicated = 1, True  # == max(1, total_hkv // tp)
    ranks_per_kv = tp // total_hkv if replicated else None
    plan = []
    for r in range(tp):
        q_heads = list(range(r * local_hq, (r + 1) * local_hq))
        kv_heads = (
            [r // ranks_per_kv]
            if replicated
            else list(range(r * local_hkv, (r + 1) * local_hkv))
        )
        plan.append(
            dict(
                q_heads=q_heads,
                kv_heads=kv_heads,
                local_hq=local_hq,
                local_hkv=local_hkv,
                replicated=replicated,
            )
        )
    return plan


def _gqa_kv_of(query_head, n_q, n_kv):
    """vLLM GQA mapping: query head -> KV head."""
    assert n_q % n_kv == 0
    return query_head // (n_q // n_kv)


@pytest.mark.parametrize("total_hq,total_hkv,tp", CONFIGS, ids=_IDS)
def test_sr_doubled_q_tp_head_mapping(total_hq, total_hkv, tp):
    d, n = 8, 3  # head_dim, tokens
    for r, p in enumerate(_shard_plan(total_hq, total_hkv, tp)):
        lhq, lhkv = p["local_hq"], p["local_hkv"]

        # A. interleave places base head i at slot 2i, adapter head i at slot 2i+1.
        qb = torch.cat([torch.full((n, d), 100.0 + h) for h in range(lhq)], dim=1)
        qa = torch.cat([torch.full((n, d), 200.0 + h) for h in range(lhq)], dim=1)
        dbl = interleave_q_heads(qb, qa, lhq, d)  # [n, 2*lhq*d]
        heads = dbl.reshape(n, 2 * lhq, d)
        for i in range(lhq):
            assert torch.all(heads[:, 2 * i, :] == 100.0 + i), (r, i, "base slot")
            assert torch.all(heads[:, 2 * i + 1, :] == 200.0 + i), (
                r,
                i,
                "adapter slot",
            )

        # B. round-trip identity.
        b2, a2 = deinterleave_heads(dbl, lhq, d)
        assert torch.equal(b2, qb) and torch.equal(a2, qa), (r, "roundtrip")

        # C. doubled-Q slot -> KV equals vanilla head i -> KV, for base AND adapter copy.
        for i in range(lhq):
            vanilla = _gqa_kv_of(i, lhq, lhkv)
            assert _gqa_kv_of(2 * i, 2 * lhq, lhkv) == vanilla, (r, i, "base->kv")
            assert _gqa_kv_of(2 * i + 1, 2 * lhq, lhkv) == vanilla, (
                r,
                i,
                "adapter->kv",
            )

        # D. co-location: every global q head a rank owns needs only a KV head it holds.
        for local_i, g in enumerate(p["q_heads"]):
            needed = _gqa_kv_of(g, total_hq, total_hkv)
            if p["replicated"]:
                assert needed == p["kv_heads"][0], (r, g, "colocation-replicated")
                local_kv_idx = 0
            else:
                assert needed in p["kv_heads"], (r, g, "colocation-divisible")
                local_kv_idx = needed - p["kv_heads"][0]
            assert _gqa_kv_of(local_i, lhq, lhkv) == local_kv_idx, (r, g, "local-map")
