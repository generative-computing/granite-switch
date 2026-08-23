# SPDX-License-Identifier: Apache-2.0
"""Pure-torch tensor ops for the Shadow-Residual vLLM decoder (SWITCH backend).

The doubled-Q head interleave/deinterleave used by the base-only-KV attention.
Kept free of any vLLM import so they can be unit-tested on CPU, and
independent of the LoRA kernel backend.

Head interleave convention
--------------------------
The SR attention runs ONE GQA kernel with 2x the query heads: base query head
``i`` sits at output position ``2i`` and adapter query head ``i`` at ``2i+1``.
This "even=base / odd=adapter" layout is REQUIRED, not cosmetic: vLLM's GQA maps
query head ``m`` to KV head ``m // (num_q_heads // num_kv_heads)``. With the
query-head count doubled, that group size doubles too, so heads ``2i`` and
``2i+1`` both map to the same KV head vanilla head ``i`` used — i.e. base and
adapter attend the same base-only K/V. A ``[base..., adapt...]`` concatenation
would instead map the adapter heads onto the wrong KV heads.
"""

import torch


def interleave_q_heads(
    q_base: torch.Tensor,
    q_adapt: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Interleave two per-stream query tensors into one doubled-Q tensor.

    Args:
        q_base, q_adapt: ``[N, num_heads * head_dim]``.
        num_heads: query heads PER STREAM (Hq).
        head_dim: per-head dim (d).

    Returns:
        ``[N, 2 * num_heads * head_dim]`` with base head ``i`` at slot ``2i`` and
        adapter head ``i`` at slot ``2i+1`` (even=base, odd=adapter).
    """
    n = q_base.shape[0]
    qb = q_base.reshape(n, num_heads, head_dim)
    qa = q_adapt.reshape(n, num_heads, head_dim)
    # New axis between head and dim -> [N, Hq, 2, d]; row-major flatten merges
    # (Hq, 2) so stream s of head i lands at head slot 2*i + s.
    stacked = torch.stack((qb, qa), dim=2)
    return stacked.reshape(n, 2 * num_heads * head_dim)


def deinterleave_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_q_heads`.

    Args:
        x: ``[N, 2 * num_heads * head_dim]`` (doubled-Q attention output).
        num_heads: query heads PER STREAM (Hq).
        head_dim: per-head dim (d).

    Returns:
        ``(base, adapt)`` each ``[N, num_heads * head_dim]`` — even heads -> base,
        odd heads -> adapter.
    """
    n = x.shape[0]
    grouped = x.reshape(n, num_heads, 2, head_dim)
    base = grouped[:, :, 0, :].reshape(n, num_heads * head_dim)
    adapt = grouped[:, :, 1, :].reshape(n, num_heads * head_dim)
    return base, adapt


__all__ = ["deinterleave_heads", "interleave_q_heads"]
