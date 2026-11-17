# SPDX-License-Identifier: Apache-2.0
"""Kerdock and Delsarte-Goethals code generation for memory-based switching."""

import torch

from .code_generator import KerdockDGCodeGenerator, verify_coherence


def recover_count_from_signal(
    counting_signal: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Recover integer count from 1/(1+n) attention signal.

    Always upcasts to fp32 for inversion: bf16 ULP=0.5 in [64,128)
    causes round-half-to-even errors. fp32 is safe to ~8.4M.
    All ops are tensor ops (torch.compile compatible, stays on device).
    """
    count = 1.0 / counting_signal.float() - 1.0
    return torch.clamp(torch.round(count).long(), 0, capacity - 1)


__all__ = ["KerdockDGCodeGenerator", "recover_count_from_signal", "verify_coherence"]
