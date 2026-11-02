# SPDX-License-Identifier: Apache-2.0
"""Switched activation probe heads for Granite Switch (vLLM backend).

An activation probe head, a linear head over a hidden state, is an optional
substitute for a LoRA adapter: an adapter slot is either a LoRA adapter or a
probe head, selected by the same control token and adapter index. Where a probe
is selected, the LoRA path no-ops.
The model supplies the configured layer's hidden states (the final normalized
state by default), and the head produces per-label scores.

Probe positions are expected to be sparse, and the weight bank is replicated
on every rank at this point; revisit if probes ever cover most tokens or
num_labels grows large.
"""

import torch
import torch.nn as nn


class SwitchedProbeHead(nn.Module):
    """Per-token probe heads selected by probe indices (vLLM).

    Compile-safe, on the flat ``[total_tokens, hidden_size]`` layout vLLM's
    scheduler produces. Index ``0`` = no probe (base -> zero logits);
    ``1+`` selects that slot's head.

    The weight bank is stacked ``[num_slots, ...]`` and indexed by
    ``(index - 1)``, matching ``SwitchedLoRALinear``.

    Args:
        hidden_size: Input feature dimension (the model hidden size).
        num_probe_slots: ``num_adapters`` (the full index space). Slot ``s``
            (0-based) serves adapter index ``s + 1``. Slots that are not
            probes simply never get selected.
        max_num_labels: Padded label width of the bank (``max`` over the per-slot
            label counts). Keeping this uniform is what lets the per-token logits
            be one fused ``einsum``. Slots with fewer labels leave their extra
            rows zero. The verdict exit slices each slot back to its real label
            count.
    """

    def __init__(
        self,
        hidden_size: int,
        num_probe_slots: int,
        max_num_labels: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_probe_slots = num_probe_slots
        self.num_labels = max_num_labels

        # Stacked bank indexed by (adapter_idx - 1), like the LoRA bank
        #   weight: [num_slots, max_num_labels, hidden_size]
        #   bias:   [num_slots, max_num_labels]
        #
        # A frozen inference bank whose real values are copied in from a trained
        # head at compose time. Zero value for both an unloaded slot's row and a
        # loaded slot's pad rows (label counts below max_num_labels).
        self.weight = nn.Parameter(
            torch.zeros(num_probe_slots, max_num_labels, hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(num_probe_slots, max_num_labels))

    def forward(
        self,
        x: torch.Tensor,
        probe_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token probe logits for selected positions.

        Branchless and loop-free so it traces cleanly under
        ``@support_torch_compile``.

        Args:
            x: Input tensor [total_tokens, hidden_size].
            probe_indices: [total_tokens]; 0 = no probe, 1+ = slot
                index (the un-offset adapter index).

        Returns:
            Logits [total_tokens, num_labels]. Positions with index 0 are zero.
        """
        is_probe = probe_indices > 0  # [total_tokens]
        slot = (probe_indices - 1).clamp_min(0)  # [total_tokens]; base -> row 0

        w_per_token = self.weight[slot]  # [total_tokens, num_labels, hidden]
        b_per_token = self.bias[slot]  # [total_tokens, num_labels]

        # Per-token linear, batched over tokens as one fused einsum.
        logits = torch.einsum("tlh,th->tl", w_per_token, x)
        logits = logits + b_per_token

        # Zero out base positions.
        logits = logits * is_probe.unsqueeze(-1).to(logits.dtype)
        return logits
