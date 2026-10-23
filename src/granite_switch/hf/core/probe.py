# SPDX-License-Identifier: Apache-2.0
"""Switched activation probe heads for Granite Switch.

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
    """Per-token probe heads selected by probe indices.

    Stacked weight bank indexed by ``(index - 1)``, like ``SwitchedLoRALinear``.
    Index ``0`` = no probe (base); ``1+`` selects that slot's head.
    Positions not selected are left at zero.

    Args:
        hidden_size: Input feature dimension (the model hidden size).
        num_probe_slots: ``num_adapters`` (the full index space), so
            ``index - 1`` addresses the bank directly like the LoRA bank. Slots
            that are not probes simply never get selected.
        max_num_labels: Padded label width of the bank — ``max`` over the
            per-slot label counts. Slots with fewer labels leave their extra
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

        # Slot s (0-based) serves adapter index s + 1, matching the LoRA bank's
        # tensor_idx = adapter_idx - 1.
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

        Args:
            x: Input tensor [batch, seq_len, hidden_size] or [num_tokens, hidden_size].
            probe_indices: Per-token probe selection [batch, seq_len]
                or [num_tokens]; 0 = no probe, 1+ = slot index.

        Returns:
            Logits with the selected slot's head applied per token:
            [batch, seq_len, num_labels] or [num_tokens, num_labels]. Positions
            with index 0 (and any not selected by this bank) are left at zero.
        """
        # Flatten for token-level processing (same as SwitchedLoRALinear).
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.hidden_size)
        else:
            x_flat = x

        logits_flat = x_flat.new_zeros((x_flat.shape[0], self.num_labels))

        idx_flat = probe_indices.reshape(-1)

        # Per-token gather/scatter loop (same as SwitchedLoRALinear.forward)
        mask = idx_flat > 0
        if mask.any():
            active = idx_flat[mask].unique()
            for adapter_idx in active:
                token_mask = idx_flat == adapter_idx
                token_indices = torch.where(token_mask)[0]

                if len(token_indices) == 0:
                    continue

                slot = adapter_idx - 1  # 1-indexed → 0-indexed, like the LoRA bank
                w = self.weight[slot]  # [num_labels, hidden_size]
                b = self.bias[slot]  # [num_labels]

                x_sel = x_flat[token_indices]  # [num_sel, hidden_size]
                # Write (not add): a probe replaces the LoRA.
                logits_flat[token_indices] = x_sel @ w.t() + b

        # Reshape back if needed.
        if len(original_shape) == 3:
            return logits_flat.view(batch_size, seq_len, self.num_labels)
        return logits_flat
