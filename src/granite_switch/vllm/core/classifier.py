# SPDX-License-Identifier: Apache-2.0
"""Switched classifier heads for Granite Switch (vLLM backend).

A classifier head is an optional substitute for a LoRA adapter: an adapter slot
is either a LoRA adapter or a classifier head, selected by the same control
token and adapter index. Where a classifier is selected, the LoRA path no-ops
and the classifier reads clean base-model hidden states.

Classifier positions are expected to be sparse, and the weight bank is replicated
on every rank at this point; revisit if classifiers ever cover most tokens or
num_labels grows large.
"""

import torch
import torch.nn as nn


class SwitchedClassifierHead(nn.Module):
    """Per-token classifier heads selected by classifier indices (vLLM).

    Compile-safe, on the flat ``[total_tokens, hidden_size]`` layout vLLM's
    scheduler produces. Index ``0`` = no classifier (base -> zero logits);
    ``1+`` selects that slot's head.

    The weight bank is stacked ``[num_slots, ...]`` and indexed by
    ``(index - 1)``, matching ``SwitchedLoRALinear``.

    Args:
        hidden_size: Input feature dimension (the model hidden size).
        num_classifier_slots: ``num_adapters`` (the full index space). Slot ``s``
            (0-based) serves adapter index ``s + 1``. Slots that are not
            classifiers simply never get selected.
        max_num_labels: Padded label width of the bank (``max`` over the per-slot
            label counts). Keeping this uniform is what lets the per-token logits
            be one fused ``einsum``. Slots with fewer labels leave their extra
            rows zero. The verdict exit slices each slot back to its real label
            count.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classifier_slots: int,
        max_num_labels: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classifier_slots = num_classifier_slots
        self.num_labels = max_num_labels

        # Stacked bank indexed by (adapter_idx - 1), like the LoRA bank
        #   weight: [num_slots, max_num_labels, hidden_size]
        #   bias:   [num_slots, max_num_labels]
        #
        # A frozen inference bank whose real values are copied in from a trained
        # head at compose time. Zero value for both an unloaded slot's row and a
        # loaded slot's pad rows (label counts below max_num_labels).
        self.weight = nn.Parameter(
            torch.zeros(num_classifier_slots, max_num_labels, hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(num_classifier_slots, max_num_labels))

    def forward(
        self,
        x: torch.Tensor,
        classifier_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token classifier logits for selected positions.

        Branchless and loop-free so it traces cleanly under
        ``@support_torch_compile``.

        Args:
            x: Input tensor [total_tokens, hidden_size].
            classifier_indices: [total_tokens]; 0 = no classifier, 1+ = slot
                index (the un-offset adapter index).

        Returns:
            Logits [total_tokens, num_labels]. Positions with index 0 are zero.
        """
        is_classifier = classifier_indices > 0  # [total_tokens]
        slot = (classifier_indices - 1).clamp_min(0)  # [total_tokens]; base -> row 0

        w_per_token = self.weight[slot]  # [total_tokens, num_labels, hidden]
        b_per_token = self.bias[slot]  # [total_tokens, num_labels]

        # Per-token linear, batched over tokens as one fused einsum.
        logits = torch.einsum("tlh,th->tl", w_per_token, x)
        logits = logits + b_per_token

        # Zero out base positions.
        logits = logits * is_classifier.unsqueeze(-1).to(logits.dtype)
        return logits
