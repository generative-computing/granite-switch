# SPDX-License-Identifier: Apache-2.0
"""The probe type's vLLM module: linear heads in one bank, ``scores = W h + b``.

One bank row per probe slot. The model calls the module at the layer a probe
reads (the final normalized state by default), on every token of the step
(branch-free, so it runs inside CUDA graphs); tokens that are not probe markers
carry row 0. The bank is replicated on every tensor-parallel rank.
"""

import torch
import torch.nn as nn


class SwitchedProbeHead(nn.Module):
    """Linear probe heads in one bank, selected per token by bank row (vLLM).

    Compile-safe, on the flat ``[total_tokens, hidden_size]`` layout vLLM's
    scheduler produces. Row ``0`` = no probe here (zero scores); ``1+``
    selects that bank row's head (``1 + head_row`` of the slot).

    The weight bank is stacked ``[num_rows, ...]`` and indexed by ``(row - 1)``.

    Args:
        hidden_size: Input feature dimension (the model hidden size).
        num_rows: Bank rows: ``1 + max(head_row)`` over the probe slots, or
            ``num_adapters`` for a checkpoint composed before streams, which
            kept one row per slot.
        max_num_labels: Padded label width of the bank (``max`` over the per-slot
            label counts). Keeping this uniform is what lets every slot's head
            run as one matmul. Slots with fewer labels leave their extra
            rows zero. The verdict exit slices each slot back to its real label
            count.
    """

    def __init__(
        self,
        hidden_size: int,
        num_rows: int,
        max_num_labels: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_rows = num_rows
        self.num_labels = max_num_labels

        # Stacked bank indexed by (row - 1)
        #   weight: [num_rows, max_num_labels, hidden_size]
        #   bias:   [num_rows, max_num_labels]
        #
        # A frozen inference bank whose real values are copied in from a trained
        # head at compose time. Zero value for both an unloaded slot's row and a
        # loaded slot's pad rows (label counts below max_num_labels).
        self.weight = nn.Parameter(torch.zeros(num_rows, max_num_labels, hidden_size))
        self.bias = nn.Parameter(torch.zeros(num_rows, max_num_labels))

    def forward(
        self,
        x: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token probe logits for selected positions.

        Branchless and loop-free so it traces cleanly under
        ``@support_torch_compile``.

        Args:
            x: Input tensor [total_tokens, hidden_size].
            head_indices: [total_tokens] bank row + 1; 0 = no probe here.

        Returns:
            Logits [total_tokens, num_labels]. Positions with index 0 are zero.
        """
        is_probe = head_indices > 0  # [total_tokens]
        slot = (head_indices - 1).clamp_min(0)  # [total_tokens]; base -> row 0

        # Every slot's head on every token as one matmul ([T, H] x [H, S*L]),
        # then each token keeps its own slot's L scores. The bank is a few
        # labels per slot, so this costs no more than one head per token and,
        # unlike gathering a weight copy per token ([T, L, H]), allocates
        # nothing that grows with the hidden size.
        #
        # The matmul reads the activation as served and sums in fp32
        # (out_dtype); the bias is added in fp32 and each score is rounded
        # once. A bf16-output matmul rounds inside the reduction, and
        # upcasting the activation first would let the compiler fuse the
        # final norm into the head and skip the activation's own bf16
        # rounding, so a compiled head would score a different input than
        # the one it serves.
        num_slots, num_labels, hidden = self.weight.shape
        weight = self.weight.reshape(num_slots * num_labels, hidden)
        all_scores = torch.mm(x, weight.t(), out_dtype=torch.float32)
        all_scores = all_scores.view(-1, num_slots, num_labels)
        index = slot.view(-1, 1, 1).expand(-1, 1, num_labels)
        logits = all_scores.gather(1, index).squeeze(1) + self.bias[slot].float()

        # Zero out base positions.
        logits = logits * is_probe.unsqueeze(-1).to(logits.dtype)
        return logits.to(x.dtype)
