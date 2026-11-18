# SPDX-License-Identifier: Apache-2.0
"""Regression plugin with parameters, persistent buffers, and ordinary copies."""

import torch
from torch import nn

from granite_switch.streams import HeadType, register_head_type


class BufferedHeads(nn.Module):
    def __init__(self, rows, outputs, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, outputs, hidden))
        # Deliberately resembles a decoder-owned buffer; the head owns it.
        self.register_buffer("adapter_token_ids", torch.zeros(rows, outputs))

    def forward(self, hidden, rows):
        idx = (rows - 1).clamp_min(0)
        return (
            torch.einsum("th,toh->to", hidden, self.weight[idx])
            + self.adapter_token_ids[idx]
        )


@register_head_type
class BufferedHeadType(HeadType):
    name = "test_buffered"

    def hf_module(self, specs, config):
        return BufferedHeads(
            1 + max(s.head_row for s in specs),
            max(s.num_outputs for s in specs),
            config.hidden_size,
        )

    def load_weights(self, module, row, state, spec):
        n = spec.num_outputs
        module.weight[row, :n].copy_(state["weight"])
        module.adapter_token_ids[row, :n].copy_(state["bias"])
