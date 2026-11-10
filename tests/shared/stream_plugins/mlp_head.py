# SPDX-License-Identifier: Apache-2.0
"""``mlp_head``: a two-layer MLP over the read state, one per slot.

An example head type, written only against the stream API
(:mod:`granite_switch.streams`) and living outside the package: adding it
changes no Granite Switch file. A slot of this type ships::

    settings: {"width": 32}            # the MLP's hidden width
    weights:  mlp_head.safetensors     # w1 [width, hidden], w2 [n, width]

and is meant for output mode ``merge`` (a learned nudge to the LM head's
logits), though ``select`` works too. Its forward has no data-dependent
branches, so the one module serves HF and vLLM (the default ``vllm_module``).
"""

import torch
from torch import nn

from granite_switch.streams import HeadType, register_head_type

MLP_HEAD_FILE = "mlp_head.safetensors"


class MLPHeads(nn.Module):
    """``scores = w2[row] @ gelu(w1[row] @ h)`` at each token, by its row.

    Banks ``w1 [rows, width, hidden]`` and ``w2 [rows, outputs, width]``, padded
    to the widest slot (padding is zero, so it adds nothing). ``rows`` holds
    ``1 + head_row`` per token, 0 for "not this type's marker" (score 0).

    Every token is scored against the whole bank, then each keeps its own
    row: the bank is small (one row per slot), so this costs little and needs
    neither a branch nor a per-token copy of the weights.
    """

    def __init__(self, hidden_size: int, rows: int, width: int, outputs: int):
        super().__init__()
        self.hidden_size, self.width, self.outputs = hidden_size, width, outputs
        self.w1 = nn.Parameter(torch.zeros(rows, width, hidden_size))
        self.w2 = nn.Parameter(torch.zeros(rows, outputs, width))

    def forward(self, hidden: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        own = torch.arange(hidden.shape[0], device=hidden.device)
        idx = (rows - 1).clamp_min(0)
        h = torch.einsum("th,rwh->trw", hidden.to(self.w1.dtype), self.w1)[own, idx]
        h = torch.nn.functional.gelu(h)
        scores = torch.einsum("tw,row->tro", h, self.w2)[own, idx]
        return scores * (rows > 0).unsqueeze(-1).to(scores.dtype)


@register_head_type
class MLPHeadType(HeadType):
    name = "mlp_head"
    weights_file = MLP_HEAD_FILE

    def check(self, spec, config):
        width = spec.settings.get("width")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError(
                f"mlp_head slot {spec.slot} needs settings.width, a positive int; "
                f"got {width!r}."
            )

    def hf_module(self, specs, config):
        return MLPHeads(
            hidden_size=config.hidden_size,
            rows=1 + max(s.head_row for s in specs),
            width=max(s.settings["width"] for s in specs),
            outputs=max(s.num_outputs for s in specs),
        )

    def load_weights(self, module, row, state, spec):
        if set(state) != {"w1", "w2"}:
            raise ValueError(
                f"{MLP_HEAD_FILE} must hold w1 and w2; got {sorted(state)}."
            )
        width, n = spec.settings["width"], spec.num_outputs
        w1, w2 = state["w1"], state["w2"]
        if tuple(w1.shape) != (width, module.hidden_size) or tuple(w2.shape) != (
            n,
            width,
        ):
            raise ValueError(
                f"{MLP_HEAD_FILE}: w1 {tuple(w1.shape)} / w2 {tuple(w2.shape)} do not "
                f"match width {width}, hidden {module.hidden_size}, {n} outputs."
            )
        with torch.no_grad():
            module.w1[row, :width].copy_(w1.to(module.w1))
            module.w2[row, :n, :width].copy_(w2.to(module.w2))
