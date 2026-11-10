# SPDX-License-Identifier: Apache-2.0
"""The activation probe: a linear head per slot, ``scores = W @ h + b``.

The first head type on the stream API. Its modules are the probe heads the
backends already had (``SwitchedProbeHead``), so a probe checkpoint keeps its
weight names (``model.probe_head.weight`` / ``.bias``) and its numbers.

Weights: a slot folder ships ``probe_head.safetensors`` with ``weight``
``[n_labels, hidden]`` and ``bias`` ``[n_labels]``. Output: ``select`` over the
label words' token ids -- the request emits one label word.
"""

import torch

from ..registry import HeadType, register_head_type

#: The trained head a probe folder ships, and the keys it must hold.
PROBE_HEAD_FILE = "probe_head.safetensors"
PROBE_HEAD_KEYS = ("weight", "bias")


@register_head_type
class ProbeType(HeadType):
    name = "probe"
    weights_file = PROBE_HEAD_FILE

    #: The model's attribute for this type's module: kept for checkpoint names.
    module_name = "probe_head"

    @staticmethod
    def _shape(specs, config):
        rows = 1 + max(s.head_row for s in specs)
        if config.streams is None:
            # A pre-streams checkpoint's bank has a row per slot, probe or not.
            rows = config.num_adapters
        width = max(s.num_outputs for s in specs)
        return config.hidden_size, rows, width

    def hf_module(self, specs, config):
        from granite_switch.hf.core.probe import SwitchedProbeHead

        hidden, rows, width = self._shape(specs, config)
        return SwitchedProbeHead(hidden, rows, width)

    def vllm_module(self, specs, config):
        from granite_switch.vllm.core.probe import SwitchedProbeHead

        hidden, rows, width = self._shape(specs, config)
        return SwitchedProbeHead(hidden, rows, width)

    def load_weights(self, module, row, state, spec):
        if set(state) != set(PROBE_HEAD_KEYS):
            raise ValueError(
                f"{PROBE_HEAD_FILE} must hold exactly {sorted(PROBE_HEAD_KEYS)}; "
                f"got {sorted(state)}."
            )
        n, hidden = spec.num_outputs, module.hidden_size
        weight, bias = state["weight"], state["bias"]
        if tuple(weight.shape) != (n, hidden) or tuple(bias.shape) != (n,):
            raise ValueError(
                f"{PROBE_HEAD_FILE}: weight {tuple(weight.shape)} / bias "
                f"{tuple(bias.shape)} do not match {n} labels x hidden {hidden}."
            )
        with torch.no_grad():
            module.weight[row, :n].copy_(weight.to(module.weight))
            module.bias[row, :n].copy_(bias.to(module.bias))
