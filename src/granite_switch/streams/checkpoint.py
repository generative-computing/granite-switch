# SPDX-License-Identifier: Apache-2.0
"""Strict loading of replicated head parameters and persistent buffers on vLLM."""

import torch
from torch import nn

from .routing import module_attr


class HeadWeightLoader:
    """Consume head tensors before the decoder applies its own weight mappings.

    HF and vLLM heads share state-dict keys and shapes. Persistent buffers are
    part of that contract too; decoder loaders otherwise inspect parameters
    only. Every head tensor is required, even if its name resembles a LoRA
    parameter or a buffer that the base model regenerates.
    """

    def __init__(self, model: nn.Module) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        prefixes = []
        for name in model._head_types:
            attr = module_attr(name)
            prefix = f"model.{attr}."
            prefixes.append(prefix)
            state = getattr(model, attr).state_dict(keep_vars=True)
            self.tensors.update((prefix + key, tensor) for key, tensor in state.items())
        self.prefixes = tuple(prefixes)
        self.loaded: set[str] = set()

    def load(self, name: str, value: torch.Tensor) -> bool:
        """Copy a head tensor; return False for a decoder-owned key."""
        if not name.startswith(self.prefixes):
            return False
        if name not in self.tensors:
            raise ValueError(f"Unexpected head checkpoint tensor {name!r}.")
        target = self.tensors[name]
        if target.shape != value.shape:
            raise ValueError(
                f"Head checkpoint tensor {name!r} has shape {tuple(value.shape)}; "
                f"expected {tuple(target.shape)}."
            )
        with torch.no_grad():
            target.copy_(value)
        self.loaded.add(name)
        return True

    def finish(self) -> set[str]:
        """Require every serialized head tensor before serving any request."""
        missing = self.tensors.keys() - self.loaded
        if missing:
            raise ValueError(f"Missing head checkpoint tensors: {sorted(missing)}.")
        return self.loaded
