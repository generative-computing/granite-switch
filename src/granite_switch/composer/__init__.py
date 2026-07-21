# SPDX-License-Identifier: Apache-2.0
"""Compose utilities for Granite Switch models.

This module provides utilities for composing Granite Switch checkpoints from
base models and LoRA adapters.
"""

from .arch import (
    ArchDescriptor,
    ModuleDescriptor,
    granite_dense_arch,
    granite_moe_hybrid_arch,
    resolve_arch,
)
from .compose_utils import GraniteSwitchComposer
from .weight_remapper import AdapterRemapper, RemapResult

__all__ = [
    "AdapterRemapper",
    "ArchDescriptor",
    "GraniteSwitchComposer",
    "ModuleDescriptor",
    "RemapResult",
    "granite_dense_arch",
    "granite_moe_hybrid_arch",
    "resolve_arch",
]
