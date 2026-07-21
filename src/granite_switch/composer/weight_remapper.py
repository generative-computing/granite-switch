# SPDX-License-Identifier: Apache-2.0
"""Adapter name remapping utilities.

``AdapterRemapper`` is generated programmatically from
:class:`~granite_switch.composer.arch.ModuleDescriptor` list.  Single source of
truth for adapter name remapping — maps PEFT parameter names to Switch
target names based on module descriptors.
"""

import re
from dataclasses import dataclass


@dataclass
class RemapResult:
    """Result of adapter name remapping, with optional split metadata.

    For standard (non-split) mappings, ``split_slices`` is None and
    ``target_name`` is the final parameter name.

    For split mappings (e.g., fused ``input_linear`` → 2 slices),
    ``target_name`` is the base name without slice index (e.g.,
    ``model.layers.0.shared_mlp.input_linear.lora_A_slices``), and
    ``split_slices`` indicates how many slices to produce:

    - ``split_type="duplicate"``: copy the same tensor to each slice
      (used for lora_A where both gate and up share the input space).
    - ``split_type="chunk_dim0"``: chunk tensor along dim=0 into N slices
      (used for lora_B where output dim is [gate_size | up_size]).
    """

    target_name: str
    split_slices: int | None = None
    split_type: str | None = None


# ---------------------------------------------------------------------------
# AdapterRemapper — generated from ModuleDescriptor list
# ---------------------------------------------------------------------------


class AdapterRemapper:
    """Adapter name remapper generated from module descriptors.

    Given a list of :class:`~granite_switch.composer.arch.ModuleDescriptor` and a
    PEFT source prefix, builds compiled regex rules that map PEFT parameter
    names to Switch target names.
    """

    def __init__(self, groups, peft_source_prefix: str = "base_model.model.model."):
        self._rules = []
        self._build_rules(groups, peft_source_prefix)

    def _build_rules(self, groups, prefix: str):
        """Build compiled regex rules from module descriptors."""
        for g in groups:
            n_slices = g.effective_num_slices
            inner = g.target_inner_path  # e.g., "inner." or ""
            attr = g.effective_attr_name
            src_parent = g.effective_source_parent

            if g.is_split:
                # Split module: 1 PEFT module → N slices with split metadata
                peft_mod = g.peft_modules[0]
                for ab in ("lora_A", "lora_B"):
                    pattern = self._make_pattern(prefix, src_parent, peft_mod, ab)
                    target_template = (
                        f"model.layers.{{layer}}.{g.parent}.{attr}.{inner}{ab}_slices"
                    )
                    if ab == "lora_A":
                        split_info = {"slices": n_slices, "type": "duplicate"}
                    else:
                        split_info = {"slices": n_slices, "type": "chunk_dim0"}
                    self._rules.append((pattern, target_template, split_info))

            elif g.is_sliced:
                # Multi-PEFT → numbered slices (qkv, gate_up)
                for slice_idx, peft_mod in enumerate(g.peft_modules):
                    for ab in ("lora_A", "lora_B"):
                        pattern = self._make_pattern(prefix, src_parent, peft_mod, ab)
                        target_template = (
                            f"model.layers.{{layer}}.{g.parent}.{attr}."
                            f"{inner}{ab}_slices.{slice_idx}"
                        )
                        self._rules.append((pattern, target_template, None))

            else:
                # Non-sliced: plain lora_A / lora_B
                peft_mod = g.peft_modules[0]
                for ab in ("lora_A", "lora_B"):
                    pattern = self._make_pattern(prefix, src_parent, peft_mod, ab)
                    target_template = (
                        f"model.layers.{{layer}}.{g.parent}.{attr}.{inner}{ab}"
                    )
                    self._rules.append((pattern, target_template, None))

    @staticmethod
    def _make_pattern(prefix: str, parent: str, peft_mod: str, ab: str):
        """Build compiled regex for a PEFT source parameter name.

        Pattern: ``{prefix}layers.{layer}.{parent}.{peft_mod}.{ab}.weight``
        """
        # Escape dots in literal segments
        escaped_prefix = re.escape(prefix)
        escaped_parent = re.escape(parent)
        escaped_mod = re.escape(peft_mod)
        escaped_ab = re.escape(ab)
        regex = (
            f"^{escaped_prefix}layers\\.(?P<layer>\\d+)\\."
            f"{escaped_parent}\\.{escaped_mod}\\.{escaped_ab}\\.weight$"
        )
        return re.compile(regex)

    def remap_adapter_name(self, src_name: str) -> RemapResult | None:
        """Remap a single adapter weight name to its target name.

        Args:
            src_name: Source parameter name (PEFT format).

        Returns:
            ``RemapResult`` with target name and optional split metadata,
            or ``None`` if no mapping matches.
        """
        for compiled_re, target_template, split_info in self._rules:
            match = compiled_re.match(src_name)
            if match:
                target_name = target_template.format(**match.groupdict())
                if split_info:
                    return RemapResult(
                        target_name=target_name,
                        split_slices=split_info["slices"],
                        split_type=split_info["type"],
                    )
                return RemapResult(target_name=target_name)
        return None
