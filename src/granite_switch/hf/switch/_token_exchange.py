# SPDX-License-Identifier: Apache-2.0
"""Token-exchange helpers, shared by SingleSwitch and the MultiSwitch engines.

These are the small pure functions behind the runtime token-exchange: the
switch reads the ORIGINAL ``input_ids`` for adapter selection, then rewrites
each control-token id to its substitute id so the decoder embeds a clean
sequence and never knows a control token existed.

Contract:
  - ``build_substitute_lut(config)`` -> vocab-sized LongTensor with ``-1`` at
    non-control ids and the substitute id at each adapter control id, or
    ``None`` when no substitute ids are configured.
  - ``apply_token_exchange(lut, input_ids)`` -> ``input_ids`` with each control
    token rewritten to its substitute id. Branch-free (``torch.where``) for
    torch.compile / @support_torch_compile compatibility.

SingleSwitch keeps its own inline LUT construction for now; the MultiSwitch
engines (scan + coded) share this module so the token-exchange behavior is
identical across all engines.
"""

import torch


def build_substitute_lut(config) -> torch.Tensor | None:
    """Build the control-id -> substitute-id lookup table from config.

    Mirrors the SingleSwitch ``__init__`` LUT block. Only the real adapter
    control tokens (``adapter_token_ids``) are mapped; a base-reset token, if
    present, is intentionally NOT rewritten here (its substitution is handled
    by its own substitute entry when present, and it never needs an OOD guard
    because it is only placed at in-distribution turn boundaries).
    """
    ctrl_ids = getattr(config, "adapter_token_ids", None)
    sub_ids = getattr(config, "adapter_substitute_token_ids", None)
    if ctrl_ids is None or sub_ids is None:
        return None

    max_ctrl_id = max(ctrl_ids)
    lut_size = max(getattr(config, "vocab_size", 0), max_ctrl_id + 1)
    lut = torch.full((lut_size,), -1, dtype=torch.long)
    for ctrl_id, sub_id in zip(ctrl_ids, sub_ids):
        lut[ctrl_id] = sub_id
    return lut


def apply_token_exchange(
    lut: torch.Tensor | None, input_ids: torch.Tensor
) -> torch.Tensor:
    """Rewrite each control token's id to its substitute id via ``lut``.

    No data-dependent branch: ``torch.where`` runs every step (the decoder is
    wrapped in @support_torch_compile in vLLM, which forbids ``tensor.any()``
    short-circuits). When ``lut`` is ``None`` the input is returned unchanged.
    """
    if lut is None:
        return input_ids
    sub_id_per_pos = lut[input_ids]
    is_control = sub_id_per_pos >= 0
    return torch.where(is_control, sub_id_per_pos, input_ids)
