# SPDX-License-Identifier: Apache-2.0
"""Token-exchange helpers, shared by every switch engine on both backends.

These are the small pure functions behind the runtime token-exchange: the
switch reads the ORIGINAL ``input_ids`` for adapter selection, then rewrites
each control-token id to its substitute id so the decoder embeds a clean
sequence and never knows a control token existed.

Backend-neutral on purpose. The table is a pure function of three config
fields (``vocab_size``, ``adapter_token_ids``, ``adapter_substitute_token_ids``),
so it belongs at the same tier as :mod:`granite_switch.config` rather than
inside either backend — all four switch engines (HF/vLLM x Single/Multi) build
the identical table and must agree on its size.

Contract:
  - ``build_control_to_substitute_lut(config)`` -> LongTensor with ``-1`` at
    non-control ids and the substitute id at each adapter control id, or
    ``None`` when no substitute ids are configured.
  - ``rebuild_control_to_substitute_lut(switch, config)`` -> re-derive the
    table on an already-constructed switch after a vocabulary change.
  - ``apply_token_exchange(lut, input_ids)`` -> ``input_ids`` with each control
    token rewritten to its substitute id. Branch-free (``torch.where``) for
    torch.compile / @support_torch_compile compatibility.

Named to match the buffer the switches register (``control_to_substitute_lut``)
so grepping either name finds the whole mechanism.
"""

import torch


def build_control_to_substitute_lut(config) -> torch.Tensor | None:
    """Derive the control->substitute lookup table from *config*.

    Shape ``[max(vocab_size, max_ctrl_id + 1)]``: ``-1`` at every non-control id
    and the substitute id at each control slot.

    **The ``max`` is load-bearing, not defensive.** It fires on every compose. The
    switch is constructed from a config whose ``vocab_size`` was copied verbatim
    from the base checkpoint, while ``add_control_tokens`` has already appended the
    control ids *past* that number — so at construction time
    ``max_ctrl_id + 1 > vocab_size`` always holds, and sizing at ``vocab_size``
    alone would make ``lut[ctrl_id] = sub_id`` below raise ``IndexError``. For
    granite-4.1-3b with two adapters that is ``max(100352, 100354) -> 100354``.

    So this function and :func:`~granite_switch.composer.validator.validate_control_lut`
    encode the same invariant at two different points in the lifecycle, and they
    are meant to disagree in between:

    1. **here, pre-resize** — the table is deliberately longer than
       ``config.vocab_size``, because the vocabulary has not caught up yet;
    2. ``resize_token_embeddings`` grows ``vocab_size`` to cover the new tokens;
    3. :func:`rebuild_control_to_substitute_lut` re-derives the table, which now
       comes out at exactly ``vocab_size`` (the ``max`` no longer fires);
    4. **validate_control_lut, pre-save** — demands strict equality, which by then
       is the only correct answer.

    Read either site alone and the two rules look contradictory. They are not: a
    table longer than ``vocab_size`` is correct at step 1 and a bug at step 4. The
    window between them is closed by step 3, which is why compose calls
    ``refresh_switch_control_lut`` before validating.

    Only the real adapter control tokens (``adapter_token_ids``) are mapped; a
    base-reset token, if present, is intentionally NOT rewritten here (its
    substitution is handled by its own substitute entry when present, and it
    never needs an OOD guard because it is only placed at in-distribution turn
    boundaries).

    Returns ``None`` when *config* carries no token-exchange mapping, in which
    case the switch leaves ``input_ids`` untouched. An empty id list counts as
    no mapping — checked with a falsiness test rather than ``is None`` because
    ``max(())`` raises.

    Single source of truth for the sizing rule: anything that changes
    ``vocab_size``, ``adapter_token_ids`` or ``adapter_substitute_token_ids``
    must re-derive the table (see :func:`rebuild_control_to_substitute_lut`).
    """
    if config is None:
        return None
    ctrl_ids = getattr(config, "adapter_token_ids", None)
    sub_ids = getattr(config, "adapter_substitute_token_ids", None)
    if not ctrl_ids or not sub_ids:
        return None

    lut_size = max(getattr(config, "vocab_size", 0), max(ctrl_ids) + 1)
    lut = torch.full((lut_size,), -1, dtype=torch.long)
    for ctrl_id, sub_id in zip(ctrl_ids, sub_ids):
        lut[ctrl_id] = sub_id
    return lut


def rebuild_control_to_substitute_lut(switch, config=None) -> bool:
    """Re-derive *switch*'s control->substitute table after a vocabulary change.

    A switch sizes its table from ``config.vocab_size`` at construction, so
    anything that grows the vocabulary afterwards — notably
    ``resize_token_embeddings`` when compose adds control and marker tokens —
    leaves the buffer shorter than the config it will be saved alongside.

    That matters because the buffer is persistent. On ``from_pretrained``, a
    stored tensor whose shape disagrees with the freshly-constructed one is
    discarded and the buffer is left as uninitialised memory (there is no
    ``_init_weights`` rule for it), so every id reads as a control id and the
    rewrite sends out-of-range ids into the embedding gather. Re-derive the
    table before saving so the checkpoint and its config agree.

    A free function rather than a method because rebuilding is something
    *compose* does *to* a switch once, not runtime behavior of the switch --
    and because there are four independent switch classes across two backends
    with no common base. Duck-typed on the buffer, so it works on all of them
    and cannot be present on one engine and missing on another.

    Args:
        switch: Any switch module carrying a ``control_to_substitute_lut``.
        config: Config to derive from. Falls back to ``switch.config``.

    Returns:
        ``True`` if a table was rebuilt, ``False`` if there is no
        token-exchange mapping to rebuild.
    """
    lut = build_control_to_substitute_lut(
        config if config is not None else getattr(switch, "config", None)
    )
    if lut is None:
        return False

    existing = getattr(switch, "control_to_substitute_lut", None)
    if existing is not None:
        # Already a registered buffer: assign through, keeping its device.
        # nn.Module.__setattr__ writes back into _buffers for a known name.
        switch.control_to_substitute_lut = lut.to(device=existing.device)
        return True

    # Constructed without a mapping, so the name is a plain None attribute
    # rather than a buffer. register_buffer refuses a name that already exists
    # outside _buffers, so drop the placeholder first.
    if "control_to_substitute_lut" in switch.__dict__:
        del switch.__dict__["control_to_substitute_lut"]
    switch.register_buffer("control_to_substitute_lut", lut)
    return True


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
