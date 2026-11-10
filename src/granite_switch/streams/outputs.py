# SPDX-License-Identifier: Apache-2.0
"""What a head stream does to its row's logits.

Two output modes, applied by the same function on both backends:

* ``select`` -- the row becomes the head's scores on ``token_ids`` and ``-inf``
  everywhere else, so the sampler can only emit one of those tokens. This is
  the activation probe: one label word per request.
* ``merge`` -- the head's scores are added to the LM head's row on
  ``token_ids``, scaled by ``weight``; every other token keeps its logit.

The head returns one score per entry of ``token_ids`` (``num_outputs`` of them,
padded to the checkpoint's widest head; padded columns are never read).

The rewrite reads nothing back to the host: every table is built once from the
config (:func:`output_tables`), and each row gathers its slot's entries from
them. That keeps the logits step free of GPU waits and lets it run inside a
captured graph.
"""

import torch

OUTPUT_MODES = ("select", "merge")


def output_tables(specs, num_slots: int) -> dict:
    """Per 1-based slot (row 0 and non-head slots unused), from the stream specs.

    Returns ``{"ids", "cols", "select", "merge", "weight"}``:

    * ``ids`` / ``cols`` ``[num_slots + 1, width]`` long -- a head slot's
      output token ids and the score column each one takes. Padded to the
      widest head by repeating the first id and column, so a padded write
      repeats a real one with the same value.
    * ``select`` / ``merge`` ``[num_slots + 1]`` bool -- the slot's output mode.
    * ``weight`` ``[num_slots + 1]`` float -- a merge slot's weight.
    """
    heads = [s for s in specs if s.is_head]
    width = max((s.num_outputs for s in heads), default=0) or 1
    ids = torch.zeros(num_slots + 1, width, dtype=torch.long)
    cols = torch.zeros(num_slots + 1, width, dtype=torch.long)
    select = torch.zeros(num_slots + 1, dtype=torch.bool)
    merge = torch.zeros(num_slots + 1, dtype=torch.bool)
    weight = torch.zeros(num_slots + 1)
    for s in heads:
        n, pad = s.num_outputs, width - s.num_outputs
        ids[s.slot] = torch.tensor([*s.token_ids, *[s.token_ids[0]] * pad])
        cols[s.slot] = torch.tensor([*range(n), *[0] * pad])
        select[s.slot] = s.mode == "select"
        merge[s.slot] = s.mode == "merge"
        weight[s.slot] = s.weight
    return {
        "ids": ids,
        "cols": cols,
        "select": select,
        "merge": merge,
        "weight": weight,
    }


def apply_head_scores(
    logits: torch.Tensor,
    slots: torch.Tensor,
    scores: torch.Tensor,
    tables: dict,
) -> torch.Tensor:
    """Write each head row's scores into its ``logits`` row, in place; return it.

    Branch-free, with no host reads. Each row gathers its slot's output ids and
    score columns. A ``select`` row is blanked to ``-inf`` and gets its scores;
    a ``merge`` row adds ``weight * scores`` to its LM logits; any other row
    writes back the values it already had.

    Args:
        logits: ``[rows, vocab]``.
        slots: ``[rows]`` long, each row's 1-based slot (0 or a non-head slot
            for rows the heads leave alone). Must lie in ``[0, num_slots]``.
        scores: ``[rows, >= width]`` each row's head scores (only head rows'
            are read).
        tables: :func:`output_tables`, on ``logits``' device.
    """
    is_select = tables["select"][slots].unsqueeze(-1)
    is_merge = tables["merge"][slots].unsqueeze(-1)
    ids = tables["ids"][slots]
    own = scores.gather(1, tables["cols"][slots]).to(logits.dtype)
    logits.masked_fill_(is_select, float("-inf"))
    current = logits.gather(1, ids)
    weight = tables["weight"][slots].unsqueeze(-1).to(logits.dtype)
    new = torch.where(
        is_select, own, torch.where(is_merge, current + weight * own, current)
    )
    logits.scatter_(1, ids, new)
    return logits
