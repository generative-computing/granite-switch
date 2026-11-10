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
"""

import torch

OUTPUT_MODES = ("select", "merge")


def apply_head_scores(
    logits: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    scores: torch.Tensor,
    outputs: dict,
) -> torch.Tensor:
    """Write each head row's scores into ``logits``, in place, and return it.

    Args:
        logits: ``[num_rows, vocab]`` -- one logits row per position.
        rows: ``[n]`` row index of each head output in ``logits``.
        slots: ``[n]`` 1-based slot of each head output.
        scores: ``[n, width]`` head scores, padded to the widest head.
        outputs: ``{slot: (mode, token_ids LongTensor, weight float)}`` for every
            head slot, built once from the config (``StreamSpec.output``).
    """
    for i, (r, slot) in enumerate(zip(rows.tolist(), slots.tolist())):
        mode, token_ids, weight = outputs[slot]
        token_ids = token_ids.to(logits.device)
        own = scores[i, : token_ids.numel()].to(logits.dtype)
        if mode == "select":
            logits[r] = float("-inf")
            logits[r, token_ids] = own
        else:  # merge
            logits[r, token_ids] = logits[r, token_ids] + weight * own
    return logits
