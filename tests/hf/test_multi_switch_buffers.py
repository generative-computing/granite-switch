# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch buffers must survive a save_pretrained -> from_pretrained round trip.

Regression test for a silent mis-routing bug: ``codebook`` and
``control_to_substitute_lut`` were registered ``persistent=False``, so they were
absent from the ``state_dict``. ``from_pretrained`` materializes modules on the
meta device and then fills only tensors present in the checkpoint, so whatever
``__init__`` computed for a non-persistent buffer is discarded and the buffer
comes back ALL ZEROS.

Consequences of the zeroed buffers:
  * ``codebook`` -> every memory key (``code(n) * gain``) and query (``code(n)``)
    is the zero vector, so all attention logits are 0, the retrieval softmax goes
    UNIFORM over the causally-visible positions, and each position AVERAGES the
    expert ids it can see instead of selecting the most recent one. Observed on a
    real composed checkpoint: ``[T,A,T,B,T]`` produced memory_raw
    ``[0, .504, .337, .756, .605]`` -> rounded to ``[0,1,0,1,1]`` instead of
    ``[0,1,1,2,2]``.
  * ``control_to_substitute_lut`` -> the LUT uses ``-1`` as its "not a control
    token" sentinel, so an all-zero LUT rewrites EVERY token id to 0.

Why the existing suites missed it: the bare-switch tests construct the switch
directly (``create_switch``) and never round-trip a checkpoint, so the buffers
keep their ``__init__`` values. Only a real save/load exercises the failure.
CPU-only and fast (tiny synthetic geometry).
"""

import pytest
import torch

from granite_switch.hf import GraniteSwitchForCausalLM
from tests.shared.generation_models import DENSE_CFG, make_switch_model
from tests.shared.multi_switch_cases import ATOK_NO_BASE, TEXT_TOKEN

NUM_ADAPTERS = 2


def _overrides(base_cfg):
    return {
        "vocab_size": max(DENSE_CFG["vocab_size"], max(ATOK_NO_BASE) + 1),
        "num_adapters": NUM_ADAPTERS,
        "adapter_ranks": [8] * NUM_ADAPTERS,
        "adapter_token_ids": list(ATOK_NO_BASE),
        "adapter_names": [f"adapter_{i}" for i in range(NUM_ADAPTERS)],
        "adapter_substitute_token_ids": [1, 2],
        "switch_type": "multi",
        # +2 layers: the coded switch owns 2 cache slots (counting + memory).
        "num_hidden_layers": len(base_cfg["layer_types"]) + 2,
        "layer_types": ["attention", "attention"] + base_cfg["layer_types"],
    }


@pytest.fixture
def round_tripped(tmp_path):
    """(reloaded_model, original_model) after a real save/load round trip."""
    base_cfg = dict(DENSE_CFG)
    model, config = make_switch_model(base_cfg, _overrides(base_cfg))
    model.model.adapter_token_ids.data = torch.tensor(
        config.adapter_token_ids, dtype=torch.long
    )
    out = tmp_path / "ckpt"
    model.save_pretrained(out)
    reloaded = GraniteSwitchForCausalLM.from_pretrained(out)
    return reloaded, model


def test_codebook_survives_round_trip(round_tripped):
    """The reloaded codebook must equal the deterministic construction."""
    reloaded, original = round_tripped
    got = reloaded.model.switch.codebook
    assert not torch.all(got == 0), (
        "codebook came back ALL ZEROS after from_pretrained -- the buffer is "
        "not persistent, so the memory softmax will go uniform and average adapters"
    )
    torch.testing.assert_close(got.float(), original.model.switch.codebook.float())


def test_codebook_rows_are_unit_norm(round_tripped):
    """Kerdock/DG codewords are unit-norm; a zeroed/rescaled buffer is not."""
    reloaded, _ = round_tripped
    cb = reloaded.model.switch.codebook.float()
    norms = cb.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)


def test_substitute_lut_survives_round_trip(round_tripped):
    """The LUT must keep its -1 sentinel and its control->substitute entries."""
    reloaded, original = round_tripped
    lut = reloaded.model.switch.control_to_substitute_lut
    assert lut is not None
    assert not torch.all(lut == 0), (
        "control_to_substitute_lut came back ALL ZEROS -- token exchange would "
        "rewrite every token id to 0 (the 'not a control' sentinel is -1)"
    )
    torch.testing.assert_close(lut, original.model.switch.control_to_substitute_lut)
    # Non-control ids keep the -1 sentinel; control ids map to substitutes.
    assert int(lut[TEXT_TOKEN]) == -1
    for ctrl, sub in zip(ATOK_NO_BASE, [1, 2]):
        assert int(lut[ctrl]) == sub


def test_routing_correct_after_round_trip(round_tripped):
    """The whole point: routing still selects latest-wins after a reload."""
    reloaded, _ = round_tripped
    a, b = ATOK_NO_BASE
    seq = [TEXT_TOKEN, a, TEXT_TOKEN, b, TEXT_TOKEN]
    with torch.no_grad():
        reloaded(input_ids=torch.tensor([seq]))
    got = reloaded.model._last_adapter_indices[0].tolist()
    assert got == [0, 1, 1, 2, 2], (
        f"routing after round trip is {got}, expected [0,1,1,2,2]; fractional/"
        "averaged indices indicate a zeroed codebook"
    )
