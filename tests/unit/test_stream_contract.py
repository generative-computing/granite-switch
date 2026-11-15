# SPDX-License-Identifier: Apache-2.0
"""Public head contract failures and exact transport, without either backend."""

import pytest
import torch

from granite_switch.streams import HeadType, get_head_type, register_head_type, registry
from granite_switch.streams.outputs import fit_head_scores
from granite_switch.streams.spec import parse_entries
from granite_switch.streams.transport import pack_slots, unpack_slots


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    get_head_type("probe")
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))


@pytest.mark.parametrize("name", ["bad.name", "bad-name", 12, True])
def test_type_name_is_an_identifier(name):
    candidate = type("Candidate", (HeadType,), {"name": name})
    with pytest.raises(ValueError, match="name must be an identifier"):
        register_head_type(candidate)


@pytest.mark.parametrize("attr", ["", "bad.name", "_head_width", 7])
def test_module_name_is_a_public_identifier(attr):
    candidate = type(
        "Candidate", (HeadType,), {"name": "candidate", "module_name": attr}
    )
    with pytest.raises(ValueError, match="module_name must be a public identifier"):
        register_head_type(candidate)


def test_module_name_cannot_shadow_another_type():
    class Candidate(HeadType):
        name = "candidate"
        module_name = "probe_head"

    with pytest.raises(ValueError, match="already used by head type 'probe'"):
        register_head_type(Candidate)


@pytest.mark.parametrize("output", [[], "select", 3, False, None])
def test_output_must_be_a_mapping(output):
    with pytest.raises(ValueError, match="output must be a dict"):
        parse_entries([{"type": "probe", "head_row": 0, "output": output}], 1)


@pytest.mark.parametrize("ids", [3, "123", False, None])
def test_token_ids_must_be_a_sequence(ids):
    with pytest.raises(ValueError, match="token_ids must be a list"):
        parse_entries(
            [{"type": "probe", "head_row": 0, "output": {"token_ids": ids}}], 1
        )


@pytest.mark.parametrize(
    "scores",
    [
        torch.zeros(2, 1),
        torch.zeros(1, 2),
        torch.zeros(2),
        torch.zeros(2, 2, dtype=torch.long),
        None,
    ],
)
def test_invalid_output_is_not_silently_padded_or_broadcast(scores):
    with pytest.raises(ValueError, match="Head 'example'.*at least 2 columns"):
        fit_head_scores(scores, 2, 2, torch.zeros(2, 3), "example")


def test_valid_output_is_padded_only_for_other_types():
    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    fitted = fit_head_scores(scores, 2, 2, torch.zeros(2, 4), "example")
    torch.testing.assert_close(
        fitted, torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]])
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_slot_transport_is_exact_across_byte_boundaries(dtype):
    slots = torch.cat([torch.arange(65538), torch.tensor([2**24 + 1, 2**31 - 1])])
    packed = pack_slots(slots, 4, dtype)
    assert packed.dtype == dtype
    torch.testing.assert_close(unpack_slots(packed), slots)


def test_slot_transport_and_output_fit_compile_without_host_reads():
    def run(slots, scores):
        decoded = unpack_slots(pack_slots(slots, 2, torch.bfloat16))
        return decoded, fit_head_scores(scores, scores.shape[0], 2, scores, "example")

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    slots = torch.tensor([0, 255, 256, 257, 513])
    scores = torch.ones(5, 2)
    decoded, fitted = compiled(slots, scores)
    torch.testing.assert_close(decoded, slots)
    torch.testing.assert_close(fitted, scores)


def test_invalid_profile_bytes_do_not_index_the_output_tables():
    packed = torch.tensor([[float("nan"), 0], [float("inf"), 0], [-1, 0], [256, 0]])
    assert unpack_slots(packed).tolist() == [0, 0, 0, 0]
