# SPDX-License-Identifier: Apache-2.0
"""Public head contract failures, without either backend."""

import pytest
import torch

from granite_switch.streams import HeadType, get_head_type, register_head_type, registry
from granite_switch.streams.outputs import fit_head_scores
from granite_switch.streams.spec import parse_entries


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


def test_output_fit_compiles_without_host_reads():
    def run(scores):
        return fit_head_scores(scores, scores.shape[0], 2, scores, "example")

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    scores = torch.ones(5, 2)
    torch.testing.assert_close(compiled(scores), scores)
