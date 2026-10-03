# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``GraniteSwitchModel._classifier_read_points`` (vLLM).

The read runs outside the compiled region and only needs ``input_ids``, the
final hidden states and ``query_start_loc``, so it is exercised here with a fake
forward context instead of a served model. Serving coverage, including chunked
prefill, lives in ``test_classifier_chunked_prefill.py``.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

import vllm.forward_context

from granite_switch.vllm.granite_switch_model import GraniteSwitchModel

# Two classifier slots: control id 90 fires slot 2, 91 fires slot 3.
_SLOT_OF = {90: 2, 91: 3}
_MODEL = SimpleNamespace(
    config=SimpleNamespace(classifier_control_token_ids=list(_SLOT_OF))
)


def _read(monkeypatch, slices, query_start_loc=None):
    """Run the read over one pass made of the given per-request token slices."""
    ids = torch.tensor([t for s in slices for t in s])
    # The switch's index stream: a marker's slot from the marker onward, per request.
    indices = []
    for s in slices:
        current = 0
        for t in s:
            current = _SLOT_OF.get(t, current)
            indices.append(current)
    classifier_indices = torch.tensor(indices)
    hidden = torch.arange(len(ids), dtype=torch.float32).unsqueeze(1)  # row i == i
    if query_start_loc is None:
        bounds = [0]
        for s in slices:
            bounds.append(bounds[-1] + len(s))
        query_start_loc = bounds
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(query_start_loc=torch.tensor(query_start_loc))
    )
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: ctx)
    return GraniteSwitchModel._classifier_read_points(
        _MODEL, classifier_indices, hidden, ids
    )


def test_reports_each_marker_to_its_requests_row(monkeypatch):
    rows, hidden, slots, num_reqs = _read(monkeypatch, [[5, 6, 90], [7, 8], [9, 91]])
    # Markers sit at flat 2 and 6, the last rows of requests 0 and 2.
    assert rows.tolist() == [0, 2]
    assert hidden.squeeze(1).tolist() == [2.0, 6.0]
    assert slots.tolist() == [2, 3]
    assert num_reqs == 3


def test_pass_without_a_marker_returns_none(monkeypatch):
    assert _read(monkeypatch, [[5, 6], [7]]) is None


def test_marker_before_the_last_token_raises(monkeypatch):
    with pytest.raises(RuntimeError, match=r"request\(s\) \[1\]"):
        _read(monkeypatch, [[5, 90], [90, 6, 7]])


def test_two_markers_raise_even_when_one_is_last(monkeypatch):
    with pytest.raises(RuntimeError, match="not their last token"):
        _read(monkeypatch, [[90, 5, 91]])


def test_empty_slice_does_not_shift_rows(monkeypatch):
    # A padded query_start_loc repeats its final offset: the third slice is empty.
    rows, _, slots, num_reqs = _read(
        monkeypatch, [[5, 90], [6]], query_start_loc=[0, 2, 3, 3]
    )
    assert rows.tolist() == [0]
    assert slots.tolist() == [2]
    assert num_reqs == 3


def test_no_metadata_returns_none(monkeypatch):
    ctx = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: ctx)
    out = GraniteSwitchModel._classifier_read_points(
        _MODEL, torch.tensor([2]), torch.zeros(1, 1), torch.tensor([90])
    )
    assert out is None
