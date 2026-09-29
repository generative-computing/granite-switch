# SPDX-License-Identifier: Apache-2.0
"""Compose-time loading of trained classifier-head weights.

A "classifier" slot is the classifier-head analog of a LoRA adapter: it shares
the same control-token / adapter-index machinery, but instead of LoRA weights it
carries a small linear head (``weight [n_labels, hidden]`` + ``bias [n_labels]``)
that emits a per-request verdict. Each classifier slot may declare a DIFFERENT
number of labels; the head bank is padded to ``max_classifier_labels`` (like the
LoRA bank pads ranks to ``max_lora_rank``), so a slot's trained head lands in the
top ``n_labels`` rows of its bank row and the pad rows stay zero. The compose
pipeline builds the ``SwitchedClassifierHead`` from config and transfers the
trained head weights from each classifier directory's
``classifier_head.safetensors`` into bank row ``slot = adapter_index - 1``.

These tests build the model through :class:`GraniteSwitchComposer` (per CLAUDE.md
rule #5 — never hand-assemble the config) against the real ``granite-4.0-micro``
base, with a *synthetic* classifier head whose weights make one label win
deterministically, and assert:

1. ``classifier_head.weight[slot, :n_labels]`` / ``.bias[slot, :n_labels]`` exactly
   equal the synthetic tensors, and the pad rows above ``n_labels`` stay zero.
2. A classifier control-token forward emits the winning label's token id (the HF
   verdict exit scatters the head's argmax onto that slot's
   ``classifier_label_token_ids``).
3. In a mixed ``[lora, classifier]`` compose the LoRA slot's bank row is populated
   while the classifier slot's LoRA row stays zero (compose orders LoRAs first, so
   slot numbering is stable), and the classifier row lands in its slot.
4. Two classifier slots with DIFFERENT label counts (2 and 3) each land padded
   correctly and each emit a token id from their own label set.

Marked slow + requires_model: composing the ~3B micro base is expensive, but it is
the only sanctioned construction path.
"""

import glob
import json
import os

import pytest
import torch

import granite_switch.hf  # noqa: F401 — registers AutoModel/AutoConfig
from granite_switch.composer.compose_utils import GraniteSwitchComposer
from granite_switch.composer.weight_transfer import CLASSIFIER_HEAD_FILE

BASE_MODEL = "ibm-granite/granite-4.0-micro"
# The mixed test embeds a real LoRA, which must match its base. The RAG library
# ships granite-4.1-3b LoRAs, so the mixed case composes against that base.
MIXED_BASE = "ibm-granite/granite-4.1-3b"
MIXED_LORA_TARGET = "granite-4.1-3b"
RAG_LIBRARY = "ibm-granite/granite-lib-rag-r1.0"

pytestmark = [pytest.mark.slow, pytest.mark.requires_model]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tokenizer(model=BASE_MODEL):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model)
    except Exception as e:  # pragma: no cover - network/env dependent
        pytest.skip(f"could not fetch tokenizer {model!r}: {e}")


def _hidden_size(model=BASE_MODEL):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model).hidden_size
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not fetch config {model!r}: {e}")


def _single_token_labels(tokenizer, candidates, count=2):
    """Pick ``count`` label words that each encode to exactly one token id.

    The classifier verdict exit writes at one logit position per request, so a
    label must be a single token. Returns ``(names, ids)``.
    """
    names, ids = [], []
    for word in candidates:
        enc = tokenizer.encode(word, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            names.append(word)
            ids.append(enc[0])
        if len(names) == count:
            return names, ids
    pytest.skip(f"could not find {count} single-token labels among {candidates}")


def _write_synthetic_head(directory, weight, bias, labels):
    """Write a classifier slot dir: classifier_head.safetensors (+ config)."""
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    save_file(
        {"weight": weight.contiguous(), "bias": bias.contiguous()},
        os.path.join(directory, CLASSIFIER_HEAD_FILE),
    )
    # Optional manifest metadata — the composer reads kinds/labels from kwargs,
    # not this file, but a real classifier dir carries it, so include it.
    with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        json.dump({"kind": "classifier", "labels": list(labels)}, f)


def _winning_head(num_labels, hidden, winner, dtype=torch.float32):
    """Head whose winning label wins for ANY hidden state (bias-only).

    All weight rows are zero, so each label's logit reduces to its bias alone —
    independent of the (real, arbitrary-magnitude) hidden state. Row ``winner``
    gets a large positive bias and the rest a large negative one, so ``argmax``
    over labels is deterministically ``winner``. A zero weight row is a valid
    trained artifact shape and is the only construction that is robust to the
    hidden state's sign and scale on a real base model.
    """
    weight = torch.zeros((num_labels, hidden), dtype=dtype)
    bias = torch.full((num_labels,), -10.0, dtype=dtype)
    bias[winner] = 10.0
    return weight, bias


def _classifier_prompt_ids(tokenizer, control_token_id):
    """`<some prompt><control>` as a batch of input ids for a detect forward.

    End-locator layout, matching what the chat template emits: the control token
    (marker) goes AFTER the last content token, so the verdict is read at
    ``marker - 1`` -- the position the head is trained on. A marker at position 0
    has no content token before it and is rejected by the backend.
    """
    body = tokenizer.encode("Is this text safe?", add_special_tokens=False)
    ids = [*body, control_token_id]
    return torch.tensor([ids], dtype=torch.long)


def _find_rag_lora_dir():
    """Locate a real cached LoRA adapter dir for the mixed-case test.

    Reads the local HF cache directly (via ``scan_cache_dir``) rather than
    ``snapshot_download``: the RAG library is large and only partially cached
    here, and ``snapshot_download(local_files_only=True)`` refuses an incomplete
    snapshot even when the one adapter this test needs is present. Skip cleanly
    if no matching adapter is cached.
    """
    from huggingface_hub import scan_cache_dir

    snapshot_dirs = []
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == RAG_LIBRARY:
                snapshot_dirs.extend(str(rev.snapshot_path) for rev in repo.revisions)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not scan HF cache for {RAG_LIBRARY!r}: {e}")

    for root in snapshot_dirs:
        hits = glob.glob(
            os.path.join(
                root, "**", MIXED_LORA_TARGET, "lora", "adapter_model.safetensors"
            ),
            recursive=True,
        )
        # Follow the cache symlink to the real blob; skip broken (missing) ones.
        hits = [h for h in sorted(hits) if os.path.exists(os.path.realpath(h))]
        if hits:
            return os.path.dirname(hits[0])

    pytest.skip(
        f"no cached {MIXED_LORA_TARGET} LoRA adapter with weights in {RAG_LIBRARY!r}"
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_classifier_only_compose_loads_trained_head(tmp_path):
    """A single classifier slot: weights land in the bank and the verdict fires."""
    tokenizer = _tokenizer()
    hidden = _hidden_size()

    label_names, label_ids = _single_token_labels(
        tokenizer, ["safe", "unsafe", "yes", "no", "true", "false"]
    )
    num_labels = len(label_names)
    winner = 1  # second label wins
    weight, bias = _winning_head(num_labels, hidden, winner)

    clf_dir = str(tmp_path / "detect")
    _write_synthetic_head(clf_dir, weight, bias, label_names)

    # A control token id for the single classifier slot. Any spare vocab id
    # works for construction; the switch fires on this id, which the prompt
    # helper places after the content (end-locator layout).
    control_id = tokenizer.vocab_size - 5

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=BASE_MODEL,
        adapter_paths=[clf_dir],
        adapter_token_ids=[control_id],
        adapter_substitute_token_ids=[control_id],  # inert for a classifier slot
        adapter_names=["detect"],
        adapter_kinds=["classifier"],
        # Full-width per-slot label count (single classifier slot -> [num_labels]);
        # ragged label metadata is one list per classifier slot.
        classifier_label_token_ids=[label_ids],
    )
    model.eval()

    # (1) The trained head is in bank row slot = index - 1 = 0, top n_labels rows.
    head = model.model.classifier_head
    assert head is not None, "classifier_head was not built"
    torch.testing.assert_close(
        head.weight[0, :num_labels].float(), weight.to(head.weight.dtype).float()
    )
    torch.testing.assert_close(
        head.bias[0, :num_labels].float(), bias.to(head.bias.dtype).float()
    )

    # (2) A detect forward emits the winning label's token id.
    input_ids = _classifier_prompt_ids(tokenizer, control_id).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    emitted = int(out.logits[0, -1].argmax().item())
    assert emitted == label_ids[winner], (
        f"expected winning label id {label_ids[winner]} ({label_names[winner]}), "
        f"got {emitted}"
    )


def test_mixed_lora_and_classifier_preserves_slot_numbering(tmp_path):
    """[lora, classifier]: LoRA slot 0 populated, classifier slot 1 loaded."""
    tokenizer = _tokenizer(MIXED_BASE)
    hidden = _hidden_size(MIXED_BASE)
    lora_dir = _find_rag_lora_dir()

    label_names, label_ids = _single_token_labels(
        tokenizer, ["safe", "unsafe", "yes", "no", "true", "false"]
    )
    num_labels = len(label_names)
    winner = 0
    weight, bias = _winning_head(num_labels, hidden, winner)

    clf_dir = str(tmp_path / "detect")
    _write_synthetic_head(clf_dir, weight, bias, label_names)

    lora_ctrl = tokenizer.vocab_size - 6
    clf_ctrl = tokenizer.vocab_size - 5

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=MIXED_BASE,
        adapter_paths=[lora_dir, clf_dir],
        adapter_token_ids=[lora_ctrl, clf_ctrl],
        adapter_substitute_token_ids=[lora_ctrl, clf_ctrl],
        adapter_names=["rag_lora", "detect"],
        adapter_kinds=["lora", "classifier"],
        # Full width: None on the LoRA slot, ids on the classifier slot.
        classifier_label_token_ids=[None, label_ids],
    )
    model.eval()

    head = model.model.classifier_head
    assert head is not None

    # Classifier slot is index 1 -> bank row 1; slot 0 (the LoRA slot) is never
    # touched by the classifier transfer, so it stays at the bank's zero init.
    torch.testing.assert_close(
        head.weight[1, :num_labels].float(), weight.to(head.weight.dtype).float()
    )
    torch.testing.assert_close(
        head.bias[1, :num_labels].float(), bias.to(head.bias.dtype).float()
    )
    # Slot 0's classifier row must stay all-zero (never loaded) — a stronger check
    # than "not equal to the synthetic head" now that the bank is zero-init.
    assert head.weight[0].abs().sum() == 0, (
        "the LoRA slot's classifier row was overwritten"
    )
    assert head.bias[0].abs().sum() == 0, (
        "the LoRA slot's classifier bias was overwritten"
    )

    # The LoRA bank row for slot 0 must be populated, while the classifier
    # slot's LoRA row (slot 1) must stay unloaded. lora_B is the discriminator:
    # it is ZERO-initialized and only becomes nonzero when an adapter is loaded
    # (lora_A is kaiming-initialized, so an unloaded row is nonzero there too and
    # can't distinguish loaded from unloaded). So: slot 0 lora_B nonzero (loaded),
    # slot 1 lora_B all-zero (classifier slot, never received LoRA weights).
    lora_B = _first_lora_b_tensor(model)
    assert lora_B is not None, "no stacked LoRA B tensor found on the model"
    assert lora_B[0].abs().sum() > 0, "LoRA slot 0 bank row is unexpectedly all-zero"
    assert lora_B[1].abs().sum() == 0, (
        "classifier slot 1 LoRA row is not zero — slot numbering leaked"
    )

    # A detect forward on the classifier control token emits the winning label.
    input_ids = _classifier_prompt_ids(tokenizer, clf_ctrl).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    emitted = int(out.logits[0, -1].argmax().item())
    assert emitted == label_ids[winner]


def test_ragged_label_counts_pad_and_fire_per_slot(tmp_path):
    """Two classifier slots with DIFFERENT label counts (2 and 3).

    Each head lands in the top rows of its (padded) bank row, the pad rows above
    its own label count stay zero, and a detect forward on each slot's control
    token emits a token id from THAT slot's label set — proving the padded
    einsum/scatter never leaks a phantom (padded) label across slots.
    """
    tokenizer = _tokenizer()
    hidden = _hidden_size()

    # Slot A: 2 labels; slot B: 3 labels. Distinct token ids across both sets.
    names_a, ids_a = _single_token_labels(
        tokenizer, ["safe", "unsafe", "yes", "no", "true", "false"], count=2
    )
    names_b, ids_b = _single_token_labels(
        tokenizer,
        ["good", "bad", "neutral", "high", "low", "medium", "left", "right", "up"],
        count=3,
    )
    n_a, n_b = len(names_a), len(names_b)  # 2, 3
    max_labels = max(n_a, n_b)

    winner_a, winner_b = 1, 2  # last label of each set wins
    w_a, b_a = _winning_head(n_a, hidden, winner_a)
    w_b, b_b = _winning_head(n_b, hidden, winner_b)

    dir_a = str(tmp_path / "detect_a")
    dir_b = str(tmp_path / "detect_b")
    _write_synthetic_head(dir_a, w_a, b_a, names_a)
    _write_synthetic_head(dir_b, w_b, b_b, names_b)

    ctrl_a = tokenizer.vocab_size - 5
    ctrl_b = tokenizer.vocab_size - 6

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=BASE_MODEL,
        adapter_paths=[dir_a, dir_b],
        adapter_token_ids=[ctrl_a, ctrl_b],
        adapter_substitute_token_ids=[ctrl_a, ctrl_b],
        adapter_names=["detect_a", "detect_b"],
        adapter_kinds=["classifier", "classifier"],
        classifier_label_token_ids=[ids_a, ids_b],
    )
    model.eval()

    head = model.model.classifier_head
    assert head is not None
    # Bank is padded to max_labels; each slot's head fills its top n rows.
    assert head.weight.shape[1] == max_labels

    torch.testing.assert_close(
        head.weight[0, :n_a].float(), w_a.to(head.weight.dtype).float()
    )
    torch.testing.assert_close(
        head.bias[0, :n_a].float(), b_a.to(head.bias.dtype).float()
    )
    torch.testing.assert_close(
        head.weight[1, :n_b].float(), w_b.to(head.weight.dtype).float()
    )
    torch.testing.assert_close(
        head.bias[1, :n_b].float(), b_b.to(head.bias.dtype).float()
    )

    # Slot A has fewer labels than max: its pad rows (n_a..max-1) must stay zero.
    assert torch.count_nonzero(head.weight[0, n_a:]) == 0, "slot A pad rows not zero"
    assert torch.count_nonzero(head.bias[0, n_a:]) == 0, "slot A pad bias not zero"

    device = next(model.parameters()).device
    # Slot A fires -> emits one of A's ids (never a B id, never a padded slot).
    with torch.no_grad():
        out_a = model(
            input_ids=_classifier_prompt_ids(tokenizer, ctrl_a).to(device),
            use_cache=False,
        )
    emitted_a = int(out_a.logits[0, -1].argmax().item())
    assert emitted_a == ids_a[winner_a], f"slot A emitted {emitted_a}, want {ids_a}"
    assert emitted_a not in ids_b, "slot A leaked a slot-B label"

    # Slot B fires -> emits one of B's ids, including its 3rd (only-in-B) label.
    with torch.no_grad():
        out_b = model(
            input_ids=_classifier_prompt_ids(tokenizer, ctrl_b).to(device),
            use_cache=False,
        )
    emitted_b = int(out_b.logits[0, -1].argmax().item())
    assert emitted_b == ids_b[winner_b], f"slot B emitted {emitted_b}, want {ids_b}"


def _first_lora_b_tensor(model):
    """Return the first stacked LoRA-B bank tensor [num_adapters, 1, out, r].

    lora_B is zero-init and only nonzero once an adapter is loaded, so it cleanly
    distinguishes a loaded LoRA slot from an unloaded (classifier) slot — unlike
    lora_A, which is kaiming-init and nonzero even when never loaded.
    """
    for name, buf in model.named_parameters():
        if "lora_B" in name and buf.dim() == 4:
            return buf.detach()
    for name, buf in model.named_buffers():
        if "lora_B" in name and buf.dim() == 4:
            return buf.detach()
    return None
