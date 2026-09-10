# SPDX-License-Identifier: Apache-2.0
"""Real compose of ``--enable-audio`` — the combination that crashed.

``compose --enable-audio`` used to die at the end of ``build()``:

    AttributeError: 'MultiSwitch' object has no attribute
    'rebuild_control_to_substitute_lut'

The control->substitute table is sized ``max(base_vocab_size, max_ctrl_id + 1)``
and control ids are appended, so after N control tokens it already equals
``len(tokenizer)`` and the rebuild is skipped — text-only compose is fine.
``<|audio|>`` adds a token that is *not* a control token, pushing the vocabulary
one past the last control id, which makes the table stale and sends compose into
the rebuild that used to be missing on ``MultiSwitch``.

Markers are ``slow`` + ``requires_model`` + ``audio``, matching
``test_compose_e2e.py`` and deliberately NOT env-gated: gating is what hid the
bug. ``tests/composer/test_control_lut_refresh.py`` covers the same call site in
seconds on a tiny model; this is the end-to-end proof that a real compose of the
combination completes and ships a loadable checkpoint.

Cost: one compose of the default base model (module-scoped).
"""

import json
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.audio]

# Same adapter library as test_compose_e2e.py, with the default base model.
# --technology-filter lora keeps the build to the cheapest useful shape.
ADAPTER_LIBRARY = "ibm-granite/granite-lib-rag-r1.0"
BUILD_TIMEOUT = 3600
_LUT_KEY = "model.switch.control_to_substitute_lut"


def _compose(output_dir):
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--adapters",
        ADAPTER_LIBRARY,
        "--technology-filter",
        "lora",
        "--enable-audio",
        "--output",
        str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=BUILD_TIMEOUT)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-3000:])

    # The regression itself: this returned non-zero with an AttributeError
    # traceback for MultiSwitch (the only engine).
    assert result.returncode == 0, (
        f"compose --enable-audio failed (exit {result.returncode}).\n"
        f"STDOUT tail:\n{result.stdout[-2000:]}\n"
        f"STDERR tail:\n{result.stderr[-2000:]}"
    )
    return output_dir


@pytest.fixture(scope="module")
def audio_multi(tmp_path_factory):
    return _compose(tmp_path_factory.mktemp("audio-multi") / "model")


def _lut_numel(output_dir):
    """Read the saved LUT's length without loading the whole checkpoint."""
    from safetensors import safe_open

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard = index["weight_map"].get(_LUT_KEY)
        assert shard is not None, (
            f"{_LUT_KEY} is missing from the checkpoint index. It is a persistent "
            f"buffer; if it stopped being saved, from_pretrained leaves it "
            f"uninitialised and every token reads as a control token."
        )
        files = [output_dir / shard]
    else:
        files = [output_dir / "model.safetensors"]

    for path in files:
        with safe_open(str(path), framework="pt") as f:
            if _LUT_KEY in f.keys():
                return f.get_slice(_LUT_KEY).get_shape()[0]
    pytest.fail(f"{_LUT_KEY} not found in {[p.name for p in files]}")


def _assert_consistent(output_dir):
    config = json.loads((output_dir / "config.json").read_text())

    vocab_size = config["vocab_size"]
    ctrl_ids = config["adapter_token_ids"]
    lut_numel = _lut_numel(output_dir)

    # The table must match the config it ships beside, or from_pretrained
    # discards it — see validate_control_lut for why that is unrecoverable.
    assert lut_numel == vocab_size, (
        f"saved {_LUT_KEY} has {lut_numel} entries but config.vocab_size is "
        f"{vocab_size}; this checkpoint cannot be loaded correctly."
    )

    # Guard the guard: confirm this compose actually exercised the stale-table
    # path. The audio marker has to sit past the last control id, or the rebuild
    # was a no-op and this test stopped covering the bug.
    assert vocab_size > max(ctrl_ids) + 1, (
        f"vocab_size {vocab_size} is not past the last control id "
        f"{max(ctrl_ids)}, so the audio marker did not extend the vocabulary and "
        f"the rebuild branch was never entered. This test no longer covers the "
        f"multi+audio crash — check that --enable-audio still adds a non-control "
        f"token."
    )


@pytest.mark.xdist_group("multi_audio_compose_e2e")
def test_multi_audio_compose_ships_a_consistent_control_lut(audio_multi):
    """The regression: this compose used to raise AttributeError."""
    _assert_consistent(audio_multi)


@pytest.mark.xdist_group("multi_audio_compose_e2e")
def test_multi_audio_checkpoint_loads_without_size_overrides(audio_multi):
    """The multi+audio checkpoint round-trips under a strict load.

    No ``ignore_mismatched_sizes``: a stale buffer would be silently dropped and
    left as uninitialised memory, which is exactly what validate_control_lut
    exists to prevent.
    """
    torch = pytest.importorskip("torch")

    from granite_switch.hf import GraniteSwitchForCausalLM

    model = GraniteSwitchForCausalLM.from_pretrained(
        str(audio_multi), dtype=torch.float32
    ).eval()

    lut = model.model.switch.control_to_substitute_lut
    assert lut is not None
    assert lut.numel() == model.config.vocab_size
    for ctrl_id, sub_id in zip(
        model.config.adapter_token_ids,
        model.config.adapter_substitute_token_ids,
    ):
        assert lut[ctrl_id].item() == sub_id, ctrl_id
    # Every non-control id stays a sentinel, so ordinary tokens pass through.
    assert int((lut >= 0).sum()) == len(model.config.adapter_token_ids)
