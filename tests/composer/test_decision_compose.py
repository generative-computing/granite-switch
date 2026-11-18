# SPDX-License-Identifier: Apache-2.0
"""Decision adapters through the real composer: io.yaml ``decision:`` block -> config.json.

Synthetic adapters are composed onto fake bases (both chat-template families: 4.1
role markers, 4.2 ChatML) by the compose CLI with ``--decision-adapter``:

- an aLoRA whose invocation tokens spell ``### DECISION`` (``marker`` activation);
- a LoRA (``start``) and a Shadow Residual adapter (``generation``), with no marker.

The tests check that config.json records the spec (marker and activation taken from
the adapter itself, not repeated in io.yaml); that the vLLM strategy, run over the
real composed tokenizer, renders each read with exactly one control token in the
right place, the right prompt tail, and the right label mapping; that a disagreeing
marker or a missing block fails the compose; and that ``decision_config`` stamps an
already composed checkpoint and refuses a marker the template does not activate on.

CPU-only: tiny fake bases (tests/shared/fake_bases.py) and synthetic weights.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from granite_switch.composer.decision_config import main as stamp_main
from granite_switch.decision_spec import DecisionSpecError, check_activation
from tests.shared import model_sources
from tests.shared.base_models import dims_from_config
from tests.shared.decision_fixtures import (
    LABEL_READS,
    MARKER,
    build_strategy,
    install_fake_vllm,
    load_strategy_module,
    question,
    reference_block,
    run_read,
)
from tests.shared.synthetic_adapters import (
    PROFILES,
    alora_invocation_ids,
    attn_modules,
    base_weight_norms,
    sr_activation_anchor,
    synthesize_adapter,
)

pytestmark = pytest.mark.slow

BASES = ["granite-4.1-3b", "granite-4.2-3b"]
#: alora_header: an aLoRA whose invocation is the assistant header (the trained 4.2
#: adapters), activated before the generation prompt with no marker in the prompt.
KINDS = ["alora", "alora_header", "lora", "sr"]


def _block(kind: str, **overrides) -> dict:
    """The reference block as an adapter author writes it: no activation and no
    marker (the composer takes both from the adapter); no marker turn for LoRA/SR."""
    block = reference_block(**overrides)
    block.pop("activation", None)
    if "invocation_marker" not in overrides:
        block.pop("invocation_marker")
    if kind != "alora":
        block["messages"][1]["content"] = block["messages"][1]["content"].replace(
            "{marker}\n", ""
        )
    return block


def _adapter(root: Path, base: str, kind: str, block: dict | None) -> Path:
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    extra = {}
    if kind == "alora":
        extra["alora_invocation_tokens"] = tokenizer.encode(
            MARKER, add_special_tokens=False
        )
    elif kind == "alora_header":
        extra["alora_invocation_tokens"] = alora_invocation_ids(
            tokenizer, [{"role": "user", "content": "x"}], enable_thinking=False
        )
    elif kind == "sr":
        extra["cross_stream_rank"] = 8
        # Thinking off, as decisions read: 4.2's thinking branch ends the prompt in
        # a bare newline, which is not a single round-tripping token.
        extra["last_context_token"] = sr_activation_anchor(
            tokenizer, [{"role": "user", "content": "x"}], enable_thinking=False
        )
    path = synthesize_adapter(
        root / "decision",
        dims_from_config(AutoConfig.from_pretrained(base)),
        base_weight_norms(base),
        PROFILES["weak"],
        seed=3,
        # SR takes K/V from the base stream, so it adapts no k/v projection.
        modules=attn_modules(with_kv=kind != "sr"),
        **extra,
    )
    io = {"name": "decision", "model": None, "response_format": None}
    if block is not None:
        io["decision"] = block
    (path / "io.yaml").write_text(yaml.safe_dump(io, sort_keys=False))
    return path


def _compose(base: str, adapter: Path, out: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "granite_switch.composer.compose_granite_switch",
            "--base-model",
            base,
            "--adapters",
            str(adapter),
            "--output",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module", params=[(b, k) for b in BASES for k in KINDS], ids=str)
def composed(request, tmp_path_factory):
    """(kind, checkpoint dir) composed with ``--decision-adapter decision``."""
    base_name, kind = request.param
    base = model_sources.base_model(model_sources.FAKE, base_name)
    root = tmp_path_factory.mktemp(f"decision-{base_name}-{kind}")
    adapter = _adapter(root, base, kind, _block(kind))
    out = root / "out"
    tech = ["--technology", "lora"] if kind == "lora" else []
    r = _compose(base, adapter, out, *tech, "--decision-adapter", "decision")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return kind, out


def _tokenizer_and_config(out: Path):
    from transformers import AutoTokenizer

    from granite_switch.config import GraniteSwitchConfig

    return AutoTokenizer.from_pretrained(str(out)), GraniteSwitchConfig.from_pretrained(
        out
    )


EXPECTED_ACTIVATION = {
    "alora": "marker",
    "alora_header": "generation",
    "lora": "start",
    "sr": "generation",
}


def test_config_records_the_spec_with_the_adapters_own_activation(composed):
    kind, out = composed
    config = json.loads((out / "config.json").read_text())
    assert config["decision_adapter"] == "decision"
    spec = config["decision_specs"]["decision"]
    assert spec["activation"] == EXPECTED_ACTIVATION[kind]
    assert spec["invocation_marker"] == (MARKER if kind == "alora" else "")
    assert spec["question_types"]["noul"]["order"] == ["no", "yes"]
    io = yaml.safe_load((out / "io_configs/decision/io.yaml").read_text())
    assert "decision" in io


def test_strategy_reads_the_composed_checkpoint(composed, monkeypatch):
    """The real strategy over the real composed tokenizer: one control token per
    read, in the question turn for aLoRA, the prompt tail, the noul mapping."""
    kind, out = composed
    tok, config = _tokenizer_and_config(out)
    label_a = tok("A", add_special_tokens=False)["input_ids"][0]
    modules = install_fake_vllm(monkeypatch, tok, winners=[label_a])
    strategy = build_strategy(load_strategy_module(), config)

    (read,) = run_read(
        strategy, [question("urgent", "noul", [("yes", None), ("no", None)])]
    )
    assert read.probs[1] > 0.9  # A = "no" (false first) maps back to option 1

    (engine_input,) = modules[LABEL_READS].seen
    ids = engine_input["prompt_ids"]
    control = tok.convert_tokens_to_ids("<|decision|>")
    assert ids.count(control) == 1
    tail = strategy.tails["decision"]
    assert ids[len(ids) - len(tail) :] == tail
    if kind == "alora":
        conversation = tok.apply_chat_template(
            engine_input["messages"], tokenize=False, adapter_name="decision"
        )
        assert "<|decision|>" in conversation  # inside the question turn
    if kind in ("sr", "alora_header"):
        # In the generation prompt. The SR token replaces an anchor in the generation prompt (on 4.2 an earlier
        # token common to every thinking branch, not always the last one).
        conversation = tok.apply_chat_template(
            engine_input["messages"], tokenize=False, adapter_name="decision"
        )
        # The header aLoRA's token closes the conversation (right before the header).
        tail_only = conversation.rstrip().endswith("<|decision|>")
        assert tail_only or "<|decision|>" not in conversation


def test_composed_template_refuses_a_wrong_marker(composed):
    kind, out = composed
    if kind != "alora":
        pytest.skip("only marker activation depends on the marker")
    tok, config = _tokenizer_and_config(out)
    wrong = dict(config.decision_specs["decision"], invocation_marker="### OTHER")
    with pytest.raises(DecisionSpecError, match="inside the question turn"):
        check_activation(tok, "decision", wrong)


def test_marker_that_disagrees_with_the_adapter_fails_the_compose(tmp_path):
    base = model_sources.base_model(model_sources.FAKE, BASES[0])
    adapter = _adapter(
        tmp_path, base, "alora", _block("alora", invocation_marker="### OTHER")
    )
    r = _compose(base, adapter, tmp_path / "out", "--decision-adapter", "decision")
    assert r.returncode != 0
    assert "differs from the adapter's aLoRA invocation text" in r.stderr


def test_decision_adapter_without_a_block_fails_the_compose(tmp_path):
    base = model_sources.base_model(model_sources.FAKE, BASES[0])
    adapter = _adapter(tmp_path, base, "alora", None)
    r = _compose(base, adapter, tmp_path / "out", "--decision-adapter", "decision")
    assert r.returncode != 0
    assert "has no 'decision:' block" in r.stderr


def _unstamped_copy(composed, tmp_path, block) -> Path:
    """The composed checkpoint as an older compose would have left it."""
    _, out = composed
    ckpt = tmp_path / "ckpt"
    shutil.copytree(out, ckpt)
    config = json.loads((ckpt / "config.json").read_text())
    for key in ("decision_adapter", "decision_specs"):
        config.pop(key)
    (ckpt / "config.json").write_text(json.dumps(config))
    io = {"name": "decision", "decision": block}
    (ckpt / "io_configs/decision/io.yaml").write_text(yaml.safe_dump(io))
    return ckpt


def test_stamp_records_the_spec_in_an_existing_checkpoint(composed, tmp_path):
    from granite_switch.config import GraniteSwitchConfig

    kind, _ = composed
    # A stamp has no source adapter: an aLoRA names its marker, or declares that its
    # invocation is the assistant header.
    extra = {
        "alora": {"invocation_marker": MARKER},
        "alora_header": {"activation": "generation"},
    }.get(kind, {})
    block = _block(kind, **extra)
    block.update(extra)
    ckpt = _unstamped_copy(composed, tmp_path, block)
    assert stamp_main(["--model", str(ckpt), "--decision-adapter", "decision"]) == 0
    config = GraniteSwitchConfig.from_pretrained(ckpt)
    assert config.decision_adapter == "decision"
    assert config.decision_specs["decision"]["activation"] == EXPECTED_ACTIVATION[kind]


def test_stamp_refuses_a_marker_the_template_does_not_activate_on(composed, tmp_path):
    kind, _ = composed
    if kind != "alora":
        pytest.skip("only marker activation depends on the marker")
    ckpt = _unstamped_copy(
        composed, tmp_path, _block(kind, invocation_marker="### OTHER")
    )
    before = (ckpt / "config.json").read_text()
    assert stamp_main(["--model", str(ckpt), "--decision-adapter", "decision"]) == 1
    assert (ckpt / "config.json").read_text() == before


def test_stamp_of_an_alora_needs_its_marker(composed, tmp_path):
    kind, _ = composed
    if kind != "alora":
        pytest.skip("aLoRA only")
    ckpt = _unstamped_copy(composed, tmp_path, _block(kind))
    assert stamp_main(["--model", str(ckpt), "--decision-adapter", "decision"]) == 1


def test_stamp_refuses_an_unknown_adapter(composed, tmp_path):
    ckpt = _unstamped_copy(composed, tmp_path, _block("lora"))
    assert stamp_main(["--model", str(ckpt), "--decision-adapter", "nope"]) == 1
