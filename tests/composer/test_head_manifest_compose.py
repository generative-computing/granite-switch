# SPDX-License-Identifier: Apache-2.0
"""Head streams declared in a compose manifest, composed through the CLI.

The other head tests compose through the Python API (``streams=``). This one
runs the compose CLI on a YAML manifest that uses every head key:

* ``tox``: a probe with no ``type`` key (a head's marker always goes last);
* ``tox_lo``: the same kind of probe on the LoRA adapter (``adapter:``);
* ``nudge``: the example plugin type, in merge mode, with ``settings``.

The plugin is visible only through its ``granite_switch.streams`` entry point
(a dist-info on PYTHONPATH), so the compose process must discover it.

A tiny random base carries the real granite-4.1-3b tokenizer, which the chat
template needs, so the compose takes seconds. Requires the tokenizer files.
"""

import json
import os
import subprocess
import sys

import pytest
import torch
import yaml
from safetensors.torch import save_file

pytestmark = pytest.mark.requires_model

TOKENIZER = "ibm-granite/granite-4.1-3b"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIDTH = 8


def _io_yaml(path, name):
    (path / "io.yaml").write_text(f"name: {name}\nmodel: ~\n")


def _plugin_site(root):
    info = root / "site" / "gs_manifest_test_plugins-0.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: gs-manifest-test-plugins\nVersion: 0.0.0\n"
    )
    (info / "entry_points.txt").write_text(
        "[granite_switch.streams]\n"
        "mlp_head = tests.shared.stream_plugins.mlp_head:MLPHeadType\n"
    )
    return str(root / "site")


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    from transformers import AutoConfig, AutoTokenizer

    from tests.shared.base_models import dims_from_config, write_tiny_granite_base
    from tests.shared.synthetic_adapters import (
        PROFILES,
        attn_modules,
        base_weight_norms,
        mlp_modules,
        synthesize_adapter,
    )

    root = tmp_path_factory.mktemp("head_manifest")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    base = write_tiny_granite_base(root / "base", vocab_size=len(tok))
    tok.save_pretrained(base)
    hidden = AutoConfig.from_pretrained(str(base)).hidden_size

    dims = dims_from_config(AutoConfig.from_pretrained(str(base)))
    lora = root / "lo" / "lo" / "tiny" / "lora"
    synthesize_adapter(
        lora,
        dims,
        base_weight_norms(base),
        PROFILES["light"],
        seed=7,
        modules=attn_modules() + mlp_modules(dims),
    )
    _io_yaml(lora, "lo")

    g = torch.Generator().manual_seed(0)
    probe = {"weight": torch.randn(2, hidden, generator=g), "bias": torch.zeros(2)}
    heads = {}
    for name in ("tox", "tox_lo", "nudge"):
        d = root / name / name / "tiny" / "lora"
        d.mkdir(parents=True)
        _io_yaml(d, name)
        heads[name] = str(d)
    for name in ("tox", "tox_lo"):
        save_file(probe, os.path.join(heads[name], "probe_head.safetensors"))
    save_file(
        {
            "w1": torch.randn(WIDTH, hidden, generator=g),
            "w2": torch.randn(2, WIDTH, generator=g),
        },
        os.path.join(heads["nudge"], "mlp_head.safetensors"),
    )
    manifest = {
        "lo": {"path": str(lora), "type": "lora"},
        "tox": {"path": heads["tox"], "kind": "probe", "labels": ["yes", "no"]},
        "tox_lo": {
            "path": heads["tox_lo"],
            "kind": "probe",
            "labels": ["yes", "no"],
            "adapter": "lo",
        },
        "nudge": {
            "path": heads["nudge"],
            "kind": "mlp_head",
            "labels": ["yes", "no"],
            "output": {"mode": "merge", "weight": 0.5},
            "settings": {"width": WIDTH},
        },
    }
    (root / "adapters.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    out = root / "model"
    # Extend, don't replace, the caller's PYTHONPATH: it may name the source
    # tree under test.
    paths = [_REPO, _plugin_site(root), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "granite_switch.composer.compose_granite_switch",
            "--base-model",
            str(base),
            "--adapters",
            str(root / "adapters.yaml"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"compose failed (exit {proc.returncode})\nSTDOUT:\n{proc.stdout[-3000:]}\n"
        f"STDERR:\n{proc.stderr[-3000:]}"
    )
    return out, proc.stdout, tok


def test_compose_writes_every_head_entry(composed):
    out, stdout, tok = composed
    config = json.loads((out / "config.json").read_text())
    assert config["adapter_names"] == ["lo", "tox", "tox_lo", "nudge"]
    labels = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "no")]
    assert config["streams"] == [
        {"type": "lora"},
        {
            "type": "probe",
            "read_layer": None,
            "head_row": 0,
            "output": {"mode": "select", "token_ids": labels},
        },
        {
            "type": "probe",
            "read_layer": None,
            "head_row": 1,
            "output": {"mode": "select", "token_ids": labels},
            "adapter": "lo",
        },
        {
            "type": "mlp_head",
            "read_layer": None,
            "head_row": 0,
            "output": {"mode": "merge", "weight": 0.5, "token_ids": labels},
            "settings": {"width": WIDTH},
        },
    ]
    assert "Head streams:" in stdout
    assert "tox_lo (probe on lo)" in stdout
    assert "nudge (mlp_head) → merge" in stdout


def test_template_places_each_head(composed):
    out, _stdout, _tok = composed
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out))
    msgs = [{"role": "user", "content": "Is this rude?"}]

    def ids(**kw):
        return list(
            tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True, **kw
            )["input_ids"]
        )

    tid = {n: tok.convert_tokens_to_ids(f"<|{n}|>") for n in ("lo", "tox", "tox_lo")}
    plain = list(
        tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, return_dict=True
        )["input_ids"]
    )
    assert ids(adapter_name="tox") == [*plain, tid["tox"]]
    pair = ids(adapter_name="tox_lo")
    # The LoRA's own token where it goes alone (the start), the marker last.
    assert pair[0] == tid["lo"] and pair[-1] == tid["tox_lo"]
    assert pair.count(tid["lo"]) == 1


def test_pair_marker_runs_its_adapter(composed):
    out, _stdout, _tok = composed
    from transformers import AutoTokenizer

    import tests.shared.stream_plugins.mlp_head  # noqa: F401 -- registers the type
    from granite_switch.hf import GraniteSwitchForCausalLM

    tok = AutoTokenizer.from_pretrained(str(out))
    model = GraniteSwitchForCausalLM.from_pretrained(
        str(out), dtype=torch.float32
    ).eval()
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Is this rude?"}],
        adapter_name="tox_lo",
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    lora_rows = model.model._last_lora_indices[0].tolist()
    # The LoRA (slot 1) runs from its token, at the start, through the pair's marker.
    assert set(lora_rows) == {1}
    assert model.model._last_head_indices[0, -1].item() == 3
