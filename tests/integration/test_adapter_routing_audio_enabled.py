# SPDX-License-Identifier: Apache-2.0
"""Adapters + audio (issue #47): every default adapter routes to its own index
on an audio-enabled checkpoint.

Composes the default adapter set (RAG + Core + Guardian) with --enable-audio and
sweeps every adapter control token that resolved, asserting the switch maps
``adapter_token_ids[i]`` to index ``i + 1`` and leaves pre-control positions on
the base (``0``) — i.e. enabling audio does not perturb the token->index map.
For granite-4.1-3b that is 12 adapters: context_relevance ships no 4.1-3b flavor,
so 12 of the 13 defined adapters resolve. Complements the single-adapter sweep in
test_switch_e2e_compose and the serve-time answerability check in
test_answerability_over_audio.

Markers: slow + requires_model + gpu (opt-in via -m).
"""

import importlib.util
import json
import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.gpu]

if importlib.util.find_spec("granite_switch.hf") is None:
    pytest.skip("requires the HF backend ([hf] extra)", allow_module_level=True)


# The default adapter set is the union of the three granitelib libraries (the
# "12 adapters"): RAG + Core + Guardian.
_DEFAULT_ADAPTER_LIBRARIES = [
    "ibm-granite/granitelib-rag-r1.0",
    "ibm-granite/granitelib-core-r1.0",
    "ibm-granite/granitelib-guardian-r1.0",
]
_DEFAULT_BASE_MODEL_PAIRS = [
    ("ibm-granite/granite-4.1-3b", _DEFAULT_ADAPTER_LIBRARIES),
]


def _load_experimental_pairs():
    raw = os.environ.get("GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS", "")
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS is not valid JSON: {e}\n"
            f'Expected format: \'[{{"base":"/path","adapter":"/path"}}, ...]\''
        )
    # adapter may be a single library or a list of libraries.
    pairs = []
    for p in entries:
        adapters = p["adapter"]
        pairs.append(
            (p["base"], adapters if isinstance(adapters, list) else [adapters])
        )
    return pairs


BASE_MODEL_PAIRS = _DEFAULT_BASE_MODEL_PAIRS + _load_experimental_pairs()

COMPOSE_TIMEOUT_S = 1800
_SEQ_LEN = 8
_CTRL_POS = 1
_FILLER_TOKENS = [791, 5679, 2766, 279, 893, 389, 813, 1450]


@pytest.fixture(
    scope="module",
    params=BASE_MODEL_PAIRS,
    ids=lambda p: p[0].rsplit("/", 1)[-1],
)
def audio_switch_model(request, tmp_path_factory):
    import subprocess
    import sys

    import torch

    from granite_switch.hf import GraniteSwitchForCausalLM

    base_model, adapter_libraries = request.param
    save_dir = tmp_path_factory.mktemp(base_model.rsplit("/", 1)[-1]) / "model"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        base_model,
        "--adapters",
        *adapter_libraries,
        "--enable-audio",
        "--output",
        str(save_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=COMPOSE_TIMEOUT_S
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"compose failed for base={base_model} adapters={adapter_libraries}\n"
            f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
        )

    # ignore_mismatched_sizes: the <|audio|> token bumps vocab_size one past the
    # switch's control_to_substitute_lut buffer, but that buffer is rebuilt from
    # config in SingleSwitch.__init__, so the freshly-built one is kept intact.
    model = (
        GraniteSwitchForCausalLM.from_pretrained(
            str(save_dir), dtype=torch.bfloat16, ignore_mismatched_sizes=True
        )
        .eval()
        .cuda()
    )
    return {"base_model": base_model, "model": model, "config": model.config}


def _adapter_indices_after_forward(model, control_token_id):
    import torch

    seq = [_FILLER_TOKENS[i % len(_FILLER_TOKENS)] for i in range(_SEQ_LEN)]
    seq[_CTRL_POS] = control_token_id
    with torch.no_grad():
        model(input_ids=torch.tensor([seq], device="cuda"))
    return model.model._last_adapter_indices[0]


def test_all_adapters_route_with_audio_enabled(audio_switch_model):
    """Each adapter control token routes to its own index on an audio checkpoint."""
    config = audio_switch_model["config"]
    base_model = audio_switch_model["base_model"]

    assert getattr(config, "asr_enabled", False) is True, (
        f"checkpoint is not audio-enabled (base_model={base_model})"
    )

    token_ids = list(getattr(config, "adapter_token_ids", None) or [])
    names = list(getattr(config, "adapter_names", None) or [])
    assert token_ids, f"composed checkpoint has no adapters (base_model={base_model})"
    print(f"\n  sweeping {len(token_ids)} adapters (base_model={base_model})")

    failures = []
    for i, token_id in enumerate(token_ids):
        expected = i + 1  # adapter_token_ids[i] activates adapter i+1; 0 = base
        name = names[i] if i < len(names) else f"adapter_{expected}"
        ai = _adapter_indices_after_forward(audio_switch_model["model"], token_id)

        ok = bool((ai[:_CTRL_POS] == 0).all()) and bool(
            (ai[_CTRL_POS:] == expected).all()
        )
        print(
            f"    [{'ok' if ok else 'FAIL'}] idx {expected:>2} {name!r}: {ai.tolist()}"
        )
        if not ok:
            failures.append((expected, name, token_id, ai.tolist()))

    assert not failures, (
        f"{len(failures)} adapter(s) mis-routed with audio enabled "
        f"(base_model={base_model}); expected pre-control=0, post-control=index:\n"
        + "\n".join(
            f"  idx {idx} {name!r} (token {tok}): {indices}"
            for idx, name, tok, indices in failures
        )
    )
