# SPDX-License-Identifier: Apache-2.0
"""End-to-end vLLM serving smoke for the audio cascade (issue #47).

Drives text + adapter + audio x1/x2/x3 through one live engine — the serve-time
path the low-level tests bypass (CUDA graphs, the ASR processor inside vLLM's
EngineCore subprocess, the multi-clip splice, generation).

Composed with ``--bundle-asr`` (which implies ``--enable-audio``) from a tiny
random CTC model built locally (``tests/shared/tiny_asr.py``), so the ASR is
served from the checkpoint's ``asr/`` folder with no ASR download. This also
covers the airgapped path (issue #158) and proves vLLM does not take
``asr/*.safetensors`` for LLM weights. The real default ASR is exercised by
``test_asr_ctc_default_gpu.py``.

Deliberately does NOT assert transcript content: WER and adapter-routing
correctness are separate boxes, covered by the eval harness and the MultiSwitch
serving e2e tests. Synthetic tones keep the test asset-free.

Opt in explicitly: `pytest -m "slow and requires_model and gpu"`.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.audio,
    pytest.mark.slow,
    pytest.mark.gpu,
]

if importlib.util.find_spec("vllm") is None:
    pytest.skip("requires vLLM installed", allow_module_level=True)

from tests.shared import model_sources

# E2E model matrix: (base, adapter library). PR runs serve a fake granite-4.0-micro
# (tests/shared/model_sources.py); the release tier serves the real 4.0-micro and
# adds the real 4.1-3b as a second base.
_ADAPTER_LIBRARY = "ibm-granite/granitelib-core-r1.0"


def _pair(source, base_name):
    return (
        model_sources.base_model(source, base_name),
        model_sources.adapter_library(source, _ADAPTER_LIBRARY, base_name),
    )


_DEFAULT_BASE_MODEL_PAIRS = [
    pytest.param(("fake", "granite-4.0-micro"), id="granite-4.0-micro-fake"),
    pytest.param(
        ("real", "granite-4.0-micro"),
        marks=pytest.mark.release,
        id="granite-4.0-micro-real",
    ),
    pytest.param(
        ("real", "granite-4.1-3b"),
        marks=pytest.mark.release,
        id="granite-4.1-3b-real",
    ),
]


def _load_experimental_pairs():
    """Local pairings from GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS.

    JSON array of {"base": str, "adapter": str}, HF ids or local paths. The
    mechanism is committed; the values are not.
    """
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
    return [
        pytest.param(
            ("path", (p["base"], p["adapter"])), id=p["base"].rsplit("/", 1)[-1]
        )
        for p in entries
    ]


BASE_MODEL_PAIRS = _DEFAULT_BASE_MODEL_PAIRS + _load_experimental_pairs()

COMPOSE_TIMEOUT_S = 1800  # 30 min — generous for a real compose
_TARGET_SR = 16_000


@pytest.fixture(
    scope="module",
    params=BASE_MODEL_PAIRS,
)
def audio_checkpoint(request, tmp_path_factory):
    """Compose one audio-enabled checkpoint per (base, adapter) pair.

    Goes through the compose CLI, never a hand-assembled config (CLAUDE.md
    gotcha #5). Module scope amortizes the download across the pair's cases.
    """
    import subprocess
    import sys

    from tests.shared.tiny_asr import build_tiny_asr

    kind, spec = request.param
    base_model, adapter_library = spec if kind == "path" else _pair(kind, spec)
    save_dir = tmp_path_factory.mktemp(Path(base_model).name) / "model"
    tiny_asr = build_tiny_asr(tmp_path_factory.mktemp("tiny-asr") / "asr")

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        base_model,
        "--adapters",
        adapter_library,
        "--bundle-asr",
        "--asr-model",
        str(tiny_asr),
        "--output",
        str(save_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=COMPOSE_TIMEOUT_S
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"compose failed for base={base_model} adapter={adapter_library}\n"
            f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
        )
    return {"base_model": base_model, "save_dir": save_dir}


@pytest.fixture(scope="module")
def served(audio_checkpoint):
    """Boot vLLM once for the checkpoint and share it across smoke cases.

    Tokenizer init stays ON (unlike the argmax-equivalence test): the ASR
    processor needs it to encode the prompt and the transcript.
    """
    from tests.integration.conftest import shutdown_llm

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    llm = LLM(
        model=str(audio_checkpoint["save_dir"]),
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        enforce_eager=True,  # smoke: skip CUDA-graph capture for a faster boot
    )
    try:
        yield {"llm": llm, "config": llm.llm_engine.model_config.hf_config}
    finally:
        # Shut the EngineCore subprocess down (not just empty_cache) so it hands
        # the GPU back before the next parametrized checkpoint boots its engine;
        # under Exclusive_Process a leaked engine kills the next one.
        shutdown_llm(llm)


def _tone(seconds: float = 1.0, freq: float = 440.0):
    """Deterministic mono 16 kHz waveform (no speech fixture needed)."""
    import numpy as np

    t = np.arange(int(seconds * _TARGET_SR), dtype=np.float32) / _TARGET_SR
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _one_completion(outputs):
    """Assert a single RequestOutput carrying at least one generated token."""
    assert len(outputs) == 1
    completion = outputs[0].outputs[0]
    assert len(completion.token_ids) >= 1
    return completion


def test_asr_is_served_from_the_bundle(served):
    """The bundled ASR is named in the config the engine serves."""
    config = served["config"]
    assert config.asr_bundled_path == "asr"
    assert config.asr_model_id


def test_text_only_serving(served):
    """Baseline: a plain text request serves normally (backward-compat)."""
    from vllm import SamplingParams

    outputs = served["llm"].generate(
        "The capital of France is",
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    _one_completion(outputs)


def test_adapter_control_token_serving(served):
    """An adapter control token routes through the switch under serving.

    Routing correctness is the MultiSwitch serving e2e tests' job; the bar here
    is that the switch path runs in the live engine and still generates.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    config = served["config"]
    if not getattr(config, "adapter_token_ids", None):
        pytest.skip("composed checkpoint has no adapters")

    tokenizer = served["llm"].get_tokenizer()
    text_ids = tokenizer.encode("Summarize the document.", add_special_tokens=False)
    # LORA control tokens sit at the sequence start (CLAUDE.md gotcha #3).
    prompt = TokensPrompt(prompt_token_ids=[config.adapter_token_ids[0], *text_ids])

    outputs = served["llm"].generate(
        prompt, SamplingParams(max_tokens=8, temperature=0.0)
    )
    _one_completion(outputs)


@pytest.mark.parametrize("num_clips", [1, 2, 3])
def test_audio_clip_serving(served, num_clips):
    """Audio x1/x2/x3: N markers + N clips transcribe, splice, and generate.

    Content is not asserted; the bar is a completed, well-formed request.
    """
    from vllm import SamplingParams

    ceiling = int(getattr(served["config"], "asr_max_audio_clips", 32) or 32)
    if num_clips > ceiling:
        pytest.skip(f"checkpoint clip ceiling {ceiling} < {num_clips}")

    marker = "<|audio|>"
    prompt = {
        "prompt": marker * num_clips + " What was said?",
        "multi_modal_data": {
            "audio": [
                (_tone(freq=220.0 * (i + 1)), _TARGET_SR) for i in range(num_clips)
            ]
        },
    }
    outputs = served["llm"].generate(
        prompt, SamplingParams(max_tokens=8, temperature=0.0)
    )
    _one_completion(outputs)
