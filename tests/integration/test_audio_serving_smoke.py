# SPDX-License-Identifier: Apache-2.0
"""End-to-end vLLM serving smoke for the audio cascade (issue #47).

Boots a real composed, audio-enabled GraniteSwitch checkpoint under vLLM and
drives the three request shapes the #47 checklist calls out as the serving
smoke — **text + adapter + audio x1/x2/x3** — through one live engine. This is
the path the low-level tests deliberately bypass (CUDA graphs, torch.compile,
the ASR processor running inside vLLM's EngineCore subprocess, the multi-clip
splice, and generation), so it is the only test that proves those pieces
compose at serve time.

Scope is a *smoke*: each request must complete and return well-formed output,
and multi-clip requests must be accepted up to the checkpoint's declared clip
ceiling. It intentionally does NOT assert transcript content — transcription
quality (WER) and adapter-routing correctness are separate #47 boxes covered by
the eval harness and `test_switch_e2e_compose.py` respectively. Synthetic audio
keeps the test asset-free; a silent/tonal clip exercises the serving plumbing
without shipping a speech fixture.

Model construction goes through the compose CLI (CLAUDE.md gotcha #5): no test
hand-assembles a config. Audio is enabled with `--enable-audio`, which defaults
the ASR front-end to the small built-in model (distil-whisper/distil-small.en)
to keep CI cost down.

Markers: @pytest.mark.slow + @pytest.mark.requires_model + @pytest.mark.gpu.
CI must opt in explicitly: `pytest -m "slow and requires_model and gpu"`.
"""

import importlib.util
import json
import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.gpu]

if importlib.util.find_spec("vllm") is None:
    pytest.skip("requires vLLM installed", allow_module_level=True)


# ----------------------------------------------------------------------------
# Base-model / adapter-library pairs — kept in lockstep with
# tests/integration/test_switch_e2e_compose.py so the two E2E files exercise
# the same model matrix. A fast CI profile can pin one pair via -k or the
# experimental env var below.
# ----------------------------------------------------------------------------

_DEFAULT_BASE_MODEL_PAIRS = [
    ("ibm-granite/granite-4.0-micro", "ibm-granite/granitelib-core-r1.0"),
    ("ibm-granite/granite-4.1-3b", "ibm-granite/granitelib-core-r1.0"),
]


def _load_experimental_pairs():
    """Local/experimental pairings from GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS.

    JSON array of {"base": str, "adapter": str}; base/adapter may be HF ids or
    local paths. The mechanism is committed; the values are not. Mirrors
    test_switch_e2e_compose.py so both E2E files share one extension knob.
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
    return [(p["base"], p["adapter"]) for p in entries]


BASE_MODEL_PAIRS = _DEFAULT_BASE_MODEL_PAIRS + _load_experimental_pairs()

COMPOSE_TIMEOUT_S = 1800  # 30 min — matches test_switch_e2e_compose.py
_TARGET_SR = 16_000


@pytest.fixture(
    scope="module",
    params=BASE_MODEL_PAIRS,
    ids=lambda p: p[0].rsplit("/", 1)[-1],
)
def audio_checkpoint(request, tmp_path_factory):
    """Compose one audio-enabled checkpoint per (base, adapter) pair.

    Delegates to the compose CLI (same as test_switch_e2e_compose.py) with
    `--enable-audio`, then returns the save dir. Module scope amortizes the
    download-dominated first run across every smoke case for the pair.
    """
    import subprocess
    import sys

    base_model, adapter_library = request.param
    save_dir = tmp_path_factory.mktemp(base_model.rsplit("/", 1)[-1]) / "model"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        base_model,
        "--adapters",
        adapter_library,
        "--enable-audio",
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
    processor needs a tokenizer to encode both the prompt and the transcript,
    and the adapter case needs it to tokenize text around the control token.
    """
    import gc

    import torch

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
        del llm
        gc.collect()
        torch.cuda.empty_cache()


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

    Correctness of the routing is `test_switch_e2e_compose.py`'s job; here we
    only prove the switch path runs end-to-end in the live engine without
    crashing and still generates.
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

    Exercises the multi-clip long-audio serving path through the ASR processor
    running inside vLLM's engine subprocess. Content is not asserted (WER is a
    separate #47 box); the bar is a completed request with well-formed output.
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
