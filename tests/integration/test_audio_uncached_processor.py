# SPDX-License-Identifier: Apache-2.0
"""Audio serving with vLLM's multimodal processor cache DISABLED.

This is the configuration that actually exercises
``AudioMultiModalProcessor._hf_processor_applies_updates``. vLLM reaches the HF
processor two ways and only one of them consults that hook:

* cached path — hardcodes ``is_update_applied = False``, never asks
* uncached path — takes ``is_update_applied`` from the hook

Every other audio test runs vLLM's defaults, where the cache is on, so they pass
whether or not the hook is overridden. With the cache off and the base hook's
``True``, vLLM skips applying our ``PromptReplacement`` and then reports the item
as missing::

    RuntimeError: Expected there to be 1 audio prompt placeholders corresponding
    to 1 audio items, but instead found 0 prompt placeholders!

Startup profiling passes a *string* prompt, so on the uncached path the engine
fails before it can serve anything — booting at all is a large part of the guard.

Assertions here are structural (request completes, marker was replaced), never
transcript content: ASR output is not deterministic enough to assert on.

Opt in explicitly: `pytest -m "slow and requires_model and gpu"`.
"""

import importlib.util
import os

import pytest

pytestmark = [
    pytest.mark.audio,
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
]

if importlib.util.find_spec("vllm") is None:
    pytest.skip("requires vLLM installed", allow_module_level=True)


# One small model: this covers a *configuration* dimension, not a model matrix,
# and each engine boot here is expensive.
_BASE_MODEL = "ibm-granite/granite-4.0-micro"
_ADAPTER_LIBRARY = "ibm-granite/granitelib-core-r1.0"

COMPOSE_TIMEOUT_S = 1800  # matches the sibling E2E fixtures
_TARGET_SR = 16_000
_AUDIO_MARKER = "<|audio|>"


def _cache_disabling_kwargs():
    """LLM kwargs that turn off the multimodal processor cache.

    The knob was renamed across the vLLM range this project supports (0.19.x and
    0.20.x are both allowed in pyproject): older builds expose
    ``disable_mm_preprocessor_cache``, newer ones ``mm_processor_cache_gb``.
    Returns an empty dict when neither exists, so the caller can skip rather than
    silently exercise the cached path.
    """
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs

    names = {f.name for f in dataclasses.fields(EngineArgs)}
    if "mm_processor_cache_gb" in names:
        return {"mm_processor_cache_gb": 0}
    if "disable_mm_preprocessor_cache" in names:
        return {"disable_mm_preprocessor_cache": True}
    return {}


def _cache_disabled_state(llm):
    """Whether the running engine really has the processor cache off.

    ``True``/``False`` when determinable, ``None`` when this vLLM exposes neither
    setting where we look. The field moved: it lives on ``MultiModalConfig``
    (nested under ``model_config.multimodal_config``) in newer vLLM, so checking
    only ``model_config`` silently answers "unknown" and the whole module would
    look green while running the cached path — testing nothing.
    """
    model_config = llm.llm_engine.model_config
    for obj in (getattr(model_config, "multimodal_config", None), model_config):
        if obj is None:
            continue
        if hasattr(obj, "mm_processor_cache_gb"):
            return obj.mm_processor_cache_gb == 0
        if hasattr(obj, "disable_mm_preprocessor_cache"):
            return bool(obj.disable_mm_preprocessor_cache)
    return None


@pytest.fixture(scope="module")
def audio_checkpoint(tmp_path_factory):
    """Compose one audio-enabled checkpoint through the compose CLI."""
    import subprocess
    import sys

    save_dir = tmp_path_factory.mktemp(_BASE_MODEL.rsplit("/", 1)[-1]) / "model"
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        _BASE_MODEL,
        "--adapters",
        _ADAPTER_LIBRARY,
        "--enable-audio",
        "--output",
        str(save_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=COMPOSE_TIMEOUT_S
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"compose failed for base={_BASE_MODEL} adapter={_ADAPTER_LIBRARY}\n"
            f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
        )
    return save_dir


@pytest.fixture(scope="module")
def served_uncached(audio_checkpoint):
    """Boot vLLM with the processor cache off, and prove it is off.

    Reaching the ``yield`` is itself the startup-profiling guard: profiling runs a
    dummy audio item through full processing with a string prompt, exactly the
    combination that fails when the hook is left at its default.

    The cache check lives here rather than in a test so it *gates* every case. As
    a separate test it would only skip itself, leaving the rest of the module
    green while silently running the cached path.
    """
    import gc

    import torch

    cache_kwargs = _cache_disabling_kwargs()
    if not cache_kwargs:
        pytest.skip(
            "installed vLLM exposes neither mm_processor_cache_gb nor "
            "disable_mm_preprocessor_cache; cannot disable the processor cache"
        )

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    llm = LLM(
        model=str(audio_checkpoint),
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        enforce_eager=True,  # boot speed; orthogonal to the cache setting
        **cache_kwargs,
    )
    try:
        state = _cache_disabled_state(llm)
        if state is False:
            pytest.fail(
                f"{cache_kwargs!r} was accepted but the processor cache is still "
                f"enabled; this module would exercise the cached path instead"
            )
        if state is None:
            pytest.skip(
                "cannot confirm the processor-cache setting on this vLLM, so this "
                "module cannot establish that it is testing the uncached path"
            )
        yield {
            "llm": llm,
            "config": llm.llm_engine.model_config.hf_config,
            "tokenizer": llm.get_tokenizer(),
        }
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def marker_id(served_uncached):
    """The ``<|audio|>`` token id, verified to be a real single token.

    ``convert_tokens_to_ids`` returns the *unk* id for an unregistered token
    rather than None, so a checkpoint composed without audio would hand back a
    plausible-looking id and the marker assertions would be meaningless.
    """
    tokenizer = served_uncached["tokenizer"]
    token_id = tokenizer.convert_tokens_to_ids(_AUDIO_MARKER)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    assert token_id is not None and token_id >= 0, (
        f"{_AUDIO_MARKER} is not in the tokenizer"
    )
    assert token_id != unk_id, (
        f"{_AUDIO_MARKER} resolved to the unk id ({unk_id}) — the checkpoint was "
        f"not composed with --enable-audio"
    )
    encoded = tokenizer.encode(_AUDIO_MARKER, add_special_tokens=False)
    assert encoded == [token_id], (
        f"{_AUDIO_MARKER} does not encode to exactly one token: {encoded}"
    )
    return token_id


def _tone(seconds: float = 1.0, freq: float = 440.0):
    """Deterministic mono 16 kHz waveform (no speech fixture needed)."""
    import numpy as np

    t = np.arange(int(seconds * _TARGET_SR), dtype=np.float32) / _TARGET_SR
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float = 1.0):
    """A clip with no speech in it at all."""
    import numpy as np

    return np.zeros(int(seconds * _TARGET_SR), dtype=np.float32)


def _generate_with_audio(llm, waveform):
    from vllm import SamplingParams

    return llm.generate(
        {
            "prompt": f"{_AUDIO_MARKER} What was said?",
            "multi_modal_data": {"audio": [(waveform, _TARGET_SR)]},
        },
        SamplingParams(max_tokens=8, temperature=0.0),
    )


def _assert_marker_replaced(outputs, marker_id):
    """The request completed and the marker is gone from the final prompt.

    ``RequestOutput.prompt_token_ids`` is the post-processing prompt (vLLM builds
    the engine request from the processed inputs), so an absent marker means the
    prompt replacement really was applied. Length is not asserted: a clip with no
    speech legitimately collapses to the one-token fallback.
    """
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) >= 1

    prompt_ids = list(outputs[0].prompt_token_ids or [])
    assert prompt_ids, "engine returned no prompt_token_ids to inspect"
    assert marker_id not in prompt_ids, (
        f"marker id {marker_id} survived into the final prompt — the audio "
        f"placeholder was not applied"
    )


def test_audio_request_splices_transcript_uncached(served_uncached, marker_id):
    """An audio request completes and the marker is replaced, cache off.

    The direct positive signal for the hook override: with the base hook the
    replacement is skipped and vLLM raises before returning anything.
    """
    outputs = _generate_with_audio(served_uncached["llm"], _tone())
    _assert_marker_replaced(outputs, marker_id)


def test_silent_clip_uncached(served_uncached, marker_id):
    """Silence still yields a usable placeholder on the uncached path.

    The empty-transcript fallback fires on both processor paths, so it is worth
    pinning here too: a clip transcribing to "" must not collapse to a
    zero-length placeholder.
    """
    outputs = _generate_with_audio(served_uncached["llm"], _silence())
    _assert_marker_replaced(outputs, marker_id)


def test_text_only_request_uncached(served_uncached):
    """Disabling the cache must not disturb ordinary text requests."""
    from vllm import SamplingParams

    outputs = served_uncached["llm"].generate(
        "The capital of France is",
        SamplingParams(max_tokens=8, temperature=0.0),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) >= 1
