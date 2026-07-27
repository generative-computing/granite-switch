# SPDX-License-Identifier: Apache-2.0
"""Adapters + audio: the RAG answerability adapter judging a query against a
document delivered as audio (issue #47).

Covers the "adapters + audio" box. Where test_audio_serving_smoke.py only proves
an adapter control token doesn't crash the audio path, this checks the switch
reaches the *correct* verdict when the RAG document arrives as speech.

Document-as-audio: ``documents=[{"text": "<|audio|>"}]`` — the Granite template
renders each document with ``doc | tojson``, so the marker lands in the
``<documents>`` block where the ASR processor splices the transcript. The
``<|answerability|>`` control token is inserted by the same template (aLoRA
fallback path). Ground truth is one committed speech clip (fixtures/, generated
with SpeechT5 — MIT) driving both classes; the questions key on content that
transcribes cleanly, so this tests the decision, not transcription accuracy
(WER).

Markers: slow + requires_model + gpu (opt-in via -m).
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.gpu]

if importlib.util.find_spec("vllm") is None:
    pytest.skip("requires vLLM installed", allow_module_level=True)


_DEFAULT_BASE_MODEL_PAIRS = [
    ("ibm-granite/granite-4.1-3b", "ibm-granite/granitelib-rag-r1.0"),
]


def _load_experimental_pairs():
    """Extra (base, adapter) pairs from GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS.

    JSON array of {"base": str, "adapter": str}; mirrors the other E2E files.
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

COMPOSE_TIMEOUT_S = 1800  # matches the other E2E compose fixtures
_AUDIO_MARKER = "<|audio|>"

# Committed speech clip; spoken sentence is "The Eiffel Tower is located in the
# city of Paris in France." The location words transcribe cleanly (the proper
# noun does not, hence the questions key on the location, not the name).
_AUDIO_FIXTURE = Path(__file__).parent / "fixtures" / "eiffel_tower_paris.wav"
_ANSWERABILITY_CASES = [
    ("In which city is the tower located?", "answerable"),
    ("What is the boiling point of water?", "unanswerable"),
]


@pytest.fixture(
    scope="module",
    params=BASE_MODEL_PAIRS,
    ids=lambda p: p[0].rsplit("/", 1)[-1],
)
def audio_rag_checkpoint(request, tmp_path_factory):
    """Compose one audio-enabled RAG checkpoint per (base, adapter) pair."""
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
def served(audio_rag_checkpoint):
    """Boot vLLM once for the checkpoint and share it across the cases."""
    import gc

    import torch

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    llm = LLM(
        model=str(audio_rag_checkpoint["save_dir"]),
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    try:
        yield {
            "llm": llm,
            "config": llm.llm_engine.model_config.hf_config,
            "tokenizer": llm.get_tokenizer(),
        }
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _answerability_adapter_name(config):
    # Discovery derives the name from the library layout — match rather than
    # hard-code, and skip loudly if this checkpoint has no answerability adapter.
    names = list(getattr(config, "adapter_names", None) or [])
    for name in names:
        if "answerab" in name.lower():
            return name
    pytest.skip(f"no answerability adapter in composed checkpoint (adapters={names})")


@pytest.fixture(scope="module")
def speech_clip():
    """The committed 16 kHz mono speech clip as (waveform, sample_rate)."""
    import soundfile as sf

    waveform, sr = sf.read(_AUDIO_FIXTURE, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return {"waveform": waveform, "sr": sr}


def _build_answerability_prompt(tokenizer, adapter_name, question):
    # documents=[{"text": "<|audio|>"}] places one audio marker in the template's
    # <documents> block; adapter_name arms the answerability control-token insert.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        documents=[{"text": _AUDIO_MARKER}],
        add_generation_prompt=True,
        adapter_name=adapter_name,
        tokenize=False,
    )


def _parse_label(text):
    # Adapter emits the bare enum ("answerable"/"unanswerable"), maybe quoted.
    # Match the longer label first so "answerable" doesn't shadow "unanswerable".
    norm = text.strip().strip('"').strip().lower()
    if "unanswerable" in norm:
        return "unanswerable"
    if "answerable" in norm:
        return "answerable"
    return norm  # unrecognized — surfaced by the assertion


@pytest.mark.parametrize(
    "question,expected",
    _ANSWERABILITY_CASES,
    ids=[c[1] for c in _ANSWERABILITY_CASES],
)
def test_answerability_over_audio_document(served, speech_clip, question, expected):
    """Answerability adapter reaches the correct verdict on an audio document."""
    from vllm import SamplingParams

    adapter_name = _answerability_adapter_name(served["config"])
    prompt = _build_answerability_prompt(served["tokenizer"], adapter_name, question)

    outputs = served["llm"].generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [(speech_clip["waveform"], speech_clip["sr"])]
            },
        },
        SamplingParams(max_tokens=8, temperature=0.0),
    )

    assert len(outputs) == 1
    raw = outputs[0].outputs[0].text
    label = _parse_label(raw)
    assert label == expected, (
        f"answerability mismatch for {question!r}: got {label!r} "
        f"(raw={raw!r}), expected {expected!r}"
    )
