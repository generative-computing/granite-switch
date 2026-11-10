# SPDX-License-Identifier: Apache-2.0
"""The ASR hook gate must key on the base class's SHAPE, not on a name existing.

vLLM reshaped ``BaseMultiModalProcessor``'s HF-processor hooks three times, and
the eras are not ordered by whether a given attribute is present:

* 0.26-0.28 (A) -- ``_call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)``
* 0.29-0.30 (B) -- ``_call_hf_processor`` deleted
* 0.31+     (C) -- ``_call_hf_processor(hf_data, hf_kwargs)`` reintroduced, and
  now invoked *by* ``_apply_hf_processor_main(mm_items, hf_kwargs)``

So ``hasattr(base, "_call_hf_processor")`` is true for A and C and false for B --
it cannot tell A from C. Reading era C as "this is 0.26-0.28" sends its
two-argument call into era A's four-argument hook, where ``args[0]`` is the mm
data rather than the prompt and ``tokenizer.encode`` raises ``TypeError`` on
every audio request. That one is loud, and a single audio test catches it.

The quiet one is era C's ``"audios"`` -> ``"audio"`` rename. ``_get_hf_mm_inputs``
normalizes vLLM's plural modality key to the singular one HF processors take, and
it runs BEFORE our hook -- so reading the plural yields no clips, the hook hands
back an empty ``BatchFeature``, and nothing raises where the mistake was made.
The request fails much further on, as ``IndexError`` in ``_get_prompt_updates``:
a cumsum over an empty ``audio_num_tokens``, indexed for an audio item vLLM knows
exists. That is why this file asserts the routing OUTCOME -- which era each stub
resolves to, and which key the hook reads -- rather than that some hook got
called.

These tests stub the base class instead of importing vLLM: the discriminator is
the signature of the base hook, so a stub shaped like each released era is a
faithful input, and the test stays in the CPU tier where no vLLM is installed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from collections.abc import Mapping

import pytest

pytestmark = pytest.mark.audio


def _load_processor_module():
    """Import ``audio/processor.py`` with its vLLM/transformers deps stubbed.

    The module imports vLLM at module scope and vLLM is not installable on the
    CPU tier, so the few names it binds at import time are faked. Only
    ``_base_hook_era`` is under test here and it touches none of them.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "src/granite_switch/vllm/audio/processor.py"

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    class _Any:
        def __init__(self, *a, **k): ...
        def __class_getitem__(cls, item):
            return cls

    # processor.py does `from .asr import ...`, so it needs a real parent
    # package. Build a synthetic one whose __path__ points at the audio dir, so
    # the sibling import resolves to the actual asr.py without dragging in
    # granite_switch.vllm.__init__ (which imports vLLM).
    pkg = mod("gs_audio_pkg")
    pkg.__path__ = [str(path.parent)]

    stubs = {
        "gs_audio_pkg": pkg,
        "vllm": mod("vllm"),
        "vllm.multimodal": mod("vllm.multimodal", MULTIMODAL_REGISTRY=object()),
        "vllm.multimodal.inputs": mod(
            "vllm.multimodal.inputs",
            MultiModalFieldConfig=_Any,
            MultiModalKwargsItems=_Any,
        ),
        "vllm.multimodal.parse": mod(
            "vllm.multimodal.parse",
            MultiModalDataDict=_Any,
            MultiModalDataItems=_Any,
            MultiModalDataParser=_Any,
        ),
        "vllm.multimodal.processing": mod(
            "vllm.multimodal.processing",
            BaseDummyInputsBuilder=_Any,
            BaseMultiModalProcessor=_Any,
            BaseProcessingInfo=_Any,
            PromptReplacement=_Any,
            PromptUpdate=_Any,
        ),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("gs_audio_pkg.processor", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["gs_audio_pkg.processor"] = m
        spec.loader.exec_module(m)
        return m
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


proc = _load_processor_module()


# --- base classes shaped like each released era ---------------------------------


class EraA:
    """vLLM 0.26-0.28."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ): ...

    def _apply_hf_processor_main(
        self,
        prompt,
        mm_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
        *,
        enable_hf_prompt_update=True,
    ): ...

    def _get_hf_mm_data(self, mm_items): ...


class EraB:
    """vLLM 0.29-0.30: the hook is gone."""

    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs): ...

    def _get_hf_mm_data(self, mm_items): ...


class EraC:
    """vLLM 0.31+: the hook is back, with a different contract."""

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ): ...

    def _apply_hf_processor_main(self, mm_items, hf_kwargs): ...

    def _get_hf_mm_inputs(self, mm_items, hf_kwargs): ...

    def _finalize_hf_mm_data(
        self, hf_data, hf_kwargs, passthrough_data, processed_data=None
    ): ...


class TestEraDetection:
    def test_era_a_detected(self):
        assert proc._base_hook_era(EraA) == proc._ERA_CALL_HF

    def test_era_b_detected(self):
        assert proc._base_hook_era(EraB) == proc._ERA_MAIN_ONLY

    def test_era_c_detected(self):
        assert proc._base_hook_era(EraC) == proc._ERA_CALL_HF_MM

    def test_era_a_and_c_are_distinguished(self):
        """The whole point: both HAVE the attribute, so hasattr cannot separate
        them. If these ever collapse, era C's two-argument call reaches era A's
        code path and every audio request raises."""
        assert hasattr(EraA, "_call_hf_processor")
        assert hasattr(EraC, "_call_hf_processor")
        assert proc._base_hook_era(EraA) != proc._base_hook_era(EraC)

    def test_only_era_b_self_extracts(self):
        """Eras A and C both re-enter ``_call_hf_processor``, so only B may take
        the branch that reads ``kwargs["hf_processor_mm_kwargs"]`` -- the key era
        C renamed."""
        self_extracting = {
            name: proc._base_hook_era(era) == proc._ERA_MAIN_ONLY
            for name, era in [("A", EraA), ("B", EraB), ("C", EraC)]
        }
        assert self_extracting == {"A": False, "B": True, "C": False}

    def test_unsignaturable_hook_falls_back_to_helper_marker(self):
        """A hook whose signature cannot be read must not default to era A on a
        base class that is clearly era C."""

        class Weird:
            _call_hf_processor = staticmethod(print)  # builtin: no useful signature

            def _get_hf_mm_inputs(self, mm_items, hf_kwargs): ...

        assert proc._base_hook_era(Weird) == proc._ERA_CALL_HF_MM
