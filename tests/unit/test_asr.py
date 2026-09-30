# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the audio ASR backend (granite_switch.vllm.audio.asr).

The module under test has no vLLM dependency, but it lives under the
``granite_switch.vllm`` package whose ``__init__`` imports vLLM. To keep this a
fast CPU-tier unit test that runs without the vLLM extra installed, we load the
leaf module directly by file path rather than through the package.
"""

import contextlib
import importlib.util
import pathlib
from unittest import mock

import numpy as np
import pytest

# Load asr.py directly (bypasses granite_switch.vllm.__init__ -> vLLM import).
_ASR_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src/granite_switch/vllm/audio/asr.py"
)
_spec = importlib.util.spec_from_file_location("gs_asr_under_test", _ASR_PATH)
asr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asr)

pytestmark = pytest.mark.audio


class TestCoerceAudio:
    def test_array_plus_rate(self):
        a = np.zeros(1600, dtype=np.float32)
        arr, sr = asr._coerce_audio(a, 16000)
        assert sr == 16000 and arr is a

    def test_tuple_form(self):
        a = np.zeros(800, dtype=np.float32)
        arr, sr = asr._coerce_audio((a, 8000), None)
        assert sr == 8000 and arr is a

    def test_list_input_becomes_ndarray(self):
        arr, sr = asr._coerce_audio([0.0] * 10, 16000)
        assert isinstance(arr, np.ndarray) and sr == 16000

    def test_missing_sampling_rate_raises(self):
        with pytest.raises(ValueError):
            asr._coerce_audio(np.zeros(10, dtype=np.float32), None)

    def test_bad_tuple_length_raises(self):
        with pytest.raises(ValueError):
            asr._coerce_audio((np.zeros(10), 1, 2), None)


class TestAsNumpy:
    def test_passthrough_ndarray(self):
        a = np.arange(5)
        assert asr._as_numpy(a) is a

    def test_list(self):
        assert np.array_equal(asr._as_numpy([1, 2, 3]), np.array([1, 2, 3]))

    def test_duck_typed_tensor(self):
        class FakeTensor:
            def __init__(self, x):
                self._x = x

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._x

        ft = FakeTensor(np.arange(4))
        assert np.array_equal(asr._as_numpy(ft), np.arange(4))


class TestMonoAndResample:
    def test_downmix_to_mono_float32(self):
        stereo = np.ones((2, 100), dtype=np.float64)
        mono = asr._to_mono_float32(stereo)
        assert mono.shape == (100,) and mono.dtype == np.float32

    def test_resample_noop_at_target(self):
        a = np.zeros(1600, dtype=np.float32)
        assert asr._resample(a, 16000, 16000) is a

    @pytest.mark.parametrize("orig_sr,target_sr", [(8000, 16000), (44100, 16000)])
    def test_resample_real_vllm_backend(self, orig_sr, target_sr):
        # Real resampling via vLLM's AudioResampler (CPU): a 1s tone keeps its
        # duration and pitch at the new rate.
        pytest.importorskip("vllm.multimodal.audio")
        freq = 220.0
        t = np.linspace(0, 1.0, orig_sr, endpoint=False, dtype=np.float32)
        tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
        out = asr._resample(tone, orig_sr, target_sr)
        assert abs(len(out) - target_sr) <= max(4, target_sr // 100)
        assert np.isfinite(out).all()
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(len(out), 1.0 / target_sr)[spectrum.argmax()]
        assert abs(peak_hz - freq) < 5.0


class TestTranscriber:
    def test_transcribe_strips_and_uses_target_rate(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        fake_pipe = mock.Mock(return_value={"text": "  hello world  "})
        t._pipeline = fake_pipe  # inject so load() is a no-op

        out = t.transcribe(np.zeros(1600, dtype=np.float32), sampling_rate=16000)
        assert out == "hello world"
        passed = fake_pipe.call_args_list[-1][0][0]
        assert passed["sampling_rate"] == 16000

    def test_load_is_idempotent_when_pipeline_set(self):
        # Once the pipeline is loaded, load() must early-return (no rebuild).
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        sentinel = object()
        t._pipeline = sentinel
        t.load()
        assert t._pipeline is sentinel


class TestChunkedTranscribe:
    """self_chunks=False routes through the split/transcribe/merge chunker."""

    def _fake_pipe_transcriber(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        # Each segment "transcribes" to a token tagged by its sample length, so we
        # can see how many windows were produced and that merge stitched them.
        t._pipeline = lambda inp, **k: {"text": f"seg{len(inp['raw'])}"}
        return t

    def test_self_chunks_true_is_single_call(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        calls = []
        t._pipeline = lambda inp, **k: (calls.append(len(inp["raw"])) or {"text": "x"})
        # 70s clip; with self_chunks the whole thing goes in one call.
        t.transcribe(
            np.zeros(70 * 16000, dtype=np.float32),
            sampling_rate=16000,
            self_chunks=True,
        )
        assert len(calls) == 1
        assert calls[0] == 70 * 16000

    def test_non_self_chunking_splits_and_merges(self):
        t = self._fake_pipe_transcriber()
        # 70s @16k, 30s window, 5s overlap -> 3 windows: 480000, 480000, 320000
        # samples. The two identical 30s window texts collapse at the seam; the
        # 20s remainder is appended.
        out = t.transcribe(
            np.zeros(70 * 16000, dtype=np.float32),
            sampling_rate=16000,
            self_chunks=False,
            chunk_length_s=30.0,
            chunk_overlap_s=5.0,
        )
        assert out == "seg480000 seg320000"

    def test_short_clip_single_window(self):
        t = self._fake_pipe_transcriber()
        out = t.transcribe(
            np.zeros(5 * 16000, dtype=np.float32),
            sampling_rate=16000,
            self_chunks=False,
            chunk_length_s=30.0,
            chunk_overlap_s=5.0,
        )
        assert out == "seg80000"


class TestUnsupportedArchitectureError:
    """A transformers too old for the default model must say so actionably."""

    def test_unrecognized_architecture_becomes_actionable_importerror(self):
        boom = mock.Mock(
            side_effect=ValueError(
                "The checkpoint you are trying to load has model type "
                "`granite_speech5_ctc` but Transformers does not recognize this "
                "architecture."
            )
        )
        with _patched_pipeline(boom):
            t = asr.ASRTranscriber(model_id=asr.DEFAULT_ASR_MODEL_ID, device="cpu")
            with pytest.raises(ImportError) as excinfo:
                t.load()
        message = str(excinfo.value)
        assert "transformers>=5.16" in message
        assert "audio" in message  # names the extra that pins it

    def test_other_value_errors_are_left_alone(self):
        boom = mock.Mock(side_effect=ValueError("some unrelated pipeline problem"))
        with _patched_pipeline(boom):
            with pytest.raises(ValueError, match="some unrelated pipeline problem"):
                asr.ASRTranscriber(model_id="m", device="cpu").load()


class TestBackendKindDrivesCallKwargs:
    """A CTC backend gets neither a chunk window nor decode kwargs; a generative
    one gets both. Guards the two ways handing chunk_length_s to a CTC pipeline
    goes wrong: chunked CTC rescales stride by inputs_to_logits_ratio (absent on
    the default checkpoint, so it silently falls back to 1 and mis-trims every
    seam), and a CTC pipeline has no generate() to take decode kwargs at all."""

    def _transcriber(self, pipeline_type, pipeline_kwargs=None):
        factory = mock.Mock(return_value=mock.Mock(type=pipeline_type))
        with _patched_pipeline(factory):
            t = asr.ASRTranscriber(
                model_id="m", device="cpu", pipeline_kwargs=pipeline_kwargs
            )
            t.load()
        # Re-point at a recorder now that load() has classified the backend.
        t._pipeline = mock.Mock(return_value={"text": "hi"})
        return t

    @pytest.mark.parametrize("pipeline_type", ["ctc", "ctc_with_lm"])
    def test_ctc_gets_no_chunk_window_and_no_decode_kwargs(self, pipeline_type):
        t = self._transcriber(pipeline_type)
        assert t._is_ctc is True
        t.transcribe(
            np.zeros(1600, dtype=np.float32),
            sampling_rate=16000,
            generate_kwargs={"language": "fr"},
            self_chunks=True,
        )
        kwargs = t._pipeline.call_args.kwargs
        assert "chunk_length_s" not in kwargs
        assert "generate_kwargs" not in kwargs

    def test_seq2seq_gets_chunk_window_and_decode_kwargs(self):
        t = self._transcriber("seq2seq_whisper")
        assert t._is_ctc is False
        t.transcribe(
            np.zeros(1600, dtype=np.float32),
            sampling_rate=16000,
            generate_kwargs={"language": "fr"},
            self_chunks=True,
        )
        kwargs = t._pipeline.call_args.kwargs
        assert kwargs["chunk_length_s"] == asr.SEQ2SEQ_CHUNK_LENGTH_S
        assert kwargs["generate_kwargs"] == {"language": "fr"}

    def test_explicit_pipeline_window_is_not_repeated_at_call_time(self):
        # Already bound into the pipeline at construction; passing it again would
        # override the checkpoint's own choice.
        t = self._transcriber("seq2seq_whisper", pipeline_kwargs={"chunk_length_s": 15})
        t.transcribe(np.zeros(1600, dtype=np.float32), sampling_rate=16000)
        assert "chunk_length_s" not in t._pipeline.call_args.kwargs

    def test_construction_passes_no_chunk_window(self):
        # The window is a call-time decision now, since it depends on the backend
        # kind, which is only known once the pipeline exists.
        factory = mock.Mock(return_value=mock.Mock(type="ctc"))
        with _patched_pipeline(factory):
            asr.ASRTranscriber(model_id="m", device="cpu").load()
        assert "chunk_length_s" not in factory.call_args.kwargs


class TestSinglePassCeiling:
    """The default 120s window is the boundary between 'backend handles it whole'
    and 'our chunker splits it'."""

    def _recording_transcriber(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        t._is_ctc = True
        t.calls = []
        t._pipeline = lambda inp, **k: (
            t.calls.append(len(inp["raw"])) or {"text": f"seg{len(inp['raw'])}"}
        )
        return t

    def test_clip_at_the_ceiling_reaches_the_backend_whole(self):
        t = self._recording_transcriber()
        n = int(asr.DEFAULT_CHUNK_LENGTH_S) * 16000
        t.transcribe(np.zeros(n, dtype=np.float32), sampling_rate=16000)
        assert t.calls == [n]

    def test_clip_past_the_ceiling_is_split(self):
        t = self._recording_transcriber()
        n = int(asr.DEFAULT_CHUNK_LENGTH_S * 2) * 16000
        t.transcribe(np.zeros(n, dtype=np.float32), sampling_rate=16000)
        assert len(t.calls) > 1
        assert max(t.calls) <= int(asr.DEFAULT_CHUNK_LENGTH_S) * 16000


class TestTranscriberCache:
    def test_same_key_returns_same_instance(self):
        a = asr.get_transcriber("m", "cpu")
        b = asr.get_transcriber("m", "cpu")
        assert a is b

    def test_default_model_id_resolution(self):
        t = asr.get_transcriber(None, "cpu")
        assert t.model_id == asr.DEFAULT_ASR_MODEL_ID

    def test_different_device_distinct_instance(self):
        a = asr.get_transcriber("m", "cpu")
        b = asr.get_transcriber("m", "cuda:0")
        assert a is not b

    def test_pipeline_kwargs_stored_on_instance(self):
        t = asr.get_transcriber("m", "cpu", pipeline_kwargs={"chunk_length_s": 15})
        assert t.pipeline_kwargs == {"chunk_length_s": 15}

    def test_pipeline_kwargs_are_part_of_cache_key(self):
        # Different pipeline_kwargs → different cached pipeline (they change how
        # the pipeline is constructed), same kwargs → same instance.
        a = asr.get_transcriber("pk", "cpu", pipeline_kwargs={"chunk_length_s": 15})
        b = asr.get_transcriber("pk", "cpu", pipeline_kwargs={"chunk_length_s": 30})
        c = asr.get_transcriber("pk", "cpu", pipeline_kwargs={"chunk_length_s": 15})
        assert a is not b
        assert a is c

    def test_pipeline_kwargs_key_is_order_independent(self):
        a = asr.get_transcriber("pk2", "cpu", pipeline_kwargs={"x": 1, "y": 2})
        b = asr.get_transcriber("pk2", "cpu", pipeline_kwargs={"y": 2, "x": 1})
        assert a is b


class TestFreeze:
    def test_dict_order_independent(self):
        assert asr._freeze({"a": 1, "b": 2}) == asr._freeze({"b": 2, "a": 1})

    def test_nested_and_list(self):
        frozen = asr._freeze({"a": [1, 2], "b": {"c": 3}})
        # Result must be hashable (usable as a dict key).
        assert hash(frozen) == hash(asr._freeze({"b": {"c": 3}, "a": [1, 2]}))


class TestGenerateKwargsPassthrough:
    def test_generate_kwargs_forwarded_to_pipeline_call(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        fake_pipe = mock.Mock(return_value={"text": "hola"})
        t._pipeline = fake_pipe
        t.transcribe(
            np.zeros(1600, dtype=np.float32),
            sampling_rate=16000,
            generate_kwargs={"language": "es"},
        )
        # generate_kwargs is forwarded to the pipeline call as a kwarg.
        assert fake_pipe.call_args_list[-1].kwargs["generate_kwargs"] == {
            "language": "es"
        }

    def test_empty_generate_kwargs_not_passed(self):
        # CTC / non-generative backends must not receive a generate_kwargs kwarg.
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        fake_pipe = mock.Mock(return_value={"text": "ok"})
        t._pipeline = fake_pipe
        t.transcribe(np.zeros(1600, dtype=np.float32), sampling_rate=16000)
        assert "generate_kwargs" not in fake_pipe.call_args_list[-1].kwargs
        t.transcribe(
            np.zeros(1600, dtype=np.float32), sampling_rate=16000, generate_kwargs={}
        )
        assert "generate_kwargs" not in fake_pipe.call_args_list[-1].kwargs


class TestResolveGenerateKwargs:
    def test_config_defaults_only(self):
        out = asr.resolve_generate_kwargs({"language": "de", "task": "transcribe"})
        assert out == {"language": "de", "task": "transcribe"}

    def test_none_config_is_empty(self):
        assert asr.resolve_generate_kwargs(None) == {}

    def test_top_level_language_overrides_config(self):
        out = asr.resolve_generate_kwargs({"language": "de"}, {"language": "fr"})
        assert out == {"language": "fr"}

    def test_nested_request_allowlisted_keys_merge(self):
        out = asr.resolve_generate_kwargs(
            {"language": "de"},
            {"asr_generate_kwargs": {"task": "translate"}},
        )
        assert out == {"language": "de", "task": "translate"}

    def test_disallowed_request_keys_dropped(self):
        # A client cannot inject arbitrary generation options.
        out = asr.resolve_generate_kwargs(
            {"language": "de"},
            {"asr_generate_kwargs": {"num_beams": 99, "task": "translate"}},
        )
        assert out == {"language": "de", "task": "translate"}
        assert "num_beams" not in out

    def test_request_wins_over_config(self):
        out = asr.resolve_generate_kwargs(
            {"language": "de", "task": "transcribe"},
            {"asr_generate_kwargs": {"language": "ja"}},
        )
        assert out["language"] == "ja"
        assert out["task"] == "transcribe"

    def test_config_not_mutated(self):
        cfg = {"language": "de"}
        asr.resolve_generate_kwargs(cfg, {"language": "fr"})
        assert cfg == {"language": "de"}


@contextlib.contextmanager
def _patched_pipeline(factory):
    """Patch both lookup paths ``ASRTranscriber.load`` may resolve through.

    ``load()`` does ``from transformers import pipeline`` at call time, which
    reads the top-level attribute; transformers is a lazy module, so that
    attribute may not exist yet and gets resolved from ``transformers.pipelines``
    on first access. Both therefore need patching.

    **The order matters and is load-bearing.** ``transformers.pipeline`` must be
    patched FIRST. ``mock.patch.__enter__`` records the current value so it can
    restore it, and if ``transformers.pipelines.pipeline`` were replaced first,
    resolving ``transformers.pipeline`` would return *that mock* and record it as
    the original -- which the patch then faithfully restores on exit, leaving the
    mock installed for the rest of the process. See
    ``TestPatchedPipelineRestores``.
    """
    with (
        mock.patch("transformers.pipeline", factory),
        mock.patch("transformers.pipelines.pipeline", factory),
    ):
        yield


class TestPatchedPipelineRestores:
    """``_patched_pipeline`` must leave ``transformers.pipeline`` as it found it.

    Regression guard for a leak that was expensive to diagnose. With the two
    patches in the wrong order the helper restored its own mock instead of the
    real function, so every later real ``pipeline()`` call in the session got it.
    Because one of the mocks in this file raises the "does not recognize this
    architecture" ValueError, ``_unsupported_architecture_error`` then reported a
    bogus "requires transformers>=5.16" ImportError from an unrelated GPU test --
    naming the installed version as too old for itself.

    The leak only surfaces when ``transformers.pipeline`` is not already cached on
    the top-level module, which is why a full-suite run reproduced it and this
    file alone did not. The test forces that precondition instead of depending on
    collection order.
    """

    def test_pipeline_attribute_is_restored(self):
        import transformers

        real = transformers.pipeline  # resolve once, to compare against
        # Force the lazy-resolve path, which is what makes the ordering matter.
        transformers.__dict__.pop("pipeline", None)

        sentinel = mock.Mock(side_effect=ValueError("this mock must not escape"))
        with _patched_pipeline(sentinel):
            from transformers import pipeline as inside

            assert inside is sentinel, "patch did not take effect"

        from transformers import pipeline as after

        assert after is not sentinel, (
            "_patched_pipeline leaked its mock onto transformers.pipeline; "
            "check the patch order in the helper"
        )
        assert after is real

    def test_pipelines_submodule_attribute_is_restored(self):
        import transformers.pipelines

        real = transformers.pipelines.pipeline
        sentinel = mock.Mock(side_effect=ValueError("this mock must not escape"))
        with _patched_pipeline(sentinel):
            assert transformers.pipelines.pipeline is sentinel
        assert transformers.pipelines.pipeline is real


class TestResolveTorchDtype:
    """asr_dtype resolution. bfloat16-on-CUDA is the default, but overridable."""

    def test_auto_on_cuda_is_bfloat16(self):
        # bfloat16, not float16: it is the default checkpoint's own dtype and
        # keeps float32's exponent range next to the encoder's BatchNorm layers.
        torch = pytest.importorskip("torch")
        assert asr._resolve_torch_dtype(None, "cuda:0") is torch.bfloat16
        assert asr._resolve_torch_dtype("auto", "cuda") is torch.bfloat16

    def test_auto_on_cpu_is_float32(self):
        torch = pytest.importorskip("torch")
        assert asr._resolve_torch_dtype(None, "cpu") is torch.float32

    def test_explicit_float32_overrides_cuda_default(self):
        # The BatchNorm-encoder case: CUDA must not force half precision.
        torch = pytest.importorskip("torch")
        assert asr._resolve_torch_dtype("float32", "cuda:0") is torch.float32

    def test_explicit_bfloat16(self):
        torch = pytest.importorskip("torch")
        assert asr._resolve_torch_dtype("bfloat16", "cpu") is torch.bfloat16

    @pytest.mark.parametrize(
        "name,attr",
        [
            ("fp16", "float16"),
            ("half", "float16"),
            ("bf16", "bfloat16"),
            ("fp32", "float32"),
            ("FLOAT32", "float32"),
        ],
    )
    def test_aliases_and_case(self, name, attr):
        torch = pytest.importorskip("torch")
        assert asr._resolve_torch_dtype(name, "cpu") is getattr(torch, attr)

    def test_unknown_name_raises(self):
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match="Unsupported asr_dtype"):
            asr._resolve_torch_dtype("int8", "cpu")

    def test_dtype_is_part_of_cache_key(self):
        a = asr.get_transcriber("dt", "cuda:0", dtype="float32")
        b = asr.get_transcriber("dt", "cuda:0", dtype="float16")
        c = asr.get_transcriber("dt", "cuda:0", dtype="float32")
        assert a is not b
        assert a is c

    def test_load_passes_resolved_dtype(self):
        torch = pytest.importorskip("torch")
        factory = mock.Mock(return_value=mock.Mock())
        with _patched_pipeline(factory):
            asr.ASRTranscriber(model_id="m", device="cuda:0", dtype="float32").load()
        assert factory.call_args.kwargs["torch_dtype"] is torch.float32

    def test_pipeline_kwargs_torch_dtype_still_wins(self):
        torch = pytest.importorskip("torch")
        factory = mock.Mock(return_value=mock.Mock())
        with _patched_pipeline(factory):
            asr.ASRTranscriber(
                model_id="m",
                device="cpu",
                dtype="float32",
                pipeline_kwargs={"torch_dtype": torch.bfloat16},
            ).load()
        assert factory.call_args.kwargs["torch_dtype"] is torch.bfloat16


class TestLoadMergesPipelineKwargs:
    def test_pipeline_kwargs_override_defaults(self):
        # load() must merge config-supplied pipeline_kwargs over the built-in
        # defaults (e.g. override chunk_length_s, add extra kwargs).
        pytest.importorskip("torch")
        fake_pipe_factory = mock.Mock(return_value=mock.Mock())
        with _patched_pipeline(fake_pipe_factory):
            t = asr.ASRTranscriber(
                model_id="m",
                device="cpu",
                pipeline_kwargs={"chunk_length_s": 15, "batch_size": 4},
            )
            t.load()
        kwargs = fake_pipe_factory.call_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["task"] == "automatic-speech-recognition"
        assert kwargs["chunk_length_s"] == 15  # overrode the default 30
        assert kwargs["batch_size"] == 4  # extra kwarg passed through
