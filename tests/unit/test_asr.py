# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the audio ASR backend (granite_switch.vllm.audio.asr).

The module under test has no vLLM dependency, but it lives under the
``granite_switch.vllm`` package whose ``__init__`` imports vLLM. To keep this a
fast CPU-tier unit test that runs without the vLLM extra installed, we load the
leaf module directly by file path rather than through the package.
"""

import importlib.util
import pathlib
from unittest import mock

import numpy as np
import pytest

# Load asr.py directly (bypasses granite_switch.vllm.__init__ -> vLLM import).
_ASR_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/granite_switch/vllm/audio/asr.py"
)
_spec = importlib.util.spec_from_file_location("gs_asr_under_test", _ASR_PATH)
asr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asr)


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
            def __init__(self, x): self._x = x
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self._x

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

    def test_resample_without_librosa_raises_clear_error(self):
        # When librosa is unavailable, a non-target rate must raise a clear error.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("no librosa")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="librosa"):
                asr._resample(np.zeros(800, dtype=np.float32), 8000, 16000)


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
        t._pipeline = lambda inp, **k: {"text": "seg%d" % len(inp["raw"])}
        return t

    def test_self_chunks_true_is_single_call(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        calls = []
        t._pipeline = lambda inp, **k: (calls.append(len(inp["raw"])) or {"text": "x"})
        # 70s clip; with self_chunks the whole thing goes in one call.
        t.transcribe(np.zeros(70 * 16000, dtype=np.float32), sampling_rate=16000,
                     self_chunks=True)
        assert len(calls) == 1
        assert calls[0] == 70 * 16000

    def test_non_self_chunking_splits_and_merges(self):
        t = self._fake_pipe_transcriber()
        # 70s @16k, 30s window, 5s overlap -> 3 windows: 480000, 480000, 320000
        # samples. The two identical 30s window texts collapse at the seam; the
        # 20s remainder is appended.
        out = t.transcribe(
            np.zeros(70 * 16000, dtype=np.float32), sampling_rate=16000,
            self_chunks=False, chunk_length_s=30.0, chunk_overlap_s=5.0,
        )
        assert out == "seg480000 seg320000"

    def test_short_clip_single_window(self):
        t = self._fake_pipe_transcriber()
        out = t.transcribe(
            np.zeros(5 * 16000, dtype=np.float32), sampling_rate=16000,
            self_chunks=False, chunk_length_s=30.0, chunk_overlap_s=5.0,
        )
        assert out == "seg80000"


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
        assert fake_pipe.call_args_list[-1].kwargs["generate_kwargs"] == {"language": "es"}

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


class TestLoadMergesPipelineKwargs:
    def test_pipeline_kwargs_override_defaults(self):
        # load() must merge config-supplied pipeline_kwargs over the built-in
        # defaults (e.g. override chunk_length_s, add extra kwargs).
        pytest.importorskip("torch")
        fake_pipe_factory = mock.Mock(return_value=mock.Mock())
        # transformers is a lazy module: `from transformers import pipeline`
        # re-resolves to transformers.pipelines.pipeline, so patch there.
        with mock.patch("transformers.pipelines.pipeline", fake_pipe_factory):
            t = asr.ASRTranscriber(
                model_id="m",
                device="cpu",
                pipeline_kwargs={"chunk_length_s": 15, "batch_size": 4},
            )
            t.load()
        kwargs = fake_pipe_factory.call_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["task"] == "automatic-speech-recognition"
        assert kwargs["chunk_length_s"] == 15   # overrode the default 30
        assert kwargs["batch_size"] == 4        # extra kwarg passed through
