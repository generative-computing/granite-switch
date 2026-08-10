# Audio Input (Alpha)

Granite Switch can accept **audio input** through a single vLLM model load — no
separate speech server, no change to how developers deploy or call the model.

This is an **alpha**: a speech-to-text *cascade*. Audio is transcribed to text by
a small ASR model and the transcript is fed to the LLM as ordinary tokens. It is
intentionally simple and requires no training. The "proper" upgrade (feeding a
trained projection of a speech encoder's embeddings straight into the LLM) reuses
the same hooks — see [Design](#design) below.

## Installing

The audio path needs `soundfile` and `librosa` on top of the vLLM backend — they
decode and resample the incoming waveform. They live in the `audio` extra, which is
**not** part of `vllm`, so a plain `uv sync --extra vllm` gives you a checkpoint that
fails on any non-16 kHz input:

```bash
# Serving an audio-enabled checkpoint
uv sync --extra vllm --extra audio     # or --extra vllm20 --extra audio

# Development / running the test suite (the dev groups include audio already)
uv sync --group dev                    # vLLM 0.19.x
uv sync --group dev-vllm20             # vLLM 0.20.x
```

## Building an audio-enabled checkpoint

Add `--enable-audio` when composing:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.0-micro \
  --built-in-adapters core \
  --enable-audio \
  --output ./granite-switch-audio
```

This adds the `<|audio|>` marker token to the tokenizer and writes the audio
settings into `config.json` so the checkpoint is self-describing:

```json
{ "asr_enabled": true, "asr_model_id": null, "asr_device": "cpu" }
```

- `asr_model_id` — HF id of the speech-to-text model (default: a small built-in
  `distil-whisper/distil-small.en`). Override with `--asr-model <hf-id>`, e.g.
  `openai/whisper-small` for multilingual.
- `asr_device` — `cpu` (default) keeps vLLM's GPU KV-cache budget clean; set
  `--asr-device cuda:0` to run transcription on GPU (watch GPU memory).
- `asr_dtype` — precision the ASR weights load in. Unset (default) derives it
  from the device: `float16` on CUDA, `float32` on CPU. Half precision halves
  the ASR weight footprint and is what the Whisper-family defaults expect, but
  it is not universally safe — an encoder with **BatchNorm** layers raises
  `Expected weight to have type Float but got Half`, since BatchNorm will not
  promote a float16 weight against float32 features. Such a checkpoint needs
  `--asr-dtype float32`. Accepted: `auto`, `float16`, `bfloat16`, `float32`.

Audio capability is **gated per checkpoint** by `asr_enabled`: a checkpoint built
without `--enable-audio` reports no audio modality and never loads the ASR model.

### Tuning the ASR model

Two optional config fields let a checkpoint carry ASR tuning so no code change is
needed to swap or steer any HF `automatic-speech-recognition` model:

- `asr_pipeline_kwargs` — extra kwargs merged into the `transformers.pipeline(...)`
  **construction** (e.g. `chunk_length_s`, `batch_size`). These change how the
  pipeline is built, so they are folded into the transcriber cache key.
- `asr_generate_kwargs` — **decode-time** defaults applied on every transcription
  (e.g. `language`, `task` for a multilingual Whisper). Applied at call time, so
  one loaded pipeline is reused. Ignored by non-generative backends (e.g. CTC).

Set them at compose time (JSON), which writes them into `config.json`:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --adapters ... \
  --asr-model openai/whisper-large-v3 \
  --asr-pipeline-kwargs '{"chunk_length_s": 15}' \
  --asr-generate-kwargs '{"language": "de", "task": "transcribe"}'
```

Because they live in `config.json`, an existing audio checkpoint can be retuned by
editing that file directly — no re-compose and no patched package:

```json
{ "asr_enabled": true, "asr_model_id": "openai/whisper-large-v3",
  "asr_pipeline_kwargs": {"chunk_length_s": 15},
  "asr_generate_kwargs": {"language": "de", "task": "transcribe"} }
```

### Long audio & multiple clips

The transcript is spliced into the prompt as ordinary text tokens — it is **not**
truncated to fit. A request behaves exactly like a long text request: if the
prompt plus the transcript(s) leaves no room for the answer within the served
`max_model_len`, vLLM rejects it with its standard prompt-length error (HTTP 400).
Shorten the audio or serve with a larger `--max-model-len`. Relevant config fields
(all optional, sensible defaults):

- `asr_max_audio_clips` (default `32`) — how many audio clips one request may
  carry; each is spliced at its own `<|audio|>` marker. `--limit-mm-per-prompt`
  may lower this per deployment but cannot raise it above the declared value.
  Clips cost no extra KV (transcripts are ordinary text tokens bounded by the
  context); the ceiling guards against one request triggering an unbounded number
  of synchronous transcriptions.
- `asr_max_audio_seconds_per_clip` (default `600.0`) — longest single clip.
- `asr_max_total_audio_seconds` (default `1800.0`) — longest total across all
  clips in one request.
- `asr_max_audio_samples` (default `0` = derive from the total above at 16 kHz) —
  absolute decoded-sample cap, as a rate-independent backstop.

These three bound *duration*, which the clip count does not. They are enforced
**before any transcription runs**, which matters because vLLM's prompt-length
check happens after preprocessing: without them a caller could have a multi-hour
file fully transcribed and only then rejected — a free denial-of-service lever,
and a synchronous block of vLLM's input path for as long as the transcription
takes. An over-long request is refused with a message naming the offending size
and the knob that rejected it.

Raise them if you serve genuinely long recordings; the defaults are a policy
choice, not a technical limit. Note that long *single* clips are still handled by
chunking (below) — these limits cap the input, not the transcript.

**Long single clips** are handled two ways, selected by `asr_self_chunks`:

- `asr_self_chunks: true` (default) — the backend chunks internally. The Whisper
  pipeline does this via `chunk_length_s` with timestamp-based stitching, so our
  chunker is bypassed.
- `asr_self_chunks: false` — route audio through the **encoder-agnostic** chunker:
  split into overlapping windows (`asr_chunk_length_s`, default `30.0`;
  `asr_chunk_overlap_s`, default `5.0`), transcribe each, and merge with
  overlap de-duplication. Use this for a backend with a fixed input window (e.g. a
  speech encoder that cannot self-chunk); the transcript stitching then lives
  above the backend so any backend inherits long-audio support.

These are settable at compose time and are equally editable in `config.json`:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --adapters ... --enable-audio \
  --asr-max-audio-clips 4 \
  --asr-no-self-chunks --asr-chunk-length-s 20 --asr-chunk-overlap-s 3
```

### Per-request language (multilingual)

For one deployment that serves many languages, a request can override the config
default via `mm_processor_kwargs`. Only `language` and `task` are honored from a
request (an allowlist — clients cannot inject arbitrary generation options); the
config default supplies everything else, and request values win:

```python
out = llm.generate({
    "prompt": "Transcript of the audio: <|audio|>\nAnswer:",
    "multi_modal_data": {"audio": [(audio, sr)]},
    "mm_processor_kwargs": {"language": "fr"},   # this request, French
}, SamplingParams(max_tokens=128))
```

The same cached pipeline serves every language — the decode kwargs are applied per
call, so there is no per-language reload.

## Calling it

### Python (offline)

```python
from granite_switch.vllm import register; register()
from vllm import LLM, SamplingParams
import soundfile as sf

llm = LLM(model="./granite-switch-audio")          # one model load
audio, sr = sf.read("question.wav")                # numpy array + sample rate

out = llm.generate({
    "prompt": "Transcript of the audio: <|audio|>\nAnswer:",
    "multi_modal_data": {"audio": [(audio, sr)]},
}, SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
```

The `<|audio|>` marker is where the transcript is spliced in.

### OpenAI-compatible server / chat API

```bash
vllm serve ./granite-switch-audio --port 8000
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
resp = client.chat.completions.create(
    model="granite-switch-audio",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Answer the question in the audio."},
        {"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}},
    ]}],
)
print(resp.choices[0].message.content)
```

The chat template emits the `<|audio|>` marker for audio content parts
(`audio` / `input_audio` / `audio_url`), so the processor splices the transcript
in automatically — callers send standard chat messages, no manual marker needed.

## Design

Per request, before the scheduler allocates KV cache:

1. vLLM's multimodal pipeline hands the audio to our processor
   (`granite_switch.vllm.audio`).
2. The processor runs ASR → transcript → token ids.
3. A `PromptReplacement` swaps the `<|audio|>` marker for those transcript token
   ids. The scheduler then sizes KV for the **real** length — the audio "window"
   is variable and decided at runtime, not reserved in advance.
   A clip with no recognizable speech in it — silence, music, noise, or a clip
   too short to hold a word — transcribes to the empty string. Since every audio
   item has to occupy at least one prompt position (vLLM discards a zero-length
   placeholder and then rejects the request), those clips are replaced with a
   single space instead: the model sees an audio turn that said nothing, rather
   than an error.
4. The model's `embed_multimodal` supplies embeddings for those positions. In the
   alpha that is simply the transcript's own token embeddings (identical to
   embedding them as text). **This is the seam the future encoder reuses:** swap
   `embed_multimodal` to return `projection(speech_encoder(audio))` and the rest
   of the machinery is unchanged.

The decoder, switch, and LoRA paths are untouched — they only ever see text
tokens.

## Limitations (alpha)

- **Cascade, not end-to-end.** Prosody/emotion/uncertainty are lost; ASR errors
  propagate to the LLM. Two models run sequentially (ASR then LLM).
- **English by default** (`distil-whisper/distil-small.en`). Use `--asr-model`
  with a multilingual model and set the language via `asr_generate_kwargs` (or
  per request via `mm_processor_kwargs`; see *Tuning the ASR model* above).
- **HF `pipeline` backends only.** Any `automatic-speech-recognition` pipeline
  model works via config alone; a non-pipeline backend (cloud STT, faster-whisper,
  a custom encoder) still needs a code-level plug point — tracked as future work.
- Multiple clips share one context window: the per-clip transcript budget is the
  context split across the request's clips, so many/long clips together are bound
  by `max_model_len` (see *Long audio & multiple clips* above).
- Chunk-merge de-duplication is text-level (word overlap at each seam); it can
  mis-handle a phrase legitimately repeated across a window boundary. Whisper's
  internal timestamp stitching (`asr_self_chunks: true`) is more precise.

## Audio + adapters

Audio requests route through adapters exactly like text requests. The model sets
`requires_raw_input_tokens = True` so vLLM passes the raw `input_ids` to the
forward pass on the multimodal path; the switch then detects adapter control
tokens as usual, and `embed_input_ids` applies the same token-exchange rewrite
(control → substitute id) used for text — so an audio request that activates an
adapter behaves identically to the text equivalent.

## Tests

Everything on the audio path carries the `audio` marker, so the whole tier selects
in one command regardless of where the tests live:

```bash
# All audio tests (13 of them need a GPU and a real checkpoint)
pytest -m audio -v -s --tb=short

# CPU tier only — runs in a few seconds
pytest -m "audio and not gpu" -v -s --tb=short
```

- `tests/unit/test_asr.py` — CPU unit tests for the ASR backend (audio coercion,
  resampling, transcription with a mocked pipeline, pipeline-kwargs cache keying,
  and per-request decode-kwargs resolution). No GPU/vLLM required.
- `tests/unit/test_config.py` — round-trips `asr_pipeline_kwargs` /
  `asr_generate_kwargs` through save/load.
- End-to-end (GPU): compose an `--enable-audio` checkpoint, then an audio request
  through vLLM produces an answer and text-only requests are unaffected.
