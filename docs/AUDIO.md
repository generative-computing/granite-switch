# Audio Input (Alpha)

Granite Switch can accept **audio input** through a single vLLM model load — no
separate speech server, no change to how developers deploy or call the model.

This is an **alpha**: a speech-to-text *cascade*. Audio is transcribed to text by
a small ASR model and the transcript is fed to the LLM as ordinary tokens. It is
intentionally simple and requires no training. The "proper" upgrade (feeding a
trained projection of a speech encoder's embeddings straight into the LLM) reuses
the same hooks — see [Design](#design) below.

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

The transcript token budget is derived from the **served context window**, not a
fixed cap. Per request, the audio may occupy roughly
`max_model_len − asr_generation_reserve_tokens − prompt_tokens`, split across the
request's clips. Relevant config fields (all optional, sensible defaults):

- `asr_max_audio_clips` (default `32`) — how many audio clips one request may
  carry; each is spliced at its own `<|audio|>` marker. `--limit-mm-per-prompt`
  may lower this per deployment but cannot raise it above the declared value.
  Clips cost no extra KV (transcripts are ordinary text tokens bounded by the
  context); the ceiling guards against one request triggering an unbounded number
  of synchronous transcriptions.
- `asr_generation_reserve_tokens` (default `8192`) — context held back for the
  generated answer (and prompt overhead) when sizing the transcript budget.

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

- `tests/unit/test_asr.py` — CPU unit tests for the ASR backend (audio coercion,
  resampling, transcription with a mocked pipeline, pipeline-kwargs cache keying,
  and per-request decode-kwargs resolution). No GPU/vLLM required.
- `tests/unit/test_config.py` — round-trips `asr_pipeline_kwargs` /
  `asr_generate_kwargs` through save/load.
- End-to-end (GPU): compose an `--enable-audio` checkpoint, then an audio request
  through vLLM produces an answer and text-only requests are unaffected.
