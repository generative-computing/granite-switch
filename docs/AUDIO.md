# Audio Input (Alpha)

Granite Switch can accept **audio input** through a single vLLM model load — no
separate speech server, no change to how developers deploy or call the model.

This is an **alpha**: a speech-to-text *cascade*. Audio is transcribed to text by
a small ASR model and the transcript is fed to the LLM as ordinary tokens. It is
intentionally simple and requires no training. The "proper" upgrade (feeding a
trained projection of a speech encoder's embeddings straight into the LLM) reuses
the same hooks — see [Design](#design) below.

## Installing

The audio path needs vLLM's audio deps (`av`, `soundfile`, `resampy`, `scipy`) to
decode and resample the incoming waveform. They come from vLLM's own `[audio]`
extra, which the `audio` extra here pulls in (as `vllm[audio]`). A plain
`uv sync --extra vllm` omits them, so it gives you a checkpoint that fails on any
non-16 kHz input.

The `audio` extra also requires **transformers >= 5.16**, the release that added
`granite_speech5_ctc` — the architecture of the default ASR model. On an older
transformers the first transcription raises an `ImportError` naming the fix
(the rest of the package still works on an older release, which is why the
requirement sits on the extra rather than the core dependency).

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
  --base-model ibm-granite/granite-4.1-3b \
  --built-in-adapters core \
  --enable-audio \
  --output ./granite-switch-audio
```

This adds the `<|audio|>` marker token to the tokenizer and writes the audio
settings into `config.json` so the checkpoint is self-describing:

```json
{ "asr_enabled": true, "asr_model_id": null, "asr_device": "cuda" }
```

- `asr_model_id` — HF id of the speech-to-text model (default:
  `ibm-granite/granite-speech-5.0-470m-turboctc`, a 470M English conformer CTC
  encoder). Override with `--asr-model <hf-id>`, e.g. `openai/whisper-small` for
  multilingual.
- `asr_device` — `cuda` (default): the default encoder is small and its speed
  comes from running on GPU. Set `--asr-device cpu` to leave vLLM's whole GPU
  memory budget to the KV cache — transcription is then several times slower
  (measured ~3x realtime on a laptop CPU, i.e. a 10-minute clip takes minutes).
  On GPU, mind that vLLM pre-allocates its KV cache first, so a tight
  `--gpu-memory-utilization` can leave too little for the ASR weights.
- `asr_dtype` — precision the ASR weights load in. Unset (default) derives it
  from the device: `bfloat16` on CUDA, `float32` on CPU. bfloat16 because it is
  the default checkpoint's own dtype (no conversion implied) and because it keeps
  float32's exponent range, which is the safer choice for an encoder carrying
  **BatchNorm** in every conv block. Note that float16 is *not* rejected by this
  model — measured on an A100 (torch 2.10 / transformers 5.16) it loads and
  transcribes correctly — so bfloat16 is a considered default, not a hard
  requirement. A different encoder may still hit
  `Expected weight to have type Float but got Half`, since BatchNorm will not
  promote a float16 weight against float32 features; such a checkpoint needs
  `--asr-dtype float32`. Accepted: `auto`, `float16`, `bfloat16`, `float32`.

Audio capability is **gated per checkpoint** by `asr_enabled`: a checkpoint built
without `--enable-audio` reports no audio modality and never loads the ASR model.

`--enable-audio` is the **only** flag that switches audio on. The other `--asr-*`
options configure the cascade; they do not enable it. Passing one without
`--enable-audio` composes a text-only checkpoint and the value is not written — so
`--asr-model openai/whisper-small` on its own gets you no audio. This is
deliberate: one explicit flag decides, rather than the decision being inferred
from which options happen to be set.

### Tuning the ASR model

Two optional config fields let a checkpoint carry ASR tuning so no code change is
needed to swap or steer any HF `automatic-speech-recognition` model:

- `asr_pipeline_kwargs` — extra kwargs merged into the `transformers.pipeline(...)`
  **construction** (e.g. `chunk_length_s`, `batch_size`). These change how the
  pipeline is built, so they are folded into the transcriber cache key.
- `asr_generate_kwargs` — **decode-time** defaults applied on every transcription
  (e.g. `language`, `task` for a multilingual Whisper). Applied at call time, so
  one loaded pipeline is reused. Dropped for a CTC backend (the default), which
  has no ``generate()`` to steer.

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

**Long single clips** are handled two ways, selected by `asr_self_chunks`:

- `asr_self_chunks: false` (default) — route audio through the
  **encoder-agnostic** chunker: split into overlapping windows
  (`asr_chunk_length_s`, default `120.0`; `asr_chunk_overlap_s`, default `5.0`),
  transcribe each, and merge with overlap de-duplication. A clip at or under the
  window is a single segment and reaches the backend whole, so the CTC default
  handles everything up to two minutes in one pass and only longer clips are
  split. The window is what bounds activation memory: measured on CPU, peak RSS
  was ~1.4GB at 60s of audio, ~2.3GB at 300s and ~3.5GB at 600s.
- `asr_self_chunks: true` — the backend handles long audio itself. For a
  generative backend that means its own timestamp-based stitching (Whisper), which
  is more precise than our text-level merge. For a CTC backend it means feeding an
  arbitrarily long clip in one pass — its block attention keeps cost linear in
  duration, so this is a memory-for-accuracy trade rather than a hard limit.

The HF pipeline's *own* CTC chunking is deliberately never used: it rescales chunk
stride by the model's `inputs_to_logits_ratio`, which the CTC default does not
publish, so the pipeline falls back to `1` and trims every seam at the wrong
offset. `chunk_length_s` therefore reaches only a generative backend, and only at
call time — once the pipeline exists and its kind is known.

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

Both Granite chat-template families are supported, detected at compose time:

| Family | Role markers | How the marker is emitted |
|---|---|---|
| `granite_format` | `<\|start_of_role\|>` | An `elif` added to the existing content-part loop |
| `chatml` | `<\|im_start\|>` | A flattening block, since ChatML has no content-part loop |

**Nothing on the audio path is architecture-specific.** The compose-time gate is
`model_type.startswith("granite")` and the injection above keys off the *detected
template family*, never the architecture — so a dense base and a pure sparse MoE
base (`granitemoe`, no `shared_mlp`) go down identical code, and the marker's
output-row fixup only ever touches embedding rows. What does differ is the
adapter surface, not the audio: see *Audio + adapters* below.

A base whose tokenizer carries **no chat template at all** is *not* refused:
`configure_audio_chat_template` warns and returns, and compose completes. The
checkpoint then carries `asr_enabled: true` while its template emits no
`<|audio|>` marker, so an audio content part on the chat path is dropped rather
than transcribed — offline `llm.generate` with a hand-written marker still works.
Compose from the instruct-tuned sibling, or supply a template first. (A template
that *is* present but whose family cannot be identified does raise.)

The ChatML template consumes `message.content` as a string
(`{%- set content = message.content | string %}`), so a multimodal parts *list*
would otherwise render as that list's Python repr — audio payload included —
rather than as a marker. Compose rebuilds `content` from the parts instead. Part
order is preserved, so each transcript is spliced in where its clip sat relative
to the text.

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

### The marker's output row

`<|audio|>` is a new vocabulary entry, so `resize_token_embeddings` appends a row
for it — sampled from the distribution of the *trained* rows (`mean_resizing=True`
since transformers 4.46). Left alone, the marker would carry an arbitrary,
compose-run-dependent output logit despite never having been trained, and nothing
suppresses it at generation time. A generated `<|audio|>` in a reply that is fed
back on a later turn makes the marker/audio-item counts disagree and the request
is rejected.

Compose therefore copies a reserved `<|unused_N|>` row into the marker's row, so
its logit is identical to a token the base model was trained not to emit, for
every hidden state. On a tied-embedding base that row is shared with the input
embedding, which is inert here: the marker is replaced by transcript ids before
the decoder runs, and a marker without a matching audio item is rejected
up-front, so the marker's input row is never read.

**Confirmed with the Granite model authors:** `<|unused_N|>` ids are reserved and
the model is trained not to emit them, so borrowing one of their rows is the
intended use. This was previously inferred from checkpoint structure — the
measurement that motivated it is in `tokenizer_setup.py` next to
`_RESERVED_UNUSED_TOKEN_RE` and still worth reading, but it is corroboration now
rather than the basis.

If a vocabulary has no reserved slots, compose warns and leaves the row as
generated. Note the inventory is not stable across releases (4.1 has 69 unused
ids, 4.2 has 72), so nothing should depend on a specific count or id range —
`find_reserved_never_emitted_token_id` looks them up each time. The rows are
present on `granitemoe` bases too, so this policy needs no architecture-specific
fallback.

## Limitations (alpha)

- **Cascade, not end-to-end.** Prosody/emotion/uncertainty are lost; ASR errors
  propagate to the LLM. Two models run sequentially (ASR then LLM).
- **English only by default** (`ibm-granite/granite-speech-5.0-470m-turboctc`),
  and being CTC it has no language/task knobs at all, so the per-request
  `language` override is inert. For other languages use `--asr-model` with a
  multilingual generative model and set the language via `asr_generate_kwargs` (or
  per request via `mm_processor_kwargs`; see *Tuning the ASR model* above).
- **Transcripts from the CTC default are lowercase and unpunctuated**
  (`what is the capital of israel`). They are spliced into the prompt as ordinary
  text, so the LLM reads them that way. A generative backend such as Whisper
  restores case and punctuation.
- **HF `pipeline` backends only.** Any `automatic-speech-recognition` pipeline
  model works via config alone; a non-pipeline backend (cloud STT, faster-whisper,
  a custom encoder) still needs a code-level plug point — tracked as future work.
- Multiple clips share one context window: the per-clip transcript budget is the
  context split across the request's clips, so many/long clips together are bound
  by `max_model_len` (see *Long audio & multiple clips* above).
- Chunk-merge de-duplication is text-level (word overlap at each seam); it can
  mis-handle a phrase legitimately repeated across a window boundary. A generative
  backend's internal timestamp stitching (`asr_self_chunks: true`) is more
  precise, but is unavailable for the CTC default — hence the wide 120s window,
  which leaves most clips seam-free.

## Audio + adapters

Audio requests route through adapters exactly like text requests. The model sets
`requires_raw_input_tokens = True` so vLLM passes the raw `input_ids` to the
forward pass on the multimodal path; the switch then detects adapter control
tokens as usual, and `embed_input_ids` applies the same token-exchange rewrite
(control → substitute id) used for text — so an audio request that activates an
adapter behaves identically to the text equivalent.

On a **pure sparse MoE** base the adapter surface is attention-only (`qkv_proj`,
`o_proj`), because there is no `shared_mlp` for the MLP-side groups to attach to —
see [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md#pure-sparse-moe-granitemoe). Where
no adapter library targets such a base yet, compose an adapter-free audio skin
with `--built-in-adapters base --enable-audio`: the marker, its output row and the
control-LUT sizing are all independent of how many adapters are present.

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
- `tests/integration/test_asr_ctc_default_gpu.py` (GPU, downloads the ~1GB
  checkpoint) — the default CTC model through `ASRTranscriber`: bfloat16 on CUDA,
  CTC classification, a correct transcript with client decode kwargs dropped, the
  float16/BatchNorm guard, and the 120s single-pass/chunked boundary.
- End-to-end (GPU): compose an `--enable-audio` checkpoint, then an audio request
  through vLLM produces an answer and text-only requests are unaffected.
