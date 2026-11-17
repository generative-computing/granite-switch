# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [semantic versioning](https://semver.org/).

---

## [Unreleased]

### Breaking

- Audio checkpoints must name their ASR model. Serving no longer falls back to a
  built-in default when `asr_model_id` is null; it fails at startup with the fix.
  Checkpoints composed before this release need `asr_model_id` and
  `asr_model_revision` added to `config.json`, or a re-compose (see
  docs/AUDIO.md, *Migrating a checkpoint composed before 0.1.2*).
- Checkpoints composed with this release need this release to serve: older
  versions ignore the new ASR fields and load the ASR unpinned from the Hub.
- `DEFAULT_ASR_MODEL_ID` moved from `granite_switch.vllm.audio` to
  `granite_switch.config`. `ASRTranscriber` and `get_transcriber` now require
  `model_id`.
- Probe internals moved with the stream API: `granite_switch.probe_routing` is
  `granite_switch.streams.split` (`build_head_slot_lut`, `split_head_indices`);
  `SwitchedProbeHead` lives in `granite_switch.streams.types.probe.hf_head` and
  `.vllm_head`; the HF model's debug attribute `_last_probe_indices` is
  `_last_head_indices`. The derived config fields `probe_num_labels_per_slot`,
  `max_probe_labels` and `probe_control_token_ids` are gone; use
  `granite_switch.streams.routing` (`head_control_token_ids`, `max_head_outputs`).

### Added

- The stream API (`granite_switch.streams`, docs/STREAM_API.md). `config.json`
  holds one stream entry per slot (`streams`): an adapter, or a head stream such
  as the activation probe. A new head type plugs in with a `HeadType`, its
  weights and a config entry, registered through the `granite_switch.streams`
  entry-point group, with no change to Granite Switch. Heads write their scores
  into their marker's logits row (`select` or `merge`), read the base model,
  and take no LoRA rows.
  A stream entry with a type, mode or field this version doesn't know is refused.
  Checkpoints without `streams` load as before.
- Compose manifest keys for head streams: `kind`, `labels`, `layer`, `output`,
  `settings`.
- `--bundle-asr` composes the ASR model's weights into the checkpoint (`asr/`)
  for airgapped serving with no Hub access; `asr_bundled_path` in the config.
- `--asr-revision` and the `asr_model_revision` config field.

### Fixed

- Head plugins reject out-of-vocabulary output IDs, colliding module names,
  ambiguous manifest entries, and malformed score tensors before they can
  silently change results. The built-in probe rejects unsupported settings.
- Compose loads plugin weights under `torch.no_grad()`. vLLM loads head
  parameters and persistent buffers strictly, before decoder-specific mappings.
- vLLM carries head slot ids exactly from the last layer to the logits step. vLLM's
  MultiSwitch numbers slots in the model dtype, exactly only up to 256 in bf16, so a
  checkpoint with more slots is now refused at load instead of running slot 257 as
  slot 256 (fp16 holds 2,048).
- `Conversation` handles head turns: a head reads only its own turn (under
  `PRESERVE_MIXED_HISTORY`, a prompt whose kept history holds control tokens is
  rendered from text), and a recorded head turn joins the transcript as text, so
  its marker no longer stays inside later prompts. `completion_payload` asks for
  one token for a head turn.
- The default ASR model is pinned to its English-only Hub revision (`947f59a`).
  It loaded unpinned, so the Hub repo's 2026-10-21 multilingual release silently
  changed every transcript (now capitalized and punctuated) and broke the GPU
  tests.
- GPU ASR tests no longer assert transcript text, which is model quality, not
  something these tests own.

### Known limitations

- vLLM does not reject misplaced or repeated head markers in manually built
  prompts; a marker that is not last can silently lose the head's verdict.
  This affects every head type, not ordinary adapter-only requests. Use the
  composed template or `Conversation`.
  Request validation is intentionally deferred; see
  [the stream API limitation](docs/STREAM_API.md#known-gap-vllm-head-marker-validation-deferred).

## [0.1.0] — MVP

Initial MVP release. API is not yet stable.
