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

### Added

- `--bundle-asr` composes the ASR model's weights into the checkpoint (`asr/`)
  for airgapped serving with no Hub access; `asr_bundled_path` in the config.
- `--asr-revision` and the `asr_model_revision` config field.

### Fixed

- The default ASR model is pinned to its English-only Hub revision (`947f59a`).
  It loaded unpinned, so the Hub repo's 2026-10-21 multilingual release silently
  changed every transcript (now capitalized and punctuated) and broke the GPU
  tests.
- GPU ASR tests no longer assert transcript text, which is model quality, not
  something these tests own.

## [0.1.0] — MVP

Initial MVP release. API is not yet stable.
