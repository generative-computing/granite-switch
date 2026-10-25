# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [semantic versioning](https://semver.org/).

---

## [Unreleased]

### Fixed

- The default ASR model (`ibm-granite/granite-speech-5.0-470m-turboctc`) now loads
  at a pinned Hub revision, its English-only release. Unpinned, it had moved to a
  multilingual release on 2026-10-21 that capitalizes and punctuates transcripts.
  See `docs/AUDIO.md` for the `config.json` workaround for 0.1.1.

## [0.1.0] — MVP

Initial MVP release. API is not yet stable.
