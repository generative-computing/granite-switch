# CLAUDE.md — composer/

Compose system: builds Granite Switch checkpoints from a base model + LoRA adapters. Loaded
automatically when reading any file under `src/granite_switch/composer/`.

## End-to-End Tests Must Use Compose Infrastructure

No test should manually assemble `GraniteSwitchConfig` or call `transfer_base_weights` directly.
All model construction must go through `GraniteSwitchComposer` so that the compose pipeline
itself is what's being tested. If the composer can't handle a use case (e.g., zero-adapter
skinning), extend the composer — don't work around it in tests.

## Composing Models

```bash
python -m granite_switch.composer.compose_granite_switch \
  --adapters ibm-granite/granitelib-rag-r1.0
```
