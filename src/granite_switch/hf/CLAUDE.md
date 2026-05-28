# CLAUDE.md — hf/

HuggingFace backend for training and debugging. Loaded automatically when reading any file under `src/granite_switch/hf/`.

## HF Attention Backends and Causal Masking

The eager backend does NOT handle `attention_mask=None` as causal — it treats `None` as no mask
(full attention). SDPA and FlashAttention handle `attention_mask=None` correctly via `is_causal`
attribute on the module.

The HF stress tests (`tests/hf/test_single_switch.py`) auto-detect which attention backends work
on the current platform by probing each with a k=-inf GQA call at import time. Unavailable
backends are skipped.

## Fused Projections (Not Bit-Exact with Upstream HF)

The GraniteSwitch HF backend uses fused QKV and gate-up projections, symmetric with the vLLM
backend architecture. Upstream HuggingFace `GraniteMoeHybridForCausalLM` uses separate
projections. Fused projections change the floating-point reduction order, so bit-exact skinning
equivalence with the upstream HF model is not achievable. The vLLM skinning equivalence tests
are the authoritative check — both the upstream and skinned models use the same fused-projection
architecture there. The HF skinning tests in `tests/composer/test_skinning_equivalence.py` are
skipped for this reason.
