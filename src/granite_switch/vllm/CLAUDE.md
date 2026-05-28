# CLAUDE.md — vllm/

vLLM backend for production inference. Loaded automatically when reading any file under `src/granite_switch/vllm/`.

## Adapter Index Convention (vLLM-specific)

Punica kernels use `-1` = no adapter. Internal conversion from the shared convention:
`adapter_indices - 1` (so the shared `0` = no adapter becomes `-1` for Punica).

## Known Limitation: TP Row-Parallel Bias Doubling

`SwitchedLoRALinear`'s row-parallel bypass path passes bias to all TP ranks instead of
suppressing it for rank > 0. After all-reduce this doubles the bias. Not affected: all Granite
architectures (4.0, 4.1) use `attention_bias=False` and `mlp_bias=False`.

## Deployment

```bash
# Verify plugin registration
python -c "from vllm.plugins import load_general_plugins; \
           from vllm import ModelRegistry; \
           load_general_plugins(); \
           print('OK' if 'GraniteSwitchForCausalLM' in ModelRegistry.get_supported_archs() else 'FAIL')"

# Start API server
python -m vllm.entrypoints.openai.api_server \
  --model ./granite-with-all-aloras \
  --port 8000
```
