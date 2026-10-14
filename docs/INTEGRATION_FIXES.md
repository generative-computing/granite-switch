# Integration-Suite Fixes

Summary of the integration-related fixes on
`bugfix/139-fa3-schedule-full-cudagraph-prefix-caching-fa3_fixing`. Every item here
is a consequence of one host fact: the shared GPU host runs its H100s in NVIDIA
**Exclusive_Process** compute mode. Only one OS process may hold a device's primary
CUDA context at a time; any other process that tries to acquire the same device dies
with `cudaErrorDevicesUnavailable` ("CUDA-capable device(s) is/are busy or
unavailable"). Crucially, `torch.cuda.empty_cache()` frees cached *memory* but **not**
the primary context — only the holding process exiting releases it.

For the deeper diagnosis behind these fixes, see
[docs/FIXING_VLLM.md](FIXING_VLLM.md) (clusters 1–3).

## 1. Deterministic engine teardown between parametrized checkpoints

**Files:** `tests/integration/conftest.py` (`shutdown_llm`),
`tests/integration/test_audio_serving_smoke.py`,
`tests/integration/test_audio_uncached_processor.py`

The two audio serving fixtures are `scope="module"` and parametrized over several
checkpoints. Their old teardown was `del llm; gc.collect(); torch.cuda.empty_cache()`.
That does not tear down vLLM's `EngineCore` **subprocess** — the engine only exits via
its own `atexit` hook (i.e. at pytest exit, not at a fixture boundary). So when the
fixture rebuilt for the next checkpoint, the previous engine was still holding the GPU,
and under Exclusive_Process the new `EngineCore` died with
`cudaErrorDevicesUnavailable`.

Fix: both fixtures now call `shutdown_llm(llm)`, which explicitly invokes the engine
core's `shutdown()` so the subprocess exits and hands the device back before the next
engine boots. It is best-effort and version-tolerant (the private shutdown path has
moved across vLLM releases; a missing attribute is a no-op, with the `atexit` hook as
backstop).

## 2. Keep the parent pytest process CUDA-context-free (integration collection policy)

**File:** `tests/integration/conftest.py` (`pytest_collection_modifyitems`,
`ISOLATED_MODULES`)

The vLLM-serving integration tests do their CUDA work in an `EngineCore` subprocess.
But two integration modules create a CUDA context **directly in the test process**
because they drive a HuggingFace/transformers model in-process:

- `test_adapter_routing_audio_enabled.py` — `.cuda()` on the HF model
- `test_asr_ctc_default_gpu.py` — `ASRTranscriber(device="cuda")`

Collected in-parent, either one poisons every vLLM-subprocess test ordered after it —
and alphabetical collection puts both ahead of the serving tests, which is exactly how
the suite went red.

Fix: this conftest reuses the proven isolation machinery from `tests/vllm/conftest.py`
(`_IsolatedModuleItem`, `_CHILD_ENV_FLAG`). Each listed module is fronted as a single
item that shells out to `pytest <module>` in a fresh subprocess, so its context lives
and dies inside that child and the parent stays clean regardless of collection order.
The isolated subprocess modules are ordered **first**, so each fully exits (releasing
the GPU) before any in-parent serving engine acquires it.

## 3. ASR driver-process device downgrade (cuda → cpu under Exclusive_Process)

**Files:** `src/granite_switch/vllm/audio/asr.py` (`gpu_is_exclusive_process`,
`resolve_driver_device`), `src/granite_switch/vllm/audio/processor.py` (`_asr_device`),
`tests/unit/test_asr.py` (`TestExclusiveProcessDetection`, `TestResolveDriverDevice`)

In vLLM serving, the audio multimodal preprocessing (the ASR transcriber) runs in the
**driver** process, while the `EngineCore` subprocess already owns the device. Under
Exclusive_Process the driver cannot acquire a second context on that device, so loading
the ASR model onto CUDA in the driver dies with `cudaErrorDevicesUnavailable` (and leaks
a partial parent context).

Fix: `_asr_device()` now routes the requested device through `resolve_driver_device()`,
which downgrades `"cuda"` → `"cpu"` only when a visible GPU is in Exclusive_Process mode.
The ASR encoder is small and runs on short clips, so CPU in the driver is a correct,
working fallback. Default compute mode and any explicit non-`"cuda"` device pass through
unchanged.

Detection is done via `nvidia-smi --query-gpu=index,compute_mode` **on purpose** — a
`torch.cuda` probe would create exactly the parent context we are trying to avoid. It
honours `CUDA_VISIBLE_DEVICES` and is conservative: any error (no `nvidia-smi`, no GPU,
parse failure) returns `False`, leaving the requested device unchanged.

## 4. Wait for device readiness, not just free memory

**File:** `tests/shared/vllm_gpu_mem.py` (`device_is_free`, `wait_device_free`,
`shutdown_llm`, `GPU_DEVICE_READY_TIMEOUT`)

Even after an `EngineCore` child releases its memory, it can briefly still hold the
device's primary context: the FIXING_VLLM cluster-2 evidence is a device reporting
~78.7/79.2 GiB free yet failing the *next* engine's first host→device copy with
`cudaErrorDevicesUnavailable`. **Memory free ≠ device acquirable.**

Fix: after the existing memory-reclaim wait, `shutdown_llm` now also calls
`wait_device_free`, which blocks (bounded by `GPU_DEVICE_READY_TIMEOUT`, default 60 s)
until no compute process holds a visible GPU per
`nvidia-smi --query-compute-apps`. Like the memory wait, it is queried through
`nvidia-smi` rather than a `torch.cuda` probe (same parent-context hazard), honours
`CUDA_VISIBLE_DEVICES`, and on timeout prints a warning and returns rather than raising —
the engine start itself remains the authoritative gate.

## 5. FA3 FULL-cudagraph correctness (branch's primary target, #139)

**Files:** `src/granite_switch/vllm/__init__.py`, `src/granite_switch/vllm/fa3_schedule.py`
(plus `docs/FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md`)

FlashAttention 3 computes wrong attention when captured inside a FULL CUDA graph on this
vLLM line, so under FULL cudagraph + prefix caching the adapter streams get garbage
tokens. The fix reports the FA2 cudagraph-support level (`UNIFORM_BATCH`) for a
`granite_switch` model, which downgrades FULL to `FULL_AND_PIECEWISE` — attention runs
eagerly and output matches the reference. This is the main work of the branch; the
integration-suite GPU fixes above are the supporting changes that let the suite run
cleanly on the Exclusive_Process host.

## Status

- Engine teardown, parent-process isolation, ASR device downgrade, and device-readiness
  wait (items 1–4): implemented on this branch.
- FA3 schedule (item 5): the branch's primary #139 target.
