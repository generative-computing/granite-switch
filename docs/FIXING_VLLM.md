# Fixing the vLLM test suite

Working notes on the non-passing vLLM tests. Source run analyzed:
`test-logs/20261012-125229/vllm.log` + `vllm.xml` (90 run · 3 failed · 12 errored
· 8 skipped). Branch: `bugfix/139-fa3-schedule-full-cudagraph-prefix-caching`.

The 15 non-passing tests fall into three independent clusters. Only cluster 1 is a
real code problem; clusters 2 and 3 are environmental on that run but not yet proven
harmless, so they stay open until a clean re-run clears them.

## Cluster 1 — FA3 schedule (REAL, this branch's work-in-progress)

Tests:
- `test_fa3_full_cudagraph_prefix_caching_is_correct`
- `test_fa3_config_matrix_matches_reference`
- the `test_suite` wrapper that drives them

Symptom: the candidate config (FULL cudagraph + prefix caching) diverges almost
totally from the `eager + no-prefix` reference.
- `ctrl=310` emitted 13 tokens the reference never produced; `ctrl=311` emitted 2.
- `max JSD = 1.000000` (threshold 0.05); `mean(1-Jaccard) ≈ 0.85–0.89` (threshold 0.2).

Interpretation: exactly the symptom bug #139 describes — FA3's ahead-of-time
scheduler is sized for the base model's head count, not the switch's own
(doubled-Q / extra) heads, so under FULL cudagraph + prefix caching the adapter
streams get garbage. Expected-red until the `fa3_schedule` patch (commits 2297e94,
82fce04, 984f40e, 5d262cd on this branch) is complete. This is the intended target
of the branch.

## Cluster 2 — Granite4 equivalence (REAL test-infra race; NOT external contention)

Tests: `_granite4_fullsize_tests.py` and `_granite4_mini_tests.py`
(`4.0-1b`, `4.0-350m`, `4.0-micro` across several suites).

Surface error: `RuntimeError: Engine core initialization failed`.
Real root cause:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
(cudaErrorDevicesUnavailable)
```

### Reproduced on a quiet node — so it is NOT external contention
First diagnosis (Oct 12 run) guessed "busy GPU, external." The re-run
(`test-logs/20261013-105827/`) **reproduced it with both GPUs idle (0 MiB, 0% util)
before and after**, deterministically, on back-to-back engines. So it is a real race
in the test harness, not another job on the card. `CUDA_VISIBLE_DEVICES=1,0` is set
in the environment.

### Exact failure point
The crash is the engine core's **first CUDA op**, in
`BuiltinLogitsProcessor.__init__` → `_device_tensor([], torch.int32)` →
`async_tensor_h2d(..., non_blocking=True)` → `t.to(device, non_blocking=True)`. A
trivial host→device copy fails, i.e. a fresh CUDA context cannot come up on the
device at that instant.

### Mechanism
`run_equivalence_integration` (`tests/shared/vllm_equivalence.py`) builds **two**
`vllm.LLM()` engines per test, back to back (upstream leg, then switch leg), via
`run_vllm_logprobs`. In the re-run log, the first engine inits fine
(`init engine ... took 11.08 s`) and shuts down cleanly (SIGTERM, teardown); the
**second** engine's EngineCore child then dies on that first h2d copy with
`cudaErrorDevicesUnavailable`. The `*WeightTransfer` variants (no engine) PASS.

### Why the existing mitigation does not catch it
`shutdown_llm` + `gpu_mem_util` (`tests/shared/vllm_gpu_mem.py`) wait for **memory**
to come back (`torch.cuda.mem_get_info()` free-fraction ≥ `GPU_MEM_RECLAIM_TARGET`).
That is a *memory* guard. This failure is a *device-availability* race: the log shows
the next engine starting with **78.67/79.18 GiB free** (memory fully reclaimed) and
still hitting `cudaErrorDevicesUnavailable`. Memory can be back while the device is
momentarily unavailable to a NEW CUDA context in the window right after an EngineCore
child exits. The memory wait is necessary but not sufficient.

### Candidate fixes (to decide)
1. After `shutdown_llm`, additionally poll device *readiness* — not just free memory —
   before starting the next engine: e.g. a tiny `torch.zeros(1, device=...)` /
   `torch.cuda.synchronize()` probe in a bounded retry loop, treating
   `cudaErrorDevicesUnavailable` as "wait and retry" rather than fatal.
2. Wrap the `LLM()` construction in `run_vllm_logprobs` in a bounded retry on
   `cudaErrorDevicesUnavailable` specifically (not a blanket retry).
3. Isolate each engine in its own subprocess (as the FA3/multi-adapter suites already
   do with worker scripts) so device teardown is a process exit, not an in-process
   context handoff. Heaviest change; most robust.

Option 1 or 2 is the smallest fix; 1 is preferable because it addresses the actual
precondition (device readiness) rather than papering over it with a catch.
Still open — not yet implemented.

### Evidence in `test-logs/20261012-125229/vllm.log`

The Oct 12 run log carries the full chain that pins this as a device-readiness race,
not a memory shortage:

- **Two engines back-to-back, second one dies.** The failing test's traceback
  (`vllm.log:970-991`) is `run_equivalence_integration` (`vllm_equivalence.py:369`)
  → `run_vllm_logprobs` (`vllm_equivalence.py:301`) → `LLM(` (`:301`→`llm.py:338`),
  i.e. the engine constructed for the second (switch) leg. The first (upstream) leg's
  EngineCore came up and shut down cleanly just before — e.g. pid=2079349 inits
  (`init engine ... took 14.75 s`, `vllm.log:918`) and tears down at 13:17:53
  (SIGTERM → "exiting busy loop", `vllm.log:929-932`).
- **The crash is the engine core's first CUDA op.** pid=2082537 dies in
  `BuiltinLogitsProcessor.__init__` (`builtin.py:175`) → `_device_tensor([], torch.int32)`
  (`builtin.py:227`) → `async_tensor_h2d` → `t.to(device, non_blocking=True)`
  (`torch_utils.py:586`) with `torch.AcceleratorError: CUDA ... busy or unavailable`
  / `cudaErrorDevicesUnavailable` (`vllm.log:1019-1030`). A trivial host→device copy —
  a fresh CUDA context simply cannot come up on the device at that instant.
- **Memory was fully reclaimed — so it is not a memory guard failure.** Every engine
  that *did* start logged **`Free memory on device (78.67/79.18 GiB) on startup`**
  (`vllm.log:227, 280, 448, 502, 579, 657, 710, 916`, …). The device had ~78.7/79.2 GiB
  free across the whole block; the failures are not OOM. The existing
  `shutdown_llm` + `gpu_mem_util` guard waits for exactly this memory to return, and it
  had returned — yet the next context still failed. Memory-free ≠ device-ready.
- **The no-engine variants pass.** Immediately after the `cudaErrorDevicesUnavailable`
  (`vllm.log:1039-1044`), the `*WeightTransfer` tests (which build no engine) PASS,
  confirming the failure is specific to standing up a *new* CUDA context, not to the
  model/weights.
- **Scope.** All of `_granite4_fullsize_tests.py` + `_granite4_mini_tests.py`
  (`4.0-1b/350m/micro` across the equivalence suites) fail this way:
  `RuntimeError: Engine core initialization failed` at
  `vllm.log:1013-1015, 1093-1094, 1166-1167, 1239-1240, 1312, 1384, 1456, 1528-1529,
  1601-1602, 1674-1675`, each with its own `cudaErrorDevicesUnavailable` in a distinct
  EngineCore pid (2082537, 2083884, 2085186, 2086494, 2087384, 2088235, 2089212,
  2090534, 2091863, 2093181).

This matches the quiet-node re-run (`test-logs/20261013-105827/`): memory back, device
momentarily unavailable to a *new* context in the window right after an EngineCore child
exits. Confirms candidate #1 (poll device readiness, not just free memory) is the fix.

### RESOLUTION (2026-10-13): two distinct causes, candidate #3 taken

The device-readiness/retry mitigations (candidates #1/#2) were committed but did
**not** fix it, because the diagnosis above was incomplete. Two separate causes:

**Cause A — the pytest process held a CUDA context.** The real precondition was
not "device momentarily unavailable after a child exits" but that the *pytest
process itself* created a primary CUDA context: `run_vllm_logprobs` called
`gpu_mem_util()` / `shutdown_llm()` (→ `torch.cuda.mem_get_info` / `empty_cache`)
**in-process**. Under Exclusive_Process, once the pytest process owns a context it
owns the device, so every `EngineCore` child it spawns afterward dies on its first
h2d copy — which is why the failures are at engine init and the no-engine
WeightTransfer tests always pass. (The old in-process retry also never fired: it
matched `"unavailable"/"busy"` in the parent exception string, but that string is
`"Engine core initialization failed..."`; the CUDA text is only in child stderr.)

Fix: **candidate #3**. Each engine now runs in `tests/shared/_vllm_logprobs_worker.py`,
one fresh process per engine. `run_vllm_logprobs` writes a JSON spec, shells out,
and `torch.load`s a CPU tensor back; it never touches `torch.cuda`. `gpu_mem_util`
/ `shutdown_llm` moved into the worker (the process that legitimately holds the
context). Verified: importing `vllm_equivalence` leaves `torch.cuda.is_initialized()`
False.

**Cause B — vLLM's non-eager init flakes under Exclusive_Process.** Isolating the
engines exposed a second, independent failure: a FRESH worker's EngineCore still
intermittently dies with `cudaErrorDevicesUnavailable` at the same h2d copy, but
ONLY on the non-eager path (`enforce_eager=False` → torch.compile + CUDA-graph
capture). `enforce_eager=True` is rock-solid. The bad window is time-correlated
(clusters right after a non-eager teardown), can persist 2-3 min across consecutive
fresh processes, and could not be reproduced on demand on a healthy device. Forcing
eager everywhere is not allowed: the adapter-tier-live tests
(`TestZeroAdapterNoHiding`, `TestZeroAdapterEquivalence`) require non-eager for the
`get_tolerances` 1e-5 gate (torch.compile normalizes the fused-SWITCH reduction
order; eager diverges ~1e-3).

Mitigations for Cause B, both in the worker/runner:
1. The worker pins `CUDA_VISIBLE_DEVICES` to a single device before importing
   torch/vllm (`_pin_single_visible_device`). Each worker is TP=1 so it needs one
   GPU; pinning removes vLLM's cross-device NCCL init step and halves the
   device-acquisition surface the race acts on. Opt out with `GS_LOGPROBS_NO_PIN=1`.
2. `run_vllm_logprobs` retries the transient `cudaErrorDevicesUnavailable` with a
   **fresh worker process** each attempt (`GS_LOGPROBS_WORKER_RETRIES`,
   `GS_LOGPROBS_WORKER_RETRY_DELAY`), keyed on the worker's full stdout+stderr (so
   it actually matches, unlike the old in-process retry). Only this error retries;
   anything else raises immediately.

### RESOLVED (2026-10-14): "Cause B" was a mis-diagnosis — it is Cause A, one level down

The "self-sustaining non-eager window" story below was **wrong**. There is no
non-eager-specific flake and no time-correlated window. Cause B is the *same bug as
Cause A*, recurring one process level down: the **worker parent process creates a
CUDA context before `LLM()` spawns its `EngineCore` child**, and under
Exclusive_Process the parent then owns the device so the child's first CUDA op dies
with `cudaErrorDevicesUnavailable`.

Where the parent context came from: `_vllm_logprobs_worker.cmd_run` calls
`gpu_mem_util()` **before** `LLM(...)`, and the old `gpu_mem_util` called
`torch.cuda.mem_get_info()` — which creates the worker's primary CUDA context
(`torch.cuda.is_initialized()` flips to True). The child then dies at
`gpu_worker.init_device()` → `MemorySnapshot(device=...)` →
`torch.accelerator.get_memory_info()`.

Minimal, deterministic repro on a fully idle Exclusive_Process GPU: a parent that
spawns a child which creates a CUDA context. Parent CUDA-free → child **OK**; parent
first ran `torch.cuda.mem_get_info()` → child **fails** `cudaErrorDevicesUnavailable`,
every time. Confirmed on the real worker: before the fix, BOTH eager and non-eager
failed; after the fix, both pass. Eager only ever "looked rock-solid" by timing luck
in standalone probes that happened not to call `gpu_mem_util` first — it was never
eager-vs-non-eager.

**THE FIX.** `tests/shared/vllm_gpu_mem.py::gpu_mem_util` now reads free/total memory
via `nvidia-smi` (`--query-gpu=memory.free,memory.total`, new `_nvsmi_free_total_bytes`)
instead of `torch.cuda.mem_get_info`, so the worker parent never creates a CUDA
context before the EngineCore child spawns. (`shutdown_llm`'s `torch.cuda` calls run
*after* generate and immediately before the worker exits, so a late parent context
there blocks nothing.) Verified `gpu_mem_util()` leaves `is_initialized()` False.

Consequences:
- The non-eager tolerance constraint is a non-issue: non-eager engines run fine once
  the parent is CUDA-free. **No eager-split and no recovery gap are needed** — the
  levers below were chasing a phantom.
- `run_vllm_logprobs` additionally round-robins the pinned GPU across engines/retries
  (`GS_LOGPROBS_PIN_DEVICE`, honoured by the worker's `_pin_single_visible_device`;
  `GS_LOGPROBS_NO_ROUND_ROBIN=1` to disable). This is secondary belt-and-suspanders:
  it keeps the two workers of a pair on different cards so a teardown and the next
  acquire never contend for one device. Not load-bearing; the CUDA-free parent is.
- The fresh-worker retry (candidate #2) is kept as-is: harmless, and still a backstop
  for any genuinely transient acquire failure.

### (superseded) Cause B hypothesised as SELF-SUSTAINING under engine cadence

Kept for the record; **the fix above supersedes this** — the mechanism described
here is not real (there is no non-eager window; it was the parent-context bug all
along). A full `tests/vllm/test_granite4_mini.py` run with pin+retry still failed
essentially every equivalence test, which was read as:

- A single non-eager engine recovers in ~1-2 min; the suite spawns the next
  immediately so the device "never recovers". **(Not actually what was happening.)**
- Could not be reproduced on demand on a healthy device. **(Because the trigger was
  `gpu_mem_util`'s context, present in the suite path, absent in the ad-hoc probes.)**

Levers once considered (now unnecessary): (a) a 60-90s recovery gap between engines;
(b) eager where bit-exact (num_adapters=0); (c) both.

## Cluster 3 — Multi-adapter equivalence (NOT the bug the committed fix targeted)

Tests: `test_multi_adapter_equivalence.py` for `granite-4.1-3b` (lora + sr), 4 tests
each → 12 errors. All errored **on setup**, in the `prepare` step:

```
subprocess.TimeoutExpired: Command '[... _multi_adapter_equivalence_worker.py
  prepare --flavor lora --base ibm-granite/granite-4.1-3b ...]' timed out after 1800 seconds
```

`PREPARE_TIMEOUT = 1800` (`tests/vllm/test_multi_adapter_equivalence.py`).

### What `prepare` actually does
`cmd_prepare` → `_prepare_synthetic` (`tests/vllm/_multi_adapter_equivalence_worker.py`):
1. `_local_base(base)` → `snapshot_download(base)` (full checkpoint, **no**
   `allow_patterns`).
2. `base_weight_norms(base_dir)` (`tests/shared/synthetic_adapters.py:201`) — one
   disk pass over the ~6.4 GB checkpoint, fp32 upcast + Frobenius norm per 2D
   `.layers.*` weight.
3. Synthesize the adapter set from fitted profiles; write `adapters.json`.

### Why the committed fix does NOT cover this
The fix in `_local_base` (removing `allow_patterns`) addressed a cold-cache
**crash** — a filtered download omitted `chat_template.jinja`, so `prepare` died with
"chat_template is not set". That fix is committed and clean. This run is a different
failure: a warm-cache **timeout**, not a crash.

### Evidence the download was NOT the bottleneck on this run
Both 3B checkpoints were fully present in `$HF_HOME/hub`
(`/proj/comodel/antonp/hf_cache`) **before** the run:
- `granite-4.1-3b`: weights 6.4 GB, blob mtimes **Oct 8 05:44**.
- `granite-4.2-3b`: weights 6.9 GB, blob mtimes **Oct 11 12:37**.
- (Initial measurement confusion: `stat`/`du` without `-L`/`readlink -f` reported
  tiny sizes because they measured the symlink entries, not the blob targets. With
  `readlink -f` the real safetensors are 4.99 GB + 1.81 GB etc. — complete.)

The Oct 12 run's logged activity spans only `13:05:36 → 13:34:04`, yet `prepare`
hit the full 1800 s. So on a warm cache the subprocess spent 30 min somewhere other
than the download. `_run_step` prints `result.stdout` only **after** the subprocess
returns, so a timeout prints nothing — we are blind to where it hung.

### Leading hypotheses (unproven)
1. Node-level resource contention: the same run was hitting `cudaErrorDevicesUnavailable`
   (cluster 2). Under heavy parallel GPU/memory/IO pressure the 6.4 GB disk pass +
   fp32 upcasts in `base_weight_norms` could stall badly.
2. A CUDA-init hang if some import in the `prepare` path initializes CUDA on a busy
   device (prepare is nominally CPU-only synthesis, so this is less likely).
3. A genuine hang in a synthesis helper.

### Next step to pin it down (not yet run)
Run `prepare` alone, directly, with timing, on a quiet node:

```bash
time /proj/comodel/antonp/venv/bin/python \
  tests/vllm/_multi_adapter_equivalence_worker.py \
  prepare --flavor lora --base ibm-granite/granite-4.1-3b --work-dir /tmp/ma_probe
```

Fast completion ⇒ the Oct 12 timeout was contention. A real hang ⇒ a bug in the
prepare path. Either way, if `prepare` is legitimately slow even when healthy, the
1800 s budget may need splitting (download vs. synthesis) or raising.

### Re-run outcome (`test-logs/20261013-105827/`, quiet node, warm cache)
Cluster 3 did **not** reproduce. The `prepare` step for `granite-4.1-3b-lora`
completed, and the `build`/compose step then produced a full 4-adapter checkpoint
(`syn-light`/`syn-strong`/`syn-medium`/`syn-weak`, ranks 16/64/32/16) with a compose
report. So the Oct 12 1800 s timeout was node-level contention (the same resource
pressure that drives cluster 2), not a logic bug or hang in the prepare path. The
committed `allow_patterns` fix remains correct for the cold-cache crash it targeted.

Caveat: the multi-adapter test bodies' later `run` steps spin up vLLM engines, so
they can still hit the cluster-2 device race. Watch those outcomes separately.

## Status

- Cluster 1: in progress on this branch (#139).
- Cluster 2: **RESOLVED (2026-10-14).** Not a device-availability race and not a
  non-eager flake — the worker *parent* created a CUDA context (via `gpu_mem_util` →
  `torch.cuda.mem_get_info`) before spawning its `EngineCore` child, which blocks the
  child under Exclusive_Process. Fix: `gpu_mem_util` reads memory via `nvidia-smi`, so
  the parent stays CUDA-free. See the "RESOLVED (2026-10-14)" subsection above.
- Cluster 3: resolved as environmental on the re-run — prepare + compose complete on
  a quiet node. No prepare-path change needed. (If a healthy prepare is still slow,
  consider splitting the 1800 s budget into download vs. synthesis.)
