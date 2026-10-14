# FA3 Attention Is Wrong When Captured in a FULL CUDA Graph (vLLM 0.26)

## Summary

On vLLM 0.26 under FlashAttention **3**, Granite Switch emits wrong tokens whenever
`cudagraph_mode=FULL` is used. The output diverges — grossly, from the very first
prefill position — from the eager / `FULL_AND_PIECEWISE` reference.

- The divergence is in the **attention op as captured inside the full CUDA graph**.
  Running attention outside the graph fixes it: eager, `PIECEWISE`, and
  `FULL_AND_PIECEWISE` (which keeps the rest of the model in the graph but runs
  attention eagerly) all match the reference token-for-token.
- Prefix caching is **not** required to trigger it (an earlier writeup claimed FULL
  *and* prefix caching were both needed; a GPU reproduction showed FULL alone, with
  prefix caching off, diverges identically).
- It is **not** Granite-Switch-specific in mechanism: a plain Granite base served
  through this same vLLM 0.26 + FA3 build diverges under FULL in exactly the same way.
  It is an upstream FA3 full-cudagraph correctness problem on this vLLM line.

The fix is ours and narrow: for a `granite_switch` model we make the FlashAttention
backend report the FA2 cudagraph-support level (`UNIFORM_BATCH`) instead of FA3's
`ALWAYS`, so vLLM downgrades `FULL` to `FULL_AND_PIECEWISE` and runs attention eagerly.
See `src/granite_switch/vllm/fa3_schedule.py`.

## Status at a glance

| Item | State |
| --- | --- |
| Root-cause mechanism | **Reproduced and confirmed on an H100 (FA3 active).** The fault is captured FA3 attention under FULL, not the AOT schedule shape (see below). |
| Trigger | `cudagraph_mode=FULL` + FlashAttention 3. Prefix caching is irrelevant; a vanilla Granite base reproduces it too. |
| The fix | **Written and verified**: downgrade FA3 FULL -> FULL_AND_PIECEWISE for granite_switch (`fa3_schedule.py`). The #139 repro test passes with it, fails without it. |
| Regression tests | **Passing** on an FA3-capable box (`tests/vllm/test_fa3_schedule.py` + worker); they SKIP where FA3 is not active. |

**One sentence for a busy reader:** FA3 computes wrong attention when that attention is
captured inside a FULL CUDA graph on vLLM 0.26, so for Granite Switch we tell vLLM the
backend only supports `UNIFORM_BATCH` cudagraphs, which drops attention out of the graph
(FULL -> FULL_AND_PIECEWISE) and restores correct output.

## Correction to an earlier (disproven) diagnosis

An earlier version of this document, and the first `fa3_schedule.py` implementation,
attributed the bug to FA3's **ahead-of-time (AOT) schedule** being sized from the model
config's attention shape rather than the Shadow-Residual decoder's doubled query heads,
and "fixed" it by resizing `num_heads_q`/`num_heads_kv`/`headdim` on the metadata builder.
A GPU reproduction disproved this:

- With the resize in place the builder *was* sized correctly (`num_heads_q=8` for the SR
  group, the counting/memory heads' own `head_size`), confirmed by instrumenting
  `build()` — yet FULL still diverged, token-for-token identically to no patch at all.
- Disabling AOT entirely (`aot_schedule=False`, `scheduler_metadata=None`,
  `max_num_splits=0`) did **not** help either: attention was still captured in the graph.
- A plain Granite base (no switch, no SR, no doubled heads) diverges under FULL+FA3 too,
  so a doubled-Q schedule mismatch cannot be the cause.

The schedule shape was a red herring. What matters is only whether the attention op runs
inside or outside the FULL graph, which is governed by `get_cudagraph_support`.

## The experiment (reproduced on an H100)

Each setting below drives the SR MultiSwitch worker checkpoint
(`tests/vllm/_fa3_schedule_worker.py`) and is compared against the `(eager, prefix-off)`
reference. "Diverges" means the teacher-forced top-k distributions disagree grossly from
the first prefill position (JSD ≈ 0.8–1.0), with the greedy continuation out of the
reference's allowed set:

| vLLM 0.26 + FA3 setting                    | Matches `(eager, prefix-off)` reference? |
| ------------------------------------------ | ---------------------------------------- |
| `cudagraph_mode=FULL`, prefix caching on   | **No — diverges**                        |
| `cudagraph_mode=FULL`, prefix caching off  | **No — diverges (identically)**          |
| `FULL_AND_PIECEWISE` (vLLM default), on/off | Yes                                     |
| `PIECEWISE`, on/off                        | Yes                                      |
| eager (no CUDA graphs), on/off             | Yes                                      |

Read as an isolation test: the **only** variable that flips the result is whether
attention is captured inside the FULL graph (`FULL`) or run eagerly (every other mode,
including `FULL_AND_PIECEWISE`, which still captures the rest of the model). Prefix caching
makes no difference. A vanilla Granite base (no adapters) reproduces the same `FULL`
divergence, so nothing about the switch, the SR doubled-Q heads, or the counting/memory
heads is required.

## Root cause (confirmed on-GPU)

`FlashAttentionMetadataBuilder.get_cudagraph_support` reports
`AttentionCGSupport.ALWAYS` when FA3 is active and `UNIFORM_BATCH` otherwise
(`vllm/v1/attention/backends/flash_attn.py:341-345`). `ALWAYS` is what permits
`cudagraph_mode=FULL` to capture the attention op for mixed prefill-decode batches. On this
vLLM 0.26 + FA3 build that captured attention is numerically wrong — a single 49-token
prefill is padded into a uniform-decode-shaped graph (observed: `num_actual_tokens=56,
num_reqs=56` where `FULL_AND_PIECEWISE` sees the true `num_actual_tokens=49, num_reqs=1`)
and the attention output no longer matches eager.

The AOT schedule's `num_heads_q`/`headdim` are **not** the cause. Instrumenting `build()`
showed the schedule was already sized correctly (whether or not the old resize patch ran),
and disabling AOT altogether did not change the divergence — because the attention op was
still inside the captured graph either way.

## Is this a bug in FlashAttention 3 or in Granite Switch?

- **Primarily upstream (FA3 + vLLM 0.26 FULL path).** A plain Granite base diverges too,
  so the fault is in the generic captured-FA3-attention path, not in anything Granite
  Switch adds.
- **Ours to work around.** FULL is a Granite Switch deployment choice (latency of tiny
  decode steps), and the backend's broad FA3-`ALWAYS` promise is wrong for us on this line.
  We narrow it for `granite_switch` only, leaving every other model's FA3 FULL support
  intact.

## Reproducing this requires FlashAttention 3 (and that is the hard part)

The bug only exists under **FlashAttention 3**. This matters because it is easy to run
the whole thing and see *nothing wrong* — not because the bug is absent, but because the
trigger never fired. Here is the chain, in plain terms:

1. `cudagraph_mode=FULL` is what captures attention inside the CUDA graph (where the
   frozen, wrongly-sized schedule does its damage).
2. **FULL is only honored when the attention backend is FA3.** With FlashAttention 2,
   vLLM reports the backend only supports `UNIFORM_BATCH` cudagraphs and **silently
   downgrades `FULL` to `FULL_AND_PIECEWISE`** — the exact setting that is *correct* in
   the table above. So on FA2, you get correct output and learn nothing.
3. FA3 is selected only when **both** hold (`vllm/v1/attention/backends/fa_utils.py`):
   - the GPU is **SM90 / Hopper** (e.g. H100), and
   - the installed `vllm_flash_attn` actually has a **loadable FA3 kernel** for the
     running torch/CUDA. `is_fa_version_supported(3)` must return true; otherwise vLLM
     falls back to FA2.

**Where it was finally reproduced.** On an H100 (SM90) with vLLM 0.26, torch
2.11+cu130, FA3 loads (`FA_VERSION=3`) and the bug reproduces: `FULL` diverges, every
other mode matches. Earlier attempts on Vela had `FA_VERSION=2` (FA3 kernel present but
not loadable for that CUDA build), so `FULL` silently downgraded and nothing was tested —
which is why the regression tests skip, not pass, when FA3 is absent.

**How to tell, in one line.** The tests print the selected version at load:
```
FA_VERSION=2  (FULL cudagraph requires FA_VERSION=3)
```
If you see `FA_VERSION=2` (or the `CUDAGraphMode.FULL is not supported ... UNIFORM_BATCH`
warning), the run is **inconclusive** — fix the environment before trusting any result.
vLLM exposes `fa_version_unsupported_reason(3)` (in
`vllm.vllm_flash_attn.flash_attn_interface`), which returns the exact reason FA3 was
rejected; printing it is the fastest way to find what the environment is missing.

**Consequence for the tests.** Because a run without FA3 proves nothing, the regression
tests **skip** (rather than fail) when `FA_VERSION != 3`. That way a red result always
means a real regression, never just an environment that lacks FA3.

## The fix

`src/granite_switch/vllm/fa3_schedule.py` wraps
`FlashAttentionMetadataBuilder.get_cudagraph_support` so that, **for a `granite_switch`
model only**, it returns `AttentionCGSupport.UNIFORM_BATCH` instead of FA3's
`AttentionCGSupport.ALWAYS`. vLLM then downgrades `cudagraph_mode=FULL` to
`FULL_AND_PIECEWISE` on its own (logging `CUDAGraphMode.FULL is not supported ...
setting cudagraph_mode=FULL_AND_PIECEWISE`), which keeps the MLP/projection regions in the
graph but runs attention eagerly. Every other model keeps FA3's native support.

```python
original = get_support.__func__

def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
    hf = getattr(vllm_config.model_config, "hf_config", None)
    if getattr(hf, "model_type", None) == "granite_switch":
        return AttentionCGSupport.UNIFORM_BATCH   # FULL -> FULL_AND_PIECEWISE
    return original(cls, vllm_config, kv_cache_spec)

builder.get_cudagraph_support = classmethod(get_cudagraph_support)
```

It is wired as the first action of `register()` in
`src/granite_switch/vllm/__init__.py`.

### Version safety

- Soft-returns without marking `_PATCHED` on a genuine `ImportError` (vLLM absent /
  CPU-only env), so a later call in a vLLM-capable process retries.
- If `FlashAttentionMetadataBuilder`, its `get_cudagraph_support`, or
  `AttentionCGSupport.UNIFORM_BATCH` is missing (a vLLM rename/move), it emits a loud
  `logger.warning` and does **not** mark `_PATCHED` — a silent no-op would reintroduce
  wrong tokens under FA3 + FULL.
- Idempotent: a `_granite_switch_cg_support` flag on the builder guards re-wrapping.

## Files

- `src/granite_switch/vllm/fa3_schedule.py` — the `get_cudagraph_support` downgrade patch.
- `src/granite_switch/vllm/__init__.py` — calls `patch_flash_attn_schedule()` first thing
  in `register()`.
- `tests/vllm/test_fa3_schedule.py` + `tests/vllm/_fa3_schedule_worker.py` — the repro /
  regression tests (SKIP when FA3 is not active).

## Verification

GPU tests require an FA3-capable (Hopper) box; they SKIP when `FA_VERSION != 3`, so the
first thing to confirm in any run is `FA_VERSION=3` in the log — otherwise the result is
inconclusive.

**Reproduction / regression (authoritative):** `tests/vllm/test_fa3_schedule.py` composes
a small SR MultiSwitch checkpoint and runs the matrix
`{FULL, FULL_AND_PIECEWISE, eager} x {prefix on/off}` against an `(eager, prefix-off)`
reference. Each setting is teacher-forced over the *same* token sequence (the reference's
own output defines each adapter's "allowed set"):

- **Pre-fix:** `FULL` (prefix on *and* off) diverges — tokens the reference never
  produced, teacher-forced JSD ≈ 0.8–1.0 from the first prefill position. Every other cell
  matches.
- **Post-fix:** every cell, `FULL` included, matches the reference token-for-token
  (verified on an H100: both tests pass).

**Note (CLAUDE.md gotcha #11):** the FULL-cudagraph GPU test runs real graphs (not
`enforce_eager`), so it cannot read the eager-only debug attributes — it asserts on
generated tokens/logprobs, not on `_debug_*`.

## Why it matters

FULL was chosen for the launch-overhead savings on tiny decode steps. The downgrade to
`FULL_AND_PIECEWISE` keeps everything except the attention op in the CUDA graph, so most of
that saving is retained; the cost is the per-attention-launch overhead, which is the price
of correctness until the upstream FA3 full-cudagraph path is fixed on this vLLM line. The
alternative — leaving `FULL` on and getting wrong tokens — is not an option.

## Caveat

The bug lives in vLLM 0.26's FA3 full-cudagraph path, which changed by 0.30. On an
FA3-capable 0.30 box the upstream bug could already be absent; the patch is still safe
there (it only ever makes attention *eager*, never wrong), but if a future vLLM line is
confirmed correct under FA3 + FULL, the model-type gate can be narrowed to the affected
versions to recover full-graph attention. The regression test is meaningful **per vLLM
line**; keep it in the FA3-capable CI leg.
