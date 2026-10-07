# FA3 AOT Schedule Is Wrong Under FULL CUDA Graphs + Prefix Caching (vLLM 0.26)

## Summary

On vLLM 0.26, Granite Switch (Shadow-Residual / MultiSwitch checkpoints) emits wrong
tokens **only** when `cudagraph_mode=FULL` **and** prefix caching are both enabled.
Disabling either one makes outputs match vLLM 0.19 exactly.

- A token outside the adapter's allowed set (token 0) is produced, raising a `KeyError`
  downstream — observed on 99.8% of decisions in the failing environment.
- Even where no token is out of range, distributions diverge from the reference
  environment.

The fault is at the seam between Granite Switch's attention layers and FlashAttention 3's
ahead-of-time (AOT) scheduler. The fixable part is ours: the `fa3_schedule.py` patch,
which exists only on the vLLM 0.19 branch (`feature/doom-voice`) and must be
re-introduced, adapted, on the 0.26 line.

## Status at a glance

| Item | State |
| --- | --- |
| Root-cause mechanism | **Traced** from pinned vLLM 0.26 source + our attention shapes (see below). Not yet reproduced on a GPU. |
| Original experiment (the table below) | Observed by the team on an environment where **FlashAttention 3 was active**. |
| Our reproduction attempts on Vela | **Blocked**: FA3 does not load on the Vela image, so `FULL` silently downgrades and the bug cannot appear there. See *"Reproducing this requires FlashAttention 3"*. |
| The fix | **Designed, not yet written.** |
| Regression tests | **Written** (`tests/vllm/test_fa3_schedule.py` + worker), but can only pass/fail meaningfully on an FA3-capable box. |

**One sentence for a busy reader:** we understand *why* the bug happens and have a fix
designed, but we have not yet been able to re-run it on a GPU because the only trigger
condition — FlashAttention 3 driving a FULL CUDA graph — hasn't loaded in the
environments we've tried.

## The experiment

The same 12 batches of 1–5 adapters on one growing game history, under each setting:

| vLLM 0.26 setting                          | Outputs outside allowed set | Style adapters matching exactly (ref: 9) |
| ------------------------------------------ | --------------------------- | ---------------------------------------- |
| **ours:** `cudagraph_mode=FULL`, prefix caching on | **6 of 33**         | **4**                                    |
| `FULL_AND_PIECEWISE` (vLLM default)        | 0                           | 9                                        |
| eager (no CUDA graphs)                     | 0                           | 9                                        |
| prefix caching off                         | 0                           | 9                                        |
| async scheduling off                       | 6 of 33                     | 4                                        |
| old environment, vLLM 0.19 (reference)     | 0                           | 9                                        |

Read as an interaction test: turning off **either** FULL or prefix caching matches 0.19
exactly. Turning off async scheduling changes nothing, so async scheduling is **ruled
out**. The bug needs both FULL and prefix caching at once.

## Root cause (validated against pinned v0.26.0 source)

FlashAttention 3 computes an **ahead-of-time schedule** — a work partition handed to the
kernel — sized from the attention shape. vLLM's `FlashAttentionMetadataBuilder.__init__`
derives that shape from the **model config**, not from the group's real layers
(`vllm/v1/attention/backends/flash_attn.py:370-375`).

For Granite Switch the dominant mismatch is the **Shadow-Residual decoder group**.
`src/granite_switch/vllm/decoder/shadow_residual/decoder.py:138` builds:

```python
# Doubled query heads (base + adapter interleaved) against base-only K/V.
self.attn = Attention(
    2 * self.num_heads,   # query heads are DOUBLED
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    ...
)
```

Query heads are doubled, but `num_heads_q` comes from
`model_config.get_num_attention_heads()`, which reports the **un-doubled** count. So FA3's
schedule is built for **half** the real query heads. (`headdim` is already correct because
the repo overrides `get_head_size()` to `projection_head_dim`.)

### Why it is a single-shape group, not a mixed one

KV-cache groups are keyed by the frozen `KVCacheSpec`, whose `AttentionSpec` subtype
carries `head_size` and `num_kv_heads` (`vllm/v1/kv_cache_interface.py:99,175-178`;
grouping at `vllm/v1/core/kv_cache_utils.py:1205`). Layers with different `head_size`
therefore land in **separate, single-shape groups**:

- The switch counting head (`head_size=counting_head_dim`) and memory head
  (`head_size=memory_head_dim`) are in **different** single-shape groups
  (`src/granite_switch/vllm/switch/multi.py:285,299`), not one mixed builder.
- The dominant failure is a **single-shape group whose shape disagrees with the model
  config** (SR doubled-Q). This is the correction path, not the mixed-group path.

### Why FULL + prefix caching specifically

1. `__init__` sets `aot_schedule=True` (FA3 present, `flash_attn.py:379`); the
   `use_full_cuda_graph and aot_schedule` branch (`:400`) allocates a persistent
   `scheduler_metadata` buffer (sized from batch only — shape-independent) and forces
   `max_num_splits` to a config constant (`:418`).
2. `build()` reads `num_heads_q/kv/headdim` **live** and passes them to
   `get_scheduler_metadata` (`:518-520`); under FULL the result is frozen into the
   persistent buffer and reused by the captured graph (`:641-649`).
3. With a wrong `num_heads_q`, that frozen partition is invalid for a 1-token
   prefix-cached decode (`max_seqlen_q=1`, large `max_seqlen_k`, one request) → token-0 /
   out-of-set garbage.

Non-FULL recomputes a fresh, self-consistent split on every call
(`max_num_splits=0`, FA3 heuristics), so the wrong head count is largely self-correcting.
Fresh-prompt batches mask the bug because the prefill shape dominates the partition — only
the small prefix-cached decode exposes it.

> **Note.** An earlier writeup framed this as an init-ordering race where the patch flips
> `aot_schedule=False` "too late." That is not the cause on 0.26: the post-hoc disable is
> tolerated because `build()`'s write-back is guarded, so a stale buffer is simply never
> written. The real corruption is a **wrong-but-single shape kept with AOT enabled** in the
> SR group.

## Is this a bug in FlashAttention 3 or in Granite Switch?

- **Not FA3.** It sizes a schedule from the shape it is handed. Handed the model-config
  shape for a group whose real shape differs, it computes a schedule that does not fit —
  but it has no way to know the SR decoder doubles its query heads.
- **Granite Switch.** The fix is ours: size FA3's AOT schedule from the group's own layers.
  The `fa3_schedule.py` patch already does this on vLLM 0.19; it is absent on the 0.26 line
  and must be re-introduced and adapted.

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

**What we hit on Vela.** On an H100 (SM90 ✓), vLLM still selected FA2, so `FULL`
downgraded and the bug could not appear — on both vLLM 0.26 and 0.30. The cause was
**not** a torch version mismatch (a common wrong guess): vLLM 0.26 pins `torch==2.11`
and the pod had exactly that. The remaining suspect is the **compiled CUDA of the bundled
`vllm_flash_attn` kernel** versus the pod's CUDA — i.e. the FA3 kernel is present but will
not load for that CUDA build.

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

Re-introduce `fa3_schedule.py` on the 0.26 line, keeping the
wrap-`__init__`-after-original structure, with three changes:

1. **Single-shape group (`len(shapes) == 1`)** — overwrite
   `num_heads_q/num_heads_kv/headdim` from the group's own layers. This corrects the SR
   doubled-Q group and keeps AOT on (the FULL buffer is shape-independent, so it stays
   valid). This is the latency-preserving win.
2. **Genuinely mixed-shape group (`len(shapes) > 1`, rare)** — disable AOT
   **consistently**: `self.aot_schedule = False`, `self.scheduler_metadata = None`,
   `self.max_num_splits = 0`, so the builder state is self-consistent whatever path
   `build()` takes under FULL, regardless of future vLLM reordering.
3. **Guard** — bail out early when the group has no `Attention` layers or when
   `aot_schedule` is already False.

```python
layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
if not layers:
    return
shapes = {(a.num_heads, a.num_kv_heads, a.head_size) for a in layers.values()}
if len(shapes) == 1:
    self.num_heads_q, self.num_heads_kv, self.headdim = shapes.pop()
else:
    self.aot_schedule = False
    self.scheduler_metadata = None
    self.max_num_splits = 0
```

### Version safety

The current patch sets `_PATCHED = True` even on the `builder is None` branch, which hides
the case where a future vLLM renames or moves the builder — the patch would silently
install a no-op and corrupt tokens again. The 0.26 version must:

- soft-return without marking `_PATCHED` on a genuine `ImportError` (vLLM absent / CPU-only
  env);
- if vLLM is present but the expected symbol or attributes are missing, **not** mark
  `_PATCHED` and emit a loud `logger.warning` (prefer raising — a silent miss produces
  wrong tokens);
- after `original(...)`, assert the builder exposes `num_heads_q`, `num_heads_kv`,
  `headdim`, `aot_schedule`, and each layer exposes `num_heads`, `num_kv_heads`,
  `head_size`;
- optionally warn when `vllm.__version__` is outside `>=0.26,<0.31`.

### Branch

Create `fix/fa3-aot-schedule-0.26` off `feature/vllm-0.26-to-0.30` (pins `vllm>=0.26,<0.31`,
matches CI legs `vllm26`/`vllm30`). That branch currently lacks `fa3_schedule.py` and its
`register()` does not call the patch, so this reintroduces and wires the module adapted to
0.26 — not a merge of the 0.19 file.

## Files to modify

- `src/granite_switch/vllm/fa3_schedule.py` — **create** (0.26-adapted patch).
- `src/granite_switch/vllm/__init__.py` — call `patch_flash_attn_schedule()` first thing in
  `register()` (absent on the 0.26 branch).
- `tests/vllm/test_fa3_schedule.py` + `tests/vllm/_fa3_schedule_worker.py` — **create**
  (zero existing coverage), following the repo's `test_*.py` + `_*_worker.py` subprocess
  convention.

Reference only: `decoder/shadow_residual/decoder.py:138` (doubled-Q source),
`switch/multi.py:285,299` (counting/memory head shapes).

## Verification

vLLM is not installable in the local dev environment; GPU tests run via the `gsw-tests`
MCP tool.

**This requires an FA3-capable environment** (see *"Reproducing this requires
FlashAttention 3"*). The tests skip when `FA_VERSION != 3`, so the first thing to confirm
in any run is `FA_VERSION=3` in the log — otherwise the result is inconclusive.

**Reproduction / regression (authoritative):** serve a composed SR MultiSwitch checkpoint
on vLLM 0.26 with `enable_prefix_caching=True`, and run the toggle matrix
`{FULL, non-FULL} x {prefix cache on, off}`. Each setting is compared to an
`(eager, prefix-off)` reference run; the reference's own output defines each adapter's
"allowed set", and every setting is teacher-forced over the *same* token sequence so the
distributions are comparable:

- Fresh-prompt batch → passes pre- and post-fix.
- One game/conversation served alone, then a 1-token prefix-cached decode → **fails
  pre-fix** (tokens the reference never produced; distributions diverge), **passes
  post-fix**: tokens match the reference's allowed set and distributions match within the
  fused-vs-native floor.
- Only `FULL + prefix-on` should differ pre-fix; the other cells match the reference.
- Re-run the 12-batch experiment: expect 0 outputs outside the allowed set and 9/9 style
  adapters matching (vs 6/33 and 4/9 pre-fix).

**Patch unit test (vLLM-26 CI leg):** build a minimal `VllmConfig` with
`hf_config.model_type == "granite_switch"` and a group whose `Attention` layers report the
doubled SR `num_heads`; instantiate `FlashAttentionMetadataBuilder` with the patch
installed and assert:

- `builder.num_heads_q` equals the layer (doubled) value, not the model-config value;
- for a synthetic mixed group: `aot_schedule is False`, `scheduler_metadata is None`,
  `max_num_splits == 0`;
- idempotency: calling `patch_flash_attn_schedule()` twice leaves one wrapper;
- version-safety: with the builder symbol monkey-missing, the patch logs/raises and does
  not set `_PATCHED`.

**Note (CLAUDE.md gotcha #11):** the FULL-cudagraph GPU test runs real graphs (not
`enforce_eager`), so it cannot read the eager-only debug attributes — assert on generated
tokens/logprobs, not on `_debug_*`.

## Why it matters

FULL was chosen for latency: launch overhead dominates these tiny decode steps. On 0.19 a
single decision ran ~6 ms with FULL versus ~17 ms without; the patch's own 0.26 numbers put
a correct FULL decision at 7.8 ms p50 against FlashAttention 2's 23.6 ms. Falling back to
the vLLM default (`FULL_AND_PIECEWISE`) buys correctness at a real latency cost. This fix
keeps FULL both fast and correct.

## Caveat

The mechanism is traced from pinned v0.26.0 source plus the repo's attention shapes; it
has **not yet been reproduced on a GPU**, because the environments tried so far did not
load FlashAttention 3 (see *"Reproducing this requires FlashAttention 3"*). The
reproduction above (fails pre-fix, passes post-fix) is what will convert "traced and
consistent" into "confirmed" — once it runs with `FA_VERSION=3`.

Two things could still shift the picture when it does run:

- If the SR group turns out not to be the dominant trigger in practice, the per-group
  correction still fixes **any** single-shape group whose shape disagrees with the model
  config, so the fix holds — only the test's specific assertions would need adjusting.
- The bug lives in vLLM 0.26's metadata builder, which changed by 0.30. On an FA3-capable
  0.30 box the bug could already be absent upstream; a pass there would not disprove the
  0.26 problem. The test is therefore meaningful **per vLLM line**, and the demo's own
  line (0.26) is the one that matters.
