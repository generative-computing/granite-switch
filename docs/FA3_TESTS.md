# FA3 FULL-CUDA-Graph Investigation: Full Writeup

A complete account of debugging issue #139 on an FA3-capable (Hopper/H100) box:
what was failing, why the first fix did not help, how the real cause was found, and
the fix that was applied. This complements
[FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md](FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md), which is the
canonical bug/fix reference; this document is the narrative of the debugging session,
kept so the reasoning (and the dead ends) are not lost.

- **Branch:** `bugfix/139-fa3-schedule-full-cudagraph-prefix-caching-fa3_fixing`
- **Environment:** NVIDIA H100 80GB (SM90), vLLM 0.26.0, torch 2.11.0+cu130, `FA_VERSION=3`
- **Target test:** `tests/vllm/test_fa3_schedule.py::test_fa3_full_cudagraph_prefix_caching_is_correct`

---

## Status on this branch: the FAP downgrade is in place and it works

This (fix) branch keeps the applied fix: `fa3_schedule.py` reports `UNIFORM_BATCH` for a
`granite_switch` model, so vLLM downgrades `cudagraph_mode=FULL → FULL_AND_PIECEWISE`,
attention runs eagerly, and the target test **passes** under `FA_VERSION=3`. Verified this
session: with the downgrade **reverted**, the test fails exactly as #139 originally reported
(`FA_VERSION=3`, 22 tokens outside the allowed set, teacher-forced JSD ≈ 0.87/0.78, max JSD
1.0) — so the downgrade is load-bearing, not cosmetic.

The downgrade is a **pragmatic mitigation**, not a root-cause fix: it keeps the mislabeled
batch (see §4a) out of the FULL graph rather than correcting how the batch is described. The
underlying defect is upstream in vLLM (`_is_uniform_decode` misclassifies a padded prefill as
a uniform-decode batch). Root-cause tracing — and the test of whether the upstream classifier
fix alone cures our FA3 symptom, or whether a second FA3-specific bug survives — is being done
on a separate drill-down branch (`experiment/139-...`), so this branch stays a clean, working
mitigation. The confirmed-so-far mechanism is recorded in §4 / §4a below.

---

## 0. Preliminaries: vLLM backends, kernels, and CUDA-graph modes

This writeup assumes the vocabulary below. If you already know what FA3 and
`cudagraph_mode=FULL` are, skip to the TL;DR; otherwise read this first, because the whole
bug is about the *interaction* of these knobs.

### 0.1 How vLLM runs a model: attention backend vs. the rest

vLLM splits a forward pass into two kinds of work:

- The **attention op** — the scaled dot-product over the KV cache. This is handed to a
  pluggable **attention backend** (FlashAttention, FlashInfer, Triton, xFormers, …). The
  backend owns the KV-cache layout (PagedAttention), the masking, and the actual GPU kernel
  that does the softmax-weighted sum.
- **Everything else** — embeddings, the Q/K/V and gate/up/down projections, LoRA, MLP,
  norms, the logits head. These are ordinary PyTorch/Triton ops in the model definition
  (for Granite Switch, the fused projections and the SWITCH LoRA kernel).

The two are configured independently. The attention *backend* decides how the dot-product
is computed; the *cudagraph mode* (below) decides how much of the whole forward pass is
frozen into a replayable CUDA graph. The #139 bug sits precisely at the seam between them.

### 0.2 FlashAttention 2 vs. 3 (FA2 / FA3), and `FA_VERSION`

**FlashAttention** is the default attention backend. It comes in two kernel generations:

- **FA2** — the portable generation; runs on Ampere (SM80, A100), Hopper, and later.
- **FA3** — a Hopper-specific (SM90, H100) rewrite that exploits the newer tensor-core and
  async-copy hardware for a faster attention kernel.

vLLM picks **FA3 only when both** hold (`vllm/v1/attention/backends/fa_utils.py`): the GPU
is **SM90/Hopper**, *and* the installed `vllm_flash_attn` ships a **loadable FA3 kernel**
for the running torch/CUDA build. Otherwise it silently falls back to FA2. The tests print
the selected generation as **`FA_VERSION=2`** or **`FA_VERSION=3`** at load — this is the
single most important line to check, because **the #139 bug only exists under FA3**, and an
FA2 run looks green while proving nothing (see §0.5).

### 0.3 CUDA graphs, and why vLLM uses them

Launching each GPU kernel from Python has fixed per-launch overhead. On a tiny decode step
(one token, a handful of kernels) that launch overhead can dominate the actual math. A
**CUDA graph** records a sequence of kernel launches once and then *replays* the whole
sequence with a single call, erasing the per-launch cost. The catch: a captured graph is
frozen — fixed shapes, fixed addresses — so it only applies to batch shapes that were
captured ahead of time, and anything shape-dependent baked in at capture time stays baked
in.

### 0.4 `cudagraph_mode`: how much of the pass is captured

This is the central knob. It controls **which parts of the forward pass go inside the CUDA
graph** — and in particular whether the *attention op* is captured or left to run eagerly:

| `cudagraph_mode` | What is captured | Where attention runs |
| --- | --- | --- |
| `eager` | nothing — every op launched normally | eagerly (no graph at all) |
| `PIECEWISE` | the model is split into graph pieces *around* attention | eagerly (outside the graph) |
| `FULL_AND_PIECEWISE` | the whole model **except** attention | eagerly (graph split around attn) |
| `FULL` | the **entire** forward pass, attention included | **inside** the captured graph |

`FULL` is the most aggressive: it captures attention too, so even the attention launch
overhead disappears — which is exactly why Granite Switch deploys with it (tiny decode
steps, latency-sensitive). `FULL_AND_PIECEWISE` is vLLM's default: it still graph-captures
the compute-heavy projection/MLP regions but deliberately *excludes* attention, running it
eagerly. The practical difference for this bug is one bit: **does the attention op run
inside the frozen graph (`FULL`) or outside it (every other mode)?**

### 0.5 `AttentionCGSupport`: the backend's veto over `FULL`

A backend is not obliged to let itself be FULL-captured. Each one advertises a
**`AttentionCGSupport`** level from `FlashAttentionMetadataBuilder.get_cudagraph_support`
that tells vLLM how much CUDA-graph capture its attention kernel can tolerate:

- **`ALWAYS`** — "capture my attention in a FULL graph for any batch, including mixed
  prefill+decode." **FA3 reports this.**
- **`UNIFORM_BATCH`** — "only safe for uniform-shaped (pure-decode) batches." **FA2 reports
  this.** When the backend reports `UNIFORM_BATCH`, vLLM **silently downgrades
  `FULL → FULL_AND_PIECEWISE`** (logging `CUDAGraphMode.FULL is not supported ... setting
  cudagraph_mode=FULL_AND_PIECEWISE`), dropping attention back out of the graph.

This is why **FA2 boxes never saw #139**: FA2's `UNIFORM_BATCH` makes `FULL` quietly become
`FULL_AND_PIECEWISE`, so attention runs eagerly (correct) regardless of what the deployment
asked for. Only FA3's `ALWAYS` honors `FULL` literally and puts attention in the graph —
which, on this vLLM 0.26 build, is where it computes wrong results. The fix (§5) is exactly
to make the backend report `UNIFORM_BATCH` for `granite_switch`, borrowing FA2's own veto.

### 0.6 Prefix caching

**Prefix caching** reuses the KV cache for a shared prompt prefix across requests instead
of recomputing it. The original #139 writeup suspected it was part of the trigger (the
production config runs with it on); the GPU reproduction showed it is **irrelevant** —
`FULL` diverges identically with prefix caching on or off. It is mentioned here only
because it appears in the test matrix and the original theory.

### 0.7 The AOT (ahead-of-time) schedule — a term from the *disproven* theory

FA3 precomputes a **work partition** ("AOT schedule") that tells the kernel how to split
attention work across the GPU, sized from the attention shape (`num_heads_q`,
`num_heads_kv`, `headdim`). The first #139 fix blamed a mis-sized AOT schedule (§3) and was
disproven on-GPU. The term recurs throughout §3–§4; it is **not** the cause, but you need
to know what it refers to to follow the narrative.

### 0.8 Shadow Residual's doubled query heads

Granite Switch's Shadow-Residual (SR) decoder builds its `Attention` with **doubled query
heads** (`Attention(2 * self.num_heads, ...)`, base-only K/V — one query stream for the
base, one for the adapter). The original theory tied the bug to the AOT schedule being
sized for the un-doubled head count; a plain Granite base (no doubling) reproducing the bug
is what killed that theory (§3.2d). See [SR_ARCHITECTURE.md](SR_ARCHITECTURE.md) for the SR
dual-stream design.

---

## TL;DR

- The test was failing because, under **FlashAttention 3 + `cudagraph_mode=FULL`**, the
  model's attention — *when captured inside the full CUDA graph* — computes grossly wrong
  results (teacher-forced JSD ≈ 0.78–0.87, out-of-set tokens), diverging from the eager
  reference at the **first** token.
- The **previous fix** (resize FA3's ahead-of-time schedule for the Shadow-Residual
  decoder's doubled query heads) was **designed from source-tracing and never run on a
  GPU**. On the GPU it was shown to be addressing a **non-cause**: the schedule was already
  sized correctly, disabling it changed nothing, prefix caching turned out irrelevant, and
  a plain Granite base (no doubled heads at all) reproduced the same divergence.
- The **real cause** (now confirmed by a live metadata dump on an H100, not just inferred —
  see §4a): this is an **upstream vLLM bug**, not a Granite Switch bug. vLLM's
  `_is_uniform_decode` (`gpu_model_runner.py`) classifies a batch by **shape alone**. Under
  `cudagraph_mode=FULL` vLLM **pads a prefill up to a captured graph size and describes it as
  a uniform-decode batch** where `num_reqs == num_actual_tokens` — with a `query_start_loc`
  (`[0, 49, 49, 49, …]`) that is internally inconsistent with that request count. FA3, which
  advertises `AttentionCGSupport.ALWAYS`, honors `FULL` and runs its captured attention on
  this **mislabeled padded batch**, computing over the wrong per-request layout → garbage
  tokens. Every mode that runs attention *outside* the FULL graph (`eager`, `PIECEWISE`,
  `FULL_AND_PIECEWISE`) describes the same prefill honestly (`num_reqs=1`,
  `num_actual_tokens=49`) and is correct.
- It is **not** "the AOT schedule is mis-sized for the doubled heads" (head counts are
  identical and correct in the broken and the correct paths) and **not** merely "attention
  is captured" (capture is the *trigger*; the *fault* is the mislabeled padded batch that the
  FULL path hands the kernel). The AOT schedule is built *from* the bad descriptor, so it is
  the conduit, not the root.
- **Upstream:** vllm-project/vllm **#53051** documents the identical `_is_uniform_decode`
  shape-only misclassification (from an unrelated hybrid-Mamba model), fixed by **#53059**
  (also require every request to be past its prompt, so aliased prefills fall back to
  PIECEWISE). The offending function is byte-identical in this 0.26 build. vLLM's own source
  comment (`gpu_model_runner.py`, near the capture dispatch) already flags that
  `num_tokens == num_reqs` "looks like a uniform decode batch to the dispatcher."
- The **applied fix**: for a `granite_switch` model only, report `UNIFORM_BATCH` (FA2's
  level) from `get_cudagraph_support`, so vLLM downgrades `FULL → FULL_AND_PIECEWISE`.
  Attention then runs eagerly on an honestly-described batch (correct); the rest of the model
  stays graph-captured. Other models keep FA3's native `ALWAYS`. This is **broader** than the
  upstream fix (it drops *all* attention out of the FULL graph for our model, not just the
  misclassified prefills); once this vLLM line carries #53059 the gate can be removed.

---

## 1. What the test does

`tests/vllm/test_fa3_schedule.py` composes a small Shadow-Residual (SR) MultiSwitch
checkpoint through `GraniteSwitchComposer` (CLAUDE.md gotcha #5) and runs it through vLLM
under several settings, each in a fresh subprocess (`_fa3_schedule_worker.py`) so only one
model is resident on the GPU at a time.

Two runs are compared:

| Run | Setting |
| --- | --- |
| **Reference** | `enforce_eager=True`, prefix caching **off** — no CUDA graphs; known-good (matches vLLM 0.19). |
| **Candidate** (the #139 repro) | `cudagraph_mode=FULL`, prefix caching **on** — the production serving config, chosen for latency. |

Scoring has two gates:

1. **Allowed-set gate.** Each greedily-decoded candidate token must be a token the
   reference also produced for that adapter. (Mirrors the production "token-0 KeyError":
   a token outside the adapter's allowed vocabulary subset.)
2. **Distribution gate.** Teacher-forced top-k distributions (JSD + Jaccard over
   `k ∈ {1,5,10,20}`) must match within a small floor. Both runs are teacher-forced over
   the **same** `prefix + ctrl + reference-continuation` sequence, so position *i* is
   directly comparable.

The whole test is gated behind `require_fa3`: it **SKIPs** unless `FA_VERSION == 3`,
because the bug only exists under FA3, and FA3 needs a Hopper GPU (compute 9.x). A green
result on a non-FA3 box would be meaningless — hence skip, not pass.

**Outcomes:** `SKIPPED` = no FA3 here (not applicable); `FAILED` = FA3 active and outputs
diverge (#139 present); `PASSED` = FA3 active and everything matches.

---

## 2. The observed failure

```
FA3 skip-guard: FA_VERSION=3 reason=''
  tokens outside the reference's allowed set: 15
  ctrl=310 emitted 13 token(s) the reference never produced (e.g. [114, 196, 22, 268, 301])
  ctrl=311 emitted  2 token(s) the reference never produced (e.g. [131, 224])
  ctrl=310 k=1: mean JSD=0.871507 > 0.003   max JSD=1.000000 > 0.05
  ctrl=311 k=1: mean JSD=0.779099 > 0.003   max JSD=1.000000 > 0.05
```

Two things about the magnitude matter:

- **JSD ≈ 0.78–0.87, max JSD = 1.0.** JSD here is in bits and bounded at 1.0; 1.0 means
  the two top-k distributions are *completely disjoint*. This is **not** a near-tie greedy
  argmax flip (the benign fused-vs-native noise the test was explicitly built to tolerate,
  CLAUDE.md gotcha #9). It is a wholesale divergence.
- It begins at the **very first token** of the sequence — including deep inside the shared
  prefix, before any adapter control token.

---

## 3. The previous fix, and why it did not help

### 3.1 What the previous fix believed

The committed `fa3_schedule.py` and the original doc carried this theory:

> FA3 computes an *ahead-of-time (AOT) schedule* — a work partition for the kernel — sized
> from the attention shape. vLLM's `FlashAttentionMetadataBuilder.__init__` reads that shape
> from the **model config** (`num_attention_heads`, head size, …). But SR's decoder builds
> its attention with **doubled query heads**:
>
> ```python
> # src/granite_switch/vllm/decoder/shadow_residual/decoder.py:180
> self.attn = Attention(2 * self.num_heads, ...)   # query heads DOUBLED, base-only K/V
> ```
>
> So the schedule is sized for 4 query heads when the real attention has 8. Under FULL the
> mis-sized schedule is frozen into the captured graph → wrong attention on small
> prefix-cached decodes.

Its fix: wrap `__init__`; for a single-shape granite_switch group, overwrite
`num_heads_q / num_heads_kv / headdim` with the group's real shape (read from the actual
`Attention` layers); for a genuinely mixed-shape group, disable AOT.

**Critically, this was designed by reading pinned vLLM source and never executed on a GPU.**
The original doc's own status table said *"The fix: Designed, not yet written"* and
*"not yet been reproduced on a GPU"*.

### 3.2 Four observations on the GPU that disproved it

With an FA3-capable H100 available, the path was instrumented and controlled experiments
run. Each knocks out a load-bearing part of the old theory.

**(a) The resize was working — and FULL still diverged.** Logging in the patch and in
vLLM's `build()` showed all three granite_switch attention groups detected and resized:

```
GSW-FA3-DEBUG group=[model.layers.*.self_attn.attn] shapes={(8,4,64)} config(nq=4,nkv=4,hd=64)   ← SR doubled-Q → resized to 8
GSW-FA3-DEBUG group=[switch.multi.0]                shapes={(1,1,32)} config(nq=4,...)             ← counting head → resized
GSW-FA3-DEBUG group=[switch.multi.1]                shapes={(1,1,64)} config(nq=4,...)             ← memory head  → resized
GSW-BUILD tag=model... nq=8 nkv=4 hd=64 ...   ← build() really used nq=8
```

`num_heads_q=8` was fed to `get_scheduler_metadata`. The schedule was correctly sized.
**Output still diverged, token-for-token identically to no patch at all.** If the schedule
shape were the cause, fixing it would have changed something. It changed nothing.

**(b) Disabling AOT entirely did not help.** Forcing the "mixed group" branch
(`aot_schedule=False`, `scheduler_metadata=None`, `max_num_splits=0`) for *all*
granite_switch groups still diverged. So it is not "the AOT schedule is wrong" — it is not
AOT at all: removing AOT leaves the bug, because attention is still *captured*.

**(c) Prefix caching is not the trigger.** The doc claimed FULL *and* prefix caching were
both needed. Running `FULL` + prefix **off** and comparing teacher-forced distributions to
the reference gave **57/64 and 52/64 diverging positions — identical to FULL + prefix on.**
Prefix caching is irrelevant; FULL alone does it.

**(d) A vanilla Granite base — no switch, no SR, no doubled heads — also diverges.** The
decisive one. Loading the plain Granite base directly:

```
VANILLA eager:  [139, 49,  317, 165, 303, 7, ...]
VANILLA FULL:   [139, 193, 74,  50,  165, 303, ...]   ← diverges at position 1
```

A model with 4 query heads and no doubling diverges the same way. A "doubled-Q schedule
mismatch" cannot explain a model that has no doubling. The doubled-Q theory is dead.

### 3.3 The clue the instrumentation surfaced

Comparing the *same* 49-token prefill under the two cudagraph modes:

| mode | what `build()` saw |
| --- | --- |
| `FULL_AND_PIECEWISE` (correct) | `num_actual_tokens=49, num_reqs=1, max_q=49` — a normal single-sequence prefill |
| `FULL` (broken) | `num_actual_tokens=56, num_reqs=56, max_q=49` — padded to 56 and described as **56 requests** |

Under FULL, vLLM pads the prefill up to a captured graph size (56) and describes it as a
uniform-decode-shaped batch where `num_reqs == num_actual_tokens`. The captured FA3
attention operates on this mis-described, padded batch, and the result is wrong. §4a
reproduces this descriptor deliberately and ties it to the exact vLLM classifier
(`_is_uniform_decode`) and upstream issue.

---

## 4. The real root cause

The only variable that flips the result is **whether the attention op runs inside the
captured CUDA graph**:

| vLLM 0.26 + FA3 setting | attention runs… | matches `(eager, prefix-off)` reference? |
| --- | --- | --- |
| `FULL`, prefix on | **inside** the graph | **No — diverges** |
| `FULL`, prefix off | **inside** the graph | **No — diverges (identically)** |
| `FULL_AND_PIECEWISE`, on/off | eagerly (graph split around attn) | Yes |
| `PIECEWISE`, on/off | eagerly | Yes |
| `eager`, on/off | eagerly | Yes |

The gate controlling capture is `FlashAttentionMetadataBuilder.get_cudagraph_support`:

```python
# vllm/v1/attention/backends/flash_attn.py:341
_cudagraph_support = (
    AttentionCGSupport.ALWAYS            # FA3 — "capture me in a FULL graph for any batch"
    if get_flash_attn_version() == 3
    else AttentionCGSupport.UNIFORM_BATCH  # FA2 — makes vLLM downgrade FULL → FULL_AND_PIECEWISE
)
```

FA3 advertises `ALWAYS`; `FULL` honors it and captures attention for mixed prefill-decode.
FA2 advertises `UNIFORM_BATCH`, which is exactly why **non-Hopper boxes never saw this
bug**: they fall back to FA2, `FULL` silently downgrades to `FULL_AND_PIECEWISE`, and the
correct (eager-attention) path runs. The bug surfaces only when FA3 is live and takes
`FULL` literally.

But "the captured path is numerically wrong" is a *localization*, not a mechanism — a
replayed CUDA graph runs the same kernel on the same inputs, so capture per se cannot
corrupt arithmetic. The §4a dump shows what actually goes wrong: capture changes the
**batch descriptor** the kernel is handed.

**Whose bug is it — vLLM's, not Granite Switch's.** The misclassification lives entirely in
vLLM's `_is_uniform_decode` (`gpu_model_runner.py`), a check that classifies a batch by
shape alone and never consults whether the requests are actually decoding. No Granite Switch
code is in that path; a plain Granite base hits it. It *bites* Granite Switch only because
(1) FA3's `ALWAYS` lets `FULL` capture attention at all, (2) Granite Switch deploys with
`FULL` for the per-launch latency on tiny decode steps, and (3) this vLLM 0.26 line predates
the upstream fix. Granite Switch carries the patch as a regulator on a defect it does not
own, until the vLLM dependency is bumped.

### Confirming the mechanism

Forcing the builder to report `UNIFORM_BATCH` (monkeypatch) made vanilla Granite under
`FULL` match the eager reference exactly:

```
WARNING ... CUDAGraphMode.FULL is not supported with FlashAttentionBackend backend
            (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE
VANILLA FULL (downgraded): [139, 49, 317, 165, 303, 7, 217, 6, 102, 128, 269, 88]   ← == eager reference
```

---

## 4a. Direct confirmation: dumping the batch descriptor on an H100

§3.3's `num_reqs=56` clue originally came from throwaway logging. It has since been
reproduced deliberately: `FlashAttentionMetadataBuilder.build` was wrapped to log the batch
descriptor for the **same 49-token prefill** under each cudagraph mode (engine run
in-process via `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the monkeypatch reaches the metadata
builder; the `granite_switch` downgrade patch was undone so `FULL` stays `FULL`). The real
prefill `build()` call (the one with `max_query_len=49`, not the capture-time warmup
dummies) showed:

| field | **`FULL`** (broken) | **`FULL_AND_PIECEWISE`** (correct) |
| --- | --- | --- |
| `num_reqs` | **56** | **1** |
| `num_actual_tokens` | **56** (49 padded + 7) | 49 |
| `max_query_len` | 49 | 49 |
| `query_start_loc[:10]` | `[0, 49, 49, 49, 49, …]` | `[0, 49]` |
| `num_reqs == num_actual_tokens`? | **True** (looks uniform-decode) | False |
| `aot_schedule` | True | True |
| `num_heads_q` / `headdim` | 4 / 64 | 4 / 64 |
| generated tokens | `[259, 127, 210, 134]` ✗ | `[143, 128, 15, 143]` ✓ (= eager ref) |

Two things this pins down that §3–§4 could only argue:

1. **The fault is the mislabeled padded batch, observed directly.** Under `FULL` the 49-token
   prefill is padded to 56 and described as **56 requests** of ~1 token each, yet
   `query_start_loc = [0, 49, 49, …]` packs all 49 real tokens into request 0 and gives the
   other 55 zero width. That descriptor is internally inconsistent: `num_reqs=56` tells FA3
   to iterate 56 request-slots over a cu_seqlens that only has one non-empty request. The
   kernel computes over the wrong per-request layout → `[259, 127, 210, 134]`.
2. **AOT / head count is *not* the variable.** `aot_schedule=True` and
   `num_heads_q=4, headdim=64` are **identical** in the broken and the correct run — only the
   batch descriptor (`num_reqs`, padding) differs. This is the experiment that cleanly
   separates "mislabeled batch" from "mis-sized AOT schedule," which the original
   disproof (§3.2) could not fully do because under FA3 `aot_schedule` and
   `use_full_cuda_graph` are welded together in `FlashAttentionMetadataBuilder.__init__`.

### Upstream: the same `_is_uniform_decode` bug

The exact classifier is `_is_uniform_decode` (`vllm/v1/worker/gpu_model_runner.py`), whose
condition in this 0.26 build is:

```python
return (
    (max_num_scheduled_tokens == uniform_decode_query_len)
    and (num_tokens == max_num_scheduled_tokens * num_reqs)   # 56 == 1 * 56  → True
)
```

A shape-only test: a padded prefill with `num_tokens == num_reqs` satisfies it and is routed
into the FULL graph. vLLM's own source comment at the capture dispatch already warns that
`num_tokens == num_reqs` "looks like a uniform decode batch to the dispatcher."

- **vllm-project/vllm#53051** reports the identical function misclassifying a fresh prefill
  as uniform decode (from an unrelated hybrid-Mamba/GDN model with MTP spec-decode, where the
  alias is `num_tokens == (1+k) * num_reqs`). Same root cause, a different kernel downstream.
- **vllm-project/vllm#53059** is the upstream fix: in addition to the shape match, require
  every request to be past its prompt (`num_computed_tokens >= num_prompt_tokens`), so aliased
  prefills fall back to the PIECEWISE path "where prefills belong anyway." A defense-in-depth
  suggestion in the thread: assert a FULL-graph replay never sees `num_prefills > 0`, turning
  silent corruption into a loud error.

This is why the applied fix (§5) is correct but *broader* than strictly necessary: reporting
`UNIFORM_BATCH` drops **all** attention out of the FULL graph for `granite_switch`, whereas
#53059 keeps FULL for genuine decodes and ejects only the misclassified prefills. Once this
vLLM line carries #53059, the `granite_switch` gate in `fa3_schedule.py` can be removed.

Reproduce with `scratch/fa3_metadata_probe.py` (gitignored; `--raw` undoes the downgrade so
`FULL` is honored and the mislabeling is visible):

```bash
WD=$(mktemp -d)
python tests/vllm/_fa3_schedule_worker.py build --work-dir $WD
python scratch/fa3_metadata_probe.py --work-dir $WD --cudagraph FULL --raw                # num_reqs=56, wrong tokens
python scratch/fa3_metadata_probe.py --work-dir $WD --cudagraph FULL_AND_PIECEWISE --raw  # num_reqs=1,  correct tokens
```

---

## 5. The applied fix

`src/granite_switch/vllm/fa3_schedule.py` now wraps `get_cudagraph_support` (not
`__init__`). For a `granite_switch` model **only**, it reports `UNIFORM_BATCH` instead of
FA3's `ALWAYS`:

```python
original = get_support.__func__          # the classmethod, unwrapped

def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
    hf = getattr(vllm_config.model_config, "hf_config", None)
    if getattr(hf, "model_type", None) == "granite_switch":
        return uniform_batch             # FULL -> FULL_AND_PIECEWISE
    return original(cls, vllm_config, kv_cache_spec)

builder.get_cudagraph_support = classmethod(get_cudagraph_support)
builder._granite_switch_cg_support = True
```

vLLM then downgrades on its own (logging `CUDAGraphMode.FULL is not supported ... setting
cudagraph_mode=FULL_AND_PIECEWISE`): attention drops out of the captured graph and runs
eagerly (correct), while the compute-heavy MLP/projection regions stay captured.

### Why this is the right locus (not a workaround)

- `get_cudagraph_support` is exactly vLLM's knob for a backend to say "do not capture my
  attention in a FULL graph." FA2 already uses it to the same end. We are telling vLLM the
  truth: for this model, on this build, FA3's FULL-captured attention is unusable.
- Gated on `model_type == "granite_switch"`, so every other model keeps FA3's native
  `ALWAYS` and its fully-fast FULL cudagraphs. (Verified: a `llama` config still returns
  `ALWAYS`; `granite_switch` returns `UNIFORM_BATCH`.)
- Fixes the problem at its real cause (attention capture), so it covers the SR decoder
  group, the counting/memory heads, and vanilla attention alike — not just the one shape
  the old patch targeted.

### Version safety (preserved from the original design intent)

- Soft-returns **without** marking `_PATCHED` when vLLM is genuinely absent (CPU-only /
  import error), so a later vLLM-capable process retries.
- Emits a loud `logger.warning` and does **not** mark `_PATCHED` if
  `FlashAttentionMetadataBuilder`, its `get_cudagraph_support`, or
  `AttentionCGSupport.UNIFORM_BATCH` has moved — a silent no-op there would quietly
  reintroduce wrong tokens.
- Idempotent via the `_granite_switch_cg_support` flag on the builder.

### The trade-off

`FULL` was chosen for the per-launch overhead savings on tiny decode steps. The downgrade
keeps everything *except* attention in the graph, so most of that saving survives; the cost
is the per-attention-launch overhead. That is the price of correctness until the upstream
FA3 full-cudagraph path is fixed on this vLLM line. Leaving `FULL` on and emitting wrong
tokens is not an option. The model-type gate can later be narrowed to the affected vLLM
versions if a future line is confirmed correct under FA3 + FULL (the bug lives in 0.26's
path, which changed by 0.30).

---

## 6. Verification

End-to-end, the SR model under `FULL` + prefix now matches the eager reference
**token-for-token** (16/16 on both adapters):

```
CTRL 310 fix:  [143, 128, 15, 143, 267, 28, 85, 15, 308, 13, 216, 299, 27, 223, 314, 314]
CTRL 310 ref:  [143, 128, 15, 143, 267, 28, 85, 15, 308, 13, 216, 299, 27, 223, 314, 314]   ✓
CTRL 311 fix:  [98, 15, 142, 123, 64, 4, 180, 292, 27, 291, 127, 309, 98, 23, 120, 267]
CTRL 311 ref:  [98, 15, 142, 123, 64, 4, 180, 292, 27, 291, 127, 309, 98, 23, 120, 267]     ✓
```

Both tests pass:

- `test_fa3_full_cudagraph_prefix_caching_is_correct` — `tokens outside the reference's
  allowed set: 0`, distributions match. **PASSED** (`FA_VERSION=3`).
- `test_fa3_config_matrix_matches_reference` — every `{FULL, FULL_AND_PIECEWISE, eager} ×
  {prefix on/off}` cell matches the reference. **PASSED**.

**CLAUDE.md gotcha #11:** the FULL-cudagraph test runs real graphs (not `enforce_eager`),
so it asserts on generated tokens / logprobs, never on the eager-only MultiSwitch
`_debug_*` attributes.

---

## 7. How to reproduce the diagnosis

All of the following require `FA_VERSION=3` (Hopper). On a non-FA3 box the tests SKIP and
prove nothing.

```bash
# 0. Confirm FA3 is live.
python tests/vllm/_fa3_schedule_worker.py probe        # expect FA_VERSION=3

# 1. The target test (red without the fix, green with it).
pytest tests/vllm/test_fa3_schedule.py::test_fa3_full_cudagraph_prefix_caching_is_correct -v -s --tb=short -x

# 2. Full matrix regression.
pytest tests/vllm/test_fa3_schedule.py -v -s --tb=short
```

To re-run the isolation experiments by hand (worker steps are standalone subprocesses):

```bash
WD=$(mktemp -d)
python tests/vllm/_fa3_schedule_worker.py build   --work-dir $WD
python tests/vllm/_fa3_schedule_worker.py run     --work-dir $WD --tag ref_eager_prefixoff --cudagraph eager --prefix off
python tests/vllm/_fa3_schedule_worker.py run     --work-dir $WD --tag full_on             --cudagraph FULL  --prefix on
python tests/vllm/_fa3_schedule_worker.py run     --work-dir $WD --tag full_off            --cudagraph FULL  --prefix off   # also diverges
python tests/vllm/_fa3_schedule_worker.py run     --work-dir $WD --tag fap_on              --cudagraph FULL_AND_PIECEWISE --prefix on  # matches
python tests/vllm/_fa3_schedule_worker.py compare --work-dir $WD --ref ref_eager_prefixoff --cand full_on --label "FULL vs eager"
```

To see the mislabeled batch descriptor directly (the §4a experiment):

```bash
python scratch/fa3_metadata_probe.py --work-dir $WD --cudagraph FULL --raw                # num_reqs=56 (mislabeled), wrong tokens
python scratch/fa3_metadata_probe.py --work-dir $WD --cudagraph FULL_AND_PIECEWISE --raw  # num_reqs=1,  correct tokens
```

Any throwaway diagnostic scripts belong in `scratch/` (gitignored), never in `tests/`
(CLAUDE.md: `tests/` is for curated regression tests only). `fa3_metadata_probe.py` lives in
`scratch/` for exactly this reason.

---

## 8. Files changed

| File | Change |
| --- | --- |
| `src/granite_switch/vllm/fa3_schedule.py` | Replaced the AOT-schedule-resize wrapper with a `get_cudagraph_support` downgrade (FA3 `ALWAYS` → `UNIFORM_BATCH` for `granite_switch`). New docstring records the confirmed mechanism and the disproven theory. |
| `src/granite_switch/vllm/__init__.py` | Updated the `register()` comment to describe the real fix (the call site is unchanged — still `patch_flash_attn_schedule()` first thing in `register()`). |
| `docs/FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md` | Rewritten: corrected root cause, added a "disproven diagnosis" section, updated the experiment table (reproduced on H100; prefix caching irrelevant; vanilla Granite reproduces), fix, and verification. |
| `docs/FA3_TESTS.md` | This document — the debugging narrative. §4a adds the direct batch-descriptor dump, the `_is_uniform_decode` tie-in, and the upstream #53051/#53059 linkage. |
| `scratch/fa3_metadata_probe.py` | Diagnostic (gitignored): wraps `FlashAttentionMetadataBuilder.build` to dump the batch descriptor per cudagraph mode. Not a regression test; kept for re-running the §4a comparison. |

---

## 9. Lessons

- **A source-traced theory that has never touched a GPU is a hypothesis, not a fix.** The
  original fix was plausible and internally consistent, yet the first real reproduction
  falsified it four independent ways.
- **Isolate the one variable — then dump it, don't just infer it.** The matrix
  (`{cudagraph mode} × {prefix on/off}`) plus a vanilla-model control pinned the cause to
  "attention captured in a FULL graph." But that was a *localization*; the direct
  descriptor dump (§4a) showed the actual fault is a **mislabeled padded batch**
  (`num_reqs == num_actual_tokens` with an inconsistent `query_start_loc`), with AOT and head
  count held identical across the broken and correct runs — the clean separation the
  four-way disproof could only approximate.
- **"The captured path is wrong" is where to look, not what is wrong.** A replayed graph runs
  the same kernel on the same inputs; capture can only change the *inputs*. Here it changes
  the batch descriptor. Chasing the descriptor, not the capture, found the real cause.
- **Know whose bug it is.** This is an upstream vLLM classifier bug (`_is_uniform_decode`,
  issue #53051, fix #53059), not a Granite Switch bug. The patch is a scoped regulator until
  the vLLM dependency carries the fix; frame it that way so it is removed, not kept forever.
- **Prefer the knob the framework already provides.** `get_cudagraph_support` is vLLM's own
  mechanism for exactly this ("my attention can't be FULL-captured"); FA2 uses it. The fix
  is a narrow, truthful declaration, gated to our model, not a monkeypatch of attention
  internals — though it is broader than the upstream fix, which keeps FULL for real decodes.
