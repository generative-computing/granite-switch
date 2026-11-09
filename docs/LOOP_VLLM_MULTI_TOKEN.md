# Loop Slot on vLLM: Emitting the Whole Answer in One Step

Status: implemented and served on vLLM 0.26.0 (A100), branch
`feature/loop-vllm-multi-token-emission-investigation`. Results are in section 8.
vLLM sources read: 0.26.0, checked against 0.30.0 (the two ends of the supported range).

## 1. The problem

A loop slot reads the base model's hidden states at the 81 grid cells of a
Sudoku prompt and solves the puzzle with a small looped transformer
(`hf/core/loop.py`). On the HF backend the answer is written into the 81
prompt rows of the logits. That works because one HF forward scores every
position.

vLLM does not work that way. In each engine step it:

1. runs the forward on the scheduled tokens;
2. computes logits only at `logits_indices`, which is one row per request
   (the last scheduled token);
3. samples **one** token per request;
4. hands that one token to the scheduler, which appends it and checks stop.

So 81 digits would normally take 81 steps:

```
step 0   prefill: prompt + <|sudoku|>   -> the head solves the grid here -> digit 1
step 1   decode:  digit 1               -> digit 2 ... but where is the answer now?
...
step 80  decode:  digit 80              -> digit 81
step 81  decode:  digit 81              -> EOS
```

The solve can only happen in step 0. It needs the read layer's hidden states
at the 81 grid cells, and they exist only in that pass. Later steps see only
K/V.

Requirements, as stated:

* Emit the 81-digit answer as ordinary output tokens.
* Keep it inside the model and engine on the host. No client logic and no
  CPU-side solving.
* Emitting all solution tokens at once matters more than KV cache.

## 2. What vLLM offers for "more than one token per step"

| Path | Multi-token? | Why it does or does not fit |
|---|---|---|
| Plain decode | no | one sampled token per request per step |
| Speculative decoding (V1 runner) | up to 1+k per step (k ≤ 128) | Drafts are proposed after a step and checked in the next one, so it takes 2 steps (prefill, then a verify forward over 82 tokens), not 1. Custom drafters (`custom_class`) get CPU token lists, not hidden states. Medusa gets hidden states, but only at the last position. Granite Switch already refuses `speculative_config` on probe checkpoints (probe packing assumes the ordinary sampler). |
| Prompt logprobs | scores, not tokens | a side output; the request's text stays empty |
| Logits processor holding per-request state | one token per step | Python state on the host plus a GPU→CPU copy of the answer; that is the "back to software" we want to avoid |
| **V2 model runner: `ModelState.custom_sampler()`** | **any number per step** | see below |

### The hook that makes it possible

vLLM's V2 GPU model runner (the default for dense models; our Vela logs
show "Using V2 Model Runner") lets a **model** supply its own `ModelState`
through `get_model_state_cls()`. That `ModelState` may replace the sampler
through `custom_sampler()`. DiffusionGemma already uses this hook to commit
a whole canvas of tokens in one step.

The rest of the V2 pipeline already carries a variable number of tokens per
request:

* **The sampler output** is `sampled_token_ids [num_reqs, W]` plus
  `num_sampled [num_reqs]` and `num_rejected [num_reqs]`.
* **`post_update`** (a Triton kernel) loops over `num_sampled`. It writes
  every token into the request's token history and sets the last one as the
  next step's input. It advances `num_computed_tokens` by
  `query_len − num_rejected`.
* **The output copy** trims each row to `num_sampled` before the scheduler
  sees it.
* **The scheduler** (`_update_request_with_output`) appends the tokens one
  at a time. It runs `check_stop` after each, so EOS, stop tokens and
  `max_tokens` cut the burst correctly.
* **Spec-decode accounting** is skipped, because nothing was "scheduled as
  draft".

Two things do assume one token per step, and the design handles both
(section 3.3):

* **The PP broadcast buffer** is sized by `num_speculative_steps + 1`.
* **The async scheduler's placeholder count** assumes one sampled token per
  request per step.

## 3. The solution: "answer burst" sampling

In the step whose input holds the loop marker, the compiled forward keeps
the grid's read-layer rows. The custom sampler then solves the grid on the
GPU and commits the whole answer for that request in the same step: 81
digits, then EOS.

```
step 0 (prefill, marker is the last prompt token)
  forward (compiled, CUDA graph):
    decoder runs; the read layer's rows are copied to a GPU buffer
  sampler (custom, GPU):
    no loop request in this step -> ordinary sampling (known on the host)
    otherwise: gather each loop request's 81 grid rows from the buffer,
               loop head solves them -> 81 digits + EOS per request
    loop rows:  sampled[r, :82] = answer, num_sampled = 82
    other rows: ordinary sampling,        num_sampled = 1
  scheduler: appends 82 tokens, stops at EOS   ->  request done, 1 step
```

Only one engine step is needed. There is no draft step, no verify step and
no copy of the answer to the CPU before output. The usual output copy takes
the tokens to the scheduler, as it does for every sampled token.

### 3.1 Where the answer comes from

The compiled forward does one thing for the loop slot: it keeps the read
layer. Everything else runs after the forward, eagerly, in the sampler.

* **In the forward: copy the read layer.** At the configured read layer
  (embeddings −1, a decoder layer, or the final norm), the forward copies
  the rows of every scheduled token into a persistent buffer:
  `loop_read[:T].copy_(h)`.
  * This uses the same hooks as the probe groups
    (`layer_output(state, lora_indices)`).
  * It is one copy, the size of the batch's hidden states. It has static
    shapes and no branch, so it is safe under CUDA graphs.
  * Buffer: `max_num_batched_tokens × hidden_size` (8,192 × 2,560 in bf16
    is 42 MB), one per distinct read layer, only in loop checkpoints.
* **On the host: which rows are loop requests.**
  * When a request is added, the model state records whether its prompt
    ends with a loop marker.
  * The batch says which requests are prefilling in this step
    (`is_prefilling_np`).
  * So the sampler knows the loop rows without reading anything back from
    the GPU.
* **Grid rows.** A loop request's last scheduled token is its marker, so
  its logits row `logits_indices[r]` is the marker's flat position `m`.
  * The grid is the 81 tokens that end `end_offset` tokens before it, at
    flat positions `m − end_offset − 81 .. m − end_offset − 1`.
  * They are valid only if they were scheduled in the **same forward**:
    `m − end_offset − 81 ≥ query_start_loc[r]`. Section 3.4 guarantees it.
* **Hand-off buffer.** `loop_read` has a stable address outside the graph
  pool and is written with `copy_`. This is the pattern DeepSeek-V4 uses
  (`_mtp_hidden_buffer`) to hand per-token state from the compiled forward
  to the code after it.
  * Packing the rows into extra hidden-state columns, as the probe does,
    would not work here: vLLM keeps only the last row per request.
  * There is no fixed limit on loop requests per step: the sampler gathers
    as many as the step holds.
* **Per-request solve.** The sampler solves the loop requests' grids as one
  batch, and each row stops at its own convergence. The HF rule waits for
  the whole batch.
  * The freeze is on the GPU: a converged row's state stops changing (the
    head's `step` already takes an `active` mask). One request's answer
    then does not depend on the others, except for bf16 batch-size
    numerics, which vLLM has everywhere.
  * The loop stops when no row is active. It checks every 50 iterations,
    which keeps host syncs few.
  * `max_iter` stays as configured (35,000). HF iteration counts: puzzle
    #2000 took 74–2,177 iterations over 8 seeds.
  * Per-request halting may score slightly differently from the HF batch
    rule; the GPU test measures it.
* **Deterministic start.** FPRM starts from a random truncated normal. In
  vLLM that would make an answer depend on the batch, the step and the
  preemption history. We draw the start once at load time from a fixed
  seed: a buffer `z0 [prefix+81, W]` that every request shares. One seed is
  as accurate as another: the held-out eval used fresh random starts and
  gave 77.5%.
* **Answer.** `argmax(head_logits)` mapped through `output_token_ids` gives
  81 token ids. The request's EOS id, recorded when the request was added,
  makes 82.

### 3.2 The sampler (outside the graph, on the GPU)

`LoopBurstSampler` wraps the default V2 `Sampler`. It delegates every other
attribute to the wrapped sampler: the runner also calls `add_request`,
`apply_staged_writes` and the penalties state.

1. Run the default sampler: one token per request, as usual. This keeps
   temperature, top-p, logprobs and penalties for every other request.
2. If no row is a loop request in its prefill (known on the host), return
   that output unchanged.
3. Otherwise gather the loop rows' grids from `loop_read`, solve them with
   the loop head, and map each answer to tokens.
4. Build `sampled_token_ids [num_reqs, W]`:
   * Loop rows get the 81 digits plus EOS, and `num_sampled = 82`.
   * Other rows get the default token in column 0, and `num_sampled = 1`
     (0 for chunked-prefill rows, as before).
   * `num_rejected` stays 0 on every row. The runner advances a request's
     computed-token count by `query_len − num_rejected`, which is the
     prefill length here, so the counter stays right.
     * Computing `num_rejected` as `num_logits − num_sampled`, as the
       rejection sampler does, would give −81 on a loop row. That would
       count 81 tokens as computed that have no K/V.
     * It does not matter for a loop request, which ends in this step. It
       would matter if a burst were ever truncated without EOS.
   * The burst tokens never get K/V: the request finishes at EOS before any
     forward would read them. That is fine, because KV cache is the lower
     priority.
5. Logprobs: the loop rows' logprob tensors keep their single row. The
   scheduler slices `len(new_token_ids)` rows per request, so loop requests
   that ask for logprobs must be rejected or padded. The plan rejects them
   when the request is added (section 5).

Apart from the solve, everything here is a few `torch.where`/`gather` ops
on GPU tensors. The answer never goes to the CPU before the usual output
copy.

### 3.3 Wiring: the model brings its own state

* `GraniteSwitchForCausalLM.get_model_state_cls()` returns
  `LoopSlotModelState`. That is a subclass of vLLM's `DefaultModelState`
  with two changes:
  * `custom_sampler()` returns `(LoopBurstSampler(sampler, model), None)`;
  * `add_request()` checks loop-request preconditions (section 5).
* The class is chosen only when the checkpoint has a loop slot. Otherwise
  the model keeps the default state.
* **No speculative or diffusion config.** It is tempting to widen V2's
  output through `num_speculative_tokens` (or `diffusion_config`). That is
  wrong: the scheduler reads the same number and would schedule up to 81
  placeholder "draft" tokens on every decode step. Our sampler never needs
  them.
* **Nothing in the output path is width-limited**, so we leave the number
  alone:
  * the output copy moves the sampler's `[num_reqs, W]` matrix whatever its
    width;
  * `post_update` loops over `num_sampled`;
  * the scheduler appends each request's list token by token.
* **The one width-limited buffer** is the PP broadcast
  (`num_speculative_steps + 1`), and PP is out of scope (section 5).
* **Runner and scheduling mode.**
  * The V2 runner is already the default for dense Granite Switch
    checkpoints; the Vela vLLM logs say "Using V2 Model Runner".
  * Some settings select V1 instead: stock torch.compile, ngram
    speculation, and MoE bases, which default to V1 (vLLM's env override
    `VLLM_USE_V2_MODEL_RUNNER=1` selects V2; untested here). On V1 the hook
    is never called and the loop slot would quietly emit ordinary tokens.
  * So the config hook below checks `vllm_config.use_v2_model_runner` for
    loop checkpoints and raises a clear error otherwise.
* **Async scheduling** (on by default) is not compatible as is.
  * The async scheduler adds one output placeholder per request per step,
    then subtracts `len(new_tokens)` when the step's output arrives.
  * A burst of 82 would go below zero and trip an assert. Every loop burst
    ends in EOS, but the subtraction happens before the request is freed.
  * So the first version turns async scheduling off for loop checkpoints,
    through a `VerifyAndUpdateConfig` hook for `GraniteSwitchForCausalLM`.
    vLLM's own per-architecture hook runs before vLLM settles async mode.
  * A later version could keep async on by adding the burst width to the
    placeholder count.

### 3.4 Making the read valid: the whole grid in one forward

The answer needs the 81 grid rows and the marker in the same forward.

| Hazard | Effect | Handling |
|---|---|---|
| Chunked prefill splits the prompt | grid rows are in an earlier step | Loop checkpoints set `enable_chunked_prefill = False` (scheduler config, in the same hook). The prompt (~110 tokens) is then always scheduled whole. vLLM then needs `max_num_batched_tokens ≥ max_model_len`, or longer prompts never schedule, so the hook raises the budget to match. |
| Prefix-cache hit on the grid | grid rows are not recomputed, only their K/V exists | Prefix caching off for loop checkpoints (same hook). Lighter alternative: vLLM's per-request `skip_reading_prefix_cache`. It defaults on only for prompt-logprob requests, so a loop request would have to set it, and the first version does not rely on that. |
| Preemption (recompute) | the prompt plus earlier output is prefilled again; the marker is no longer last | The answer is committed in step 0, together with EOS, so a loop request never decodes and is never preempted mid-answer. A request preempted **before** its step 0 is simply re-run; the fixed start makes the answer the same. |
| Grid fewer than 81 tokens | gather would cross into another request | Checked on the host when the request is added: a loop request needs `end_offset + 81` tokens before its marker, or it is rejected. |

### 3.5 Why this is novel

* Speculative decoding emits many tokens per step only after a
  draft-and-verify round trip, and only for tokens a draft model guessed.
* Diffusion LMs emit a canvas only after many denoising steps.
* Here the **model itself** computes a complete, final multi-token answer
  from one forward's hidden states, and the engine commits it in that same
  step through the sampler hook. No draft step, no verify step, no
  acceptance test: the answer is final by construction, because the head
  that produced it is the model.

As far as we know, no vLLM model does "solve in the prefill, emit the
answer as a token burst in the same step". DiffusionGemma is the closest
precedent, and it uses the hook for a different reason: committing a
converged canvas.

## 4. Alternatives considered

1. **Oracle self-speculation.** Use spec decoding with k = 81: the loop head
   drafts its answer, and the target's verify logits are forced to agree.
   * It works on the V1 runner, but the drafts can only be proposed after
     the prefill's bookkeeping. So it takes 2 steps: the prefill, then a
     verify forward over all 82 tokens.
   * It also needs:
     * a hidden-state path into a custom drafter (vLLM only gives
       `custom_class` CPU token lists);
     * forced verify logits at 81 positions;
     * the probe slot's `speculative_config` guard lifted.
   * Async scheduling is disabled for custom drafters.
   * Twice the steps of the burst sampler, an extra 82-token forward, and
     more engine changes.
2. **KV-cache state (the MultiSwitch way).** Write the answer into a
   loop-cache attention layer at the marker. Each decode step reads digit k
   back by attention and forces it.
   * Fully "inside the model", and it works on any runner.
   * It still takes 81 steps, one per digit. The user ranked emitting all
     tokens at once above KV-cache elegance.
   * It stays the fallback if the V2 hook is ever removed.
3. **Logits processor with per-request state.** Rejected: host-side Python
   state, plus a GPU→CPU copy per request, and still 81 steps.

## 5. Limits and guards

* **Greedy-only answer.** Temperature or top-p do not apply to the 81
  digits; the head's argmax is the answer. Other requests in the same batch
  sample normally.
* **Logprobs** for loop requests: rejected at `add_request` in the first
  version.
* **Structured output and grammars** for loop requests: rejected; the
  scheduler would feed 82 tokens to a grammar.
* **`max_tokens < 82`**: `check_stop` cuts the burst and the answer is
  truncated, which is correct behavior. `max_tokens` is the only way a
  burst can end before EOS. A request cut that way is also finished, so the
  burst never leaves a running request whose last 81 tokens lack K/V.
* **Room for the burst.** The scheduler reserves room for one sampled
  token, not 82.
  * `post_update` writes every committed token into a per-request history
    of `max_model_len` slots, with no bounds mask.
  * So the sampler clamps each burst to
    `min(82, prompt_len + max_tokens − total_len)`. `max_tokens` already
    fits within `max_model_len`.
  * This keeps that write in bounds and makes the GPU history agree with
    what the scheduler keeps.
* **`ignore_eos` or `min_tokens > 82`** would let a loop request keep
  decoding after the burst, on tokens without K/V. Rejected at
  `add_request`.
* **Pipeline parallelism**: the PP broadcast sends
  `[num_reqs, num_speculative_steps + 1]`, which is 1 column here, too
  narrow for a burst. The read layer would also have to be on the last
  rank. The first version supports PP = 1 only and raises otherwise.
* **Tensor parallelism**: the head is small (width 512) and runs replicated
  on every rank on identical inputs. The answer is identical, and only the
  last rank's sampler uses it.
* **One loop request = one answer.** A loop request is a single-turn "solve
  this grid" call. Multi-turn loop chats are out of scope.

## 6. Cost

* The solve runs once per loop request, in the step that prefills it.
  * Each iteration is a 2-layer, width-512 block over 97 tokens (16 prefix
    + 81 grid), on all of the step's loop requests at once.
  * Iterations until convergence: typically hundreds to a few thousand
    (puzzle #2000: 74–2,177).
* The time per iteration on an A100 is still to be measured. It is the
  number that decides the prefill latency of a loop request.
* **Steps without a loop request do not pay for the solve.**
  * A captured CUDA graph cannot branch, so the solve cannot sit inside the
    graph.
  * Instead the compiled forward only copies the read layer into a
    persistent buffer, which is cheap and safe to capture.
  * The solve runs after the forward, in the sampler, and only on steps
    that hold a loop prefill. The host already knows which steps those
    are.
  * Every step pays for the copy, one hidden-state-sized `copy_`, but only
    in checkpoints with a loop slot.
* Decoding is 1 step instead of 82.

## 7. Plan

1. **vLLM loop head.** Reuse the HF `LoopHead` module as is. It is plain
   PyTorch and runs eagerly in the sampler, so no port is needed.
   * Weights: `loop_heads.heads.<slot>.*`.
   * Keep the head in fp32 with bf16 autocast for the solve, as HF does
     (`_keep_in_fp32_modules_strict`). vLLM would otherwise cast it to the
     model dtype.
   * Add a fixed start buffer and a per-request convergence exit.
2. **Model wiring** (`vllm/granite_switch_model.py`):
   * copy the read layer into the hand-off buffer inside the compiled
     forward;
   * load the head weights (today they are silently skipped).
3. **Burst sampler + model state** (`vllm/loop_burst.py`):
   * `LoopSlotModelState`: records which requests are loop requests, and
     each one's EOS id;
   * `LoopBurstSampler`: grid gather, solve, burst;
   * the request guards.
4. **Config hook** (`vllm/__init__.py`): register `VerifyAndUpdateConfig`
   for loop checkpoints. It:
   * checks that the V2 runner is selected;
   * turns off async scheduling (first version);
   * turns off chunked prefill, with a token budget of at least
     `max_model_len`;
   * turns off prefix caching.

   It does **not** set any speculative or diffusion width.
5. **Tests**:
   * CPU: the sampler-merge logic (`num_sampled`/`num_rejected` per row);
     the per-request solve vs the HF solve with the same start.
   * GPU on Vela: a composed Sudoku checkpoint in `vllm.LLM`, eager and
     CUDA-graph, mixed batches (loop + ordinary prompts). Check that the
     answer equals the HF answer with the same start, comes out in one
     step, and that the held-out accuracy matches.
6. **Benchmark on Vela**: throughput and latency, burst vs 82-step
   decoding.

## 8. Results (vLLM 0.26.0, one A100 80 GB)

The trained Sudoku head (the 77.5% head of the HF evaluation) was composed
onto `granite-4.1-3b` and served with `vllm.LLM`
(`tutorials/scripts/loop_sudoku/serve_vllm.py`, `run_vllm_eval.sh`).

| Run | Puzzles | Answer as one burst | Exact | Cells right | Blank cells right |
|---|---|---|---|---|---|
| Held-out test, CUDA graphs | 20,000 | **100%** | **75.27%** | 90.5% | 86.3% |
| Smoke, CUDA graphs | 64 | 100% | 76.6% | 92.0% | 88.3% |
| Smoke, eager | 64 | 100% | 79.7% | 92.6% | 89.2% |
| HF backend, same head (random starts) | 20,000 | (81 logit rows) | 77.5% | 91.2% | |

* **One burst.** Every loop request finished in its prefill step with 82
  tokens: the 81 digits, then EOS, finish reason `stop`.
* **Other requests are unaffected.** Two ordinary chat prompts in the same
  batch decoded normally ("The capital of France is Paris.").
* **The engine settings applied as designed**: V2 model runner, async
  scheduling off, chunked prefill off, prefix caching off.
* **Agreement with HF.** 14 of 16 answers equal the HF backend's answer
  from the same fixed start (15 of 16 eager). The difference is in the
  base model's hidden states, not in the burst:
  * vLLM's read layer differs from HF's by about 2% (relative, bf16), from
    different attention and matmul kernels;
  * the head solves vLLM's read exactly as the server does: 16 of 16
    answers match when HF's head is given the rows vLLM read;
  * the puzzles that differ are the hard ones, with hundreds of solver
    iterations. They are also sensitive within HF: reading the same
    hidden states in fp32 instead of bf16 flips 2 of 16 HF answers.
  * Either way can be right. On these 16 the server solved 15 and HF 14
    (CUDA graphs).
* **Accuracy.** 75.3% served on vLLM vs 77.5% on the HF backend. The gap
  is within what a different solver start gives.
* **Speed.** 1.47 puzzles/s at `max_num_seqs=64`. The solve dominates:
  hundreds to thousands of head iterations per puzzle.
  * One request at a time, the median loop request took 0.19 s (prefill,
    solve and all 82 tokens). An ordinary 82-token decode took 0.67 s.
  * Hard puzzles take 1–3 s, all in the solver loop.
* **Tests.** 43 tests passed on the GPU pod, including the burst tests and
  the HF `solve_each` tests.
