# Loop Slot on vLLM: Emitting the Whole Answer in One Step

Status: implemented and served on vLLM (A100), branch
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

The sampler that answers a loop request is supplied by the model, and it
returns the request's whole answer, 81 digits then EOS, in the step that
finishes the prompt. One engine step; no draft or verify step.

```
each prefill step (the prompt may be split across several):
  model state (host):   which scheduled tokens are this request's grid cells
                        -> a fixed-size copy plan for the step
  forward (compiled, CUDA graph):
    at the read layer, a custom op copies the planned rows into the
    request's grid buffer (81 x hidden, kept across steps)
the step that finishes the prompt:
  sampler: ordinary rows -> vLLM's sampler, 1 token
           loop rows     -> loop head solves the buffered grid
                            -> 81 digits + EOS, num_sampled = 82
  scheduler: appends 82 tokens, stops at EOS  -> request done
```

### 3.1 Reading the grid

* **Finding a loop request (host).** When the worker adds a request, the
  model state checks that the prompt ends with a loop slot's control token.
  The grid's position follows from `grid_size` and `end_offset`.
* **The grid buffer.** `max_num_seqs × 81` rows of hidden size, in the model
  dtype, plus one sink row. Row `slot·81 + i` holds cell `i` of the request
  in runner slot `slot`. The buffer persists across steps.
* **The plan (host).** Before each forward, the model state intersects each
  loop request's scheduled tokens with its grid range. It writes two index
  vectors of fixed length into persistent device buffers: source token and
  destination row. Unused entries point at the sink.
* **The copy (device).** A registered custom op, `index_copy_` from the read
  layer into the buffer, runs inside the compiled forward at the configured
  read layer: embeddings (−1), a decoder layer, or the final norm.
  * It has fixed shapes and no branch, so it is safe under CUDA graphs.
  * Entries past the step's real tokens also land in the sink.
* **Chunked prefill.** Each chunk copies the grid cells it holds. The grid is
  complete when the prompt is, wherever the chunk boundaries fall.
* **Prefix caching** stays on for the engine. A prefix-cache hit would skip
  the grid's forward, so loop requests alone skip prefix-cache reads.

### 3.2 Solving and answering (the sampler)

`LoopSampler` wraps vLLM's V2 sampler and delegates everything else to it.

1. Run vLLM's sampler for every row, as usual.
2. If no loop request finishes its prompt in this step, return its output.
3. Otherwise solve those requests' buffered grids, one call per slot.
   * `LoopHead.solve_rows`: each row starts from a truncated normal drawn
     from its own seed. The seed is the request's `seed`, 0 when unset.
   * Each row stops on its own once it has converged, after `min_iter`
     updates.
   * So a request's answer does not depend on its batch, and it repeats
     when the request runs again.
4. Each cell's best output token, then EOS, capped at `max_tokens` and at
   `max_model_len − prompt_len`, becomes the row's sampled tokens.
   * `num_sampled` is the answer length, and `num_rejected` stays 0.
5. **Logprobs** are supported: each cell carries the log-softmax over the
   digit tokens, as the HF logit rows give, and EOS carries 0.
6. The answer is the head's argmax; sampling temperature does not apply to
   it.

### 3.3 Wiring

* **Model state.** `GraniteSwitchForCausalLM.get_model_state_cls()` returns
  `LoopModelState`, a subclass of vLLM's `DefaultModelState`, but only for a
  checkpoint with a loop slot. vLLM looks the hook up with `hasattr`.
* **Config hook.** A `VerifyAndUpdateConfig` subclass is registered for the
  Granite Switch architectures. For a loop checkpoint it sets the scheduler
  class to `make_loop_scheduler`; any other checkpoint is untouched.
* **Scheduler.** `make_loop_scheduler` returns a subclass of vLLM's sync or
  async scheduler, following the engine's `async_scheduling` setting.
  * Loop requests skip prefix-cache reads.
  * On a loop request's multi-token output, the async variant first books
    the extra placeholders, and the request then finishes as stopped.
  * Answer tokens therefore never need K/V.
* **No speculative or diffusion width.** Widening V2's output through
  `num_speculative_tokens` would make the scheduler pad every decode step
  with draft placeholders. The output path is not width-limited:
  * the output copy moves any width;
  * `post_update` loops over `num_sampled`;
  * the scheduler appends token by token.

### 3.4 What stays valid

| Hazard | Handling |
|---|---|
| Chunked prefill splits the grid | each chunk copies its own cells; the plan covers each cell once |
| Prefix-cache hit on the grid | loop requests skip prefix-cache reads; other requests keep the cache |
| Preemption | the request is found again when re-added, its slot state is cleared, and the grid is read again |
| vLLM's dummy and warmup batches | matched by request id, so they never trigger a solve |
| The V1 runner, PP, speculative decoding | refused when the model is built |

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

Alternatives 1 and 2 together were built and measured separately, on
`feature/loop-vllm-extra-kv-layer` (local, not continued).

* **How it works.** The answer is kept in an extra KV-cache attention layer,
  and a custom drafter proposes it.
* **Result.** 2 engine steps per answer, all 1,620,000 drafts accepted, and
  75.36% exact on the same 20,000 puzzles.
* **Costs.** It needs the V1 runner and a drafter config passed by the
  caller. It turns chunked prefill and prefix caching off for the engine.

## 5. Limits

* **The solve runs on the engine's critical path.** While a step's loop
  requests solve (up to 35,000 iterations for a hard puzzle), the other
  requests wait.
* **Not supported with loop slots:**
  * pipeline parallelism;
  * speculative decoding;
  * the V1 model runner;
  * loop slots that read different layers or have different grids.
* **No handling or tests** for tensor parallelism or structured outputs.
* **`ignore_eos`:** the request is finished anyway.
* **Scheduler interface.** It is subclassed for the loop rules, and vLLM
  marks it non-public, so a vLLM upgrade may need changes.

## 6. Cost

* **The solve.** It runs once per loop request, in the step that finishes
  its prompt.
  * Each iteration is a 2-layer, width-512 block over 97 tokens (16 prefix +
    81 grid).
  * Iterations to convergence: typically tens to a few thousand (puzzle
    #2000: 74–2,177).
* **The read.** Each forward of a loop checkpoint pays for one small
  fixed-size `index_copy_`.
* **Decoding.** A loop answer takes 1 engine step instead of 82.

## 7. Code

| Piece | File |
|---|---|
| Grid buffer, copy op, model state, sampler, scheduler subclasses | `src/granite_switch/vllm/loop.py` |
| Config hook | `src/granite_switch/vllm/loop_config.py`, registered in `vllm/__init__.py` |
| Request finding, copy plan, answer tokens and logprobs (no vLLM import) | `src/granite_switch/loop_serving.py` |
| Per-row seeded solve | `LoopHead.solve_rows` in `hf/core/loop.py`; HF solves the same way with `model.loop_seeds` |
| Read-layer copy in the forward | `vllm/granite_switch_model.py` |
| Serving eval | `tutorials/scripts/loop_sudoku/serve_vllm.py`, `run_serve_eval.sh` |
| CPU tests | `tests/unit/test_loop_serving.py` |

## 8. Results

The trained Sudoku head was composed onto `granite-4.1-3b` and served with
`vllm.LLM`. It is the head that scores 77.5% exact on the HF evaluation.

### This implementation (vLLM 0.30.0, one A100)

| Check | Result |
|---|---|
| Exact, 20,000 held-out puzzles `[2000, 22000)` | **75.28%** (HF evaluation: 77.5%) |
| Cells right / blank cells right | 90.5% / 86.2% |
| Answer | 82 tokens on every puzzle: 81 digits, then EOS |
| Tokens per engine step, engine driven step by step | loop request: **82 in one step**; ordinary request: 1 per step |
| Ordinary prompts, with vs without loop requests in the engine | 3 of 4 identical; the 4th is a near-tie ("Hola" vs "Hola.") |
| Same answer as HF, same seeded start | 24 of 32 puzzles (79.1 of 81 cells on average) |
| Chunked prefill (budget 48) vs whole prompts, 64 puzzles | grids equal up to bf16 rounding (median 2.7% relative); 47 of 64 answers identical |
| Time for 20,000 puzzles | 4.2 h (0.76 s per puzzle) |

### The first version (vLLM 0.26.0, one A100)

The first version on this branch (commit `4a8f921`, reverted in favour of
this one) used the same hook. It read the whole read layer instead of only
the grid, and turned chunked prefill, prefix caching and async scheduling
off for the engine.

* **Accuracy.** 20,000 held-out puzzles: 75.27% exact, 90.5% of cells.
  Every answer came out in one step.
* **Agreement with HF.** 14 of 16 answers equal HF's from the same fixed
  start.
  * vLLM's read layer differs from HF's by about 2% (relative, bf16), from
    different kernels.
  * HF's head, given the rows vLLM read, reproduced all 16 server answers.
* **Speed, one request at a time.** A loop request took 0.19 s median, vs
  0.67 s for an ordinary 82-token decode.

### What the results show

* **Both versions: 75.3% vs 77.5%.** The 2.2-point gap comes from the
  serving solve rules, not from vLLM.
  * A third, separate implementation (answer kept in a KV-cache layer)
    measured it: HF scores 75.05% when it solves with the serving rules
    (fixed start, each puzzle stopping on its own).
  * The HF evaluation stops only when the whole batch has converged, so
    every puzzle ran the full 35,000 iterations.
  * Stopping each puzzle once it converges is the likely cause. It is not
    yet tested on its own (raise `min_iter` and serve again).
* **Answers are not bit-for-bit repeatable.** The solver iterates hundreds
  to thousands of times and magnifies rounding differences: vLLM vs HF
  kernels, chunked vs whole prompts, batch composition. Accuracy over many
  puzzles is the stable number; identical 64-puzzle runs scored 73.4–78.1%.
