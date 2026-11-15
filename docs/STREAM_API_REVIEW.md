# Stream API branch review

Reviewed on 2026-11-15, resuming the second Claude review at `14d4548` on
`feature/add-stream-api`. Scope: single-token head plugins, including heads on
base, LoRA, aLoRA and Shadow Residual states. Block streams remain separate work.

## Assessment

The architecture supports the intended extension point: a plugin supplies a
`HeadType`, modules and weights; the core owns routing, read layers and output
modes. The probe exercises the same interface as external plugins. Keep that
separation. Adding more type-specific branches to the backends would weaken it.

The branch needed stricter boundaries before production use. Several mistakes
could previously survive composition and either fail during inference or
silently produce the wrong answer. This revision fixes the cases below. It is
not yet validated for production serving: GPU execution of this revision and
the supported dependency environment are still release gates.

## Disposition of Claude's second review

| Finding | Disposition |
|---|---|
| Output token IDs outside the vocabulary | Fixed: shared stream validation checks the vocabulary bound before model execution. |
| Plugin module names overwrite each other or model attributes | Fixed: registration validates names and uniqueness; attachment checks existing model attributes after core initialization. |
| Composer calls plugin weight copies with gradients enabled | Fixed: the core owns `torch.no_grad()` around each plugin loader. |
| Head manifest fields without a head kind are silently ignored | Fixed: discovery rejects these entries and reports the required `kind`. |
| bf16 rounds slot IDs above 256 | Fixed: IDs travel as exact byte-valued columns, then decode to integers at the logits boundary. CPU round trips and full-graph tracing pass; CUDA validation is pending. |
| Probe accepts arbitrary settings | Fixed: the built-in type rejects all settings; the MLP example also rejects unknown keys. |
| Malformed streams produce an unrelated adapter-kind error | Fixed: config and composer parse entries before deriving kinds or indexing them. |
| Duplicate implementations of the head-slot mask | Fixed: the switch helper uses the shared routing table, retaining the no-head path for standalone switch configs. |
| vLLM tests construct full models by hand | Fixed for the new full-model stream cases: they compose synthetic artifacts and load their actual checkpoint state. Method tests in `test_probe_capture_safety.py` deliberately retain small stubs; composition would not strengthen those isolated checks. |
| Repeated head parameter-prefix logic | Fixed: composer transfer and validation use the routing helper and reuse the result. |

## Additional issues fixed

- **Persistent plugin buffers:** both vLLM decoder loaders now load the complete
  head state before decoder mappings and skip rules. Missing, unexpected or
  misshaped head tensors fail at startup. A regression plugin deliberately uses
  a buffer name that resembles a decoder-owned buffer.
- **Malformed plugin outputs:** both backends require floating, two-dimensional
  scores with the correct token count and enough columns for the type's outputs.
  Missing scores can no longer become zeros or broadcast across tokens silently.
- **Ambiguous manifest input:** reject unknown head fields, non-mapping settings
  or output, invalid labels, and simultaneous labels and explicit output IDs.
  YAML examples quote `"yes"` and `"no"` so the parser preserves strings.
- **Artifact routing:** reject malformed stream counts and head declarations for
  empty built-in adapter slots. Head weight transfer checks distinct, valid slots.
- **Documentation:** clarify module naming, output shapes, persistent state,
  framework-owned weight copying and installation in every worker environment.

## Remaining release gates

1. **Run this revision on GPUs.** On both supported vLLM lines, exercise eager
   and CUDA-graph serving, mixed batches, chunked prefill, prefix-cache reuse,
   nonfinal read layers, select/merge outputs, and heads on each adapter kind.
   Include the new slot-257 and persistent-buffer cases. Repeat TP=2 and PP=2;
   the packed output and loader changes must be checked at those boundaries.
   Earlier GPU successes belong to the previous revision, not this one.
2. **Run CPU suites with the declared dependencies.** Local validation uses
   Python 3.11, torch 2.11.0 and transformers 5.8.0. The project declares
   transformers 5.16 through 5.17.0. Passing tests here cannot establish support
   for the declared environment.
3. **Raw-prompt validation is intentionally deferred (2026-11-15).** vLLM can
   silently lose the verdict when a head marker is not last, and it does not
   reject repeated markers. This applies to heads on base, LoRA, aLoRA and
   Shadow Residual, not ordinary adapter-only requests. Use the composed
   template or `Conversation`; manual head prompts must contain one head marker
   at the end. No validation fix is included in this branch. Before supporting
   arbitrary raw prompts, add host-side validation and serving tests. Passing
   GPU tests alone will not close this known gap. See the
   [API limitation](STREAM_API.md#known-gap-vllm-head-marker-validation-deferred)
   for examples and the supported workaround.
4. **Confirm compatibility claims.** Compare a real adapter-only checkpoint with
   the pre-stream implementation. Exercise old probe checkpoints on the new
   reader. Avoid promising that old readers support every new stream checkpoint:
   compact banks and custom head types need an explicit compatibility boundary.

## Follow-up improvements

- Publish a small installable plugin starter with a fresh-process discovery,
  compose, save/reload and serving example. The tested MLP source and packaging
  instructions already demonstrate the contract, but a starter would make the
  first plugin easier to build.
- Define plugin/checkpoint version compatibility before independently releasing
  plugins whose state layouts or settings change. Type names alone do not pin a
  Python package version.
- Measure memory and latency by token count, head bank size and read-layer count.
  vLLM executes each type on all tokens at each selected layer. The current
  examples are suitable starting points, not a cost guarantee for larger heads.

## Validation of this revision

**981 distinct CPU tests passed** across the selected suites; repeated focused
runs are not added to this total.

| Scope | Passed |
|---|---:|
| Unit suite, excluding unavailable model/GPU/slow/deep cases and the vLLM-dependent bitmask file | 335 |
| HF: MultiSwitch, model forward, probe, stream API, conversation routing | 405 |
| Composer: adapter loader, head manifest CLI, SR end-to-end, base reset, control LUT refresh, tokenizer setup, chat template | 241 |

The unit run also skipped 1,450 GPU/environment cases. The large real-model
`test_probe_compose.py` suite was deselected; the head manifest CLI test ran
separately with its cached tokenizer and passed all three cases. No GPU tests
were executed. vLLM method tests also need the unavailable vLLM dependency;
changed GPU test sources were syntax-checked only.

Additional CPU checks composed and reloaded LoRA and SR fixtures with exact
state equality, and composed a 257-head checkpoint and verified slot 257's HF
verdict. Slot byte transport passed bf16/fp16/fp32 boundary cases and full-graph
`torch.compile` tracing using the eager backend; this is not CUDA graph evidence.

All pre-commit checks passed, including formatting, lint, SPDX headers, link and
import validation, and the lockfile hook. Tests ran incrementally with
`-v -s --tb=short -x`, `PYTHONPATH=src` and offline Hub access.
See [STREAM_API.md](STREAM_API.md) for the user-facing API contract.
