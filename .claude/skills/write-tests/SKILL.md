---
description: Write tests for a commit, PR, or feature. Use when the user asks to write, add, or generate tests for code changes.
argument-hint: [commit-url or feature]
allowed-tools: Read Glob Grep Bash Write Edit
---
 
## Phase 1: Discovery
 
Before writing any test code, you MUST:
 
1. **Read the code under test** — understand all branches, error paths, and edge cases
2. **Check existing tests** — search the relevant `tests/` subdirectory to avoid duplication
3. **Identify the broader scope** — what other modules interact with this code? What flows does it participate in?
   Read every call site of the functions under test and verify: (a) every argument a caller passes is
   actually accepted by the function, and (b) every user-facing parameter (CLI flag, config field,
   public API arg) is threaded all the way through to the internal function that acts on it.
   **Parameter wiring audit (MANDATORY when applicable).** Produce the table below in your response
   BEFORE writing any test code IF AND ONLY IF **both** conditions hold:
   - the code under test sits between an outer caller (CLI parser, public API entrypoint, config
     loader) and at least one further internal stage it delegates to, AND
   - at least one parameter from the outer caller is supposed to influence a decision made by that
     internal stage (not just be stored or passed through unchanged)
   If the function is a leaf (does its own work and returns, no inner stage to delegate to), or if
   the parameters are local to the function (not coming from an outer caller), SKIP the audit —
   the wiring risk doesn't exist. Small helpers, math utilities, single-step transforms, and
   isolated logic do not need this table.
   **How to fill the table — read carefully, this is where audits fail:**
   - **Source of the parameter list** = the OUTERMOST entrypoint, NOT the immediate caller of the
     function under test. For a CLI tool: read every `add_argument` in the argparser. For a public
     API: read the top-level function's signature. Walking up only one level is the most common
     mistake — it misses old parameters that should flow into newly-introduced inner functions.
   - **One column per inner function the commit added or modified**, not just the function under
     test. A commit that introduces three new inner functions needs three "Reaches" columns. This
     is what catches cross-parameter gaps: an old CLI flag may already flow into one new function
     but silently miss another.
   - **Derive parameters from the entrypoint, NOT from the diff.** The diff shows only what
     changed; the audit's value is in finding parameters that *didn't* change but *should have*
     been wired into the new code paths.
   ```
   Parameter wiring audit. Entrypoint: <outermost_function_or_CLI>
   New/modified inner functions in this commit: <fn_A>, <fn_B>, <fn_C>
   | Parameter        | Defined at | Reaches fn_A? | Reaches fn_B? | Reaches fn_C? | Test needed? |
   |------------------|------------|---------------|---------------|---------------|--------------|
   | --foo            | CLI        | ✓             | ✓             | n/a           | no           |
   | --bar (pre-existing) | CLI    | ✓             | ✗ MISSING     | n/a           | YES (fn_B)   |
   | --baz (new)      | CLI        | ✓             | ✓             | ✓             | no           |
   ```
 
   The "Reaches" verdict requires tracing the call chain — not a guess. Mark `n/a` only when the
   parameter is semantically irrelevant to that function (e.g. a logging flag and a math helper).
   Every `✗ MISSING` row becomes a wiring test in Phase 3.
4. **Determine resource requirements** — does testing this need GPU, a real model, network access, or just CPU?
5. **Check registered markers** — run `grep -A 20 '\[tool.pytest' pyproject.toml` and note which
   markers are defined. Only use markers that are registered; do not invent new ones.
---
 
## Phase 2: Plan Test Coverage
 
Design tests along TWO axes:
 
### Axis 1: Resource Tier (determines markers)
 
| Tier | Markers | Where it runs |
|------|---------|---------------|
| Local-fast | `@pytest.mark.local_fast` | Any laptop, all CI |
| Local-slow | `@pytest.mark.slow` | CI, patient devs |
| GPU-fast | `@pytest.mark.gpu` | GPU machine |
| GPU-slow | `@pytest.mark.gpu`, `@pytest.mark.slow` | GPU cluster CI |
| Network | `@pytest.mark.network` | Online environments |
| Real-model | `@pytest.mark.requires_model` | Machines with checkpoints |
| Multi-GPU | `@pytest.mark.multi_gpu`, `@pytest.mark.gpu` | Multi-GPU cluster |
 
### Axis 2: Scope (determines file placement)
 
| Scope | Directory | Tests what |
|-------|-----------|-----------|
| Unit | `tests/unit/` | Pure logic, config, math — NO model instantiation |
| Component | `tests/hf/`, `tests/vllm/`, or `tests/composer/` | Single module with tiny random-weight models |
| Integration | `tests/integration/` | Cross-module, real compose pipeline, HF-to-vLLM equivalence |
 
**You MUST produce at least one local-fast test.** For non-trivial features, also produce a slower/broader variant.
 
### When to Add Integration Tests
 
Add integration tests ONLY if:
 
- Unit tests can't verify end-to-end correctness (e.g., HF → vLLM equivalence)
- The feature involves multiple modules with complex interactions
- Real model behavior differs from random-weight behavior
- The test validates a user-facing workflow (compose → load → infer)
Skip integration tests if unit/component tests already prove correctness.
 
---
 
## Phase 3: Write Tests
 
### Marker Rules (apply consistently)
 
**Only use markers that are registered in `pyproject.toml` (checked in Phase 1 step 5). Never
apply a marker that is not in the project's marker list — it will produce warnings and may silently
skip tests under strict-markers mode.**
 
Apply a marker when a test requires a non-default resource:
 
- **`gpu`**: Test calls `.cuda()`, uses vLLM, or requires `torch.cuda.is_available()`
- **`slow`**: Test takes >30 seconds (real compose, large seq_len, real inference)
- **`network`**: Test makes real HTTP calls (HF Hub, external APIs) — NOT mocked calls
- **`requires_model`**: Test needs a real pretrained checkpoint (not random weights)
- **`multi_gpu`**: Test needs 2+ CUDA devices (tensor/pipeline parallelism)
- If a test has `skipif(not _CUDA_AVAILABLE)`, it MUST also have `@pytest.mark.gpu`
**Every test MUST have at least one marker — no exceptions.** Use `@pytest.mark.local_fast` as
the default for pure CPU tests with no special resource requirements. A test can have multiple
markers (e.g., `@pytest.mark.gpu` + `@pytest.mark.slow`). Apply markers at the **class level**
when all methods share the same resource requirements; add method-level markers only when a
specific method differs.
 
### Coverage Goal: Every Code Path + Edge Cases
 
The goal is **comprehensive coverage** — every branch, every error path, every edge case that could behave differently. Avoid redundancy by using `parametrize` to collapse inputs that exercise the *same* code path into one test, but never skip a case that exercises a *different* path.
 
**Edge case checklist** — for every function under test, systematically ask:
- Empty collections (`[]`, `{}`, `""`)
- `None` / missing keys
- Zero and negative numbers
- Boundary values (min/max rank, seq_len, batch size, num_adapters)
- Invalid combinations (mismatched shapes, wrong dtypes, out-of-range indices)
- Repeated / idempotent calls (does calling twice break state?)
- Error paths — every `raise` should have a test
- Swallowed exceptions — when an error is caught and execution continues with a fallback (e.g.,
  `except Exception: value = default`), test the fallback behavior, not just that no exception
  propagates
- **Multi-stage consistency** — when a feature has sequential stages (download → discover, encode →
  decode, filter → process), verify that a control parameter affects **all** stages, not just 
  the one it's nearest to in the code. For each control parameter, set it to a non-default value
  and assert the output reflects that value at EVERY stage — not just the final one. If removing
  the parameter from any intermediate call would still let existing tests pass, you're missing
  coverage.
Use `parametrize` to express many cases concisely, not to reduce coverage:
 
```python
# BAD: only tests one input per branch, misses edge cases
def test_valid(): ...
 
# GOOD: all inputs that exercise different paths, expressed concisely
@pytest.mark.parametrize("value,expected", [
    ("normal", "ok"),
    ("",       ValueError),   # empty string → different branch
    (None,     TypeError),    # None → different branch
])
def test_all_paths(value, expected): ...
```
 
### Parameter-Wiring Tests — Guard Against Silent Bypass
 
Phase 1 step 3(b) asks you to verify that every user-facing parameter is threaded all the way
through. For each such parameter, **translate that verification into an explicit test**:
 
1. Set the parameter to a **non-default value** at the outermost function being tested
2. Mock the inner function whose behavior the parameter should **override** — so the default
   behavior would produce the wrong output if the parameter is silently dropped
3. Assert the output reflects the **non-default value**, not the default
This catches the bug class where a parameter exists in the public API but is never forwarded to
the layer that actually uses it. The test fails with `TypeError` (parameter doesn't exist yet) or
wrong output (parameter accepted but ignored) — both are signals of missing wiring.
 
**Naming**: `test_<param>_overrides_<default>` or `test_<param>_propagates_to_<output>`
 
**When to skip**: Only when the parameter passes through with no default to override (e.g., a
required arg that simply becomes a key in a dict). If removing the forwarding call would break an
existing test, no extra wiring test is needed.
 
**Pipeline insertion check**: When the commit inserts a new function into an existing pipeline,
the Phase 1 parameter wiring audit is your source of truth. Every `✗ MISSING` row from that table
becomes a wiring test here — a parametrized test with one row per unwired parameter is the right
shape. A wiring gap is silent: old parameters are accepted at the top level but never reach the
new code path, so existing tests pass while the feature is broken.
 
### Coding Conventions
 
- **Fixtures**: Use `tiny_config` from `tests/conftest.py` for fast CPU tests
- **HF models**: Use `make_switch_model()` from `tests/shared/generation_models.py`
- **Shared logic**: Use mixin pattern from `tests/shared/` when test applies to multiple backends
- **Parametrize**: Use `@pytest.mark.parametrize` for covering multiple flows efficiently
- **Class grouping**: Group related tests in classes organized by concern
- **Module fixtures**: Use `scope="module"` for expensive setup (compose, model loading)
- **Network mocking**: Mock all network calls UNLESS explicitly testing real Hub interaction
- **vLLM subprocess**: Use subprocess isolation pattern to keep CUDA context out of parent pytest
- **Precision**: Use `torch.testing.assert_close(atol=, rtol=)` for tensor comparisons
- **Naming**: `test_<what>_<condition>_<expected>` — descriptive, scannable
- **Docstrings**: Brief docstring on each test explaining what property is verified
- **No redundant imports**: Check what the target file already imports
- **Absent kwarg assertion**: To assert a kwarg was NOT forwarded, use
  `assert "kwarg" not in mock.call_args.kwargs` — more precise than checking call count alone
### Granularity Rule
 
**Always produce fast local tests.** Add a slower/heavier tier only when fast tests provably
cannot cover the gap — apply the "When to Add Integration Tests" criteria from Phase 2 as your
gate. If fast tests already prove correctness, stop there; do not add a `slow`/`requires_model`
class just for completeness.

Ask explicitly: *what would a fast test miss that a slow test catches?* If the answer is
"nothing", skip the slow tier.

### Example Structure
 
```python
# tests/composer/test_my_feature.py
import pytest
from granite_switch.composer.my_module import my_function
 
@pytest.mark.local_fast
class TestMyFeature:
    """CPU tests — cover every branch, edge case, and error path."""
 
    def test_basic_happy_path(self):
        """my_function returns expected output for standard input."""
        result = my_function(simple_input)
        assert result == expected
 
    def test_edge_case_empty_input(self):
        """my_function handles empty input gracefully."""
        result = my_function([])
        assert result == []
 
    @pytest.mark.parametrize("variant", ["a", "b", "c"])
    def test_all_variants(self, variant):
        """Each variant produces valid output."""
        ...
 
 
# Add this class ONLY if fast tests cannot prove end-to-end correctness
# (e.g. real model precision, cross-module interaction, user-facing workflow).
@pytest.mark.slow
@pytest.mark.requires_model
class TestMyFeatureIntegration:
    """Slower tests — add only when unit/component tests leave a real gap."""
 
    def test_real_model_round_trip(self):
        """Feature works end-to-end with a real checkpoint."""
        ...
```
 
---
 
## Phase 3.5: Review Written Tests
 
Before running anything, read back what you wrote and apply this checklist. The goal:
**minimum scaffolding, maximum distinct paths covered.**
 
### Redundancy scan — collapse duplicates with `parametrize`
 
For every pair of tests, ask: *do they exercise the same code branch with inputs that differ only
in value, not in path?* If yes, they must be merged into a single `@pytest.mark.parametrize`.
 
```python
# BAD: three separate methods, identical structure, only the value changes
def test_forwards_lora(self):   ...assert captured == "lora"
def test_forwards_alora(self):  ...assert captured == "alora"
def test_forwards_none(self):   ...assert captured is None
 
# GOOD: one method, three rows — same coverage, no duplication
@pytest.mark.parametrize("tech,expected", [
    ("lora",  "lora"),
    ("alora", "alora"),
    (None,    None),
])
def test_technology_filter_forwarded(self, tech, expected): ...
```
 
A test is NOT redundant if it exercises a **different code branch**, even if it looks similar.
Merge only when the branch taken is identical across cases.
 
### Scaffolding audit — remove what isn't needed
 
For every mock or fixture in each test, ask: *if I removed this, would the test still reach
its assertion without crashing on an unrelated call?* If yes, remove it.
 
Common over-mocking patterns to eliminate:
- Mocking a function that isn't called on the path under test
- Patching return values that are never read by the assertion
- Setting up model attributes that are never accessed before `SystemExit`
Extract identical setup blocks into a **fixture or helper** — don't copy-paste patch stacks.
 
### Coverage gap check
 
After consolidating, re-read the code under test one more time and ask:
 
- Is there a branch (`if`, `elif`, `else`, early `return`, `raise`) not yet exercised?
- Is there a parameter whose absence vs. presence leads to different behavior?
- Does the None/empty/missing case go through a different path than a normal value?
Each distinct path must have at least one row in the test matrix.
 
---
 
## Phase 4: Verify
 
After writing tests, determine which situation applies:
 
### Situation A — tests for code already in the local branch
 
Run the fast tests and confirm they **pass**:
```
pytest <test_file> -v -s --tb=short -x -m "not slow and not requires_model"
```
 
### Situation B — regression tests for a commit/PR not yet merged locally
 
The tests **must fail** on the current code. That failure is proof of coverage.
Run the fast tests and confirm they fail with the **right error** (e.g. `TypeError: unexpected
keyword argument` or a wrong-value assertion — not an import error or unrelated crash):
```
pytest <test_file> -v -s --tb=short -m "not slow and not requires_model"
```
Do **NOT** apply the source changes from the incoming commit to make the tests green.
That defeats the purpose: a test that only passes after you patch the code yourself proves
nothing. The tests will turn green automatically once the PR is merged.
 
In both situations:
- Confirm markers are applied correctly: `pytest <test_file> --collect-only -q`
- Check that slow/GPU tests are properly gated and will skip gracefully on CPU machines
---
 
## Reference: Existing Test Utilities
 
| Utility | Location | Use for |
|---------|----------|---------|
| `tiny_config` fixture | `tests/conftest.py` | Minimal config for CPU tests |
| `tiny_config_no_adapters` fixture | `tests/conftest.py` | Zero-adapter config |
| `make_switch_model()` | `tests/shared/generation_models.py` | Build HF model with random weights |
| `DENSE_CFG`, `basic_overrides()` | `tests/shared/generation_models.py` | Standard test configs |
| LoRA case mixins | `tests/shared/lora_cases.py` | Backend-agnostic LoRA tests |
| Switch case mixins | `tests/shared/single_switch_cases.py` | Backend-agnostic switch tests |
| `_tree_response()` pattern | `tests/composer/test_selective_download.py` | Mocking HF Hub tree calls |
 