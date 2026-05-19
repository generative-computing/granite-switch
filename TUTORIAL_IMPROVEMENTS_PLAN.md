# Tutorial Improvements Plan - Based on Luis Lastras Feedback

**Date:** 2026-05-19
**Context:** Feedback from walkthrough of `hello_mellea.ipynb`
**Branch:** `improve-tutorials` (public granite-switch repo)

---

## 📋 Feedback Summary

Luis's key observations:
1. ✅ `01_hello_mellea` works better as starting point than HF notebooks
2. ⚠️ vLLM launch is slow + requires A100 (drag for users)
3. ⚠️ Root cause: HF backend not working in mellea + no hosted model solution
4. 🔧 Specific changes needed (see below)

---

## 🎯 Required Changes

### 1. Remove "Intrinsics" Terminology

**Issue:** The word "intrinsics" shows up in mellea, but we moved away from it to avoid introducing new concepts while explaining Granite Switch.

**Affected notebooks:**
- ✅ `hello_mellea.ipynb` - Title mentions "Using Mellea Intrinsics"
- ✅ `hello_adapter.ipynb` - May reference intrinsics indirectly
- ✅ `granite_switch_with_hf.ipynb` - May reference intrinsics from Core library

**Files to check:**
```bash
cd granite-switch/tutorials/notebooks
grep -i "intrinsic" *.ipynb
```

**Replacement strategy:**
- "intrinsic" → "adapter"
- "intrinsic wrapper" → "adapter wrapper" or "mellea wrapper"
- "intrinsic call" → "adapter call"
- "intrinsic AST node" → "adapter invocation" or just "Intrinsic node" (when referring to mellea API class)

**Locations in hello_mellea.ipynb:**
- Cell `intro`: Title "Using Mellea Intrinsics" → "Using Mellea with Granite Switch"
- Cell `intro`: "invoking **mellea intrinsics**" → "invoking **mellea adapters**"
- Cell `intro`: "intrinsics from the Guardian library" → "adapters from the Guardian library"
- Cell `intro`: "How to call an intrinsic through its high-level wrapper" → "How to call an adapter through its high-level wrapper"
- Cell `intro`: "the low-level `Intrinsic` AST node" → "the low-level invocation API" or keep `Intrinsic` (class name)
- Cell `intro`: "list of intrinsic wrappers" → "list of adapter wrappers"
- Cell `1bab556a6a1eda5d`: "invoke an intrinsic mellea doesn't wrap yet" → "invoke an adapter mellea doesn't wrap yet"
- Cell `reference-intrinsics`: "Other mellea intrinsic wrappers" → "Other mellea adapter wrappers"
- Cell `reference-intrinsics`: "new intrinsics can be added" → "new adapters can be added"
- Cell `695e3d0155280a60`: "chains these intrinsics end-to-end" → "chains these adapters end-to-end"
- Cell `695e3d0155280a60`: "the intrinsics framework" → "the adapter framework"

**Locations in hello_adapter.ipynb:**
- Cell `intro`: "guardian-core intrinsic" → "guardian-core adapter" (if present)

**Locations in granite_switch_with_hf.ipynb:**
- Cell `bd878b44`: `from mellea.stdlib.components.intrinsic import rag` → likely OK (import path)
- Cell `d1605862`: "intrinsics from the Core library" → "adapters from the Core library" (if present)

---

### 2. Add vLLM Wait Time Guidance

**Issue:** No guidance on how long to wait - users get impatient.

**Location:** `hello_mellea.ipynb`, cell `launch-vllm-heading`

**Current text:**
```markdown
## 1 · Launch vLLM server

Start the Granite Switch model on port 8000. The server runs in the background; `wait_for_server` polls `/health` until it is ready.
```

**Improved text:**
```markdown
## 1 · Launch vLLM server

Start the Granite Switch model on port 8000. The server runs in the background; `wait_for_server` polls `/health` until it is ready.

⏱️ **This takes ~3 minutes** on first run (model download + loading). Subsequent runs are faster (~30-60 seconds). The cell below will block until the server responds - no action needed.
```

**Alternative location:** Add to cell `launch-vllm` as a print statement before wait:
```python
print("⏱️  Starting vLLM server - this takes ~3 minutes on first run...")
print("    (model download + loading; subsequent runs: ~30-60 sec)")
print("    Waiting for server health check...")

if not wait_for_server(VLLM_PORT):
    tail_log("/content/vllm_server.log")
```

---

### 3. Merge Cells 1+2, Move Imports to Cell 3

**Issue:** Too many small cells at start - feels cluttered. Streamline for better flow.

**Current structure:**
```
Cell 0 (heading): "0 · Install and set up"
Cell install-tutorial-deps: %pip install
Cell hf-login: notebook_login()
Cell vllm-helper-setup: from granite_switch.tutorials... + kill_stale + print_gpu
Cell 1 (heading): "1 · Launch vLLM server"
Cell launch-vllm: VLLM_MODEL = ..., launch_vllm(), wait_for_server()
Cell 2 (heading): "2 · Configuration"
Cell config: Imports + env setup + print
Cell 3 (heading): "3 · Connect to vLLM backend"
```

**Proposed structure:**
```
Cell 0 (heading): "0 · Install and set up"
Cell install-tutorial-deps: %pip install
Cell hf-login: notebook_login()

Cell 1 (heading): "1 · Launch vLLM server"
Cell launch-vllm-combined:
  - from granite_switch.tutorials... + kill_stale + print_gpu
  - VLLM_MODEL = ..., launch_vllm(), wait_for_server()
  - (merge vllm-helper-setup + launch-vllm)

Cell 2 (heading): "2 · Configuration and imports"
Cell config:
  - All imports (mellea, json, os, Path, etc.)
  - VLLM_BASE_URL, VLLM_MODEL_NAME config
  - print statements

Cell 3 (heading): "3 · Connect to vLLM backend"
```

**Rationale:**
- Fewer cells = more streamlined
- Imports are typically grouped at top in Python, but here it makes sense to have them after vLLM setup since they depend on it
- Users see the vLLM launch as one logical step

---

### 4. Fix query_rewrite Example

**Issue:** Current example misleading - shows simplification, not context-based rewriting.

**Current example:**
```python
query = "I want to ask you something. what is...mmmm the the main city(capital you call it,right?) of France?"
# Rewritten: "What is the capital of France?"
```

**Problem:** This shows query *cleaning*, not query *decontextualization*. The design point of query_rewrite is:
- **Passthrough** when query is already self-contained
- **Rewrite based on context** when query has pronouns/references that need resolution

**Better example - Multi-turn conversation:**
```python
# Build conversation context
ctx = ChatContext()
ctx = ctx.add(MelleaMessage("user", "What is the capital of France?"))
ctx = ctx.add(MelleaMessage("assistant", "The capital of France is Paris."))

# Now ask a follow-up with a pronoun
query = "What river does it sit on?"

# Without context-aware rewriting, "it" is ambiguous
# With query_rewrite, "it" → "Paris"
rewritten = rag.rewrite_question(query, ctx, backend)
print(f"original:  {query}")
print(f"rewritten: {rewritten}")
# Expected: "What river does Paris sit on?"
```

**Alternative example - from 00_hello_adapter (Rex the dog):**
```python
# Conversation context
ctx = ChatContext()
ctx = ctx.add(MelleaMessage("user", "I have a dog named Rex. He spends a lot of time in the backyard."))
ctx = ctx.add(MelleaMessage("assistant", "Rex must love exploring!"))

# Follow-up with pronouns
query = "Is he more likely to get fleas because of that?"

# query_rewrite resolves pronouns using context
rewritten = rag.rewrite_question(query, ctx, backend)
print(f"original:  {query}")
print(f"rewritten: {rewritten}")
# Expected: "Is Rex more likely to get fleas because he spends a lot of time in the backyard?"
```

**Keep the with/without wrapper comparison (6a/6b)** - Luis liked that part. Just fix the example query to show context-based rewriting.

---

### 5. Additional Improvements (Not from Luis, but obvious)

#### 5a. Add visual diagram to granite_switch_with_hf.ipynb

**Current:** Text-only explanation of adapter flow
**Improvement:** Add flow diagram showing Turn 1-5 sequence

```markdown
## Adapter Flow in This Notebook

┌─────────────────────────────────────────────────────┐
│ Turn 1: "What's the expense ratio?"                 │
│   → Answer with docs                                │
│   → context-attribution (map answer → sources)      │
├─────────────────────────────────────────────────────┤
│ Turn 2: "What's a glide path?"                      │
│   → Answer                                          │
│   → uncertainty (confidence score)                  │
├─────────────────────────────────────────────────────┤
│ Turn 3: "Should I put my 401k in this?"             │
│   → Answer                                          │
│   → policy-guardrails (compliance check)            │
├─────────────────────────────────────────────────────┤
│ Turn 4: "Summarize with constraints"                │
│   → Answer                                          │
│   → requirement-check (constraint satisfaction)     │
├─────────────────────────────────────────────────────┤
│ Turn 5: Fact-check summary                          │
│   → factuality-detection (error flagging)           │
│   → factuality-correction (fix if needed)           │
└─────────────────────────────────────────────────────┘
```

#### 5b. Add error handling to granite_switch_with_hf.ipynb

**Current:** No try/except around generation
**Improvement:** Wrap `generate_turn` calls

```python
def generate_turn(messages, adapter=None, documents=None, max_new_tokens=64):
    """Render a chat prompt with the named adapter active and greedy-decode."""
    try:
        kwargs = {"adapter_name": adapter} if adapter else {}
        if documents:
            kwargs["documents"] = documents
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
    except Exception as e:
        print(f"⚠️  Generation failed: {e}")
        return "[generation error]"
```

#### 5c. Add summary cell to granite_switch_with_hf.ipynb

**At end of notebook:**
```markdown
## Summary - Adapters Used

| Turn | Adapter | Purpose | Result |
|------|---------|---------|--------|
| 1 | context-attribution | Map answer to sources | 3 sentences → doc_id:0 spans |
| 2 | uncertainty | Confidence score | digit=7 → ~75% certain |
| 3 | policy-guardrails | Compliance check | "Ambiguous" (needs review) |
| 4 | requirement-check | Constraint satisfaction | "yes" (80 words, no jargon) |
| 5 | factuality-detection | Error detection | "yes" (error found) |
| 5 | factuality-correction | Fix errors | Corrected version returned |

**Key patterns demonstrated:**
- ✅ Judge pattern (temporary message variants, don't pollute history)
- ✅ Adapter chaining (detection → correction)
- ✅ Guardian screening (before each user turn)
- ✅ Document context passing
```

---

## 🔄 Implementation Order

### Phase 1: Critical Fixes (30-45 min)
1. ✅ Replace "intrinsics" → "adapters" across all 3 notebooks
2. ✅ Add vLLM wait time guidance (01_hello_mellea)
3. ✅ Fix query_rewrite example to show context-based rewriting (01_hello_mellea)

### Phase 2: Structure Improvements (30-45 min)
4. ✅ Merge cells 1+2, move imports (01_hello_mellea)
5. ✅ Add flow diagram (02_granite_switch_with_hf)

### Phase 3: Polish (15-30 min)
6. ✅ Add error handling (02_granite_switch_with_hf)
7. ✅ Add summary cell (02_granite_switch_with_hf)

**Total estimated time:** 75-120 minutes

---

## 📝 Testing Checklist

After changes:
- [ ] Run 00_hello_adapter end-to-end (local or Colab)
- [ ] Run 01_hello_mellea end-to-end (Colab with A100)
- [ ] Run 02_granite_switch_with_hf end-to-end (local GPU)
- [ ] Verify no mentions of "intrinsic" except in import paths / class names
- [ ] Verify vLLM wait message appears
- [ ] Verify query_rewrite shows context-based rewriting with pronouns
- [ ] Verify cell count reduced in 01_hello_mellea

---

## 🎯 Success Criteria

- [ ] All occurrences of "intrinsic" (concept) replaced with "adapter"
- [ ] vLLM wait time clearly communicated (~3 min first run)
- [ ] Cell structure streamlined (01_hello_mellea: 1+2 merged)
- [ ] query_rewrite example demonstrates context-based rewriting (not cleanup)
- [ ] Flow diagram added to 02_granite_switch_with_hf
- [ ] Error handling added to generation helper
- [ ] Summary table added to 02_granite_switch_with_hf

---

## 📂 Files to Modify

```
granite-switch/tutorials/notebooks/
├── hello_adapter.ipynb        # Minor: terminology fix
├── hello_mellea.ipynb         # Major: all 4 changes
└── granite_switch_with_hf.ipynb  # Medium: optional improvements
```

---

## 🔗 Related Context

- **Luis's observation:** "it works much better as a starting point" (01_hello_mellea vs HF path)
- **Root limitation:** HF backend not working in mellea + no hosted model solution
- **Trade-off:** vLLM is slow to launch but necessary for production-speed demos
- **Luis liked:** with/without wrapper comparison in query_rewrite section

---

## 💡 Future Considerations (Not in scope)

1. **Hosted model solution** - Would eliminate vLLM launch wait
2. **HF backend in mellea** - Would enable faster local demos without vLLM
3. **Smaller model for tutorials** - 3B still requires A100; could compose 1B for faster demos
4. **Pre-warmed Colab instances** - HuggingFace could host pre-configured Colab with model cached
