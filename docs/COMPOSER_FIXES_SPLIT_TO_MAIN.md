# Two composer fixes that belong to main, not to this branch

Both were developed on `feature/multiswitch-kv-policy` and have been **removed from it**.
Each now lives on its own branch off `origin/main`, pushed, because each fixes a bug that
is on `main` and neither has anything to do with the KV-history policy work.

| fix | branch | commit | removed here by |
|---|---|---|---|
| control-token embedding rows are random after resize | `bugfix/control-token-embedding-init` | `c4db141` | `190567a` |
| Pass 2 drops the invocation's first CHARACTER, not its first TOKEN | `bugfix/alora-invocation-tail` | `41a6883` | `7a7cea3` |

```
origin/main ──┬── bugfix/control-token-embedding-init   c4db141  (pushed)
              │      composer: _init_control_token_embeddings + CPU unit test
              │
              ├── bugfix/alora-invocation-tail          41a6883  (pushed)
              │      composer: alora_invocation_tail + corrected tests
              │
              └── ... ── feature/multiswitch-kv-policy
                           190567a  revert: embedding init off this branch
                           7a7cea3  revert: invocation tail off this branch
```

Read either without checking it out:

```bash
git show bugfix/control-token-embedding-init:src/granite_switch/composer/compose_granite_switch.py
git show bugfix/alora-invocation-tail:tests/composer/test_chat_template.py
```

---

## 1. Control-token embedding rows are random after resize

**Branch:** `bugfix/control-token-embedding-init` (`c4db141`)

`resize_token_embeddings` gives every newly-added control token a **random** input
embedding row. On this branch and on `main`, nothing corrects it — see
[`compose_granite_switch.py:1041`](../src/granite_switch/composer/compose_granite_switch.py),
where the resize is followed only by the untied-`lm_head` helper at line 1053, which
touches the output side and only when `tie_word_embeddings` is false:

```
STEP 3
  model.resize_token_embeddings(new_vocab_size)      <- new rows are RANDOM

  embed_tokens.weight            lm_head.weight
  ┌──────────────────┐           ┌──────────────────┐
  │ 0..V-1  base      │          │ 0..V-1  base      │
  │ V+0  <|adapter_0|>│ random   │ V+0  <|adapter_0|>│ random
  │ V+1  <|adapter_1|>│ random   │ V+1  <|adapter_1|>│ random
  └──────────────────┘           └──────────────────┘
       ^ nobody fixes these           ^ initialize_untied_control_token_lm_head_rows
         on either path                 fixes these, untied only
```

Two composes of the same base + adapters therefore differ, and a low-confidence
downstream position can resolve a different top-1 token between them. HF and vLLM,
running different kernels, then disagree on that one position — an argmax-equivalence
failure with no code change behind it.

**The fix.** `_init_control_token_embeddings` (`compose_granite_switch.py:181` on the fix
branch) copies each control token's token-exchange **substitute** row into its own row,
right after the resize (`:992`). The substitute is the token the control token is
rewritten to at the input side, so the row is in-distribution; it is config-derived
rather than sampled, so it is identical across runs. It fires on the **tied** path too
(granite-4.0/4.1), which the untied-only helper never reaches.

| path | before | after |
|---|---|---|
| tied (4.0, 4.1) | random `embed` row, shared with head | `embed[ctrl] = embed[sub]` |
| untied (4.2) | random `embed` row; head corrected | both corrected |
| base slot (`sub == ctrl`) | n/a | skipped, nothing to copy |

**Tests** (fix branch): `tests/composer/test_control_token_embedding_init.py`, 6 CPU tests
— the copy on the tied path, the untied `lm_head` rows, non-control rows untouched,
determinism across two different random draws, and the self-substitute / negative /
out-of-range skips.

**Consequence while it is off this branch:** checkpoints composed here have random
control-token rows again, so HF-vs-vLLM argmax equivalence is flaky by construction.
Nothing on this branch measures that any more. The reporting-only **C1** check that did
(`tests/vllm/test_newms_verify.py` — no assert, as its own docstring admitted) was removed
as characterization, so the 6 CPU tests on the fix branch above are the only coverage of
the initialization.

---

## 2. Pass 2 drops the invocation's first CHARACTER, not its first TOKEN

**Branch:** `bugfix/alora-invocation-tail` (`41a6883`)

The runtime swap puts the **first invocation token's** embedding at the control token's
position, so the text Pass 2 emits after the control token must start one *token* in.
This branch and `main` slice one *character* —
[`tokenizer_setup.py:478`](../src/granite_switch/composer/tokenizer_setup.py):

```python
+ """ = _parts[0] + ns.adapter_token + ns.adapter_invocation_text[1:] + _parts[1] %}
                                                                 ^^^^ character rule
```

The two rules coincide only when the first character tokenizes alone — a property of the
vocabulary's merges, not of the code.

```
<|req_check|>requirements>req1
^^^^^^^^^^^^^ becomes '<' at runtime, reconstructing ['<', 'requirements', '>']
```

**Latent today, and measured.** On all six cached Granite tokenizers (4.1-3b, 4.1-8b,
4.0-micro, 4.0-h-tiny, switch-4.1-3b-preview, 3.3-2b-instruct) every invocation the
library ships starts with a lone `'<'`, so the fix changes no current render:

| invocation | tokens (4.1-3b) | first token | rules agree |
|---|---|---|---|
| `<requirements>` | `['<', 'requirements', '>']` | `'<'` | yes |
| `<certainty>` | `['<', 'cert', 'ainty', '>']` | `'<'` | yes |
| `<guardian>` | `['<', 'guard', 'ian', '>']` | `'<'` | yes |
| `<context>` | `['<', 'context', '>']` | `'<'` | yes |

It is the **next** invocation string that pays, and Granite's own structural markers show
how much — single vocab entries on 4.1-3b and 4.0-micro:

| marker | tokens | token tail | character tail | what the char rule renders |
|---|---|---|---|---|
| `</documents>` | `['</documents>']` | `''` | `'/documents>'` | `['</documents>', '/', 'documents', '>']` |
| `</think>` | `['</think>']` | `''` | `'/think>'` | `['</think>', '/', 'think', '>']` |

Four tokens where the adapter was trained on one. The failure is a quality regression at
the swap site, not an error, so nothing reports it.

**The fix.** `alora_invocation_tail(text, tokenizer)` (`tokenizer_setup.py:163` on the fix
branch) slices by `len(tokenizer.decode([ids[0]]))`, stores the result on the adapter map
as `invocation_tail` at compose time, and Pass 2 reads it (`:476`) instead of slicing in
Jinja. A caller handing over a stub tokenizer falls back to the character rule **with a
warning** rather than silently.

### One correction, if you review that branch

The original commit (`c775046`, now reverted here) justified itself with `<context>`
tokenizing as `['<context', '>']` on granite-4.1-3b. It does not. `'<context'` is a real
vocab entry (id 35628), but BPE never produces it for `'<context>'`:

```python
>>> tok("<context>", add_special_tokens=False).input_ids
[27, 2196, 29]                      # ['<', 'context', '>']
>>> tok.convert_ids_to_tokens(35628)
'<context'                          # in the vocab, never emitted here
```

So its anti-regression test was failing on this branch: `assert [27, 2196, 29] != [27,
2196, 29]`. The invariant was right; the example was not. `41a6883` argues from
`</documents>` / `</think>` instead and pins them in `_DISCRIMINATORS`
(`tests/composer/test_chat_template.py:272`), which is what stops a revert to the
character rule from passing. **Verify a tokenization by encoding the string, never by
vocab membership.**

### Second consumer

`feature/per-message-adapter` auto-emits invocation text at two more Jinja sites that
still slice by character, so it needs this rule more than `main` does. With the fix on a
branch off `main`, that branch picks it up by merging `main`. The working note
`docs/INVOCATION_TAIL_MERGE_NOTE.md` (untracked, deliberately) carries the details.
