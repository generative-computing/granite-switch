# Per-message adapter field in the chat template

Date: 2026-08-16
Status: proposed
Scope: `granite_switch.composer.tokenizer_setup` (template generator), the composed
`config.json`, and `granite_switch.conversation`

Filename follows the repo's `UPPER_CASE.md` rule with the spec convention's date prefix.

## 1. Problem

A control token is markup that only exists in a rendered prompt. A conversation stored as
`messages` holds role and content text, and text carries no control tokens. So a re-render
on turn 2 drops turn 1's control token: the positions it governed are reinterpreted as
base, and their KV blocks stop matching.

Today the only fix is `Conversation` with `KVHistoryPolicy.PRESERVE_MIXED_HISTORY`, which
keeps the ids it already sent and appends a delta. That works and is measured, but it makes
the caller hold state, and it rules out every host that speaks `messages`:
`/v1/chat/completions`, Mellea, any OpenAI-compatible client. `Conversation.chat_payload`
raises under PRESERVE for exactly this reason.

The root cause is a data-structure limit, not a policy choice. The template's entire adapter
state is two scalars:

| variable | line in a composed `chat_template.jinja` | meaning |
|---|---|---|
| `adapter_token` / `adapter_type` | 19-23 | resolved from one `adapter_name` kwarg |
| `ns.alora_target_idx` | 41, reassigned at 104 | **one** integer: the single message to rewrite |

One integer cannot describe two turns with two adapters. That is the first problem.

### 1b. The second problem: a silent fallback that hides a wrong activation point

The invocation text appears in a composed template only as something to *search for* (lines
100, 103, 108) and then to `rsplit` off and put back (134, 136) — net zero. The template never
introduces it. So a caller who passes ordinary content, with no invocation text in it, makes
Pass 1 find nothing, `alora_target_idx` stays `-1`, and the boundary fallback fires.

Measured on `granite-switch-4.1-3b-preview` with content `"Rate the incident report."` and
nothing else:

| `adapter_name` | its real invocation | rendered result |
|---|---|---|
| `uncertainty` | `<certainty>` (user message) | `…report.<\|end_of_text\|>\n<\|uncertainty\|><\|start_of_role\|>assistant…` |
| `requirement-check` | `<requirements>` (user message) | `…report.<\|end_of_text\|>\n<\|requirement-check\|><\|start_of_role\|>assistant…` |
| `guardian-core` | `<guardian>` (user message) | `…report.<\|end_of_text\|>\n<\|guardian-core\|><\|start_of_role\|>assistant…` |
| `query_rewrite` | the assistant role marker | `…report.<\|end_of_text\|>\n<\|query_rewrite\|><\|start_of_role\|>assistant…` |

All four are identical, and no invocation text appears in any of them. Three of those adapters
were trained to activate at a point inside the user message; they are instead activating at the
assistant boundary, without their invocation text. The distinction between a user-trigger
adapter and an assistant-boundary adapter collapses silently, because `-1` means both "this
adapter activates at the boundary" and "the caller did not supply the trigger".

Only a caller who knows to hand-write the invocation text avoids this —
`tutorials/notebooks/hello_adapter.ipynb` does, via `build_guardian_block`. That is the
diligent path, not the default one.

## 2. Non-goals

* Not replacing `Conversation`. A re-render's cache reuse is contingent on byte-identity; a
  kept transcript's is structural. Section 8 quantifies the difference.
* Not changing any switch, LoRA layer, model class, or engine code. This is prompt
  construction only.
* Not the `control_to_substitute_lut` ⟺ tail matched-pair assert (section 11).
* Not ChatML / Granite 4.2 end-to-end validation, still deferred.
* Not base-reset / return-to-base routing, which lives on its own branch.

## 3. The change

Replace "one insertion site, found by scanning" with "one insertion site per message that
declares an adapter". Placement per site uses the rules that already exist:

```
for each message i carrying adapter=X:
    X is aLoRA, invocation text occurs in message i   ->  insert the control token before its
                                                          last occurrence (Pass 2 today,
                                                          lines 133-136)
    X is aLoRA, invocation text absent from message i ->  EMIT control token + invocation
                                                          text at the content BOUNDARY, at X's
                                                          declared placement, default PREFIX.
                                                          The content itself is NOT modified
                                                          (see "Where it goes" below)
    X is aLoRA, invocation IS the assistant role      ->  control token at the boundary, with
                                                          nothing to emit (the fallback today,
                                                          lines 179-180). This is now the ONLY
                                                          route to the boundary: a user-trigger
                                                          adapter must never land there by
                                                          accident (section 1b)
    X is LoRA                                         ->  reject (section 7)

adapter_name kwarg: unchanged. Applies to the turn being generated.
```

Two consequences, both deliberate.

**Message content stays free of adapter markup, and is never rewritten.** The caller writes
their own words plus an adapter name; the template *emits* the control token and the
invocation text as prompt structure at the content boundary, immediately after
`<|end_of_role|>` for the prefix placement — the same way role markers are emitted. Today's
Pass 2 does string surgery on the content (`rsplit` on the invocation text, then rejoin,
line 136); on this path there is nothing to split and the content passes through untouched.
Content mutation survives only for the legacy case where a caller wrote the invocation text
inline themselves.

Measured on the real checkpoint, guardian-core: emitting `<|guardian-core|><guardian>` at the
boundary of an untouched body renders byte-identically to today's hand-built block —
identical text and identical ids — while the caller's content drops from
`'<guardian>As a judge agent, assess the text...'` to `'As a judge agent, assess the text...'`.

Note the two markups are different things and only one is a control token:

| | what it is | tokens | control token? |
|---|---|---|---|
| `<certainty>` | invocation text — ordinary prose | 4: `['<','cert','ainty','>']` | no, not even special |
| `<\|uncertainty\|>` | control token — fires the switch | 1: id 100361 | yes, and special |

A caller writing the invocation text inline still gets today's placement, so nothing that
works now breaks; auto-insertion only fills in what was left out.

**Where it goes: PREFIX, and it is adapter-specific.** The only usage verified in this repo is
a prefix. `tutorials/notebooks/hello_adapter.ipynb` has the caller build this by hand:

```python
def build_guardian_block(criteria: str) -> str:
    schema = "If the text meets the criteria, return 'yes'; otherwise, return 'no'."
    return (
        f"<guardian>{JUDGE_SYSTEM}\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Scoring Schema: {schema}"
    )
```

`<guardian>` opens the block; the judge system prompt, criteria and scoring schema follow it,
and all of them must be processed *under* the adapter. Mellea states the same contract
independently (`mellea/stdlib/components/intrinsic/guardian.py:4-13`): those adapters
"require a `<guardian>`-prefixed envelope as the last user message of the request".

So default to prefix. But do not assume prefix is universal — no verified usage exists in this
repo for `<certainty>` or `<requirements>`, and the position is a property of how each adapter
was trained, not of the mechanism. The composer therefore records a per-adapter
`invocation_placement` (`prefix` | `suffix`) in the template's `adapter_map` alongside
`invocation_text`, defaulting to `prefix`. Getting it wrong moves where the adapter switches
on, which is silent: the prompt still renders and the adapter still answers.

The envelope *body* (the judge system prompt, criteria, schema) stays the caller's or
Mellea's job. It is separate metadata — `io_configs/<adapter>/io.yaml`'s `instruction:` field
— and it is unpopulated for most adapters: 79 of the 166 `io.yaml` files available locally
have `instruction: ~`, `guardian-core`, `uncertainty` and `requirement-check` among them.
This spec inserts the invocation text only.

**The scan disappears, and with it a failure class.** Today Pass 1 reassigns
`ns.alora_target_idx` while iterating (line 104), so the *last* matching message wins. When
the newest user message does not repeat the invocation text, the last match is an older
message and the token is placed into already-sent history — which breaks
`PRESERVE_MIXED_HISTORY` outright and silently re-attributes history under any policy. With
placement declared per message and the invocation text supplied rather than searched for,
there is nothing to find and nowhere else to put it.

The emitted form after the control token — full invocation text or its tail — is unchanged
and continues to follow the checkpoint's own mechanism (see section 8's two generations).

## 4. Before and after

Measured on `ibm-granite/granite-switch-4.1-3b-preview`, adapter `uncertainty`, invocation
text `<certainty>`. Two user turns and one recorded answer.

### Caller

Today has two paths and neither is good. The BEFORE block below is the second one.

| | content the caller writes | where `uncertainty` activates | every turn expressible? |
|---|---|---|---|
| **today, typical** | `"Rate the incident report."` | **assistant boundary, with no `<certainty>` at all** — the silent fallback of section 1b | no |
| **today, diligent** | `"<certainty>Rate the incident report."` | correctly, before `<certainty>` | no — one `adapter_name` per render |
| **proposed** | `"Rate the incident report."` | correctly, before an emitted `<certainty>` | yes |

So the change does two things at once: it makes history expressible, and it removes the
typical path's wrong activation point. The caller ends up writing exactly what they would have
written in the typical path, and getting what the diligent path produces.

```python
# BEFORE - the DILIGENT path. <certainty> is written by hand because the template
# will not supply it; omit it and the adapter silently activates at the assistant
# boundary instead (section 1b). Even so, only this turn's adapter is expressible,
# so turn 1's control token cannot be reproduced.
messages = [
  {"role": "user",      "content": "<certainty>Rate the incident report."},   # <-- hand-written
  {"role": "assistant", "content": "High confidence."},
  {"role": "user",      "content": "<certainty>Now as JSON."},                # <-- hand-written
]
tok.apply_chat_template(messages, add_generation_prompt=True, adapter_name="uncertainty")

# AFTER - content is the caller's own words, nothing else. The template emits both
# the control token and the invocation text at the content boundary.
messages = [
  {"role": "user",      "content": "Rate the incident report.", "adapter": "uncertainty"},
  {"role": "assistant", "content": "High confidence."},
  {"role": "user",      "content": "Now as JSON.", "adapter": "uncertainty"},
]
tok.apply_chat_template(messages, add_generation_prompt=True, adapter_name="uncertainty")
```

The caller never writes `<|uncertainty|>` and no longer needs to know that `uncertainty`
activates on the string `<certainty>`. Measured: emitting the invocation text from the adapter
name produces a byte-identical prompt to a caller writing it inline — identical text and
identical ids — so this is ergonomics and correctness, not a change to the stream.

### Rendered output

Prefix placement, as section 3 specifies.

```
BEFORE  (39 ids, control tokens at [25])
<|start_of_role|>user<|end_of_role|><certainty>Rate the incident report.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>High confidence.<|end_of_text|>
<|start_of_role|>user<|end_of_role|><|uncertainty|><certainty>Now as JSON.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>

AFTER   (40 ids, control tokens at [3, 26])
<|start_of_role|>user<|end_of_role|><|uncertainty|><certainty>Rate the incident report.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>High confidence.<|end_of_text|>
<|start_of_role|>user<|end_of_role|><|uncertainty|><certainty>Now as JSON.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

Turn 1's region routes to `uncertainty` in AFTER and to base in BEFORE. The two streams first
differ at token 3, and AFTER is exactly **one** id longer — the retained control token, and
nothing else. Prefix placement is cleaner than suffix here for a mechanical reason worth
knowing: the token lands immediately after `<|end_of_role|>`, so it splits no merge. The same
example with the invocation as a suffix costs two ids per turn, because the control token
lands between a space and a `<` and breaks the merged token `Ġ<` into `Ġ` + `<`.

### Why AFTER is exactly right

AFTER is byte-identical to what `Conversation` under `PRESERVE_MIXED_HISTORY` sends, and a
single-pass encode of it reproduces those ids exactly — so emitting a control token mid-stream
does not create a non-canonical seam. That holds for the prefix render above
(`encode(AFTER) == transcript ids`, 40 ids) and for the suffix form on three real adapters:

| adapter | invocation | transcript | `encode(decode(transcript))` | identical |
|---|---|---|---|---|
| `factuality-detection` | `<guardian>` | 63 ids | 63 ids | yes |
| `uncertainty` | `<certainty>` | 63 ids | 63 ids | yes |
| `requirement-check` | `<requirements>` | 61 ids | 61 ids | yes |

Both positions round-trip, which is what makes `invocation_placement` a free choice
mechanically — it matters for adapter fidelity (section 12), not for tokenisation.

This is the property the implementation must preserve, and section 10 turns it into a test.

## 5. Caller contract

* `messages[i]["adapter"]` is an adapter name from the checkpoint's `adapter_map`, meaning
  "this turn was produced with that adapter active". It is the **only** adapter-related thing
  a caller writes: no control tokens, and no invocation text unless they want to control its
  position within the message.
* The field goes on the **user** message. That is where Pass 2 already rewrites content, so
  the change generalises the existing rule. Putting it on the assistant message would force
  the template to map a field on message *i* to an insertion in message *i-1* for no gain.
* Absent field means no control token for that message, i.e. base — today's behaviour.
* `adapter_name` keeps its exact current meaning and is still how the turn being generated
  selects its adapter.

## 6. Data flow

```
messages (+ per-message adapter)  --.
                                    |
adapter_name (this turn)  ----------+--> apply_chat_template --> text --> ids --> server
                                    |
adapter_map (baked into template) --'

no Conversation, no client-side id bookkeeping, no state
```

Compared with the wrapper path, which stays available:

```
Conversation._sent_ids (kept ids)  +  delta(text)  -->  ids  -->  server
```

## 7. Error handling

| condition | behaviour | why |
|---|---|---|
| `adapter` names something not in `adapter_map` | render fails loudly | a silent base render is indistinguishable from success and would be found only as a quality regression |
| `adapter` is a LoRA-technology adapter | reject | a LoRA's token belongs at sequence position 0 and applies to the whole sequence; it cannot describe one turn. Matches the refusal already enforced in `conversation.py` |
| aLoRA declared on a message whose content lacks the invocation text | emit control token + invocation text at the content boundary, at that adapter's declared placement (default prefix), content unmodified | this is the expected case, not an error: callers should not have to know an adapter's invocation string. Silently inserting nothing is what the current `_parts \| length > 1` guard at line 135 would do |
| an adapter whose correct invocation placement is unknown | ships as `prefix`, the only verified pattern, and is overridable by writing the invocation inline | a wrong placement moves where the adapter switches on and is silent — the prompt renders and the adapter answers, just later than it was trained to |
| per-message fields sent to a checkpoint whose template predates this change | ignored silently by that template | unavoidable: published templates cannot be changed. Mitigated by the capability flag in section 8 |

Templates cannot raise usefully, so "render fails loudly" and "reject" must be implemented
where the check can report: the composer's template validator for compose-time checks, and
`Conversation` for the client path. A server-side render has no client check, which is the
argument for the capability flag being mandatory rather than advisory.

## 8. Backwards compatibility

A message list with no per-message fields has nothing to iterate, so the render goes down
exactly today's path. This is a byte-identity property, pinned by a test rather than
asserted here.

Published checkpoints keep their existing templates and will ignore the field. Clients must
therefore be able to detect support. The composer writes it into `config.json`:

```json
"chat_template_features": ["per_message_adapter"]
```

A client that wants history routing checks for that feature and falls back to
`RE_PREFILL`-without-history-routing (today's behaviour) or to `Conversation` when it is
absent.

Two generations of checkpoint exist and are both correct, which is why the feature list is
keyed to the template and not to the model:

| generation | mechanism | buffers | template emits |
|---|---|---|---|
| published previews | hiding matrix only | `model.adapter_hiding_matrix`, no LUT | control token + **full** invocation text |
| current composer | token exchange, both modes | `control_to_substitute_lut` | control token + **tail** |

The engine already bridges them: `apply_token_exchange` returns its input unchanged when the
LUT is `None` (`hf/switch/_token_exchange.py:57-58`).

## 9. Effect on the two policies

| | routing of history | cache reuse |
|---|---|---|
| `RE_PREFILL` today | lost — history reads as base | not applicable |
| `RE_PREFILL` + per-message field | correct | exact until the model emits a non-canonical token, then lost from that block onward |
| `PRESERVE_MIXED_HISTORY` | correct | structural, always |

The middle row's caveat is not hypothetical. Text to ids is not injective, and the Granite
vocabulary contains tokens the encoder will never produce for their own text:

```
'(sync'    id 98333  ->  re-encodes to [7, 13293]   = ['(', 'sync']
'_UID'     id 70982  ->  re-encodes to [62, 6599]   = ['_', 'UID']
'.median'  id 82896  ->  re-encodes to [13, 56751]  = ['.', 'median']
```

A model may emit any of them; a re-render will not. Measured consequence on a 62-token
turn-2 prompt whose recorded answer contained `.median` as one token: divergence at token
37, so reuse ends at `floor(37/16)*16 = 32` and 30 tokens whose text is byte-identical are
recomputed. The cache is a chain — a block's key includes its parent's hash — so one
differing token invalidates every block after it. A kept transcript is immune because it
stores id 82896 rather than re-deriving it.

Net effect: the default policy stops silently losing history's adapters, and `Conversation`
narrows to the case it is uniquely good for.

## 10. Components and testing

| file | change |
|---|---|
| `src/granite_switch/composer/tokenizer_setup.py` | Pass 1 becomes a per-message walk; Pass 2 fires for each declaring message; the fallback generalises to "next assistant boundary". The bulk of the work |
| the composer's `config.json` write | add `chat_template_features` |
| `src/granite_switch/conversation.py` | emit `adapter` on user messages so `RE_PREFILL` also routes history correctly. Small |
| `tests/composer/test_chat_template.py` | placement unit tests per kind |
| `tests/integration/` | the byte-identity regression below |

Tests, in the order they should be written:

1. **No fields changes nothing.** A message list with no `adapter` fields renders
   byte-identically to the current template for the same inputs. Guards every existing
   caller.
2. **Per-message render equals the transcript.** For a real composed checkpoint, the
   per-message render's ids equal `Conversation(PRESERVE).build_prompt()` for the same
   conversation. The measured targets are in section 4: 63/63, 63/63, 61/61 identical.
3. **Boundary emission equals hand-written.** Content without the invocation text plus
   `adapter: X` renders byte-identically to content with it written inline at X's declared
   placement. Measured for both: suffix (`uncertainty`) and prefix (`guardian-core`, the
   `hello_adapter.ipynb` shape), identical text and identical ids each time.
4. **Content is never rewritten on the declared path.** Assert the rendered output contains
   each message's content as a verbatim substring. This is what keeps Mellea's cached blocks
   matching (section 14) and it is cheap to check.
5. **Placement per kind.** aLoRA with the invocation inline in the declaring message; aLoRA
   without it (auto-appended); assistant-boundary aLoRA; base turns; and LoRA rejected.
6. **Case B cannot occur.** A conversation whose newest message lacks the invocation text
   gets this turn's token emitted at that message's boundary and leaves every earlier message
   byte-unchanged — the failure the scan causes today.
7. **A user-trigger adapter never lands at the assistant boundary.** Declaring `uncertainty`
   on a message with plain content must emit `<certainty>`, not fall through to the boundary.
   Pins the section 1b regression, which today makes `uncertainty`, `requirement-check`,
   `guardian-core` and the genuinely-boundary `query_rewrite` render identically.
8. **Capability flag round-trips** through compose and is readable from `config.json`.

All eight are CPU-only. Tests 2, 3, 4 and 7 need a composed checkpoint but no GPU; they are
tokenizer-level comparisons.

## 11. Out of scope, tracked separately

**Matched-pair validation.** A checkpoint's template and its buffers are a matched pair and
nothing validates the pairing. Crossing them fails silently in opposite directions: a tail
template on a hiding-matrix checkpoint deletes the `<` the adapter was trained on; a
full-text template on a token-exchange checkpoint sends it twice. Both render plausibly and
degrade the adapter quietly. Wanted: a load-time assert that `control_to_substitute_lut`
present ⟺ the template uses the tail. Different bug, different blast radius, separate
change.

## 12. Open question for adapter owners

`invocation_placement` defaults to `prefix` because that is the only pattern verified here
(`hello_adapter.ipynb`, and Mellea's guardian contract). For each library adapter with a
user-message invocation — `uncertainty`, `requirement-check`, `guardian-core`,
`policy-guardrails`, `factuality-detection`, `factuality-correction` — someone who knows the
training data has to confirm prefix or suffix. Until then, a caller who writes the invocation
inline keeps full control, so the default being wrong degrades ergonomics rather than
correctness for those callers. It does silently degrade anyone relying on auto-insertion.

## 13. Risks

* The template generator is Jinja assembled from Python strings. A per-message walk is more
  logic in that layer, which is awkward to test directly — hence test 1 as a regression net.
* `messages` gains a non-standard field. Clients that validate message schemas strictly may
  reject it; the capability flag tells them whether it is worth sending, not whether their
  own validator will allow it.
* `invocation_placement` defaults may be wrong for adapters nobody confirms (section 12). The
  failure is silent.

## 14. A collision this incidentally resolves

Mellea's KV reuse caches raw content strings and asserts each still appears verbatim in the
new render (`mellea/backends/huggingface.py:896-906, 936`):

```python
assert key in current_suffix, (
    "Could happen but would be rare. related to the other assert in this block."
)
```

Today an aLoRA control token is spliced *inside* a user message, so a cached block whose
content contains the invocation text stops appearing verbatim and that assert fires. Emitting
at the boundary instead leaves content byte-identical, so cached blocks keep matching. Worth
noting because it is an argument for boundary emission independent of ergonomics — the two
systems are compatible only if content is not rewritten.
