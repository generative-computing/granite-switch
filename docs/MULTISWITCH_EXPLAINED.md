# MultiSwitch: the engine, the backends, and the limits

One reference for coarse-grained multi-adapter routing in Granite Switch, as the code stands today: how the two attention heads compute an adapter index per token, what the HF and vLLM backends do differently, how multi-turn conversations reuse KV, and which limits are real. Every number below was read from the source or produced by running it.


*Converted from HTML on branch `bugfix/multi-switch-default`. Engine and backend
facts were read at `1c86ee9` on `feature/multiswitch-kv-policy`; the `switch_type`
default and the `config.py` / `modeling_granite_switch.py` line references were
re-verified against this branch. Other line references are carried over unchanged
and have not been re-audited.*

**Contents**

1. [What MultiSwitch is, and how it differs from SingleSwitch](#1-what-multiswitch-is-and-how-it-differs-from-singleswitch)
2. [The engine: two attention heads and a token rewrite](#2-the-engine-two-attention-heads-and-a-token-rewrite)
3. [Latest-wins routing, and returning to base](#3-latest-wins-routing-and-returning-to-base)
4. [The two backends, and the one thing that differs](#4-the-two-backends-and-the-one-thing-that-differs)
5. [Under vLLM continuous batching](#5-under-vllm-continuous-batching)
6. [Multi-turn: cache reuse and the Conversation API](#6-multi-turn-cache-reuse-and-the-conversation-api)
7. [Control-token placement](#7-control-token-placement)
8. [Limits that are real today](#8-limits-that-are-real-today)
9. [What is tested, and where](#9-what-is-tested-and-where)
10. [Reproducing every number here](#10-reproducing-every-number-here)

> **If you read only this box.** MultiSwitch replaces SingleSwitch's single-transition routing with a *coded memory*: two tiny attention heads turn "how many control tokens have I seen?" into an address, and read back the adapter written at that address. Routing is piecewise-constant, latest-wins, and a pure function of the token ids. The control token is then rewritten to a substitute id so the decoder never embeds it. One engine ships (`switch_type="multi"`, Kerdock/DG codes); two backends implement it; the only substantive behavioural difference between them is arithmetic precision, which caps a vLLM request at **188 retained control tokens**.

## 1. What MultiSwitch is, and how it differs from SingleSwitch

Both switches answer the same question -- *which adapter applies at each token position?* -- and both are selected by one config field:

```python
# src/granite_switch/hf/switch/__init__.py  (vllm/switch/__init__.py is the twin)
switch_type = config.switch_type
if switch_type == "multi":
    return MultiSwitch(**common)
return SingleSwitch(**common)
```

Note that `create_switch` has no default of its own to fall back on. Two different
places decide `switch_type`, and they answer different questions:

| | decides | value |
|---|---|---|
| `config.py` parameter default (`DEFAULT_SWITCH_TYPE`) | what a **new** checkpoint becomes | `"multi"` |
| `GraniteSwitchConfig.from_dict` | how a `config.json` with **no** `switch_type` key is read | `"single"` |

Only a checkpoint composed before `c5e78c6` (2026-07-30, the commit that added the
field) lacks the key -- which includes both published previews,
`ibm-granite/granite-switch-4.1-3b-preview` and
`barha/granite-switch-4.0-350m-demo`. Such a checkpoint has
`num_hidden_layers = L + 1`, because `_switch_cache_layers("single") == 1`. Reading
it as multi subtracts `MultiSwitch.num_cache_layers == 2` instead, so the model
builds one decoder layer too few (41 -> 39 for the 3b preview) and routes through
counting and memory heads that were never trained. Both failures are silent, which
is why `tests/unit/test_switch_type_default.py` pins the seam.

|  | SingleSwitch (`"single"`) | MultiSwitch (`"multi"`, **default**) |
|---|---|---|
| Transitions per request | one: base -> adapter | arbitrarily many: base -> A -> B -> base -> ... |
| Mechanism | +/-gain cumsum over one attention head | counting head + Kerdock/DG coded memory head |
| Two control tokens in one sequence | averages them and mis-routes | the case it exists for; resolved latest-wins |
| KV-cache slots consumed | `num_cache_layers == 1` | `num_cache_layers == 2` (counting + memory) |
| Return to base mid-sequence | no mechanism (`vllm/switch/single.py:139-140`) | yes, with a base-reset control token (section 3) |

### The config surface

All of it is plain attributes on `GraniteSwitchConfig`, with defaults, read via `getattr` so an older checkpoint still loads:

| field | default | what it sets | defined |
|---|---|---|---|
| `switch_type` | `"multi"` | selects this engine; validated against `("single","multi")` | `config.py:70,122-127` |
| `ms_code_m` | `6` | Kerdock order. m=6 -> code dim N=64, capacity 2048 | `config.py:75,129` |
| `ms_code_type` | `"kerdock"` | codebook family | `config.py:76,130` |
| `ms_memory_gain` | `28.0` | scale on the memory head's keys; 16.0 is the smallest that still retrieves exactly at capacity, 8.0 does not | `config.py:77,131` |
| `ms_counting_head_dim` | `32` | counting head dim; floored at 32 for FlashAttention | `config.py:78,132` |
| `adapter_token_ids` | -- | `num_adapters` ids, or `num_adapters + 1` with a leading base-reset slot | `config.py:134-158` |
| `adapter_substitute_token_ids` | -- | what each control token is rewritten to; required once `num_adapters > 0` | `config.py:159-192` |

### What the checkpoint carries that SingleSwitch's does not

Two registered buffers, both `persistent=True` deliberately:

| buffer | shape | why persistent |
|---|---|---|
| `codebook` | `[2048, 64]` fp32 unit-norm codewords, ~512 KB | `from_pretrained` materializes modules on the meta device and fills only tensors present in the checkpoint. A non-persistent buffer comes back **all zeros**, which makes every memory logit 0, the retrieval softmax uniform, and each position average the expert ids it can see instead of selecting the latest (`hf/switch/multi.py:209-220`). |
| `control_to_substitute_lut` | vocab-sized `long`, `-1` at non-control ids | same failure mode, worse: `-1` is the "not a control token" sentinel, so an all-zero LUT rewrites *every* token id to 0 (`hf/switch/multi.py:273-282`). |

The switch also owns two logical cache slots rather than one, and the decoder layers are offset past them:

```python
# src/granite_switch/hf/modeling_granite_switch.py:343-344
layer_offset = self.switch.num_cache_layers      # 2 for MultiSwitch, 1 for SingleSwitch
num_decoder_layers = config.num_hidden_layers - layer_offset
```

## 2. The engine: two attention heads and a token rewrite

There is no router network and no learned parameter. The whole engine is arithmetic dressed as two single-head attention calls, so it inherits the backend's batching, masking and KV cache for free.

*The three stages of one `MultiSwitch.forward`.*
```
  input_ids  ────────────────────────────────────────────────┐
      │                                                      │ (read, never modified)
      ▼                                                      │
  ┌─ STAGE 1: counting head ─────────────────────────┐        │
  │  anchor at position 0 holds  V = 1               │        │
  │  every control token holds   K unmasked, V = 0   │        │
  │  every token queries one-hot on dim 0            │        │
  │            ──►  signal = 1/(1+n)                 │        │
  │            ──►  n = round(1/signal - 1)          │        │
  └──────────────────────┬───────────────────────────┘        │
                         │  n  =  write address               │
                         ▼                                    │
  ┌─ STAGE 2: coded memory head ─────────────────────┐        │
  │  at a control token:  K = code(n) * 28           │        │
  │                       V = expert_id              │        │
  │  at every token:      Q = code(n)                │        │
  │            ──►  softmax concentrates on address n│        │
  │            ──►  raw = expert_id written there    │        │
  └──────────────────────┬───────────────────────────┘        │
                         │  round + clamp[0, num_adapters]    │
                         ▼                                    ▼
              adapter_indices [B,S]              ┌─ STAGE 3: token exchange ─┐
                                                 │  control id ──► substitute│
                                                 └──────────┬────────────────┘
                                                            ▼
                                                  modified_input_ids [B,S]
                                                  (what the decoder embeds)
```

### A real trace

Three adapters, no base-reset slot, run on CPU against the real `MultiSwitch`. Control ids are 101/102/103, substitutes 11/12/13. Output is verbatim (section 10 has the script):

```
capacity 2048  memory_dim 64  counting_head_dim 32  memory_head_dim 64
offset 1  num_cache_layers 2

input_ids        [50, 101, 60, 61, 102, 70, 71]
adapter_indices  [ 0,   1,  1,  1,   2,  2,  2]
modified_ids     [50,  11, 60, 61,  12, 70, 71]
```

| pos | input id | what it is | n (write address) | expert_id written | adapter_index |
|---|---|---|---|---|---|
| 0 | 50 | ordinary token, *and* the counting anchor | 0 | -- | 0 -- base |
| 1 | 101 | `<\|a1\|>` control token | 1 | 1 | 1 |
| 2--3 | 60, 61 | ordinary | 1 | -- | 1 -- reads address 1 |
| 4 | 102 | `<\|a2\|>` control token | 2 | 2 | 2 |
| 5--6 | 70, 71 | ordinary | 2 | -- | 2 -- reads address 2 |

### Stage 1, in detail

Position 0 is the anchor. It is derived from `cache_position` / `positions`, not from a special `<|init|>` token, and it is not hidden from the decoder:

```
# src/granite_switch/hf/switch/multi.py:329-331
is_counting_anchor = (cache_position == 0).unsqueeze(0).expand(bsz, -1)
```

Then the three tensors. Keys default to a mask value and are un-masked only at the anchor and at control tokens; values are 1 only at the anchor; queries are one-hot on dim 0, so `Q.K` reduces to `K[0]`:

```
# src/granite_switch/hf/switch/multi.py:348-376  (fp32, mandatory -- see section 4)
key_states_count   = full((B,1,S,32), -1e9)               # masked by default
key_states_count[:,0,:,0] = where(anchor_or_control, 0, -1e9)  # un-mask participants
value_states_count[:,0,:,0] = is_counting_anchor.float()       # v = 1 at the anchor only
query_states_count[:,0,:,0] = 1.0                              # one-hot => Q.K = K[0]
```

A query at a position that can see the anchor plus `n` control tokens attends uniformly over `1 + n` keys, of which exactly one carries `V=1`. The output is therefore `1/(1+n)`, and inverting it recovers the integer:

```python
# src/granite_switch/hf/switch/codes/__init__.py:19
count = 1.0 / counting_signal.float() - 1.0
return torch.clamp(torch.round(count).long(), 0, capacity - 1)
```

> **Why `-1e9` and not `-inf`**
>
> IEEE-754 defines `0 * +/-inf = NaN`, and the one-hot query has zeros at exactly the dimensions where the mask sits. `-1e9` gives `0 * -1e9 = 0` and still `exp(-1e9) ~= 0` in the softmax, so it masks without poisoning the product (`hf/switch/multi.py:72-76`, `vllm/switch/multi.py:108-112`).

Note what `n` does *not* depend on: absolute position, sequence length, or batch composition. Only the count of control tokens causally before the query. That is what makes the same routing survive prefill, decode, chunked prefill and co-batching.

### Stage 2, in detail

The address is turned into a Kerdock/DG codeword by a plain buffer lookup, so there is no numpy and no per-call loop on the forward path:

```python
# src/granite_switch/hf/switch/multi.py:476-518
all_code_vectors = self.codebook[write_addresses]              # [B,S,64] unit-norm
key_states_memory[:,0,:,:64]   = all_code_vectors * 28.0 * write_mask   # control tokens only
value_states_memory[:,0,:,0]   = expert_ids * is_control_token
query_states_memory[:,0,:,:64] = all_code_vectors              # every token asks
```

Kerdock/DG codewords have provably low mutual coherence, so a query of `code(n)` against keys of `code(m) * 28` produces a softmax that concentrates on `m == n`. The attended value is the `expert_id` most recently written at address `n`; `round` and `clamp[0, num_adapters]` make it an integer index (`hf/switch/multi.py:539-541`).

### Stage 3: token exchange

Selection reads the *original* ids; the rewrite happens last, so the decoder embeds a clean sequence and never knows a control token existed:

```python
# src/granite_switch/hf/switch/_token_exchange.py:47-60
def apply_token_exchange(lut, input_ids):
    if lut is None:
        return input_ids            # unconfigured checkpoints pass through unchanged
    sub_id_per_pos = lut[input_ids]
    is_control = sub_id_per_pos >= 0
    return torch.where(is_control, sub_id_per_pos, input_ids)
```

Branch-free on purpose: the vLLM decoder is wrapped in `@support_torch_compile`, which forbids `tensor.any()` short-circuits. Both backends import this one module, so the behaviour is identical across them.

> **Token exchange, not a hiding matrix**
>
> MultiSwitch does not use the group-based KV-hiding path. `grep -rn hiding src/` returns two comments and no mechanism; the control token is neutralised by having its id replaced before embedding, which is why `modified_ids` above shows `11` and `12` where the control tokens were.

## 3. Latest-wins routing, and returning to base

Because each position reads the value at *its own* address, and addresses only ever increase, routing is piecewise-constant and changes only at a control token. It never drifts back to base on its own.

> **What the adapter_indices vector can never look like**
>
> `[0,0,1,1,**0**,1,1,...]` is unreachable. An isolated 0 in the middle of an adapter run requires a control token sitting at that position that writes expert id 0 -- which means a checkpoint composed with `--base-reset-token` and a caller who places `<|base_reset|>` there.

The engine has always supported that layout: it reads the *length* of `adapter_token_ids` and derives a static offset once, in `__init__`, so `forward` stays branch-free:

```python
# src/granite_switch/hf/switch/multi.py:182-188
if ctrl_ids is not None and len(ctrl_ids) == num_adapters + 1:
    self._expert_id_offset = 0   # base-reset layout; argmax IS the expert id
else:
    self._expert_id_offset = 1   # no base slot; slot i fires adapter i+1
```

Same three adapters as section 2, recomposed with a leading base slot (ids 100..103). Verbatim output:

```
offset(base-reset layout) 0

input_ids        [50, 101, 60, 100, 70, 102, 80]
adapter_indices  [ 0,   1,  1,  0,  0,   2,  2]
                        │        │            └─ <|a2|> fires adapter 2
                        │        └─ <|base_reset|> writes expert id 0 -> back to base
                        └─ <|a1|> fires adapter 1
```

| layer | who does what |
|---|---|
| compose | `--base-reset-token` (multi only) prepends `<\|base_reset\|>`, giving `num_adapters + 1` control ids with the base slot first. Off by default. `compose_granite_switch.py:655-660`, `tokenizer_setup.py:178-209`, gated by `validate_base_reset_switch_type`. |
| chat template | never emits it. `configure_chat_template` only knows about adapters, and base-reset is not an adapter. |
| `Conversation` | never emits it either (`conversation.py:62-67`). Under `PRESERVE_MIXED_HISTORY` an earlier adapter therefore carries forward into the new user turn rather than reverting to base. |
| caller | places it, or the gap does not happen. |

Ordinary aLoRA chat does not need it: one `apply_chat_template` call emits a control token for the current turn only, so prior turns carry none and already read as base. It matters when a single sequence holds several control tokens -- agentic per-step switching, or a preserved multi-turn history.

## 4. The two backends, and the one thing that differs

The two files are deliberate twins -- same stages, same order, same masking constant, and the codes and token-exchange helpers are imported from the HF side as the single source of truth (`vllm/switch/multi.py:99-106`). What differs is the attention they call, and therefore the dtype they are allowed to compute in.

|  | HF `hf/switch/multi.py` | vLLM `vllm/switch/multi.py` |
|---|---|---|
| Attention | `ALL_ATTENTION_FUNCTIONS["sdpa"]`, hand-built Q/K/V | two real `vllm.Attention` modules (`switch.multi.0`, `switch.multi.1`) |
| Q/K/V dtype | `torch.float32`, forced (`:348-376`, `:483-518`) | `vllm_config.model_config.dtype` -- in practice bf16 (`:151`, `:362`) |
| Backend override | SDPA forced for **both** heads, not the model's configured backend | whatever attention backend vLLM selected |
| Autocast | explicitly disabled around both calls (`:450-455`) | not applicable |
| Causal mask | built internally in fp32 over `kv_len`, ignoring the model's mask | vLLM's own attention metadata |
| KV cache | `DynamicCache` slots `layer_idx` / `layer_idx + 1` | paged KV cache, two cache groups |
| Positions | `cache_position`, defaults to `arange(seq_len)` | `positions`, **required** -- `None` raises |
| Counting exact to | n = 4095+ (no failure in the range tested) | **n = 188**; 189 aliases |

### Why the HF side forces so much

Three separate corruptions were traced to letting the heads run in the model's dtype, and each one produced a plausible-looking wrong answer rather than an error:

1. **The model's bf16 attention mask.** Feeding a bf16 mask to fp32 Q/K/V makes SDPA run the whole attention in bf16; the recovered `n` comes out off by one, so routing lags a position -- `[T,A,T,B,T]` gives `[0,0,0,1,1]` instead of `[0,1,1,2,2]` (`:405-416`).
2. **The model's configured backend.** An earlier version dispatched the memory head through `config._attn_implementation`. flash_attention_2 requires fp16/bf16, silently downcast, destroyed the codebook separation, and routed every token to base -- the isolated switch tests passed while the full model returned all-zero indices (`:392-402`).
3. **Enclosing bf16 autocast.** The memory keys are `code(n) * 28` and the mask is large and negative; in bf16 these overflow the softmax to NaN intermittently, per kernel, which rounds and clamps to 0 (`:442-449`).

The HF mask is also built over the *key* length, not the query length. That is load-bearing rather than cosmetic: at a decode step with `q_len=1` and `kv_len=6`, a `q_len x q_len` mask made SDPA raise `"(*bias): last dimension must be contiguous"` -- a shape complaint, not a stride problem. Every forward-only test passed because prefill has `q_len == kv_len` (`:417-436`).

### Why the vLLM side cannot do the same

Its heads are paged-KV attention modules, so their Q/K/V dtype is fixed by the engine's KV-cache dtype. Serving a bf16 checkpoint quantizes the `1/(1+n)` signal in the cache, and 188 is the arithmetic consequence rather than an oversight. That is section 8's first limit.

## 5. Under vLLM continuous batching

The switch writes none of vLLM's per-request machinery. It builds constant Q/K/V and calls `vllm.Attention`, which means per-request isolation is inherited, not implemented.

### The anchor is the whole trick

vLLM flattens a batch into one `[total_tokens]` tensor, but `positions` restarts at 0 for each request. So `positions == 0` puts exactly one counting anchor in each request -- which is precisely what `1/(1+n)` needs.

*Two requests in one flattened forward. Each gets its own anchor and its own count.*
```
flat index :  0    1    2    3    4  │  5    6    7    8
positions  :  0    1    2    3    4  │  0    1    2    3
             ▲ anchor (req A)         │ ▲ anchor (req B)
input_ids  :  t  <|a1|>  t    t  <|a2|>│  t    t  <|a1|>  t
n          :  0    1    1    1    2  │  0    0    1    1
idx        :  0    1    1    1    2  │  0    0    1    1
                 request A            │     request B
                 (attention confined per request by vLLM, not by us)
```

This only holds if the caller forwards the *real* positions, so `forward` refuses to synthesize them:

```python
# src/granite_switch/vllm/switch/multi.py:375-383
if positions is None:
    raise ValueError(
        "MultiSwitch.forward() requires per-request `positions`: ...")
```

The comment above it states the failure being prevented: a fabricated `arange(total_tokens)` anchors only the first request, every later request counts against a missing baseline, and the mis-routing is silent -- it only diverges once a request carries three or more control tokens (`:366-374`).

### Decode carries the adapter through the cache

At a decode step the control token is long gone from `input_ids`. Routing still works because both heads are real paged-KV attention: the single new query attends over the cached anchor and control-token keys, recovers the same `n`, and reads back the same expert id. The cache is the mechanism, not an obstacle.

> **Measured, not argued**
>
> `granite-switch-internal/vela_yamls/ci/multiswitch-serving-routing.yaml` on 1 GPU: **zero of 1623 checked positions mis-routed**, across five prompts at 100% trace coverage each (24/24, 61/61, 413/413, 614/614, 396/396), over 23 pure-decode and 6 mixed forwards with up to 5 concurrent requests in one forward. The 23 is the same count as a historical 23/23 *failure*, so this exercises the workload that once broke. Recorded at `vllm/switch/multi.py:77-85`.

Two of those prompts generated different *text* solo versus co-batched while their routing was identical. That is bf16 sampling landing on a near-tie, not a routing leak -- which is why the e2e test reports text differences and asserts only on routing. Do not try to establish routing correctness from generated text: a control token routes the prefill region, so continuations differ between adapters even when decode reverts to base.

### Chunked prefill

Chunked prefill splits a long prompt across forwards, so a control token and the tokens it routes can land in different batches. It works for the same reason decode does -- the counting head reads `positions`, which vLLM keeps per request across chunks. Exercised on purpose by the serving e2e with `--enable-chunked-prefill` and a small `--max-num-batched-tokens`.

### What is regenerated versus what persists

| regenerated every forward (the switch stores nothing) | persisted by vLLM across forwards |
|---|---|
| `is_counting_anchor`, `is_control_token`, `expert_ids`, all six Q/K/V tensors, `write_addresses`, `adapter_indices` | the two heads' keys and values, in the paged KV cache, under vLLM's own block tables |

## 6. Multi-turn: cache reuse and the Conversation API

### The one rule

> vLLM reuses a cached block when **the token prefix leading to it is identical**. Nothing else is consulted -- not the adapter, not the turn, not the request. And because adapter routing is computed from those same tokens, identical tokens always mean identical routing, so a reused block is always exactly what recomputing would have produced.

Adapters are weights inside the checkpoint selected by control tokens in the stream, so they are invisible to the cache key: `hash_block_tokens` mixes the parent hash, the block's token ids, and `extra_keys` (vLLM LoRA name, multimodal ids, `cache_salt`, prompt-embeds digest) -- none of which Granite Switch sets.

So the whole question reduces to: *where does turn 2's token stream stop matching turn 1's?* And the answer is the missing control token, because **one render emits one control token**, for the current turn's adapter (`tokenizer_setup.py:302-306`, asserted at `tests/composer/test_chat_template.py:453` -- "Only one control token in the entire output").

```
turn 1 sent:  ... glide path? <|uncertainty|>certainty> <|end_of_text|> ... assistant: answer1
turn 2 sent:  ... glide path? <certainty> <|end_of_text|> ... assistant: answer1 ...
              └─── reused ───┘✗ diverges here -> everything after is prefilled

measured on the real Granite template: 59 shared chars of turn 1's 140
```

> **The wrong mental model**
>
> "Switching adapter invalidates the cache" is backwards. Appending the *new* turn's control token is harmless -- routing is causal, so nothing before it moves. **Losing the old one is the whole story.**

### The two policies

`src/granite_switch/conversation.py` exists to make that choice explicit rather than accidental:

|  | `RE_PREFILL` (default) | `PRESERVE_MIXED_HISTORY` |
|---|---|---|
| How turn 2 is built | reuse the base-demoted ids already sent, plus this turn's delta | the ids already sent, plus this turn's delta |
| Earlier control tokens | dropped (history reads as base) | kept |
| History reads as | base, and is recomputed from the first adapter turn on | the adapter that produced it |
| Transport | token ids only -- `/v1/completions` with `prompt=[ids]`, or `model.generate(input_ids=...)` | same |
| Requires | -- | -- (no technology enforcement; see below) |
| Prefix-cache reuse | deep history reused as ids (lags one turn); a LoRA/earlier-trigger turn re-prefills | more -- every turn's control token is kept, so no turn re-prefills |

The delta is derived, never hardcoded -- the module renders the conversation with and without the new turn and subtracts, so it inherits any future template change for free. Whether the new turn can be appended is decided by *where its control token lands*; when it cannot, the turn falls back to a full render instead of raising:

```python
# src/granite_switch/conversation.py
prev = render(messages WITHOUT the new turn)          # what has already been sent
full = render(messages WITH the new turn, adapter=...)

if _appendable(full, prev):                            # control token lands in the delta
    delta = full[len(prev):]                           # append it to the reused ids
else:
    full_render()                                      # LoRA (token at index 0), or
                                                       # a non-append-only template -> raise
```

Under RE_PREFILL the reused prefix is the history *demoted to base* (control tokens dropped); under PRESERVE it is the history *with control tokens kept*. Both reuse the exact ids already sent, so the prefix cache hits. RE_PREFILL's base form of a turn is only sent the turn after it, so its id reuse lags one turn.

### The API surface

Three calls per turn. The caller never writes a control token, a token id, or a payload.

```python
from granite_switch import Conversation, KVHistoryPolicy

conv = Conversation(tokenizer, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY,
                    config=model.config)
conv.user("Rate it. <certainty>")          # 1. add the user message
ids = conv.build_prompt(adapter="unc")      # 2. render + place the control token
conv.record_answer(model_ids, adapter="unc") # 3. commit what the model produced
```

| member | what it does | mutates the transcript? |
|---|---|---|
| `user(text)` / `system(text)` | appends a message | `messages` only |
| `build_prompt(adapter=...)` | returns the ids to send | no -- a discarded prompt (a judge or guardian call) leaves the conversation untouched |
| `record_answer(ids_or_text, adapter=...)` | commits the answer and turn-end tokens | yes -- pass the model's *ids*; re-encoding detokenized text does not always reproduce them |
| `completion_payload(...)` | a ready `/v1/completions` body with `prompt=[ids]` -- the only transport | same as `build_prompt` |
| `generated_control_tokens` | `[]` = checked, none. `[ids]` = the model emitted these. `None` = not checkable, the answer was recorded as text | read-only |

There is no `chat_payload`: the `/v1/chat/completions` endpoint re-renders and re-tokenizes server-side, which cannot carry the ids both policies now reuse as a prefix. (The endpoint itself is unaffected; a caller who wants it just does not route through `Conversation`.)

`build_prompt` returns a `PromptTokenIds` -- a `list[int]` subclass carrying one fact a bare list cannot: `requires_token_ids`, now **always true**. Both policies reuse the ids already sent as a prefix, so re-rendering them server-side reproduces a different prefix and silently degrades cache reuse, with no error and no symptom beyond a fallen cache-hit rate.

### When a turn cannot be appended, it falls back to a full render

Two placements put this turn's control token *inside* the already-sent prefix, so no delta can carry it. Neither raises: the turn falls back to a full render -- a re-prefill under PRESERVE, which drops the preserved control tokens for that turn and recomputes history as base. Nothing detects the adapter technology; the append test (`_appendable`) simply fails on control-token position.

**LoRA, any turn.** A LoRA adapter's control token is emitted at sequence position 0, inside the already-sent prefix on every turn after the first. It can never be a delta, so a LoRA turn always full-renders (`conversation.py`, `_appendable`).

**The aLoRA trap: the trigger is in an EARLIER turn.** When the invocation text sits in a past user message, the control token is inserted into the history region, not the new turn:

```
char                                45  46
                                     │  │
already sent : ...<|end_of_role|>Rate it. < c e r t a i n t y > <|end_of_text|>
new render   : ...<|end_of_role|>Rate it. <|u n c|> c e r t a i n t y > <|end_of_text|>
                                    same  ^^^^^^^ INSERTED here, inside the sent region

both renders start with "<" (the token the control token replaces), so they agree to
char 45 and disagree from char 46 -- of 134. The new render is not the old render with
something appended; it is the old render with a token stuffed into the middle. There
is no tail to slice off, so the turn re-prefills.
```

**The one case that still raises: a template that is not append-only even with NO adapter.** Granite 4.2 with thinking-truncation on rewrites history when a turn is added, so even a no-adapter render fails the prefix relation. That is the discriminator between a placement fallback (re-prefill) and a template that cannot be served: the latter raises, naming the flag to set, because a silent re-prefill would hide it (`conversation.py`, `_explain_no_prefix`).

### The measured matrix

Turn 1 is always aLoRA `unc` with `"Rate it. <certainty>"`, answered. Only turn 2 varies.

| turn-2 adapter | turn-2 text | result under PRESERVE |
|---|---|---|
| `unc` -- aLoRA, trigger in *this* turn | `"As JSON, how sure? <certainty>"` | appends -- 41 ids, control tokens at 8 and 33 |
| `unc` -- aLoRA, trigger in an *earlier* turn | `"As JSON."` | re-prefills -- token lands at 8 (history region); 32 ids, `reprefills`=1 |
| `req` -- aLoRA, assistant boundary | `"As JSON."` | appends -- 32 ids, control tokens at 8 and 29 |
| `ctx` -- LoRA | `"As JSON."` | re-prefills -- token at position 0; 32 ids, `reprefills`=1 |
| `None` -- base turn | `"As JSON."` | appends -- 32 ids, control token at 8 only (turn 1's, preserved) |

### Cheat sheet

| I have... | policy | what to do |
|---|---|---|
| a LoRA adapter, any turn | either | nothing special; under PRESERVE that turn re-prefills instead of preserving (token at position 0) |
| an aLoRA whose trigger is the assistant role marker | either | nothing special -- placement is always in the new turn |
| an aLoRA with a user-message trigger, turn 1 | either | nothing special -- turn 1 has no prefix to protect |
| an aLoRA with a user-message trigger, turn >=2 | `PRESERVE` | put the trigger text in *this* turn's user message, or the turn re-prefills |
| ...and I cannot edit the user's message | `RE_PREFILL` | accept recomputing history; that policy re-renders anyway |
| mixed adapters across turns | `PRESERVE` | fine -- an unappendable turn (LoRA, or an earlier trigger) re-prefills on its own |

### The HF equivalent

Under HF, per-turn `generate()` starts a brand-new cache, so turn 1 is prefilled again. Threading `past_key_values` into the next `generate()` is supported by the model (`modeling_granite_switch.py:283-284, 413` -- the model creates a `DynamicCache` whenever `use_cache` is set and returns it) and used by no caller.

## 7. Control-token placement

Placement is entirely the chat template's decision. The composer writes the rule into the template; `Conversation` never hardcodes a position, and the template never mentions `switch_type`.

| adapter kind | control token goes | in the NEW turn? | rule |
|---|---|---|---|
| LoRA | sequence position 0, before the first role marker | never | routing carries forward, so position 0 is how "applies to everything" is expressed |
| aLoRA, user-message trigger | before the trigger text, in the *last* message containing it | only if the trigger is in this turn | the adapter was trained to switch on exactly that text |
| aLoRA, assistant-boundary trigger | just before the generation prompt | always | fallback path, `alora_target_idx == -1` |

```python
# two-pass Jinja placement -- src/granite_switch/composer/tokenizer_setup.py:302-311
pass 1  (before the message loop)  scan for the last user message containing the decoded
                                   invocation text; store its index in ns.alora_target_idx
                                   (stays -1 when not found)
pass 2  (inside the loop)          when the current message is the target, split content on
                                   the invocation text and rejoin with the control token
                                   inserted before the final occurrence
fallback (before add_generation_prompt)  fires when alora_target_idx == -1
```

### Why a rendered aLoRA reads `<|unc|>certainty>`

The runtime swap replaces the control token's embedding with the *first invocation token's* embedding. So the text emitted after the control token must omit that first unit, or the stream carries it twice -- an out-of-distribution pattern at exactly the swap site:

```
trained on:   <requirements>req1              ['<', 'requirements', '>', ...]
rendered:     <|req_check|>requirements>req1
              ^^^^^^^^^^^^^ becomes '<' at runtime, reconstructing the invocation
```

On this branch the omitted unit is the first **character**, sliced in the emitted Jinja itself:

```
# src/granite_switch/composer/tokenizer_setup.py:478  (Pass 2, inside the emitted template)
= _parts[0] + ns.adapter_token + ns.adapter_invocation_text[1:] + _parts[1]
```

> **The character rule and the token rule are not the same rule**
>
> They coincide only when the first character tokenizes alone. Measured across six cached Granite tokenizers (4.1-3b, 4.1-8b, 4.0-micro, 4.0-h-tiny, switch-4.1-3b-preview, 3.3-2b-instruct), the first token of every library invocation -- `<requirements>`, `<certainty>`, `<guardian>`, `<context>` -- is `'<'`, so the two rules agree on everything currently shipped. The exposure is **latent, not active**.
>
> Two corrections to earlier write-ups of this, both worth knowing: `<context>` is *not* a counterexample -- `'<context'` is a real vocab entry (id 35628 on granite-4.1-3b) but BPE never produces it for `'<context>'`, which encodes as `['<', 'context', '>']`. The real counterexamples are Granite's structural markers, which are single vocab entries:
>
> ```
> </documents>   ['</documents>']    token tail=''   char tail='/documents>'
> </think>       ['</think>']        token tail=''   char tail='/think>'
>
> under the character rule, a </documents> invocation renders FOUR tokens
> where the adapter was trained on ONE.
> ```

The token-based rule exists but is **not on this branch**: `c775046` landed it here, `7a7cea3` reverted it, and it now lives as `41a6883` on `bugfix/alora-invocation-tail`, branched off `origin/main`. So do not look for an `alora_invocation_tail()` helper in this tree -- there is none.

> **A checkpoint's template and its buffers are a matched pair**
>
> With a LUT (what the current composer builds) the emitted tail omits the invocation's first unit. With no LUT -- the published previews, which use a hiding matrix -- the control token keeps its own embedding and the *full* invocation text must follow it. The engine bridges both, because `apply_token_exchange` returns its input unchanged when the LUT is `None`. Crossing the two degrades the adapter silently in either direction, and **nothing validates the pairing today**.

## 8. Limits that are real today

### bf16 caps a vLLM request at 188 retained control tokens -- *(re-prefilled client-side only)*

The counting signal takes the KV-cache dtype under vLLM (section 4). bf16 carries 8 significand bits, so values near `1/(1+n)` are spaced about `2^-8` apart relatively while consecutive addresses differ by `1/(1+n)`. Past n = 188 those cross. Verbatim:

```
torch.bfloat16   first mis-recovered n: 189
torch.float16    first mis-recovered n: 1464
torch.float32    first mis-recovered n: None

  n=   0  1/(1+n)=1.000000  bf16=1.000000  recovered=0
  n=   1  1/(1+n)=0.500000  bf16=0.500000  recovered=1
  n= 188  1/(1+n)=0.005291  bf16=0.005280  recovered=188   <-- last exact
  n= 189  1/(1+n)=0.005263  bf16=0.005249  recovered=190   <-- aliases onto 190
  n= 190  1/(1+n)=0.005236  bf16=0.005249  recovered=190
```

Crossing it does *not* show up as a lag. The same recovered address is used to write the codeword and to read it back -- `multi.py` builds both `k_memory` and `q_memory` from one `codebook[write_addresses]` -- so n = 189 writes at 190, its followers read 190, and routing is **correct**. The address set is still all-distinct. What breaks is the **collision** one count later, where 189 and 190 both recover as 190:

```
control tokens in request:  188        189             190
address set:                {0..188}   {0..188, 190}   {0..188, 190, 190}
                            distinct   distinct        COLLIDE
routing:                    correct    correct         broken
```

Two control tokens then key the same codeword, so the memory head returns the *mean* of the two expert ids they wrote, and `round()` lands on an arbitrary adapter for every token in both their spans. The functional cliff is therefore **190** control tokens in one request, not 189 -- and no error is raised at any point. Handling is client-side only:

```
# src/granite_switch/conversation.py:110, :320, :704-707
MAX_RETAINED_CONTROL_TOKENS = 188   # _reprefill above it; _assert_control_budget raises above it too
```

`Conversation` does not raise at that bound: it **re-prefills**. Past 188 control tokens the turn is rebuilt as a full render of the transcript, history's control tokens are dropped, the count resets to one, and preserving resumes from the new baseline. The cost is one full recompute of the conversation and the loss of history's adapter attribution -- both reported, via a `WARNING` and `Conversation.reprefills`, because a silent reset is indistinguishable from PRESERVE working. The raise survives only for a prompt that is *already* a full render and still over budget, which needs control-token text recorded into a message.

One constant, compared with `>` in both places, so 188 is legal and only 189 is not: a request carrying C control tokens gives its last one the write address C, so C = 188 is exactly recoverable and re-prefilling it would discard a prefix for nothing. Because the cliff is 190 rather than 189, 188 leaves **one token of headroom** for a control token the model emits mid-answer -- it joins this same request during decode, and one is survivable. Two in a single answer are not: they reach 190 and collide. Turn N+1 cannot repeat that, because `record_answer` retains the raw answer ids under PRESERVE, control tokens included, so the next `build_prompt` counts 190, trips the bound, and re-prefills before the request is sent. Buying more headroom means lowering `MAX_RETAINED_CONTROL_TOKENS` below the bf16 fact, which turns `tests/unit/test_counting_ceiling.py` red on purpose.

`Conversation` is a helper, not part of the engine. A request that goes straight to `POST /v1/completions` with token ids -- which is exactly what PRESERVE instructs callers to send -- never reaches any of this. Moving it into `MultiSwitch.forward` would cover every caller, and is not merely a behaviour change on the serving path: a host-side check there is absent from the compiled graph (see the debug-attribute row below), and a raise there kills the EngineCore for every in-flight request, since `vllm/v1/engine/core.py`'s `_process_engine_step` wraps `step_fn()` in no `try`.

### Two different ceilings, often confused

| limit | value | bounds | set by |
|---|---|---|---|
| codebook capacity | 2048 | distinct addresses a codeword can name -- the *memory* head | `ms_code_m` (Kerdock m=6 -> N=64, capacity 2048) |
| counting precision | 188 (bf16) | addresses the signal can be inverted to -- the *counting* head | KV-cache dtype |

Under vLLM the smaller one governs, so the advertised 2048 is not reachable there.

### `--kv-cache-dtype fp8` is unguarded -- *(no check)*

Two values in these heads flow through the paged KV cache and both exceed fp8 e4m3's maximum of about 448:

| value | magnitude | source |
|---|---|---|
| masking constant | `-1e9` | `vllm/switch/multi.py:112` (`_NEG_INF`) |
| memory key scale | `code(n) * 28` | `ms_memory_gain` default 28.0 |

An fp8 cache would saturate the mask and collapse the codeword separation the gain was chosen to guarantee. Nothing currently rejects the flag.

### One render emits one control token -- *(template limit, not engine)*

The template keys off a single scalar `adapter_name`, so one `apply_chat_template` call yields one token for `single` and `multi` alike. Reaching MultiSwitch's actual shape -- several control tokens in one sequence -- means assembling token ids rather than rendering, which is what PRESERVE does and what `tests/hf/test_multi_switch_alora.py` does by concatenating two renders. So: *one token per render*, not one token per request.

### Control tokens are freely generatable -- *(by design)*

There is no runtime suppression -- the model can emit any control token during generation, and feeding an append-only transcript back in changes routing. Filter or reject explicitly. `Conversation.generated_control_tokens` reports what it can see, and distinguishes "checked, none" from "not checkable" (the answer was recorded as text).

### Base-reset is composable but never placed for you

See section 3. `PRESERVE_MIXED_HISTORY` therefore yields A' -- the earlier adapter carrying into the new user turn -- rather than a real base gap, unless the caller inserts `<|base_reset|>` itself.

### Inherited limits that also apply

| limit | effect |
|---|---|
| Position 0 in a hiding group | `hidden_count` off by one, so a 1-position RoPE offset. Accepted: adapter detection is exact and RoPE is robust to small shifts. |
| TP row-parallel bias doubling | `SwitchedLoRALinear`'s bypass path passes bias to all TP ranks; after all-reduce it doubles. Not hit by Granite 4.0/4.1, which set `attention_bias=False` and `mlp_bias=False`. |
| Blocks, not tokens | only complete 16-token blocks are reusable, so reuse ends at the last full block before divergence -- up to 15 extra tokens recomputed. |
| Debug write addresses need eager | `_debug_write_addresses` / `_debug_counting_signal` sit behind `not torch.compiler.is_compiling()`, and the switch runs inside the `@support_torch_compile` region, so a default-configured server omits them. Every routing trace that reads them requires `--enforce-eager`; without it they are `None` and any assertion over them is vacuous. |
| Best-effort caching | blocks are evicted under memory pressure and cannot be pinned. A "hit" is never guaranteed; on a miss the same tokens recompute to the same values, so only cost changes. |
| ChatML / Granite 4.2 | the `<\|im_start\|>` template family has never been exercised end to end for either policy. |

## 9. What is tested, and where

Test counts are `grep -c '^\s*def test_'` over the tree. Two files shrank in the test-removal commit: `tests/hf/test_multi_switch.py` (12, was 15) and `tests/composer/test_switch_cache_layers.py` (3, was 4) -- each dropped duplicates of the 188-bound arithmetic already pinned in `tests/unit/test_counting_ceiling.py`. The same commit deletes `tests/vllm/test_newms_verify.py` and its worker -- a characterization suite that reported a C1-C4 routing scorecard without asserting a verdict. Its one load-bearing check, *C3 decode carry*, is now an assert in `test_multi_switch_serving.py`.

| file | tests | what it pins |
|---|---|---|
| **unit -- CPU, fast** |  |  |
| `tests/unit/test_counting_ceiling.py` | 8 | the 188 bound as arithmetic, per dtype; RE_PREFILL never accumulates |
| `tests/unit/test_conversation_policy.py` | 37 | the placement matrix (append vs re-prefill) and RE_PREFILL id reuse |
| `tests/unit/test_conversation_span_attribution.py` | 19 | which adapter owns which span |
| `tests/unit/test_conversation_transport.py` | 9 | ids-only transport; `chat_payload` removed |
| `tests/unit/test_conversation_generated_control_tokens.py` | 9 | model-emitted control tokens; the three-state report |
| `tests/unit/test_conversation_lora_rule.py` | 8 | a LoRA turn falls back to a full render |
| **HF backend -- CPU** |  |  |
| `tests/hf/test_multi_switch.py` | 12 | the engine across attention backends; `TestReturnToBaseCoded` |
| `tests/hf/test_multi_switch_alora.py` | 9 | two aLoRA adapters in *one* sequence |
| `tests/hf/test_multi_switch_buffers.py` | 4 | the codebook and LUT surviving a save/load round trip |
| `tests/hf/test_multi_switch_e2e.py` | 9 | compose -> load -> forward |
| `tests/hf/test_multi_switch_generate.py` | 6 | the `kv_len` mask fix; a prefill-only control token routing every later decode step |
| `tests/hf/test_multi_switch_mixed_tech.py` | 7 | LoRA + aLoRA in one checkpoint |
| `tests/hf/test_multi_switch_realistic.py` | 7 | chat template, real multi-turn, long prompts |
| `tests/hf/test_conversation_routing.py` | 4 | the policy's ids actually routing as claimed |
| **composer** |  |  |
| `tests/composer/test_base_reset_token.py` | 11 | the engine driven by exactly what compose emits |
| `tests/composer/test_switch_cache_layers.py` | 3 | `num_cache_layers == 2` accounting |
| `tests/composer/test_tokenizer_setup.py` | 21 | control-token lists, substitute ids, base-reset alignment |
| `tests/composer/test_chat_template.py` | 26 | placement; *one* control token per render (`:453`); the exact rendered form `<\|req_check\|>requirements>` |
| **vLLM -- GPU** |  |  |
| `tests/vllm/test_multi_switch.py` | 11 | the engine under `vllm.Attention` |
| `tests/vllm/test_multi_switch_serving.py` | 20 | chunked prefill, decode, mixed serving shapes; `TestDecode::test_decode_carries_adapter` (`:246,261`) asserts the prefill adapter holds on *every* decode step |
| `tests/vllm/test_conversation_routing.py` | 4 | the policy's ids routing under vLLM |
| **integration -- GPU, real composed checkpoint** |  |  |
| `tests/integration/test_multi_switch_serving_e2e.py` | 1 | every forward traced from *inside* the engine process; adapter index compared per position against ground truth |
| `tests/integration/test_multi_switch_vllm_generate.py` | 4 | a live vLLM engine via `llm.generate` |
| `tests/integration/test_multi_switch_alora_cache.py` | 1 | aLoRA prefix-cache behaviour |
| `tests/integration/test_conversation_prefix_cache.py` | 1 | the measured 80/99 vs 64/98 reuse |
| `tests/integration/test_conversation_concurrency.py` | 2 | PRESERVE under interleaved traffic and under eviction |

> **Why the serving e2e traces from inside the engine**
>
> An in-process monkeypatch cannot reach the EngineCore child, so the test installs a `sitecustomize.py` on `PYTHONPATH`. The file records the failure it was built for: **identical output text with 23/23 wrong decode routing**. That is the reason it asserts on traced routing rather than on generated text.

### GPU verification jobs

| job | what it runs |
|---|---|
| `granite-switch-internal/vela_yamls/ci/testall.yaml` | the full suite on 1 GPU |
| `granite-switch-internal/vela_yamls/ci/multiswitch-serving-routing.yaml` | the 1623-position decode-routing verdict in section 5 |
| `granite-switch-internal/vela_yamls/ci/multiswitch-decode-routing.yaml` | decode routing in isolation |
| `granite-switch-internal/vela_yamls/ci/multiswitch-compose-notebooks.yaml` | the three compose notebooks, with a sentinel check |
| `granite-switch-internal/vela_yamls/ci/multi-turn-multiswitch-notebook.yaml` | `multi_turn_multiswitch.ipynb` end to end |
| `granite-switch-internal/vela_yamls/ci/probe-first-token.yaml` | token-0 logprobs, to tell a sampling near-tie from an unapplied adapter |
| `granite-switch-internal/vela_yamls/ci/diagnose-serving-leak.yaml` | why `multiswitch_serving` saw batching change its output (it was the near-tie) |

### Tutorials

| notebook | covers |
|---|---|
| `tutorials/notebooks/hello_multiswitch.ipynb` | first contact: compose, load, route |
| `tutorials/notebooks/multi_turn_multiswitch.ipynb` | the two KV-history policies across turns |
| `tutorials/notebooks/multiswitch_serving.ipynb` | per-request isolation, paged KV, chunked prefill |

## 10. Reproducing every number here

The routing traces in section 2 and section 3, on CPU, no GPU and no checkpoint:

```python
import torch
from granite_switch.hf.switch.multi import MultiSwitch

class Cfg:
    adapter_token_ids = [101, 102, 103]
    adapter_substitute_token_ids = [11, 12, 13]
    vocab_size = 200
    hidden_size = 256; num_attention_heads = 4
    ms_code_m = 6; ms_code_type = "kerdock"
    ms_memory_gain = 28.0; ms_counting_head_dim = 32

sw = MultiSwitch(num_adapters=3, config=Cfg(), layer_idx=0).eval()
print(sw.capacity, sw.memory_dim, sw._expert_id_offset, sw.num_cache_layers)

ids = torch.tensor([[50, 101, 60, 61, 102, 70, 71]])
with torch.no_grad():
    idx, mod = sw(ids, torch.tensor(Cfg.adapter_token_ids))
print(idx.tolist()[0], mod.tolist()[0])

# base-reset layout: num_adapters + 1 ids, base slot first
class Cfg2(Cfg):
    adapter_token_ids = [100, 101, 102, 103]
    adapter_substitute_token_ids = [10, 11, 12, 13]
sw2 = MultiSwitch(num_adapters=3, config=Cfg2(), layer_idx=0).eval()
with torch.no_grad():
    idx2, _ = sw2(torch.tensor([[50, 101, 60, 100, 70, 102, 80]]),
                  torch.tensor(Cfg2.adapter_token_ids))
print(sw2._expert_id_offset, idx2.tolist()[0])
```

The counting ceiling in section 8:

```python
import torch
for dt in (torch.bfloat16, torch.float16, torch.float32):
    bad = [n for n in range(4096)
           if int(torch.round(1.0 / torch.tensor(1.0/(1.0+n)).to(dt).float() - 1.0)) != n]
    print(dt, "first mis-recovered n:", bad[0] if bad else None)
```

The Conversation figures in section 6: build a `Conversation` over `tests/shared/conversation_stubs.py`, then render `_messages[:_sent_messages]` and `_messages` and compare. The measured cache-reuse numbers come from `tests/integration/test_conversation_prefix_cache.py` on 1xA100 with a real composed checkpoint.

**Provenance.** Config, engine, backend and `Conversation` facts read from `src/` at `1c86ee9`. The section 2 and section 3 routing traces and the section 8 dtype table were produced by running the snippets in section 10. The decode-routing verdict is `granite-switch-internal/vela_yamls/ci/multiswitch-serving-routing.yaml` on 1 GPU, recorded at `vllm/switch/multi.py:77-85`. Cache-key and block-size facts were read from vLLM `main`, not the pinned `>=0.19.1,<0.21.0`. Character offsets and render strings in section 6 and section 7 are verbatim output from the stub tokenizer in `tests/shared/conversation_stubs.py`.

**Scope.** This replaces thirteen earlier MultiSwitch documents, several of which described a proposal rather than the code (the `Conversation` API docs), or engines that were removed (`cceaeb7`, "drop the coded suffix now that scan is gone"). Those are archived in the `granite-switch-internal` repository under `docs/`, on branch `docs/multiswitch-archive`, and are history rather than reference.
