# Decision Adapters

A decision adapter answers a labelled question with one token: the reply's first
token is read over the option labels and softmaxed. vLLM 0.32+ serves such reads on
its native `/v1/systemone` route; Granite Switch plugs its adapters into that route
through `granite_switch.vllm.structured_decisions` (a `vllm.general_plugins` entry
point, installed with the package).

Granite Switch keeps **no adapter-specific knowledge in code**. How an adapter is
asked — its marker, its turns, how options are written, its labels, which question
types it answers — is declared by the adapter in its `io.yaml`, resolved by the
composer, and stored in the checkpoint's `config.json`. Retraining an adapter with a
different marker or layout is an `io.yaml` change, not a Granite Switch change.

## 1. Declare the layout in `io.yaml`

Add a `decision:` block next to the adapter's existing io.yaml fields. This is the
block for the Granite Switch 4.1-8b decision adapter:

```yaml
decision:
  # Where the control token lands: marker (aLoRA invocation text in a user turn),
  # start (LoRA), generation (Shadow Residual, or an aLoRA whose invocation is the
  # assistant header). Optional at compose time: the composer derives it.
  activation: marker
  # marker activation: optional at compose time (taken from adapter_config.json's
  # alora_invocation_tokens, and checked against it if given). Required for a stamp.
  invocation_marker: "### DECISION"
  system_prompt: ~                 # optional system turn, used verbatim
  messages:                        # placeholders: {state} {instructions} {options} {marker}
    - role: user
      content: "State:\n{state}"
    - role: user
      content: "{marker}\nQuestion:\n{instructions}\n\nOptions:\n{options}\n\nReply with exactly one option label."
  option_line: "{label}: {text}"   # one line per option
  option_separator: "\n"
  empty_state: "(empty)"           # rendered when the request has no state
  labels: letters                  # A..Z; or an explicit list of single-token labels
  request_instructions: reject     # reject | system (escaped, after system_prompt)
  question_types:                  # the types the adapter answers; any other is a 400
    choice: {}
    noul:
      order: ["no", "yes"]         # render false first; answers map back to vLLM's order
      fallback_text: {"no": "No / false", "yes": "Yes / true"}
    score: {}
```

Quote `"yes"` and `"no"`: YAML reads them unquoted as booleans.

| Key | Meaning |
|---|---|
| `activation` | `marker`, `start` or `generation` (see the comment above). The composer derives it from the adapter and refuses a block that disagrees. |
| `messages` | The turns of one question, in order. Roles `user`/`assistant`; the last must be `user`. `{options}` exactly once; `{marker}` at most once, in a user turn (required for `marker` activation). Literal braces are `{{` / `}}`. |
| `option_line` | `{label}` and `{text}`, once each. `{text}` is the option's description, else its `fallback_text`, else its name. |
| `labels` | `letters` (A..Z), or a list of at least two single-token strings in the order the adapter was trained on. On vLLM 0.32 at most 26 of them are ever used (see section 5). |
| `question_types.<type>.order` | Optional, `noul` only: the order its `"yes"`/`"no"` options are rendered in. Default: vLLM's order (`yes` first). |
| `question_types.<type>.fallback_text` | Optional. Text for an option with no description, by option name. An option it does not name, or every option when it is omitted, falls back to its name (a `noul` question then shows `yes` / `no`). Omitting it is never an error, so set it when the adapter was trained on other text, as the 4.1-8b one was (`No / false`, `Yes / true`). |

Only `messages` and `question_types` are required. At compose time `activation` and
`invocation_marker` come from the adapter (a stamp needs them, see section 3); every
other key defaults to the value shown above. A `question_types` entry may be empty
(`choice: {}`).

All caller text (state, question instructions, option descriptions, request
instructions) is escaped before it is placed: every added token and every decision
marker gets a U+2060 word joiner after its first character, so caller text cannot
forge a turn, a control token, or a second activation. The block's own text is
trusted.

## 2. Compose

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-8b \
  --adapters ./decision --decision-adapter decision \
  --output ./granite-switch-decision
```

`--decision-adapter` is repeatable; the first is the default. For each, the composer
validates the block, resolves the aLoRA marker from the adapter's
`adapter_config.json`, renders a probe question through the composed chat template,
and fails unless exactly one `<|name|>` control token lands inside the question turn.
The result is written to `config.json`:

```json
"decision_adapter": "decision",
"decision_specs": {"decision": {"invocation_marker": "### DECISION", "messages": [...], ...}}
```

## 3. Or stamp an existing checkpoint

A checkpoint composed before decision support (for example the public
`ibm-granite/granite-switch-4.1-8b`) needs no recompose. Add the block to its
`io_configs/<name>/io.yaml`, then stamp it. The adapter's technology comes from the
checkpoint's `adapter_index.json`. The source adapter is not there to supply an aLoRA
marker, so an aLoRA block names its `invocation_marker`, or declares
`activation: generation` when its invocation is the assistant header.

```bash
python -m granite_switch.composer.decision_config --model ./granite-switch-4.1-8b \
  --decision-adapter decision
```

The stamp runs the same probe on the checkpoint's own tokenizer and only then rewrites
`config.json`.

## 4. Serve and ask

```bash
vllm serve ./granite-switch-decision --logprobs-mode raw_logprobs
```

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "./granite-switch-decision",
  "state": "My payouts have failed three times.",
  "questions": {"queue": {"type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {"payments": "Payout failures and refunds",
                 "account": "Login and account access", "other": null}}}}'
```

The strategy reads the default decision adapter. A request may pick another declared
one with `"chat_template_kwargs": {"adapter_name": "<name>"}`; the only other key it
accepts is `enable_thinking: false`. A server whose checkpoint declares no decision
adapter starts normally and answers `/v1/systemone` with an error naming the compose
and stamp commands. A declared adapter that does not activate stops the server at
startup.

`diagnostics.<question>.label_mass` is the probability the model put on the option
labels; near 1.0 means the adapter is active and the layout matches its training.

## 5. Limits on vLLM 0.32

vLLM checks every question before the strategy reads it. Its `build_question` caps a
question at `min(max_options, len(label_set))`, where `max_options` comes from the
strategy's `limits()` (read once at server start) and `label_set` is vLLM's own label
set for the question type:

| Type | vLLM 0.32 `label_set` | Most options a question can have |
|---|---|---|
| `choice` | A..Z | 26 |
| `score` | digits 0..9 | 10 levels |
| `noul` | `yes`, `no` | 2 |

A larger question is refused with a 400 ("at most N options for this model"), whatever
labels the adapter's spec declares. The strategy therefore reports
`max_options = min(widest label set, 26)`, so it never advertises options vLLM 0.32
cannot serve. A spec with fewer labels than that is enforced per question when it is
rendered ("only N labels").

Supporting more than 26 options takes three steps:

1. **Move to vLLM 0.33.** Raise the `vllm` extra and the `dev-latest` group past
   `<0.33`, and import the strategies from their new 0.33 location (see section 6).
   Return `DecisionLimits(..., labeled=False)`
   from `limits()`: the strategy labels options itself, and that flag makes vLLM cap a
   question at `max_options` alone instead of at its own `label_set`.
2. **Declare a wider label set the adapter was trained on.** Give the `decision:`
   block an explicit `labels` list (or a new label scheme) with more than 26
   single-token labels, in the adapter's trained order. Confirm with whoever trained
   the adapter that it saw those labels; untrained labels give unreliable reads.
3. **Read wide questions in chunks.** vLLM caps `logprob_token_ids` at 128 token ids
   per request, so a question with more than 128 options must be read as several
   requests over the same prompt (sharing the prefix cache) and its label logprobs
   merged before the softmax.

## 6. vLLM versions

The decision strategy supports **vLLM 0.32 only**, the line the `vllm` extra's
`<0.33` ceiling allows. `register_structured_decisions` runs in every Granite Switch
vLLM server and behaves per line:

| vLLM | Behaviour |
|---|---|
| < 0.32 | No structured-decisions API: registration is a silent no-op. |
| 0.32 | The strategy registers for every Granite Switch architecture. |
| ≥ 0.33 | The strategies module has moved, so the import fails. Registration logs a **warning** that the strategy is NOT registered, and `/v1/systemone` does not serve decision adapters. |

On 0.33 the strategy warns and does not fail, on purpose. The plugin loads in every
Granite Switch server, and most declare no decision adapter, so failing would break
ordinary serving for an opt-in feature. At plugin load the model config does not
exist yet, so the plugin cannot tell whether this checkpoint declares decision
adapters. Only an install outside the `<0.33` pin (for example `--no-deps`) reaches
this path, and the warning names it.

0.33 support is deliberately not added ahead of the repo's own vLLM 0.33 bump.
Importing the new path alone would claim support for a vLLM line on which neither
this strategy nor the model backend has GPU evidence. The bump itself is small; this
is what vLLM v0.33.0 changes for the strategy:

- `strategies.py` and `question_types.py` move from
  `vllm.entrypoints.generate.structured_decisions` to
  `vllm.entrypoints.openai.decisions`. Import the 0.33 path first, then fall back to
  the 0.32 one.
- The API the strategy subclasses is unchanged. `NextTokenStrategy`'s `_tokenizer`,
  `_read_request`, `_render`, `_read_input`, `_sampling_params` and `read` keep their
  signatures. `ReadContext` keeps its fields and `prompt_tail` is identical.
  `QuestionRead` gains only an optional `slot`. `LABELS`, `StructuredDecisionError`
  and `label_softmax` stay in `question_types`.
- `vllm.entrypoints.generate.label_reads.next_token_label_reads` does not move. It
  gains only an optional `slots` keyword.
- `DecisionLimits` gains `labeled: bool = True`, and `build_question` takes
  `labeled`. With `labeled=False` a question is capped at `max_options` alone (see
  section 5).
- Run the request battery on 0.33 on a GPU before the `vllm` extra's ceiling moves.
