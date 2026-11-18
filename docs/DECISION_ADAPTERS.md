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
| `labels` | `letters` (A..Z), or a list of at least two single-token strings in the order the adapter was trained on. vLLM 0.32 caps a choice at 26 options and a score at 10 levels. |
| `question_types.<type>.order` | `noul` only: the order its `"yes"`/`"no"` options are rendered in. |
| `question_types.<type>.fallback_text` | Text for an option with no description, by option name. |

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
