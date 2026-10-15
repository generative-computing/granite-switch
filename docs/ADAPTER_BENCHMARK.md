# Adapter Accuracy Benchmark

This page explains how to measure the accuracy of trained intrinsic adapters
through one granite-switch commit, how that compares with the same adapters
outside granite-switch, and how the results page is built.

## Background

Granite Switch embeds adapters into one checkpoint and serves them with vLLM.
A change to the composer, the chat template or the vLLM backend can change
what an adapter outputs, without any test failing.

The benchmark catches this. For one commit it:

1. composes already-trained adapters with that commit's composer,
2. runs each adapter's eval set through that commit's vLLM backend,
3. scores the outputs and adds one row to the results page.

Nothing is trained. The adapters and eval sets are fixed ("staged") once, so
two commits are always compared on the same inputs.

Each adapter is measured in three forms:

| Technology | Configuration |
|---|---|
| LoRA | all-linear, r=16 |
| aLoRA (activated LoRA) | all-linear, r=32 |
| SR (Shadow Residual) | q,o + MLP, r=32, cross-stream r=32, shared KV |

The base model, the intrinsics and their headline metrics are listed in
[adapters.yaml](../benchmarks/adapter_eval/adapters.yaml).

### Columns

The table has two sections. **Task Quality** is how well the adapters do
their tasks: per intrinsic, 10 columns in four groups:

| Group | Columns | What it shows | Per commit |
|---|---|---|---|
| granite-switch (vLLM) | LoRA, aLoRA, SR | the adapters composed with the commit, under its vLLM backend | yes |
| HF + PEFT | LoRA, aLoRA, SR | the same checkpoints without granite-switch, with Hugging Face transformers and PEFT | no |
| Base | one | the base model with no adapter, told the answer format | no |
| Gain ratio | LoRA, aLoRA, SR | (granite-switch − Base) ÷ (HF + PEFT − Base) | yes |

**Serving Quality - Throughput** is how fast they are served, per
technology rather than per intrinsic, in two blocks of 9 columns:

| Block | Each cell |
|---|---|
| Decode 128 tokens, no switching, batch 32 | decode tokens per second ([Decode throughput](#decode-throughput)) |
| Decode 512 tokens, switch size 32, concurrency 8 | the p95 seconds for an agent switching adapters to finish ([Agents switching adapters](#agents-switching-adapters)) |

Each block has the same three groups:

| Group | Columns | What it shows |
|---|---|---|
| granite-switch (vLLM) | LoRA, aLoRA, SR | the technology's adapters composed with the commit |
| PEFT (vLLM) | LoRA, aLoRA | the same adapters served by stock vLLM as PEFT LoRAs; SR shows `—`, as stock vLLM has no SR |
| Speedup | LoRA, aLoRA | how much faster granite-switch is, in whole percent (+34% is 1.34 times the tokens per second, or agents done in 1/1.34 of the time; a negative value is slower, -13% being agents done in 1/0.87 of the time); SR shows `—` |

Both blocks are measured in each commit's run, both engines on the same GPU,
by the switch benchmark's own scripts, copied into `benchmarks/` unchanged but
for one flag: how many GPUs an engine spreads over, for a model too big for
one (`tensor_parallel_size`, below).

HF + PEFT and Base are the **reference columns**. They show what the
checkpoints score outside granite-switch, and what the base model scores
alone. They do not depend on the commit, so they are computed once and
repeated on every row (see [Reference columns](#reference-columns)).

The **gain ratio** is the share of an adapter's gain over the base model that
granite-switch keeps. For example, with Base at 72.5, HF + PEFT at 84.7 and
granite-switch at 84.8, the ratio is (84.8 − 72.5) ÷ (84.7 − 72.5) = 1.01. If
a commit drops granite-switch to 78.6, the ratio falls to 0.50, while HF + PEFT
stays put: the commit broke something. A ratio needs a gain to divide by: when
HF + PEFT beats the base model by less than 1 point, the cell shows `n/a`.

### The page

- **One tab per base model.** The link remembers the tab, e.g.
  `.../benchmarks/#granite-4.2-3b`.
- **Filters** above the table, for every tab: **Show** picks the sections
  (Task Quality, Serving Quality); **Adapters** picks the intrinsics shown in
  Task Quality, and **Columns** its column groups. Those two grey out while Task
  Quality is hidden, and the page keeps at least one column on show.
- **The `i` next to a commit** opens its run details: the commit, the library
  versions (vLLM, torch, transformers), the GPU, each adapter checkpoint (ranks
  and the start of its weights checksum, never its storage path), and the
  reference run.
- A diagram at the top shows the pipeline: pick adapters, compose, evaluate,
  next to the reference path.

## How one run works

```
local machine                          GPU pod (Vela)
-------------                          --------------
submit.sh bench main
  cache check ── hit ──> stop
  render + submit job  ───────────>    clone the commit, uv sync --frozen
                                       find staged adapters, check formats
                                       convert SR activation (if needed)
                                       rename MLP weights (if needed)
                                       check weight names against the base
                                       compose #1: LoRA + aLoRA adapters
                                       compose #2: SR adapters
                                       generate (vLLM, greedy) per adapter
                                       score, print the results block
  follow the pod log   <───────────
  extract results block
  merge into data.json, render page
```

A few points matter:

- **Two composes per commit.** A checkpoint runs either dual-stream (SR) or
  single-stream (LoRA, aLoRA) decoders. The composer refuses to mix them.
- **SR activation may be converted.** The internal trainer turns SR on at
  the invocation tokens. Usually these are
  `<|start_of_role|>assistant<|end_of_role|>`; for guardian they are
  `<guardian>`, inside the user message. The composer turns SR on at one
  anchor token instead: the last token of the generation prompt
  (`<|end_of_role|>`). So before composing, the run gives each such
  checkpoint that anchor. The generated output is the same, because SR takes
  K/V from the base stream only: turning it on later in the prompt changes
  nothing the model generates from. The staged checkpoint is not changed. The
  run record lists the converted cells.
- **MLP weight names may be converted.** The internal trainer saves MLP
  weights one level up: `layers.0.gate_proj` instead of the base model's
  `layers.0.mlp.gate_proj`. The composer maps only the names it knows and
  leaves the rest out, without an error. Such an adapter would compose with
  its attention part only. So before composing, the run renames these
  weights in a per-run copy. Only the file header changes; the tensor bytes
  are copied as they are. The run record lists the renamed cells.
- **Every adapter weight must name a base weight.** After the renaming, the
  run checks each LoRA weight against the base model's weight names. A
  checkpoint with any unknown name becomes an error cell rather than being
  composed with part of its weights missing. SR's cross-stream weights have
  no base weight by design and are not checked.
- **The harness is not the commit's.** The job ships the local copy of
  `benchmarks/` inside itself. It runs with the commit's virtualenv. So every
  commit is measured by the same harness, even commits older than it.
- **Adapters are picked by name.** Each prompt is rendered with the chat
  template's adapter name. The run checks that the adapter's control token is
  in the prompt. An unknown name would otherwise fall back to the base model.
- **Results travel through the log.** The pod prints one JSON block between
  `=== ADAPTER_BENCH_RESULTS_BEGIN ===` and `=== ADAPTER_BENCH_RESULTS_END ===`.
  Predictions and full score reports stay on the storage volume.

### Results cells

Each (intrinsic, technology) pair is one cell. A cell is one of:

```json
{"accuracy": 0.867, "n": 450, "score_version": 1}
{"skipped": "adapter not staged"}
{"error": "generation failed"}
```

`score_version` is the version of the intrinsic's scoring that produced the
numbers (see [Re-scoring saved answers](#re-scoring-saved-answers)). Cells
from before it existed count as version 1.

A row also holds its decode throughput, per technology and engine:

```json
"throughput": {
  "lora": {"gs": {"tokens_per_s": 3984.2, "median_s": 1.028,
                  "runs_s": [1.03, 1.028, 1.027, 1.031, 1.026],
                  "batch": 32, "generated_tokens": 128, "prompt_tokens": 1,
                  "adapters": 4, "requests_per_adapter": {...},
                  "gpu": "NVIDIA A100-SXM4-80GB",
                  "preflight": {...}, "provenance": {...}},
           "native": {...}},
  "sr": {"skipped": "no adapter staged"}
}
```

An engine that failed holds `{"error": ...}`, e.g. a refused gate.

And its switching run, the same way: per technology and engine, the p95 and
median seconds to complete, the agents and waves, the switches, tool calls and
re-prefilled prompt tokens summed over the agents, and the settings.

Metrics are stored as fractions and shown as percentages (`86.7`). On the
page:

| Shown | Meaning |
|---|---|
| `86.7` | scored; hover for every metric. The best of an intrinsic's 7 accuracy columns is bold. |
| `0.98` | a gain ratio; hover for both gains |
| `n/a` | a gain ratio with too small a gain to divide by |
| `—` | skipped: no adapter or eval set for this cell (reason on hover) |
| `·` | not run: for a commit, e.g. a new intrinsic before an `--only` run; in the reference columns, not computed yet or out of date (reason on hover) |
| `error` | the run failed for this cell (reason on hover) |

## Reference columns

### What runs

One job computes all of them: 5 intrinsics × 4 columns, 20 cells. No
granite-switch commit is involved.

```
local machine                          GPU pod (Vela), 4 GPUs
-------------                          ----------------------
submit.sh reference
  cache check ── hit ──> stop
  render + submit job  ───────────>    virtualenv: pinned torch, transformers, peft
                                       unpack the SR model code (pinned commit)
                                       find staged adapters, check formats
                                       convert copies for PEFT (if needed)
                                       generate (HF, greedy), one cell per GPU
                                       score, print the reference block
  follow the pod log   <───────────
  extract reference block
  merge into data.json, render page
```

Per column:

- **LoRA:** the base model with the checkpoint loaded by PEFT.
- **aLoRA:** the same. PEFT turns the adapter on at its invocation tokens.
  The run checks that every prompt contains them; without them PEFT would
  run the base model.
- **SR:** a standalone Hugging Face implementation of the Shadow Residual
  model, with the checkpoint loaded by PEFT. The checkpoint's invocation
  tokens are dropped first, in a copy: PEFT would read them as aLoRA and keep
  the adapter off before them. The generated output does not change, for the
  same reason as the SR activation conversion above.
- **Base:** the base model, no adapter, with one more user turn at the end:
  an instruction naming the answer format its scorer expects. Only the
  adapters were trained on that format. Without it, the base model answers in
  prose and scores near zero: answerability 4.4, the share of rows whose
  expected label is neither answerable nor unanswerable. With it, it answers
  in the format, though without the quotes around an answerability label;
  since benchmark v2 the scorer accepts a label either way. For requirement
  check the instruction replaces the row's own terse last request; for the
  others it follows the conversation. The texts are private, like the judge
  prompt: they live in
  `local/base_instructions.json`, travel with each reference job, and the
  results record only their checksum.

The same as in the granite-switch run:

- the staged checkpoints and eval sets, and the MLP weight renaming;
- the prompts: the base tokenizer's chat template, with the same documents
  and tools;
- greedy decoding in bfloat16, with the same token budgets;
- the scorers.

One more check: every checkpoint weight must be in the loaded model, with its
saved value. PEFT loads a weight that names no module of the model without an
error. Without this check, a cell could silently run with part of its adapter.

### The SR model code

The SR column needs model code that is not in this repository. The job ships
it from a local checkout, at a pinned commit: `SR_REPO` and `SR_REF` in
`local.env`. Only the model package is shipped, and nothing of it is
committed here. The results record only its commit sha.

Without that code, the SR cells are error cells. The other columns still run.

### When the reference is re-computed

The page data holds one reference, with the two versions it was computed
for: `bench_version` and `reference_version` from `adapters.yaml`. The page
shows it on every row of its `bench_version`.

Bump `reference_version` when something changes only the reference numbers:

- the pinned library versions (in `vela/pod_entry.sh`),
- the SR model code commit (`SR_REF`),
- the HF generation code (`hf_generate.py`),
- the base model's instructions (`local/base_instructions.json`).

Version 2 added those instructions.

A `bench_version` bump also needs a new reference. Until it is computed, rows
of the new version show `·` in the reference columns.

## One-time setup

1. Copy the settings template and fill it in:

   ```bash
   cp benchmarks/adapter_eval/vela/local.env.example benchmarks/adapter_eval/vela/local/local.env
   ```

   `local/` is gitignored. It holds the cluster names, storage paths and
   secret names. Never put a secret value in it; the judge key is read in the
   pod from a Kubernetes secret, by name.

2. Log in to the cluster with `oc login`, in the namespace set in
   `local.env`.

3. Add the Helm repository that provides the `mlbatch/pytorchjob-generator`
   chart, and check that `helm template` can find it.

The local side needs only `bash`, `oc`, `helm` and `uv` (or any Python with
PyYAML, set as `LOCAL_PYTHON`).

## Commands

Everything goes through
[submit.sh](../benchmarks/adapter_eval/vela/submit.sh).

| Command | What it does |
|---|---|
| `submit.sh discover` | Lists adapter checkpoints and eval files under the source roots. Read-only. |
| `submit.sh stage [--replace]` | Copies the picks in `local/selection.json` into the bench root. |
| `submit.sh bench <ref> [flags]` | Benchmarks one commit and publishes its row. |
| `submit.sh reference [flags]` | Computes the reference columns and publishes them. |
| `submit.sh rescore [--only a,b]` | Scores saved answers again after a scoring change, and publishes the new scores. Generates nothing. |
| `submit.sh script <ref> <file.py>` | Runs one local Python file on a GPU pod, with the commit installed and the bench root mounted. For one-off checks; the output is only in the log. |
| `submit.sh fetch <job>` | Resumes following a job (after Ctrl-C) and collects its output. |
| `submit.sh stage --release <rev>` | Fetches a Hugging Face release into its folder (see [Hugging Face releases](#hugging-face-releases)). |
| `submit.sh bench --release <rev> [flags]` | Benchmarks the release as published, on its own serving commit, and publishes its row. |
| `submit.sh reference --release <rev> [flags]` | Computes the release's HF + PEFT and Base columns. |

Any command takes `--dry-run`: it renders the job and stops. Any command
also takes `--model <id>`, for a model other than the first (see
[Models](#models)).

`bench` flags:

| Flag | Effect |
|---|---|
| `--limit N` | Only the first N rows of each eval set. For smoke runs; never published. |
| `--only a,b` | Only these intrinsics. The results merge into the commit's existing row. `throughput` is the throughput block: alone, it measures just that. |
| `--no-cache` | Runs even if the commit already has a complete row. |
| `--no-publish` | Keeps the results in `local/results/` without touching the page. |
| `--extra-args "..."` | Passes flags to the in-pod driver, e.g. `--enforce-eager`. |

`reference` takes the same flags, with one difference: `--only` also takes
single cells, as `intrinsic/column`. For example, `--only answerability,guardian_core/sr`
runs the 4 answerability columns and guardian-core's SR column. The cells
merge into the stored reference. When only some cells are missing, the cache
check prints the `--only` value that runs just them.

A job that fails is kept for 24 hours, for its logs. A job that succeeds is
deleted.

### First-time flow

Staging happens once, before the first benchmark:

```bash
benchmarks/adapter_eval/vela/submit.sh discover
```

This writes `local/discovery.json` (all candidates) and
`local/selection.draft.json` (a suggested pick per cell). Review the draft,
then save it as `local/selection.json`.

How the draft picks:

- **Adapters, by their weights.** A checkpoint fits a cell when its rank,
  adapted modules and base model match the cell, e.g. LoRA is r=16 on
  q,k,v,o + MLP over `granite-4.1-3b`. Intermediate `checkpoint-N` folders
  are not searched. Among fitting runs, a run named after the expected
  configuration wins, then the newest.
- **SR, only with shared K/V.** This backend always takes SR's K/V from the
  base stream. A run trained with its own adapter K/V would compose without
  error and give wrong outputs. The checkpoint does not record which kind it
  is; only the run's name does. So an SR run is used only when its path says
  `sharedkv`. The draft picks only such runs, `stage` refuses any other SR
  pick, and a run skips an SR cell whose staged record does not say shared
  K/V.
- **Eval sets, from the runs.** A run's own predictions file holds exactly
  the rows it was scored on, so the draft takes the eval set from there. The
  model's outputs are removed at staging. A note flags an intrinsic whose
  runs were scored on different rows.
- **Intrinsic names.** An intrinsic is read from the run's path. A source
  root written as `<intrinsic>=<dir>` in `local.env` names it for runs whose
  path does not.

The selection looks like this:

```json
{
  "adapters": {"answerability": {"lora": "<checkpoint dir>", "alora": "<checkpoint dir>"}},
  "eval": {"answerability": "<eval file>.jsonl"}
}
```

Then copy the picks into place and run a short smoke test:

```bash
benchmarks/adapter_eval/vela/submit.sh stage
```

```bash
benchmarks/adapter_eval/vela/submit.sh bench main --limit 20
```

Staging copies files rather than linking them, so retraining a source run
cannot change the benchmark underneath it. Each staged folder records its
source path and file checksums.

### Benchmarking a commit

```bash
benchmarks/adapter_eval/vela/submit.sh bench main
```

On success the page data and page are updated locally. Commit them to record
the row:

```bash
git add docs/benchmarks && git commit -s -m "Adapter benchmark: <sha>"
```

### Computing the reference columns

Once per benchmark version, after staging. A short smoke run first:

```bash
benchmarks/adapter_eval/vela/submit.sh reference --limit 20
```

Then the full run:

```bash
benchmarks/adapter_eval/vela/submit.sh reference
```

Commit the result the same way:

```bash
git add docs/benchmarks && git commit -s -m "Adapter benchmark: reference"
```

The largest eval sets (guardian-core and answerability, thousands of rows
each) set the pace.

## Models

The benchmark covers several base models, listed under `models` in
[adapters.yaml](../benchmarks/adapter_eval/adapters.yaml). Each model has its
own:

- **bench root**, with its own staged adapters and eval sets;
- **selection file**, `local/selection.<id>.json`;
- **`bench_version`**, so staging for one model leaves the others' rows
  current;
- **tab on the page**, with its own rows and reference columns.

The first model is the default of every command. For another one, pass
`--model`, and give its settings in `local.env` with a suffix: the model id
with `-` and `.` as `_`. For example, for `granite-4.2-3b`:

```bash
BENCH_ROOT__granite_4_2_3b=$PVC_MOUNT/<path>/adapter-bench/granite-4.2-3b/v1
BASE_MODEL_PATH__granite_4_2_3b=$PVC_MOUNT/<path>/granite-4.2-3b
```

`BENCH_ROOT__<id>` is required: the first model's plain settings are never
used for another model. Without `BASE_MODEL_PATH__<id>` the pod downloads the
base model by name. `ADAPTER_SOURCE_ROOTS`, `EVAL_SOURCE_ROOTS` and `SR_REF`
take the suffix too, and keep the first model's value without one.

Then stage and run as usual, with `--model`:

```bash
benchmarks/adapter_eval/vela/submit.sh stage --model granite-4.2-3b
```

```bash
benchmarks/adapter_eval/vela/submit.sh bench main --model granite-4.2-3b
```

```bash
benchmarks/adapter_eval/vela/submit.sh reference --model granite-4.2-3b
```

### Prompts per model

A model's chat template decides how a prompt carries its documents, and the
benchmark has to build prompts the way the checkpoints were trained:

- **Granite 4.1** renders a `documents=` argument itself, into the system
  turn. Its prompts are passed through as they are.
- **Granite 4.2** has no `documents=`, and opens a reasoning block unless
  `enable_thinking` is false. Its checkpoints were trained with reasoning off
  and the documents placed in the messages:
  - LoRA and aLoRA, trained on 4.1's data, with the documents in the system
    turn, in the very block 4.1's template writes there;
  - SR, by its own trainer, with the documents in a tool message.

  So each technology gets its own form; the base model gets the system form.

| Form | Where the documents go |
|---|---|
| `system` (4.2 LoRA, aLoRA, base) | the system turn: 4.1's preamble, each document as one JSON line within `<documents>` tags, 4.1's closing instruction; after the row's own system message, if any |
| `tool_json_after_question` (4.2 SR) | one tool message after the question, all documents as a JSON list |
| `tool_text_before_question` | one tool message per document, its text, before the question (the first 4.2 aLoRA, retired at v3) |

The `prompt` entry of a model in `adapters.yaml` sets this. Both generation
paths, granite-switch under vLLM and HF + PEFT, build their prompts the same
way ([prompts.py](../benchmarks/adapter_eval/prompts.py)).

### A model's own settings

Everything else is benchmark-wide, unless a model sets it for itself. Each of
these is optional, and without it the model gets the benchmark-wide
behaviour:

| Setting | What it changes |
|---|---|
| `technologies.<tech>.source` | how the page describes the model's adapters of that technology |
| `technologies.<tech>.shape` | the rank, cross rank, adapted modules, K/V mode, expert routing (`share_moe_routing`) and config names its checkpoints must have, over `EXPECTED` in `stage.py` |
| `resources.<mode>` | the pod size of a launcher mode: `gpus`, `cpus`, `memory` |
| `serving.throughput`, `serving.switching` | `false` leaves that serving block out: not measured, not needed for a cache hit, no columns |
| `reference` | `token_budget` and `max_batch` of the HF + PEFT generation, and `serial_loads`: load one cell's model at a time, for a large base |
| `tensor_parallel_size` | how many GPUs each of its vLLM engines spreads its weights over, for a model too big for one: its generation, both engines' throughput and switching. Its bench pods need that many (`resources.bench.gpus`). The row's details show it next to the GPU. The HF + PEFT columns still run a cell per GPU. |

For example, for a mixture-of-experts model whose adapters can target
attention only, and whose SR adapters must route experts with the base
stream:

```yaml
    technologies:
      lora: {source: "LoRA (q,k,v,o, r=32)", shape: {rank: 32, modules: qkvo, names: [lora_qkvo_r32]}}
      sr:
        source: "Shadow Residual (q,o, r=32, cross r=32, shared KV and routing)"
        shape: {rank: 32, cross_rank: 32, modules: qo, shared_kv: true,
                share_moe_routing: true, names: [sr_qo_c32_sharedkv]}
    resources: {bench: {memory: 384Gi}}
    serving: {throughput: false, switching: false}
```

A declared shape is checked when a pick is staged: a checkpoint of another
shape is refused, including one whose `all-linear` targets also wrapped a
mixture-of-experts router. With `share_moe_routing: true`, an SR checkpoint
must say so in its adapter config, when staged and on every run:
granite-switch always takes an SR adapter's expert routing from the base
stream, so one trained to route on its own would run with no error and give
wrong answers.

### Private models

A model that is not public yet is defined in `local/models.private.yaml`,
with the same fields as in `adapters.yaml`:

```yaml
models:
  - id: <id>
    name: <hub id>
    label: <tab label>
    bench_version: 1
```

Every command then knows it (`--model <id>`), and ships the file with its
jobs. It is never published: `submit.sh` does not merge its results, which
stay in `local/results/`, and the page and its data leave it out.

### One model's adapters never run on another

A bench root records the model it was staged for (`model.json`), and every
run refuses a root staged for another model. Without that check, one model's
adapters would load onto another base model with no error. A root staged
before there were several models counts as the first model's.

## Hugging Face releases

The main table measures the adapters this benchmark stages. Composed
granite-switch checkpoints are also published on Hugging Face, with the
adapter libraries they were composed from. Each model's tab has a second
table for them, **Hugging Face releases**: a row per release, named by its
Hub revision rather than by a git commit. A model with no release says so.

### What runs

| Columns | What they are |
|---|---|
| granite-switch (vLLM) | the release's checkpoint as published, served by the granite-switch commit that loads it, or by a granite-switch version from PyPI; each intrinsic under the release's own name for its adapter |
| HF + PEFT | the library adapters it was composed from, at the revisions its BUILD.md records, under PEFT on the base model at its revision of the time; for a release composed from the same adapter files as an earlier one, that one's (`reference_from`) |
| Base | the base model alone, as in the main table |
| Throughput, switching | the main table's two cells, over all of the release's own adapters: its checkpoint against stock vLLM serving the same library adapters (an aLoRA without its invocation tokens) |

A release has one adapter per intrinsic, of its own technology; the others
show "not in this release". The eval sets and prompts are the model's own,
so the accuracy columns compare with the main table's. The speed columns do
not: the main table's are per technology, and its switching cell is over 64
synthetic adapters.

### A release's entry

Each release is an entry of `releases` in `adapters.yaml`:

```yaml
releases:
  - model: granite-4.1-3b
    repo: ibm-granite/granite-switch-4.1-3b-preview
    revision: 7a3ac02e07868411424ac89440397b475da66fa7
    base_revision: c0650403e44e78ec0262dab1c90914c65b196c4e
    serve_with: 24aa420f71b6703f96e40533d83041ce311c30ae
    sources:
      ibm-granite/granitelib-rag-r1.0: 2f0b2c79c6731068625aca8045c2eb2e8912b353
      ...
    adapters: {answerability: answerability, guardian_core: guardian-core, ...}
```

- **`revision`**, **`base_revision`**, **`serve_with`**, `reference_from`
  and every `sources` revision are full shas: a branch would move under the
  row.
- **`serve_with`** or **`serve_pypi`**, one of the two: what serves it.
  - `serve_with`, a commit: a checkpoint loads only on code that still reads
    its format. The first 4.1 previews were composed for the single-switch
    engine, which `6013c7ff` (#131) removed; its parent, `24aa420f`, is the
    last commit that loads them.
  - `serve_pypi`, a version (`"0.1.1"`): the pod installs that granite-switch
    from PyPI, as its users install it, with the `vllm`, `audio` and `hf`
    extras (a checkpoint with ASR loads its audio model when it starts), and
    no checkout. The release's details box links the version.
- **`reference_from`** (optional): an earlier release of the same model whose
  HF + PEFT and Base columns this one shows. Use it when both were composed
  from the same adapter files on the same base revision, e.g. the 8B preview
  recomposed on 2026-10-15: its libraries' newer revisions changed only model
  cards and one `io.yaml`, which the reference never reads.
  - `submit.sh reference --release` runs nothing for it.
  - Publishing its row checks every adapter's weights checksum against the
    reference run it shows, and refuses a difference: then drop
    `reference_from` and run its own.
- **`sources`**: the libraries its BUILD.md names, at their revisions. Each
  adapter's folder there is `<name>/<base>/<technology>`, from the
  checkpoint's `adapter_index.json`.
- **`adapters`**: the release's name for each of the benchmark's intrinsics
  (`guardian-core` for `guardian_core`).
- A later Hub commit that changes only the model card needs nothing. A new
  release is a new entry.
- **A release of a base model with no staged adapters** (the 4.1 8B
  preview): the base model gets an entry under `models` (`granite-4.1-8b`)
  and a bench root of its own (`BENCH_ROOT__<id>`) holding only the eval
  sets. Stage them with a `local/selection.<id>.json` that has only `eval`,
  the same sources as the other models', then stage the release. Its tab
  shows an empty table of commits above its releases.
- **A release too big for one GPU** (the 4.1 30B preview, 60 GB of bf16
  weights): its model sets `tensor_parallel_size: 2` and
  `resources: {bench: {gpus: 2, ...}}`, so each engine of its bench runs
  spreads over two GPUs; its `reference` settings give the HF + PEFT cells a
  smaller `token_budget` and `serial_loads`.

### Commands

```bash
benchmarks/adapter_eval/vela/submit.sh stage --release 7a3ac02e
benchmarks/adapter_eval/vela/submit.sh bench --release 7a3ac02e --limit 20
benchmarks/adapter_eval/vela/submit.sh reference --release 7a3ac02e --limit 20
```

For a model with no bench root yet, `submit.sh stage --model <id>` first,
for its eval sets.

`stage` fetches the release into `RELEASE_ROOT/<model>/<repo name>/<revision>`
(set in `local.env`):

```
checkpoint/                   the composed checkpoint, at the release's revision
native/<adapter>/             each adapter from its library, for stock vLLM
adapters/<intrinsic>/<tech>/  the benchmark's intrinsics among them, for HF + PEFT
eval/<intrinsic>/             the model's eval sets, copied from its bench root
model.json, release.json      the model; what was fetched, from which Hub commits
```

It refuses a library revision the release's BUILD.md does not name, and an
adapter whose files are not of the technology the release says. `bench` and
`reference` then run in that folder; `--limit N` makes a smoke run, never
published. A release with `reference_from` needs no `reference` run.

### Cache

A release's row is a cache hit when it has the current `release_version`,
every cell (one scored with an older `score_version` runs again: releases are
not re-scored), and its throughput and switching measured with the current
settings; its reference columns, when they have the current
`reference_version` and `release_version` (with `reference_from`, those of the
release it names). Bump `release_version` when a
change alters how releases are benchmarked: every release runs again.

## Cache rules

A commit is a **cache hit** when its row has:

- the current `bench_version` from `adapters.yaml`,
- every cell present, with no error cells. Skipped cells count as done, and
- its throughput, measured with the current settings.

On a hit, `bench` prints the row and submits nothing. Some examples:

| Stored row | Result |
|---|---|
| same version, all cells scored or skipped | hit |
| same version, one cell is an error | miss, re-run |
| older version | miss; the old row stays on the page, in grey |
| no row | miss |

Publishing refuses a `--limit` run and a run whose version differs from
`adapters.yaml`. A failed run publishes nothing, so the next run retries.

The reference columns follow the same rules, with both versions. They are a
hit when their `bench_version` and `reference_version` match `adapters.yaml`
and every cell is present, with no error cells.

## Decode throughput

The throughput block is measured by the switch benchmark's own driver,
[`benchmarks/bench_switch_repro.py`](../benchmarks/bench_switch_repro.py),
copied unchanged from the staging repository's `feature/switch-benchmark`
branch but for one internal link. Copy it again to update it; never edit it
here. [throughput.py](../benchmarks/adapter_eval/throughput.py) only builds
its command line for this benchmark's adapters and reads its record back.

It measures the cell of that benchmark's batch-decode figure (throughput for
generating 128 tokens with one adapter active, no switching): granite-switch
against stock vLLM serving the same adapters, at one batch size. Per
technology, two of its arms, on the model's staged adapters (N of them):

| | granite-switch (vLLM) | PEFT (vLLM): stock vLLM |
|---|---|---|
| LoRA | `gs-lora-vllm`: a checkpoint of the LoRA adapters | `native-lora`: the same LoRA checkpoints |
| aLoRA | `gs-lora-vllm`: a checkpoint of the aLoRA adapters | `native-lora`: the aLoRA checkpoints without their invocation tokens |
| SR | `gs-sr-vllm`: a checkpoint of the SR adapters | none |

Stock vLLM has no SR implementation, so SR has no PEFT (vLLM) number and no
speedup. That benchmark's `native-sr` arm serves SR's weights as a plain LoRA,
without SR's second stream; it is a control there, not SR, and is not run
here.

The checkpoints are composed with the commit's composer; stock vLLM loads all
N adapters (`max_loras` = N).

**The cell**, its prompt-1 decode cell:

- a batch of **32** one-token prompts: the adapter's control token under
  granite-switch, one filler token with its `LoRARequest` under stock vLLM;
  each request on one of the N adapters, by its seeded draw (the same for
  both engines);
- exactly **128 tokens generated** for each (`min_tokens` = `max_tokens`);
- **2 warm-up runs, then 5 timed runs**; tok/s = batch × tokens ÷ the median
  run's seconds, its formula;
- its pinned engine: no prefix caching, a 4,096-token context, 8,192 batched
  tokens, 256 sequences, CUDA graphs up to batch 1,024, 90% of GPU memory,
  vLLM's engine in its own process;
- its gates first (prefix caching and the CUDA-graph size resolved as asked,
  the adapter count, `max_loras` covering N, the first adapter live); a
  failed gate shows as an error cell.

**How it is run**, as its sweep runs each block (`run_switch_repro_sweep.sh`,
`run_block`): one process per engine, the two one after the other on the
commit's GPU, the compile caches (vLLM, Triton, FlashInfer, TorchInductor)
cleared before each; which engine goes first alternates by technology.

**Where it differs from that benchmark**, each on purpose:

- one batch size, 32, where its figure sweeps 1 to 64;
- this model's trained adapters, N per technology, where it uses 12
  synthetic rank-32 adapters per checkpoint; throughput falls as N grows, so
  the numbers compare across commits, not with its figure;
- an aLoRA row, which it leaves out because granite-switch serves aLoRA and
  LoRA alike: ours have their own rank. Stock vLLM gets the aLoRA weights
  without the invocation tokens, which would keep them off after a
  one-token prompt; an active aLoRA decodes as a LoRA;
- only adapter-active cells (100%), not its idle (0%) ones, and no drift
  anchors, since a commit's engines run within minutes, not a sweep's hours.

The settings are `throughput` in `adapters.yaml`. A row counts as done only
with every technology's throughput measured with the current batch and
generated length, so rows from before throughput existed are cache misses.
`submit.sh bench <ref> --only throughput` measures just the throughput of a
row, without its accuracy run (`--only switching`, the switching block).

Running it adds about 20 minutes to a commit's run: a checkpoint per
technology, and an engine start per engine and technology.

## Agents switching adapters

The second throughput block is one cell of the switch benchmark's
concurrency-by-switching grid (its `grid_concurrency` figure), measured by its
own scripts, copied from the staging repository's `feature/switch-benchmark`
branch, unchanged but for a `--tensor-parallel-size` flag (default 1) on
`bench_switching_grid.py` and `bench_agent_sim.py`, which their shared engine
takes:

| Script | Its part |
|---|---|
| `benchmarks/make_synth_fleet.py` | its synthetic adapters (it needs `tests/shared/base_models.py` and `synthetic_adapters.py`, shipped with it) |
| `benchmarks/verify_composed.py` | its check of a composed checkpoint |
| `benchmarks/gen_switching_grid.py` | the agents' workload |
| `benchmarks/bench_switching_grid.py` | the run of one engine over the cell |
| `benchmarks/bench_agent_sim.py` | the agent loop, engine, gates and timing it uses |

[switching.py](../benchmarks/adapter_eval/switching.py) builds their command
lines and reads their rows back.

**The workload**, its, at one cell: **8 agents** at a time, 16 in all (two
waves, so the p95 is over 16 agents, as in its grid). Each agent has a prompt
of its own, log-spaced from 256 to 16,384 tokens, generates **512 tokens**,
and switches adapter **every 32 tokens**, picking adapters it has not used
yet first, with tool calls (and their latencies) at its rate between runs.
The figure runs 1,024 tokens over concurrency 1 to 64; the settings are
`switching` in `adapters.yaml`.

**The adapters**: its **64 synthetic adapters** of rank 32 per technology
(`lora`, `alora`, `sr`), as in its figure. Their timing does not depend on
their weights, and the cost being measured grows with their number: a first
visit to an adapter re-prefills the agent's whole context. They are built
once per model and builder version, and kept on the storage volume. The SR
ones get the SR anchor the accuracy run gives the real SR adapters, through
the builder's `--last-context-token`: the builder's own renders the chat
template without the SR prompt options, which on Granite 4.2 ends on a newline
it rejects. The two agree on 4.1 (`<|end_of_role|>`); on 4.2 it is
`</think>`.

**The engines**, per technology:

| | granite-switch (vLLM) | PEFT (vLLM): stock vLLM |
|---|---|---|
| LoRA | a checkpoint of the 64 LoRA adapters, composed by the commit and checked by `verify_composed.py`; switching like stock vLLM, by control token (below) | its `native-lora` arm: a new request at every switch, with the adapter's LoRARequest |
| aLoRA | the same, with the 64 aLoRA adapters | the same, the aLoRA adapters without their invocation tokens |
| SR | its `shadow-residual` arm on a checkpoint of the 64 SR adapters: one adapter for the whole task, since SR switches without a re-prefill | none |

Both re-prefill at a switch: the agent's prompt and everything it generated,
under the new adapter. With prefix caching on, stock vLLM's cache is keyed on
the LoRA and a revisit reuses what that adapter computed before; under
granite-switch the control token is the first token, so the cache does the
same.

**The granite-switch LoRA and aLoRA arm** is the one thing not in that
benchmark, which has a granite-switch arm for SR only.
[gs_switch.py](../benchmarks/adapter_eval/gs_switch.py) runs its cell loop,
agent loop (as its native-lora arm), engine, gates and timing unchanged,
through an engine that names each request's adapter by its control token
rather than a LoRARequest.

**The cell** is its figure's: the p95 time for an agent to finish, by nearest
rank. The speedup is stock vLLM's p95 over granite-switch's, in whole percent.

**A warm-up run first.** Each engine runs the cell twice and only the second
run is timed (`warmup_runs`; the first run's p95 is kept beside it). Their
driver runs a job's cells in order in one engine and resets the prefix cache
before each, so listing the cell twice is all it takes. Their figure's sweep
runs 49 cells per engine, so all but its first cell find the adapters loaded
and the kernels compiled; a single cell measured cold would charge those
one-off costs to the cell. They are not small next to the gap being measured:
cold, stock vLLM reads each adapter from the storage volume on its first use.
Measured cold on b7e54212, its first wave of agents finished 14 to 22 seconds
after granite-switch's, while its second wave, with every adapter loaded,
finished level with granite-switch's or ahead of it.

Running it adds roughly 35 to 50 minutes to a commit's run: three
64-adapter checkpoints to compose and check, and five engines.
`submit.sh bench <ref> --only switching` measures just this block.

## Re-scoring saved answers

A commit's run generates only its own columns, the granite-switch (vLLM)
ones. The HF + PEFT and Base columns are the reference: computed once per
model, and again only when their version changes. Every run also keeps its
answers on the storage volume.

So each kind of change re-runs only what it affects. Three versions in
`adapters.yaml` say what changed:

| What changed | Bump | What runs again |
|---|---|---|
| Generation for every column: an eval set, the prompts, generation settings, a checkpoint | the model's `bench_version` | every commit's row, and the reference |
| Only the reference columns (see [above](#when-the-reference-is-re-computed)) | `reference_version` | the reference only |
| Only how one intrinsic is scored | that intrinsic's `score_version` | nothing is generated: the saved answers are scored again |

For a scoring change:

1. Change the scorer and bump the intrinsic's `score_version`.
2. Run `submit.sh rescore`, once per model (`--model <id>`). It lists the
   scored cells whose `score_version` is older, then runs one small pod.
   The pod reads each cell's saved answers and scores them with the new
   scorer. It takes minutes, not hours.
3. The new scores replace the old ones on the page. The row's run details
   record which cells were scored again.

`--only a,b` scores those intrinsics again whatever their version, e.g. after
a scorer bug fix that kept the version.

Only rows of the current `bench_version` and the current reference are
scored again; older rows stay as they were. Some cells are left as they are:

- error cells, which have no answers. They need generating (`bench --only`).
- a cell whose saved answers are not found. The log names it, and the next
  `rescore` tries it again.
- a cell whose row ran again in the meantime. Its new run already used the
  new scorer.

## Adding an intrinsic or an adapter

1. Add an entry to
   [adapters.yaml](../benchmarks/adapter_eval/adapters.yaml): id, name,
   scorer, headline metric and `max_new_tokens`.
2. If no scorer fits, add one under
   [scorers/](../benchmarks/adapter_eval/scorers/__init__.py) and register it.
3. Add the new picks to `local/selection.json` (`selection.<model>.json` for
   another model) and run `submit.sh stage`. Cells already staged from the
   same picks are kept, so only the new ones are copied; a pick that differs
   from what is staged fails unless `--replace` is given, which overwrites it.
4. Bump `bench_version` if the change alters existing numbers. That covers a
   new eval set, new generation settings, or a replaced checkpoint. A
   scoring change alone bumps the intrinsic's `score_version` instead (see
   [Re-scoring saved answers](#re-scoring-saved-answers)). A pure addition
   does not need a bump: run the new intrinsic with `--only`.

To remove an intrinsic, delete its entry from `adapters.yaml`, with a comment
saying why. The page stops showing it, and no version changes: its cells stay
in the page data, and `stage` skips its picks, for when it comes back.
Hallucination detection and query rewrite are left out this way.

## Local folder layout

All under `benchmarks/adapter_eval/vela/`, all gitignored:

```
local/
  local.env              cluster, storage and secret names; the SR code checkout
  selection.json         the reviewed picks for `stage`
  selection.<id>.json    the same, for another model (also discovery.<id>.json, ...)
  judge_prompt.txt       query-rewrite judge prompt (shipped at stage time)
  base_instructions.json the base model's answer-format instructions (shipped
                         with each reference job)
  discovery.json         output of `discover`
  selection.draft.json   suggested picks from `discover`
  jobs/<job>.env         what `fetch` needs to resume a job
  logs/<job>.log         full pod logs
  results/<job>.json     results blocks from bench, reference and rescore runs
  rescore/targets.json   the cells the last `rescore` scored again
  stage/<job>.json       stage reports, a release's included
.rendered/               rendered Helm values and job specs
```

## Public and private

This repository is public. So:

- **Public:** the harness, the page, and the page data. The page shows only
  scores, commit metadata and generic skip reasons.
- **Private (in `local/`):** namespace, image, volume and paths, secret
  names, the judge endpoint, the judge prompt and the base model's
  instructions, and models that are not public yet
  (`local/models.private.yaml`).
- **Private, and not in this repository at all:** the SR model code of the
  reference column. The job ships it from a local checkout. Only its commit
  sha is published.

## Tests

The harness has CPU unit tests: cache rules, merging, page rendering, every
scorer, checkpoint conversions, staging, job rendering, and the reference
run's driver.

```bash
pytest tests/unit/test_adapter_benchmark.py -v -s --tb=short -x
```

Composing and generating, with vLLM or HF + PEFT, need a GPU and are only
exercised by a real run.

## Known gaps

- **The vLLM and HF + PEFT columns will differ slightly.** They use the same
  checkpoints, prompts and greedy decoding. But different kernels and
  batching change bfloat16 rounding, which can flip near-ties. A large gap is
  worth investigating.
- **All adapters come from the internal trainer.** Their MLP weight names,
  and SR's activation, are converted as described above. Moving to
  shadow-residual trainer checkpoints for SR later replaces checkpoints, so
  it needs a `bench_version` bump.
- **Query rewrite is left out.** An LLM judge grades it, and it leaves a
  different 12–17% of rows ungraded each run, so the same answers score 1–3
  points apart. Its scorer and judge settings stay, for a steadier judge.
  Only jobs that score with the judge get its key, so for now none do.
- **Throughput is measured at one batch size.** The switch benchmark sweeps
  batch sizes; this page shows batch 32, its standard point, to keep one
  number per cell.

## Running it from GitHub

The benchmark can also run on the self-hosted GPU runner, started from any
terminal with `gh`. The workflow
([adapter-benchmark.yaml](../.github/workflows/adapter-benchmark.yaml)) lives
on this branch, so it has no Run button in the Actions tab and no PR-comment
command; GitHub offers those only for workflows on `main`.

```bash
gh workflow run adapter-benchmark.yaml --repo generative-computing/granite-switch --ref feature/adapter-benchmark -f sha=6013c7ff82a73020f1d1fb00dcdc323398fffb38 -f model=granite-4.2-3b
```

| Input | Effect |
|---|---|
| `sha` | the full commit sha to benchmark (required) |
| `model` | a model id from `adapters.yaml`; empty for the first |
| `pr_number` | also report on this pull request |
| `no_cache` | `true` to run although the commit already has its row |

The workflow:

1. checks the role (Maintain or Admin) with the runner image's script;
2. checks out this branch and runs `submit.sh bench <sha> --no-publish`,
   which stops early on a cache hit;
3. merges the row into the page data and pushes it to this branch, so the
   page updates;
4. with a `pr_number`, posts the result on that pull request: a comment per
   model (`publish.py summary`), edited by later runs. It also sets a commit
   status `adapter-benchmark/<model>`.

GitHub dispatches a workflow from a branch other than `main` only once it has
run there. A push that changes the workflow file runs a small job that does
just that.

### What it needs

- **The private settings:** the repository secret `ADAPTER_BENCH_LOCAL_ENV`,
  holding a `local.env` like the one in `vela/local/`, for every model it
  should run. A repository admin sets it, from a local copy:

  ```bash
  gh secret set ADAPTER_BENCH_LOCAL_ENV --repo generative-computing/granite-switch < benchmarks/adapter_eval/vela/local/local.env
  ```

  Without the secret, the workflow reads `/opt/gsw/adapter-bench/` on the
  runner. Like every repository secret, it is readable by a workflow that
  someone with write access writes.
- **The GPU runner online**, with `oc` logged in with rights to create and
  follow jobs in the benchmark's namespace (the GPU tests' jobs run in the
  same one), `helm` and `git`. The workflow adds the job chart's `helm`
  repository and installs `uv` itself.

### What stays manual

- **The reference columns** (`submit.sh reference`): once per model and
  version, and they need the private SR model code.
- **Staging adapters** (`submit.sh stage`).

### Safety

- The log is public. It prints only `submit.sh`'s status lines: the pod log,
  which names internal storage paths, stays in a file on the runner.
- The commit under test runs only in the cluster pod; the runner runs this
  branch's code.
- The workflow comes from this branch, so whoever can push to it can change
  what runs on the GPU runner. Protect the branch.
