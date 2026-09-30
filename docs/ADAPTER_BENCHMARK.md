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

Each intrinsic has 7 columns on the page:

| Group | Columns | What runs | Per commit |
|---|---|---|---|
| granite-switch (vLLM) | LoRA, aLoRA, SR | the adapters composed with the commit, under its vLLM backend | yes |
| HF + PEFT | LoRA, aLoRA, SR | the same checkpoints without granite-switch, with Hugging Face transformers and PEFT | no |
| Base | one | the base model with no adapter | no |

The last two groups are the **reference columns**. They show what the
checkpoints score outside granite-switch, and what the base model scores
alone. For example, if a commit drops aLoRA on answerability from 88 to 70
while HF + PEFT aLoRA stays at 88, the commit broke something. The reference
columns do not depend on the commit, so they are computed once and repeated
on every row (see [Reference columns](#reference-columns)).

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
{"accuracy": 0.867, "n": 450}
{"skipped": "adapter not staged"}
{"error": "generation failed"}
```

Metrics are stored as fractions and shown as percentages (`86.7`). On the
page:

| Shown | Meaning |
|---|---|
| `86.7` | scored; hover for every metric. The best of an intrinsic's 7 columns is bold. |
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
- **Base:** the base model, no adapter.

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
- the HF generation code (`hf_generate.py`).

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
| `submit.sh script <ref> <file.py>` | Runs one local Python file on a GPU pod, with the commit installed and the bench root mounted. For one-off checks; the output is only in the log. |
| `submit.sh fetch <job>` | Resumes following a job (after Ctrl-C) and collects its output. |

Any command takes `--dry-run`: it renders the job and stops. Any command
also takes `--model <id>`, for a model other than the first (see
[Models](#models)).

`bench` flags:

| Flag | Effect |
|---|---|
| `--limit N` | Only the first N rows of each eval set. For smoke runs; never published. |
| `--only a,b` | Only these intrinsics. The results merge into the commit's existing row. |
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
base model by name.

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

A bench root records the model it was staged for (`model.json`), and every
run refuses a root staged for another model. Without that check, one model's
adapters would load onto another base model with no error. A root staged
before there were several models counts as the first model's.

## Cache rules

A commit is a **cache hit** when its row has:

- the current `bench_version` from `adapters.yaml`, and
- every cell present, with no error cells. Skipped cells count as done.

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

## Adding an intrinsic or an adapter

1. Add an entry to
   [adapters.yaml](../benchmarks/adapter_eval/adapters.yaml): id, name,
   scorer, headline metric and `max_new_tokens`.
2. If no scorer fits, add one under
   [scorers/](../benchmarks/adapter_eval/scorers/__init__.py) and register it.
3. Add the new picks to `local/selection.json` and run `submit.sh stage`.
   Existing cells are kept; `--replace` overwrites them.
4. Bump `bench_version` if the change alters existing numbers. That covers a
   new eval set, a changed scorer, new generation settings, or a replaced
   checkpoint. A pure addition does not need a bump: run the new intrinsic
   with `--only`.

## Local folder layout

All under `benchmarks/adapter_eval/vela/`, all gitignored:

```
local/
  local.env              cluster, storage and secret names; the SR code checkout
  selection.json         the reviewed picks for `stage`
  selection.<id>.json    the same, for another model (also discovery.<id>.json, ...)
  judge_prompt.txt       query-rewrite judge prompt (shipped at stage time)
  discovery.json         output of `discover`
  selection.draft.json   suggested picks from `discover`
  jobs/<job>.env         what `fetch` needs to resume a job
  logs/<job>.log         full pod logs
  results/<job>.json     results blocks from bench and reference runs
  stage/<job>.json       stage reports
.rendered/               rendered Helm values and job specs
```

## Public and private

This repository is public. So:

- **Public:** the harness, the page, and the page data. The page shows only
  scores, commit metadata and generic skip reasons.
- **Private (in `local/`):** namespace, image, volume and paths, secret
  names, the judge endpoint and the judge prompt.
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
- **Query rewrite needs its judge.** Without a judge endpoint and key, that
  intrinsic is skipped.
- **No throughput yet.** Cells are dictionaries, so more metrics can be added
  without breaking old rows.

## Later: PR-comment trigger

A `/benchmark` PR comment, like `/gpu-test`, is planned. It needs the workflow
files on `main` and support in the GPU runner. See [CICD.md](CICD.md) for the
existing GPU-test flow.
