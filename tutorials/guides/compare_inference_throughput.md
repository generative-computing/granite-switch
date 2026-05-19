# ALORA vs LoRA Race

Benchmark two Granite Switch models — one using **ALORA** (which defers adapter
activation to save prefill time) and one using standard **LoRA** — on a 6-step RAG
pipeline under concurrent load.

Each conversation runs 5 turns through:
guardian &rarr; query rewrite &rarr; ChromaDB retrieval &rarr; answerability &rarr; clarification &rarr; generation

## Pre-recorded race

[**Watch the animated replay**](https://generative-computing.github.io/granite-switch/race_live.html)
of a 32-conversation race (no setup required).

## Reproduce in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/alora_vs_lora_race.ipynb)

The notebook runs both servers sequentially on a single A100 GPU and produces
`race_live.html` (animated replay) and `race_report.html` (static summary).

## Run locally with Rich live display

### Prerequisites

- Two GPUs (one per server) for simultaneous mode, or one GPU for sequential mode
- Install dependencies:
  ```bash
  pip install -e ".[vllm]"
  pip install mellea chromadb rich tqdm transformers httpx
  ```
- Build the ChromaDB index (once):
  ```bash
  python build_govt_chroma.py
  ```
- Compose the LoRA-only model. See [`../notebooks/compose_granite_switch.ipynb`](../notebooks/compose_granite_switch.ipynb) — pass the three libraries (`granitelib-rag-r1.0`, `granitelib-core-r1.0`, `granitelib-guardian-r1.0`) with `--technology-filter lora` to force every adapter to its standard LoRA variant, and set `--output ./granite-switch-lora-only`.

### Simultaneous race (two H100 GPUs)

Start both servers, then run the benchmark with the Rich live terminal display:

```bash
# Start servers
bash launch_servers_race.sh ./granite-switch-lora-only

# Run the race — H100 suggested parameters
python bench_pipeline_race.py -n 32 -c 24 -k 15
```

### Sequential mode (one GPU)

Run one server at a time — useful when only a single GPU is available:

```bash
# ALORA leg
vllm serve ibm-granite/granite-switch-4.1-3b-preview --port 8111
python bench_pipeline_race.py --mode sequential --server "ALORA (8111)" -n 32 -c 24 -k 15

# Stop the ALORA server, then start the LoRA leg
vllm serve ./granite-switch-lora-only --port 8112
python bench_pipeline_race.py --mode sequential --server "LORA (8112)" \
  --lora-model ./granite-switch-lora-only -n 32 -c 24 -k 15
```

Results are merged across runs — `race_live.html` replays both legs as if they
raced simultaneously.

### CLI options

| Flag | Description |
|------|-------------|
| `-n, --runs` | Number of conversations (default: 16) |
| `-c, --concurrency` | Max concurrent requests per server (default: 8) |
| `-k, --top-k` | Documents to retrieve per query (default: 10) |
| `--no-live` | Disable Rich live display (for notebooks) |
| `--alora-model` | Override ALORA model path |
| `--lora-model` | Override LoRA model path |

**Suggested parameters by GPU:**

| GPU | `-n` | `-c` | `-k` |
|-----|------|------|------|
| A100 (Colab) | 16 | 8 | 10 |
| H100 | 32 | 24 | 15 |

## Next Steps

- **[Hello Adapter](../notebooks/hello_adapter.ipynb)** - minimal embedded-adapter invocation via the HuggingFace backend
- **[Using Mellea with Granite Switch](mellea_with_granite_switch.md)** - deeper Mellea integration details
- **[Bring Your Own Adapter](bring_your_own_adapter.md)** - train a custom adapter and compose it in
