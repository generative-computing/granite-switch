# Agent race: granite-switch SR against stock vLLM LoRA

[agent_race.html](agent_race.html) replays a recorded race between two ways of
serving agents that switch adapters: granite-switch's Shadow Residual (SR) and
stock vLLM serving the same adapters as LoRAs. It looks and works like the
[ALORA-vs-LoRA race](race_live.html): a column per lane, a bar per agent, a
metrics panel per lane, and play, speed and scrub controls. The page opens
straight from disk: the race is embedded in it.

## What races

**The workload** is the adapter benchmark's "Decode 512 tokens, switch size 32,
concurrency 8" cell (see [Agents switching adapters](ADAPTER_BENCHMARK.md#agents-switching-adapters)),
from the switch benchmark's own generator and agent loop, copied unchanged into
`benchmarks/`:

- **16 agents per lane, at most 8 at a time.** The next agent starts as soon as
  any running one finishes. (The benchmark releases its agents in two fixed
  waves of 8 instead.)
- **Each agent** has a prompt of its own (256 to 16,384 tokens), generates 512
  tokens, and switches adapter every 32 tokens over 64 synthetic adapters of rank
  32, with its tool calls and round trips as real sleeps.

**The lanes:**

| Lane | Serving | At an adapter switch |
|---|---|---|
| `gs-sr-vllm` | granite-switch SR: the 64 SR adapters composed by the commit, the switch benchmark's shadow-residual arm | nothing: one generation runs until a tool call. As in the switch benchmark, each agent keeps one adapter for its task, since SR's K/V come from the base model and a switch is taken as free. |
| `lora-vllm` | stock vLLM serving the 64 LoRA adapters (its native-lora arm) | ends the request and starts a new one, prefilling the agent's whole context again; vLLM's prefix cache is keyed on the LoRA, so only an adapter served before hits it |

**How it is measured:** both lanes run on one GPU, one after the other, each in
an engine of its own with fresh compile caches and after one untimed warm-up
run of the whole fleet, with the prefix cache reset before every run. The page
replays them side by side, each from its own start.

## What the page shows

- **A bar per agent**, filled by the tokens it has generated, colored by what it
  is doing: waiting for a slot, prefilling (with how many prompt tokens it
  submitted), decoding (with its adapter), on a round trip, on a tool call, or
  done (with its time).
- **A footer per lane**: agents done, running and waiting, then the lane's time.
- **A metrics panel per lane**, sampled every half second: agents running,
  prefilling, decoding and waiting on a tool or round trip; tokens per second;
  the average time to first token of its last 8 requests; the prompt tokens the
  engine re-prefilled so far; and its prefix-cache hit rate, from vLLM's own
  count of cached prompt tokens.
- **The result**, once both lanes are done: each lane's time and the ratio.

## Running it

1. Run the race on a pod, through the benchmark's script mode, which installs
   the commit and ships the harness:

   ```bash
   benchmarks/adapter_eval/vela/submit.sh script b7e54212f5d73b9d971c00ba2cb58b6612178f33 benchmarks/adapter_eval/race/run_race.py
   ```

   It takes the workload's settings from `switching` in `adapters.yaml` and the
   synthetic adapters from the benchmark's cache (built if missing), composes and
   checks the SR checkpoint, and runs both lanes. Add
   `--extra-args "--model granite-4.2-3b"` for another model.

2. Put the race into the page, from the pod's saved log:

   ```bash
   python -m benchmarks.adapter_eval.race.build_page benchmarks/adapter_eval/vela/local/logs/<job>.log
   ```

   It refuses a race that carries an absolute path anywhere.

3. Open `docs/agent_race.html`.

## What is recorded

`benchmarks/adapter_eval/race/lane.py` runs a lane: the agent loop
(`run_agent`), unchanged, through a recording engine that notes each request's
start, first token, tokens and end, and the engine's count of cached prompt
tokens. The gaps between an agent's requests are its round trips and tool
calls, named from its schedule.

The page holds the race's events (seconds from each lane's start), each lane's
agents (their prompt lengths, switches, tool calls and times), and the run's
public details: model, commit, GPU, library versions, settings. No path on the
run's storage, no host name.

| File | Its part |
|---|---|
| `benchmarks/adapter_eval/race/lane.py` | one lane: slots, recording, metrics |
| `benchmarks/adapter_eval/race/run_race.py` | the pod's run: workload, adapters, SR checkpoint, both lanes |
| `benchmarks/adapter_eval/race/build_page.py` | the race into the page |
| `docs/agent_race.html` | the page |
| `tests/unit/test_agent_race.py` | the race end to end on CPU, with a fake engine |
