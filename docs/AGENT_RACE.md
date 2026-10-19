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

- **Agents run in slots:** the next agent starts as soon as any running one
  finishes. (The benchmark's switching block runs 16, released in two fixed
  waves of 8.)
- **Each agent** has a prompt of its own (256 to 16,384 tokens), generates its
  tokens, and switches adapter every 32 over 64 synthetic adapters of rank 32,
  with its tool calls and round trips as real sleeps.
- **Both lanes run the same agents:** they read one workload file, so an agent's
  prompt, adapter schedule and length are the same in both.

The page holds five races:

| Agents | Per lane | At a time | Tokens per agent | Options (`--extra-args`) |
|---|---|---|---|---|
| the switching block's | 16 | 8 | 512 | `--agents 16` |
| the switching block's | 64 | 8 | 512 | (the defaults) |
| the switching block's | 64 | 16 | 1,024 | `--concurrency 16 --decode 1024` |
| the switching block's | 64 | 16 | 128, 256, 512 or 1,024, 16 agents each | `--concurrency 16 --decode-mix 128,256,512,1024` |
| the paper's ReAct agents | 64 | 16 | 128, 256, 512 or 1,024, 16 agents each | `--agents-kind react --concurrency 16 --decode-mix 128,256,512,1024` |

A mixed workload runs the generator once per length, for an equal share of the
agents with a seed of its own, then shuffles the agents together with a fixed
seed, so the lengths interleave and a rerun gives the same workload.

**The ReAct agents** are the switch benchmark's agent study's
(`benchmarks/gen_agent_configs.py`, copied unchanged; built by
`benchmarks/adapter_eval/race/react.py`). Each walks a ReAct cycle of typed
steps, at the generator's lengths and switch rates:

| Step | Median tokens | Switches adapter | Inside the step |
|---|---|---|---|
| think | 120 | 35% of the time | 1 + Poisson(2) segments, a switch each |
| prepare_call | 10 | 95% | one segment |
| act | 0: a tool call | | the tool's latency, slept for real |
| observe | 35 | 80% | 1 + Poisson(0.5) segments |
| reflect | 90 | 70% | 1 + Poisson(1.5) segments |

over its three cycles (classic5, toolheavy10, shortadapters), choosing adapters
novelty-first over the 64. Three settings are the race's, so its agents are
comparable with the other races': 128 to 1,024 tokens (the paper's agents
generate 10,000), the switching workload's prompts (256 to 16,384 tokens), and a
round trip within one datacenter (the paper draws a mix of distances). The
generator closes every tenth of an agent's tokens on a step boundary, for the
paper's stopwatch; here an agent is one chunk, closed by its last step, and
every other step keeps its drawn length. They switch about every 35 tokens.

**The lanes:**

| Lane | Serving | At an adapter switch |
|---|---|---|
| `gs-sr-vllm` | granite-switch SR: the 64 SR adapters composed by the commit, the switch benchmark's shadow-residual arm | nothing: its K/V come from the base model, so one generation runs until a tool call. The page shows its adapter switching on the same schedule as stock vLLM's, as many times; the switch itself is not implemented yet, so the run kept one adapter per agent, and on the GPU SR costs the same with or without one. |
| `lora-vllm` | stock vLLM serving the 64 LoRA adapters (its native-lora arm) | ends the request and starts a new one, prefilling the agent's whole context again; vLLM's prefix cache is keyed on the LoRA, so only an adapter served before hits it |

**Adapter switching (rtt):** stock vLLM sets a LoRA per request, so it can't
change adapter inside a generation. At each switch, its tokens go back to the
agent's software, which sends a new request on the next adapter. That round trip
is what the page calls an adapter switching (rtt): 2 to 10 ms (the same
datacenter), drawn uniformly per switch and slept for real, so the GPU serves
the other agents meanwhile. A tool call (act) is the other wait: both lanes stop
for it, sleeping the same tools' latencies. SR has no adapter switching (rtt):
its adapter changes in place.

**How it is measured:** both lanes run on one GPU, one after the other, each in
an engine of its own with fresh compile caches and after one untimed warm-up
run of the whole fleet, with the prefix cache reset before every run. The page
replays them side by side, each from its own start.

## What the page shows

- **The experiment**, in a few lines, and a diagram of the race's agents: the
  switching block's (its prompt, 32-token generations switching adapter at each, a
  tool call between) or, for the ReAct race, a ReAct agent's steps at their own
  lengths; then what each lane does at a switch, and what an adapter switching
  (rtt) takes. Hovering a step or a metric's label explains it.
- **A race per configuration**, chosen above the controls, each a recorded run.
  The link keeps the choice (e.g. `#64-16-mix`).
- **A bar per agent**, filled by the tokens it has generated, colored by what it
  is doing: waiting for a slot; prefilling, with the prompt tokens it computes and
  those vLLM's prefix cache already held (e.g. `prefill 3 tok · 2.6k cached`);
  generating, with its step and adapter (e.g. `think (generation) ad26`, or
  `generation ad26` for the switching block's agents); on an adapter switching
  (rtt); on a tool call (act); or done, with its time.
- **A footer per lane**: agents done, running and waiting, then the lane's time.
- **A metrics panel per lane**, sampled every half second: agents running,
  prefilling, decoding and waiting on a tool call or an adapter switching (rtt);
  tokens per second; the adapter switches so far (the same in both lanes at the
  end: for the switching block's agents one per 32 tokens they generate, so 16
  for 512 tokens)
  and the requests behind them (stock vLLM issues one per switch, SR one per run
  between tool calls);
  the average time to first token of its last 8 requests; and, from vLLM's own
  count of each request's cached prompt tokens:
  - **Re-prefilled**: prompt tokens computed again after an agent's first
    request. The first prefills the agent's own prompt, which no cache could
    hold; it is the same in both lanes (about 67k tokens for 16 agents, 267k
    for 64) and is left out.
  - **Context hit**: the share of the context sent again that the prefix cache
    already held. On the recorded races, 99.8% or more for SR (only the end of
    each context, short of a full 16-token cache block, is recomputed) and
    between 3.6% and 16.2% for stock vLLM LoRA, whose cache is keyed on the
    adapter (highest for the ReAct agents, whose steps more often stay on the
    adapter or the base model they ran before).
- **The result**, once both lanes are done: each lane's time and the ratio.

## Running it

1. Run the race on a pod, through the benchmark's script mode, which installs
   the commit and ships the harness. A job ships the race's code only when its
   script is in `race/`, with the switch benchmark's scripts; every other job
   leaves it out, keeping the harness within its size limit:

   ```bash
   benchmarks/adapter_eval/vela/submit.sh script b7e54212f5d73b9d971c00ba2cb58b6612178f33 benchmarks/adapter_eval/race/run_race.py
   ```

   It takes the workload's settings from `switching` in `adapters.yaml` and the
   synthetic adapters from the benchmark's cache (built if missing), composes and
   checks the SR checkpoint, and runs both lanes. Add
   `--extra-args "--model granite-4.2-3b"` for another model, and the options in
   the table above for the other races.

2. Put the race into the page, from the pod's saved log:

   ```bash
   python -m benchmarks.adapter_eval.race.build_page benchmarks/adapter_eval/vela/local/logs/<job>.log
   ```

   The page keeps one race per configuration: this run replaces the race of
   its own and keeps the others. It refuses a race that
   carries an absolute path anywhere.

   The 16-agent race was recorded (2026-10-11) before the recorder kept
   switches. They were added from its own workload, regenerated by the same
   generator with the same seed and checked against the recorded agents and
   stock vLLM's adapters, at its recorded token marks, exactly where the
   recorder now puts them; its times and counts are as recorded:

   ```bash
   python -m benchmarks.adapter_eval.race.build_page --add-switches <grid.jsonl>
   ```

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
| `benchmarks/adapter_eval/race/react.py` | the ReAct race's agents, from the switch benchmark's generator |
| `benchmarks/gen_agent_configs.py` | that generator, copied unchanged |
| `benchmarks/adapter_eval/race/build_page.py` | the race into the page |
| `docs/agent_race.html` | the page |
| `docs/generation/generate_race_diagram.py` | the page's diagrams, and `docs/media/agent_race_diagram*.svg` |
| `tests/unit/test_agent_race.py` | the race end to end on CPU, with a fake engine |
