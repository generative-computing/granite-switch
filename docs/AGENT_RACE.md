# Agent race: granite-switch SR against stock vLLM LoRA

[agent_race.html](agent_race.html) replays recorded races between two ways of
serving agents that switch adapters: granite-switch's Shadow Residual (SR) and
stock vLLM serving the same adapters as LoRAs. It looks and works like the
[ALORA-vs-LoRA race](race_live.html): a column per lane, a bar per agent, a
metrics panel per lane, and play, speed and scrub controls. The page opens
straight from disk: its races are embedded in it.

The code that runs and records the races lives with the adapter benchmark, on
branch `feature/adapter-benchmark` (`benchmarks/adapter_eval/race/`), because it
runs on that benchmark's harness and jobs. This page and its diagrams are all
that main holds.

## What races

**The lanes:**

| Lane | Serving | At an adapter switch |
|---|---|---|
| `gs-sr-vllm` | granite-switch SR: 64 SR adapters composed into one checkpoint, the switch benchmark's shadow-residual arm | nothing: its K/V come from the base model, so one generation runs until a tool call |
| `lora-vllm` | stock vLLM serving the 64 adapters as LoRAs (its native-lora arm) | ends the request, waits the round trip to the agent, and starts a new one, prefilling the agent's whole context again; vLLM's prefix cache is keyed on the LoRA, so only an adapter served before hits it |

**Adapter switching (rtt):** stock vLLM sets a LoRA per request, so it can't
change adapter inside a generation. At each switch, its tokens go back to the
agent's software, which sends a new request on the next adapter. That round trip
is what the page calls an adapter switching (rtt): 2 to 10 ms (the same
datacenter), drawn per switch and slept for real, so the GPU serves the other
agents meanwhile. A tool call (act) is the other wait: both lanes stop for it,
sleeping the same tools' latencies. SR has no adapter switching (rtt): its
adapter changes in place.

**The agents** run in slots: the next starts as soon as any running one
finishes. Each has a prompt of its own (256 to 16,384 tokens), and both lanes
run the same agents, with the same prompts, lengths, adapter schedules and
tool calls.

**The page holds five races**, all on Granite 4.1 3B at commit b7e54212, one
A100-80GB, vLLM 0.26.0:

| Agents | Per lane | At a time | Tokens per agent | SR | stock vLLM | SR sooner by |
|---|---|---|---|---|---|---|
| the paper's ReAct agents, classic5 | 64 | 16 | 128, 256, 512 or 1,024, 16 each | 90.4 s | 282.4 s | 3.1× |
| fixed steps | 16 | 8 | 512 | 28.4 s | 91.8 s | 3.2× |
| fixed steps | 64 | 8 | 512 | 114.4 s | 365.0 s | 3.2× |
| fixed steps | 64 | 16 | 1,024 | 129.1 s | 747.7 s | 5.8× |
| fixed steps | 64 | 16 | 128, 256, 512 or 1,024, 16 each | 89.4 s | 335.9 s | 3.8× |

**The ReAct agents** are the switch benchmark's agent study's, from its generator,
round its classic5 cycle (think, prepare_call, act, observe, reflect, and around
again), at the generator's step lengths and switch rates:

| Step | Median tokens | Switches adapter | Inside the step |
|---|---|---|---|
| think | 120 | 35% of the time | 1 + Poisson(2) segments, a switch each |
| prepare_call | 10 | 95% | one segment |
| act | 0: a tool call | | the tool's latency, slept for real |
| observe | 35 | 80% | 1 + Poisson(0.5) segments |
| reflect | 90 | 70% | 1 + Poisson(1.5) segments |

Three settings are the race's, so its agents compare with the other races': 128
to 1,024 tokens (the paper's agents generate 10,000), the prompts above, and a
round trip within one datacenter (the paper draws a mix of distances). They
switch about every 41 tokens.

**The fixed-steps agents** are the adapter benchmark's switching workload: each
generates 32-token steps on a new adapter each, over 64 synthetic adapters of
rank 32, with a tool call after every 256 tokens.

**How it is measured:** both lanes run on one GPU, one after the other, each in
an engine of its own with fresh compile caches and after one untimed warm-up
run of the whole fleet, with the prefix cache reset before every run. The page
replays them side by side, each from its own start.

## What the page shows

- **At the top, only the demo:** a sentence on what races, the outcome (e.g.
  "granite-switch SR finished 3.1× sooner"), three tiles on why (prompt tokens
  computed again, requests, context already cached, for each lane), the chosen
  race's settings one labeled line each, a track per lane that fills as its
  agents finish, and the race. It opens on the ReAct race. The rest is behind
  buttons:
  - **How it works:** the experiment in a few lines, and a diagram of the race's
    agents: for the ReAct race, the agent as a state machine (LangGraph-style:
    its steps as nodes, its cycle as the edges) above a recorded agent's steps
    at their own lengths; for the others, an agent's 32-token steps; then what
    each lane does at a switch, and what an adapter switching (rtt) takes.
  - **The models:** each lane's setup: the base model (Granite 4.1 3B, 40
    layers), the 64 adapters, their rank (32, alpha 64) and the modules they
    adapt (SR: q, o, gate, up and down, with a cross-stream shunt per layer and
    no k or v; stock vLLM: q, k, v, o, gate, up and down).
  - **Metrics:** a panel per lane, below.
  - **More races:** the four fixed-steps races.

  Hovering a step or a metric's label explains it. The link keeps the chosen
  race (e.g. `#64-16-mix`).
- **A bar per agent**, filled by the tokens it has generated, colored by what it
  is doing: waiting for a slot; prefilling, with the prompt tokens it computes and
  those vLLM's prefix cache already held (e.g. `prefill 3 tok · 2.6k cached`);
  generating, with its step and adapter (e.g. `think (generation) ad26`, or
  `generation ad26` for the fixed-steps agents); on an adapter switching (rtt);
  on a tool call (act); or done, with its time.
- **The metrics panel per lane**, sampled every half second: agents running,
  prefilling, decoding and waiting; tokens per second; the adapter switches so
  far (the same in both lanes at the end) and the requests behind them (stock
  vLLM one per switch, SR one per stretch between tool calls); the average time
  to first token of the last 8 requests; and, from vLLM's own count of each
  request's cached prompt tokens:
  - **Re-prefilled**: prompt tokens computed again after an agent's first
    request. The first prefills the agent's own prompt, which no cache could
    hold; it is the same in both lanes and is left out.
  - **Context hit**: the share of the context sent again that the prefix cache
    already held: 99.8% or more for SR on every race (only the end of each
    context, short of a full 16-token cache block, is recomputed), and 3.6% to
    16.6% for stock vLLM.
- **The result**, once both lanes are done: each lane's time and the ratio.

## What the page shows that the runs do not do

- **SR is shown switching.** Its adapter changes on the same schedule as stock
  vLLM's, as many times. The switch itself is not implemented yet, so the runs
  kept one adapter per agent; on the GPU, SR costs the same with or without one,
  because its K/V come from the base model and its generation runs straight
  through the switch points. The page says so.
- **The 16-agent race's switches were added afterwards.** It was recorded before
  the recorder kept switches; they were added from its own workload,
  regenerated by the same generator with the same seed and checked against the
  recorded agents and stock vLLM's adapters, at the recorded token marks. Its
  times and counts are as recorded.

## A new race

On `feature/adapter-benchmark`, a race job runs both lanes on a pod and prints
the race into its log, and `benchmarks/adapter_eval/race/build_page.py` adds it
to this page, keeping one race per configuration (see that branch's
`docs/AGENT_RACE.md`). The updated page then comes to main in a PR of its own.

The diagrams are generated by
[generate_race_diagram.py](generation/generate_race_diagram.py), here on main:
`machine`, `grid` and `react` print each as the page holds it, and
`--standalone` writes one as an image of its own (`docs/media/agent_race_*.svg`).

The page holds the races' events (seconds from each lane's start), each lane's
agents (their prompt lengths, switches, tool calls and times), and the runs'
public details: model, commit, GPU, library versions, settings. No path on the
runs' storage, no host name.
