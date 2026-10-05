#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit 8533b5e) and kept unchanged, so this benchmark
# measures switching exactly as it does. Update it only by copying it again.
"""Fleets for the concurrency x switching-frequency grid.

One fleet per cell of the grid. A cell is (concurrency, span): `concurrency` agents released together,
each generating L tokens in segments of exactly `span` tokens, so the switch count per agent is L/span
and the only thing varying across a row of the grid is how often the adapter changes.

DIFFERENT FROM gen_agent_configs.py, which this deliberately does not extend. That generator samples a
ReAct trajectory -- node types, variable spans, a tool mix -- because the agent-runtime figure is about a
realistic workload. This grid is a controlled sweep: span is the independent variable, so it is FIXED
within a cell rather than drawn. Sharing a generator between the two would mean one of them silently
getting the other's span model.

What is held identical to the agent study, so the two figures can be read against each other:
  * the prompt ladder is log-spaced over 256..16,384 tokens -- the agent study's lower bound and one
    octave below its top, which is what shadow-residual's smaller KV cache allows at concurrency 64 --
    and each agent's filler is seeded on its own
    id so no two agents share a prefix (prefix caching is content-addressed; a shared filler would hand
    the re-prefilling arm back most of the cost being measured).
  * tool calls at the agent fleet's own rate, a median of 37 per 10,000 generated tokens.
  * the adapter pool is M distinct adapters with novelty-first selection: a first visit to an adapter
    recomputes the whole prefix, a revisit only what grew since that adapter's own previous turn, so
    which adapter comes next is what decides native's cost.

Deployment distance is ONE category here, not a mix: "same cluster / same datacenter", 2-10 ms effective,
which is the case an organization actually deploys. Drawn uniformly, as are tool latencies -- the shape
barely matters, since both arms stop at the same tool boundaries and sleep the same drawn sequence, so
the tool distribution cancels in the ratio. The switch round trip is native-only, and there uniform over
2-10 ms means 6 ms against log-uniform's 4.7, which is marginally conservative for shadow-residual.
"""

import argparse
import json
import pathlib
import random

#: Effective round-trip for "same cluster / same datacenter", milliseconds. Category 3 of rtt_table.tsv.
RTT_MS = (2.0, 10.0)
#: Tool latency ranges, milliseconds. Same three short categories as the agent fleet.
TOOL_MS = {
    "calculator": (1.0, 10.0),
    "sql_lookup": (5.0, 50.0),
    "rag_no_rerank": (20.0, 80.0),
}
#: Tool calls per generated token, from the agent fleet's median of 37 per 10,000.
TOOL_RATE = 37 / 10_000
#: Segments per node. The agent study lets a node hold several switches; here a node IS a segment, so the
#: grid's span is exactly the distance between switches and nothing averages it away.


#: Prompt ladder: 16 rungs log-spaced over the agent study's range, and the COUNT is what matters.
#:
#: It must tile every agent count in the grid, because a cell holds max(concurrency, 16) agents -- 16, 32 or
#: 64 -- and 16 divides all three. Then every cell holds a whole number of copies of the ladder and so has
#: an identical prompt distribution, which is the property that lets figure 1 attribute a rise in p95 to
#: contention. Sizing the ladder to the concurrency instead makes the prompt mix a second independent
#: variable: every agent carries the median prompt at C=1 while spanning the full range at C=64.
#:
#: The agent-runtime fleet's own 20-value ladder was tried first, to put literally the same prompts through
#: both studies, and it does not tile: cycling 20 values across 16 agents stops at 8,192 while 64 agents
#: reach 32,768, so the range itself varied with concurrency. Same range, same endpoints, 16 rungs instead.
#: The top rung is 16,384, not the agent study's 32,768, and the constraint is SHADOW-RESIDUAL's KV cache.
#: That checkpoint carries 64 rank-32 adapters in its weights, so it allocates 536,656 tokens of cache where
#: native, loading the base model with LoRA slots, gets 786,304 -- 32% less. At concurrency 64 a ladder
#: topping out at 32,768 needs 537,132 tokens, which is 100% of what SR has; the pilot aborted on exactly
#: that, and the abort is the gate working, since past ~85% vLLM preempts and swaps rather than failing and
#: every timing would have been inflated with nothing to show it.
#:
#: 16,384 brings concurrency 64 to 332,988 tokens, 62% of SR's cache. The alternative was capping concurrency
#: at 32 and keeping the agent study's range, which throws away the most contended point -- the one where the
#: effect is largest -- to preserve a nuisance variable. Shrinking the ladder instead changes every cell
#: uniformly, so the grid stays internally consistent; what it costs is that the grid spans 256..16,384 where
#: the agent study spans 256..32,768, and any comparison between them has to say so.
PROMPT_LO, PROMPT_HI, LADDER_RUNGS = 256, 16_384, 16


def ladder(n=LADDER_RUNGS, lo=PROMPT_LO, hi=PROMPT_HI):
    """`n` prompt lengths log-spaced over [lo, hi], ascending."""
    if n == 1:
        return [int(round((lo * hi) ** 0.5))]
    return [int(round(lo * (hi / lo) ** (k / (n - 1)))) for k in range(n)]


def build_agent(
    agent_id, wave, prompt_tokens, span, decode, adapters, rng, novelty=0.9
):
    """One agent: `decode` tokens in segments of `span`, with tool boundaries at the fleet's rate."""
    n_seg = decode // span
    tool_every = (
        max(1, int(round(1 / (TOOL_RATE * span)))) if TOOL_RATE * span < 1 else 1
    )
    nodes, seen, cum, recompute = [], {}, 0, 0
    for i in range(n_seg):
        # Novelty-first, as in the agent fleet: an adapter this agent has not used yet is preferred, so
        # first visits accumulate early and revisits dominate later -- which is what makes M the lever on
        # native's re-prefill rather than the switch count alone.
        unused = [a for a in range(adapters) if f"ad{a}" not in seen]
        if unused and rng.random() < novelty:
            cur = f"ad{rng.choice(unused)}"
        elif seen:
            cur = min(seen, key=lambda k: seen[k])
        else:
            cur = f"ad{rng.randrange(adapters)}"
        first = cur not in seen
        recompute += (prompt_tokens + cum) if first else (cum - seen[cur])
        nodes.append(
            dict(
                type="segment",
                span=span,
                switch=True,
                tool=False,
                segments=[span],
                segments_detail=[
                    dict(
                        adapter=cur,
                        tokens=span,
                        switch=True,
                        first_visit=first,
                        native_recompute=(prompt_tokens + cum)
                        if first
                        else (cum - seen[cur]),
                    )
                ],
            )
        )
        cum += span
        seen[cur] = cum
        if (i + 1) % tool_every == 0 and i + 1 < n_seg:
            nodes.append(
                dict(
                    type="tool",
                    span=0,
                    switch=False,
                    tool=True,
                    segments_detail=[
                        dict(
                            adapter=None,
                            tokens=0,
                            switch=False,
                            first_visit=False,
                            native_recompute=0,
                        )
                    ],
                )
            )
    toolbox = [
        dict(name=f"t{i:02d}", category=c) for i, c in enumerate(sorted(TOOL_MS))
    ]
    n_tools = sum(1 for n in nodes if n["tool"])
    return dict(
        agent_id=agent_id,
        wave=wave,
        concurrency=None,  # filled by the caller; the cell identity travels with the record
        span=span,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode,
        n_segments=n_seg,
        n_switches=n_seg,
        n_nodes=len(nodes),
        n_tool_nodes=n_tools,
        n_first_visits=sum(
            1 for n in nodes for d in n["segments_detail"] if d["first_visit"]
        ),
        native_recompute_tokens=recompute,
        effective_span=round(decode / max(1, n_seg), 1),
        rtt_category=3,
        rtt_ms=list(RTT_MS),
        cycle=f"grid-s{span}",
        toolbox=toolbox,
        tool_draws=[rng.randrange(len(toolbox)) for _ in range(n_tools + 2)],
        nodes=nodes,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--span", type=int, required=True)
    ap.add_argument("--decode", type=int, default=1024)
    ap.add_argument("--adapters", type=int, default=64)
    ap.add_argument(
        "--min-agents",
        type=int,
        default=16,
        help="run extra waves at low concurrency until the cell holds at least this many agents, so a "
        "p95 is taken over a comparable sample at every concurrency. At concurrency 1 a p95 over one "
        "agent is that agent.",
    )
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.decode % args.span:
        raise SystemExit(
            f"FATAL: decode {args.decode} is not a multiple of span {args.span}; the last segment would "
            f"be short and the switch count would not be decode/span."
        )

    rng = random.Random(args.seed ^ (args.concurrency << 8) ^ args.span)
    total = max(args.concurrency, args.min_agents)
    waves = -(-total // args.concurrency)  # ceil
    # Prompts are assigned from a SHUFFLED list, not by walking the ascending ladder.
    #
    # Walking it in order makes each wave a contiguous slice, so at concurrency 2 the last wave holds the two
    # largest prompts and the first the two smallest -- the giants contend only with each other. The pilot
    # showed it plainly: p95 went 74.8s at C=1 to 181.0s at C=2 while the median did not move, which is not
    # contention rising but the worst wave becoming uniquely bad. At C>=16 a cell is one wave holding the whole
    # ladder, so only the low-concurrency points were distorted -- which is exactly where figure 1 starts. A
    # deterministic shuffle makes every wave a representative sample at every concurrency, which is what
    # comparing across concurrency requires.
    rungs = ladder()
    prompts = (rungs * (-(-total // len(rungs))))[:total]
    random.Random(args.seed ^ 0x5EED).shuffle(prompts)
    fleet, aid = [], 0
    for w in range(waves):
        for _ in range(args.concurrency):
            if aid >= total:
                break
            a = build_agent(
                aid,
                w,
                prompts[aid],
                args.span,
                args.decode,
                args.adapters,
                rng,
            )
            a["concurrency"] = args.concurrency
            fleet.append(a)
            aid += 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(a) + "\n" for a in fleet))
    rc = sum(a["native_recompute_tokens"] for a in fleet)
    gen = sum(a["decode_tokens"] for a in fleet)
    print(
        f"C={args.concurrency} span={args.span}: {len(fleet)} agents in {waves} wave(s), "
        f"{fleet[0]['n_switches']} switches each, {fleet[0]['n_tool_nodes']} tool calls each"
    )
    print(
        f"  prompts {min(a['prompt_tokens'] for a in fleet):,}..{max(a['prompt_tokens'] for a in fleet):,}"
        f"   native recompute {rc / 1e6:.1f}M = {rc / gen:.0f}x generated   -> {out}"
    )


if __name__ == "__main__":
    main()
