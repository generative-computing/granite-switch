#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit 5cf9aef) and kept unchanged, so this benchmark
# builds ReAct agents exactly as it does. Update it only by copying it again.
"""Generate the agent fleet for the multi-agent ReAct experiment.

One config per agent, written as JSONL so the driver reads it and BOTH arms replay the identical fleet:
same prompts, same spans, same node types, same tool draws, same deployment categories. Two arms on
different fleets would not be a comparison.

WHAT IS FIXED HERE AND WHY

Spans are engineered so every decile boundary falls exactly on a span boundary. Within each decile chunk
spans are drawn from the node-type distribution until the chunk is nearly full, then the final span in
the chunk is set to close it exactly. That distorts roughly one span per chunk in LENGTH only, never in
node type, which is a smaller price than interpolating a stopwatch between boundaries.

KV BUDGET is the binding constraint and is checked, not assumed. A measured 667,632 tokens fit on one
80GB A100 at TP=1 for granite-4.1-3b bf16; past ~85% of that vLLM preempts and swaps, which silently
inflates exactly the timings being measured. With 33 concurrent agents and L=10k, decode alone is
330,000 tokens, leaving ~237k for 33 prompts -- so the prompt ladder is capped to fit and the script
refuses to emit a fleet that would not.

Node types carry their own span distributions and their own switch probability, because a "prepare
function call" node needs its adapter almost always and is a handful of tokens, while "think" may not
switch at all and runs long. Both facts move the result, so neither is hardcoded to a single number.
"""

import argparse
import json
import math
import pathlib
import random

# --- measured capacity, from the round-trip job's own log -------------------------------------
KV_TOKENS_TP1 = 667_632
SAFE_FRACTION = 0.85

# --- RTT categories, effective column of rtt_table.tsv (ms). Log-uniform within each range. ----
# The effective column, not the raw one: an LLM call pays serialization, HTTP and (cat 6) a gateway.
RTT_MS = {
    1: (0.2, 1.0),  # same process
    2: (1.0, 5.0),  # same host, separate process
    3: (2.0, 10.0),  # same cluster / datacenter
    4: (5.0, 30.0),  # same region, different zone
    5: (50.0, 200.0),  # cross-region
    6: (
        100.0,
        500.0,
    ),  # public API with gateway/auth/queue -- run as its own variant, see the doc
}

# --- tools: each agent carries TOOLS_PER_AGENT distinct tools, each of one of these categories -----
#
# A real ReAct agent has a sizeable toolbox, not one tool: twenty distinct tools, each belonging to a
# latency category, and every act node draws which tool it reaches for. Only the category sets the
# latency, so what a fleet-level result depends on is the category mix, not the tool identities -- but
# the identities are recorded anyway so both arms replay the same sequence and so a per-tool breakdown
# is possible later without regenerating.
#
# Tool latency is identical in both arms -- both stop at act nodes -- so none of this changes a
# comparison. It is realism, and it keeps between-agent variance in the prompt and cycle axes.
#
# It would stop being a detail the moment a SLOW tool joined: ~37 act nodes x 1 s for a web search is
# ~37 s per agent, which starts diluting the gap being measured. All three below are fast enough not to.
TOOLS_PER_AGENT = 20
TOOLS = {
    "calculator": (1.0, 10.0),
    "sql_lookup": (5.0, 50.0),
    "rag_no_rerank": (20.0, 80.0),
}

#: Node types. `segs_lambda` is the Poisson mean for EXTRA switches inside the node, on top of the one
#: at its start -- a long "think" really does move between base and several adapters mid-thought, and
#: modelling that as one atomic node understated the switch count. Short nodes get 0: a 10-token
#: prepare_call cannot be split at a 4-token floor, and pretending otherwise would invent boundaries.
#:
#: A tool node generates no tokens -- it is a boundary plus a tool latency, paid by BOTH arms.
MIN_SEGMENT = 4
NODE_TYPES = {
    "think": dict(median=120, sigma=0.55, p_switch=0.35, tool=False, segs_lambda=2.0),
    "prepare_call": dict(
        median=10, sigma=0.45, p_switch=0.95, tool=False, segs_lambda=0.0
    ),
    "act": dict(median=0, sigma=0.0, p_switch=0.00, tool=True, segs_lambda=0.0),
    "observe": dict(median=35, sigma=0.50, p_switch=0.80, tool=False, segs_lambda=0.5),
    "reflect": dict(median=90, sigma=0.55, p_switch=0.70, tool=False, segs_lambda=1.5),
}

#: ReAct cycles to draw from. The composition sets tokens-per-switch, which is what native pays per
#: unit of work -- so it is the single strongest lever on the result and is varied across the fleet
#: rather than fixed. Measured gap at batch 32 runs 3.2x at span 32 and 4.9x at span 8.
CYCLES = {
    "classic5": ["think", "prepare_call", "act", "observe", "reflect"],
    "toolheavy10": [
        "think",
        "prepare_call",
        "act",
        "observe",
        "prepare_call",
        "act",
        "observe",
        "prepare_call",
        "act",
        "reflect",
    ],
    "shortadapters": [
        "think",
        "prepare_call",
        "prepare_call",
        "prepare_call",
        "act",
        "observe",
        "prepare_call",
        "prepare_call",
        "reflect",
    ],
}


def lognormal_span(rng, median, sigma, lo=4, hi=512):
    """A log-normal span truncated to [lo, hi], rounded to an integer >= 1."""
    v = median * math.exp(rng.gauss(0.0, sigma))
    return int(max(lo, min(hi, round(v))))


def prompt_ladder(n, budget):
    """n prompt lengths spanning small to very large, summing to <= budget.

    Geometric so the fleet covers a wide range rather than clustering, then scaled down if it would
    not fit. Returns integers.
    """
    raw = [2 ** (8 + (i * 7) // max(1, n - 1)) for i in range(n)]  # 256 .. 32768
    total = sum(raw)
    if total > budget:
        scale = budget / total
        raw = [max(128, int(v * scale)) for v in raw]
    return raw


def split_segments(rng, span, lam):
    """Split one node's span into sub-spans, each an internal switch.

    Count is 1 + Poisson(lam), capped so every segment holds at least MIN_SEGMENT tokens -- so a short
    node stays whole rather than being cut into boundaries that could not exist. Sizes come from a
    Dirichlet-ish normalisation of uniform draws, then the remainder lands on the last segment so the
    node's total is preserved EXACTLY. Node totals are what the decile alignment rests on.
    """
    if lam <= 0 or span < 2 * MIN_SEGMENT:
        return [span]
    n = 1 + min(
        rng.poisson(lam) if hasattr(rng, "poisson") else _poisson(rng, lam),
        span // MIN_SEGMENT - 1,
    )
    if n <= 1:
        return [span]
    w = [rng.random() for _ in range(n)]
    tot = sum(w)
    segs = [max(MIN_SEGMENT, int(span * x / tot)) for x in w]
    drift = span - sum(segs)
    segs[-1] += drift
    if segs[-1] < MIN_SEGMENT:  # give back rather than emit a sub-minimum segment
        return [span]
    assert sum(segs) == span
    return segs


def _poisson(rng, lam):
    """Knuth's Poisson sampler -- random.Random has no poisson()."""
    import math

    ell, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= ell:
            return k
        k += 1


def build_spans(rng, total, cycle, deciles=10):
    """Spans whose cumulative sums hit every decile boundary EXACTLY.

    Fills each decile chunk with draws from the node-type distributions, cycling node types, then sets
    the last span in the chunk to close it exactly. Tool nodes contribute 0 tokens but are recorded in
    order, because both arms stop at them.
    """
    assert total % deciles == 0, "total decode must divide evenly into deciles"
    chunk = total // deciles
    nodes, ci = [], 0
    for _ in range(deciles):
        used = 0
        while True:
            ntype = cycle[ci % len(cycle)]
            ci += 1
            spec = NODE_TYPES[ntype]
            if spec["tool"]:
                nodes.append(dict(type=ntype, span=0, switch=False, tool=True))
                continue
            span = lognormal_span(rng, spec["median"], spec["sigma"])
            remaining = chunk - used
            if span >= remaining:
                # close the chunk exactly; distorts this span's LENGTH only, never its type
                nodes.append(
                    dict(
                        type=ntype,
                        span=remaining,
                        switch=rng.random() < spec["p_switch"],
                        tool=False,
                        chunk_closer=True,
                        segments=split_segments(rng, remaining, spec["segs_lambda"]),
                    )
                )
                used = chunk
                break
            nodes.append(
                dict(
                    type=ntype,
                    span=span,
                    switch=rng.random() < spec["p_switch"],
                    tool=False,
                    segments=split_segments(rng, span, spec["segs_lambda"]),
                )
            )
            used += span
    assert sum(n["span"] for n in nodes) == total
    return nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument(
        "--waves",
        type=int,
        default=5,
        help="number of sequential waves; wave sizes are BALANCED across them",
    )
    ap.add_argument("--decode", type=int, default=10_000)
    ap.add_argument(
        "--adapters",
        type=int,
        default=64,
        help="pool size M. The lever on native's re-prefill: at most M of an agent's "
        "switches can be first visits, the rest are forced revisits that hit the "
        "name-keyed prefix cache. CEILING 64 -- see the comment below.",
    )
    ap.add_argument(
        "--novelty",
        type=float,
        default=0.9,
        help="probability of preferring an adapter this agent has not used yet",
    )
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="scratch/agent_fleet.jsonl")
    args = ap.parse_args()

    # M IS CAPPED AT 64 BY THE KERNEL, not by taste. The SWITCH fused-LoRA kernel encodes each module's
    # applicable adapters as tile bits in an int64, so lora_kernel_meta.register_remap_tables asserts
    # at most 64 kernel-local adapters per module. Every gs-* arm goes through that kernel, and this
    # fleet is homogeneous -- every adapter targets every module -- so "per module" equals M here.
    #
    # It is a LOAD-TIME limit, and the composer does not know about it: switch-sr-100a-r32 composed
    # cleanly and reported VERIFY_OK, then vLLM refused it. So a checkpoint existing is not evidence it
    # can be served.
    #
    # native-lora alone could hold more, since it uses vLLM's own LoRA path rather than the SWITCH
    # kernel. It must not: unequal pools between the arms is what produced the ~77% re-prefill artefact
    # in SWITCH_MULTI_TURN.md 5.0, where the arms rotated over different-sized pools and it read as a
    # measurement fault. Both arms hold the same M or the comparison means nothing.
    if args.adapters > 64:
        raise SystemExit(
            f"FATAL: --adapters {args.adapters} exceeds the 64 the SWITCH kernel's int64 tile bitmask "
            f"supports per module (lora_kernel_meta.py register_remap_tables). A checkpoint this large "
            f"composes and verifies, then fails to load in vLLM."
        )

    rng = random.Random(args.seed)

    # BALANCED waves, five of twenty. The default matters more than it looks: `concurrent` is the
    # LARGEST wave, and it sets the prompt ladder, so 100 agents over 3 waves builds a 34-rung ladder
    # while 5 waves builds a 20-rung one over the same KV budget -- coarser rungs, larger prompts, and
    # a different experiment. Changing --waves therefore changes every agent's prompt, not just how the
    # work is grouped. Keep this default in step with the committed fleet in docs/switch_data_agent_sim.
    #
    # Balanced rather than fixed-size: 100 agents at a fixed 33 per wave leaves a fourth wave holding
    # ONE agent, and a lone agent has the whole GPU to itself -- it would finish roughly 30x faster
    # than any other and drag the p95 the whole design rests on. Sizes are split as evenly as possible
    # and the KV budget is checked against the LARGEST wave.
    base, extra = divmod(args.agents, args.waves)
    wave_sizes = [base + (1 if w < extra else 0) for w in range(args.waves)]
    concurrent = max(wave_sizes)

    # KV budget: the binding constraint, checked rather than assumed
    safe = int(KV_TOKENS_TP1 * SAFE_FRACTION)
    decode_load = concurrent * args.decode
    prompt_budget = safe - decode_load
    if prompt_budget <= concurrent * 128:
        raise SystemExit(
            f"FATAL: {concurrent} concurrent x {args.decode} decode = {decode_load:,} tokens "
            f"leaves {prompt_budget:,} for prompts out of a safe {safe:,} "
            f"({SAFE_FRACTION:.0%} of a measured {KV_TOKENS_TP1:,}). Raise --waves or lower --decode."
        )

    # the ladder is sized for the LARGEST wave, so every wave fits
    per_wave = prompt_ladder(concurrent, prompt_budget)

    # agent -> (wave, slot within that wave)
    placement = []
    for w, size in enumerate(wave_sizes):
        placement.extend((w, slot) for slot in range(size))

    fleet = []
    for i in range(args.agents):
        wave, slot = placement[i]
        cycle_name = rng.choice(list(CYCLES))
        cycle = CYCLES[cycle_name]
        nodes = build_spans(rng, args.decode, cycle)
        # the agent's toolbox: TOOLS_PER_AGENT distinct tools, each assigned a latency category, then
        # which tool every act node reaches for. Recorded so both arms replay identical tool latencies.
        toolbox = [
            dict(name=f"t{t:02d}", category=rng.choice(sorted(TOOLS)))
            for t in range(TOOLS_PER_AGENT)
        ]
        tool_draws = [rng.randrange(TOOLS_PER_AGENT) for n in nodes if n["tool"]]
        # ONE ADAPTER PER SEGMENT, chosen novelty-first. An agent working through a task reaches for
        # capabilities it has not used yet rather than re-drawing uniformly, so unused adapters are
        # preferred and the least-recently-used one is the fallback once the pool is exhausted.
        #
        # This is the choice that sets native's bill. The prefix cache is keyed on lora_name, so a
        # FIRST visit recomputes prompt + everything generated so far, while a revisit recomputes only
        # what grew since that adapter's own previous turn. At most M switches can be first visits.
        # `cur` starts on BASE and persists across nodes: a node whose p_switch did not fire simply
        # continues on whatever was already active. Base is a first-class cache key -- an unbound
        # request has its own prefix-cache namespace, so returning to base recovers base's prefix
        # exactly as returning to an adapter recovers that adapter's.
        seen: dict[str, int] = {}
        cur = "base"
        cum = 0
        switches = 0
        for node in nodes:
            if node["tool"]:
                node["segments_detail"] = []
                continue
            detail = []
            for si, seg in enumerate(node.get("segments", [node["span"]])):
                is_switch = node["switch"] if si == 0 else True
                if is_switch:
                    unused = [j for j in range(args.adapters) if f"ad{j}" not in seen]
                    if unused and rng.random() < args.novelty:
                        cur = f"ad{rng.choice(unused)}"
                    elif seen:
                        cur = min(seen, key=seen.get)
                    else:
                        cur = f"ad{rng.randrange(args.adapters)}"
                    switches += 1
                    first = cur not in seen
                    # A FIRST visit recomputes the whole prefix -- prompt plus everything generated so
                    # far -- because vLLM keys its prefix cache on the LoRA name and has nothing under
                    # this one. A revisit recomputes only what grew since this adapter's own last turn.
                    recompute = per_wave[slot] + cum if first else cum - seen[cur]
                else:
                    first, recompute = False, 0
                detail.append(
                    dict(
                        adapter=cur,
                        tokens=seg,
                        switch=is_switch,
                        first_visit=first,
                        native_recompute=recompute,
                    )
                )
                cum += seg
                seen[cur] = cum
            node["segments_detail"] = detail
        tool_nodes = sum(1 for n in nodes if n["tool"])
        segs = [d for n in nodes for d in n["segments_detail"]]
        fleet.append(
            dict(
                agent_id=i,
                n_segments=len(segs),
                n_first_visits=sum(1 for d in segs if d["first_visit"]),
                native_recompute_tokens=sum(d["native_recompute"] for d in segs),
                wave=wave,
                rtt_category=rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 4, 2, 1])[0],
                toolbox=toolbox,
                tool_draws=tool_draws,
                cycle=cycle_name,
                prompt_tokens=per_wave[slot],
                decode_tokens=args.decode,
                n_nodes=len(nodes),
                n_switches=switches,
                n_tool_nodes=tool_nodes,
                effective_span=round(args.decode / max(1, switches), 1),
                nodes=nodes,
            )
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for a in fleet:
            fh.write(json.dumps(a) + "\n")

    # --- census, so the fleet is inspectable before any GPU time -----------------------------
    import statistics as st

    print(
        f"wrote {out}  ({len(fleet)} agents in {len(wave_sizes)} wave(s) of {wave_sizes})"
    )
    print(
        f"  KV: decode {decode_load:,} + prompts {sum(per_wave):,} = "
        f"{decode_load + sum(per_wave):,} of safe {safe:,} "
        f"({(decode_load + sum(per_wave)) / KV_TOKENS_TP1:.0%} of measured capacity)"
    )
    print(
        f"  prompts per wave: {min(per_wave):,} .. {max(per_wave):,}  (sum {sum(per_wave):,})"
    )
    sw = [a["n_switches"] for a in fleet]
    es = [a["effective_span"] for a in fleet]
    nn = [a["n_nodes"] for a in fleet]
    print(f"  switches/agent: min {min(sw)} median {int(st.median(sw))} max {max(sw)}")
    print(f"  effective span: min {min(es)} median {st.median(es)} max {max(es)}")
    print(f"  nodes/agent:    min {min(nn)} median {int(st.median(nn))} max {max(nn)}")
    sg = [a["n_segments"] for a in fleet]
    fv = [a["n_first_visits"] for a in fleet]
    rc = [a["native_recompute_tokens"] for a in fleet]
    print(f"  segments/agent: min {min(sg)} median {int(st.median(sg))} max {max(sg)}")
    print(
        f"  first visits/agent: median {int(st.median(fv))} of {int(st.median(sw))} switches "
        f"({st.median(fv) / st.median(sw):.0%})"
    )
    print(
        f"  native recompute: {sum(rc):,} tokens vs {sum(a['decode_tokens'] for a in fleet):,} "
        f"generated = {sum(rc) / sum(a['decode_tokens'] for a in fleet):.1f}x useful work"
    )
    cat_counts, draw_cats, distinct = {}, {}, []
    for a in fleet:
        for t in a["toolbox"]:
            cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1
        for idx in a["tool_draws"]:
            c = a["toolbox"][idx]["category"]
            draw_cats[c] = draw_cats.get(c, 0) + 1
        distinct.append(len({a["toolbox"][i]["name"] for i in a["tool_draws"]}))
    print(
        f"  tools per agent: {TOOLS_PER_AGENT}   "
        f"categories across the fleet: "
        + "  ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()))
    )
    print(
        "  act-node draws by category: "
        + "  ".join(f"{k}={v:,}" for k, v in sorted(draw_cats.items()))
    )
    print(
        f"  distinct tools actually used per agent: min {min(distinct)} "
        f"median {int(st.median(distinct))} max {max(distinct)} of {TOOLS_PER_AGENT}"
    )
    for key in ("cycle", "rtt_category"):
        c = {}
        for a in fleet:
            c[a[key]] = c.get(a[key], 0) + 1
        print(
            f"  {key}: "
            + "  ".join(
                f"{k}={v}" for k, v in sorted(c.items(), key=lambda x: str(x[0]))
            )
        )
    # every agent must hit its deciles exactly, or the stopwatch design is void
    bad = []
    for a in fleet:
        cum, marks = 0, set()
        for n in a["nodes"]:
            cum += n["span"]
            marks.add(cum)
        if not all(a["decode_tokens"] * d // 10 in marks for d in range(1, 11)):
            bad.append(a["agent_id"])
    print(
        f"  decile alignment: {'ALL AGENTS OK' if not bad else f'FAILED for {bad[:5]}'}"
    )


if __name__ == "__main__":
    main()
