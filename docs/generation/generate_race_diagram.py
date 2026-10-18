#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the agent race page's diagrams (docs/agent_race.html).

Two diagrams, one per kind of agent, each drawn from a recorded agent:

* ``grid``: the switching workload's agent, generating 32-token segments on a
  new adapter each, with a tool call after every 256 tokens;
* ``react``: the paper's ReAct agent, its steps (think, prepare_call, act,
  observe, reflect) at their own lengths, each segment on its adapter.

Below each, what each lane does at an adapter switch, and the round trip. The
colors are the page's phase colors, so a diagram matches the race's bars.

Usage:
    python docs/generation/generate_race_diagram.py grid      # inline, for the page
    python docs/generation/generate_race_diagram.py react
    python docs/generation/generate_race_diagram.py --standalone grid \\
        > docs/media/agent_race_diagram.svg                   # an image of its own
"""

import sys

DEC, PRE, RTT, TOOL, DONE = "#4fc3f7", "#ef5350", "#80deea", "#fff176", "#66bb6a"
BASE = "#9e9e9e"
W = 1024
# The grid race's agent 6 (64 agents, 512 tokens): its adapters, in order.
GRID_ADAPTERS = [
    "ad25", "ad57", "ad34", "ad14", "ad24", "ad4", "ad2", "ad40",
    "ad41", "ad10", "ad7", "ad47", "ad45", "ad26", "ad54", "ad56",
]  # fmt: skip
# The ReAct race's agent 4 (512 tokens, the classic5 cycle): its first cycle,
# each step's segments as (adapter, tokens).
REACT_STEPS = [
    ("think", [("base", 35), ("ad26", 18), ("ad40", 65), ("ad19", 50)]),
    ("prepare_call", [("ad9", 17)]),
    ("act", []),
    ("observe", [("ad18", 4), ("ad54", 11), ("ad20", 13)]),
    ("reflect", [("ad29", 103), ("ad36", 56)]),
]
STEP_HELP = {
    "think": "think (generation): reasons about the task. Median 120 tokens; "
    "switches adapter as it starts 35% of the time, and again inside it: "
    "1 + Poisson(2) segments.",
    "prepare_call": "prepare_call (generation): writes the tool call. Median 10 "
    "tokens; its own adapter 95% of the time; too short to split.",
    "act": "act: the tool call. Calculator (1&#8211;10 ms), SQL lookup (5&#8211;50 ms) or RAG "
    "retrieval (20&#8211;80 ms), slept for real. Both lanes make the same calls.",
    "observe": "observe (generation): reads the tool's result. Median 35 tokens; "
    "switches 80% of the time; 1 + Poisson(0.5) segments.",
    "reflect": "reflect (generation): weighs the result. Median 90 tokens; "
    "switches 70% of the time; 1 + Poisson(1.5) segments.",
}
out = []


def text(x, y, s, cls, anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    out.append(f'<text x="{x:g}" y="{y:g}" class="{cls}"{a}>{s}</text>')


def block(x, y, w, h, title, fill=None, stroke=None, label=None, cls="d-chip"):
    if fill:
        paint = f'fill="{fill}"'
    else:
        paint = f'fill="none" stroke="{stroke}" stroke-width="1.5"'
    inner = f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="2" {paint}/>'
    if label:
        inner += (
            f'<text x="{x + w / 2:g}" y="{y + h / 2 + 4:g}" class="{cls}" '
            f'text-anchor="middle">{label}</text>'
        )
    out.append(f"<g><title>{title}</title>{inner}</g>")


def arrow(x1, x2, y):
    out.append(
        f'<path d="M{x1:g},{y:g} H{x2:g}" class="d-arrow" marker-end="url(#dArrow)"/>'
    )


def bracket(x1, x2, y, label):
    out.append(f'<path d="M{x1:g},{y:g} v4 H{x2:g} v-4" class="d-rule"/>')
    text((x1 + x2) / 2, y + 18, label, "d-sub", "middle")


def prompt_block(y):
    block(
        0, y, 100, 30,
        "The agent's own prompt, 256 to 16,384 tokens, shared with no other agent. "
        "Its first request prefills it, in both lanes.",
        stroke=PRE, label="prompt", cls="d-lab",
    )  # fmt: skip
    text(50, y + 54, "256&#8211;16,384 tok", "d-sub", "middle")


def done_block(x, y, tokens):
    block(
        x, y, 64, 30, "Done: the agent's last token. Its slot goes to the next agent.",
        stroke=DONE, label="done", cls="d-lab",
    )  # fmt: skip
    text(x + 32, y + 54, tokens, "d-sub", "middle")


def grid_agent(y):
    """The switching workload's agent: 32-token segments, a tool call at 256."""
    prompt_block(y)
    arrow(102, 118, y + 15)

    def run(x0, names):
        for i, ad in enumerate(names):
            block(
                x0 + i * 43, y, 40, 30, f"generation: 32 tokens on adapter {ad}",
                fill=DEC, label=ad,
            )  # fmt: skip
        return x0 + len(names) * 43 - 3

    text(120, y - 5, "generation", "d-step")
    end1 = run(120, GRID_ADAPTERS[:8])
    bracket(120, end1, y + 36, "256 tokens · switching adapter every 32")
    arrow(end1 + 2, end1 + 18, y + 15)
    tx = end1 + 20
    text(tx, y - 5, "act", "d-step")
    block(tx, y, 92, 30, STEP_HELP["act"], fill=TOOL, label="tool call")
    text(tx + 46, y + 54, "1&#8211;80 ms", "d-sub", "middle")
    arrow(tx + 94, tx + 110, y + 15)
    x2 = tx + 112
    text(x2, y - 5, "generation", "d-step")
    end2 = run(x2, GRID_ADAPTERS[8:])
    bracket(x2, end2, y + 36, "another 256 tokens")
    arrow(end2 + 2, end2 + 18, y + 15)
    done_block(end2 + 20, y, "512 tok")
    text(
        0, y + 78,
        "These agents are simplified: every step is a 32-token generation on a new "
        "adapter. At 1,024 tokens: 3 tool calls; at 128 or 256: none.",
        "d-note",
    )  # fmt: skip


def react_agent(y):
    """The paper's ReAct agent: its first cycle, each step at its own length (a short
    step drawn at a readable minimum), its name above it."""
    prompt_block(y)
    arrow(102, 118, y + 15)
    x, scale = 120.0, 1.3  # pixels per token
    for i, (step, segs) in enumerate(REACT_STEPS):
        tokens = sum(t for _, t in segs)
        start = x
        if step == "act":
            block(x, y, 64, 30, STEP_HELP["act"], fill=TOOL, label="tool")
            text(x + 32, y + 54, "1&#8211;80 ms", "d-sub", "middle")
            x += 64
        else:
            widths = [max(t * scale, 26) for _, t in segs]
            for (ad, t), w in zip(segs, widths, strict=True):
                where = "the base model" if ad == "base" else "adapter " + ad
                block(
                    x, y, w - 2, 30,
                    f"{STEP_HELP[step]} This segment: {t} tokens on {where}.",
                    fill=BASE if ad == "base" else DEC,
                    label=ad if w >= 34 else None,
                )  # fmt: skip
                x += w
            n = len(segs)
            bracket(start, x - 2, y + 36, f"{n} seg" if x - start < 70 else
                    f"{n} segment" + "s" * (n > 1))  # fmt: skip
        # Step names alternate between two rows, so short steps' names don't meet;
        # an upper-row name has a tick down to its step.
        name = step if step == "act" else f"{step} ({tokens})"
        row = 14 * (i % 2)
        text(start, y - 5 - row, name, "d-step")
        if row:
            out.append(
                f'<path d="M{start + 3:g},{y - row + 1:g} V{y - 1:g}" class="d-rule"/>'
            )
        arrow(x + 1, x + 13, y + 15)
        x += 15
    text(x, y + 20, "… cycles again", "d-sub")
    done_block(W - 76, y, "128&#8211;1,024 tok")
    text(
        0, y + 78,
        "Agent 4's first cycle (classic5), at scale. The other cycles: toolheavy10 "
        "(3 tool calls a cycle) and shortadapters (5 prepare_calls).",
        "d-note",
    )  # fmt: skip


def lanes(y, segs):
    """Each lane at a switch, over the agent's first segments."""
    text(0, y, "AT EACH SWITCH · WHERE THE LANES DIFFER", "d-h")
    legend = [
        (DEC, "generation"), (PRE, "prefill"), (RTT, "adapter switching (rtt)"),
        (TOOL, "act (tool call)"),
    ]  # fmt: skip
    widths = [9 + 5 + len(s) * 6.3 for _, s in legend]
    lx = W - 6 - (sum(widths) + 16 * (len(legend) - 1))
    for (color, s), w in zip(legend, widths, strict=True):
        out.append(
            f'<rect x="{lx:g}" y="{y - 8}" width="9" height="9" rx="1" fill="{color}"/>'
        )
        text(lx + 14, y, s, "d-sub")
        lx += w + 16
    x0, h = 130, 24

    def seg(x, yy, ad):
        block(x, yy, 48, h, f"generation on {ad}", fill=DEC if ad != "base" else BASE,
              label=ad)  # fmt: skip

    ly = y + 11
    text(0, ly + 9, "lora-vllm", "d-lane")
    text(0, ly + 25, "stock vLLM LoRA", "d-sub")
    x = x0
    for i, ad in enumerate(segs[:5]):
        seg(x, ly, ad)
        x += 50
        if i == 4:
            break
        block(
            x, ly, 8, h,
            "adapter switching (rtt): the request ends, its tokens go back to the "
            "agent's software, which sends a new request on the next adapter. "
            "2&#8211;10 ms, slept for real.",
            fill=RTT,
        )  # fmt: skip
        x += 10
        block(
            x, ly, 100, h,
            "prefill: the context computed again, except what vLLM's prefix cache "
            "holds for this adapter (its cache is per adapter).",
            fill=PRE, label="prefill again",
        )  # fmt: skip
        x += 102
    text(x + 6, ly + 16, "…", "d-sub")
    text(
        x0, ly + 40,
        "A new request per adapter: an adapter switching (rtt), then a prefill of the "
        "context again.",
        "d-note",
    )  # fmt: skip

    sy = ly + 56
    text(0, sy + 9, "gs-sr-vllm", "d-lane")
    text(0, sy + 25, "granite-switch SR", "d-sub")
    x = x0
    for ad in segs[:7]:
        seg(x, sy, ad)
        x += 50
    block(
        x, sy, 10, h,
        "act (tool call): the request ends; the next one finds its context cached",
        fill=TOOL,
    )  # fmt: skip
    x += 12
    for ad in segs[7:14]:
        seg(x, sy, ad)
        x += 50
    block(x + 2, sy, 60, h, "done", stroke=DONE, label="done", cls="d-lab")
    text(
        x0, sy + 40,
        "One request until a tool call: its K/V come from the base model, so the "
        "adapter* changes in place.",
        "d-note",
    )  # fmt: skip
    text(W - 6, sy + 40, "time →", "d-sub", "end")
    return sy + 64


def rtt_note(y):
    text(0, y, "ADAPTER SWITCHING (RTT) · WHAT IS TAKEN", "d-h")
    lines = [
        "Stock vLLM sets a LoRA per request, so it can't change adapter inside a "
        "generation.",
        "At each switch, its tokens go back to the agent's software, which sends a new "
        "request on the next adapter.",
        "That round trip takes 2&#8211;10 ms (the same datacenter), drawn per switch and "
        "slept for real: the GPU serves other agents meanwhile.",
        "SR has none: it stops only at a tool call, as stock vLLM does.",
    ]
    for i, line in enumerate(lines):
        text(0, y + 16 + i * 14, line, "d-note")
    text(
        0, y + 80,
        "* Shown switching as often as LoRA. SR's switch isn't implemented yet, so "
        "the run kept one adapter; on the GPU, SR costs the same either way.",
        "d-note",
    )  # fmt: skip
    return y + 86


def diagram(kind) -> int:
    """Draws the diagram; returns its height."""
    text(0, 11, "EACH AGENT · THE SAME IN BOTH LANES", "d-h")
    if kind == "react":
        top = 46  # two rows of step names above the steps
        react_agent(top)
        segs = [ad for _, ss in REACT_STEPS for ad, _ in ss]
    else:
        top = 34
        grid_agent(top)
        segs = GRID_ADAPTERS
    end = lanes(top + 116, segs)
    return rtt_note(end + 10) + 6


STANDALONE = """<style>
  text { font-family: 'IBM Plex Mono', Consolas, 'Cascadia Code', Monaco, monospace; }
  .d-h    { fill: #9a9a9a; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; }
  .d-step { fill: #bdbdbd; font-size: 11px; font-weight: 600; }
  .d-lane { fill: #9cdcfe; font-size: 12px; font-weight: 600; }
  .d-chip { fill: #1e1e1e; font-size: 10px; }
  .d-lab  { fill: #cccccc; font-size: 11px; }
  .d-sub  { fill: #8a8a8a; font-size: 10.5px; }
  .d-note { fill: #8a8a8a; font-size: 11px; }
  .d-arrow, .d-rule { stroke: #666; stroke-width: 1; fill: none; }
</style>
<rect x="-16" y="-16" width="WIDTH" height="HEIGHT" fill="#1e1e1e"/>"""


def main(argv):
    standalone = "--standalone" in argv
    kinds = [a for a in argv if not a.startswith("--")] or ["grid"]
    kind = kinds[0]
    if kind not in ("grid", "react"):
        raise SystemExit(f"unknown diagram {kind!r}: grid or react")
    h = diagram(kind)
    view = f"-16 -16 {W + 32} {h + 32}" if standalone else f"0 0 {W} {h}"
    size = f' width="{W + 32}" height="{h + 32}"' if standalone else ""
    who = (
        "Each ReAct agent prefills its own prompt, then walks its steps: think, "
        "prepare_call, act (a tool call), observe and reflect, each at its own length, "
        "switching adapter between and inside steps."
        if kind == "react"
        else "Each agent prefills its own prompt, then generates in 32-token steps, "
        "switching adapter at each, with a tool call after every 256 tokens."
    )
    print(
        "\n".join(
            [
                f'<svg viewBox="{view}"{size} role="img" '
                f'aria-labelledby="dTitle-{kind} dDesc-{kind}" '
                'xmlns="http://www.w3.org/2000/svg">',
                f'<title id="dTitle-{kind}">The simulated agent, and what each lane '
                "does at an adapter switch</title>",
                f'<desc id="dDesc-{kind}">{who} At each switch, stock vLLM ends the '
                "request, waits an adapter switching round trip (2&#8211;10 ms) and prefills "
                "the context again; granite-switch SR keeps one request running until a "
                "tool call.</desc>",
                '<defs><marker id="dArrow" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 '
                'L0,10 z" fill="#777"/></marker></defs>',
                *(
                    [
                        STANDALONE.replace("WIDTH", str(W + 32)).replace(
                            "HEIGHT", str(h + 32)
                        )
                    ]
                    if standalone
                    else []
                ),
                *out,
                "</svg>",
            ]
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
