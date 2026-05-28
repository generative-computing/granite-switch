#!/usr/bin/env python3
"""Generate docs/compose_benchmark_short.svg — animated benchmark showcase.

Shows 18 slots (6 per model family) cycling through all 12 Granite Libraries
adapters. Uses SMIL <animate> for Safari compatibility.

Usage:
    python docs/generate_benchmark_svg.py > docs/compose_benchmark_short.svg
"""

# ── DATA ──────────────────────────────────────────────────────────────────────

MODELS = {
    "3b": {"label": "Granite 4.1 3B", "params": "3B parameters · Apache 2.0"},
    "8b": {"label": "Granite 4.1 8B", "params": "8B parameters · Apache 2.0"},
    "30b": {"label": "Granite 4.1 30B", "params": "30B parameters · Apache 2.0"},
}

# All 12 adapters from the catalog.
# "improved" uses aLoRA where available, falls back to LoRA.
# Values normalized to 0-100 scale (0-1 metrics × 100).
ADAPTERS = {
    "qr": {
        "label": "Query Rewrite", "lib": "raglib", "type": "alora",
        "snippet": "rag.rewrite_question(query, ctx, backend)",
        "base": {"3b": 61, "8b": 67, "30b": 64},
        "improved": {"3b": 86, "8b": 84, "30b": 87},
    },
    "qc": {
        "label": "Query Clarification", "lib": "raglib", "type": "alora",
        "snippet": "rag.clarify_query(query, docs, ctx, backend)",
        "base": {"3b": 65, "8b": 48, "30b": 76},
        "improved": {"3b": 95, "8b": 96, "30b": 95},
    },
    "an": {
        "label": "Answerability", "lib": "raglib", "type": "alora",
        "snippet": "rag.check_answerability(query, docs, ctx, backend)",
        "base": {"3b": 59, "8b": 66, "30b": 65},
        "improved": {"3b": 91, "8b": 91, "30b": 92},
    },
    "hd": {
        "label": "Hallucination Det.", "lib": "raglib", "type": "lora",
        "snippet": "rag.flag_hallucinated_content(response, docs, ctx, backend)",
        "base": {"3b": 42, "8b": 45, "30b": 54},
        "improved": {"3b": 71, "8b": 74, "30b": 83},
    },
    "ci": {
        "label": "Citations", "lib": "raglib", "type": "lora",
        "snippet": "rag.find_citations(answer, docs, ctx, backend)",
        "base": {"3b": 10, "8b": 18, "30b": 55},
        "improved": {"3b": 64, "8b": 66, "30b": 71},
    },
    "ca": {
        "label": "Context Attr.", "lib": "corelib", "type": "lora",
        "snippet": "core.find_context_attributions(answer, docs, ctx, backend)",
        "base": {"3b": 34, "8b": 27, "30b": 65},
        "improved": {"3b": 92, "8b": 96, "30b": 90},
    },
    "rc": {
        "label": "Req. Check", "lib": "corelib", "type": "alora",
        "snippet": "core.requirement_check(ctx, backend, requirements)",
        "base": {"3b": 51, "8b": 54, "30b": 57},
        "improved": {"3b": 77, "8b": 77, "30b": 78},
    },
    "un": {
        "label": "Uncertainty", "lib": "corelib", "type": "alora",
        "snippet": "core.check_certainty(ctx, backend)",
        "base": {"3b": 63, "8b": 64, "30b": 65},
        "improved": {"3b": 77, "8b": 78, "30b": 80},
    },
    "gc": {
        "label": "Guardian Core", "lib": "guardianlib", "type": "alora",
        "snippet": 'guardian.check(ctx, backend, "harm")',
        "base": {"3b": 1, "8b": 67, "30b": 69},
        "improved": {"3b": 78, "8b": 80, "30b": 81},
    },
    "fd": {
        "label": "Factuality Det.", "lib": "guardianlib", "type": "alora",
        "snippet": "guardian.factuality_detection(ctx, backend)",
        "base": {"3b": 8, "8b": 25, "30b": 51},
        "improved": {"3b": 81, "8b": 83, "30b": 83},
    },
    "fc": {
        "label": "Factuality Corr.", "lib": "guardianlib", "type": "alora",
        "snippet": "guardian.factuality_correction(ctx, backend)",
        "base": {"3b": 0, "8b": 19, "30b": 13},
        "improved": {"3b": 19, "8b": 23, "30b": 26},
    },
    "pg": {
        "label": "Policy Guardrails", "lib": "guardianlib", "type": "alora",
        "snippet": "guardian.policy_guardrails(ctx, backend, policy)",
        "base": {"3b": 82, "8b": 80, "30b": 88},
        "improved": {"3b": 95, "8b": 95, "30b": 96},
    },
}

LIBS = {
    "corelib":     {"fill": "#8A3FFC", "text": "white",   "sub": "#d4bbff",
                    "pill_bg": "#EDE7FF", "pill_text": "#8A3FFC", "bar": "#8A3FFC", "pct": "#8A3FFC"},
    "raglib":      {"fill": "#009D9A", "text": "white",   "sub": "#9ef0f0",
                    "pill_bg": "#E0F7F6", "pill_text": "#009D9A", "bar": "#009D9A", "pct": "#009D9A"},
    "guardianlib": {"fill": "#F1C21B", "text": "#161616", "sub": "#7a5f00",
                    "pill_bg": "#FEF3C7", "pill_text": "#B45309", "bar": "#F1C21B", "pct": "#B45309"},
}

# 9 slots: Latin square rotation — 3 adapter groups × 3 models.
# Each group is assigned to each model exactly once; models cycle every slot so
# consecutive slots always show a different model AND different adapters.
# Every adapter-model pair appears exactly once (36 unique combinations, zero repeats).
#
# Groups (lib-balanced: ~2 raglib, 1 corelib, 1 guardianlib each):
#   A: qr, an, ca, gc  (raglib×2, corelib×1, guardianlib×1)
#   B: qc, hd, rc, fd  (raglib×2, corelib×1, guardianlib×1)
#   C: ci, un, fc, pg  (raglib×1, corelib×1, guardianlib×2)
#
# Rotation:  3b→A  8b→B  30b→C  |  3b→B  8b→C  30b→A  |  3b→C  8b→A  30b→B
SLOTS = [
    # ── Cycle 1 ──
    {"model": "3b",  "adapters": ["qr", "an", "ca", "gc"]},
    {"model": "8b",  "adapters": ["qc", "hd", "rc", "fd"]},
    {"model": "30b", "adapters": ["ci", "un", "fc", "pg"]},
    # ── Cycle 2 ──
    {"model": "3b",  "adapters": ["qc", "hd", "rc", "fd"]},
    {"model": "8b",  "adapters": ["ci", "un", "fc", "pg"]},
    {"model": "30b", "adapters": ["qr", "an", "ca", "gc"]},
    # ── Cycle 3 ──
    {"model": "3b",  "adapters": ["ci", "un", "fc", "pg"]},
    {"model": "8b",  "adapters": ["qr", "an", "ca", "gc"]},
    {"model": "30b", "adapters": ["qc", "hd", "rc", "fd"]},
]

# Verify: all 12 adapters appear at least once
_seen = set()
for _s in SLOTS:
    _seen.update(_s["adapters"])
assert len(_seen) == 12, f"Only {len(_seen)} adapters covered: missing {set(ADAPTERS.keys()) - _seen}"


# ── LAYOUT CONSTANTS ──────────────────────────────────────────────────────────

SLOT_SECONDS = 10
TOTAL_DUR = len(SLOTS) * SLOT_SECONDS

SVG_HEIGHT = 490

# Bar geometry
BAR_X = 310
BAR_TRACK_W = 430
BAR_H = 16
BAR_RX = 4
PCT_X = 750

# Row positions (bottom to top: row 0 = bottom)
ROW_Y = [274, 200, 126, 52]          # baseline bar y
IMP_ROW_Y = [294, 220, 146, 72]      # improved bar y (baseline + 20)
BASE_TEXT_Y = [286, 212, 138, 64]    # baseline pct text y (ROW_Y + 12)
IMP_TEXT_Y = [306, 232, 158, 84]     # improved pct text y (IMP_ROW_Y + 12)

ADAPTER_Y = [196, 144, 92, 40]       # adapter box y
ADAPTER_TEXT1_Y = [216, 164, 112, 60]
ADAPTER_TEXT2_Y = [234, 182, 130, 78]

# Stagger delays for adapter rows (seconds into slot when row appears)
# Baseline bars finish growing at 1.5s (0.5s delay + 1.0s grow).
# 1s pause → first adapter tile at 2.5s.
ADAPTER_STAGGER = [2.5, 3.5, 4.5, 5.5]

GROW_SECONDS = 1.0  # duration of bar grow animation

# Code panel (below main content)
CODE_PANEL_Y = 390      # top of code block background rect
CODE_PANEL_H = 92       # height of code block rect
CODE_HEADER_Y = 404     # y for "MELLEA INVOCATION" label
CODE_LINE_Y = [421, 438, 455, 472]  # y per adapter row (row 0 first = top); 17px spacing


# ── HELPERS ───────────────────────────────────────────────────────────────────

def bar_width(pct_val):
    """Convert percentage (0-100) to bar pixel width (430px track)."""
    return round(pct_val / 100 * BAR_TRACK_W)


def smil_visibility(active_slot, stagger_s=0):
    """Return <animate> element that makes element visible only during active_slot."""
    n = len(SLOTS)
    if stagger_s == 0:
        values = ";".join("visible" if i == active_slot else "hidden" for i in range(n))
        key_times = ";".join(f"{i/n:.6f}" for i in range(n))
    else:
        points = []
        for i in range(n):
            slot_start = i / n
            if i == active_slot:
                appear_at = slot_start + stagger_s / TOTAL_DUR
                points.append((slot_start, "hidden"))
                points.append((appear_at, "visible"))
            else:
                points.append((slot_start, "hidden"))
        values = ";".join(v for _, v in points)
        key_times = ";".join(f"{t:.6f}" for t, _ in points)

    key_times = key_times.replace("0.000000", "0")
    return (f'<animate attributeName="visibility" values="{values}" '
            f'keyTimes="{key_times}" calcMode="discrete" '
            f'dur="{TOTAL_DUR}s" repeatCount="indefinite"/>')


def smil_animate_fill(values):
    """Return an <animate> element for fill color with discrete calcMode."""
    n = len(values)
    vals = ";".join(values)
    key_times = ";".join(f"{i/n:.6f}" for i in range(n))
    key_times = key_times.replace("0.000000", "0")
    return (f'<animate attributeName="fill" values="{vals}" '
            f'keyTimes="{key_times}" calcMode="discrete" '
            f'dur="{TOTAL_DUR}s" repeatCount="indefinite"/>')


def smil_animate_width_grow(values, delay_s=0):
    """Return <animate> for width that grows from 0 to target over GROW_SECONDS.

    SMIL spec requires keyTimes to span exactly [0, 1] for calcMode="linear".
    """
    n = len(values)
    points = []
    for i in range(n):
        slot_start = i / n
        grow_start = slot_start + delay_s / TOTAL_DUR
        grow_end = slot_start + (delay_s + GROW_SECONDS) / TOTAL_DUR
        next_slot = (i + 1) / n
        hold_end = next_slot - 0.0001

        points.append((slot_start, 0))
        if delay_s > 0:
            points.append((grow_start, 0))
        points.append((grow_end, values[i]))
        if i < n - 1:
            points.append((hold_end, values[i]))
    # Last keyTime must be exactly 1.0
    points.append((1.0, values[-1]))

    vals = ";".join(str(v) for _, v in points)
    key_times = ";".join(f"{t:.6f}" for t, _ in points)
    key_times = key_times.replace("0.000000", "0").replace("1.000000", "1")
    return (f'<animate attributeName="width" values="{vals}" '
            f'keyTimes="{key_times}" calcMode="linear" '
            f'dur="{TOTAL_DUR}s" repeatCount="indefinite"/>')


# ── SCHEDULE COMMENT ──────────────────────────────────────────────────────────

def generate_schedule_comment():
    """Generate the schedule comment block."""
    lines = []
    lines.append("<!--")
    lines.append(f"  SCHEDULE — {SLOT_SECONDS}s per slot · {TOTAL_DUR}s total ({len(SLOTS)} slots)")
    lines.append(f"  Generated by: python docs/generate_benchmark_svg.py")
    lines.append(f"  Animation: SMIL <animate> (no CSS keyframes)")
    lines.append("")
    for i, slot in enumerate(SLOTS):
        model = slot["model"]
        labels = [f"{ADAPTERS[sid]['label']}" for sid in slot["adapters"]]
        lines.append(f"  Slot {i+1:2d} [{model:>3s}]: {labels[0]:20s} {labels[1]:20s} {labels[2]:20s} {labels[3]}")
    lines.append("-->")
    return "\n".join(lines)


# ── LEFT PANEL ────────────────────────────────────────────────────────────────

def generate_left_panel():
    """Generate left panel: base model box + adapter boxes with SMIL animations."""
    lines = []
    lines.append("")
    lines.append("<!-- LEFT PANEL -->")
    lines.append('<text x="137" y="28" text-anchor="middle" font-size="10" fill="#8d8d8d" letter-spacing="1">CUSTOM GRANITE SWITCH MODEL</text>')
    lines.append("")

    # ── Base model box (static rect, text toggled per slot) ──
    lines.append("<!-- Base model box -->")
    lines.append('<rect x="22" y="248" width="230" height="106" rx="12" fill="#0F62FE"/>')

    for si, slot in enumerate(SLOTS):
        model = slot["model"]
        minfo = MODELS[model]
        lines.append(f'<g visibility="hidden">')
        lines.append(f'  {smil_visibility(si)}')
        lines.append(f'  <text x="137" y="277" text-anchor="middle" font-size="16" font-weight="700" fill="white">{minfo["label"]}</text>')
        lines.append(f'  <text x="137" y="297" text-anchor="middle" font-size="10" fill="#a8c4ff">base model</text>')
        lines.append(f'  <text x="137" y="315" text-anchor="middle" font-size="10" fill="#6ea6ff">{minfo["params"]}</text>')
        lines.append(f'  <text x="137" y="336" text-anchor="middle" font-size="9" fill="#4f83cc">ibm-granite \u00b7 HuggingFace</text>')
        lines.append(f'</g>')
    lines.append("")

    # ── Adapter boxes (4 rows, staggered appearance) ──
    for row_idx in range(4):
        y = ADAPTER_Y[row_idx]
        t1y = ADAPTER_TEXT1_Y[row_idx]
        t2y = ADAPTER_TEXT2_Y[row_idx]
        stagger = ADAPTER_STAGGER[row_idx]

        adapter_ids = [slot["adapters"][row_idx] for slot in SLOTS]
        fills = [LIBS[ADAPTERS[aid]["lib"]]["fill"] for aid in adapter_ids]

        lines.append(f"<!-- Adapter box row {row_idx} (appears {stagger}s into slot) -->")
        lines.append(f'<g visibility="hidden">')
        # Visibility with stagger
        vis_points = []
        for i in range(len(SLOTS)):
            slot_start = i / len(SLOTS)
            appear_at = slot_start + stagger / TOTAL_DUR
            vis_points.append((slot_start, "hidden"))
            vis_points.append((appear_at, "visible"))
        vis_values = ";".join(v for _, v in vis_points)
        vis_kt = ";".join(f"{t:.6f}" for t, _ in vis_points).replace("0.000000", "0")
        lines.append(f'  <animate attributeName="visibility" values="{vis_values}" keyTimes="{vis_kt}" calcMode="discrete" dur="{TOTAL_DUR}s" repeatCount="indefinite"/>')
        lines.append(f'  <rect x="22" y="{y}" width="230" height="48" rx="8" fill="{fills[0]}">')
        lines.append(f'    {smil_animate_fill(fills)}')
        lines.append(f'  </rect>')

        # Text variants
        for si, aid in enumerate(adapter_ids):
            info = ADAPTERS[aid]
            lib = LIBS[info["lib"]]
            lines.append(f'  <g visibility="hidden">')
            lines.append(f'    {smil_visibility(si, stagger)}')
            lines.append(f'    <text x="137" y="{t1y}" text-anchor="middle" font-size="12" font-weight="700" fill="{lib["text"]}">{info["label"]}</text>')
            lines.append(f'    <text x="137" y="{t2y}" text-anchor="middle" font-size="9" fill="{lib["sub"]}">{info["lib"]}</text>')
            lines.append(f'  </g>')
        lines.append(f'</g>')
    lines.append("")

    return "\n".join(lines)


# ── RIGHT PANEL ───────────────────────────────────────────────────────────────

def generate_right_panel():
    """Generate right panel: headers, bars, labels, percentages with SMIL."""
    lines = []
    lines.append("")
    lines.append('<line x1="280" y1="15" x2="280" y2="365" stroke="#e0e0e0" stroke-width="1"/>')
    lines.append("")
    lines.append("<!-- RIGHT PANEL -->")

    # ── Right panel header ──
    for si, slot in enumerate(SLOTS):
        model = slot["model"]
        minfo = MODELS[model]
        lines.append(f'<g visibility="hidden">')
        lines.append(f'  {smil_visibility(si)}')
        lines.append(f'  <text x="530" y="28" text-anchor="middle" font-size="10" fill="#8d8d8d" letter-spacing="1">PERFORMANCE \u00b7 {minfo["label"].upper()}</text>')
        lines.append(f'</g>')
    lines.append("")

    # ── Bars and labels for each row ──
    for row_idx in range(4):
        base_y = ROW_Y[row_idx]
        imp_y = IMP_ROW_Y[row_idx]
        label_y = imp_y + BAR_H + 14
        stagger = ADAPTER_STAGGER[row_idx]

        adapter_ids = [slot["adapters"][row_idx] for slot in SLOTS]
        model_ids = [slot["model"] for slot in SLOTS]

        base_widths = [bar_width(ADAPTERS[adapter_ids[si]]["base"][model_ids[si]]) for si in range(len(SLOTS))]
        imp_widths = [bar_width(ADAPTERS[adapter_ids[si]]["improved"][model_ids[si]]) for si in range(len(SLOTS))]
        imp_fills = [LIBS[ADAPTERS[adapter_ids[si]]["lib"]]["bar"] for si in range(len(SLOTS))]

        lines.append(f"<!-- Row {row_idx} bars -->")

        # Track backgrounds (static)
        lines.append(f'<rect x="{BAR_X}" y="{base_y}" width="{BAR_TRACK_W}" height="{BAR_H}" rx="{BAR_RX}" fill="#f4f4f4"/>')
        lines.append(f'<rect x="{BAR_X}" y="{imp_y}" width="{BAR_TRACK_W}" height="{BAR_H}" rx="{BAR_RX}" fill="#f4f4f4"/>')

        # Baseline bar — grows from 0
        lines.append(f'<rect x="{BAR_X}" y="{base_y}" width="0" height="{BAR_H}" rx="{BAR_RX}" fill="#c6c6c6">')
        lines.append(f'  {smil_animate_width_grow(base_widths, delay_s=0.5)}')
        lines.append(f'</rect>')

        # Improved bar — grows when adapter tile appears
        lines.append(f'<rect x="{BAR_X}" y="{imp_y}" width="0" height="{BAR_H}" rx="{BAR_RX}" fill="{imp_fills[0]}">')
        lines.append(f'  {smil_animate_width_grow(imp_widths, delay_s=stagger)}')
        lines.append(f'  {smil_animate_fill(imp_fills)}')
        lines.append(f'</rect>')

        # Baseline percentage — appears immediately
        for si in range(len(SLOTS)):
            aid = adapter_ids[si]
            model = model_ids[si]
            base_val = ADAPTERS[aid]["base"][model]
            lines.append(f'<g visibility="hidden">')
            lines.append(f'  {smil_visibility(si)}')
            lines.append(f'  <text x="{PCT_X}" y="{BASE_TEXT_Y[row_idx]}" text-anchor="end" font-size="10" font-weight="600" fill="#8d8d8d">{base_val}%</text>')
            lines.append(f'</g>')

        # Improved percentage — appears when bar finishes growing
        for si in range(len(SLOTS)):
            aid = adapter_ids[si]
            model = model_ids[si]
            info = ADAPTERS[aid]
            lib = LIBS[info["lib"]]
            imp_val = info["improved"][model]
            lines.append(f'<g visibility="hidden">')
            lines.append(f'  {smil_visibility(si, stagger + GROW_SECONDS)}')
            lines.append(f'  <text x="{PCT_X}" y="{IMP_TEXT_Y[row_idx]}" text-anchor="end" font-size="10" font-weight="700" fill="{lib["pct"]}">{imp_val}%</text>')
            lines.append(f'</g>')

        # Row labels + badges
        for si in range(len(SLOTS)):
            aid = adapter_ids[si]
            info = ADAPTERS[aid]
            lib = LIBS[info["lib"]]

            type_label = "aLoRA" if info["type"] == "alora" else "LoRA"
            type_x = BAR_X + len(info["label"]) * 7 + 10
            type_w = 34 if info["type"] == "alora" else 28
            pill_x = type_x + type_w + 6
            pill_w = 64 if info["lib"] == "guardianlib" else (46 if info["lib"] == "corelib" else 40)

            lines.append(f'<g visibility="hidden">')
            lines.append(f'  {smil_visibility(si)}')
            lines.append(f'  <text x="{BAR_X}" y="{label_y}" font-size="11" font-weight="600" fill="#161616">{info["label"]}</text>')
            lines.append(f'  <rect x="{type_x}" y="{label_y - 11}" width="{type_w}" height="14" rx="4" fill="{lib["pill_bg"]}"/>')
            lines.append(f'  <text x="{type_x + type_w // 2}" y="{label_y - 1}" text-anchor="middle" font-size="8" font-weight="600" fill="{lib["pill_text"]}">{type_label}</text>')
            lines.append(f'  <rect x="{pill_x}" y="{label_y - 11}" width="{pill_w}" height="14" rx="4" fill="{lib["pill_bg"]}"/>')
            lines.append(f'  <text x="{pill_x + pill_w // 2}" y="{label_y - 1}" text-anchor="middle" font-size="8" font-weight="600" fill="{lib["pill_text"]}">{info["lib"]}</text>')
            lines.append(f'</g>')

        lines.append("")

    # Legend at bottom (static)
    lines.append('<text x="310" y="372" font-size="12" fill="#a8a8a8">prompted baseline \u2192 with aLoRA/LoRA adapter</text>')
    return "\n".join(lines)


# ── CODE PANEL ────────────────────────────────────────────────────────────────

_CODE_FONT_SIZE = 12
_CODE_FONT = "'JetBrains Mono','Fira Code','SF Mono',Consolas,Menlo,'Courier New',monospace"

# Syntax-highlight palette (light background)
_C_MODULE = "#6f6f6f"   # module prefix: rag., core., guardian.
_C_PAREN  = "#8d8d8d"   # parentheses and commas
_C_ARG    = "#393939"   # positional argument names
_C_STRING = "#198038"   # string literals e.g. "harm"


def _tokenize(snippet):
    """Split snippet into (text, role) tokens where role drives color selection."""
    paren = snippet.index("(")
    call_site = snippet[:paren]
    args_raw = snippet[paren + 1:-1]

    tokens = []
    if "." in call_site:
        module, func = call_site.split(".", 1)
        tokens.append((module + ".", "module"))
        tokens.append((func, "func"))
    else:
        tokens.append((call_site, "func"))

    tokens.append(("(", "paren"))
    args = [a.strip() for a in args_raw.split(",")]
    for i, arg in enumerate(args):
        tokens.append((arg, "str" if arg.startswith('"') else "arg"))
        if i < len(args) - 1:
            tokens.append((", ", "paren"))
    tokens.append((")", "paren"))
    return tokens


def _render_snippet(snippet, base_x, y, func_color):
    """Return a syntax-highlighted SVG <text> element for a code snippet.

    tspan elements have no explicit x — the browser handles glyph advance widths
    correctly for whichever monospace font it resolves.
    """
    color_map = {"module": _C_MODULE, "func": func_color,
                 "paren": _C_PAREN, "arg": _C_ARG, "str": _C_STRING}
    spans = [f'<tspan fill="{color_map[role]}">{text}</tspan>'
             for text, role in _tokenize(snippet)]
    return (f'<text x="{base_x}" y="{y}" font-family="{_CODE_FONT}" font-size="{_CODE_FONT_SIZE}">'
            + "".join(spans) + "</text>")


def generate_code_panel():
    """Generate animated code panel below the performance bars.

    Shows the Mellea invocation for each adapter row, appearing with the same
    stagger delay as the corresponding adapter tile in the left panel.
    """
    lines = []
    lines.append("")
    lines.append("<!-- CODE PANEL -->")
    lines.append(f'<line x1="22" y1="{CODE_PANEL_Y - 2}" x2="798" y2="{CODE_PANEL_Y - 2}" stroke="#e8e8e8" stroke-width="1"/>')
    lines.append(f'<rect x="22" y="{CODE_PANEL_Y}" width="776" height="{CODE_PANEL_H}" rx="6" fill="#f4f4f4"/>')
    lines.append(f'<text x="34" y="{CODE_HEADER_Y}" font-size="9" fill="#8d8d8d" letter-spacing="1">MELLEA INVOCATION</text>')

    for row_idx in range(4):
        stagger = ADAPTER_STAGGER[row_idx]
        cy = CODE_LINE_Y[row_idx]
        adapter_ids = [slot["adapters"][row_idx] for slot in SLOTS]

        for si, aid in enumerate(adapter_ids):
            info = ADAPTERS[aid]
            lib = LIBS[info["lib"]]
            lines.append(f'<g visibility="hidden">')
            lines.append(f'  {smil_visibility(si, stagger)}')
            lines.append(f'  {_render_snippet(info["snippet"], 52, cy, lib["pct"])}')
            lines.append(f'</g>')

    lines.append("")
    return "\n".join(lines)


# ── MAIN SVG GENERATION ───────────────────────────────────────────────────────

def generate_svg():
    """Generate the complete SVG with SMIL animations."""
    parts = []
    parts.append(f'<svg width="820" height="{SVG_HEIGHT}" viewBox="0 0 820 {SVG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">')
    parts.append(generate_schedule_comment())

    parts.append("<defs><style>")
    parts.append("  text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}")
    parts.append("</style></defs>")
    parts.append("")

    parts.append(f'<rect width="820" height="{SVG_HEIGHT}" rx="10" fill="#ffffff"/>')
    parts.append(f'<rect width="820" height="{SVG_HEIGHT}" rx="10" fill="none" stroke="#e0e0e0" stroke-width="1"/>')

    parts.append(generate_left_panel())
    parts.append(generate_right_panel())
    parts.append(generate_code_panel())
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    print(generate_svg())
