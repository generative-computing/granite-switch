"""Display helpers for the govt RAG pipeline tutorials (03_01, 03_02, 03_03).

Pure formatting / pretty-printing only — no pipeline logic. Each tutorial uses a
different `show_intermediates` variant to match its pipeline shape:

  - `show_intermediates_simple`     — 03_01 (no guardian, no retries)
  - `show_intermediates_sequential` — 03_02 (harm + scope guardian, no retries)
  - `show_intermediates_loops`      — 03_03 (harm guardian + scope/answer retry loops)

`show_answer`, `show_history`, and `Conversation` work for all three pipelines
(blocked-state branches are no-ops when `r["blocked"]` is absent).
"""

import json

from IPython.display import Markdown, display


def _is_clear(clarification):
    """rag.clarify_query returns 'CLEAR' when no clarification is needed; accept prefix variants like 'CLEARLY'."""
    return clarification.strip().upper().startswith("CLEAR")


def show_answer(r):
    """Pretty-print a single pipeline result. Handles all four terminal states."""
    lines = [f"**Q:** {r['query']}", "---"]
    if r.get("blocked"):
        lines.append(f"⛔ **BLOCKED** — {r['block_reason']}")
    elif r.get("unanswerable"):
        lines.append(
            f"🔍 **Not in corpus** — `answerability={r['answerability']}`\n\n"
            f"> I don't have enough information in my knowledge base to answer that."
        )
    elif r.get("needs_clarification"):
        lines.append(f"❓ **Clarification needed:**\n\n> {r['clarification']}")
    else:
        lines.append(f"**A:** {r.get('answer', '')}")
    display(Markdown("\n\n".join(lines)))


def show_history(conv):
    """Render a Conversation's history as formatted Markdown."""
    if not conv.history:
        display(Markdown("*(conversation history is empty)*"))
        return
    md = ["---", f"### Conversation history — {len(conv.history)//2} turn(s)", "---"]
    for m in conv.history:
        role = "👤 **User**" if m["role"] == "user" else "🤖 **Assistant**"
        doc_note = f" *({len(m['documents'])} docs)*" if m.get("documents") else ""
        md.append(f"{role}{doc_note}\n\n> {m['content']}")
    display(Markdown("\n\n".join(md)))


def show_intermediates_simple(r, top_k):
    """03_01 simple pipeline: rewrite -> retrieve -> answerability -> clarify -> answer -> citations."""
    md = ["---", f"### Intermediates — *{r['query']}*", "---"]

    md.append(f"**[1] Query Rewrite**\n\n"
              f"| | |\n|---|---|\n"
              f"| original | {r['query']} |\n"
              f"| rewritten | {r.get('rewritten_query')} |")

    docs = r.get("documents", [])
    md.append(f"\n**[2] ChromaDB Retrieval** — {len(docs)} doc(s) (top {top_k}, cosine sim)")
    if docs:
        md.append(f"\n<details><summary>Show all {len(docs)} documents</summary>\n")
        for i, d in enumerate(docs):
            md.append(f"<details><summary>Document {i+1}</summary>\n\n```\n{d}\n```\n\n</details>\n")
        md.append("</details>")

    answerability = r.get("answerability")
    if answerability is not None:
        badge = "answerable" if not r.get("unanswerable") else "unanswerable"
        md.append(f"\n**[3] Answerability** — {badge} &nbsp;&nbsp; `verdict={answerability}`")
    if r.get("unanswerable"):
        display(Markdown("\n\n".join(md)))
        return

    clar = r.get("clarification", "")
    badge = "CLEAR" if _is_clear(clar) else "needs clarification"
    md.append(f"\n**[4] Clarification** — {badge}")
    if r.get("needs_clarification"):
        md.append(f"\n> {clar}")
        display(Markdown("\n\n".join(md)))
        return

    ans = r.get("answer", "")
    md.append(f"\n**[5] Answer** — {len(ans)} chars\n\n> {ans}")

    citations = r.get("citations", [])
    md.append(f"\n**[6] Citations** — {len(citations)} found")
    if citations:
        md.append(f"\n<details><summary>Show citations JSON</summary>\n\n```json\n{json.dumps(citations, indent=2)}\n```\n\n</details>")
    else:
        md.append("\n*(none)*")

    display(Markdown("\n\n".join(md)))


def show_intermediates_sequential(r, top_k):
    """03_02 sequential pipeline: harm + scope guardian -> rewrite -> retrieve -> answerability -> clarify -> answer -> citations."""
    md = ["---", f"### Intermediates - *{r['query']}*", "---"]

    harm_score = r.get("guardian_harm_score", 0)
    harm_badge = "🟢 safe" if harm_score < 0.5 else "🔴 harmful"
    md.append(f"**[1a] Guardian - Harm** - {harm_badge} &nbsp;&nbsp; `score={harm_score:.3f}` &nbsp;&nbsp; (full-conversation eval)")

    if r.get("blocked") and "Harmful" in r.get("block_reason", ""):
        md.append(f"\n> ⛔ **BLOCKED:** {r['block_reason']}")
        display(Markdown("\n\n".join(md)))
        return

    scope_score = r.get("guardian_scope_score", 0)
    scope_badge = "🟢 in-scope" if scope_score >= 0.5 else "🔴 out-of-scope"
    md.append(f"\n**[1b] Guardian - Scope** - {scope_badge} &nbsp;&nbsp; `score={scope_score:.3f}`")

    if r.get("blocked"):
        md.append(f"\n> ⛔ **BLOCKED:** {r['block_reason']}")
        display(Markdown("\n\n".join(md)))
        return

    md.append(f"\n**[2] Query Rewrite**\n\n"
              f"| | |\n|---|---|\n"
              f"| original | {r['query']} |\n"
              f"| rewritten | {r.get('rewritten_query')} |")

    docs = r.get("documents", [])
    md.append(f"\n**[3] ChromaDB Retrieval** - {len(docs)} doc(s) (top {top_k}, cosine sim)")
    if docs:
        md.append(f"\n<details><summary>📚 Show all {len(docs)} documents</summary>\n")
        for i, d in enumerate(docs):
            md.append(f"<details><summary>📄 Document {i+1}</summary>\n\n```\n{d}\n```\n\n</details>\n")
        md.append("</details>")

    answerability = r.get("answerability")
    if answerability is not None:
        badge = "✅ answerable" if not r.get("unanswerable") else "🔍 unanswerable"
        md.append(f"\n**[4] Answerability** - {badge} &nbsp;&nbsp; `verdict={answerability}`")
    if r.get("unanswerable"):
        display(Markdown("\n\n".join(md)))
        return

    clar = r.get("clarification", "")
    badge = "✅ CLEAR" if _is_clear(clar) else "❓ needs clarification"
    md.append(f"\n**[5] Clarification** - {badge}")
    if r.get("needs_clarification"):
        md.append(f"\n> {clar}")
        display(Markdown("\n\n".join(md)))
        return

    ans = r.get("answer", "")
    md.append(f"\n**[6] Answer** - {len(ans)} chars\n\n> {ans}")

    citations = r.get("citations", [])
    md.append(f"\n**[7] Citations** - {len(citations)} found")
    if citations:
        md.append(f"\n<details><summary>🔖 Show citations JSON</summary>\n\n```json\n{json.dumps(citations, indent=2)}\n```\n\n</details>")
    else:
        md.append("\n*(none)*")

    display(Markdown("\n\n".join(md)))


def show_intermediates_loops(r, top_k):
    """03_03 loops pipeline: harm guardian -> scope retry loop -> rewrite -> answerability retry loop -> clarify -> answer -> citations."""
    md = ["---", f"### Intermediates — *{r['query']}*", "---"]

    # [1] Harm
    harm_score = r.get("guardian_harm_score", 0)
    harm_badge = "safe" if harm_score < 0.5 else "harmful"
    md.append(f"**[1] Guardian — Harm** — {harm_badge} &nbsp;&nbsp; `score={harm_score:.3f}`")

    if r.get("blocked") and "Harmful" in r.get("block_reason", ""):
        md.append(f"\n> BLOCKED: {r['block_reason']}")
        display(Markdown("\n\n".join(md)))
        return

    # [2] Scope retry loop
    scope_attempts = r.get("scope_attempts", [])
    if scope_attempts:
        n = len(scope_attempts)
        last = scope_attempts[-1]
        passed = last["score"] >= 0.5
        badge = "in-scope" if passed else "out-of-scope"
        md.append(f"\n**[2] Scope retry loop** — {badge} &nbsp;&nbsp; ({n} attempt(s))")
        md.append("\n| Attempt | Query | Score | Result |")
        md.append("|---------|-------|-------|--------|")
        for i, att in enumerate(scope_attempts):
            result = "in-scope" if att["score"] >= 0.5 else "out-of-scope"
            md.append(f"| {i+1} | {att['query'][:60]}{'...' if len(att['query'])>60 else ''} | {att['score']:.3f} | {result} |")

    if r.get("blocked"):
        md.append(f"\n> BLOCKED: {r['block_reason']}")
        display(Markdown("\n\n".join(md)))
        return

    # [3] Query Rewrite
    md.append(f"\n**[3] Query Rewrite**\n\n"
              f"| | |\n|---|---|\n"
              f"| original | {r['query']} |\n"
              f"| rewritten | {r.get('rewritten_query')} |")

    # [4] Answerability retry loop
    ans_attempts = r.get("answerability_attempts", [])
    if ans_attempts:
        n = len(ans_attempts)
        last = ans_attempts[-1]
        passed = last["verdict"] != "unanswerable"
        badge = "answerable" if passed else "unanswerable"
        md.append(f"\n**[4] Answerability retry loop** — {badge} &nbsp;&nbsp; ({n} attempt(s))")
        md.append("\n| Attempt | Query | Verdict |")
        md.append("|---------|-------|---------|")
        for i, att in enumerate(ans_attempts):
            md.append(f"| {i+1} | {att['query'][:60]}{'...' if len(att['query'])>60 else ''} | {att['verdict']} |")

    if r.get("unanswerable"):
        display(Markdown("\n\n".join(md)))
        return

    docs = r.get("documents", [])
    md.append(f"\n**Retrieval** — {len(docs)} doc(s) (top {top_k}, cosine sim)")
    if docs:
        md.append(f"\n<details><summary>Show all {len(docs)} documents</summary>\n")
        for i, d in enumerate(docs):
            md.append(f"<details><summary>Document {i+1}</summary>\n\n```\n{d}\n```\n\n</details>\n")
        md.append("</details>")

    # [5] Clarification
    clar = r.get("clarification", "")
    badge = "CLEAR" if _is_clear(clar) else "needs clarification"
    md.append(f"\n**[5] Clarification** — {badge}")
    if r.get("needs_clarification"):
        md.append(f"\n> {clar}")
        display(Markdown("\n\n".join(md)))
        return

    # [6] Answer
    ans = r.get("answer", "")
    md.append(f"\n**[6] Answer** — {len(ans)} chars\n\n> {ans}")

    # [7] Citations
    citations = r.get("citations", [])
    md.append(f"\n**[7] Citations** — {len(citations)} found")
    if citations:
        md.append(f"\n<details><summary>Show citations JSON</summary>\n\n```json\n{json.dumps(citations, indent=2)}\n```\n\n</details>")
    else:
        md.append("\n*(none)*")

    display(Markdown("\n\n".join(md)))


class Conversation:
    """Stateful chat wrapper — calls `run_pipeline` and prints the answer.

    The pipeline function differs across tutorials (simple/sequential/loops),
    so it's injected at construction time. `show_answer` is shared.
    """

    def __init__(self, run_pipeline):
        self.run_pipeline = run_pipeline
        self.history = []

    def ask(self, query):
        print(f"[turn {len(self.history)//2 + 1}  |  history: {len(self.history)} msg(s)]")
        r = self.run_pipeline(query, self.history)
        show_answer(r)

        if r.get("blocked"):
            return r  # blocked turns are not recorded

        if r.get("unanswerable"):
            reply = "I don't have enough information in my knowledge base to answer that."
        elif r.get("needs_clarification"):
            reply = r["clarification"]
        else:
            reply = r.get("answer", "")

        self.history.append({"role": "user",      "content": query, "documents": r.get("documents")})
        self.history.append({"role": "assistant", "content": reply})
        print(f"→ history now has {len(self.history)} message(s)")
        return r
