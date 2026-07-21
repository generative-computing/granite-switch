"""Display helpers for the govt RAG flow tutorial.

Formatting / pretty-printing only. Blocked-state branches in
`show_answer` are no-ops when `r["blocked"]` is absent.
"""

import json

from IPython.display import Markdown, display
from mellea.stdlib.components.chat import Message as MelleaMessage


def _is_clear(clarification):
    """rag.clarify_query returns 'CLEAR' when no clarification is needed; accept prefix variants like 'CLEARLY'."""
    return clarification.strip().upper().startswith("CLEAR")


def show_answer(r):
    """Pretty-print a single flow result. Handles all four terminal states."""
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


def show_history(ctx):
    """Render a Mellea `ChatContext` as formatted Markdown."""
    messages = [m for m in ctx.as_list() if isinstance(m, MelleaMessage)]
    if not messages:
        display(Markdown("*(conversation history is empty)*"))
        return
    md = ["---", f"### Conversation history — {len(messages) // 2} turn(s)", "---"]
    for m in messages:
        role = "👤 **User**" if m.role == "user" else "🤖 **Assistant**"
        docs = m._docs or []
        doc_note = f" *({len(docs)} docs)*" if docs else ""
        md.append(f"{role}{doc_note}\n\n> {m.content}")
    display(Markdown("\n\n".join(md)))


def show_intermediates(r, top_k):
    """Flow: harm + scope guardian -> rewrite -> retrieve -> answerability -> clarify -> answer -> citations."""
    md = ["---", f"### Intermediates - *{r['query']}*", "---"]

    harm_score = r.get("guardian_harm_score", 0)
    harm_badge = "🟢 safe" if harm_score < 0.5 else "🔴 harmful"
    md.append(
        f"**[1a] Guardian - Harm** - {harm_badge} &nbsp;&nbsp; `score={harm_score:.3f}` &nbsp;&nbsp; (full-conversation eval)"
    )

    if r.get("blocked") and "Harmful" in r.get("block_reason", ""):
        md.append(f"\n> ⛔ **BLOCKED:** {r['block_reason']}")
        display(Markdown("\n\n".join(md)))
        return

    scope_score = r.get("guardian_scope_score", 0)
    scope_badge = "🟢 in-scope" if scope_score >= 0.5 else "🔴 out-of-scope"
    md.append(
        f"\n**[1b] Guardian - Scope** - {scope_badge} &nbsp;&nbsp; `score={scope_score:.3f}`"
    )

    if r.get("blocked"):
        md.append(f"\n> ⛔ **BLOCKED:** {r['block_reason']}")
        display(Markdown("\n\n".join(md)))
        return

    md.append(
        f"\n**[2] Query Rewrite**\n\n"
        f"| | |\n|---|---|\n"
        f"| original | {r['query']} |\n"
        f"| rewritten | {r.get('rewritten_query')} |"
    )

    docs = r.get("documents", [])
    md.append(
        f"\n**[3] ChromaDB Retrieval** - {len(docs)} doc(s) (top {top_k}, cosine sim)"
    )
    if docs:
        md.append(f"\n<details><summary>📚 Show all {len(docs)} documents</summary>\n")
        for i, d in enumerate(docs):
            md.append(
                f"<details><summary>📄 Document {i + 1}</summary>\n\n```\n{d}\n```\n\n</details>\n"
            )
        md.append("</details>")

    answerability = r.get("answerability")
    if answerability is not None:
        badge = "✅ answerable" if not r.get("unanswerable") else "🔍 unanswerable"
        md.append(
            f"\n**[4] Answerability** - {badge} &nbsp;&nbsp; `verdict={answerability}`"
        )
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
        md.append(
            f"\n<details><summary>🔖 Show citations JSON</summary>\n\n```json\n{json.dumps(citations, indent=2)}\n```\n\n</details>"
        )
    else:
        md.append("\n*(none)*")

    display(Markdown("\n\n".join(md)))
